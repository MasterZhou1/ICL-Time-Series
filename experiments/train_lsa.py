"""Training script for LSA Transformer on synthetic AR data.

This script performs the following steps:
1.  Generates a long stationary AR(p) series.
2.  Splits the series into training, validation, and test sets.
3.  Saves the training and test series to the output directory for reuse.
4.  Creates PyTorch DataLoaders for the LSA model.
5.  Trains an `LSATransformerWithHankel` model.
6.  Saves the best model checkpoint and a training curve plot.

Usage (example):
    python experiments/train_lsa.py --p 7 --layers 2 --history-len 15 --epochs 100
"""
from __future__ import annotations
import os
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

import sys

# Ensure repo root is on sys.path before importing project modules (works from experiments/)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.common import setup_project_path
setup_project_path(levels_up=2)

from data.ar_sims import ARDataGenerator
from data.ar_dataloader import ARForecastDataset
from models.lsa_transformer import LSATransformerWithHankel
from utils.wandb_utils import WandbRun
from utils.common import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSA Transformer on synthetic AR data.")
    parser.add_argument("--p", type=int, default=7, help="AR order / context length for the LSA model.")
    parser.add_argument("--history-len", type=int, default=None, help="History length for the LSA model. Defaults to p + 2.")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSA layers in the trained model.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise std for the AR data generator.")
    parser.add_argument("--series-len", type=int, default=50000, help="Total length of the synthetic AR series.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Fraction of the series for training.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of the series for validation.")
    parser.add_argument("--test-split", type=float, default=None, help="Fraction of the series for testing. If None, uses 1 - train_split - val_split.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs.")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory for data, checkpoints, and plots.")
    parser.add_argument("--use-softmax", action="store_true", help="Enable softmax attention (default: linear LSA)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional directory containing shared train/test .npy files")
    return parser.parse_args()

def main():
    """Main function to train and evaluate the LSA model."""
    args = parse_args()

    # Handle test_split
    if args.test_split is None:
        args.test_split = 1.0 - args.train_split - args.val_split

    if args.history_len is None:
        args.history_len = args.p + 2

    if args.history_len <= args.p:
        raise ValueError(f"history-len ({args.history_len}) must be greater than p ({args.p}).")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Seeding and Save Configuration ---
    set_global_seed(args.seed)
    import json
    # Augment config with data_path if provided (relative to output_dir)
    config_to_save = vars(args).copy()
    if args.data_dir is not None:
        try:
            rel_path = str(Path(args.data_dir).resolve().relative_to(output_dir.resolve().parent))
        except Exception:
            # Fallback to plain relative path
            rel_path = str(Path(args.data_dir))
        config_to_save['data_path'] = rel_path
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # --- 2. Data Generation ---
    if args.data_dir is not None:
        print(f"Loading data from shared directory: {args.data_dir}")
        data_dir = Path(args.data_dir)
        train_series = np.load(data_dir / "train_series.npy")
        test_series = np.load(data_dir / "test_series.npy")
    else:
        print(f"Generating AR({args.p}) data...")
        generator = ARDataGenerator(p=args.p, sigma=args.sigma, sequence_length=args.series_len, random_seed=args.seed)
        series = generator.generate_series()
        train_end = int(len(series) * args.train_split)
        train_series, test_series = series[:train_end], series[train_end:]
        np.save(output_dir / "train_series.npy", train_series)
        np.save(output_dir / "test_series.npy", test_series)
        print(f"Data saved to {output_dir}/")

    # --- 3. Model Training ---
    print("Training Start.")
    
    train_dataset = ARForecastDataset(train_series, p=args.p, history_len=args.history_len)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_series = test_series # Use test set as validation for simplicity in this script
    val_dataset = ARForecastDataset(val_series, p=args.p, history_len=args.history_len)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = LSATransformerWithHankel(p=args.p, L=args.layers, use_softmax=args.use_softmax).to(args.device)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
    except Exception:
        pass
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    
    # Load .env file explicitly before WANDB initialization
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available
    # Use WANDB_PROJECT from .env file, fallback to hardcoded name
    project_name = os.getenv("WANDB_PROJECT", "icl-time-series")
    
    with WandbRun(project=project_name, config=vars(args), group="ad-hoc", name=f"seed{args.seed}_p{args.p}_n{args.history_len}_L{args.layers}{'_softmax' if args.use_softmax else ''}") as wb:
        for epoch in trange(args.epochs, desc="Training", unit="epoch"):
            model.train()
            total_train_loss = 0.0
            for history, target in train_dl:
                history, target = history.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                preds = model.predict(history)
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for history, target in val_dl:
                    history, target = history.to(args.device), target.to(args.device)
                    preds = model.predict(history)
                    loss = criterion(preds, target)
                    epoch_val_loss += loss.item() * len(history)
            if wb:
                wb.log({
                    'train/loss': total_train_loss / max(1, len(train_dl)),
                    'val/loss': epoch_val_loss / max(1, len(val_dataset)),
                    'epoch': epoch + 1
                }, step=epoch + 1)

            current_val_loss = epoch_val_loss / len(val_dataset)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}.\n")

    # --- 4. Run Evaluation ---
    # print("Running final evaluation...")
    results_json_path = output_dir / "results.json"
    # eval_cmd = [
    #     sys.executable, str(repo_root / "experiments/evaluate_models.py"),
    #     "--artifacts-dir", str(output_dir),
    #     "--save-results", str(results_json_path),
    #     "--no-plots"
    # ]
    
    # result = subprocess.run(eval_cmd, capture_output=True, text=True)
    # if result.returncode != 0:
    #     print("Evaluation script failed:")
    #     print(result.stderr)
    # else:
    #     print(f"Evaluation complete. Results saved to {results_json_path}")

    # --- 5. Persist comprehensive training metrics into results.json for downstream comparison ---
    try:
        if results_json_path.exists():
            with open(results_json_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}

        # Add comprehensive training parameters
        results.update({
            'best_val_loss': float(best_val_loss),
            'epochs_trained': int(epoch + 1 if 'epoch' in locals() else 0),
            'use_softmax': bool(args.use_softmax),
            'experiment': 'lsa_training',
            # Training hyperparameters
            'learning_rate': float(args.lr),
            'batch_size': int(args.batch_size),
            'patience': int(args.patience),
            'device': str(args.device),
            # Data parameters
            'p': int(args.p),
            'history_len': int(args.history_len),
            'layers': int(args.layers),
            'sigma': float(args.sigma),
            'series_len': int(args.series_len),
            'train_split': float(args.train_split),
            'val_split': float(args.val_split),
            'test_split': 1.0 - args.train_split - args.val_split,
            # Model configuration
            'model_type': 'LSATransformerWithHankel',
            'seed': int(args.seed),
            # Training metadata
            'early_stopped': patience_counter >= args.patience if 'patience_counter' in locals() else False,
            'converged': patience_counter < args.patience if 'patience_counter' in locals() else True
        })
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to update results.json with training metrics: {e}")
        pass

if __name__ == "__main__":
    main()
