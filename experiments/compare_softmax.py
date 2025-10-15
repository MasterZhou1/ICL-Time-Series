#!/usr/bin/env python3
"""
Compare Linear LSA vs Softmax Attention under identical settings and data.

For each seed, this script:
  1) Generates a single AR(p) dataset with the given seed
  2) Trains two models with identical hyperparameters and loaders:
     - Linear LSA (use_softmax=False)
     - Softmax Attention (use_softmax=True)
  3) Saves artifacts into informative directories (in current working directory):
     {subdir}/seed{seed}_p{p}_n{n}_L{L}/
       ├── shared/train_series.npy, test_series.npy (single copy)
       ├── linear/best_model.pt, model_config.json
       └── softmax/best_model.pt, model_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys

# Ensure repo root is on sys.path before importing project modules (works from experiments/)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.common import setup_project_path, set_global_seed
setup_project_path(levels_up=2)

from data.ar_sims import ARDataGenerator
from data.ar_dataloader import ARForecastDataset
from models.lsa_transformer import LSATransformerWithHankel
from experiments.train_lsa import main as train_lsa_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Linear LSA vs Softmax attention fairly.")
    # Model/data
    parser.add_argument("--p", type=int, required=True, help="AR order / context length")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSA layers")
    parser.add_argument("--history-len", type=int, default=None, help="History length (defaults to p+2 if None)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise std for AR generator")
    parser.add_argument("--series-len", type=int, default=50000, help="Total length of the synthetic series")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    parser.add_argument("--subdir", type=str, default="compare_softmax", help="Subdirectory name for experiments (created in current directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p = args.p
    layers = args.layers
    history_len = args.history_len if args.history_len is not None else p + 2
    device = args.device

    base_dir = Path.cwd() / args.subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    set_global_seed(int(seed))

    # 1) Data generation (once per seed)
    gen = ARDataGenerator(p=p, sigma=args.sigma, sequence_length=args.series_len, random_seed=int(seed))
    series = gen.generate_series()
    train_end = int(len(series) * args.train_split)
    train_series, test_series = series[:train_end], series[train_end:]

    # 2) Datasets and loaders with identical settings
    # train_ds = ARForecastDataset(train_series, p=p, history_len=history_len)
    # val_ds = ARForecastDataset(test_series, p=p, history_len=history_len)
    # Elegantly keep loaders identical: no shuffle (fixed order) so both models see the same batches
    # train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    # val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3) Informative run directory
    run_root = base_dir / f"seed{seed}_p{p}_n{history_len}_L{layers}"
    shared_dir = run_root / "shared"
    linear_dir = run_root / "lsa"
    softmax_dir = run_root / "softmax"
    shared_dir.mkdir(parents=True, exist_ok=True)
    linear_dir.mkdir(parents=True, exist_ok=True)
    softmax_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets to shared directory (single copy)
    np.save(shared_dir / "train_series.npy", train_series)
    np.save(shared_dir / "test_series.npy", test_series)

    # 4) Train Linear LSA (reuse training entrypoint)
    linear_args = [
        "--p", str(p),
        "--layers", str(layers),
        "--history-len", str(history_len),
        "--sigma", str(args.sigma),
        "--series-len", str(args.series_len),
        "--train-split", str(args.train_split),
        "--val-split", str(args.val_split),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", str(device),
        "--patience", str(args.patience),
        "--out", str(linear_dir),
        "--seed", str(seed),
        "--data-dir", str(shared_dir),
    ]
    # Temporarily patch sys.argv to call train_lsa.main in-process
    _argv_backup = sys.argv
    try:
        sys.argv = [str(train_lsa_main.__module__)] + linear_args
        train_lsa_main()
    finally:
        sys.argv = _argv_backup
    # best val and epochs are written to results.json; also persist minimal fields locally
    try:
        with open(linear_dir / "results.json", 'r') as f:
            _lin_results = json.load(f)
        best_val_linear = float(_lin_results.get('best_val_loss', float('inf')))
        epochs_linear = int(_lin_results.get('epochs_trained', 0))
    except Exception:
        best_val_linear, epochs_linear = float('inf'), 0
    with open(linear_dir / "model_config.json", 'w') as f:
        json.dump({
            'experiment': 'compare_softmax',
            'variant': 'linear',
            'seed': int(seed),
            'p': p,
            'layers': layers,
            'history_len': history_len,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'sigma': args.sigma,
            'series_len': args.series_len,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'use_softmax': False,
            'data_path': str(shared_dir.relative_to(run_root)),
            'best_val_loss': float(best_val_linear),
            'epochs_trained': int(epochs_linear),
        }, f, indent=2)

    # 5) Train Softmax Attention
    # Reset seed to keep initial model/optimizer states reproducible
    set_global_seed(int(seed))
    softmax_args = [
        "--p", str(p),
        "--layers", str(layers),
        "--history-len", str(history_len),
        "--sigma", str(args.sigma),
        "--series-len", str(args.series_len),
        "--train-split", str(args.train_split),
        "--val-split", str(args.val_split),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", str(device),
        "--patience", str(args.patience),
        "--out", str(softmax_dir),
        "--seed", str(seed),
        "--use-softmax",
        "--data-dir", str(shared_dir),
    ]
    _argv_backup = sys.argv
    try:
        sys.argv = [str(train_lsa_main.__module__)] + softmax_args
        train_lsa_main()
    finally:
        sys.argv = _argv_backup
    try:
        with open(softmax_dir / "results.json", 'r') as f:
            _sm_results = json.load(f)
        best_val_softmax = float(_sm_results.get('best_val_loss', float('inf')))
        epochs_softmax = int(_sm_results.get('epochs_trained', 0))
    except Exception:
        best_val_softmax, epochs_softmax = float('inf'), 0
    with open(softmax_dir / "model_config.json", 'w') as f:
        json.dump({
            'experiment': 'compare_softmax',
            'variant': 'softmax',
            'seed': int(seed),
            'p': p,
            'layers': layers,
            'history_len': history_len,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'sigma': args.sigma,
            'series_len': args.series_len,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'use_softmax': True,
            'data_path': str(shared_dir.relative_to(run_root)),
            'best_val_loss': float(best_val_softmax),
            'epochs_trained': int(epochs_softmax),
        }, f, indent=2)

    # 6) Summary comparison file at the run root
    summary = {
        'seed': int(seed),
        'p': p,
        'layers': layers,
        'history_len': history_len,
        'linear_best_val_loss': float(best_val_linear),
        'softmax_best_val_loss': float(best_val_softmax),
        'delta_softmax_minus_linear': float(best_val_softmax - best_val_linear),
    }
    with open(run_root / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Seed {seed}: linear={best_val_linear:.6f} softmax={best_val_softmax:.6f} "
          f"Δ={best_val_softmax - best_val_linear:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


