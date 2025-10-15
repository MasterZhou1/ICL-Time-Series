"""
Evaluate and compare the performance of trained LSA Transformer, Softmax Attention,
and Linear AR models on a fixed test set.

This script supports two modes:
1. Single model evaluation: Compare LSA Transformer against Linear AR baseline
2. Softmax comparison: Compare LSA Transformer, Softmax Attention, and Linear AR models

This script first checks for the required data and model artifacts. It then proceeds
with the evaluation, which includes:
1. Teacher-Forcing: One-step-ahead prediction using true historical values.
2. Chain-of-Thought (Autoregressive): Multi-step prediction using the model's
   own previous outputs as input.

The script produces comparison plots and prints MSE for all models.

Usage (examples):
    # Single model evaluation (default)
    python experiments/evaluate_models.py
    python experiments/evaluate_models.py --artifacts-dir my_experiment --cot-steps 100

    # Softmax comparison mode
    python experiments/evaluate_models.py --compare-softmax experiments/compare_softmax/seed42_p5_n8_L1 --cot-steps 50

    # Save results
    python experiments/evaluate_models.py --compare-softmax experiments/compare_softmax/seed42_p5_n8_L1 --save-results results.json
"""
import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import subprocess
from tqdm import tqdm
import json

# Ensure repo root is on sys.path before importing project modules (works from experiments/)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.common import setup_project_path
from utils.common import set_global_seed
setup_project_path(levels_up=2)

from models.linear_ar import LinearARModel
from models.lsa_transformer import LSATransformerWithHankel
from data.ar_dataloader import ARForecastDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare LSA, Softmax, and Linear AR models.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory containing trained model and data.")
    parser.add_argument("--compare-softmax", type=str, default=None, help="Directory containing compare_softmax experiment (e.g., 'experiments/compare_softmax/seed42_p5_n8_L1') to compare LSA vs Softmax models.")
    parser.add_argument("--cot-steps", type=int, default=50, help="Number of steps for Chain-of-Thought evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots.")
    return parser.parse_args()

def ensure_artifacts_exist(args):
    """Check for artifacts and provide instructions if they don't exist."""
    artifacts_path = Path(args.artifacts_dir)

    # Check if this is a compare_softmax experiment
    if args.compare_softmax:
        compare_softmax_path = Path(args.compare_softmax)
        if not compare_softmax_path.exists():
            print(f"Compare softmax directory not found: {compare_softmax_path}")
            print("Please run compare_softmax.py first to generate the required artifacts.")
            sys.exit(1)

        # Check for required directories
        linear_dir = compare_softmax_path / "lsa"
        softmax_dir = compare_softmax_path / "softmax"
        shared_dir = compare_softmax_path / "shared"

        missing_dirs = []
        if not linear_dir.exists():
            missing_dirs.append(f"linear/ in {compare_softmax_path}")
        if not softmax_dir.exists():
            missing_dirs.append(f"softmax/ in {compare_softmax_path}")
        if not shared_dir.exists():
            missing_dirs.append(f"shared/ in {compare_softmax_path}")

        if missing_dirs:
            print("Required directories not found in compare_softmax experiment:")
            for d in missing_dirs:
                print(f"  - {d}")
            sys.exit(1)

        # Check for required files
        linear_checkpoint = linear_dir / "best_model.pt"
        linear_config = linear_dir / "model_config.json"
        softmax_checkpoint = softmax_dir / "best_model.pt"
        softmax_config = softmax_dir / "model_config.json"
        train_data = shared_dir / "train_series.npy"
        test_data = shared_dir / "test_series.npy"

        missing_files = []
        if not linear_checkpoint.exists():
            missing_files.append(f"best_model.pt in {linear_dir}")
        if not linear_config.exists():
            missing_files.append(f"model_config.json in {linear_dir}")
        if not softmax_checkpoint.exists():
            missing_files.append(f"best_model.pt in {softmax_dir}")
        if not softmax_config.exists():
            missing_files.append(f"model_config.json in {softmax_dir}")
        if not train_data.exists():
            missing_files.append(f"train_series.npy in {shared_dir}")
        if not test_data.exists():
            missing_files.append(f"test_series.npy in {shared_dir}")

        if missing_files:
            print("Required files not found in compare_softmax experiment:")
            print("Please run compare_softmax.py first:")
            print("Example: python experiments/compare_softmax.py --p 5 --layers 1 --history-len 8 --seed 42")
            for f in missing_files:
                print(f"  - {f}")
            sys.exit(1)

        return {
            'linear_checkpoint': linear_checkpoint,
            'linear_config': linear_config,
            'softmax_checkpoint': softmax_checkpoint,
            'softmax_config': softmax_config,
            'data_dir': shared_dir,
            'is_compare_softmax': True
        }
    else:
        # Original single model logic
        lsa_checkpoint = artifacts_path / "best_model.pt"
        model_config = artifacts_path / "model_config.json"

        # Check if data_path is specified in config (shared data location)
        try:
            with open(model_config, 'r') as f:
                config = json.load(f)
            data_path = config.get('data_path')
            if data_path:
                data_dir = artifacts_path.parent / data_path
            else:
                data_dir = artifacts_path
        except:
            data_dir = artifacts_path

        train_data = data_dir / "train_series.npy"
        test_data = data_dir / "test_series.npy"

        missing_files = []
        if not lsa_checkpoint.exists():
            missing_files.append(f"best_model.pt in {artifacts_path}")
        if not train_data.exists():
            missing_files.append(f"train_series.npy in {data_dir}")
        if not test_data.exists():
            missing_files.append(f"test_series.npy in {data_dir}")
        if not model_config.exists():
            missing_files.append(f"model_config.json in {artifacts_path}")

        if missing_files:
            print("Required artifacts not found. Please train a model first:")
            print("Example: python experiments/train_lsa.py --p 7 --layers 2")
            print("Missing files:")
            for f in missing_files:
                print(f"  - {f}")
            sys.exit(1)

        return {
            'lsa_checkpoint': lsa_checkpoint,
            'model_config': model_config,
            'data_dir': data_dir,
            'is_compare_softmax': False
        }

def main():
    args = parse_args()

    # Get artifacts information
    artifacts_info = ensure_artifacts_exist(args)

    # Determine the directory for saving plots
    # For compare_softmax mode, use the compare_softmax directory
    # For single model mode, use the artifacts directory
    if artifacts_info['is_compare_softmax']:
        artifacts_dir = Path(args.compare_softmax)
    else:
        artifacts_dir = Path(args.artifacts_dir)

    # Load model configuration(s)
    if artifacts_info['is_compare_softmax']:
        # Load both linear and softmax configs
        with open(artifacts_info['linear_config'], 'r') as f:
            linear_config = json.load(f)
        with open(artifacts_info['softmax_config'], 'r') as f:
            softmax_config = json.load(f)

        # Use consistent config values (they should be the same)
        config = linear_config  # or softmax_config - they should be identical except for use_softmax
        seed = int(config.get('seed', 42))
        p = config['p']
        layers = config.get('layers', config.get('lsa_layers'))
        history_len = config['history_len']
        print(f"Loaded compare_softmax configs: p={p}, layers={layers}, history_len={history_len}")
    else:
        # Original single model logic
        with open(artifacts_info['model_config'], 'r') as f:
            config = json.load(f)

        seed = int(config.get('seed', 42))
        p = config['p']
        layers = config.get('layers', config.get('lsa_layers'))
        history_len = config['history_len']
        print(f"Loaded model config: p={p}, layers={layers}, history_len={history_len}")

    # Apply saved seed for deterministic evaluation
    set_global_seed(seed)

    # 1. Load Data
    data_dir = artifacts_info['data_dir']
    print(f"Loading data from {data_dir}...")
    train_series = np.load(data_dir / "train_series.npy")
    test_series = np.load(data_dir / "test_series.npy")
    full_series = np.concatenate([train_series, test_series])
    train_size = len(train_series)


    # 2. Fit/Load Models
    print("Fitting Linear AR model...")
    linear_model = LinearARModel(p=p).fit(train_series)

    # Load LSA and/or Softmax models based on experiment type
    if artifacts_info['is_compare_softmax']:
        # Load both linear LSA and softmax models
        print(f"Loading Linear LSA model from {artifacts_info['linear_checkpoint']}...")
        lsa_model = LSATransformerWithHankel(p=p, L=layers, use_softmax=False)
        state = torch.load(artifacts_info['linear_checkpoint'], map_location=args.device)
        # Handle possible torch.compile wrapping (_orig_mod prefix)
        if any(k.startswith('_orig_mod.') for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        lsa_model.load_state_dict(state)
        lsa_model.to(args.device)
        lsa_model.eval()

        print(f"Loading Softmax model from {artifacts_info['softmax_checkpoint']}...")
        softmax_model = LSATransformerWithHankel(p=p, L=layers, use_softmax=True)
        state = torch.load(artifacts_info['softmax_checkpoint'], map_location=args.device)
        # Handle possible torch.compile wrapping (_orig_mod prefix)
        if any(k.startswith('_orig_mod.') for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        softmax_model.load_state_dict(state)
        softmax_model.to(args.device)
        softmax_model.eval()
    else:
        # Original single model logic (assumes LSA model)
        print(f"Loading LSA model from {artifacts_info['lsa_checkpoint']}...")
        lsa_model = LSATransformerWithHankel(p=p, L=layers)
        state = torch.load(artifacts_info['lsa_checkpoint'], map_location=args.device)
        # Handle possible torch.compile wrapping (_orig_mod prefix)
        if any(k.startswith('_orig_mod.') for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        lsa_model.load_state_dict(state)
        lsa_model.to(args.device)
        lsa_model.eval()

    n_preds = len(test_series)
    cot_steps = min(args.cot_steps, n_preds)  # Ensure we don't predict more steps than available in test set

    # 3. Generate Predictions
    print("Generating predictions...")
    # --- Teacher-Forcing (on the full test set) ---
    linear_tf_preds = np.array([linear_model.predict(full_series[i - p : i]) for i in range(train_size, len(full_series))])

    # Generate teacher-forcing predictions for all neural models
    tf_dataset = ARForecastDataset(full_series[train_size - history_len:], p=p, history_len=history_len)

    # LSA model predictions
    lsa_tf_loader = DataLoader(tf_dataset, batch_size=args.batch_size, shuffle=False)
    lsa_tf_preds_list = []
    with torch.no_grad():
        for history, _ in tqdm(lsa_tf_loader, desc="LSA Teacher-Forcing"):
            history = history.to(args.device)
            preds = lsa_model.predict(history)
            lsa_tf_preds_list.extend(preds.cpu().numpy())
    lsa_tf_preds = np.array(lsa_tf_preds_list)

    # Softmax model predictions (only if in compare_softmax mode)
    if artifacts_info['is_compare_softmax']:
        softmax_tf_loader = DataLoader(tf_dataset, batch_size=args.batch_size, shuffle=False)
        softmax_tf_preds_list = []
        with torch.no_grad():
            for history, _ in tqdm(softmax_tf_loader, desc="Softmax Teacher-Forcing"):
                history = history.to(args.device)
                preds = softmax_model.predict(history)
                softmax_tf_preds_list.extend(preds.cpu().numpy())
        softmax_tf_preds = np.array(softmax_tf_preds_list)
    else:
        softmax_tf_preds = None

    # --- Chain-of-Thought (for `cot_steps`) ---
    print(f"Generating CoT predictions for {cot_steps} steps...")
    linear_cot_context = train_series[-p:]
    linear_cot_preds = linear_model.predict_chain_of_thought(linear_cot_context, steps=cot_steps)

    lsa_cot_context_np = train_series[-history_len:]
    lsa_cot_context = torch.from_numpy(lsa_cot_context_np).float().to(args.device)
    lsa_cot_preds = lsa_model.predict_chain_of_thought(lsa_cot_context, steps=cot_steps).cpu().numpy().flatten()

    # Softmax model CoT predictions (only if in compare_softmax mode)
    if artifacts_info['is_compare_softmax']:
        softmax_cot_preds = softmax_model.predict_chain_of_thought(lsa_cot_context, steps=cot_steps).cpu().numpy().flatten()
    else:
        softmax_cot_preds = None
    
    # 4. Calculate MSE and CoT collapse metrics
    test_series_for_cot = test_series[:cot_steps]
    mse = lambda preds, target: np.mean((target - preds[:len(target)])**2)
    linear_tf_mse = mse(linear_tf_preds, test_series)
    lsa_tf_mse = mse(lsa_tf_preds, test_series)
    linear_cot_mse = mse(linear_cot_preds, test_series_for_cot)
    lsa_cot_mse = mse(lsa_cot_preds, test_series_for_cot)

    # Detect CoT collapse-to-mean parameters (calculate before using)
    series_std = float(np.std(train_series)) + 1e-8
    threshold = 0.05 * series_std
    window = max(5, min(20, cot_steps // 5))

    # Softmax metrics (only if in compare_softmax mode)
    softmax_tf_mse = None
    softmax_cot_mse = None
    softmax_collapse_step = None

    if artifacts_info['is_compare_softmax']:
        softmax_tf_mse = mse(softmax_tf_preds, test_series)
        softmax_cot_mse = mse(softmax_cot_preds, test_series_for_cot)

        # Detect CoT collapse-to-mean for Softmax model
        if cot_steps > 0:
            abs_softmax_preds = np.abs(softmax_cot_preds)
            for k in range(0, cot_steps - window + 1):
                if np.all(abs_softmax_preds[k:k+window] < threshold):
                    softmax_collapse_step = k
                    break
    collapse_step = None
    if cot_steps > 0:
        abs_preds = np.abs(lsa_cot_preds)
        for k in range(0, cot_steps - window + 1):
            if np.all(abs_preds[k:k+window] < threshold):
                collapse_step = k
                break

    # Detect CoT collapse-to-mean for Linear AR as well
    linear_collapse_step = None
    if cot_steps > 0:
        abs_lin_preds = np.abs(linear_cot_preds)
        for k in range(0, cot_steps - window + 1):
            if np.all(abs_lin_preds[k:k+window] < threshold):
                linear_collapse_step = k
                break

    # Gaps
    tf_gap = float(lsa_tf_mse - linear_tf_mse)
    cot_gap = float(lsa_cot_mse - linear_cot_mse)

    # Store results
    results = {
        'p': p,
        'layers': layers,
        'history_len': history_len,
        'seed': seed,
        'linear_tf_mse': float(linear_tf_mse),
        'lsa_tf_mse': float(lsa_tf_mse),
        'linear_cot_mse': float(linear_cot_mse),
        'lsa_cot_mse': float(lsa_cot_mse),
        'lsa_vs_linear_tf_gap': tf_gap,
        'lsa_vs_linear_cot_gap': cot_gap,
        'cot_steps': cot_steps,
        'lsa_cot_collapse_step': collapse_step,
        'linear_cot_collapse_step': linear_collapse_step
    }

    # Add softmax metrics if available
    if artifacts_info['is_compare_softmax']:
        results.update({
            'softmax_tf_mse': float(softmax_tf_mse) if softmax_tf_mse is not None else None,
            'softmax_cot_mse': float(softmax_cot_mse) if softmax_cot_mse is not None else None,
            'softmax_cot_collapse_step': softmax_collapse_step,
            'lsa_vs_softmax_tf_gap': float(lsa_tf_mse - softmax_tf_mse) if softmax_tf_mse is not None else None,
            'lsa_vs_softmax_cot_gap': float(lsa_cot_mse - softmax_cot_mse) if softmax_cot_mse is not None else None
        })
        # Save results in comparison_results.json (overwrite or create)
        comparison_results_path = artifacts_dir / "comparison_results.json"
        with open(comparison_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {comparison_results_path}")
    
    if artifacts_info['is_compare_softmax']:
        print("\n--- Evaluation Results (MSE) ---")
        print(f"| Metric           |   Linear AR   |   LSA Model   |  Softmax Model  |")
        print(f"|------------------|---------------|---------------|-----------------|")
        softmax_tf_mse_str = f"{softmax_tf_mse:<14.6f}" if softmax_tf_mse is not None else "       N/A      "
        softmax_cot_mse_str = f"{softmax_cot_mse:<14.6f}" if softmax_cot_mse is not None else "       N/A      "
        print(f"| Teacher-Forcing  |  {linear_tf_mse:<12.6f} |  {lsa_tf_mse:<12.6f} |  {softmax_tf_mse_str} |")
        print(f"| Chain-of-Thought |  {linear_cot_mse:<12.6f} |  {lsa_cot_mse:<12.6f} |  {softmax_cot_mse_str} | (over {cot_steps} steps)")
    else:
        print("\n--- Evaluation Results (MSE) ---")
        print(f"| Metric           |   Linear AR   |   LSA Model   |")
        print(f"|------------------|---------------|---------------|")
        print(f"| Teacher-Forcing  |  {linear_tf_mse:<12.6f} |  {lsa_tf_mse:<12.6f} |")
        print(f"| Chain-of-Thought |  {linear_cot_mse:<12.6f} |  {lsa_cot_mse:<12.6f} | (over {cot_steps} steps)")
    

    
    # 5. Create Plots (unless disabled)
    if not args.no_plots:
        # Ensure output directory exists
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Robust plotting style setup
        try:
            import seaborn as sns  # type: ignore
            sns.set_theme(style="whitegrid")
        except Exception:
            try:
                plt.style.use('seaborn-v0_8')
            except Exception:
                plt.style.use('ggplot')

        # Use explicit relative axes: context x=0..49, predictions x=51..(50+prediction_steps)
        context_len = min(25, train_size)
        rel_context_x = np.arange(0, context_len)
        cot_plot_steps = min(50, cot_steps)
        tf_plot_steps = min(cot_plot_steps, len(test_series))
        pred_x_cot = np.arange(context_len + 1, context_len + 1 + cot_plot_steps)
        pred_x_tf = np.arange(context_len + 1, context_len + 1 + tf_plot_steps)

        # Consistent colors
        gt_color = '#D62728'
        linear_color = '#1F77B4'
        lsa_color = '#2CA02C'
        softmax_color = '#d54b4b'  # Orange for softmax   

        # --- Separate Plot 1: Chain-of-Thought Values ---
        test_series_for_cot = test_series[:cot_plot_steps]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Ground Truth: single color, drawn over context and prediction ranges
        gt_x = np.concatenate([rel_context_x, pred_x_cot])
        gt_y = np.concatenate([train_series[-context_len:], test_series_for_cot])
        ax1.plot(gt_x, gt_y, color=gt_color, linewidth=1.5, label='Ground Truth', alpha=0.5, zorder=1)
        # Model predictions (contrasting colors)
        ax1.plot(pred_x_cot, linear_cot_preds[:cot_plot_steps], color=linear_color, linestyle='-', linewidth=2, label=f'Linear AR (CoT) MSE: {linear_cot_mse:.4f}', zorder=2, alpha=0.8)

        if artifacts_info['is_compare_softmax']:
            ax1.plot(pred_x_cot, lsa_cot_preds[:cot_plot_steps], color=lsa_color, linestyle='--', linewidth=2, label=f'LSA (CoT) MSE: {lsa_cot_mse:.4f}', zorder=2, alpha=0.8)
            ax1.plot(pred_x_cot, softmax_cot_preds[:cot_plot_steps], color=softmax_color, linestyle=':', linewidth=2, label=f'Softmax (CoT) MSE: {softmax_cot_mse:.4f}', zorder=2, alpha=0.8)
        else:
            ax1.plot(pred_x_cot, lsa_cot_preds[:cot_plot_steps], color=lsa_color, linestyle='--', linewidth=2, label=f'LSA (CoT) MSE: {lsa_cot_mse:.4f}', zorder=2, alpha=0.8)

        ax1.axvline(x=context_len, color='red', linestyle='--', linewidth=1.5)
        ax1.set_xlim(0, context_len + 1 + cot_plot_steps)
        ax1.set_title(f'Chain-of-Thought Prediction (Predict {cot_plot_steps} Steps)', fontsize=14)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Value')
        ax1.legend(fontsize=10)
        fig1.tight_layout()
        cot_path = artifacts_dir / f"cot_values_p{p}_L{layers}.pdf"
        fig1.savefig(cot_path, dpi=300)
        plt.close(fig1)
        print(f"Saved CoT values plot to {cot_path}")

        # --- Separate Plot 2: Teacher-Forcing Values ---
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Ground Truth: single color across both ranges
        gt_x = np.concatenate([rel_context_x, pred_x_tf])
        gt_y = np.concatenate([train_series[-context_len:], test_series[:tf_plot_steps]])
        ax2.plot(gt_x, gt_y, color=gt_color, linewidth=1.5, label='Ground Truth', alpha=0.5, zorder=1)
        # Model predictions (contrasting colors)
        ax2.plot(pred_x_tf, linear_tf_preds[:tf_plot_steps], color=linear_color, linestyle='-', linewidth=2, label=f'Linear AR (TF) MSE: {linear_tf_mse:.4f}', zorder=2, alpha=0.8)

        if artifacts_info['is_compare_softmax']:
            ax2.plot(pred_x_tf, lsa_tf_preds[:tf_plot_steps], color=lsa_color, linestyle='--', linewidth=2, label=f'LSA (TF) MSE: {lsa_tf_mse:.4f}', zorder=2, alpha=0.8)
            ax2.plot(pred_x_tf, softmax_tf_preds[:tf_plot_steps], color=softmax_color, linestyle=':', linewidth=2, label=f'Softmax (TF) MSE: {softmax_tf_mse:.4f}', zorder=2, alpha=0.8)
        else:
            ax2.plot(pred_x_tf, lsa_tf_preds[:tf_plot_steps], color=lsa_color, linestyle='--', linewidth=2, label=f'LSA (TF) MSE: {lsa_tf_mse:.4f}', zorder=2, alpha=0.8)

        ax2.axvline(x=context_len, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlim(0, context_len + 1 + tf_plot_steps)
        ax2.set_title(f'Teacher-Forcing Prediction (Predict {tf_plot_steps} Steps)', fontsize=14)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Value')
        ax2.legend(fontsize=10)
        fig2.tight_layout()
        tf_path = artifacts_dir / f"tf_values_p{p}_L{layers}.pdf"
        fig2.savefig(tf_path, dpi=300)
        plt.close(fig2)
        print(f"Saved TF values plot to {tf_path}")

        # --- Separate Plot 3: Chain-of-Thought Error Analysis (Cumulative MSE) ---
        steps = np.arange(1, cot_steps + 1)
        linear_cot_errors = linear_cot_preds - test_series[:cot_steps]
        lsa_cot_errors = lsa_cot_preds - test_series[:cot_steps]
        linear_cot_cumulative_mse = np.cumsum(linear_cot_errors**2) / steps
        lsa_cot_cumulative_mse = np.cumsum(lsa_cot_errors**2) / steps

        # Softmax cumulative MSE (if available)
        softmax_cot_cumulative_mse = None
        if artifacts_info['is_compare_softmax'] and softmax_cot_preds is not None:
            softmax_cot_errors = softmax_cot_preds - test_series[:cot_steps]
            softmax_cot_cumulative_mse = np.cumsum(softmax_cot_errors**2) / steps

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(steps, linear_cot_cumulative_mse, color=linear_color, linestyle='--', label='Linear AR (CoT)', linewidth=2)

        if artifacts_info['is_compare_softmax']:
            ax3.plot(steps, lsa_cot_cumulative_mse, color=lsa_color, linestyle='--', label='LSA Model (CoT)', linewidth=2)
            if softmax_cot_cumulative_mse is not None:
                ax3.plot(steps, softmax_cot_cumulative_mse, color=softmax_color, linestyle=':', label='Softmax Model (CoT)', linewidth=2)
        else:
            ax3.plot(steps, lsa_cot_cumulative_mse, color=lsa_color, linestyle='--', label='LSA Model (CoT)', linewidth=2)

        ax3.set_title('Chain-of-Thought Cumulative MSE over Steps', fontsize=14)
        ax3.set_xlabel('Prediction Step')
        ax3.set_ylabel('Cumulative MSE')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        cot_err_path = artifacts_dir / f"cot_cum_mse_p{p}_L{layers}.pdf"
        fig3.savefig(cot_err_path, dpi=300)
        plt.close(fig3)
        print(f"Saved CoT cumulative MSE plot to {cot_err_path}")

        # --- Separate Plot 4: Teacher-Forcing Error Analysis (Cumulative MSE) ---
        plot_steps = tf_plot_steps
        tf_steps = np.arange(1, plot_steps + 1)
        linear_tf_errors = linear_tf_preds[:plot_steps] - test_series[:plot_steps]
        lsa_tf_errors = lsa_tf_preds[:plot_steps] - test_series[:plot_steps]
        linear_tf_cumulative_mse = np.cumsum(linear_tf_errors**2) / tf_steps
        lsa_tf_cumulative_mse = np.cumsum(lsa_tf_errors**2) / tf_steps

        # Softmax cumulative MSE (if available)
        softmax_tf_cumulative_mse = None
        if artifacts_info['is_compare_softmax']:
            softmax_tf_errors = softmax_tf_preds[:plot_steps] - test_series[:plot_steps]
            softmax_tf_cumulative_mse = np.cumsum(softmax_tf_errors**2) / tf_steps

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(tf_steps, linear_tf_cumulative_mse, color=linear_color, linestyle='--', label='Linear AR (TF)', linewidth=2)

        if artifacts_info['is_compare_softmax']:
            ax4.plot(tf_steps, lsa_tf_cumulative_mse, color=lsa_color, linestyle='--', label='LSA Model (TF)', linewidth=2)
            if softmax_tf_cumulative_mse is not None:
                ax4.plot(tf_steps, softmax_tf_cumulative_mse, color=softmax_color, linestyle=':', label='Softmax Model (TF)', linewidth=2)
        else:
            ax4.plot(tf_steps, lsa_tf_cumulative_mse, color=lsa_color, linestyle='--', label='LSA Model (TF)', linewidth=2)

        ax4.set_title('Teacher-Forcing Cumulative MSE over Steps', fontsize=14)
        ax4.set_xlabel('Prediction Step')
        ax4.set_ylabel('Cumulative MSE')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        tf_err_path = artifacts_dir / f"tf_cum_mse_p{p}_L{layers}.pdf"
        fig4.savefig(tf_err_path, dpi=300)
        plt.close(fig4)
        print(f"Saved TF cumulative MSE plot to {tf_err_path}")

        # --- Commented out: Plot 5 and Plot 6 as requested ---
        # The error histogram and actual-vs-predicted scatter have been disabled per user request.
        if False:
            # Error Distribution Histogram (Teacher-Forcing only)
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            all_linear_errors = linear_tf_errors
            all_lsa_errors = lsa_tf_errors
            ax5.hist(all_linear_errors, bins=30, alpha=0.8, color='#2E8B57', label='Linear AR', density=True, edgecolor='black', linewidth=0.5)
            ax5.hist(all_lsa_errors, bins=30, alpha=0.8, color='#DC143C', label='LSA Model', density=True, edgecolor='black', linewidth=0.5)
            fig5.savefig(artifacts_dir / f"error_hist_p{p}_L{layers}.pdf", dpi=300)
            plt.close(fig5)

            # Actual vs Predicted Scatter Plot
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            ax6.scatter(test_series[:plot_steps], linear_tf_preds[:plot_steps], alpha=0.7, color='#2E8B57', s=30, label='Linear AR', edgecolors='black', linewidth=0.3)
            ax6.scatter(test_series[:plot_steps], lsa_tf_preds[:plot_steps], alpha=0.7, color='#DC143C', s=30, label='LSA Model', edgecolors='black', linewidth=0.3)
            fig6.savefig(artifacts_dir / f"actual_vs_pred_p{p}_L{layers}.pdf", dpi=300)
            plt.close(fig6)
    
    return results

if __name__ == "__main__":
    main()
