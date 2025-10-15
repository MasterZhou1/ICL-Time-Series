#!/usr/bin/env python3
"""
ICL Time Series - Missing Models Checker
Find which model checkpoints are missing
"""

import json
import argparse
from pathlib import Path
import sys

# Add project root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.config import generate_all_configs, load_config


def check_missing_models(configs_dir, experiments_dir, output_file, min_epochs: int = 0):
    """Find missing model checkpoints."""
    # Generate all expected configurations
    all_configs = generate_all_configs(configs_dir)
    configs_dir = Path(configs_dir)
    experiments_dir = Path(experiments_dir)

    # Resolve checkpoint directories from YAML (honor dynamic paths)
    try:
        context_cfg = load_config(configs_dir / "context_scaling.yaml")
        lsa_cfg = load_config(configs_dir / "lsa_layers.yaml")
        context_ckpt_root = experiments_dir / context_cfg.get('output', {}).get('checkpoints_dir', 'context_scaling/checkpoints')
        lsa_ckpt_root = experiments_dir / lsa_cfg.get('output', {}).get('checkpoints_dir', 'lsa_layers/checkpoints')
    except Exception:
        context_ckpt_root = experiments_dir / 'context_scaling/checkpoints'
        lsa_ckpt_root = experiments_dir / 'lsa_layers/checkpoints'
    
    # Check which models exist
    missing_configs = []
    for config in all_configs:
        exp = config['experiment']
        p = config['p']
        history_len = config['history_len']
        lsa_layers = config['lsa_layers']
        seed = config.get('seed', 42)

        # Build checkpoint path (include seed in directory name)
        model_dir = f"seed{seed}_p{p}_n{history_len}_L{lsa_layers}"
        base_dir = context_ckpt_root if exp == 'context_scaling' else lsa_ckpt_root
        model_path = base_dir / model_dir
        checkpoint_path = model_path / "best_model.pt"
        
        # Missing if no checkpoint
        if not checkpoint_path.exists():
            missing_configs.append(config)
            continue

        # If a minimum epoch requirement is specified, ensure training reached it
        if min_epochs and min_epochs > 0:
            results_file = model_path / 'results.json'
            epochs_trained = None
            if results_file.exists():
                try:
                    with open(results_file, 'r') as rf:
                        results = json.load(rf)
                        epochs_trained = int(results.get('epochs_trained', 0))
                except Exception:
                    epochs_trained = None
            if epochs_trained is None or epochs_trained < min_epochs:
                missing_configs.append(config)
    
    # Save missing configs
    with open(output_file, 'w') as f:
        json.dump(missing_configs, f, indent=2)
    
    print(f"Found {len(missing_configs)} missing models out of {len(all_configs)} total")
    return len(missing_configs)


def main():
    parser = argparse.ArgumentParser(description='Check for missing ICL model checkpoints')
    parser.add_argument('--configs-dir', required=True, help='Configs directory')
    parser.add_argument('--experiments-dir', required=True, help='Experiments directory')
    parser.add_argument('--output-file', required=True, help='Output JSON file')
    parser.add_argument('--min-epochs', type=int, default=0, help='Minimum epochs required to consider a model complete')
    
    args = parser.parse_args()
    check_missing_models(args.configs_dir, args.experiments_dir, args.output_file, min_epochs=args.min_epochs)


if __name__ == "__main__":
    main() 