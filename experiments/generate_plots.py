#!/usr/bin/env python3
"""
Standalone script to generate plots from existing experiment results.
Useful for generating plots without re-running experiments.
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Ensure repo root is on sys.path before importing project modules (works from experiments/)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.common import setup_project_path
setup_project_path(levels_up=2)

from utils.plotting import generate_plots_from_experiments


# Optional torch import for environments focused only on plotting
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

def get_default_device() -> str:
    if _TORCH_AVAILABLE:
        try:
            return 'cuda' if torch.cuda.is_available() else 'cpu'  # type: ignore[attr-defined]
        except Exception:
            return 'cpu'
    return 'cpu'

def main():
    parser = argparse.ArgumentParser(description='Generate plots from existing experiment results')
    parser.add_argument('--experiments-dir', default='experiments', 
                       help='Experiments directory (default: experiments)')
    parser.add_argument('--experiment-type', choices=['context_scaling', 'lsa_layers', 'all'],
                       default='all', help='Experiment type to plot (default: all)')
    parser.add_argument('--exclude', type=str, nargs='+', action='append',
                       help='Exclude parameter combinations. Format: --exclude param1=value1 param2=value2. Can be used multiple times.')
    
    args = parser.parse_args()

    # Process exclude arguments into a format that can be used to override config
    exclude_overrides = {}
    if args.exclude:
        for exclude_group in args.exclude:
            if len(exclude_group) % 2 != 0:
                print(f"Warning: Invalid exclude argument format: {exclude_group}. Expected param=value pairs.")
                continue

            for i in range(0, len(exclude_group), 2):
                param = exclude_group[i]
                value = exclude_group[i + 1]

                # Try to parse value as int, float, or keep as string
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                # Group by experiment type (we'll apply to all for now, or could be more specific)
                if 'lsa_layers' not in exclude_overrides:
                    exclude_overrides['lsa_layers'] = []
                if 'context_scaling' not in exclude_overrides:
                    exclude_overrides['context_scaling'] = []

                exclude_overrides['lsa_layers'].append({param: value})
                exclude_overrides['context_scaling'].append({param: value})

    # Resolve experiments_dir robustly from repo root or current working dir
    candidates = [
        Path(args.experiments_dir),
        repo_root / args.experiments_dir,
        repo_root / 'experiments',
        Path(__file__).resolve().parent  # the experiments/ dir containing this script
    ]
    experiments_dir: Path | None = None
    for c in candidates:
        if c.exists() and c.is_dir():
            # Accept if any expected subdir exists; do not hard-require context_scaling
            if any((c / sub).exists() for sub in ['context_scaling', 'lsa_layers']):
                experiments_dir = c
                break
            # As a fallback, accept c even if subdirs are not present yet
            experiments_dir = c
            break
    if experiments_dir is None:
        print(f"Error: Experiments directory not found. Tried: {[str(c) for c in candidates]}")
        sys.exit(1)
    
    if args.experiment_type == 'all':
        # Generate plots for all experiment types
        experiment_types = ['context_scaling', 'lsa_layers']
    else:
        experiment_types = [args.experiment_type]
    
    for experiment_type in experiment_types:
        experiment_dir = experiments_dir / experiment_type
        # Support YAML-configured subpaths if present
        configured_plots_dir = None
        try:
            from utils.config import load_config
            cfg_file = 'context_scaling.yaml' if experiment_type == 'context_scaling' else 'lsa_layers.yaml'
            exp_cfg = load_config(repo_root / 'configs' / cfg_file)
            configured_plots_dir = exp_cfg.get('output', {}).get('plots_dir')
        except Exception:
            configured_plots_dir = None

        if configured_plots_dir:
            experiment_dir = experiments_dir / Path(configured_plots_dir).parent

        if not experiment_dir.exists():
            print(f"Warning: Experiment directory not found: {experiment_dir}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Preparing visualization for {experiment_type}")
        print(f"{'='*50}")

        # Aggregate and generate plots/analysis
        try:
            generate_plots_from_experiments(experiments_dir, experiment_type,
                                           exclude_overrides.get(experiment_type, []))
            print(f"✅ Successfully generated plots for {experiment_type}")
        except Exception as e:
            print(f"❌ Error generating plots for {experiment_type}: {e}")
    
    print(f"\n{'='*50}")
    print("Plot generation completed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 