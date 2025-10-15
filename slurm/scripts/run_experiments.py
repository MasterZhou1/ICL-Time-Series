#!/usr/bin/env python3
"""
ICL Time Series Experiments Runner
Runs experiments with config generation and execution
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.experiment_runner import run_experiments_slice, run_experiments_slice_resume


def main():
    parser = argparse.ArgumentParser(description='Run ICL Time Series experiments')
    parser.add_argument('--experiments-dir', required=True, help='Experiments directory')
    parser.add_argument('--configs-dir', required=True, help='Configs directory')
    parser.add_argument('--start-idx', type=int, required=True, help='Start index for config slice')
    parser.add_argument('--end-idx', type=int, required=True, help='End index for config slice')
    parser.add_argument('--parallel-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--missing-configs-file', help='File containing missing configs (resume mode)')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda|cpu)')
    
    args = parser.parse_args()
    
    # Handle resume mode with missing configs
    if args.missing_configs_file:
        run_experiments_slice_resume(args.experiments_dir, args.missing_configs_file, args.start_idx, args.end_idx, args.parallel_workers, device=args.device)
    else:
        run_experiments_slice(args.experiments_dir, args.configs_dir, args.start_idx, args.end_idx, args.parallel_workers, device=args.device)


if __name__ == "__main__":
    main() 