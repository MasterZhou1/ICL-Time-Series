#!/usr/bin/env python3
"""
ICL Time Series Config Generator
Generates experiment configurations from YAML files
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.config import generate_all_configs


def generate_experiment_configs(configs_dir, output_file):
    """Generate all experiment configurations and save to file."""
    # Generate all configurations
    all_configs = generate_all_configs(configs_dir)

    print(f'Total expected configurations: {len(all_configs)}')

    # Save to output file
    with open(output_file, 'w') as f:
        yaml.dump(all_configs, f)
    
    return len(all_configs)


def main():
    parser = argparse.ArgumentParser(description='Generate ICL Time Series experiment configurations')
    parser.add_argument('--configs-dir', required=True, help='Directory containing config files')
    parser.add_argument('--output-file', required=True, help='Output file for generated configs')
    
    args = parser.parse_args()
    
    count = generate_experiment_configs(args.configs_dir, args.output_file)
    print(f'Generated {count} configurations and saved to {args.output_file}')


if __name__ == "__main__":
    main() 