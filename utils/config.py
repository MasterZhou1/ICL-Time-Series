"""
Unified configuration management for ICL Time Series experiments.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and merge configuration files."""
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base config if specified
    if 'base' in config:
        base_path = config_path.parent / config['base']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Deep-merge configs so experiment overrides only provided keys
        def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result: Dict[str, Any] = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = deep_update(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_update(base_config, config)
    
    return config


def generate_context_scaling_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate context scaling experiment configurations."""
    configs = []
    p_values = config['experiment']['p_values']
    history_len_offsets = config['experiment']['history_len_offsets']
    lsa_layers = config['experiment']['lsa_layers']
    seeds = config['experiment'].get('seeds', [42])
    use_softmax = bool(config['experiment'].get('use_softmax', False))
    
    for seed in seeds:
        for p in p_values:
            for offset in history_len_offsets:
                configs.append({
                    'experiment': 'context_scaling',
                    'seed': int(seed),
                    'p': p,
                    'history_len': p + offset,
                    'lsa_layers': lsa_layers,
                    'use_softmax': use_softmax,
                })
    return configs


def generate_lsa_layers_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate LSA layers experiment configurations."""
    configs = []
    p_values = config['experiment']['p_values']
    exp_cfg = config['experiment']
    lsa_layers = exp_cfg['lsa_layers']
    seeds = exp_cfg.get('seeds', [42])
    use_softmax = bool(exp_cfg.get('use_softmax', False))

    # Prefer a fixed history length if provided; otherwise fall back to ratio; else default to p + 2
    fixed_history_len = exp_cfg.get('history_len', None)
    history_len_ratio = exp_cfg.get('history_len_ratio', None)
    
    for seed in seeds:
        for p in p_values:
            for layers in lsa_layers:
                if fixed_history_len is not None:
                    history_len = int(fixed_history_len)
                elif history_len_ratio is not None:
                    history_len = int(p * float(history_len_ratio))
                else:
                    history_len = int(p) + 2

                configs.append({
                    'experiment': 'lsa_layers',
                    'seed': int(seed),
                    'p': p,
                    'lsa_layers': layers,
                    'history_len': history_len,
                    'use_softmax': use_softmax,
                })
    return configs


def generate_all_configs(configs_dir: str | Path) -> List[Dict[str, Any]]:
    """Generate all experiment configurations."""
    configs_dir = Path(configs_dir)
    
    # Load configurations
    context_config = load_config(configs_dir / "context_scaling.yaml")
    lsa_config = load_config(configs_dir / "lsa_layers.yaml")

    # Generate all configurations
    all_configs = []
    all_configs.extend(generate_context_scaling_configs(context_config))
    all_configs.extend(generate_lsa_layers_configs(lsa_config))

    return all_configs


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[2]


def get_checkpoint_directories(configs_dir: str | Path) -> tuple[str, str]:
    """Get checkpoint directories for both experiments."""
    configs_dir = Path(configs_dir)
    
    # Load configurations
    context_config = load_config(configs_dir / "context_scaling.yaml")
    lsa_config = load_config(configs_dir / "lsa_layers.yaml")
    
    # Get checkpoint directories
    context_dir = f"{context_config['output']['base_dir']}/{context_config['output']['checkpoints_dir']}"
    lsa_dir = f"{lsa_config['output']['base_dir']}/{lsa_config['output']['checkpoints_dir']}"
    
    return context_dir, lsa_dir 