"""
Elegant plotting utilities for ICL Time Series experiments.

This module provides a clean, composable API for generating publication-ready plots
with proper statistical analysis and configuration management.

Key Features:
- Clean separation of concerns (config, data processing, plotting)
- Composable functions following single-responsibility principle
- Type-safe configuration with dataclasses
- Robust error handling and graceful degradation
- No redundant code patterns
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

class BandKind(Enum):
    """Types of confidence bands."""
    CI = "ci"
    SEM = "sem"
    STD = "std"

@dataclass
class BandConfig:
    """Configuration for confidence bands and error bars."""
    kind: BandKind = BandKind.CI
    level: float = 0.95
    max_rel_ratio: float = 0.5
    smooth_window: int = 1

    # Y-axis limits
    floor_mode: str = "ols_min"
    floor_factor: float = 0.98
    ceil_mode: str = "percentile"
    ceil_factor: float = 1.05
    ceil_percentile: float = 99.5

@dataclass
class OutlierConfig:
    """Configuration for outlier detection and removal."""
    enable: bool = True
    method: str = "mad"  # "mad" or "iqr"
    k: float = 3.5
    min_group_size: int = 2


@dataclass
class ExcludeConfig:
    """Configuration for parameter exclusions."""
    exclude_combinations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PlotConfig:
    """Complete plotting configuration for an experiment type."""
    experiment_type: str
    band: BandConfig = field(default_factory=BandConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    exclude: ExcludeConfig = field(default_factory=ExcludeConfig)

    # Plot styling
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'
    ])

def setup_plotting_style():
    """Setup consistent plotting style."""
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        try:
            plt.style.use('seaborn-v0_8')
        except Exception:
            try:
                plt.style.use('ggplot')
            except Exception:
                pass  # Use matplotlib defaults
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def load_plot_config() -> Dict[str, PlotConfig]:
    """Load and parse plotting configuration from YAML."""
    try:
        from utils.config import load_config
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / 'configs' / 'plots.yaml') or {}

        configs = {}
        for exp_type, exp_cfg in cfg.get('plots', {}).items():
            # Parse band configuration (directly from exp_cfg, not nested under 'band')
            band = BandConfig(
                kind=BandKind(exp_cfg.get('kind', 'ci')),
                level=float(exp_cfg.get('level', 0.95)),
                max_rel_ratio=float(exp_cfg.get('max_rel_ratio', 0.5)),
                smooth_window=int(exp_cfg.get('smooth_window', 1)),
                floor_mode=exp_cfg.get('floor_mode', 'ols_min'),
                floor_factor=float(exp_cfg.get('floor_factor', 0.98)),
                ceil_mode=exp_cfg.get('ceil_mode', 'percentile'),
                ceil_factor=float(exp_cfg.get('ceil_factor', 1.05)),
                ceil_percentile=float(exp_cfg.get('ceil_percentile', 99.5))
            )

            # Parse outlier configuration
            outlier_cfg = exp_cfg.get('outliers', {})
            outliers = OutlierConfig(
                enable=bool(outlier_cfg.get('enable', True)),
                method=outlier_cfg.get('method', 'mad'),
                k=float(outlier_cfg.get('k', 3.0)),  # Use 3.0 as default to match YAML values
                min_group_size=int(outlier_cfg.get('min_group_size', 1))  # Use 1 as default to match YAML
            )

            # Parse exclusion configuration (allow dict or list of dicts)
            raw_exclude = exp_cfg.get('exclude', [])
            if isinstance(raw_exclude, dict):
                exclude_list = [raw_exclude]
            elif isinstance(raw_exclude, list):
                exclude_list = raw_exclude
            else:
                exclude_list = []
            exclude = ExcludeConfig(
                exclude_combinations=exclude_list
            )

            configs[exp_type] = PlotConfig(
                experiment_type=exp_type,
                band=band,
                outliers=outliers,
                exclude=exclude
            )

        return configs
    except Exception:
        # Return default configurations if loading fails
        return {
            'lsa_layers': PlotConfig('lsa_layers'),
            'context_scaling': PlotConfig('context_scaling')
        }

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================


def filter_exclusions(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    """Remove rows based on parameter exclusions."""
    if df.empty:
        return df

    for exclusion in config.exclude.exclude_combinations:
        if not isinstance(exclusion, dict):
            continue

        mask = pd.Series([True] * len(df), index=df.index)
        for param, value in exclusion.items():
            if param in df.columns:
                mask &= (df[param] == value)
            else:
                # If param doesn't exist in df, this exclusion cannot match any row
                mask &= False

        # Remove rows that match this exclusion
        df = df[~mask].copy()

    return df

def remove_outliers(df: pd.DataFrame, config: PlotConfig) -> Tuple[pd.DataFrame, int]:
    """Remove statistical outliers from dataframe."""
    if df.empty or not config.outliers.enable:
        return df, 0

    key_col = 'history_len' if config.experiment_type == 'context_scaling' else 'lsa_layers'
    required_cols = ['lsa_tf_mse', 'ols_tf_mse']
    if 'tf_gap' in df.columns:
        required_cols.append('tf_gap')

    kept_frames = []
    removed_total = 0

    for (p_val, key_val), group in df.groupby(['p', key_col]):
        if len(group) <= config.outliers.min_group_size:
            kept_frames.append(group)
            continue

        mask_all = np.ones(len(group), dtype=bool)
        for col in required_cols:
            if col not in group.columns:
                continue

            values = group[col].to_numpy(dtype=float)
            if config.outliers.method.lower() == 'iqr':
                q1, q3 = np.quantile(values, [0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - config.outliers.k * iqr
                upper = q3 + config.outliers.k * iqr
                mask = (values >= lower) & (values <= upper)
            else:  # MAD method
                median = np.median(values)
                mad = np.median(np.abs(values - median)) + 1e-12
                mask = np.abs(values - median) <= (config.outliers.k * mad)

            mask_all &= mask

        kept = group[mask_all]
        if len(kept) < config.outliers.min_group_size:
            kept_frames.append(group)
        else:
            kept_frames.append(kept)
            removed_total += (len(group) - len(kept))

    return pd.concat(kept_frames, ignore_index=True) if kept_frames else df, removed_total

def aggregate_metrics(df: pd.DataFrame, experiment_type: str) -> pd.DataFrame:
    """Aggregate metrics with proper grouping and statistics."""
    if experiment_type == 'context_scaling':
        group_cols = ['p', 'history_len']
        x_col = 'history_len'
    elif experiment_type == 'lsa_layers':
        group_cols = ['p', 'lsa_layers']
        x_col = 'lsa_layers'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    metrics = ['lsa_tf_mse', 'ols_tf_mse', 'lsa_cot_mse', 'ols_cot_mse']
    if 'tf_gap' in df.columns:
        metrics.append('tf_gap')
    if 'cot_gap' in df.columns:
        metrics.append('cot_gap')

    agg_dict = {metric: ['mean', 'std', 'count'] for metric in metrics}

    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten column names for cleaner access
    result_dict = {}

    # Add grouping columns
    for col in group_cols:
        result_dict[col] = grouped[col]

    # Add flattened metric columns
    for metric in metrics:
        for stat in ['mean', 'std', 'count']:
            # Check for both tuple format (metric, stat) and flattened format
            tuple_col = (metric, stat)
            flat_col = f"{metric}_{stat}"

            if tuple_col in grouped.columns:
                # Multi-level column exists, create flattened version
                result_dict[flat_col] = grouped[tuple_col]
            elif flat_col in grouped.columns:
                # Already flattened
                result_dict[flat_col] = grouped[flat_col]

    grouped = pd.DataFrame(result_dict)
    return grouped

def prepare_data(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    """Complete data preparation pipeline."""
    if df.empty:
        return df

    # Apply all filters and transformations
    df = filter_exclusions(df, config)

    trimmed_df, removed = remove_outliers(df, config)
    if removed > 0:
        print(f"Removed {removed} outliers for {config.experiment_type}")

    return aggregate_metrics(trimmed_df, config.experiment_type)

# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def compute_band_halfwidth(values: np.ndarray, std: np.ndarray, count: np.ndarray,
                          kind: BandKind, level: float) -> np.ndarray:
    """Compute confidence band half-width."""
    counts = np.maximum(count.astype(float), 1.0)
    std_vals = std.astype(float)

    if kind == BandKind.STD:
        multiplier = level if level > 0 else 1.0
        hw = multiplier * std_vals
    elif kind == BandKind.SEM:
        hw = std_vals / np.sqrt(counts)
    else:  # CI
        z = 1.96 if level >= 0.95 else (1.645 if level >= 0.90 else 1.0)
        hw = z * (std_vals / np.sqrt(counts))

    return np.where(np.isfinite(hw), hw, 0.0)

def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply smoothing to series."""
    if window <= 1 or len(values) <= 2:
        return values

    try:
        return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    except Exception:
        return values

def apply_band_limits(values: np.ndarray, halfwidth: np.ndarray, max_rel_ratio: float) -> np.ndarray:
    """Apply relative ratio limits to band halfwidth."""
    if not np.isfinite(max_rel_ratio) or max_rel_ratio <= 0:
        return halfwidth

    safe_values = np.maximum(np.abs(values), 1e-12)
    cap = 0.5 * max_rel_ratio * safe_values
    return np.minimum(halfwidth, cap)

def create_confidence_bands(x: np.ndarray, y: np.ndarray, std: np.ndarray,
                           count: np.ndarray, config: BandConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Create confidence band coordinates."""
    hw = compute_band_halfwidth(np.zeros_like(y), std, count, config.kind, config.level)
    hw = apply_band_limits(y, hw, config.max_rel_ratio)
    hw = smooth_series(hw, config.smooth_window)

    # Ensure positive bounds for log scale
    floor = np.maximum(np.abs(y) * 1e-6, 1e-12)
    y_lower = np.maximum(y - hw, floor)
    y_upper = y + hw

    return y_lower, y_upper

def set_axis_limits(df: pd.DataFrame, config: BandConfig, experiment_type: str) -> None:
    """Set appropriate y-axis limits."""
    try:
        # Collect all y values
        all_vals = []
        for p_val in sorted(df['p'].unique()):
            subset = df[df['p'] == p_val]
            # Use the correct column names after flattening
            lsa_col = 'lsa_tf_mse_mean' if 'lsa_tf_mse_mean' in subset.columns else 'lsa_tf_mean'
            ols_col = 'ols_tf_mse_mean' if 'ols_tf_mse_mean' in subset.columns else 'ols_tf_mean'
            all_vals.extend([subset[lsa_col].to_numpy(), subset[ols_col].to_numpy()])

        if not all_vals:
            return

        vals = np.concatenate(all_vals)
        vals = vals[np.isfinite(vals)]

        if not vals.size:
            return

        # Compute floor
        if config.floor_mode == 'ols_min':
            ols_vals_list = []
            for p_val in sorted(df['p'].unique()):
                subset = df[df['p'] == p_val]
                ols_col = 'ols_tf_mse_mean' if 'ols_tf_mse_mean' in subset.columns else 'ols_tf_mean'
                arr = subset[ols_col].to_numpy()
                if arr.size:
                    ols_vals_list.append(arr)
            ols_vals = np.concatenate(ols_vals_list) if ols_vals_list else vals
            ols_vals = ols_vals[np.isfinite(ols_vals) & (ols_vals > 0)]
            floor = float(np.min(ols_vals)) * config.floor_factor if ols_vals.size else float(np.min(vals)) * config.floor_factor
        else:
            floor = float(np.min(vals)) * config.floor_factor

        bottom = max(floor, 1e-12)

        # Compute ceiling
        if config.ceil_mode == 'percentile':
            top = float(np.percentile(vals, config.ceil_percentile) * config.ceil_factor)
        else:
            top = float(np.max(vals)) * config.ceil_factor

        if top <= bottom:
            top = bottom * 10

        # Set limits - this should work correctly now that we call it before setting log scale
        plt.ylim(bottom=bottom, top=top)

    except Exception as e:
        print(f"Warning: Could not set y-axis limits: {e}")

def plot_teacher_forcing_comparison(df: pd.DataFrame, config: PlotConfig, output_dir: Path) -> None:
    """Plot LSA vs OLS teacher-forcing performance."""
    plt.figure(figsize=(12, 8))

    x_col = 'history_len' if config.experiment_type == 'context_scaling' else 'lsa_layers'

    for i, p_val in enumerate(sorted(df['p'].unique())):
        subset = df[df['p'] == p_val].sort_values(x_col)
        x = subset[x_col].to_numpy()

        # LSA performance - use correct column names after flattening
        lsa_mean_col = 'lsa_tf_mse_mean' if 'lsa_tf_mse_mean' in subset.columns else 'lsa_tf_mean'
        lsa_std_col = 'lsa_tf_mse_std' if 'lsa_tf_mse_std' in subset.columns else 'lsa_tf_std'
        lsa_count_col = 'lsa_tf_mse_count' if 'lsa_tf_mse_count' in subset.columns else 'lsa_tf_count'

        y_lsa = subset[lsa_mean_col].to_numpy()
        std_lsa = subset[lsa_std_col].to_numpy()
        count_lsa = subset[lsa_count_col].to_numpy()
        y_lsa_lower, y_lsa_upper = create_confidence_bands(x, y_lsa, std_lsa, count_lsa, config.band)

        # OLS performance
        ols_mean_col = 'ols_tf_mse_mean' if 'ols_tf_mse_mean' in subset.columns else 'ols_tf_mean'
        ols_std_col = 'ols_tf_mse_std' if 'ols_tf_mse_std' in subset.columns else 'ols_tf_std'
        ols_count_col = 'ols_tf_mse_count' if 'ols_tf_mse_count' in subset.columns else 'ols_tf_count'

        y_ols = subset[ols_mean_col].to_numpy()
        std_ols = subset[ols_std_col].to_numpy()
        count_ols = subset[ols_count_col].to_numpy()
        y_ols_lower, y_ols_upper = create_confidence_bands(x, y_ols, std_ols, count_ols, config.band)

        color = config.colors[i % len(config.colors)]

        # Plot lines and bands
        plt.plot(x, y_lsa, 'o-', label=f'LSA (p={p_val})', linewidth=2, markersize=5, color=color)
        if np.isfinite(y_lsa_lower).any():
            plt.fill_between(x, y_lsa_lower, y_lsa_upper, color=color, alpha=0.15)

        plt.plot(x, y_ols, 's--', label=f'OLS (p={p_val})', linewidth=2, markersize=5,
                color=color, alpha=0.9)
        if np.isfinite(y_ols_lower).any():
            plt.fill_between(x, y_ols_lower, y_ols_upper, color=color, alpha=0.10)

    # Styling
    title = 'Context Scaling: LSA vs OLS Performance' if config.experiment_type == 'context_scaling' else 'LSA Layers: LSA vs OLS Performance'
    plt.title(title, fontsize=16)
    plt.xlabel('History Length (n)' if config.experiment_type == 'context_scaling' else 'Number of LSA Layers', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set axis limits BEFORE setting log scale
    set_axis_limits(df, config.band, config.experiment_type)
    # plt.yscale('log')

    filename = "context_scaling_lsa_vs_ols_tf.pdf" if config.experiment_type == 'context_scaling' else "lsa_layers_lsa_vs_ols_tf.pdf"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_gap(df: pd.DataFrame, config: PlotConfig, output_dir: Path) -> None:
    """Plot LSA performance gap vs OLS."""
    plt.figure(figsize=(12, 8))

    x_col = 'history_len' if config.experiment_type == 'context_scaling' else 'lsa_layers'

    for i, p_val in enumerate(sorted(df['p'].unique())):
        subset = df[df['p'] == p_val].sort_values(x_col)
        x = subset[x_col].to_numpy()

        # Calculate performance gap - use correct column names after flattening
        if 'tf_gap_mean' in subset.columns:
            gap = subset['tf_gap_mean'].to_numpy()
            gap_std = subset.get('tf_gap_std', pd.Series(np.zeros(len(subset)))).to_numpy()
            gap_count = subset.get('tf_gap_count', pd.Series(np.ones(len(subset)))).to_numpy()
        else:
            # Use the correct column names after flattening
            lsa_mean_col = 'lsa_tf_mse_mean' if 'lsa_tf_mse_mean' in subset.columns else 'lsa_tf_mean'
            ols_mean_col = 'ols_tf_mse_mean' if 'ols_tf_mse_mean' in subset.columns else 'ols_tf_mean'
            lsa_std_col = 'lsa_tf_mse_std' if 'lsa_tf_mse_std' in subset.columns else 'lsa_tf_std'
            ols_std_col = 'ols_tf_mse_std' if 'ols_tf_mse_std' in subset.columns else 'ols_tf_std'
            lsa_count_col = 'lsa_tf_mse_count' if 'lsa_tf_mse_count' in subset.columns else 'lsa_tf_count'
            ols_count_col = 'ols_tf_mse_count' if 'ols_tf_mse_count' in subset.columns else 'ols_tf_count'

            gap = (subset[lsa_mean_col] - subset[ols_mean_col]).to_numpy()
            comb_std = np.sqrt(np.maximum(subset[lsa_std_col], 0.0)**2 +
                             np.maximum(subset[ols_std_col], 0.0)**2)
            comb_cnt = np.minimum(subset[lsa_count_col], subset[ols_count_col])
            gap_std, gap_count = comb_std, comb_cnt

        # Create bands for gap
        gap_lower, gap_upper = create_confidence_bands(x, gap, gap_std, gap_count, config.band)

        color = config.colors[i % len(config.colors)]

        plt.plot(x, gap, 'o-', linewidth=2, markersize=5, color=color,
                label=f'TF Gap (p={p_val})')
        if np.isfinite(gap_lower).any():
            plt.fill_between(x, gap_lower, gap_upper, color=color, alpha=0.15)

    # Styling
    title = 'Context Scaling: LSA Performance Gap vs OLS' if config.experiment_type == 'context_scaling' else 'LSA Layers: LSA Performance Gap vs OLS'
    plt.title(title, fontsize=16)
    plt.xlabel('History Length (n)' if config.experiment_type == 'context_scaling' else 'Number of LSA Layers', fontsize=12)
    plt.ylabel('LSA MSE - OLS MSE (Positive = LSA Worse)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)

    # Set gap-specific limits
    if 'tf_gap_mean' in df.columns:
        gap_vals = df['tf_gap_mean']
    else:
        # Use the correct column names after flattening
        lsa_mean_col = 'lsa_tf_mse_mean' if 'lsa_tf_mse_mean' in df.columns else 'lsa_tf_mean'
        ols_mean_col = 'ols_tf_mse_mean' if 'ols_tf_mse_mean' in df.columns else 'ols_tf_mean'
        gap_vals = df[lsa_mean_col] - df[ols_mean_col]

    gap_clean = gap_vals[np.isfinite(gap_vals.to_numpy())]
    if gap_clean.size:
        lo, hi = np.quantile(gap_clean, [0.01, 0.99])
        plt.ylim(lo, hi)
        
    plt.yscale('log')
    filename = "context_scaling_performance_gap.pdf" if config.experiment_type == 'context_scaling' else "lsa_layers_performance_gap.pdf"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN API
# =============================================================================

def generate_plots(results_df: pd.DataFrame, experiment_type: str, output_dir: Path,
                   exclude_overrides: List[Dict] = None) -> None:
    """Generate all plots for an experiment type."""
    if results_df.empty:
        print(f"No results to plot for {experiment_type}")
        return
    
    # Load base configuration
    configs = load_plot_config()
    config = configs.get(experiment_type)
    if config is None:
        print(f"No configuration found for {experiment_type}")
        return

    # Apply overrides to configuration
    if exclude_overrides:
        config.exclude.exclude_combinations.extend(exclude_overrides)

    # Setup plotting style
    setup_plotting_style()
    output_dir.mkdir(exist_ok=True)

    # Prepare data
    # Apply top-level exclusions before aggregation if any
    if config.exclude.exclude_combinations:
        results_df = filter_exclusions(results_df, config)

    processed_df = prepare_data(results_df, config)

    if processed_df.empty:
        print(f"No data to plot after processing for {experiment_type}")
        return

    # Generate plots
    plot_teacher_forcing_comparison(processed_df, config, output_dir)
    plot_performance_gap(processed_df, config, output_dir)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_plots_from_experiments(experiments_dir: Path, experiment_type: str,
                                   exclude_overrides: List[Dict] = None) -> None:
    """Generate plots by aggregating results from experiment directories."""
    print(f"Generating plots for {experiment_type}...")

    # Load and aggregate results
    results_df = aggregate_results(experiments_dir, experiment_type)

    if results_df.empty:
        print(f"No results to plot for {experiment_type}")
        return

    # Apply exclusions before saving/plotting
    if exclude_overrides:
        # Create a temporary config for exclusion filtering
        temp_config = PlotConfig(experiment_type=experiment_type)
        temp_config.exclude.exclude_combinations = exclude_overrides
        results_df = filter_exclusions(results_df, temp_config)

    # Determine output directories
    try:
        from utils.config import load_config
        repo_root = Path(__file__).resolve().parents[1]
        cfg_file = 'context_scaling.yaml' if experiment_type == 'context_scaling' else 'lsa_layers.yaml'
        exp_cfg = load_config(repo_root / 'configs' / cfg_file)
        results_dir_cfg = exp_cfg.get('output', {}).get('results_dir')
        plots_dir_cfg = exp_cfg.get('output', {}).get('plots_dir')
    except Exception:
        results_dir_cfg = plots_dir_cfg = None

    results_dir = experiments_dir / (results_dir_cfg or f"{experiment_type}/results")
    results_dir.mkdir(exist_ok=True)

    # Save results
    keep_cols = [c for c in results_df.columns if c in {
        'p', 'history_len', 'lsa_layers', 'seed', 'experiment',
        'ols_tf_mse', 'lsa_tf_mse', 'ols_cot_mse', 'lsa_cot_mse', 'tf_gap', 'cot_gap',
        'cot_collapse_step', 'linear_cot_collapse_step', 'cot_steps',
        'learning_rate', 'batch_size', 'epochs_trained', 'patience', 'device',
        'sigma', 'series_len', 'train_split', 'val_split', 'test_split',
        'model_type', 'best_val_loss', 'early_stopped', 'converged'
    }]

    if keep_cols:
        results_df[keep_cols].to_csv(results_dir / f"{experiment_type}_results.csv", index=False)
    results_df.to_csv(results_dir / f"{experiment_type}_full_results.csv", index=False)

    print(f"Saved results to {results_dir}")

    # Generate plots
    plots_dir = experiments_dir / (plots_dir_cfg or f"{experiment_type}/plots")
    generate_plots(results_df, experiment_type, plots_dir, exclude_overrides)


def aggregate_results(experiments_dir: Path, experiment_type: str) -> pd.DataFrame:
    """Aggregate results from all individual model directories."""
    import json

    # Determine experiment directory
    try:
        from utils.config import load_config
        repo_root = Path(__file__).resolve().parents[1]
        cfg_file = 'context_scaling.yaml' if experiment_type == 'context_scaling' else 'lsa_layers.yaml'
        exp_cfg = load_config(repo_root / 'configs' / cfg_file)
        checkpoints_dir_cfg = exp_cfg.get('output', {}).get('checkpoints_dir')
    except Exception:
        checkpoints_dir_cfg = None

    experiment_dir = experiments_dir / (checkpoints_dir_cfg or f"{experiment_type}/checkpoints")

    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return pd.DataFrame()

    results = []
    for results_file in experiment_dir.rglob("results.json"):
        try:
            with open(results_file, 'r') as f:
                result = json.load(f)

            # Normalize schema
            if 'tf_gap' not in result and 'lsa_tf_mse' in result and 'ols_tf_mse' in result:
                result['tf_gap'] = result['lsa_tf_mse'] - result['ols_tf_mse']
            if 'cot_gap' not in result and 'lsa_cot_mse' in result and 'ols_cot_mse' in result:
                result['cot_gap'] = result['lsa_cot_mse'] - result['ols_cot_mse']

            results.append(result)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            print(f"Error reading {results_file}: {e}")

    if not results:
        print(f"No results found for {experiment_type}")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    print(f"Aggregated {len(df)} results for {experiment_type}")
    return df

