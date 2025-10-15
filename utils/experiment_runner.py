"""Unified experiment runner for ICL Time Series experiments."""

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.common import setup_project_path
from utils.common import set_global_seed
from utils.config import load_config, generate_all_configs

# Setup project imports
setup_project_path()
from data.ar_sims import ARDataGenerator
from data.ar_dataloader import ARForecastDataset
from models.lsa_transformer import LSATransformerWithHankel
from models.linear_ar import LinearARModel
from utils.wandb_utils import WandbRun


def train_single_model_gpu(config: Dict[str, Any], experiments_dir: str, device: str = "cuda") -> bool:
    """Train a single LSA model directly on GPU without subprocess overhead.
    
    This enables true GPU parallelism when called from a parallel executor.
    
    Args:
        config: Configuration dictionary with model parameters
        experiments_dir: Base experiments directory
        device: Training device (cuda or cpu)
        
    Returns:
        True if training successful, False otherwise
    """
    try:
        # Extract parameters
        p = int(config.get('p'))
        history_len = int(config.get('history_len'))
        lsa_layers = int(config.get('lsa_layers', 3))
        experiment_type = str(config.get('experiment'))
        seed = int(config.get('seed', 42))
        use_softmax = bool(config.get('use_softmax', False))

        # Apply seed for reproducibility
        set_global_seed(seed)
        
        # Load experiment-specific config for paths
        configs_dir = Path(experiments_dir).parent / "configs"
        if experiment_type == "context_scaling":
            exp_config = load_config(configs_dir / "context_scaling.yaml")
        else:
            exp_config = load_config(configs_dir / "lsa_layers.yaml")
        
        # Get output paths from YAML config
        base_dir = Path(experiments_dir)
        checkpoints_dir = exp_config.get('output', {}).get('checkpoints_dir', f"{experiment_type}/checkpoints")
        
        # Create model directory (seed-aware, informative)
        model_dir = f"seed{seed}_p{p}_n{history_len}_L{lsa_layers}{'_softmax' if use_softmax else ''}"
        output_dir = base_dir / checkpoints_dir / model_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_dir / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Get training parameters from config
        data_config = exp_config.get('data', {})
        training_config = exp_config.get('training', {})

        # Capture actual parameters used (from config dict, not config files)
        actual_sigma = config.get('sigma', data_config.get('sigma', 0.05))
        actual_series_len = config.get('series_len', data_config.get('series_len', 50000))
        actual_train_split = config.get('train_split', data_config.get('train_split', 0.7))
        actual_val_split = config.get('val_split', data_config.get('val_split', 0.15))
        actual_lr = config.get('lr', training_config.get('lr', 1e-3))
        actual_batch_size = config.get('batch_size', training_config.get('batch_size', 256))
        actual_patience = config.get('patience', training_config.get('patience', 20))
        
        # Check for cached dataset (seed-aware)
        shared_data_dir = base_dir / "shared_data" / f"p{p}" / f"seed{seed}"
        if shared_data_dir.exists():
            train_series = np.load(shared_data_dir / "train_series.npy")
            test_series = np.load(shared_data_dir / "test_series.npy")
        else:
            # Generate new dataset
            shared_data_dir.mkdir(parents=True, exist_ok=True)
            generator = ARDataGenerator(
                p=p,
                sigma=actual_sigma,
                sequence_length=actual_series_len,
                random_seed=seed,
            )
            series = generator.generate_series()
            train_end = int(len(series) * actual_train_split)
            train_series, test_series = series[:train_end], series[train_end:]
            
            # Cache dataset
            np.save(shared_data_dir / "train_series.npy", train_series)
            np.save(shared_data_dir / "test_series.npy", test_series)
        
        # Create datasets
        train_dataset = ARForecastDataset(train_series, p=p, history_len=history_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True
        )
        
        # Use test set as validation for simplicity
        val_dataset = ARForecastDataset(test_series, p=p, history_len=history_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            shuffle=False
        )
        
        # Initialize model
        model = LSATransformerWithHankel(p=p, L=lsa_layers, use_softmax=use_softmax).to(device)
        
        # Optional torch.compile for performance
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
        except Exception:
            pass
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(actual_lr)
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        patience = actual_patience
        # Allow quick test override via env var
        max_epochs = int(os.getenv('ICL_TRAIN_EPOCHS', training_config.get('epochs', 200)))
        
        # Persist series for reproducibility of this run
        try:
            import numpy as _np
            _np.save(output_dir / "train_series.npy", train_series)
            _np.save(output_dir / "test_series.npy", test_series)
        except Exception:
            pass

        epoch_count = 0
        
         # Load .env file explicitly before WANDB initialization
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv not available

        # Use WANDB_PROJECT from .env file, fallback to hardcoded name
        project_name = os.getenv("WANDB_PROJECT", "icl-time-series")
    
        with WandbRun(
            project=project_name, 
            config=config, 
            group=experiment_type,
            name=f"seed{seed}_p{p}_n{history_len}_L{lsa_layers}{'_softmax' if use_softmax else ''}"
        ) as wb:
            for epoch in range(max_epochs):
                epoch_count += 1
                # Training
                model.train()
                total_train_loss = 0.0
                for history, target in train_loader:
                    history, target = history.to(device), target.to(device)
                    optimizer.zero_grad()
                    preds = model.predict(history)
                    loss = criterion(preds, target)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for history, target in val_loader:
                        history, target = history.to(device), target.to(device)
                        preds = model.predict(history)
                        val_loss += criterion(preds, target).item() * len(history)
                
                val_loss /= len(val_dataset)
                
                # Log to wandb
                if wb:
                    wb.log({
                        'train/loss': total_train_loss / len(train_loader),
                        'val/loss': val_loss,
                        'epoch': epoch + 1
                    }, step=epoch + 1)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        # Fit OLS baseline and evaluate
        ols_model = LinearARModel(p=p).fit(train_series)
        
        # Teacher-forcing evaluation
        full_series = np.concatenate([train_series, test_series])
        train_size = len(train_series)
        
        # OLS teacher-forcing (vectorized creation of contexts for clarity)
        ols_tf_preds = np.array([ols_model.predict(full_series[i - p : i]) for i in range(train_size, len(full_series))])
        
        # LSA teacher-forcing
        model.eval()
        test_dataset = ARForecastDataset(
            full_series[train_size - history_len:], 
            p=p, 
            history_len=history_len
        )
        eval_batch = exp_config.get('evaluation', {}).get('eval_batch_size', 512)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False)
        
        lsa_tf_preds_list = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                preds = model.predict(batch_x)
                lsa_tf_preds_list.extend(preds.cpu().numpy())
        lsa_tf_preds = np.array(lsa_tf_preds_list)
        
        # Chain-of-thought evaluation
        cot_steps = min(100, len(test_series))
        
        # OLS chain-of-thought
        ols_cot_context = train_series[-p:]
        ols_cot_preds = ols_model.predict_chain_of_thought(ols_cot_context, steps=cot_steps)
        
        # LSA chain-of-thought
        lsa_cot_context = torch.from_numpy(train_series[-history_len:]).float().to(device)
        lsa_cot_preds = model.predict_chain_of_thought(lsa_cot_context, steps=cot_steps).cpu().numpy().flatten()
        
        # Calculate MSE
        ols_tf_mse = np.mean((test_series - ols_tf_preds)**2)
        lsa_tf_mse = np.mean((test_series - lsa_tf_preds)**2)
        
        test_series_cot = test_series[:cot_steps]
        ols_cot_mse = np.mean((test_series_cot - ols_cot_preds)**2)
        lsa_cot_mse = np.mean((test_series_cot - lsa_cot_preds)**2)
        
        # Save results
        results = {
            'seed': int(seed),
            'p': int(p),
            'history_len': int(history_len),
            'lsa_layers': int(lsa_layers),
            'experiment': str(experiment_type),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': int(epoch_count),
            'ols_tf_mse': float(ols_tf_mse),
            'lsa_tf_mse': float(lsa_tf_mse),
            'ols_cot_mse': float(ols_cot_mse),
            'lsa_cot_mse': float(lsa_cot_mse),
            'tf_gap': float(lsa_tf_mse - ols_tf_mse),
            'cot_gap': float(lsa_cot_mse - ols_cot_mse),
            'lsa_beats_ols_tf': bool(float(lsa_tf_mse) < float(ols_tf_mse)),
            'lsa_beats_ols_cot': bool(float(lsa_cot_mse) < float(ols_cot_mse)),
            'cot_steps': int(cot_steps),
            # Training parameters - actual values used during training
            'learning_rate': float(actual_lr),
            'batch_size': int(actual_batch_size),
            'patience': int(actual_patience),
            'device': str(device),
            'sigma': float(actual_sigma),
            'series_len': int(actual_series_len),
            'train_split': float(actual_train_split),
            'val_split': float(actual_val_split),
            'test_split': 1.0 - actual_train_split - actual_val_split,
            'model_type': 'LSATransformerWithHankel',
            'use_softmax': bool(use_softmax),
            'early_stopped': epoch_count < max_epochs,  # If didn't reach max_epochs, likely early stopped
            'converged': best_val_loss < 1.0  # Simple convergence check
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log final metrics to wandb
        if wb:
            wb.log({
                'metrics/ols_tf_mse': results['ols_tf_mse'],
                'metrics/lsa_tf_mse': results['lsa_tf_mse'],
                'metrics/ols_cot_mse': results['ols_cot_mse'],
                'metrics/lsa_cot_mse': results['lsa_cot_mse'],
                'metrics/tf_gap': results['tf_gap'],
                'metrics/cot_gap': results['cot_gap']
            })
        
        return True
        
    except Exception as e:
        print(f"Error training model {config}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False


def run_experiments_slice_gpu_parallel(experiments_dir: str, configs_dir: str, start_idx: int, 
                                      end_idx: int, parallel_workers: int = 8, device: str = "cuda") -> None:
    """Run experiments for the given slice of configurations using sequential GPU training.
    
    Due to CUDA threading issues, we train models sequentially but efficiently.
    
    Args:
        experiments_dir: Base experiments directory
        configs_dir: Configuration files directory
        start_idx: Start index of configuration slice
        end_idx: End index of configuration slice
        parallel_workers: Number of parallel workers (unused due to GPU constraints)
        device: Training device
    """
    # Generate all configs
    all_configs = generate_all_configs(configs_dir)
    configs = all_configs[start_idx:end_idx]
    
    if not configs:
        print("No configurations to run in this slice")
        return
    
    print(f"Running {len(configs)} configurations sequentially on {device}")
    
    # Train models sequentially to avoid CUDA threading issues
    completed = 0
    successful = 0
    
    for i, config in enumerate(configs):
        completed += 1
        print(f"Processing config {i+1}/{len(configs)}: {config}")
        
        try:
            success = train_single_model_gpu(config, experiments_dir, device)
            if success:
                successful += 1
                print(f"✅ Success: {config.get('experiment')} p={config.get('p')} L={config.get('lsa_layers')}")
            else:
                print(f"❌ Failed: {config.get('experiment')} p={config.get('p')} L={config.get('lsa_layers')}")
        except Exception as e:
            print(f"❌ Exception: {config.get('experiment')} p={config.get('p')} - {e}")
        
        print(f"Progress: {completed}/{len(configs)} completed, {successful} successful")
    
    print(f"Completed slice: {successful}/{len(configs)} successful")


def run_experiments_slice(experiments_dir: str, configs_dir: str, start_idx: int, 
                         end_idx: int, parallel_workers: int = 8, device: str = "cuda") -> None:
    """Run experiments for the given slice of configurations."""
    run_experiments_slice_gpu_parallel(experiments_dir, configs_dir, start_idx, end_idx, parallel_workers, device)


def run_experiments_slice_resume(experiments_dir: str, missing_configs_file: str, start_idx: int, 
                                end_idx: int, parallel_workers: int = 8, device: str = "cuda") -> None:
    """Run experiments for the given slice of missing configurations (resume mode)."""
    # Load missing configs from file
    with open(missing_configs_file, 'r') as f:
        missing_configs = json.load(f)
    
    # Get the slice of missing configs
    configs = missing_configs[start_idx:end_idx]
    
    if not configs:
        print("No missing configurations to run in this slice")
        return
    
    print(f"Running {len(configs)} missing configurations sequentially on {device}")
    
    # Train models sequentially to avoid CUDA threading issues
    completed = 0
    successful = 0
    
    for i, config in enumerate(configs):
        completed += 1
        print(f"Processing missing config {i+1}/{len(configs)}: {config}")
        
        try:
            success = train_single_model_gpu(config, experiments_dir, device)
            if success:
                successful += 1
                print(f"✅ Success: {config.get('experiment')} p={config.get('p')} L={config.get('lsa_layers')}")
            else:
                print(f"❌ Failed: {config.get('experiment')} p={config.get('p')} L={config.get('lsa_layers')}")
        except Exception as e:
            print(f"❌ Exception: {config.get('experiment')} p={config.get('p')} - {e}")
        
        print(f"Progress: {completed}/{len(configs)} completed, {successful} successful")
    
    print(f"Completed resume slice: {successful}/{len(configs)} successful")

