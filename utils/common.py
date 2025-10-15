"""Common utilities for ICL Time Series project."""

import os
import sys
from pathlib import Path
from typing import Optional
import random
import numpy as np

try:
    import torch  # type: ignore
except Exception:  # torch may be unavailable in some envs
    torch = None  # type: ignore


# Fix OpenMP library conflict on macOS (set once globally)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def setup_project_path(levels_up: int = 2) -> Path:
    """Setup project path and add to sys.path if needed.
    
    Args:
        levels_up: Number of directory levels to go up from current file
        
    Returns:
        Project root path
    """
    # Get the calling file's path
    frame = sys._getframe(1)
    try:
        current_file = Path(frame.f_globals['__file__'])
        # Calculate project root
        repo_root = current_file.resolve().parents[levels_up - 1]
    except KeyError:
        # Fallback when __file__ is not available (e.g., in -c scripts)
        repo_root = get_project_root()
    
    # Add to sys.path if not already present
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    return repo_root


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for key files
    current = Path(__file__).resolve()
    
    for parent in [current] + list(current.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    
    # Fallback to one level up from utils
    return Path(__file__).resolve().parents[1]


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Seed value to apply.
        deterministic_torch: If True, configure torch backends for determinism when available.
    """
    # Set CuBLAS workspace config for deterministic behavior before any torch operations
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, 'cuda') and torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
            if deterministic_torch:
                try:
                    torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    import torch.backends.cudnn as cudnn  # type: ignore
                    cudnn.deterministic = True
                    cudnn.benchmark = False
                except Exception:
                    pass
        except Exception:
            # Best-effort; do not fail hard on torch seeding issues
            pass
