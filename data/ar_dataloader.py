"""Data utilities for training time-series models on synthetic AR data.

This module exposes `ARForecastDataset` which creates (history, target) pairs
from a long time-series for one-step-ahead forecasting.

The dataset creates training samples where:
    • Each sample `x` (the history) is a subsequence of `history_len` contiguous points.
    • The target `y` is the single point that immediately follows that subsequence.
    • `history_len` must be > `p` to be compatible with the LSA model's internal
      Hankel matrix processing, which requires at least two columns.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class ARForecastDataset(Dataset):
    """Create (history, target) pairs from a 1-D NumPy array."""

    def __init__(self, series: np.ndarray, p: int, history_len: int):
        if series.ndim != 1:
            raise ValueError("`series` must be one-dimensional")
        if p <= 0:
            raise ValueError("Context length `p` must be positive")
        if history_len <= p:
            raise ValueError(f"`history_len` ({history_len}) must be greater than `p` ({p}).")

        self.p = p
        self.history_len = history_len
        self.series = series.astype(np.float32)

        self.n_samples = len(series) - self.history_len
        if self.n_samples <= 0:
            raise ValueError(f"Time-series (len {len(series)}) is too short for the given `history_len` ({history_len}).")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts the history (x) and target (y) for a given index."""
        start = idx
        end = idx + self.history_len
        history = self.series[start:end]
        target = self.series[end]

        return (
            torch.from_numpy(history),
            torch.tensor(target, dtype=torch.float32),
        )
