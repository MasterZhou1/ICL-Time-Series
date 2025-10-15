"""
AR(p) Time Series Data Generator.

This module provides functionality for generating synthetic autoregressive
time series data with guaranteed weak stationarity. The generator ensures
that all characteristic polynomial roots lie outside the unit circle.

The AR(p) model generated follows:
    X_t = φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t

where ε_t ~ N(0, σ²) and the coefficients φᵢ satisfy stationarity constraints.

Classes:
    ARDataGenerator: Main generator class with stationarity validation.

Functions:
    create_ar_data: Convenience function for creating AR data from config files.
"""

import numpy as np
import yaml
from typing import Tuple, Dict, Any


class ARDataGenerator:
    """Generates weakly stationary AR(p) time series data"""
    
    def __init__(self, p: int, sigma: float, sequence_length: int, 
                 random_seed: int = 42):
        """
        Initialize AR(p) data generator.
        
        Args:
            p: AR order (must be positive)
            sigma: Noise standard deviation (must be positive)
            sequence_length: Total length of time series (must be positive)
            random_seed: Random seed for reproducibility
        """
        if p <= 0 or sigma <= 0 or sequence_length <= 0:
            raise ValueError("p, sigma, and sequence_length must be positive")
            
        self.p = p
        self.sigma = sigma
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        self.ar_coeffs = self._generate_stationary_coeffs()
        
    def _generate_stationary_coeffs(self) -> np.ndarray:
        """
        Generate AR coefficients ensuring weak stationarity.
        
        Weak stationarity requires all roots of the characteristic polynomial
        φ(z) = 1 - φ₁z - φ₂z² - ... - φ_pz^p to be outside the unit circle.
        
        Returns:
            Stationary AR coefficients
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            coeffs = np.random.uniform(-0.9, 0.9, self.p)
            if self._is_stationary(coeffs):
                return coeffs
        
        # Fallback: use conservative AR(1) coefficients
        print(f"Warning: Could not find stationary AR({self.p}) coefficients. Using AR(1) with φ=0.5")
        return np.array([0.5])
    
    def _is_stationary(self, coeffs: np.ndarray) -> bool:
        """
        Check if AR coefficients satisfy weak stationarity.
        
        Args:
            coeffs: AR coefficients [φ₁, φ₂, ..., φ_p]
            
        Returns:
            True if all characteristic polynomial roots are outside unit circle
        """
        # Characteristic polynomial: φ(z) = 1 - φ₁z - φ₂z² - ... - φ_pz^p
        poly_coeffs = np.concatenate([-coeffs[::-1], [1]])
        roots = np.roots(poly_coeffs)
        
        # All roots must be outside unit circle for stationarity
        return all(abs(root) > 1.0 for root in roots)
    
    def _solve_yule_walker(self) -> float:
        """Solve Yule–Walker equations to obtain γ₀ (variance)."""
        p = self.p
        phi = self.ar_coeffs

        # Build (p+1) × (p+1) coefficient matrix A and RHS vector b
        A = np.zeros((p + 1, p + 1))
        b = np.zeros(p + 1)

        # Equation for k = 0: γ₀ - Σ φ_j γ_j = σ²
        A[0, 0] = 1.0
        A[0, 1:] = -phi
        b[0] = self.sigma ** 2

        # Equations for k = 1 … p: γ_k - Σ φ_j γ_{|k-j|} = 0
        for k in range(1, p + 1):
            A[k, k] = 1.0
            for j, phi_j in enumerate(phi, start=1):
                idx = abs(k - j)
                A[k, idx] -= phi_j
        # Solve for γ vector
        gamma = np.linalg.solve(A, b)
        return gamma[0]  # γ₀

    def _get_stationary_variance(self) -> float:
        """Calculate variance of stationary AR(p) process via Yule–Walker equations."""
        return self._solve_yule_walker()
    
    def generate_series(self, burn_in: int = 100) -> np.ndarray:
        """
        Generate AR(p) time series with burn-in period.
        
        Args:
            burn_in: Number of initial observations to discard
            
        Returns:
            AR(p) time series of length sequence_length
        """
        total_length = self.sequence_length + burn_in
        series = np.zeros(total_length)
        
        # Initialize with stationary distribution
        stationary_var = self._get_stationary_variance()
        series[:self.p] = np.random.normal(0, np.sqrt(stationary_var), self.p)
        
        # Generate AR(p) process: X_t = ΣφᵢX_{t-i} + ε_t
        for t in range(self.p, total_length):
            # Align coefficients with lags so φ₁ multiplies X_{t-1}, φ₂ multiplies X_{t-2}, ...
            ar_component = np.sum(self.ar_coeffs * series[t-self.p:t][::-1])
            series[t] = ar_component + np.random.normal(0, self.sigma)
        
        return series[burn_in:]
    
    def generate_multiple_series(self, n_series: int, burn_in: int = 100) -> np.ndarray:
        """
        Generate multiple independent AR(p) time series.
        
        Args:
            n_series: Number of series to generate
            burn_in: Number of initial observations to discard
            
        Returns:
            Array of shape (n_series, sequence_length)
        """
        series_list = []
        
        for i in range(n_series):
            np.random.seed(self.random_seed + i)
            series_list.append(self.generate_series(burn_in))
        
        return np.array(series_list)
    
    def prepare_data(self, train_split: float = 0.8, val_split: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/validation/test splits.
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            
        Returns:
            Tuple of (train_series, val_series, test_series)
        """
        if not (0 < train_split < 1 and 0 < val_split < 1 and train_split + val_split < 1):
            raise ValueError("Invalid split proportions")
            
        series = self.generate_series()
        
        train_end = int(self.sequence_length * train_split)
        val_end = int(self.sequence_length * (train_split + val_split))
        
        return (series[:train_end], 
                series[train_end:val_end], 
                series[val_end:])
    
    def get_stationarity_info(self) -> Dict[str, Any]:
        """
        Get comprehensive stationarity information.
        
        Returns:
            Dictionary with stationarity diagnostics
        """
        poly_coeffs = np.concatenate([-self.ar_coeffs[::-1], [1]])
        roots = np.roots(poly_coeffs)
        
        return {
            'ar_coefficients': self.ar_coeffs,
            'characteristic_roots': roots,
            'min_distance_from_unit_circle': min(abs(abs(root) - 1.0) for root in roots),
            'sum_of_absolute_coefficients': np.sum(np.abs(self.ar_coeffs)),
            'is_stationary': self._is_stationary(self.ar_coeffs),
            'stationary_variance': self._get_stationary_variance()
        }


def create_ar_data(config_path: str = "configs/config.yaml") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create AR data using configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train_series, val_series, test_series)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters with defaults
    dataset_config = config.get('dataset', {})
    p = dataset_config.get('p_values', [1, 4])[1]  # Default to AR(4)
    sigma = dataset_config.get('sigma_values', [0.1, 1.0])[0]  # Default to 0.1
    sequence_length = dataset_config.get('sequence_length', 1000)
    random_seed = config.get('experiments', {}).get('random_seed', 42)
    
    generator = ARDataGenerator(p, sigma, sequence_length, random_seed)
    return generator.prepare_data()


def _test_generator(generator: ARDataGenerator, name: str) -> None:
    """Helper function to test and display generator results."""
    series = generator.generate_series()
    info = generator.get_stationarity_info()
    
    print(f"{name} generator:")
    print(f"  Coefficients: {info['ar_coefficients']}")
    print(f"  Characteristic roots: {info['characteristic_roots']}")
    print(f"  Is stationary: {info['is_stationary']}")
    print(f"  Series mean: {np.mean(series):.4f}")
    print(f"  Series variance: {np.var(series):.4f}")
    print()


if __name__ == "__main__":
    print("Testing AR Data Generator")
    print("=" * 50)
    
    # Test AR(1) and AR(4) generators
    generators = [
        (ARDataGenerator(1, 0.1, 1000, 42), "AR(1)"),
        (ARDataGenerator(4, 0.1, 1000, 42), "AR(4)")
    ]
    
    for generator, name in generators:
        _test_generator(generator, name)
    
    # Test data splits
    print("Testing data splits:")
    train, val, test = generators[1][0].prepare_data()
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print()
    
    # Test multiple series
    print("Testing multiple series generation:")
    multiple = generators[1][0].generate_multiple_series(5)
    print(f"  Shape: {multiple.shape}")
    print(f"  Means: {np.mean(multiple, axis=1)}")
    print(f"  Variances: {np.var(multiple, axis=1)}") 