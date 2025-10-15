"""
Linear AR(p) Baseline Model - OLS Implementation.

This module provides a classical autoregressive model implementation using
ordinary least squares (OLS) estimation. It serves as the gold standard
baseline for comparison with neural network approaches like LSA transformers.

The model implements the standard AR(p) formulation:
    X_t = φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t

Classes:
    LinearARModel: Main AR(p) model with fit/predict methods.

Functions:
    fit_linear_ar_baseline: Convenience function for fitting AR models.
    evaluate_teacher_forcing: Teacher-forcing evaluation on test data.
    evaluate_chain_of_thought: Autoregressive evaluation on test data.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LinearARModel:
    """Linear AR(p) model using OLS estimation"""
    
    def __init__(self, p: int):
        """
        Args:
            p: AR order
        """
        self.p = p
        self.coefficients = None
        self.intercept = None
        self.fitted = False
        
    def fit(self, series: np.ndarray) -> 'LinearARModel':
        """
        Fit AR(p) model using OLS closed-form solution
        
        Args:
            series: Time series data
            
        Returns:
            self
        """
        if len(series) <= self.p:
            raise ValueError(f"Series length {len(series)} must be > AR order {self.p}")
        
        # Create lagged features
        X, y = self._create_lagged_features(series)
        
        # OLS solution using least squares (more robust to singular matrices)
        self.coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        self.intercept = 0.0  # No intercept in standard AR(p) formulation
        
        self.fitted = True
        
        return self
    
    def _create_lagged_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged features for AR(p) model
        
        y := (x_{p+1}, x_{p+2}, ..., x_n)^T
        X := ( x_p    x_{p-1}  ...  x_1    )
             ( x_{p+1}  x_p      ...  x_2    )
             (  :       :        ...  :      )
             ( x_{n-1}  x_{n-2}  ...  x_{n-p} )
        """
        n = len(series)
        
        # Create lagged matrix X
        X = np.zeros((n - self.p, self.p))
        for i in range(self.p):
            X[:, i] = series[self.p-1-i:n-1-i]
        
        # Target values y
        y = series[self.p:]
        
        return X, y
    
    def predict(self, series: np.ndarray) -> float:
        """
        Predict next value using fitted AR(p) model
        
        Args:
            series: Input series (last p values used for prediction)
            
        Returns:
            Single prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if len(series) < self.p:
            raise ValueError(f"Need at least {self.p} values for prediction")
        
        # Use last p values
        # Arrange lags so that the first element corresponds to lag 1 (x_{t-1})
        last_values = series[-self.p:][::-1]
        
        # Make one-step prediction (no intercept in standard AR formulation)
        prediction = np.sum(self.coefficients * last_values)
        
        return prediction
    

    
    def predict_chain_of_thought(self, series: np.ndarray, steps: int) -> np.ndarray:
        """
        Chain-of-Thought prediction (autoregressive, uses own predictions)
        
        Args:
            series: Initial context (last p values)
            steps: Number of steps to predict
            
        Returns:
            Predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if len(series) < self.p:
            raise ValueError(f"Need at least {self.p} values for context")
        
        # Start with last p values
        context = series[-self.p:].copy()
        predictions = []
        
        for _ in range(steps):
            # Make prediction (no intercept in standard AR formulation)
            pred = np.sum(self.coefficients * context[::-1])
            predictions.append(pred)
            
            # Update context with prediction
            context = np.append(context[1:], pred)
        
        return np.array(predictions)
    
    def evaluate(self, true_series: np.ndarray, predicted_series: np.ndarray) -> dict:
        """
        Evaluate predictions using multiple metrics
        
        Args:
            true_series: True values
            predicted_series: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        if len(true_series) != len(predicted_series):
            raise ValueError("True and predicted series must have same length")
        
        mse = mean_squared_error(true_series, predicted_series)
        mae = mean_absolute_error(true_series, predicted_series)
        
        # MAPE
        mape = np.mean(np.abs((true_series - predicted_series) / (true_series + 1e-8))) * 100
        
        # SMAPE
        smape = 2.0 * np.mean(np.abs(predicted_series - true_series) / 
                             (np.abs(true_series) + np.abs(predicted_series) + 1e-8)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape
        }





def fit_linear_ar_baseline(train_series: np.ndarray, p: int) -> LinearARModel:
    """
    Fit linear AR baseline on training data
    
    Args:
        train_series: Training time series
        p: AR order
        
    Returns:
        Fitted linear AR model
    """
    model = LinearARModel(p)
    model.fit(train_series)
    return model


def evaluate_teacher_forcing(model: LinearARModel, full_series: np.ndarray, 
                           test_start: int) -> dict:
    """
    Evaluate model using teacher forcing on test portion
    
    Args:
        model: Fitted linear AR model
        full_series: Complete time series
        test_start: Index where test portion begins
        
    Returns:
        Dictionary of evaluation results
    """
    test_series = full_series[test_start:]
    predictions = []
    
    # Teacher forcing: use true values for each prediction
    for i in range(len(test_series)):
        # Use the p values before the current test point
        context = full_series[test_start + i - model.p:test_start + i]
        pred = model.predict(context)
        predictions.append(pred)
    
    metrics = model.evaluate(test_series, predictions)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'true_values': test_series
    }


def evaluate_chain_of_thought(model: LinearARModel, full_series: np.ndarray, 
                            test_start: int) -> dict:
    """
    Evaluate model using chain-of-thought (autoregressive) on test portion
    
    Args:
        model: Fitted linear AR model
        full_series: Complete time series
        test_start: Index where test portion begins
        
    Returns:
        Dictionary of evaluation results
    """
    test_series = full_series[test_start:]
    
    # Use initial context from training data
    initial_context = full_series[test_start - model.p:test_start]
    
    # Chain-of-thought: use predictions as input for subsequent predictions
    predictions = model.predict_chain_of_thought(initial_context, len(test_series))
    
    metrics = model.evaluate(test_series, predictions)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'true_values': test_series
    }







def _unit_test_prediction_order():
    """Unit test to verify that lag ordering in prediction is correct."""
    model = LinearARModel(2)
    model.coefficients = np.array([0.3, 0.7])  # φ1, φ2
    model.fitted = True

    # Context where x_{t-1} = 2.0, x_{t-2} = 1.0
    context = np.array([1.0, 2.0])

    # Expected prediction: 0.3 * 2.0 + 0.7 * 1.0 = 1.3
    expected = 0.3 * 2.0 + 0.7 * 1.0
    pred = model.predict(context)
    assert np.isclose(pred, expected), f"Prediction order incorrect: got {pred}, expected {expected}"

    # Chain-of-thought: one-step ahead should match same expectation
    cot_pred = model.predict_chain_of_thought(context, 1)[0]
    assert np.isclose(cot_pred, expected), f"CoT prediction order incorrect: got {cot_pred}, expected {expected}"

    print("Prediction order unit test passed.\n")


if __name__ == "__main__":
    _unit_test_prediction_order()

    # Unit test for AR(5) model
    np.random.seed(42)
    
    # Parameters
    p = 5  # AR order
    n = 10000  # Total length
    train_size = 9500  # Training size
    test_size = 500   # Test size
    
    # Generate AR(5) data with known coefficients (ensuring stationarity)
    true_coeffs = [0.3, 0.2, 0.1, 0.05, 0.1]  # True AR coefficients
    series = np.random.normal(0, 0.1, n)
    
    # Add AR structure with better numerical stability
    for i in range(p, n):
        series[i] = sum(true_coeffs[j] * series[i-j-1] for j in range(p)) + np.random.normal(0, 0.05)
    
    # Split data
    train_series = series[:train_size]
    test_start = train_size
    
    # Fit model
    model = LinearARModel(p)
    model.fit(train_series)
    
    print(f"AR({p}) Model Test Results:")
    print(f"True coefficients: {true_coeffs}")
    print(f"Estimated coefficients: {model.coefficients}")
    print(f"Training size: {train_size}, Test size: {test_size}")
    
    # Evaluate using both methods
    tf_results = evaluate_teacher_forcing(model, series, test_start)
    cot_results = evaluate_chain_of_thought(model, series, test_start)
    
    print(f"\nTeacher Forcing Evaluation:")
    for metric, value in tf_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nChain-of-Thought Evaluation:")
    for metric, value in cot_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nFirst 5 predictions comparison:")
    for i in range(5):
        tf_pred = tf_results['predictions'][i]
        cot_pred = cot_results['predictions'][i]
        true_val = tf_results['true_values'][i]
        print(f"  TF: {tf_pred:.3f}, CoT: {cot_pred:.3f}, True: {true_val:.3f}") 