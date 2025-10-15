"""
Linear Self-Attention (LSA) Transformer Implementation

This module implements:
1. Linear Self-Attention (LSA) layers
2. LSA Transformer with stacked LSA layers  
3. LSA Transformer with Hankel matrix input embedding

The Hankel functionality allows:
- Creating Hankel matrices from time series sequences
- Each column represents a subsequence of length (p+1)
- Final column is zero-padded with last coordinate being 0
- Using Hankel matrices as input to LSA Transformer with (d+1) × (n+1) dimensions
- Teacher forcing: sequences → Hankel matrices → transformer → predictions
- Chain-of-Thought: autoregressive updates with new columns

Example Hankel Matrix Construction:
For sequence [1, 2, 3, 4, 5] with window_size=3 (p=2):
- Column 0: [1, 2, 3] (positions 0,1,2)
- Column 1: [2, 3, 4] (positions 1,2,3) 
- Column 2: [3, 4, 5] (positions 2,3,4)
- Column 3: [4, 5, 0] (positions 3,4,0 - zero-padded)

Architecture:
- LSATransformer: Core LSA implementation for direct tensor inputs
- LSATransformerWithHankel: Specialized for time series with Hankel matrices

Usage:
    # Standard LSA Transformer
    model = LSATransformer(d=64, L=2)
    H = torch.randn(batch_size, d+1, seq_len)
    output = model(H)
    predictions = model.predict(H)
    
    # LSA Transformer with Hankel matrix
    hankel_model = LSATransformerWithHankel(p=7, L=1)
    sequences = torch.randn(num_sequences, seq_len)
    output = hankel_model(sequences)
    
    # Teacher forcing and Chain-of-Thought
    tf_predictions = hankel_model.predict_teacher_forcing(sequences)
    cot_predictions = hankel_model.predict_chain_of_thought(sequences, steps=10)
    
    # Single sequence inference (automatically handled)
    single_seq = torch.randn(seq_len)
    single_pred = hankel_model.predict_teacher_forcing(single_seq)
    
    # Factory function
    model = create_lsa_model('LSA-Hankel', p=7, L=1)
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable, Any
from functools import wraps


def handle_single_sequence(func: Callable) -> Callable:
    """Decorator to automatically handle single sequence inputs"""
    @wraps(func)
    def wrapper(self, sequences: torch.Tensor, *args, **kwargs) -> Any:
        # Add batch dimension if single sequence
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
            single_sequence = True
        else:
            single_sequence = False
        
        # Call the original method
        result = func(self, sequences, *args, **kwargs)
        
        # Remove batch dimension if it was added
        if single_sequence and hasattr(result, 'squeeze'):
            result = result.squeeze(0)
        
        return result
    return wrapper


class LinearSelfAttention(nn.Module):
    """Linear Self-Attention layer implementing
    LSA(H) := H + (1/n) @ P @ H @ M @ (H^T @ Q @ H)

    Optional softmax attention:
    If `use_softmax=True`, applies column-wise softmax to `(H^T @ Q @ H)`
    before masking, i.e., uses `PHM Softmax(H^TQH)`.
    """
    
    def __init__(self, d: int, use_softmax: bool = False):
        """
        Args:
            d: Embedding dimension
            use_softmax: If True, use softmax attention; otherwise linear form
        """
        super().__init__()
        self.d = d
        self.use_softmax = use_softmax
        self.n = None
        self.mask = None
        
        # P := W_V, Q := W_K^T * W_Q, with P,Q ∈ ℝ^(d+1)×(d+1)
        self.P = nn.Parameter(torch.randn(d+1, d+1) * 0.01)
        self.Q = nn.Parameter(torch.randn(d+1, d+1) * 0.01)
        
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Input sequence of shape (batch_size, d+1, n+1)
            
        Returns:
            Output of shape (batch_size, d+1, n+1)
        """
        batch_size, d_actual, seq_len = H.shape
        
        if d_actual != self.d + 1:
            raise ValueError(f"Expected H to have {self.d + 1} rows (d+1), got {d_actual}")
        
        # Set n based on sequence length: n = seq_len - 1
        if self.n is None or self.n != seq_len - 1:
            self.n = seq_len - 1
            # Create causal mask M := [I_n  0]
            #                          [0    0]
            self.mask = torch.zeros(seq_len, seq_len)
            self.mask[:self.n, :self.n] = torch.eye(self.n)
        
        # Transpose H for matrix operations: (batch_size, n+1, d+1)
        H_t = H.transpose(1, 2)
        
        # Compute H^T @ Q @ H
        # Use torch.matmul for better deterministic behavior
        HtQH = torch.matmul(torch.matmul(H_t, self.Q), H)

        # Optional column-wise softmax normalization before masking, note that we have batch dimension
        # We use dim=1 to softmax over the "columns" of H^T @ Q @ H
        if self.use_softmax:
            HtQH = torch.softmax(HtQH, dim=1)
        
        # Apply causal mask M
        MHtQH = torch.matmul(self.mask.to(HtQH.device), HtQH)
        
        # Compute P @ H
        PH = torch.matmul(self.P, H)

        # Compute P @ H @ M @ (H^T @ Q @ H)
        attention_output = torch.matmul(PH, MHtQH)
        
        # Scale by 1/n and add residual connection
        output = H + attention_output / self.n
        
        return output


class LSATransformer(nn.Module):
    """L-layer Transformer composed of stacked LSA layers: TF(H) := LSA_L ∘ LSA_{L-1} ∘ ... ∘ LSA_1(H)"""
    
    def __init__(self, d: int, L: int = 1, use_softmax: bool = False):
        """
        Args:
            d: Embedding dimension
            L: Number of LSA layers
            use_softmax: If True, each LSA layer uses softmax attention
        """
        super().__init__()
        self.d = d
        self.L = L
        self.lsa_layers = nn.ModuleList([LinearSelfAttention(d, use_softmax=use_softmax) for _ in range(L)])
        
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Input sequence of shape (batch_size, d+1, n+1)
            
        Returns:
            Output of shape (batch_size, d+1, n+1)
        """
        for lsa_layer in self.lsa_layers:
            H = lsa_layer(H)
        return H
    
    def predict(self, H: torch.Tensor) -> torch.Tensor:
        """
        Extract prediction from final token coordinate: TF(H)_{d+1,(n+1)}
        
        Args:
            H: Transformer output of shape (batch_size, d+1, n+1)
            
        Returns:
            Predictions of shape (batch_size,)
        """
        return H[:, -1, -1]


class LSATransformerWithHankel(nn.Module):
    """LSA Transformer with Hankel matrix constructed from input sequences"""
    
    def __init__(self, p: int, L: int = 1, use_softmax: bool = False):
        """
        Args:
            p: Context length for Hankel matrix. Sets embedding dimension d=p.
            L: Number of LSA layers
            use_softmax: If True, each LSA layer uses softmax attention
        """
        super().__init__()
        self.p = p
        self.L = L
        self.transformer = LSATransformer(d=p, L=L, use_softmax=use_softmax)
    
    def create_hankel_matrix(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Create Hankel matrix from time series sequences
        
        Args:
            sequences: Time series sequences of shape (num_sequences, seq_len)
            
        Returns:
            Hankel matrix of shape (num_sequences, p+1, hankel_seq_len)
        """
        num_sequences, seq_len = sequences.shape
        hankel_seq_len = seq_len - self.p + 1
        
        if hankel_seq_len <= 0:
            raise ValueError(f"Sequence length ({seq_len}) must be greater than context length p ({self.p})")

        return create_hankel_matrix_from_sequences(sequences, window_size=self.p + 1)
    
    def _update_hankel_matrices(self, H_matrices: list, predictions: torch.Tensor) -> list:
        """
        Update Hankel matrices with new predictions for autoregressive generation
        
        Args:
            H_matrices: List of Hankel matrices for each sequence
            predictions: New predictions of shape (num_sequences,)
            
        Returns:
            Updated list of Hankel matrices
        """
        for seq_idx, pred in enumerate(predictions):
            # Update last unknown value with prediction
            H_matrices[seq_idx][self.p, -1] = pred

            # Append new column: shifted version of last column with zero at end
            new_col = torch.zeros(self.p + 1, dtype=H_matrices[seq_idx].dtype, device=H_matrices[seq_idx].device)
            new_col[:-1] = H_matrices[seq_idx][1:, -1]
            new_col[-1] = 0.0

            H_matrices[seq_idx] = torch.cat([H_matrices[seq_idx], new_col.unsqueeze(1)], dim=1)
        
        return H_matrices
        
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: Time series sequences of shape (num_sequences, seq_len)
            
        Returns:
            Output of shape (num_sequences, p+1, hankel_seq_len)
        """
        H = self.create_hankel_matrix(sequences)
        return self.transformer(H)
    
    @handle_single_sequence
    def predict(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: Time series sequences of shape (num_sequences, seq_len) or (seq_len,)
            
        Returns:
            Predictions of shape (num_sequences,) or scalar
        """
        output = self.forward(sequences)
        return self.transformer.predict(output)
    
    @handle_single_sequence
    def predict_teacher_forcing(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Teacher forcing prediction using Hankel matrices
        
        Args:
            sequences: Time series sequences of shape (num_sequences, seq_len) or (seq_len,)
            
        Returns:
            Predictions of shape (num_sequences,) or scalar
        """
        H = self.create_hankel_matrix(sequences)
        
        with torch.no_grad():
            output = self.transformer(H)
            predictions = self.transformer.predict(output)
        
        return predictions
    
    @handle_single_sequence
    def predict_chain_of_thought(self, initial_sequences: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Chain-of-Thought prediction with autoregressive updates
        
        Args:
            initial_sequences: Initial sequences of shape (num_sequences, seq_len) or (seq_len,)
            steps: Number of steps to predict
            
        Returns:
            Predictions of shape (num_sequences, steps) or (steps,)
        """
        num_sequences, seq_len = initial_sequences.shape
        
        # Create initial Hankel matrices
        H_raw = create_hankel_matrix_from_sequences(initial_sequences, window_size=self.p + 1)
        predictions = torch.zeros(num_sequences, steps, dtype=initial_sequences.dtype, device=initial_sequences.device)
        
        # Store Hankel matrices separately to avoid dimension issues
        H_matrices = [H_raw[i].clone() for i in range(num_sequences)]
        
        for step in range(steps):
            # Stack matrices for batch processing
            H_input = torch.stack(H_matrices, dim=0)

            with torch.no_grad():
                output = self.transformer(H_input)
                pred = self.transformer.predict(output)
                predictions[:, step] = pred
            
            # Update Hankel matrices autoregressively
            H_matrices = self._update_hankel_matrices(H_matrices, pred)
        
        return predictions


def create_hankel_matrix_from_sequences(sequences: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Create Hankel matrix from time series sequences
    
    Args:
        sequences: Time series sequences of shape (num_sequences, seq_len)
        window_size: Window size for Hankel matrix construction (p+1)
        
    Returns:
        Hankel matrix of shape (num_sequences, window_size, seq_len - window_size + 1)
    """
    num_sequences, seq_len = sequences.shape
    p = window_size - 1
    hankel_seq_len = seq_len - p + 1
    
    # Create position indices using broadcasting
    i_indices = torch.arange(p + 1, device=sequences.device).unsqueeze(1)
    j_indices = torch.arange(hankel_seq_len, device=sequences.device).unsqueeze(0)
    pos_indices = i_indices + j_indices
    
    # Create mask for valid positions
    valid_mask = pos_indices < seq_len
    
    # Initialize Hankel matrices
    hankel_matrices = torch.zeros(num_sequences, p + 1, hankel_seq_len, 
                                dtype=sequences.dtype, device=sequences.device)
    
    # Fill Hankel matrices for each sequence
    for seq_idx in range(num_sequences):
        sequence = sequences[seq_idx, :]
        valid_positions = pos_indices[valid_mask]
        hankel_matrices[seq_idx][valid_mask] = sequence[valid_positions]
    
    return hankel_matrices


def create_lsa_model(model_type: str, p: int, L: int = 1, use_softmax: bool = False) -> nn.Module:
    """
    Factory function to create LSA models
    
    Args:
        model_type: 'LSA-1', 'LSA-L', or 'LSA-Hankel'
        p: Context length for Hankel matrix, also used as embedding dimension d
        L: Number of layers (for LSA-L)
        use_softmax: If True, use softmax attention in layers
        
    Returns:
        LSA model
    """
    if model_type == 'LSA-1':
        return LSATransformer(p, L=1, use_softmax=use_softmax)
    elif model_type == 'LSA-L':
        return LSATransformer(p, L=L, use_softmax=use_softmax)
    elif model_type == 'LSA-Hankel':
        return LSATransformerWithHankel(p, L, use_softmax=use_softmax)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test LSA implementation
    torch.manual_seed(42)
    
    # Parameters
    d_for_non_hankel = 64
    p_for_hankel = 7
    L = 2
    
    print("=== Testing LSA Transformer Core Functionality ===")
    model = LSATransformer(d_for_non_hankel, L)
    
    for seq_len in [11, 15, 20]:
        H = torch.randn(4, d_for_non_hankel+1, seq_len)
        output = model(H)
        predictions = model.predict(output)
        
        print(f"seq_len={seq_len}: Input shape: {H.shape}")
        print(f"seq_len={seq_len}: Output shape: {output.shape}")
        print(f"seq_len={seq_len}: Predictions shape: {predictions.shape}")
        print()
    
    print("=== Testing LSA Transformer with Hankel ===")
    hankel_model = LSATransformerWithHankel(p=p_for_hankel, L=L)
    
    # Create random sequences for testing
    num_sequences = 10
    seq_len = 20
    sequences_tensor = torch.randn(num_sequences, seq_len)
    
    print(f"Random sequences shape: {sequences_tensor.shape}")
    
    # Test forward pass
    hankel_output = hankel_model(sequences_tensor)
    hankel_predictions = hankel_model.predict(sequences_tensor)
    
    print(f"Hankel output shape: {hankel_output.shape}")
    print(f"Hankel predictions shape: {hankel_predictions.shape}")
    
    # Test teacher forcing
    tf_predictions = hankel_model.predict_teacher_forcing(sequences_tensor)
    print(f"Teacher forcing predictions shape: {tf_predictions.shape}")
    
    # Test chain-of-thought
    cot_steps = 10
    cot_predictions = hankel_model.predict_chain_of_thought(sequences_tensor, steps=cot_steps)
    print(f"Chain-of-thought predictions shape: {cot_predictions.shape}")
    
    # Test single sequence inference (elegant handling)
    single_seq = sequences_tensor[0]  # Shape: (seq_len,)
    print(f"Single sequence: {single_seq}")
    
    single_forward = hankel_model.predict(single_seq)
    print(f"Single sequence forward pass shape: {single_forward.shape}")
    print(f"Single sequence forward pass: {single_forward}")
    
    single_pred = hankel_model.predict_teacher_forcing(single_seq)
    print(f"Single sequence prediction shape: {single_pred.shape}")
    print(f"Single sequence prediction: {single_pred}")
    
    # Test single sequence chain-of-thought
    single_cot = hankel_model.predict_chain_of_thought(single_seq, steps=cot_steps)
    print(f"Single sequence CoT predictions shape: {single_cot.shape}")
    print(f"Single sequence CoT predictions: {single_cot}")
    
    print("\n=== Testing Helper Functions ===")
    hankel_matrix = create_hankel_matrix_from_sequences(sequences_tensor, window_size=p_for_hankel+1)
    print(f"Hankel matrix shape: {hankel_matrix.shape}")
    expected_shape = (num_sequences, p_for_hankel+1, seq_len - p_for_hankel + 1)
    print(f"Expected shape: {expected_shape} - matches: {hankel_matrix.shape == expected_shape}")
    
    # Test Hankel matrix structure
    test_sequence = torch.tensor([[1, 2, 3, 4, 5]])
    test_hankel = create_hankel_matrix_from_sequences(test_sequence, window_size=3)
    
    print(f"Test sequence: {test_sequence[0]}")
    print(f"Test Hankel matrix (p=2, n=5):")
    print(test_hankel[0])
    
    expected_hankel = torch.tensor([[1, 2, 3, 4],
                                   [2, 3, 4, 5], 
                                   [3, 4, 5, 0]])
    print(f"Expected Hankel matrix:")
    print(expected_hankel)
    print(f"Structure matches: {torch.allclose(test_hankel[0], expected_hankel)}")
    
    print("\n=== Testing Factory Function ===")
    model_lsa1 = create_lsa_model('LSA-1', p=d_for_non_hankel)
    model_lsaL = create_lsa_model('LSA-L', p=d_for_non_hankel, L=2)
    model_hankel = create_lsa_model('LSA-Hankel', p=p_for_hankel, L=3)
    model_hankel_softmax = create_lsa_model('LSA-Hankel', p=p_for_hankel, L=1, use_softmax=True)
    
    print(f"LSA-1 model created: {type(model_lsa1)}")
    print(f"LSA-L model created: {type(model_lsaL)}")
    print(f"LSA-Hankel model created: {type(model_hankel)}")
    print(f"LSA-Hankel model with softmax attention created: {type(model_hankel_softmax)}")
    
    # Test Softmax Attention
    print("\n=== Testing Softmax Attention ===")
    hankel_output_softmax = model_hankel_softmax(sequences_tensor)
    print(f"Hankel output shape with softmax attention: {hankel_output_softmax.shape}")
    print(f"Hankel output with softmax attention: {hankel_output_softmax}")
    print(f"Hankel predictions with softmax attention: {model_hankel_softmax.predict(sequences_tensor)}")
    print(f"Hankel predictions with softmax attention teacher forcing: {model_hankel_softmax.predict_teacher_forcing(sequences_tensor)}")
    print(f"Hankel predictions with softmax attention chain-of-thought: {model_hankel_softmax.predict_chain_of_thought(sequences_tensor, steps=cot_steps)}")
    
    
    print("\n=== Performance Comparison ===")
    import time
    
    # Larger test data
    num_sequences = 100
    seq_len = 50
    p_perf_test = 10
    test_sequences = torch.randn(num_sequences, seq_len)
    
    # Test vectorized Hankel creation
    start_time = time.time()
    hankel_model = LSATransformerWithHankel(p=p_perf_test, L=L)
    hankel_output = hankel_model(test_sequences)
    vectorized_time = time.time() - start_time
    
    print(f"Vectorized Hankel creation time: {vectorized_time:.4f}s")
    print(f"Output shape: {hankel_output.shape}")
    
    # Test teacher forcing performance
    start_time = time.time()
    tf_predictions = hankel_model.predict_teacher_forcing(test_sequences)
    tf_time = time.time() - start_time
    
    print(f"Teacher forcing time: {tf_time:.4f}s")
    print(f"Predictions shape: {tf_predictions.shape}")
    
    print("\n✓ All tests passed!")
    
    print("\n=== Testing _update_hankel_matrices Method ===")
    # Test the _update_hankel_matrices method specifically
    test_model = LSATransformerWithHankel(p=3, L=1)
    
    # Create test Hankel matrices (2 sequences, p=3, 4 columns)
    H_matrices = [
        torch.tensor([[1, 2, 3, 4],
                     [2, 3, 4, 5],
                     [3, 4, 5, 6],
                     [4, 5, 6, 0]], dtype=torch.float),  # Last element is 0 (unknown)
        torch.tensor([[10, 20, 30, 40],
                     [20, 30, 40, 50],
                     [30, 40, 50, 60],
                     [40, 50, 60, 0]], dtype=torch.float)
    ]
    
    print(f"Original matrix shapes: {[H.shape for H in H_matrices]}")
    
    # Test predictions
    predictions = torch.tensor([7.0, 70.0])  # Predictions for the two sequences
    
    print("Original Hankel matrices:")
    for i, H in enumerate(H_matrices):
        print(f"Sequence {i}:")
        print(H)
    
    print(f"\nPredictions: {predictions}")

    
    # Update Hankel matrices
    H_matrices_copy = [H.clone() for H in H_matrices]  # Create a copy to preserve originals
    updated_H_matrices = test_model._update_hankel_matrices(H_matrices_copy, predictions)
    
    print("\nUpdated Hankel matrices:")
    for i, H in enumerate(updated_H_matrices):
        print(f"Sequence {i}:")
        print(H)
        print(f"Shape: {H.shape}")
    
    # Verify the update logic
    print("\nVerification:")
    for i, (original, updated) in enumerate(zip(H_matrices, updated_H_matrices)):
        print(f"Sequence {i}: Original shape: {original.shape}, Updated shape: {updated.shape}")
        # Check that the last unknown value was updated with prediction
        assert updated[3, -2] == predictions[i], f"Last unknown value not updated correctly for sequence {i}"
        # Check that a new column was added (shape[1] is the number of columns)
        assert updated.shape[1] == original.shape[1] + 1, f"New column not added for sequence {i}"
        # Check that the new column has the correct structure
        assert updated[3, -1] == 0.0, f"New unknown value not set to 0 for sequence {i}"
        print(f"Sequence {i}: ✓ All checks passed")
    
    print("\n✓ _update_hankel_matrices test passed!") 
