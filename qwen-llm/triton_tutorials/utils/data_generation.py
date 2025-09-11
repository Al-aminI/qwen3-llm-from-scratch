"""
📊 Data Generation Utilities

This module provides utilities for generating test data for Triton kernels
and benchmarking.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random

@dataclass
class DataConfig:
    """Configuration for data generation."""
    size: int
    dtype: torch.dtype = torch.float32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: Optional[int] = None
    distribution: str = 'normal'  # 'normal', 'uniform', 'zeros', 'ones'
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0

class DataGenerator:
    """
    📊 DATA GENERATOR
    
    A utility class for generating test data for Triton kernels.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_vector(self, config: DataConfig) -> torch.Tensor:
        """
        Generate a vector with the specified configuration.
        
        Args:
            config: Data configuration
            
        Returns:
            Generated vector tensor
        """
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        if config.distribution == 'normal':
            data = torch.randn(config.size, dtype=config.dtype, device=config.device)
            data = data * config.std + config.mean
        elif config.distribution == 'uniform':
            data = torch.rand(config.size, dtype=config.dtype, device=config.device)
            data = data * (config.max_val - config.min_val) + config.min_val
        elif config.distribution == 'zeros':
            data = torch.zeros(config.size, dtype=config.dtype, device=config.device)
        elif config.distribution == 'ones':
            data = torch.ones(config.size, dtype=config.dtype, device=config.device)
        else:
            raise ValueError(f"Unknown distribution: {config.distribution}")
        
        return data
    
    def generate_matrix(self, rows: int, cols: int, config: DataConfig) -> torch.Tensor:
        """
        Generate a matrix with the specified configuration.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            config: Data configuration
            
        Returns:
            Generated matrix tensor
        """
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        if config.distribution == 'normal':
            data = torch.randn(rows, cols, dtype=config.dtype, device=config.device)
            data = data * config.std + config.mean
        elif config.distribution == 'uniform':
            data = torch.rand(rows, cols, dtype=config.dtype, device=config.device)
            data = data * (config.max_val - config.min_val) + config.min_val
        elif config.distribution == 'zeros':
            data = torch.zeros(rows, cols, dtype=config.dtype, device=config.device)
        elif config.distribution == 'ones':
            data = torch.ones(rows, cols, dtype=config.dtype, device=config.device)
        else:
            raise ValueError(f"Unknown distribution: {config.distribution}")
        
        return data
    
    def generate_batch_matrices(self, batch_size: int, rows: int, cols: int, config: DataConfig) -> torch.Tensor:
        """
        Generate a batch of matrices with the specified configuration.
        
        Args:
            batch_size: Number of matrices in the batch
            rows: Number of rows per matrix
            cols: Number of columns per matrix
            config: Data configuration
            
        Returns:
            Generated batch tensor
        """
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        if config.distribution == 'normal':
            data = torch.randn(batch_size, rows, cols, dtype=config.dtype, device=config.device)
            data = data * config.std + config.mean
        elif config.distribution == 'uniform':
            data = torch.rand(batch_size, rows, cols, dtype=config.dtype, device=config.device)
            data = data * (config.max_val - config.min_val) + config.min_val
        elif config.distribution == 'zeros':
            data = torch.zeros(batch_size, rows, cols, dtype=config.dtype, device=config.device)
        elif config.distribution == 'ones':
            data = torch.ones(batch_size, rows, cols, dtype=config.dtype, device=config.device)
        else:
            raise ValueError(f"Unknown distribution: {config.distribution}")
        
        return data
    
    def generate_attention_data(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, 
                               config: DataConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate attention data (Q, K, V tensors).
        
        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            seq_len: Sequence length
            head_dim: Head dimension
            config: Data configuration
            
        Returns:
            Tuple of (Q, K, V) tensors
        """
        q = self.generate_batch_matrices(batch_size, num_heads, seq_len, head_dim, config)
        k = self.generate_batch_matrices(batch_size, num_heads, seq_len, head_dim, config)
        v = self.generate_batch_matrices(batch_size, num_heads, seq_len, head_dim, config)
        
        return q, k, v
    
    def generate_transformer_data(self, batch_size: int, seq_len: int, hidden_dim: int, 
                                 config: DataConfig) -> torch.Tensor:
        """
        Generate transformer input data.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            config: Data configuration
            
        Returns:
            Generated transformer input tensor
        """
        return self.generate_batch_matrices(batch_size, seq_len, hidden_dim, config)
    
    def generate_sparse_data(self, size: int, sparsity: float, config: DataConfig) -> torch.Tensor:
        """
        Generate sparse data.
        
        Args:
            size: Size of the tensor
            sparsity: Fraction of elements that should be zero
            config: Data configuration
            
        Returns:
            Generated sparse tensor
        """
        data = self.generate_vector(config)
        
        # Randomly set some elements to zero
        num_zeros = int(size * sparsity)
        zero_indices = torch.randperm(size)[:num_zeros]
        data[zero_indices] = 0.0
        
        return data
    
    def generate_sequence_data(self, batch_size: int, seq_len: int, vocab_size: int, 
                              config: DataConfig) -> torch.Tensor:
        """
        Generate sequence data (token IDs).
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            vocab_size: Vocabulary size
            config: Data configuration
            
        Returns:
            Generated sequence tensor
        """
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        # Generate random token IDs
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=config.device)
        
        return data
    
    def generate_benchmark_data(self, sizes: List[int], dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
        """
        Generate data for benchmarking.
        
        Args:
            sizes: List of sizes to generate
            dtype: Data type
            
        Returns:
            List of generated tensors
        """
        data_list = []
        
        for size in sizes:
            config = DataConfig(
                size=size,
                dtype=dtype,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                distribution='normal'
            )
            data = self.generate_vector(config)
            data_list.append(data)
        
        return data_list
    
    def generate_matrix_benchmark_data(self, sizes: List[Tuple[int, int, int]], 
                                      dtype: torch.dtype = torch.float32) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate data for matrix multiplication benchmarking.
        
        Args:
            sizes: List of (M, K, N) tuples
            dtype: Data type
            
        Returns:
            List of (A, B) matrix pairs
        """
        data_list = []
        
        for M, K, N in sizes:
            config = DataConfig(
                size=1,  # Not used for matrices
                dtype=dtype,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                distribution='normal'
            )
            
            a = self.generate_matrix(M, K, config)
            b = self.generate_matrix(K, N, config)
            data_list.append((a, b))
        
        return data_list

def generate_test_data():
    """Generate test data for various scenarios."""
    print("📊 Generating Test Data:")
    print("=" * 50)
    
    generator = DataGenerator(seed=42)
    
    # Generate vector data
    print("\n📈 Vector Data:")
    vector_config = DataConfig(size=1024, dtype=torch.float32, distribution='normal')
    vector_data = generator.generate_vector(vector_config)
    print(f"  Generated vector: {vector_data.shape}, dtype: {vector_data.dtype}")
    
    # Generate matrix data
    print("\n📈 Matrix Data:")
    matrix_config = DataConfig(size=1, dtype=torch.float32, distribution='normal')
    matrix_data = generator.generate_matrix(256, 128, matrix_config)
    print(f"  Generated matrix: {matrix_data.shape}, dtype: {matrix_data.dtype}")
    
    # Generate attention data
    print("\n📈 Attention Data:")
    attention_config = DataConfig(size=1, dtype=torch.float32, distribution='normal')
    q, k, v = generator.generate_attention_data(2, 8, 128, 64, attention_config)
    print(f"  Generated Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    
    # Generate sparse data
    print("\n📈 Sparse Data:")
    sparse_config = DataConfig(size=1024, dtype=torch.float32, distribution='normal')
    sparse_data = generator.generate_sparse_data(1024, 0.5, sparse_config)
    sparsity = (sparse_data == 0).float().mean().item()
    print(f"  Generated sparse vector: {sparse_data.shape}, sparsity: {sparsity:.2%}")
    
    # Generate benchmark data
    print("\n📈 Benchmark Data:")
    benchmark_sizes = [1024, 4096, 16384]
    benchmark_data = generator.generate_benchmark_data(benchmark_sizes)
    for i, data in enumerate(benchmark_data):
        print(f"  Size {benchmark_sizes[i]}: {data.shape}")
    
    return generator

def generate_validation_data():
    """Generate data for validation tests."""
    print("\n📊 Generating Validation Data:")
    print("=" * 50)
    
    generator = DataGenerator(seed=123)
    
    # Generate edge case data
    edge_cases = []
    
    # Single element
    single_config = DataConfig(size=1, dtype=torch.float32, distribution='normal')
    edge_cases.append(generator.generate_vector(single_config))
    
    # Zero tensor
    zero_config = DataConfig(size=100, dtype=torch.float32, distribution='zeros')
    edge_cases.append(generator.generate_vector(zero_config))
    
    # Large values
    large_config = DataConfig(size=100, dtype=torch.float32, distribution='normal', mean=1e6, std=1e3)
    edge_cases.append(generator.generate_vector(large_config))
    
    # Small values
    small_config = DataConfig(size=100, dtype=torch.float32, distribution='normal', mean=1e-6, std=1e-9)
    edge_cases.append(generator.generate_vector(small_config))
    
    print(f"  Generated {len(edge_cases)} edge case datasets")
    
    return edge_cases

if __name__ == "__main__":
    # Generate test data
    generator = generate_test_data()
    
    # Generate validation data
    edge_cases = generate_validation_data()
    
    print("\n✅ Data generation complete!")
