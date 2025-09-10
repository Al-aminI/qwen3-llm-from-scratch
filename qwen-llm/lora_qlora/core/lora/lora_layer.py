"""
LoRA layer implementation.

This module provides the core LoRA layer implementation for low-rank adaptation.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    🎯 LORA LAYER IMPLEMENTATION
    
    LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices.
    
    Mathematical Foundation:
    W = W₀ + ΔW = W₀ + BA
    Where:
    - W₀: Original frozen weights
    - B: Low-rank matrix (d × r)
    - A: Low-rank matrix (r × k)
    - r << min(d, k) (rank)
    
    Benefits:
    - Reduces trainable parameters by ~1000x
    - Maintains model performance
    - Enables efficient fine-tuning
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, 
                 alpha: float = 32.0, dropout: float = 0.1):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of LoRA matrices (r << min(in_features, out_features))
            alpha: LoRA scaling parameter
            dropout: Dropout rate for LoRA
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor (batch_size, seq_len, out_features)
        """
        # Apply LoRA: x @ A^T @ B^T
        # x: (batch, seq, in_features)
        # A: (rank, in_features) -> (in_features, rank)
        # B: (out_features, rank)
        # Result: (batch, seq, out_features)
        
        x = self.dropout(x)
        x = x @ self.lora_A.T  # (batch, seq, rank)
        x = x @ self.lora_B.T  # (batch, seq, out_features)
        x = x * self.scaling
        
        return x
    
    def get_parameter_count(self) -> int:
        """Get the number of trainable parameters in this LoRA layer."""
        return self.lora_A.numel() + self.lora_B.numel()
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB for this LoRA layer."""
        total_params = self.get_parameter_count()
        # Assuming float32 (4 bytes per parameter)
        memory_mb = (total_params * 4) / (1024 * 1024)
        return memory_mb
    
    def merge_weights(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights with original weights.
        
        Args:
            original_weight: Original weight matrix (out_features, in_features)
            
        Returns:
            Merged weight matrix
        """
        # W = W₀ + BA
        lora_weight = self.lora_B @ self.lora_A  # (out_features, in_features)
        return original_weight + lora_weight * self.scaling
    
    def reset_parameters(self):
        """Reset LoRA parameters to initial values."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def get_rank(self) -> int:
        """Get the rank of this LoRA layer."""
        return self.rank
    
    def get_scaling(self) -> float:
        """Get the scaling factor for this LoRA layer."""
        return self.scaling
