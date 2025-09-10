"""
LoRA-adapted linear layer.

This module provides a linear layer that combines original weights with LoRA adaptation.
"""

import torch
import torch.nn as nn
from typing import Optional

from .lora_layer import LoRALayer


class LoRALinear(nn.Module):
    """
    🎯 LORA LINEAR LAYER
    
    Combines original linear layer with LoRA adaptation.
    
    This layer maintains the original linear layer's functionality while adding
    LoRA adaptation for efficient fine-tuning.
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        """
        Initialize LoRA linear layer.
        
        Args:
            original_layer: Original linear layer to adapt
            rank: Rank of LoRA matrices
            alpha: LoRA scaling parameter
            dropout: Dropout rate for LoRA
        """
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original + LoRA adaptation
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor (batch_size, seq_len, out_features)
        """
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        return original_output + lora_output
    
    def get_original_layer(self) -> nn.Linear:
        """Get the original linear layer."""
        return self.original_layer
    
    def get_lora_layer(self) -> LoRALayer:
        """Get the LoRA layer."""
        return self.lora
    
    def get_parameter_count(self) -> dict:
        """Get parameter counts for analysis."""
        original_params = sum(p.numel() for p in self.original_layer.parameters())
        lora_params = self.lora.get_parameter_count()
        
        return {
            'original': original_params,
            'lora': lora_params,
            'total': original_params + lora_params,
            'lora_percentage': (lora_params / (original_params + lora_params)) * 100
        }
    
    def get_memory_usage(self) -> dict:
        """Get memory usage breakdown."""
        original_memory = sum(p.numel() for p in self.original_layer.parameters()) * 4 / (1024 * 1024)
        lora_memory = self.lora.get_memory_usage()
        
        return {
            'original_mb': original_memory,
            'lora_mb': lora_memory,
            'total_mb': original_memory + lora_memory,
            'lora_percentage': (lora_memory / (original_memory + lora_memory)) * 100
        }
    
    def merge_lora_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into original weights.
        
        Returns:
            Merged weight tensor
        """
        return self.lora.merge_weights(self.original_layer.weight)
    
    def reset_lora_parameters(self):
        """Reset LoRA parameters to initial values."""
        self.lora.reset_parameters()
    
    def set_lora_alpha(self, alpha: float):
        """Set the LoRA alpha parameter."""
        self.lora.alpha = alpha
        self.lora.scaling = alpha / self.lora.rank
    
    def get_lora_alpha(self) -> float:
        """Get the LoRA alpha parameter."""
        return self.lora.alpha
    
    def get_lora_rank(self) -> int:
        """Get the LoRA rank."""
        return self.lora.rank
