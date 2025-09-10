"""
QLoRA linear layer implementation.

This module provides the QLoRA linear layer that combines 4-bit quantization with LoRA.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .qlora_layer import QLoRALayer


class QLoRALinear(nn.Module):
    """
    🎯 QLORA LINEAR LAYER
    
    QLoRA linear layer combining 4-bit quantization with LoRA adaptation.
    
    This layer provides:
    - 4-bit quantized weights for memory efficiency
    - LoRA adaptation for fine-tuning
    - Efficient forward pass with quantized operations
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 1.0, dropout: float = 0.0):
        """
        Initialize QLoRA linear layer.
        
        Args:
            original_layer: Original linear layer to adapt
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # QLoRA layer
        self.qlora = QLoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Store original weights for reference
        self.register_buffer('original_weight', original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.register_buffer('original_bias', original_layer.bias.data.clone())
        else:
            self.register_buffer('original_bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with QLoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get QLoRA output
        qlora_output = self.qlora(x)
        
        # Add original bias if present
        if self.original_bias is not None:
            qlora_output = qlora_output + self.original_bias
        
        return qlora_output
    
    def get_qlora_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get QLoRA weights for analysis.
        
        Returns:
            Dictionary containing QLoRA weights
        """
        return self.qlora.get_quantized_weights()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage breakdown.
        
        Returns:
            Dictionary with memory usage information
        """
        return self.qlora.get_memory_usage()
    
    def reset_qlora_parameters(self):
        """Reset QLoRA parameters to initial values."""
        self.qlora.reset_parameters()
    
    def set_lora_alpha(self, alpha: float):
        """
        Set LoRA alpha scaling factor.
        
        Args:
            alpha: New alpha value
        """
        self.alpha = alpha
        self.qlora.set_lora_alpha(alpha)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'quantization_bits': 4,
            'has_bias': self.original_bias is not None
        }
