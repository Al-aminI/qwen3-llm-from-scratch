"""
QLoRA manager for model adaptation.

This module provides a manager class for applying QLoRA adaptation to entire models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

from .qlora_linear import QLoRALinear
from ..quantization.quantization_expert import QuantizationConfig


class QLoRAManager:
    """
    🎯 QLORA MANAGER
    
    Manages QLoRA adaptation for entire models.
    
    This class provides functionality to:
    - Apply QLoRA to specified modules in a model
    - Track and manage QLoRA layers
    - Analyze parameter counts and memory usage
    - Save and load QLoRA weights
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        """
        Initialize QLoRA manager.
        
        Args:
            model: The model to apply QLoRA to
            config: QLoRA configuration
        """
        self.model = model
        self.config = config
        self.qlora_layers = {}
        self.original_layers = {}
        self.applied_modules = []
    
    def apply_qlora(self, target_modules: Optional[List[str]] = None):
        """
        🎯 APPLY QLORA TO MODEL
        
        Replaces specified linear layers with QLoRA-adapted versions.
        
        Args:
            target_modules: List of module names to apply QLoRA to.
                          If None, uses config.target_modules.
        """
        if target_modules is None:
            target_modules = self.config.target_modules
        
        print(f"🎯 Applying QLoRA to model...")
        print(f"   Target modules: {target_modules}")
        print(f"   LoRA rank: {self.config.lora_rank}")
        print(f"   LoRA alpha: {self.config.lora_alpha}")
        print(f"   Quantization bits: 4")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                print(f"   Applying QLoRA to: {name}")
                
                # Store original layer
                self.original_layers[name] = module
                
                # Create QLoRA layer
                qlora_layer = QLoRALinear(
                    module,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
                
                # Replace in model
                self._replace_module(self.model, name, qlora_layer)
                self.qlora_layers[name] = qlora_layer
                self.applied_modules.append(name)
        
        print(f"✅ QLoRA applied to {len(self.qlora_layers)} modules")
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """
        Replace a module in the model by name.
        
        Args:
            model: The model containing the module
            module_name: Name of the module to replace
            new_module: New module to replace with
        """
        parts = module_name.split('.')
        current = model
        
        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the module
        setattr(current, parts[-1], new_module)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get all trainable parameters (QLoRA only).
        
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        for qlora_layer in self.qlora_layers.values():
            trainable_params.extend(qlora_layer.qlora.lora.parameters())
        return trainable_params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter counts for analysis.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage breakdown.
        
        Returns:
            Dictionary with memory usage information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        # Memory calculation (in MB)
        # Original weights: 4 bytes per parameter (fp32)
        # QLoRA weights: 0.5 bytes per parameter (4-bit) + LoRA parameters
        
        total_memory = total_params * 4 / (1024 * 1024)  # MB
        trainable_memory = trainable_params * 4 / (1024 * 1024)  # MB
        frozen_memory = total_memory - trainable_memory
        
        # Calculate quantized memory savings
        quantized_memory = 0
        for qlora_layer in self.qlora_layers.values():
            memory_info = qlora_layer.get_memory_usage()
            quantized_memory += memory_info['quantized_weights_bytes']
        
        quantized_memory_mb = quantized_memory / (1024 * 1024)
        
        return {
            'total_memory_mb': total_memory,
            'trainable_memory_mb': trainable_memory,
            'frozen_memory_mb': frozen_memory,
            'quantized_memory_mb': quantized_memory_mb,
            'memory_savings_mb': total_memory - quantized_memory_mb,
            'trainable_percentage': (trainable_memory / total_memory) * 100 if total_memory > 0 else 0
        }
    
    def get_qlora_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get QLoRA weights for saving.
        
        Returns:
            Dictionary containing QLoRA weights
        """
        qlora_weights = {}
        for name, qlora_layer in self.qlora_layers.items():
            qlora_weights[name] = {
                'lora_A': qlora_layer.qlora.lora.lora_A.data.clone(),
                'lora_B': qlora_layer.qlora.lora.lora_B.data.clone(),
                'quantized_weights': qlora_layer.get_qlora_weights()
            }
        return qlora_weights
    
    def load_qlora_weights(self, qlora_weights: Dict[str, Dict[str, torch.Tensor]]):
        """
        Load QLoRA weights.
        
        Args:
            qlora_weights: Dictionary containing QLoRA weights
        """
        for name, qlora_layer in self.qlora_layers.items():
            if name in qlora_weights:
                qlora_layer.qlora.lora.lora_A.data = qlora_weights[name]['lora_A']
                qlora_layer.qlora.lora.lora_B.data = qlora_weights[name]['lora_B']
    
    def reset_qlora_parameters(self):
        """Reset all QLoRA parameters to initial values."""
        for qlora_layer in self.qlora_layers.values():
            qlora_layer.reset_qlora_parameters()
    
    def set_lora_alpha(self, alpha: float):
        """
        Set LoRA alpha for all layers.
        
        Args:
            alpha: New alpha value
        """
        for qlora_layer in self.qlora_layers.values():
            qlora_layer.set_lora_alpha(alpha)
        self.config.lora_alpha = alpha
    
    def get_applied_modules(self) -> List[str]:
        """Get list of modules that have QLoRA applied."""
        return self.applied_modules.copy()
    
    def get_qlora_config(self) -> Dict[str, Any]:
        """Get current QLoRA configuration."""
        return {
            'lora_rank': self.config.lora_rank,
            'lora_alpha': self.config.lora_alpha,
            'lora_dropout': self.config.lora_dropout,
            'target_modules': self.config.target_modules,
            'quantization_bits': 4,
            'applied_modules': self.applied_modules
        }
    
    def print_summary(self):
        """Print a summary of QLoRA application."""
        param_counts = self.get_parameter_count()
        memory_usage = self.get_memory_usage()
        
        print(f"\n📊 QLoRA Summary:")
        print(f"   Applied to {len(self.qlora_layers)} modules")
        print(f"   Total parameters: {param_counts['total']:,}")
        print(f"   Trainable parameters: {param_counts['trainable']:,}")
        print(f"   Frozen parameters: {param_counts['frozen']:,}")
        print(f"   Trainable percentage: {param_counts['trainable_percentage']:.2f}%")
        print(f"   Total memory: {memory_usage['total_memory_mb']:.2f} MB")
        print(f"   Trainable memory: {memory_usage['trainable_memory_mb']:.2f} MB")
        print(f"   Quantized memory: {memory_usage['quantized_memory_mb']:.2f} MB")
        print(f"   Memory savings: {memory_usage['memory_savings_mb']:.2f} MB")
        print(f"   Applied modules: {self.applied_modules}")
