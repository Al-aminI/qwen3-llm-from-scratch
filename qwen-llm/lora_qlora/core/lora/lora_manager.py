"""
LoRA manager for model adaptation.

This module provides a manager class for applying LoRA adaptation to entire models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

from .lora_linear import LoRALinear
from ..quantization.quantization_expert import QuantizationConfig


class LoRAManager:
    """
    🎯 LORA MANAGER
    
    Manages LoRA adaptation for entire models.
    
    This class provides functionality to:
    - Apply LoRA to specified modules in a model
    - Track and manage LoRA layers
    - Analyze parameter counts and memory usage
    - Save and load LoRA weights
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        """
        Initialize LoRA manager.
        
        Args:
            model: The model to apply LoRA to
            config: LoRA configuration
        """
        self.model = model
        self.config = config
        self.lora_layers = {}
        self.original_layers = {}
        self.applied_modules = []
    
    def apply_lora(self, target_modules: Optional[List[str]] = None):
        """
        🎯 APPLY LORA TO MODEL
        
        Replaces specified linear layers with LoRA-adapted versions.
        
        Args:
            target_modules: List of module names to apply LoRA to.
                          If None, uses config.target_modules.
        """
        if target_modules is None:
            target_modules = self.config.target_modules
        
        print(f"🎯 Applying LoRA to model...")
        print(f"   Target modules: {target_modules}")
        print(f"   LoRA rank: {self.config.lora_rank}")
        print(f"   LoRA alpha: {self.config.lora_alpha}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                print(f"   Applying LoRA to: {name}")
                
                # Store original layer
                self.original_layers[name] = module
                
                # Create LoRA layer
                lora_layer = LoRALinear(
                    module,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
                
                # Replace in model
                self._replace_module(self.model, name, lora_layer)
                self.lora_layers[name] = lora_layer
                self.applied_modules.append(name)
        
        print(f"✅ LoRA applied to {len(self.lora_layers)} modules")
    
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
        Get all trainable parameters (LoRA only).
        
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        for lora_layer in self.lora_layers.values():
            trainable_params.extend(lora_layer.lora.parameters())
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
        # LoRA weights: 4 bytes per parameter (fp32)
        
        total_memory = total_params * 4 / (1024 * 1024)  # MB
        trainable_memory = trainable_params * 4 / (1024 * 1024)  # MB
        frozen_memory = total_memory - trainable_memory
        
        return {
            'total_memory_mb': total_memory,
            'trainable_memory_mb': trainable_memory,
            'frozen_memory_mb': frozen_memory,
            'trainable_percentage': (trainable_memory / total_memory) * 100 if total_memory > 0 else 0
        }
    
    def get_lora_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get LoRA weights for saving.
        
        Returns:
            Dictionary containing LoRA weights
        """
        lora_weights = {}
        for name, lora_layer in self.lora_layers.items():
            lora_weights[name] = {
                'lora_A': lora_layer.lora.lora_A.data.clone(),
                'lora_B': lora_layer.lora.lora_B.data.clone()
            }
        return lora_weights
    
    def load_lora_weights(self, lora_weights: Dict[str, Dict[str, torch.Tensor]]):
        """
        Load LoRA weights.
        
        Args:
            lora_weights: Dictionary containing LoRA weights
        """
        for name, lora_layer in self.lora_layers.items():
            if name in lora_weights:
                lora_layer.lora.lora_A.data = lora_weights[name]['lora_A']
                lora_layer.lora.lora_B.data = lora_weights[name]['lora_B']
    
    def merge_lora_weights(self) -> Dict[str, torch.Tensor]:
        """
        Merge LoRA weights into original weights.
        
        Returns:
            Dictionary containing merged weights
        """
        merged_weights = {}
        for name, lora_layer in self.lora_layers.items():
            merged_weights[name] = lora_layer.merge_lora_weights()
        return merged_weights
    
    def reset_lora_parameters(self):
        """Reset all LoRA parameters to initial values."""
        for lora_layer in self.lora_layers.values():
            lora_layer.reset_lora_parameters()
    
    def set_lora_alpha(self, alpha: float):
        """
        Set LoRA alpha for all layers.
        
        Args:
            alpha: New alpha value
        """
        for lora_layer in self.lora_layers.values():
            lora_layer.set_lora_alpha(alpha)
        self.config.lora_alpha = alpha
    
    def get_applied_modules(self) -> List[str]:
        """Get list of modules that have LoRA applied."""
        return self.applied_modules.copy()
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get current LoRA configuration."""
        return {
            'lora_rank': self.config.lora_rank,
            'lora_alpha': self.config.lora_alpha,
            'lora_dropout': self.config.lora_dropout,
            'target_modules': self.config.target_modules,
            'applied_modules': self.applied_modules
        }
    
    def print_summary(self):
        """Print a summary of LoRA application."""
        param_counts = self.get_parameter_count()
        memory_usage = self.get_memory_usage()
        
        print(f"\n📊 LoRA Summary:")
        print(f"   Applied to {len(self.lora_layers)} modules")
        print(f"   Total parameters: {param_counts['total']:,}")
        print(f"   Trainable parameters: {param_counts['trainable']:,}")
        print(f"   Frozen parameters: {param_counts['frozen']:,}")
        print(f"   Trainable percentage: {param_counts['trainable_percentage']:.2f}%")
        print(f"   Total memory: {memory_usage['total_memory_mb']:.2f} MB")
        print(f"   Trainable memory: {memory_usage['trainable_memory_mb']:.2f} MB")
        print(f"   Applied modules: {self.applied_modules}")
