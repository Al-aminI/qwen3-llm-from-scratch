"""
Unit tests for LoRA components.

This module provides unit tests for LoRA layers, linear layers, and manager.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lora_qlora.core.lora.lora_layer import LoRALayer
from lora_qlora.core.lora.lora_linear import LoRALinear
from lora_qlora.core.lora.lora_manager import LoRAManager
from lora_qlora.core.quantization.quantization_expert import QuantizationConfig


class TestLoRALayer(unittest.TestCase):
    """Test cases for LoRALayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 128
        self.out_features = 64
        self.rank = 16
        self.alpha = 32.0
        self.dropout = 0.1
        
        self.lora_layer = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
    
    def test_initialization(self):
        """Test LoRA layer initialization."""
        self.assertEqual(self.lora_layer.in_features, self.in_features)
        self.assertEqual(self.lora_layer.out_features, self.out_features)
        self.assertEqual(self.lora_layer.rank, self.rank)
        self.assertEqual(self.lora_layer.alpha, self.alpha)
        self.assertEqual(self.lora_layer.dropout, self.dropout)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 8
        x = torch.randn(batch_size, self.in_features)
        
        output = self.lora_layer(x)
        
        self.assertEqual(output.shape, (batch_size, self.out_features))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_parameter_count(self):
        """Test parameter count."""
        param_count = self.lora_layer.get_parameter_count()
        
        # LoRA should have fewer parameters than full layer
        full_layer_params = self.in_features * self.out_features
        self.assertLess(param_count, full_layer_params)
        
        # Should have 2 * rank * (in_features + out_features) parameters
        expected_params = 2 * self.rank * (self.in_features + self.out_features)
        self.assertEqual(param_count, expected_params)
    
    def test_reset_parameters(self):
        """Test parameter reset."""
        # Get initial parameters
        initial_lora_A = self.lora_layer.lora_A.data.clone()
        initial_lora_B = self.lora_layer.lora_B.data.clone()
        
        # Reset parameters
        self.lora_layer.reset_parameters()
        
        # Check that parameters changed
        self.assertFalse(torch.equal(initial_lora_A, self.lora_layer.lora_A.data))
        self.assertFalse(torch.equal(initial_lora_B, self.lora_layer.lora_B.data))
    
    def test_set_lora_alpha(self):
        """Test setting LoRA alpha."""
        new_alpha = 64.0
        self.lora_layer.set_lora_alpha(new_alpha)
        
        self.assertEqual(self.lora_layer.alpha, new_alpha)


class TestLoRALinear(unittest.TestCase):
    """Test cases for LoRALinear."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 128
        self.out_features = 64
        self.rank = 16
        self.alpha = 32.0
        self.dropout = 0.1
        
        # Create original linear layer
        self.original_layer = nn.Linear(self.in_features, self.out_features)
        
        # Create LoRA linear layer
        self.lora_linear = LoRALinear(
            self.original_layer,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
    
    def test_initialization(self):
        """Test LoRA linear layer initialization."""
        self.assertEqual(self.lora_linear.in_features, self.in_features)
        self.assertEqual(self.lora_linear.out_features, self.out_features)
        self.assertEqual(self.lora_linear.rank, self.rank)
        self.assertEqual(self.lora_linear.alpha, self.alpha)
        self.assertEqual(self.lora_linear.dropout, self.dropout)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 8
        x = torch.randn(batch_size, self.in_features)
        
        output = self.lora_linear(x)
        
        self.assertEqual(output.shape, (batch_size, self.out_features))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_get_lora_weights(self):
        """Test getting LoRA weights."""
        lora_weights = self.lora_linear.get_lora_weights()
        
        self.assertIn('lora_A', lora_weights)
        self.assertIn('lora_B', lora_weights)
        self.assertEqual(lora_weights['lora_A'].shape, (self.rank, self.in_features))
        self.assertEqual(lora_weights['lora_B'].shape, (self.out_features, self.rank))
    
    def test_reset_lora_parameters(self):
        """Test resetting LoRA parameters."""
        # Get initial parameters
        initial_lora_A = self.lora_linear.lora.lora_A.data.clone()
        initial_lora_B = self.lora_linear.lora.lora_B.data.clone()
        
        # Reset parameters
        self.lora_linear.reset_lora_parameters()
        
        # Check that parameters changed
        self.assertFalse(torch.equal(initial_lora_A, self.lora_linear.lora.lora_A.data))
        self.assertFalse(torch.equal(initial_lora_B, self.lora_linear.lora.lora_B.data))


class TestLoRAManager(unittest.TestCase):
    """Test cases for LoRAManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Create LoRA config
        self.config = QuantizationConfig(
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.1,
            target_modules=["0", "2", "4"]  # Target linear layers
        )
        
        # Create LoRA manager
        self.lora_manager = LoRAManager(self.model, self.config)
    
    def test_initialization(self):
        """Test LoRA manager initialization."""
        self.assertEqual(self.lora_manager.model, self.model)
        self.assertEqual(self.lora_manager.config, self.config)
        self.assertEqual(len(self.lora_manager.lora_layers), 0)
        self.assertEqual(len(self.lora_manager.applied_modules), 0)
    
    def test_apply_lora(self):
        """Test applying LoRA to model."""
        self.lora_manager.apply_lora()
        
        # Check that LoRA was applied
        self.assertGreater(len(self.lora_manager.lora_layers), 0)
        self.assertGreater(len(self.lora_manager.applied_modules), 0)
        
        # Check that target modules were replaced
        for module_name in self.lora_manager.applied_modules:
            self.assertIn(module_name, self.lora_manager.lora_layers)
    
    def test_get_trainable_parameters(self):
        """Test getting trainable parameters."""
        self.lora_manager.apply_lora()
        
        trainable_params = self.lora_manager.get_trainable_parameters()
        
        self.assertIsInstance(trainable_params, list)
        self.assertGreater(len(trainable_params), 0)
        
        # All parameters should be LoRA parameters
        for param in trainable_params:
            self.assertTrue(param.requires_grad)
    
    def test_get_parameter_count(self):
        """Test getting parameter count."""
        self.lora_manager.apply_lora()
        
        param_counts = self.lora_manager.get_parameter_count()
        
        self.assertIn('total', param_counts)
        self.assertIn('trainable', param_counts)
        self.assertIn('frozen', param_counts)
        self.assertIn('trainable_percentage', param_counts)
        
        # Trainable should be less than total
        self.assertLess(param_counts['trainable'], param_counts['total'])
        
        # Trainable percentage should be reasonable
        self.assertGreater(param_counts['trainable_percentage'], 0)
        self.assertLess(param_counts['trainable_percentage'], 100)
    
    def test_get_memory_usage(self):
        """Test getting memory usage."""
        self.lora_manager.apply_lora()
        
        memory_usage = self.lora_manager.get_memory_usage()
        
        self.assertIn('total_memory_mb', memory_usage)
        self.assertIn('trainable_memory_mb', memory_usage)
        self.assertIn('frozen_memory_mb', memory_usage)
        self.assertIn('trainable_percentage', memory_usage)
        
        # Memory usage should be positive
        self.assertGreater(memory_usage['total_memory_mb'], 0)
        self.assertGreater(memory_usage['trainable_memory_mb'], 0)
    
    def test_get_lora_weights(self):
        """Test getting LoRA weights."""
        self.lora_manager.apply_lora()
        
        lora_weights = self.lora_manager.get_lora_weights()
        
        self.assertIsInstance(lora_weights, dict)
        self.assertEqual(len(lora_weights), len(self.lora_manager.lora_layers))
        
        for module_name, weights in lora_weights.items():
            self.assertIn('lora_A', weights)
            self.assertIn('lora_B', weights)
            self.assertIsInstance(weights['lora_A'], torch.Tensor)
            self.assertIsInstance(weights['lora_B'], torch.Tensor)
    
    def test_load_lora_weights(self):
        """Test loading LoRA weights."""
        self.lora_manager.apply_lora()
        
        # Get original weights
        original_weights = self.lora_manager.get_lora_weights()
        
        # Modify weights
        for module_name, weights in original_weights.items():
            weights['lora_A'] = torch.randn_like(weights['lora_A'])
            weights['lora_B'] = torch.randn_like(weights['lora_B'])
        
        # Load modified weights
        self.lora_manager.load_lora_weights(original_weights)
        
        # Check that weights were loaded
        loaded_weights = self.lora_manager.get_lora_weights()
        for module_name in original_weights:
            self.assertTrue(torch.equal(original_weights[module_name]['lora_A'], loaded_weights[module_name]['lora_A']))
            self.assertTrue(torch.equal(original_weights[module_name]['lora_B'], loaded_weights[module_name]['lora_B']))


if __name__ == '__main__':
    unittest.main()
