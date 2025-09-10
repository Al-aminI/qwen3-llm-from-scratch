"""
Unit tests for QLoRA components.

This module provides unit tests for QLoRA layers, linear layers, and manager.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lora_qlora.core.qlora.qlora_layer import QLoRALayer
from lora_qlora.core.qlora.qlora_linear import QLoRALinear
from lora_qlora.core.qlora.qlora_manager import QLoRAManager
from lora_qlora.core.quantization.quantization_expert import QuantizationConfig


class TestQLoRALayer(unittest.TestCase):
    """Test cases for QLoRALayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 128
        self.out_features = 64
        self.rank = 16
        self.alpha = 32.0
        self.dropout = 0.1
        
        self.qlora_layer = QLoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
    
    def test_initialization(self):
        """Test QLoRA layer initialization."""
        self.assertEqual(self.qlora_layer.in_features, self.in_features)
        self.assertEqual(self.qlora_layer.out_features, self.out_features)
        self.assertEqual(self.qlora_layer.rank, self.rank)
        self.assertEqual(self.qlora_layer.alpha, self.alpha)
        self.assertEqual(self.qlora_layer.dropout, self.dropout)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 8
        x = torch.randn(batch_size, self.in_features)
        
        output = self.qlora_layer(x)
        
        self.assertEqual(output.shape, (batch_size, self.out_features))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_quantization_parameters(self):
        """Test quantization parameters."""
        # Check that quantization parameters exist
        self.assertTrue(hasattr(self.qlora_layer, 'quantization_scale'))
        self.assertTrue(hasattr(self.qlora_layer, 'quantization_zero_point'))
        
        # Check shapes
        self.assertEqual(self.qlora_layer.quantization_scale.shape, (self.out_features,))
        self.assertEqual(self.qlora_layer.quantization_zero_point.shape, (self.out_features,))
    
    def test_quantize_weights(self):
        """Test weight quantization."""
        weights = torch.randn(self.out_features, self.in_features)
        
        quantized = self.qlora_layer.quantize_weights(weights)
        
        self.assertEqual(quantized.shape, weights.shape)
        self.assertEqual(quantized.dtype, torch.uint8)
        
        # Check that values are in 4-bit range
        self.assertTrue(torch.all(quantized >= 0))
        self.assertTrue(torch.all(quantized <= 15))
    
    def test_dequantize_weights(self):
        """Test weight dequantization."""
        weights = torch.randn(self.out_features, self.in_features)
        quantized = self.qlora_layer.quantize_weights(weights)
        
        dequantized = self.qlora_layer.dequantize_weights(quantized)
        
        self.assertEqual(dequantized.shape, weights.shape)
        self.assertEqual(dequantized.dtype, torch.float32)
    
    def test_get_quantized_weights(self):
        """Test getting quantized weights."""
        quantized_weights = self.qlora_layer.get_quantized_weights()
        
        self.assertIsInstance(quantized_weights, torch.Tensor)
        self.assertEqual(quantized_weights.dtype, torch.uint8)
    
    def test_get_memory_usage(self):
        """Test getting memory usage."""
        memory_usage = self.qlora_layer.get_memory_usage()
        
        self.assertIn('lora_parameters', memory_usage)
        self.assertIn('quantization_parameters', memory_usage)
        self.assertIn('quantized_weights_bytes', memory_usage)
        self.assertIn('total_parameters', memory_usage)
        self.assertIn('memory_savings_bytes', memory_usage)
        
        # Memory savings should be positive
        self.assertGreater(memory_usage['memory_savings_bytes'], 0)
    
    def test_reset_parameters(self):
        """Test parameter reset."""
        # Get initial parameters
        initial_lora_A = self.qlora_layer.lora.lora_A.data.clone()
        initial_lora_B = self.qlora_layer.lora.lora_B.data.clone()
        initial_scale = self.qlora_layer.quantization_scale.data.clone()
        initial_zero_point = self.qlora_layer.quantization_zero_point.data.clone()
        
        # Reset parameters
        self.qlora_layer.reset_parameters()
        
        # Check that parameters changed
        self.assertFalse(torch.equal(initial_lora_A, self.qlora_layer.lora.lora_A.data))
        self.assertFalse(torch.equal(initial_lora_B, self.qlora_layer.lora.lora_B.data))
        self.assertFalse(torch.equal(initial_scale, self.qlora_layer.quantization_scale.data))
        self.assertFalse(torch.equal(initial_zero_point, self.qlora_layer.quantization_zero_point.data))
    
    def test_set_lora_alpha(self):
        """Test setting LoRA alpha."""
        new_alpha = 64.0
        self.qlora_layer.set_lora_alpha(new_alpha)
        
        self.assertEqual(self.qlora_layer.alpha, new_alpha)
        self.assertEqual(self.qlora_layer.lora.alpha, new_alpha)


class TestQLoRALinear(unittest.TestCase):
    """Test cases for QLoRALinear."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 128
        self.out_features = 64
        self.rank = 16
        self.alpha = 32.0
        self.dropout = 0.1
        
        # Create original linear layer
        self.original_layer = nn.Linear(self.in_features, self.out_features)
        
        # Create QLoRA linear layer
        self.qlora_linear = QLoRALinear(
            self.original_layer,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
    
    def test_initialization(self):
        """Test QLoRA linear layer initialization."""
        self.assertEqual(self.qlora_linear.in_features, self.in_features)
        self.assertEqual(self.qlora_linear.out_features, self.out_features)
        self.assertEqual(self.qlora_linear.rank, self.rank)
        self.assertEqual(self.qlora_linear.alpha, self.alpha)
        self.assertEqual(self.qlora_linear.dropout, self.dropout)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 8
        x = torch.randn(batch_size, self.in_features)
        
        output = self.qlora_linear(x)
        
        self.assertEqual(output.shape, (batch_size, self.out_features))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_get_qlora_weights(self):
        """Test getting QLoRA weights."""
        qlora_weights = self.qlora_linear.get_qlora_weights()
        
        self.assertIsInstance(qlora_weights, torch.Tensor)
        self.assertEqual(qlora_weights.dtype, torch.uint8)
    
    def test_get_memory_usage(self):
        """Test getting memory usage."""
        memory_usage = self.qlora_linear.get_memory_usage()
        
        self.assertIn('lora_parameters', memory_usage)
        self.assertIn('quantization_parameters', memory_usage)
        self.assertIn('quantized_weights_bytes', memory_usage)
        self.assertIn('total_parameters', memory_usage)
        self.assertIn('memory_savings_bytes', memory_usage)
    
    def test_reset_qlora_parameters(self):
        """Test resetting QLoRA parameters."""
        # Get initial parameters
        initial_lora_A = self.qlora_linear.qlora.lora.lora_A.data.clone()
        initial_lora_B = self.qlora_linear.qlora.lora.lora_B.data.clone()
        
        # Reset parameters
        self.qlora_linear.reset_qlora_parameters()
        
        # Check that parameters changed
        self.assertFalse(torch.equal(initial_lora_A, self.qlora_linear.qlora.lora.lora_A.data))
        self.assertFalse(torch.equal(initial_lora_B, self.qlora_linear.qlora.lora.lora_B.data))


class TestQLoRAManager(unittest.TestCase):
    """Test cases for QLoRAManager."""
    
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
        
        # Create QLoRA config
        self.config = QuantizationConfig(
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.1,
            target_modules=["0", "2", "4"]  # Target linear layers
        )
        
        # Create QLoRA manager
        self.qlora_manager = QLoRAManager(self.model, self.config)
    
    def test_initialization(self):
        """Test QLoRA manager initialization."""
        self.assertEqual(self.qlora_manager.model, self.model)
        self.assertEqual(self.qlora_manager.config, self.config)
        self.assertEqual(len(self.qlora_manager.qlora_layers), 0)
        self.assertEqual(len(self.qlora_manager.applied_modules), 0)
    
    def test_apply_qlora(self):
        """Test applying QLoRA to model."""
        self.qlora_manager.apply_qlora()
        
        # Check that QLoRA was applied
        self.assertGreater(len(self.qlora_manager.qlora_layers), 0)
        self.assertGreater(len(self.qlora_manager.applied_modules), 0)
        
        # Check that target modules were replaced
        for module_name in self.qlora_manager.applied_modules:
            self.assertIn(module_name, self.qlora_manager.qlora_layers)
    
    def test_get_trainable_parameters(self):
        """Test getting trainable parameters."""
        self.qlora_manager.apply_qlora()
        
        trainable_params = self.qlora_manager.get_trainable_parameters()
        
        self.assertIsInstance(trainable_params, list)
        self.assertGreater(len(trainable_params), 0)
        
        # All parameters should be QLoRA parameters
        for param in trainable_params:
            self.assertTrue(param.requires_grad)
    
    def test_get_parameter_count(self):
        """Test getting parameter count."""
        self.qlora_manager.apply_qlora()
        
        param_counts = self.qlora_manager.get_parameter_count()
        
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
        self.qlora_manager.apply_qlora()
        
        memory_usage = self.qlora_manager.get_memory_usage()
        
        self.assertIn('total_memory_mb', memory_usage)
        self.assertIn('trainable_memory_mb', memory_usage)
        self.assertIn('frozen_memory_mb', memory_usage)
        self.assertIn('quantized_memory_mb', memory_usage)
        self.assertIn('memory_savings_mb', memory_usage)
        self.assertIn('trainable_percentage', memory_usage)
        
        # Memory usage should be positive
        self.assertGreater(memory_usage['total_memory_mb'], 0)
        self.assertGreater(memory_usage['trainable_memory_mb'], 0)
        
        # Memory savings should be positive
        self.assertGreater(memory_usage['memory_savings_mb'], 0)
    
    def test_get_qlora_weights(self):
        """Test getting QLoRA weights."""
        self.qlora_manager.apply_qlora()
        
        qlora_weights = self.qlora_manager.get_qlora_weights()
        
        self.assertIsInstance(qlora_weights, dict)
        self.assertEqual(len(qlora_weights), len(self.qlora_manager.qlora_layers))
        
        for module_name, weights in qlora_weights.items():
            self.assertIn('lora_A', weights)
            self.assertIn('lora_B', weights)
            self.assertIn('quantized_weights', weights)
            self.assertIsInstance(weights['lora_A'], torch.Tensor)
            self.assertIsInstance(weights['lora_B'], torch.Tensor)
            self.assertIsInstance(weights['quantized_weights'], torch.Tensor)
    
    def test_load_qlora_weights(self):
        """Test loading QLoRA weights."""
        self.qlora_manager.apply_qlora()
        
        # Get original weights
        original_weights = self.qlora_manager.get_qlora_weights()
        
        # Modify weights
        for module_name, weights in original_weights.items():
            weights['lora_A'] = torch.randn_like(weights['lora_A'])
            weights['lora_B'] = torch.randn_like(weights['lora_B'])
        
        # Load modified weights
        self.qlora_manager.load_qlora_weights(original_weights)
        
        # Check that weights were loaded
        loaded_weights = self.qlora_manager.get_qlora_weights()
        for module_name in original_weights:
            self.assertTrue(torch.equal(original_weights[module_name]['lora_A'], loaded_weights[module_name]['lora_A']))
            self.assertTrue(torch.equal(original_weights[module_name]['lora_B'], loaded_weights[module_name]['lora_B']))


if __name__ == '__main__':
    unittest.main()
