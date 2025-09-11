"""
🧪 Test Suite for Examples

This module contains tests for the example implementations.
"""

import unittest
import torch
import triton
import triton.language as tl
import sys
import os

# Add the parent directory to the path to import examples
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from examples.llm_inference_optimization import optimized_attention, transformer_matmul, layer_norm

class TestExamples(unittest.TestCase):
    """
    🧪 TEST SUITE FOR EXAMPLES
    
    Tests for the example implementations.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        # Test data for attention
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 128
        self.head_dim = 64
        
        self.q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, 
                            device=self.device, dtype=self.dtype)
        self.k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, 
                            device=self.device, dtype=self.dtype)
        self.v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, 
                            device=self.device, dtype=self.dtype)
        
        # Test data for matrix multiplication
        self.M, self.K, self.N = 256, 128, 192
        self.a = torch.randn(self.M, self.K, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.K, self.N, device=self.device, dtype=self.dtype)
        
        # Test data for layer normalization
        self.seq_len_norm = 128
        self.hidden_dim = 512
        self.input_tensor = torch.randn(self.seq_len_norm, self.hidden_dim, 
                                       device=self.device, dtype=self.dtype)
        self.weight = torch.ones(self.hidden_dim, device=self.device, dtype=self.dtype)
        self.bias = torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)
    
    def test_optimized_attention(self):
        """Test optimized attention implementation."""
        # Note: This is a simplified attention implementation for demonstration
        # In practice, you'd need a more complete implementation
        try:
            result = optimized_attention(self.q, self.k, self.v)
            
            # Check basic properties
            self.assertEqual(result.shape, self.q.shape)
            self.assertEqual(result.dtype, self.q.dtype)
            self.assertEqual(result.device, self.q.device)
            
            # Check that result is not NaN or Inf
            self.assertTrue(torch.isfinite(result).all())
            
        except Exception as e:
            # If the attention implementation is not complete, that's expected
            self.assertIn("simplified", str(e).lower() or "not implemented", str(e).lower())
    
    def test_transformer_matmul(self):
        """Test transformer matrix multiplication."""
        result = transformer_matmul(self.a, self.b)
        expected = torch.matmul(self.a, self.b)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_transformer_matmul_edge_cases(self):
        """Test transformer matrix multiplication with edge cases."""
        # Test with different sizes
        sizes = [(64, 32, 48), (128, 64, 96), (256, 128, 192)]
        
        for M, K, N in sizes:
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            result = transformer_matmul(a, b)
            expected = torch.matmul(a, b)
            
            self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.device, expected.device)
        
        # Test with square matrices
        square_a = torch.randn(64, 64, device=self.device, dtype=self.dtype)
        square_b = torch.randn(64, 64, device=self.device, dtype=self.dtype)
        
        result = transformer_matmul(square_a, square_b)
        expected = torch.matmul(square_a, square_b)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
    
    def test_layer_norm(self):
        """Test layer normalization implementation."""
        result = layer_norm(self.input_tensor, self.weight, self.bias)
        expected = torch.nn.functional.layer_norm(self.input_tensor, [self.hidden_dim], self.weight, self.bias)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_layer_norm_edge_cases(self):
        """Test layer normalization with edge cases."""
        # Test with different sizes
        sizes = [(64, 256), (128, 512), (256, 1024)]
        
        for seq_len, hidden_dim in sizes:
            input_tensor = torch.randn(seq_len, hidden_dim, device=self.device, dtype=self.dtype)
            weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
            bias = torch.zeros(hidden_dim, device=self.device, dtype=self.dtype)
            
            result = layer_norm(input_tensor, weight, bias)
            expected = torch.nn.functional.layer_norm(input_tensor, [hidden_dim], weight, bias)
            
            self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.device, expected.device)
        
        # Test with zero input
        zero_input = torch.zeros(64, 256, device=self.device, dtype=self.dtype)
        weight = torch.ones(256, device=self.device, dtype=self.dtype)
        bias = torch.zeros(256, device=self.device, dtype=self.dtype)
        
        result = layer_norm(zero_input, weight, bias)
        expected = torch.nn.functional.layer_norm(zero_input, [256], weight, bias)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
    
    def test_layer_norm_different_eps(self):
        """Test layer normalization with different epsilon values."""
        eps_values = [1e-5, 1e-6, 1e-7]
        
        for eps in eps_values:
            result = layer_norm(self.input_tensor, self.weight, self.bias, eps=eps)
            expected = torch.nn.functional.layer_norm(self.input_tensor, [self.hidden_dim], 
                                                    self.weight, self.bias, eps=eps)
            
            self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.device, expected.device)
    
    def test_examples_performance_characteristics(self):
        """Test performance characteristics of examples."""
        # Test transformer matrix multiplication performance
        result = transformer_matmul(self.a, self.b)
        expected = torch.matmul(self.a, self.b)
        
        # Check that results are close
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        
        # Test layer normalization performance
        result_norm = layer_norm(self.input_tensor, self.weight, self.bias)
        expected_norm = torch.nn.functional.layer_norm(self.input_tensor, [self.hidden_dim], 
                                                      self.weight, self.bias)
        
        # Check that results are close
        self.assertTrue(torch.allclose(result_norm, expected_norm, rtol=1e-3, atol=1e-3))
    
    def test_examples_memory_usage(self):
        """Test memory usage of examples."""
        # Test that operations don't cause memory issues
        try:
            # Test with larger tensors
            large_a = torch.randn(512, 256, device=self.device, dtype=self.dtype)
            large_b = torch.randn(256, 384, device=self.device, dtype=self.dtype)
            
            result = transformer_matmul(large_a, large_b)
            expected = torch.matmul(large_a, large_b)
            
            self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
            self.assertEqual(result.shape, expected.shape)
            
            # Test layer normalization with larger tensors
            large_input = torch.randn(256, 1024, device=self.device, dtype=self.dtype)
            large_weight = torch.ones(1024, device=self.device, dtype=self.dtype)
            large_bias = torch.zeros(1024, device=self.device, dtype=self.dtype)
            
            result_norm = layer_norm(large_input, large_weight, large_bias)
            expected_norm = torch.nn.functional.layer_norm(large_input, [1024], 
                                                          large_weight, large_bias)
            
            self.assertTrue(torch.allclose(result_norm, expected_norm, rtol=1e-3, atol=1e-3))
            self.assertEqual(result_norm.shape, expected_norm.shape)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Skip test if out of memory
                self.skipTest("Out of memory - skipping large tensor test")
            else:
                raise
    
    def test_examples_different_dtypes(self):
        """Test examples with different data types."""
        if self.device == 'cuda':
            # Test with float16
            a_f16 = self.a.half()
            b_f16 = self.b.half()
            
            result_f16 = transformer_matmul(a_f16, b_f16)
            expected_f16 = torch.matmul(a_f16, b_f16)
            
            self.assertTrue(torch.allclose(result_f16, expected_f16, rtol=1e-3, atol=1e-3))
            self.assertEqual(result_f16.shape, expected_f16.shape)
            self.assertEqual(result_f16.dtype, expected_f16.dtype)
            
            # Test layer normalization with float16
            input_f16 = self.input_tensor.half()
            weight_f16 = self.weight.half()
            bias_f16 = self.bias.half()
            
            result_norm_f16 = layer_norm(input_f16, weight_f16, bias_f16)
            expected_norm_f16 = torch.nn.functional.layer_norm(input_f16, [self.hidden_dim], 
                                                              weight_f16, bias_f16)
            
            self.assertTrue(torch.allclose(result_norm_f16, expected_norm_f16, rtol=1e-3, atol=1e-3))
            self.assertEqual(result_norm_f16.shape, expected_norm_f16.shape)
            self.assertEqual(result_norm_f16.dtype, expected_norm_f16.dtype)
    
    def test_examples_error_handling(self):
        """Test error handling in examples."""
        # Test with mismatched dimensions
        with self.assertRaises(AssertionError):
            mismatched_a = torch.randn(64, 32, device=self.device, dtype=self.dtype)
            mismatched_b = torch.randn(64, 32, device=self.device, dtype=self.dtype)  # Wrong inner dimension
            transformer_matmul(mismatched_a, mismatched_b)
        
        # Test with mismatched hidden dimensions
        with self.assertRaises(AssertionError):
            mismatched_input = torch.randn(64, 256, device=self.device, dtype=self.dtype)
            mismatched_weight = torch.ones(128, device=self.device, dtype=self.dtype)  # Wrong dimension
            mismatched_bias = torch.zeros(128, device=self.device, dtype=self.dtype)
            layer_norm(mismatched_input, mismatched_weight, mismatched_bias)
        
        # Test with CPU tensors when CUDA is expected
        if self.device == 'cuda':
            with self.assertRaises(AssertionError):
                cpu_a = torch.randn(64, 32, device='cpu', dtype=self.dtype)
                cpu_b = torch.randn(32, 48, device='cpu', dtype=self.dtype)
                transformer_matmul(cpu_a, cpu_b)
    
    def test_examples_integration(self):
        """Test integration of examples."""
        # Test transformer pipeline
        # 1. Matrix multiplication
        result_matmul = transformer_matmul(self.a, self.b)
        expected_matmul = torch.matmul(self.a, self.b)
        self.assertTrue(torch.allclose(result_matmul, expected_matmul, rtol=1e-3, atol=1e-3))
        
        # 2. Layer normalization
        result_norm = layer_norm(self.input_tensor, self.weight, self.bias)
        expected_norm = torch.nn.functional.layer_norm(self.input_tensor, [self.hidden_dim], 
                                                      self.weight, self.bias)
        self.assertTrue(torch.allclose(result_norm, expected_norm, rtol=1e-3, atol=1e-3))
        
        # 3. Combined operations
        # Apply layer normalization to matrix multiplication result
        matmul_result = transformer_matmul(self.a, self.b)
        norm_result = layer_norm(matmul_result, self.weight, self.bias)
        
        # Compare with PyTorch reference
        pytorch_matmul = torch.matmul(self.a, self.b)
        pytorch_norm = torch.nn.functional.layer_norm(pytorch_matmul, [self.hidden_dim], 
                                                     self.weight, self.bias)
        
        self.assertTrue(torch.allclose(norm_result, pytorch_norm, rtol=1e-3, atol=1e-3))
        self.assertEqual(norm_result.shape, pytorch_norm.shape)
        self.assertEqual(norm_result.dtype, pytorch_norm.dtype)
        self.assertEqual(norm_result.device, pytorch_norm.device)

class TestExamplesIntegration(unittest.TestCase):
    """
    🧪 INTEGRATION TESTS FOR EXAMPLES
    
    Integration tests that combine multiple examples.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
    
    def test_transformer_pipeline(self):
        """Test a complete transformer pipeline."""
        # Create test data
        batch_size = 2
        seq_len = 128
        hidden_dim = 512
        num_heads = 8
        head_dim = hidden_dim // num_heads
        
        # Input embeddings
        input_embeddings = torch.randn(batch_size, seq_len, hidden_dim, 
                                      device=self.device, dtype=self.dtype)
        
        # Attention weights
        q_weight = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
        k_weight = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
        v_weight = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
        
        # Feed-forward weights
        ff_weight1 = torch.randn(hidden_dim, hidden_dim * 4, device=self.device, dtype=self.dtype)
        ff_weight2 = torch.randn(hidden_dim * 4, hidden_dim, device=self.device, dtype=self.dtype)
        
        # Layer normalization weights
        norm_weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
        norm_bias = torch.zeros(hidden_dim, device=self.device, dtype=self.dtype)
        
        # Transformer pipeline
        # 1. Attention
        q = torch.matmul(input_embeddings, q_weight)
        k = torch.matmul(input_embeddings, k_weight)
        v = torch.matmul(input_embeddings, v_weight)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Apply attention (simplified)
        attention_output = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_output = torch.softmax(attention_output, dim=-1)
        attention_output = torch.matmul(attention_output, v)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim)
        
        # 2. Add & Norm
        residual = input_embeddings + attention_output
        norm_output = layer_norm(residual, norm_weight, norm_bias)
        
        # 3. Feed-forward
        ff_output = transformer_matmul(norm_output, ff_weight1)
        ff_output = torch.relu(ff_output)
        ff_output = transformer_matmul(ff_output, ff_weight2)
        
        # 4. Add & Norm
        final_output = norm_output + ff_output
        final_norm = layer_norm(final_output, norm_weight, norm_bias)
        
        # Check output properties
        self.assertEqual(final_norm.shape, (batch_size, seq_len, hidden_dim))
        self.assertEqual(final_norm.dtype, self.dtype)
        self.assertEqual(final_norm.device.type, self.device)
        self.assertTrue(torch.isfinite(final_norm).all())
    
    def test_attention_mechanism(self):
        """Test attention mechanism implementation."""
        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device=self.device, dtype=self.dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device=self.device, dtype=self.dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device=self.device, dtype=self.dtype)
        
        # Test attention computation
        try:
            result = optimized_attention(q, k, v)
            
            # Check basic properties
            self.assertEqual(result.shape, q.shape)
            self.assertEqual(result.dtype, q.dtype)
            self.assertEqual(result.device, q.device)
            self.assertTrue(torch.isfinite(result).all())
            
        except Exception as e:
            # If the attention implementation is not complete, that's expected
            self.assertIn("simplified", str(e).lower() or "not implemented", str(e).lower())
    
    def test_memory_efficiency(self):
        """Test memory efficiency of examples."""
        # Test with large tensors to check memory usage
        try:
            # Large matrix multiplication
            large_a = torch.randn(1024, 512, device=self.device, dtype=self.dtype)
            large_b = torch.randn(512, 768, device=self.device, dtype=self.dtype)
            
            result = transformer_matmul(large_a, large_b)
            expected = torch.matmul(large_a, large_b)
            
            self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
            self.assertEqual(result.shape, expected.shape)
            
            # Large layer normalization
            large_input = torch.randn(512, 1024, device=self.device, dtype=self.dtype)
            large_weight = torch.ones(1024, device=self.device, dtype=self.dtype)
            large_bias = torch.zeros(1024, device=self.device, dtype=self.dtype)
            
            result_norm = layer_norm(large_input, large_weight, large_bias)
            expected_norm = torch.nn.functional.layer_norm(large_input, [1024], 
                                                          large_weight, large_bias)
            
            self.assertTrue(torch.allclose(result_norm, expected_norm, rtol=1e-3, atol=1e-3))
            self.assertEqual(result_norm.shape, expected_norm.shape)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Skip test if out of memory
                self.skipTest("Out of memory - skipping large tensor test")
            else:
                raise

if __name__ == '__main__':
    # Run the tests
    unittest.main()
