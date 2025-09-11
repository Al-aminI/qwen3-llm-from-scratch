"""
🧪 Test Suite for Intermediate Lessons

This module contains tests for the intermediate-level Triton tutorials.
"""

import unittest
import torch
import triton
import triton.language as tl
import sys
import os

# Add the parent directory to the path to import lessons
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lessons.intermediate.lesson_04_matrix_operations import basic_matmul, optimized_matmul, batch_matmul, matrix_transpose
from lessons.intermediate.lesson_05_advanced_memory import shared_memory_matmul, cache_friendly_reduction
from lessons.intermediate.lesson_06_kernel_fusion import fused_add_multiply, fused_matmul_activation, fused_loop

class TestIntermediateLessons(unittest.TestCase):
    """
    🧪 TEST SUITE FOR INTERMEDIATE LESSONS
    
    Tests for lessons 4-6 (Matrix Operations, Advanced Memory, Kernel Fusion)
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        # Test matrix sizes
        self.small_matrices = (64, 32, 48)
        self.medium_matrices = (256, 128, 192)
        self.large_matrices = (512, 256, 384)
        
        # Create test data
        self.small_a = torch.randn(self.small_matrices[0], self.small_matrices[1], device=self.device, dtype=self.dtype)
        self.small_b = torch.randn(self.small_matrices[1], self.small_matrices[2], device=self.device, dtype=self.dtype)
        
        self.medium_a = torch.randn(self.medium_matrices[0], self.medium_matrices[1], device=self.device, dtype=self.dtype)
        self.medium_b = torch.randn(self.medium_matrices[1], self.medium_matrices[2], device=self.device, dtype=self.dtype)
        
        self.large_a = torch.randn(self.large_matrices[0], self.large_matrices[1], device=self.device, dtype=self.dtype)
        self.large_b = torch.randn(self.large_matrices[1], self.large_matrices[2], device=self.device, dtype=self.dtype)
    
    def test_lesson_04_basic_matrix_multiplication(self):
        """Test basic matrix multiplication."""
        result = basic_matmul(self.small_a, self.small_b)
        expected = torch.matmul(self.small_a, self.small_b)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_04_optimized_matrix_multiplication(self):
        """Test optimized matrix multiplication."""
        result = optimized_matmul(self.medium_a, self.medium_b)
        expected = torch.matmul(self.medium_a, self.medium_b)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_04_batch_matrix_multiplication(self):
        """Test batch matrix multiplication."""
        batch_size = 4
        batch_a = torch.randn(batch_size, self.small_matrices[0], self.small_matrices[1], device=self.device, dtype=self.dtype)
        batch_b = torch.randn(batch_size, self.small_matrices[1], self.small_matrices[2], device=self.device, dtype=self.dtype)
        
        result = batch_matmul(batch_a, batch_b)
        expected = torch.bmm(batch_a, batch_b)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_04_matrix_transpose(self):
        """Test matrix transpose."""
        result = matrix_transpose(self.small_a)
        expected = self.small_a.t()
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_04_matrix_operations_edge_cases(self):
        """Test matrix operations with edge cases."""
        # Single element matrices
        single_a = torch.tensor([[1.0]], device=self.device, dtype=self.dtype)
        single_b = torch.tensor([[2.0]], device=self.device, dtype=self.dtype)
        result = basic_matmul(single_a, single_b)
        expected = torch.matmul(single_a, single_b)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        
        # Square matrices
        square_a = torch.randn(64, 64, device=self.device, dtype=self.dtype)
        square_b = torch.randn(64, 64, device=self.device, dtype=self.dtype)
        result = basic_matmul(square_a, square_b)
        expected = torch.matmul(square_a, square_b)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        
        # Rectangular matrices
        rect_a = torch.randn(32, 64, device=self.device, dtype=self.dtype)
        rect_b = torch.randn(64, 16, device=self.device, dtype=self.dtype)
        result = basic_matmul(rect_a, rect_b)
        expected = torch.matmul(rect_a, rect_b)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
    
    def test_lesson_05_shared_memory_matrix_multiplication(self):
        """Test shared memory matrix multiplication."""
        result = shared_memory_matmul(self.small_a, self.small_b)
        expected = torch.matmul(self.small_a, self.small_b)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3, atol=1e-3))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_05_cache_friendly_reduction(self):
        """Test cache-friendly reduction."""
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = cache_friendly_reduction(x)
        expected = torch.sum(x)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-5))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_05_advanced_memory_edge_cases(self):
        """Test advanced memory operations with edge cases."""
        # Test with different sizes
        sizes = [100, 500, 1000, 2000]
        for size in sizes:
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            result = cache_friendly_reduction(x)
            expected = torch.sum(x)
            self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-5))
        
        # Test with zero tensor
        zero_x = torch.zeros(100, device=self.device, dtype=self.dtype)
        result = cache_friendly_reduction(zero_x)
        expected = torch.sum(zero_x)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-5))
    
    def test_lesson_06_fused_add_multiply(self):
        """Test fused add-multiply operation."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        c = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = fused_add_multiply(a, b, c)
        expected = (a + b) * c
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_06_fused_matmul_activation(self):
        """Test fused matrix multiplication + activation."""
        result_relu = fused_matmul_activation(self.small_a, self.small_b, "relu")
        expected_relu = torch.relu(torch.matmul(self.small_a, self.small_b))
        
        self.assertTrue(torch.allclose(result_relu, expected_relu, rtol=1e-3, atol=1e-3))
        self.assertEqual(result_relu.shape, expected_relu.shape)
        self.assertEqual(result_relu.dtype, expected_relu.dtype)
        self.assertEqual(result_relu.device, expected_relu.device)
        
        # Test Tanh activation
        result_tanh = fused_matmul_activation(self.small_a, self.small_b, "tanh")
        expected_tanh = torch.tanh(torch.matmul(self.small_a, self.small_b))
        
        self.assertTrue(torch.allclose(result_tanh, expected_tanh, rtol=1e-3, atol=1e-3))
        self.assertEqual(result_tanh.shape, expected_tanh.shape)
        self.assertEqual(result_tanh.dtype, expected_tanh.dtype)
        self.assertEqual(result_tanh.device, expected_tanh.device)
    
    def test_lesson_06_fused_loop(self):
        """Test fused loop operation."""
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = fused_loop(x, num_iterations=5)
        
        # PyTorch reference
        expected = x
        for i in range(5):
            expected = expected * 2.0 + 1.0
            expected = torch.relu(expected)
            expected = expected * 0.5
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-4))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_06_kernel_fusion_edge_cases(self):
        """Test kernel fusion with edge cases."""
        # Test with different sizes
        sizes = [100, 500, 1000, 2000]
        for size in sizes:
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            c = torch.randn(size, device=self.device, dtype=self.dtype)
            
            result = fused_add_multiply(a, b, c)
            expected = (a + b) * c
            self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        
        # Test with zero tensors
        zero_a = torch.zeros(100, device=self.device, dtype=self.dtype)
        zero_b = torch.zeros(100, device=self.device, dtype=self.dtype)
        zero_c = torch.ones(100, device=self.device, dtype=self.dtype)
        
        result = fused_add_multiply(zero_a, zero_b, zero_c)
        expected = (zero_a + zero_b) * zero_c
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
    
    def test_lesson_06_fusion_performance_characteristics(self):
        """Test fusion performance characteristics."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        c = torch.randn(size, device=self.device, dtype=self.dtype)
        
        # Test fused operation
        result_fused = fused_add_multiply(a, b, c)
        
        # Test separate operations
        result_separate = (a + b) * c
        
        self.assertTrue(torch.allclose(result_fused, result_separate, rtol=1e-5, atol=1e-6))
        
        # Test with different activation functions
        activations = ["relu", "tanh", "sigmoid", "gelu"]
        for activation in activations:
            result = fused_matmul_activation(self.small_a, self.small_b, activation)
            self.assertEqual(result.shape, (self.small_a.shape[0], self.small_b.shape[1]))
            self.assertEqual(result.dtype, self.dtype)
            self.assertEqual(result.device, self.device)

class TestIntermediateLessonsIntegration(unittest.TestCase):
    """
    🧪 INTEGRATION TESTS FOR INTERMEDIATE LESSONS
    
    Integration tests that combine multiple intermediate lessons.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
    
    def test_lesson_integration_matrix_operations(self):
        """Test integration of matrix operations across lessons."""
        # Lesson 4: Basic matrix multiplication
        a = torch.randn(64, 32, device=self.device, dtype=self.dtype)
        b = torch.randn(32, 48, device=self.device, dtype=self.dtype)
        
        result_basic = basic_matmul(a, b)
        expected = torch.matmul(a, b)
        self.assertTrue(torch.allclose(result_basic, expected, rtol=1e-3, atol=1e-3))
        
        # Lesson 5: Shared memory matrix multiplication
        result_shared = shared_memory_matmul(a, b)
        self.assertTrue(torch.allclose(result_shared, expected, rtol=1e-3, atol=1e-3))
        
        # Lesson 6: Fused matrix multiplication + activation
        result_fused = fused_matmul_activation(a, b, "relu")
        expected_fused = torch.relu(torch.matmul(a, b))
        self.assertTrue(torch.allclose(result_fused, expected_fused, rtol=1e-3, atol=1e-3))
    
    def test_lesson_integration_memory_optimization(self):
        """Test integration of memory optimization across lessons."""
        # Test cache-friendly reduction
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result_reduction = cache_friendly_reduction(x)
        expected_reduction = torch.sum(x)
        self.assertTrue(torch.allclose(result_reduction, expected_reduction, rtol=1e-4, atol=1e-5))
        
        # Test fused operations
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        c = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result_fused = fused_add_multiply(a, b, c)
        expected_fused = (a + b) * c
        self.assertTrue(torch.allclose(result_fused, expected_fused, rtol=1e-5, atol=1e-6))
    
    def test_lesson_integration_performance_optimization(self):
        """Test integration of performance optimization across lessons."""
        # Test different matrix sizes
        sizes = [(64, 32, 48), (128, 64, 96), (256, 128, 192)]
        
        for M, K, N in sizes:
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            # Test basic matrix multiplication
            result_basic = basic_matmul(a, b)
            expected = torch.matmul(a, b)
            self.assertTrue(torch.allclose(result_basic, expected, rtol=1e-3, atol=1e-3))
            
            # Test optimized matrix multiplication
            result_optimized = optimized_matmul(a, b)
            self.assertTrue(torch.allclose(result_optimized, expected, rtol=1e-3, atol=1e-3))
            
            # Test shared memory matrix multiplication
            result_shared = shared_memory_matmul(a, b)
            self.assertTrue(torch.allclose(result_shared, expected, rtol=1e-3, atol=1e-3))
    
    def test_lesson_integration_kernel_fusion(self):
        """Test integration of kernel fusion across lessons."""
        # Test fused operations with different data types
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        c = torch.randn(size, device=self.device, dtype=self.dtype)
        
        # Test fused add-multiply
        result_fused = fused_add_multiply(a, b, c)
        expected_fused = (a + b) * c
        self.assertTrue(torch.allclose(result_fused, expected_fused, rtol=1e-5, atol=1e-6))
        
        # Test fused loop
        result_loop = fused_loop(a, num_iterations=3)
        
        # PyTorch reference
        expected_loop = a
        for i in range(3):
            expected_loop = expected_loop * 2.0 + 1.0
            expected_loop = torch.relu(expected_loop)
            expected_loop = expected_loop * 0.5
        
        self.assertTrue(torch.allclose(result_loop, expected_loop, rtol=1e-4, atol=1e-4))
        
        # Test fused matrix multiplication + activation
        mat_a = torch.randn(64, 32, device=self.device, dtype=self.dtype)
        mat_b = torch.randn(32, 48, device=self.device, dtype=self.dtype)
        
        result_matmul_act = fused_matmul_activation(mat_a, mat_b, "relu")
        expected_matmul_act = torch.relu(torch.matmul(mat_a, mat_b))
        self.assertTrue(torch.allclose(result_matmul_act, expected_matmul_act, rtol=1e-3, atol=1e-3))

if __name__ == '__main__':
    # Run the tests
    unittest.main()
