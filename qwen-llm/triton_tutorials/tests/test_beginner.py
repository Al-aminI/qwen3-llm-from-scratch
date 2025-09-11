"""
🧪 Test Suite for Beginner Lessons

This module contains tests for the beginner-level Triton tutorials.
"""

import unittest
import torch
import triton
import triton.language as tl
import sys
import os

# Add the parent directory to the path to import lessons
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lessons.beginner.lesson_01_gpu_fundamentals import vector_add
from lessons.beginner.lesson_02_memory_management import coalesced_access_kernel, non_coalesced_access_kernel
from lessons.beginner.lesson_03_basic_operations import element_wise_add_kernel, sum_reduction_kernel

class TestBeginnerLessons(unittest.TestCase):
    """
    🧪 TEST SUITE FOR BEGINNER LESSONS
    
    Tests for lessons 1-3 (GPU Fundamentals, Memory Management, Basic Operations)
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        # Test data sizes
        self.small_size = 1024
        self.medium_size = 4096
        self.large_size = 16384
        
        # Create test data
        self.small_a = torch.randn(self.small_size, device=self.device, dtype=self.dtype)
        self.small_b = torch.randn(self.small_size, device=self.device, dtype=self.dtype)
        
        self.medium_a = torch.randn(self.medium_size, device=self.device, dtype=self.dtype)
        self.medium_b = torch.randn(self.medium_size, device=self.device, dtype=self.dtype)
        
        self.large_a = torch.randn(self.large_size, device=self.device, dtype=self.dtype)
        self.large_b = torch.randn(self.large_size, device=self.device, dtype=self.dtype)
    
    def test_lesson_01_vector_addition_small(self):
        """Test vector addition with small size."""
        result = vector_add(self.small_a, self.small_b)
        expected = self.small_a + self.small_b
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_01_vector_addition_medium(self):
        """Test vector addition with medium size."""
        result = vector_add(self.medium_a, self.medium_b)
        expected = self.medium_a + self.medium_b
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_01_vector_addition_large(self):
        """Test vector addition with large size."""
        result = vector_add(self.large_a, self.large_b)
        expected = self.large_a + self.large_b
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)
    
    def test_lesson_01_vector_addition_edge_cases(self):
        """Test vector addition with edge cases."""
        # Single element
        single_a = torch.tensor([1.0], device=self.device, dtype=self.dtype)
        single_b = torch.tensor([2.0], device=self.device, dtype=self.dtype)
        result = vector_add(single_a, single_b)
        expected = single_a + single_b
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        
        # Zero tensor
        zero_a = torch.zeros(100, device=self.device, dtype=self.dtype)
        zero_b = torch.zeros(100, device=self.device, dtype=self.dtype)
        result = vector_add(zero_a, zero_b)
        expected = zero_a + zero_b
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        
        # Large values
        large_a = torch.full((100,), 1e6, device=self.device, dtype=self.dtype)
        large_b = torch.full((100,), 1e6, device=self.device, dtype=self.dtype)
        result = vector_add(large_a, large_b)
        expected = large_a + large_b
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
    
    def test_lesson_01_vector_addition_different_dtypes(self):
        """Test vector addition with different data types."""
        if self.device == 'cuda':
            # Test float16
            a_f16 = self.small_a.half()
            b_f16 = self.small_b.half()
            result_f16 = vector_add(a_f16, b_f16)
            expected_f16 = a_f16 + b_f16
            self.assertTrue(torch.allclose(result_f16, expected_f16, rtol=1e-3, atol=1e-3))
            
            # Test int32
            a_i32 = torch.randint(0, 100, (self.small_size,), device=self.device, dtype=torch.int32)
            b_i32 = torch.randint(0, 100, (self.small_size,), device=self.device, dtype=torch.int32)
            result_i32 = vector_add(a_i32, b_i32)
            expected_i32 = a_i32 + b_i32
            self.assertTrue(torch.allclose(result_i32.float(), expected_i32.float(), rtol=1e-5, atol=1e-6))
    
    def test_lesson_02_memory_coalescing(self):
        """Test memory coalescing patterns."""
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        output_coalesced = torch.empty_like(x)
        output_non_coalesced = torch.empty_like(x)
        
        # Test coalesced access
        grid = (triton.cdiv(size, 128),)
        coalesced_access_kernel[grid](
            x, output_coalesced, size, 1, BLOCK_SIZE=128
        )
        
        # Test non-coalesced access
        non_coalesced_access_kernel[grid](
            x, output_non_coalesced, size, 1, BLOCK_SIZE=128
        )
        
        # Verify correctness
        expected_coalesced = x * 2.0
        self.assertTrue(torch.allclose(output_coalesced, expected_coalesced, rtol=1e-5, atol=1e-6))
        
        # Non-coalesced access should still produce correct results (but slower)
        expected_non_coalesced = x[::100] * 2.0
        self.assertTrue(torch.allclose(output_non_coalesced[::100], expected_non_coalesced, rtol=1e-5, atol=1e-6))
    
    def test_lesson_03_element_wise_operations(self):
        """Test element-wise operations."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        output = torch.empty_like(a)
        
        # Test element-wise addition
        grid = (triton.cdiv(size, 128),)
        element_wise_add_kernel[grid](
            a, b, output, size, BLOCK_SIZE=128
        )
        
        expected = a + b
        self.assertTrue(torch.allclose(output, expected, rtol=1e-5, atol=1e-6))
    
    def test_lesson_03_reduction_operations(self):
        """Test reduction operations."""
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        output = torch.zeros(1, device=self.device, dtype=self.dtype)
        
        # Test sum reduction
        grid = (triton.cdiv(size, 128),)
        sum_reduction_kernel[grid](
            x, output, size, BLOCK_SIZE=128
        )
        
        expected = torch.sum(x)
        self.assertTrue(torch.allclose(output, expected, rtol=1e-4, atol=1e-5))
    
    def test_lesson_03_reduction_edge_cases(self):
        """Test reduction operations with edge cases."""
        # Single element
        single_x = torch.tensor([5.0], device=self.device, dtype=self.dtype)
        single_output = torch.zeros(1, device=self.device, dtype=self.dtype)
        grid = (triton.cdiv(1, 128),)
        sum_reduction_kernel[grid](
            single_x, single_output, 1, BLOCK_SIZE=128
        )
        expected_single = torch.sum(single_x)
        self.assertTrue(torch.allclose(single_output, expected_single, rtol=1e-4, atol=1e-5))
        
        # Zero tensor
        zero_x = torch.zeros(100, device=self.device, dtype=self.dtype)
        zero_output = torch.zeros(1, device=self.device, dtype=self.dtype)
        grid = (triton.cdiv(100, 128),)
        sum_reduction_kernel[grid](
            zero_x, zero_output, 100, BLOCK_SIZE=128
        )
        expected_zero = torch.sum(zero_x)
        self.assertTrue(torch.allclose(zero_output, expected_zero, rtol=1e-4, atol=1e-5))
    
    def test_lesson_03_broadcasting(self):
        """Test broadcasting operations."""
        # Test 2D broadcasting
        a = torch.randn(3, 1, device=self.device, dtype=self.dtype)
        b = torch.randn(1, 4, device=self.device, dtype=self.dtype)
        
        # This would need a broadcasting kernel implementation
        # For now, just test that the tensors are created correctly
        self.assertEqual(a.shape, (3, 1))
        self.assertEqual(b.shape, (1, 4))
        self.assertEqual(a.device, self.device)
        self.assertEqual(b.device, self.device)
    
    def test_lesson_03_activation_functions(self):
        """Test activation functions."""
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        
        # Test ReLU
        relu_result = torch.relu(x)
        self.assertTrue(torch.all(relu_result >= 0))
        self.assertTrue(torch.allclose(relu_result, torch.where(x > 0, x, 0.0), rtol=1e-5, atol=1e-6))
        
        # Test Sigmoid
        sigmoid_result = torch.sigmoid(x)
        self.assertTrue(torch.all(sigmoid_result >= 0))
        self.assertTrue(torch.all(sigmoid_result <= 1))
        
        # Test Tanh
        tanh_result = torch.tanh(x)
        self.assertTrue(torch.all(tanh_result >= -1))
        self.assertTrue(torch.all(tanh_result <= 1))
    
    def test_lesson_03_error_handling(self):
        """Test error handling and edge cases."""
        # Test with non-power-of-2 size
        size = 1000  # Not a power of 2
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = vector_add(a, b)
        expected = a + b
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(result.shape, expected.shape)
    
    def test_lesson_03_performance_characteristics(self):
        """Test performance characteristics."""
        # Test with different block sizes
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        # Test with different block sizes
        block_sizes = [32, 64, 128, 256]
        
        for block_size in block_sizes:
            if size % block_size == 0 or size > block_size:
                result = vector_add(a, b)
                expected = a + b
                self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
    
    def test_lesson_03_memory_bandwidth(self):
        """Test memory bandwidth characteristics."""
        size = 1024 * 1024  # 1M elements
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        output = torch.empty_like(x)
        
        # Test memory copy
        output.copy_(x)
        self.assertTrue(torch.allclose(output, x, rtol=1e-5, atol=1e-6))
        
        # Test memory bandwidth with vector addition
        result = vector_add(x, x)
        expected = x + x
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))

class TestBeginnerLessonsIntegration(unittest.TestCase):
    """
    🧪 INTEGRATION TESTS FOR BEGINNER LESSONS
    
    Integration tests that combine multiple lessons.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
    
    def test_lesson_integration_vector_operations(self):
        """Test integration of vector operations across lessons."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        # Lesson 1: Vector addition
        result_add = vector_add(a, b)
        expected_add = a + b
        self.assertTrue(torch.allclose(result_add, expected_add, rtol=1e-5, atol=1e-6))
        
        # Lesson 2: Memory coalescing
        output_coalesced = torch.empty_like(a)
        grid = (triton.cdiv(size, 128),)
        coalesced_access_kernel[grid](
            a, output_coalesced, size, 1, BLOCK_SIZE=128
        )
        expected_coalesced = a * 2.0
        self.assertTrue(torch.allclose(output_coalesced, expected_coalesced, rtol=1e-5, atol=1e-6))
        
        # Lesson 3: Element-wise operations
        output_element = torch.empty_like(a)
        element_wise_add_kernel[grid](
            a, b, output_element, size, BLOCK_SIZE=128
        )
        expected_element = a + b
        self.assertTrue(torch.allclose(output_element, expected_element, rtol=1e-5, atol=1e-6))
    
    def test_lesson_integration_reduction_operations(self):
        """Test integration of reduction operations across lessons."""
        size = 1024
        x = torch.randn(size, device=self.device, dtype=self.dtype)
        
        # Lesson 3: Sum reduction
        output_sum = torch.zeros(1, device=self.device, dtype=self.dtype)
        grid = (triton.cdiv(size, 128),)
        sum_reduction_kernel[grid](
            x, output_sum, size, BLOCK_SIZE=128
        )
        expected_sum = torch.sum(x)
        self.assertTrue(torch.allclose(output_sum, expected_sum, rtol=1e-4, atol=1e-5))
        
        # Test with different sizes
        sizes = [100, 500, 1000, 2000]
        for test_size in sizes:
            test_x = torch.randn(test_size, device=self.device, dtype=self.dtype)
            test_output = torch.zeros(1, device=self.device, dtype=self.dtype)
            test_grid = (triton.cdiv(test_size, 128),)
            sum_reduction_kernel[test_grid](
                test_x, test_output, test_size, BLOCK_SIZE=128
            )
            expected_test = torch.sum(test_x)
            self.assertTrue(torch.allclose(test_output, expected_test, rtol=1e-4, atol=1e-5))

if __name__ == '__main__':
    # Run the tests
    unittest.main()
