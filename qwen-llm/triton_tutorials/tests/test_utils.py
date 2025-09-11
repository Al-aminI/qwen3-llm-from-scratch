"""
🧪 Test Suite for Utilities

This module contains tests for the utility modules.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.benchmarking import BenchmarkSuite, BenchmarkResult
from utils.profiling import PerformanceProfiler, ProfileResult
from utils.validation import ValidationSuite, ValidationResult
from utils.data_generation import DataGenerator, DataConfig
from utils.performance_analysis import PerformanceAnalyzer, PerformanceMetrics

class TestBenchmarking(unittest.TestCase):
    """
    🧪 TEST SUITE FOR BENCHMARKING UTILITIES
    
    Tests for the benchmarking module.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.suite = BenchmarkSuite(warmup_runs=5, benchmark_runs=10)
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        self.assertEqual(self.suite.warmup_runs, 5)
        self.assertEqual(self.suite.benchmark_runs, 10)
        self.assertEqual(len(self.suite.results), 0)
    
    def test_benchmark_function(self):
        """Test benchmarking a function."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        result = self.suite.benchmark_function(
            triton_func, pytorch_func,
            "Vector Addition",
            a, b
        )
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.name, "Vector Addition")
        self.assertGreater(result.triton_time, 0)
        self.assertGreater(result.pytorch_time, 0)
        self.assertGreater(result.speedup, 0)
        self.assertIsNone(result.error)
    
    def test_benchmark_memory_bandwidth(self):
        """Test memory bandwidth benchmarking."""
        size = 1024
        
        def memory_copy(x, output):
            output.copy_(x)
        
        result = self.suite.benchmark_memory_bandwidth(
            memory_copy,
            "Memory Copy",
            size
        )
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.name, "Memory Copy")
        self.assertGreater(result.triton_time, 0)
        self.assertGreater(result.throughput, 0)
        self.assertIsNone(result.error)
    
    def test_benchmark_matrix_sizes(self):
        """Test benchmarking multiple matrix sizes."""
        sizes = [(64, 32, 48), (128, 64, 96)]
        
        def triton_matmul(a, b):
            return torch.matmul(a, b)
        
        def pytorch_matmul(a, b):
            return torch.matmul(a, b)
        
        results = self.suite.benchmark_matrix_sizes(
            triton_matmul, pytorch_matmul,
            "Matrix Multiplication",
            sizes
        )
        
        self.assertEqual(len(results), len(sizes))
        for result in results:
            self.assertIsInstance(result, BenchmarkResult)
            self.assertGreater(result.speedup, 0)
    
    def test_benchmark_error_handling(self):
        """Test error handling in benchmarking."""
        def error_func(x, y):
            raise ValueError("Test error")
        
        def normal_func(x, y):
            return x + y
        
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = self.suite.benchmark_function(
            error_func, normal_func,
            "Error Test",
            a, b
        )
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.name, "Error Test")
        self.assertIsNotNone(result.error)
        self.assertIn("Test error", result.error)
    
    def test_benchmark_results_management(self):
        """Test benchmark results management."""
        # Add some results
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        self.suite.benchmark_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            "Test 1",
            a, b
        )
        
        self.suite.benchmark_function(
            lambda x, y: x * y,
            lambda x, y: x * y,
            "Test 2",
            a, b
        )
        
        self.assertEqual(len(self.suite.results), 2)
        
        # Clear results
        self.suite.clear_results()
        self.assertEqual(len(self.suite.results), 0)

class TestProfiling(unittest.TestCase):
    """
    🧪 TEST SUITE FOR PROFILING UTILITIES
    
    Tests for the profiling module.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.profiler = PerformanceProfiler()
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertEqual(len(self.profiler.results), 0)
        self.assertEqual(self.profiler.gpu_available, torch.cuda.is_available())
    
    def test_profile_function(self):
        """Test profiling a function."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def test_func(x, y):
            return x + y
        
        result = self.profiler.profile_function(
            test_func,
            "Vector Addition",
            a, b
        )
        
        self.assertIsInstance(result, ProfileResult)
        self.assertEqual(result.name, "Vector Addition")
        self.assertGreater(result.execution_time, 0)
        self.assertIsNone(result.error)
    
    def test_profile_memory_bandwidth(self):
        """Test memory bandwidth profiling."""
        size = 1024
        
        def memory_copy(x, output):
            output.copy_(x)
        
        result = self.profiler.profile_memory_bandwidth(
            memory_copy,
            "Memory Copy",
            size
        )
        
        self.assertIsInstance(result, ProfileResult)
        self.assertEqual(result.name, "Memory Copy")
        self.assertGreater(result.execution_time, 0)
        self.assertGreater(result.throughput, 0)
        self.assertIsNone(result.error)
    
    def test_profile_kernel_launch_overhead(self):
        """Test kernel launch overhead profiling."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def test_kernel(x, y):
            return x + y
        
        result = self.profiler.profile_kernel_launch_overhead(
            test_kernel,
            "Vector Addition",
            num_launches=100,
            a, b
        )
        
        self.assertIsInstance(result, ProfileResult)
        self.assertEqual(result.name, "Vector Addition_launch_overhead")
        self.assertGreater(result.execution_time, 0)
        self.assertGreater(result.throughput, 0)
        self.assertIsNone(result.error)
    
    def test_profile_error_handling(self):
        """Test error handling in profiling."""
        def error_func(x, y):
            raise ValueError("Test error")
        
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = self.profiler.profile_function(
            error_func,
            "Error Test",
            a, b
        )
        
        self.assertIsInstance(result, ProfileResult)
        self.assertEqual(result.name, "Error Test")
        self.assertIsNotNone(result.error)
        self.assertIn("Test error", result.error)

class TestValidation(unittest.TestCase):
    """
    🧪 TEST SUITE FOR VALIDATION UTILITIES
    
    Tests for the validation module.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.suite = ValidationSuite(rtol=1e-5, atol=1e-6)
    
    def test_validation_suite_initialization(self):
        """Test validation suite initialization."""
        self.assertEqual(self.suite.rtol, 1e-5)
        self.assertEqual(self.suite.atol, 1e-6)
        self.assertEqual(len(self.suite.results), 0)
    
    def test_validate_function(self):
        """Test function validation."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        result = self.suite.validate_function(
            triton_func, pytorch_func,
            "Vector Addition",
            a, b
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.name, "Vector Addition")
        self.assertTrue(result.passed)
        self.assertEqual(result.max_error, 0.0)
        self.assertEqual(result.mean_error, 0.0)
        self.assertEqual(result.relative_error, 0.0)
        self.assertIsNone(result.error_message)
    
    def test_validate_shapes(self):
        """Test shape validation."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        result = self.suite.validate_shapes(
            triton_func, pytorch_func,
            "Shape Validation",
            a, b
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.name, "Shape Validation")
        self.assertTrue(result.passed)
        self.assertIsNone(result.error_message)
    
    def test_validate_dtypes(self):
        """Test dtype validation."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        result = self.suite.validate_dtypes(
            triton_func, pytorch_func,
            "Dtype Validation",
            a, b
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.name, "Dtype Validation")
        self.assertTrue(result.passed)
        self.assertIsNone(result.error_message)
    
    def test_validate_edge_cases(self):
        """Test edge case validation."""
        test_cases = [
            (torch.tensor([1.0], device=self.device, dtype=self.dtype),
             torch.tensor([2.0], device=self.device, dtype=self.dtype)),
            (torch.zeros(100, device=self.device, dtype=self.dtype),
             torch.zeros(100, device=self.device, dtype=self.dtype)),
        ]
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        results = self.suite.validate_edge_cases(
            triton_func, pytorch_func,
            "Edge Cases",
            test_cases
        )
        
        self.assertEqual(len(results), len(test_cases))
        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertTrue(result.passed)
    
    def test_validate_numerical_stability(self):
        """Test numerical stability validation."""
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        result = self.suite.validate_numerical_stability(
            triton_func, pytorch_func,
            "Numerical Stability",
            a, b
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.name, "Numerical Stability")
        self.assertTrue(result.passed)
        self.assertIsNone(result.error_message)
    
    def test_validation_error_handling(self):
        """Test error handling in validation."""
        def error_func(x, y):
            raise ValueError("Test error")
        
        def normal_func(x, y):
            return x + y
        
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        result = self.suite.validate_function(
            error_func, normal_func,
            "Error Test",
            a, b
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.name, "Error Test")
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Test error", result.error_message)
    
    def test_validation_summary(self):
        """Test validation summary."""
        # Add some results
        size = 1024
        a = torch.randn(size, device=self.device, dtype=self.dtype)
        b = torch.randn(size, device=self.device, dtype=self.dtype)
        
        self.suite.validate_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            "Test 1",
            a, b
        )
        
        self.suite.validate_function(
            lambda x, y: x * y,
            lambda x, y: x * y,
            "Test 2",
            a, b
        )
        
        summary = self.suite.get_summary()
        self.assertEqual(summary['total'], 2)
        self.assertEqual(summary['passed'], 2)
        self.assertEqual(summary['failed'], 0)
        self.assertEqual(summary['pass_rate'], 1.0)

class TestDataGeneration(unittest.TestCase):
    """
    🧪 TEST SUITE FOR DATA GENERATION UTILITIES
    
    Tests for the data generation module.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.generator = DataGenerator(seed=42)
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        self.assertEqual(self.generator.seed, 42)
    
    def test_generate_vector(self):
        """Test vector generation."""
        config = DataConfig(size=1024, dtype=self.dtype, device=self.device, distribution='normal')
        vector = self.generator.generate_vector(config)
        
        self.assertEqual(vector.shape, (1024,))
        self.assertEqual(vector.dtype, self.dtype)
        self.assertEqual(vector.device.type, self.device)
    
    def test_generate_matrix(self):
        """Test matrix generation."""
        config = DataConfig(size=1, dtype=self.dtype, device=self.device, distribution='normal')
        matrix = self.generator.generate_matrix(64, 32, config)
        
        self.assertEqual(matrix.shape, (64, 32))
        self.assertEqual(matrix.dtype, self.dtype)
        self.assertEqual(matrix.device.type, self.device)
    
    def test_generate_batch_matrices(self):
        """Test batch matrix generation."""
        config = DataConfig(size=1, dtype=self.dtype, device=self.device, distribution='normal')
        batch = self.generator.generate_batch_matrices(4, 64, 32, config)
        
        self.assertEqual(batch.shape, (4, 64, 32))
        self.assertEqual(batch.dtype, self.dtype)
        self.assertEqual(batch.device.type, self.device)
    
    def test_generate_attention_data(self):
        """Test attention data generation."""
        config = DataConfig(size=1, dtype=self.dtype, device=self.device, distribution='normal')
        q, k, v = self.generator.generate_attention_data(2, 8, 128, 64, config)
        
        self.assertEqual(q.shape, (2, 8, 128, 64))
        self.assertEqual(k.shape, (2, 8, 128, 64))
        self.assertEqual(v.shape, (2, 8, 128, 64))
        self.assertEqual(q.dtype, self.dtype)
        self.assertEqual(q.device.type, self.device)
    
    def test_generate_sparse_data(self):
        """Test sparse data generation."""
        config = DataConfig(size=1024, dtype=self.dtype, device=self.device, distribution='normal')
        sparse_data = self.generator.generate_sparse_data(1024, 0.5, config)
        
        self.assertEqual(sparse_data.shape, (1024,))
        self.assertEqual(sparse_data.dtype, self.dtype)
        self.assertEqual(sparse_data.device.type, self.device)
        
        # Check sparsity
        sparsity = (sparse_data == 0).float().mean().item()
        self.assertAlmostEqual(sparsity, 0.5, delta=0.1)
    
    def test_generate_sequence_data(self):
        """Test sequence data generation."""
        config = DataConfig(size=1, dtype=torch.int32, device=self.device, distribution='normal')
        sequence = self.generator.generate_sequence_data(2, 128, 1000, config)
        
        self.assertEqual(sequence.shape, (2, 128))
        self.assertEqual(sequence.dtype, torch.int32)
        self.assertEqual(sequence.device.type, self.device)
        self.assertTrue(torch.all(sequence >= 0))
        self.assertTrue(torch.all(sequence < 1000))
    
    def test_generate_benchmark_data(self):
        """Test benchmark data generation."""
        sizes = [1024, 4096, 16384]
        data_list = self.generator.generate_benchmark_data(sizes, self.dtype)
        
        self.assertEqual(len(data_list), len(sizes))
        for i, data in enumerate(data_list):
            self.assertEqual(data.shape, (sizes[i],))
            self.assertEqual(data.dtype, self.dtype)
            self.assertEqual(data.device.type, self.device)
    
    def test_generate_matrix_benchmark_data(self):
        """Test matrix benchmark data generation."""
        sizes = [(64, 32, 48), (128, 64, 96)]
        data_list = self.generator.generate_matrix_benchmark_data(sizes, self.dtype)
        
        self.assertEqual(len(data_list), len(sizes))
        for i, (a, b) in enumerate(data_list):
            M, K, N = sizes[i]
            self.assertEqual(a.shape, (M, K))
            self.assertEqual(b.shape, (K, N))
            self.assertEqual(a.dtype, self.dtype)
            self.assertEqual(b.dtype, self.dtype)
    
    def test_data_distributions(self):
        """Test different data distributions."""
        size = 1024
        
        # Test normal distribution
        config_normal = DataConfig(size=size, dtype=self.dtype, device=self.device, distribution='normal')
        normal_data = self.generator.generate_vector(config_normal)
        self.assertEqual(normal_data.shape, (size,))
        
        # Test uniform distribution
        config_uniform = DataConfig(size=size, dtype=self.dtype, device=self.device, distribution='uniform')
        uniform_data = self.generator.generate_vector(config_uniform)
        self.assertEqual(uniform_data.shape, (size,))
        
        # Test zeros
        config_zeros = DataConfig(size=size, dtype=self.dtype, device=self.device, distribution='zeros')
        zeros_data = self.generator.generate_vector(config_zeros)
        self.assertEqual(zeros_data.shape, (size,))
        self.assertTrue(torch.all(zeros_data == 0))
        
        # Test ones
        config_ones = DataConfig(size=size, dtype=self.dtype, device=self.device, distribution='ones')
        ones_data = self.generator.generate_vector(config_ones)
        self.assertEqual(ones_data.shape, (size,))
        self.assertTrue(torch.all(ones_data == 1))

class TestPerformanceAnalysis(unittest.TestCase):
    """
    🧪 TEST SUITE FOR PERFORMANCE ANALYSIS UTILITIES
    
    Tests for the performance analysis module.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.analyzer = PerformanceAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(len(self.analyzer.metrics), 0)
        self.assertEqual(self.analyzer.gpu_available, torch.cuda.is_available())
    
    def test_analyze_kernel(self):
        """Test kernel analysis."""
        size = 1024
        
        def test_kernel(x, output):
            output.copy_(x)
        
        metrics = self.analyzer.analyze_kernel(
            test_kernel,
            "Memory Copy",
            size,
            self.dtype,
            num_runs=10
        )
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.name, "Memory Copy")
        self.assertGreater(metrics.execution_time, 0)
        self.assertGreater(metrics.memory_bandwidth, 0)
        self.assertGreater(metrics.throughput, 0)
        self.assertGreaterEqual(metrics.efficiency, 0)
        self.assertLessEqual(metrics.efficiency, 1)
    
    def test_compare_kernels(self):
        """Test kernel comparison."""
        size = 1024
        
        def triton_func(x, y):
            return x + y
        
        def pytorch_func(x, y):
            return x + y
        
        triton_metrics, pytorch_metrics = self.analyzer.compare_kernels(
            triton_func, pytorch_func,
            "Vector Addition",
            size,
            self.dtype,
            num_runs=10
        )
        
        self.assertIsInstance(triton_metrics, PerformanceMetrics)
        self.assertIsInstance(pytorch_metrics, PerformanceMetrics)
        self.assertEqual(triton_metrics.name, "Vector Addition_triton")
        self.assertEqual(pytorch_metrics.name, "Vector Addition_pytorch")
    
    def test_analyze_scaling(self):
        """Test scaling analysis."""
        sizes = [1024, 4096, 16384]
        
        def test_kernel(x, output):
            output.copy_(x)
        
        scaling_metrics = self.analyzer.analyze_scaling(
            test_kernel,
            "Memory Copy",
            sizes,
            self.dtype,
            num_runs=10
        )
        
        self.assertEqual(len(scaling_metrics), len(sizes))
        for metrics in scaling_metrics:
            self.assertIsInstance(metrics, PerformanceMetrics)
            self.assertGreater(metrics.execution_time, 0)
    
    def test_analyzer_error_handling(self):
        """Test error handling in analyzer."""
        def error_kernel(x, output):
            raise ValueError("Test error")
        
        size = 1024
        metrics = self.analyzer.analyze_kernel(
            error_kernel,
            "Error Test",
            size,
            self.dtype,
            num_runs=10
        )
        
        self.assertIsNone(metrics)
    
    def test_analyzer_summary(self):
        """Test analyzer summary."""
        # Add some metrics
        size = 1024
        
        def test_kernel(x, output):
            output.copy_(x)
        
        self.analyzer.analyze_kernel(
            test_kernel,
            "Test 1",
            size,
            self.dtype,
            num_runs=10
        )
        
        self.analyzer.analyze_kernel(
            test_kernel,
            "Test 2",
            size,
            self.dtype,
            num_runs=10
        )
        
        self.assertEqual(len(self.analyzer.metrics), 2)
        
        # Clear metrics
        self.analyzer.clear_metrics()
        self.assertEqual(len(self.analyzer.metrics), 0)

if __name__ == '__main__':
    # Run the tests
    unittest.main()
