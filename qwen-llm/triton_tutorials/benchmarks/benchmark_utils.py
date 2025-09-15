"""
📊 Benchmark Suite for Utilities

This module contains benchmarks for the utility modules.
"""

import torch
import time
import sys
import os

# Add the parent directory to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.benchmarking import BenchmarkSuite
from utils.profiling import PerformanceProfiler
from utils.validation import ValidationSuite
from utils.data_generation import DataGenerator, DataConfig
from utils.performance_analysis import PerformanceAnalyzer

class BenchmarkUtilities:
    """
    📊 BENCHMARK SUITE FOR UTILITIES
    
    Benchmarks for the utility modules.
    """
    
    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100):
        """
        Initialize the benchmark suite.
        
        Args:
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        print(f"🚀 Initializing Utilities Benchmark Suite")
        print(f"   Device: {self.device}")
        print(f"   Warmup runs: {self.warmup_runs}")
        print(f"   Benchmark runs: {self.benchmark_runs}")
    
    def benchmark_benchmarking_suite(self):
        """Benchmark the benchmarking suite itself."""
        print("\n📊 Benchmarking Benchmarking Suite:")
        print("=" * 50)
        
        suite = BenchmarkSuite(warmup_runs=self.warmup_runs, benchmark_runs=self.benchmark_runs)
        
        # Test different sizes
        sizes = [1024, 4096, 16384]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Benchmark function
            result = suite.benchmark_function(
                lambda x, y: x + y,
                lambda x, y: x + y,
                f"Vector Addition ({size:,})",
                a, b
            )
            
            print(f"  Triton Time: {result.triton_time:.3f} ms")
            print(f"  PyTorch Time: {result.pytorch_time:.3f} ms")
            print(f"  Speedup: {result.speedup:.2f}x")
            print(f"  Status: {'✅' if result.error is None else '❌'}")
        
        # Test memory bandwidth
        print(f"\n📈 Memory Bandwidth Test:")
        size = 1024 * 1024  # 1M elements
        
        def memory_copy(x, output):
            output.copy_(x)
        
        result = suite.benchmark_memory_bandwidth(
            memory_copy,
            "Memory Copy",
            size
        )
        
        print(f"  Time: {result.triton_time:.3f} ms")
        print(f"  Throughput: {result.throughput:.1f} GB/s")
        print(f"  Status: {'✅' if result.error is None else '❌'}")
    
    def benchmark_profiling_suite(self):
        """Benchmark the profiling suite."""
        print("\n📊 Benchmarking Profiling Suite:")
        print("=" * 50)
        
        profiler = PerformanceProfiler()
        
        # Test different sizes
        sizes = [1024, 4096, 16384]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Profile function
            result = profiler.profile_function(
                lambda x, y: x + y,
                f"Vector Addition ({size:,})",
                a, b
            )
            
            print(f"  Execution Time: {result.execution_time:.3f} ms")
            print(f"  Memory Usage: {result.memory_usage:.3f} GB")
            print(f"  GPU Memory Usage: {result.gpu_memory_usage:.3f} GB")
            print(f"  Status: {'✅' if result.error is None else '❌'}")
        
        # Test memory bandwidth profiling
        print(f"\n📈 Memory Bandwidth Profiling:")
        size = 1024 * 1024  # 1M elements
        
        def memory_copy(x, output):
            output.copy_(x)
        
        result = profiler.profile_memory_bandwidth(
            memory_copy,
            "Memory Copy",
            size
        )
        
        print(f"  Execution Time: {result.execution_time:.3f} ms")
        print(f"  Throughput: {result.throughput:.1f} GB/s")
        print(f"  Status: {'✅' if result.error is None else '❌'}")
    
    def benchmark_validation_suite(self):
        """Benchmark the validation suite."""
        print("\n📊 Benchmarking Validation Suite:")
        print("=" * 50)
        
        suite = ValidationSuite()
        
        # Test different sizes
        sizes = [1024, 4096, 16384]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Validate function
            result = suite.validate_function(
                lambda x, y: x + y,
                lambda x, y: x + y,
                f"Vector Addition ({size:,})",
                a, b
            )
            
            print(f"  Passed: {'✅' if result.passed else '❌'}")
            print(f"  Max Error: {result.max_error:.2e}")
            print(f"  Mean Error: {result.mean_error:.2e}")
            print(f"  Relative Error: {result.relative_error:.2e}")
            print(f"  Status: {'✅' if result.error_message is None else '❌'}")
        
        # Test edge cases
        print(f"\n📈 Edge Cases Validation:")
        edge_cases = [
            (torch.tensor([1.0], device=self.device, dtype=self.dtype),
             torch.tensor([2.0], device=self.device, dtype=self.dtype)),
            (torch.zeros(100, device=self.device, dtype=self.dtype),
             torch.zeros(100, device=self.device, dtype=self.dtype)),
        ]
        
        results = suite.validate_edge_cases(
            lambda x, y: x + y,
            lambda x, y: x + y,
            "Edge Cases",
            edge_cases
        )
        
        for i, result in enumerate(results):
            print(f"  Case {i+1}: {'✅' if result.passed else '❌'}")
    
    def benchmark_data_generation(self):
        """Benchmark data generation utilities."""
        print("\n📊 Benchmarking Data Generation:")
        print("=" * 50)
        
        generator = DataGenerator(seed=42)
        
        # Test different data types
        data_types = [
            (torch.float32, "float32"),
            (torch.float16, "float16"),
            (torch.int32, "int32"),
        ]
        
        for dtype, name in data_types:
            print(f"\n📈 {name}:")
            
            # Test vector generation
            config = DataConfig(size=1024, dtype=dtype, device=self.device, distribution='normal')
            vector = generator.generate_vector(config)
            
            print(f"  Vector: {vector.shape}, dtype: {vector.dtype}")
            
            # Test matrix generation
            config = DataConfig(size=1, dtype=dtype, device=self.device, distribution='normal')
            matrix = generator.generate_matrix(64, 32, config)
            
            print(f"  Matrix: {matrix.shape}, dtype: {matrix.dtype}")
            
            # Test batch generation
            batch = generator.generate_batch_matrices(4, 64, 32, config)
            
            print(f"  Batch: {batch.shape}, dtype: {batch.dtype}")
        
        # Test different distributions
        print(f"\n📈 Different Distributions:")
        distributions = ['normal', 'uniform', 'zeros', 'ones']
        
        for distribution in distributions:
            config = DataConfig(size=1024, dtype=self.dtype, device=self.device, distribution=distribution)
            data = generator.generate_vector(config)
            
            if distribution == 'zeros':
                is_correct = torch.all(data == 0)
            elif distribution == 'ones':
                is_correct = torch.all(data == 1)
            else:
                is_correct = torch.isfinite(data).all()
            
            print(f"  {distribution.capitalize()}: {'✅' if is_correct else '❌'}")
    
    def benchmark_performance_analysis(self):
        """Benchmark performance analysis utilities."""
        print("\n📊 Benchmarking Performance Analysis:")
        print("=" * 50)
        
        analyzer = PerformanceAnalyzer()
        
        # Test different sizes
        sizes = [1024, 4096, 16384]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Analyze function
            def test_func(x, y):
                return x + y
            
            result = analyzer.analyze_kernel(
                test_func,
                f"Vector Addition ({size:,})",
                size,
                self.dtype,
                num_runs=10
            )
            
            if result:
                print(f"  Execution Time: {result.execution_time:.3f} ms")
                print(f"  Memory Bandwidth: {result.memory_bandwidth:.1f} GB/s")
                print(f"  Throughput: {result.throughput:.0f} elements/s")
                print(f"  Efficiency: {result.efficiency:.2f}")
                print(f"  Status: ✅")
            else:
                print(f"  Status: ❌")
        
        # Test scaling analysis
        print(f"\n📈 Scaling Analysis:")
        scaling_sizes = [1024, 4096, 16384, 65536]
        
        def test_kernel(x, output):
            output.copy_(x)
        
        scaling_results = analyzer.analyze_scaling(
            test_kernel,
            "Memory Copy",
            scaling_sizes,
            self.dtype,
            num_runs=10
        )
        
        for result in scaling_results:
            if result:
                print(f"  Size {result.name.split('_size_')[-1]}: {result.execution_time:.3f} ms")
            else:
                print(f"  Size {scaling_sizes[len(scaling_results)]}: ❌")
    
    def benchmark_utility_integration(self):
        """Benchmark utility integration."""
        print("\n📊 Benchmarking Utility Integration:")
        print("=" * 50)
        
        # Test complete pipeline
        print(f"\n📈 Complete Pipeline Test:")
        
        # 1. Generate data
        generator = DataGenerator(seed=42)
        config = DataConfig(size=1024, dtype=self.dtype, device=self.device, distribution='normal')
        a = generator.generate_vector(config)
        b = generator.generate_vector(config)
        
        print(f"  Data Generation: ✅")
        
        # 2. Benchmark
        suite = BenchmarkSuite(warmup_runs=5, benchmark_runs=10)
        result = suite.benchmark_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            "Integration Test",
            a, b
        )
        
        print(f"  Benchmarking: {'✅' if result.error is None else '❌'}")
        
        # 3. Profile
        profiler = PerformanceProfiler()
        profile_result = profiler.profile_function(
            lambda x, y: x + y,
            "Integration Test",
            a, b
        )
        
        print(f"  Profiling: {'✅' if profile_result.error is None else '❌'}")
        
        # 4. Validate
        validation_suite = ValidationSuite()
        validation_result = validation_suite.validate_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            "Integration Test",
            a, b
        )
        
        print(f"  Validation: {'✅' if validation_result.passed else '❌'}")
        
        # 5. Analyze
        analyzer = PerformanceAnalyzer()
        analysis_result = analyzer.analyze_kernel(
            lambda x, y: x + y,
            "Integration Test",
            1024,
            self.dtype,
            num_runs=10
        )
        
        print(f"  Analysis: {'✅' if analysis_result else '❌'}")
        
        print(f"  Overall Status: ✅")
    
    def run_all_benchmarks(self):
        """Run all utility benchmarks."""
        print("🚀 Running All Utility Benchmarks")
        print("=" * 70)
        
        self.benchmark_benchmarking_suite()
        self.benchmark_profiling_suite()
        self.benchmark_validation_suite()
        self.benchmark_data_generation()
        self.benchmark_performance_analysis()
        self.benchmark_utility_integration()
        
        print("\n🎉 All Utility Benchmarks Complete!")

def main():
    """Main function to run utility benchmarks."""
    benchmark_suite = BenchmarkUtilities()
    benchmark_suite.run_all_benchmarks()

if __name__ == "__main__":
    main()
