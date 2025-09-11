"""
📊 Benchmarking Utilities

This module provides utilities for benchmarking Triton kernels and comparing
performance with PyTorch implementations.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    triton_time: float
    pytorch_time: float
    speedup: float
    memory_usage: Optional[float] = None
    throughput: Optional[float] = None
    error: Optional[str] = None

class BenchmarkSuite:
    """
    📊 BENCHMARK SUITE
    
    A comprehensive benchmarking suite for Triton kernels.
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
        self.results = []
    
    def benchmark_function(self, 
                          triton_func: Callable,
                          pytorch_func: Callable,
                          name: str,
                          *args, **kwargs) -> BenchmarkResult:
        """
        Benchmark a Triton function against a PyTorch function.
        
        Args:
            triton_func: Triton implementation
            pytorch_func: PyTorch implementation
            name: Name of the benchmark
            *args: Arguments to pass to both functions
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            BenchmarkResult object
        """
        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                _ = triton_func(*args, **kwargs)
                _ = pytorch_func(*args, **kwargs)
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                triton_result = triton_func(*args, **kwargs)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000  # Convert to ms
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                pytorch_result = pytorch_func(*args, **kwargs)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000  # Convert to ms
            
            # Calculate speedup
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            result = BenchmarkResult(
                name=name,
                triton_time=triton_time,
                pytorch_time=pytorch_time,
                speedup=speedup
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = BenchmarkResult(
                name=name,
                triton_time=0.0,
                pytorch_time=0.0,
                speedup=0.0,
                error=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def benchmark_memory_bandwidth(self,
                                  func: Callable,
                                  name: str,
                                  input_size: int,
                                  dtype: torch.dtype = torch.float32) -> BenchmarkResult:
        """
        Benchmark memory bandwidth utilization.
        
        Args:
            func: Function to benchmark
            name: Name of the benchmark
            input_size: Size of input tensor
            dtype: Data type of input tensor
            
        Returns:
            BenchmarkResult with bandwidth information
        """
        try:
            # Create test data
            x = torch.randn(input_size, device='cuda', dtype=dtype)
            output = torch.empty_like(x)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = func(x, output)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = func(x, output)
            torch.cuda.synchronize()
            elapsed_time = (time.time() - start_time) / self.benchmark_runs
            
            # Calculate bandwidth
            bytes_transferred = input_size * dtype.itemsize * 2  # read + write
            bandwidth_gb_s = (bytes_transferred / elapsed_time) / (1024**3)
            
            result = BenchmarkResult(
                name=name,
                triton_time=elapsed_time * 1000,  # Convert to ms
                pytorch_time=0.0,
                speedup=0.0,
                throughput=bandwidth_gb_s
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = BenchmarkResult(
                name=name,
                triton_time=0.0,
                pytorch_time=0.0,
                speedup=0.0,
                error=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def benchmark_matrix_sizes(self,
                              triton_func: Callable,
                              pytorch_func: Callable,
                              name: str,
                              sizes: List[tuple]) -> List[BenchmarkResult]:
        """
        Benchmark multiple matrix sizes.
        
        Args:
            triton_func: Triton implementation
            pytorch_func: PyTorch implementation
            name: Base name for benchmarks
            sizes: List of (M, K, N) tuples
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for i, (M, K, N) in enumerate(sizes):
            # Create test data
            a = torch.randn(M, K, device='cuda', dtype=torch.float32)
            b = torch.randn(K, N, device='cuda', dtype=torch.float32)
            
            result = self.benchmark_function(
                triton_func, pytorch_func,
                f"{name}_({M}x{K}x{N})",
                a, b
            )
            results.append(result)
        
        return results
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n📊 Benchmark Results:")
        print("=" * 80)
        print(f"{'Name':<30} {'Triton (ms)':<12} {'PyTorch (ms)':<12} {'Speedup':<10} {'Error':<20}")
        print("-" * 80)
        
        for result in self.results:
            if result.error:
                print(f"{result.name:<30} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {result.error:<20}")
            else:
                print(f"{result.name:<30} {result.triton_time:<12.3f} {result.pytorch_time:<12.3f} {result.speedup:<10.2f}x {'':<20}")
    
    def save_results(self, filename: str):
        """Save benchmark results to a JSON file."""
        results_dict = []
        for result in self.results:
            results_dict.append({
                'name': result.name,
                'triton_time': result.triton_time,
                'pytorch_time': result.pytorch_time,
                'speedup': result.speedup,
                'memory_usage': result.memory_usage,
                'throughput': result.throughput,
                'error': result.error
            })
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")
    
    def clear_results(self):
        """Clear all benchmark results."""
        self.results = []

def benchmark_vector_operations():
    """Benchmark vector operations."""
    print("📊 Benchmarking Vector Operations:")
    print("=" * 50)
    
    suite = BenchmarkSuite()
    
    # Test different sizes
    sizes = [1024, 4096, 16384, 65536, 262144]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        # Create test data
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Benchmark addition
        suite.benchmark_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            f"Vector Addition ({size:,})",
            a, b
        )
        
        # Benchmark multiplication
        suite.benchmark_function(
            lambda x, y: x * y,
            lambda x, y: x * y,
            f"Vector Multiplication ({size:,})",
            a, b
        )
    
    suite.print_results()
    return suite

def benchmark_matrix_operations():
    """Benchmark matrix operations."""
    print("\n📊 Benchmarking Matrix Operations:")
    print("=" * 50)
    
    suite = BenchmarkSuite()
    
    # Test different matrix sizes
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    for M, K, N in sizes:
        print(f"\n📈 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
        
        # Create test data
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Benchmark matrix multiplication
        suite.benchmark_function(
            torch.matmul,
            torch.matmul,
            f"Matrix Multiplication ({M}x{K}x{N})",
            a, b
        )
    
    suite.print_results()
    return suite

def benchmark_memory_bandwidth():
    """Benchmark memory bandwidth."""
    print("\n📊 Benchmarking Memory Bandwidth:")
    print("=" * 50)
    
    suite = BenchmarkSuite()
    
    # Test different sizes
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        # Benchmark memory copy
        def memory_copy(x, output):
            output.copy_(x)
        
        suite.benchmark_memory_bandwidth(
            memory_copy,
            f"Memory Copy ({size:,})",
            size
        )
    
    suite.print_results()
    return suite

if __name__ == "__main__":
    # Run all benchmarks
    vector_suite = benchmark_vector_operations()
    matrix_suite = benchmark_matrix_operations()
    memory_suite = benchmark_memory_bandwidth()
    
    # Save results
    vector_suite.save_results("vector_benchmarks.json")
    matrix_suite.save_results("matrix_benchmarks.json")
    memory_suite.save_results("memory_benchmarks.json")
