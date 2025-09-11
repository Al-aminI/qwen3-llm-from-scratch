"""
📊 Benchmark Suite for Beginner Lessons

This module contains benchmarks for the beginner-level Triton tutorials.
"""

import torch
import triton
import triton.language as tl
import time
import sys
import os

# Add the parent directory to the path to import lessons
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lessons.beginner.lesson_01_gpu_fundamentals import vector_add
from lessons.beginner.lesson_02_memory_management import coalesced_access_kernel, non_coalesced_access_kernel
from lessons.beginner.lesson_03_basic_operations import element_wise_add_kernel, sum_reduction_kernel

class BenchmarkBeginnerLessons:
    """
    📊 BENCHMARK SUITE FOR BEGINNER LESSONS
    
    Benchmarks for lessons 1-3 (GPU Fundamentals, Memory Management, Basic Operations)
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
        
        print(f"🚀 Initializing Beginner Lessons Benchmark Suite")
        print(f"   Device: {self.device}")
        print(f"   Warmup runs: {self.warmup_runs}")
        print(f"   Benchmark runs: {self.benchmark_runs}")
    
    def benchmark_vector_addition(self):
        """Benchmark vector addition operations."""
        print("\n📊 Benchmarking Vector Addition:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536, 262144]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = vector_add(a, b)
                _ = a + b
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                triton_result = vector_add(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000  # Convert to ms
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                pytorch_result = a + b
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000  # Convert to ms
            
            # Calculate speedup
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_memory_access_patterns(self):
        """Benchmark memory access patterns."""
        print("\n📊 Benchmarking Memory Access Patterns:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            output_coalesced = torch.empty_like(x)
            output_non_coalesced = torch.empty_like(x)
            
            # Warmup
            for _ in range(self.warmup_runs):
                coalesced_access_kernel[(triton.cdiv(size, 128),)](
                    x, output_coalesced, size, 1, BLOCK_SIZE=128
                )
                non_coalesced_access_kernel[(triton.cdiv(size, 128),)](
                    x, output_non_coalesced, size, 1, BLOCK_SIZE=128
                )
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark coalesced access
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                coalesced_access_kernel[(triton.cdiv(size, 128),)](
                    x, output_coalesced, size, 1, BLOCK_SIZE=128
                )
            if self.device == 'cuda':
                torch.cuda.synchronize()
            coalesced_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark non-coalesced access
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                non_coalesced_access_kernel[(triton.cdiv(size, 128),)](
                    x, output_non_coalesced, size, 1, BLOCK_SIZE=128
                )
            if self.device == 'cuda':
                torch.cuda.synchronize()
            non_coalesced_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = non_coalesced_time / coalesced_time if coalesced_time > 0 else 0
            
            print(f"  Coalesced:    {coalesced_time:.3f} ms")
            print(f"  Non-Coalesced: {non_coalesced_time:.3f} ms")
            print(f"  Speedup:      {speedup:.2f}x")
            
            # Verify correctness
            expected_coalesced = x * 2.0
            is_correct_coalesced = torch.allclose(output_coalesced, expected_coalesced, rtol=1e-5)
            print(f"  Coalesced Correct: {'✅' if is_correct_coalesced else '❌'}")
    
    def benchmark_element_wise_operations(self):
        """Benchmark element-wise operations."""
        print("\n📊 Benchmarking Element-wise Operations:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            output = torch.empty_like(a)
            
            # Warmup
            for _ in range(self.warmup_runs):
                element_wise_add_kernel[(triton.cdiv(size, 128),)](
                    a, b, output, size, BLOCK_SIZE=128
                )
                _ = a + b
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                element_wise_add_kernel[(triton.cdiv(size, 128),)](
                    a, b, output, size, BLOCK_SIZE=128
                )
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = a + b
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            expected = a + b
            is_correct = torch.allclose(output, expected, rtol=1e-5, atol=1e-6)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_reduction_operations(self):
        """Benchmark reduction operations."""
        print("\n📊 Benchmarking Reduction Operations:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            output = torch.zeros(1, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                sum_reduction_kernel[(triton.cdiv(size, 128),)](
                    x, output, size, BLOCK_SIZE=128
                )
                _ = torch.sum(x)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                sum_reduction_kernel[(triton.cdiv(size, 128),)](
                    x, output, size, BLOCK_SIZE=128
                )
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.sum(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            expected = torch.sum(x)
            is_correct = torch.allclose(output, expected, rtol=1e-4, atol=1e-5)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_activation_functions(self):
        """Benchmark activation functions."""
        print("\n📊 Benchmarking Activation Functions:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536]
        activations = [
            ("ReLU", torch.relu),
            ("Sigmoid", torch.sigmoid),
            ("Tanh", torch.tanh),
            ("GELU", torch.nn.functional.gelu),
        ]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            
            for name, torch_func in activations:
                # Warmup
                for _ in range(self.warmup_runs):
                    _ = torch_func(x)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark PyTorch
                start_time = time.time()
                for _ in range(self.benchmark_runs):
                    _ = torch_func(x)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
                
                print(f"  {name}: {pytorch_time:.3f} ms")
    
    def benchmark_memory_bandwidth(self):
        """Benchmark memory bandwidth."""
        print("\n📊 Benchmarking Memory Bandwidth:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            output = torch.empty_like(x)
            
            # Warmup
            for _ in range(self.warmup_runs):
                output.copy_(x)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Measure time
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                output.copy_(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            elapsed_time = (time.time() - start_time) / self.benchmark_runs
            
            # Calculate bandwidth
            bytes_transferred = size * self.dtype.itemsize * 2  # read + write
            bandwidth_gb_s = (bytes_transferred / elapsed_time) / (1024**3)
            
            print(f"  Time: {elapsed_time*1000:.3f} ms")
            print(f"  Bandwidth: {bandwidth_gb_s:.1f} GB/s")
    
    def benchmark_data_types(self):
        """Benchmark different data types."""
        print("\n📊 Benchmarking Data Types:")
        print("=" * 50)
        
        if self.device != 'cuda':
            print("  Skipping data type benchmarks (CUDA not available)")
            return
        
        size = 1024
        data_types = [
            (torch.float32, "float32"),
            (torch.float16, "float16"),
            (torch.bfloat16, "bfloat16"),
            (torch.int32, "int32"),
        ]
        
        for dtype, name in data_types:
            print(f"\n📈 {name}:")
            
            # Create test data
            if dtype in [torch.int32]:
                a = torch.randint(0, 100, (size,), device=self.device, dtype=dtype)
                b = torch.randint(0, 100, (size,), device=self.device, dtype=dtype)
            else:
                a = torch.randn(size, device=self.device, dtype=dtype)
                b = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = a + b
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = a + b
            torch.cuda.synchronize()
            elapsed_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            print(f"  Time: {elapsed_time:.3f} ms")
    
    def run_all_benchmarks(self):
        """Run all beginner lesson benchmarks."""
        print("🚀 Running All Beginner Lesson Benchmarks")
        print("=" * 70)
        
        self.benchmark_vector_addition()
        self.benchmark_memory_access_patterns()
        self.benchmark_element_wise_operations()
        self.benchmark_reduction_operations()
        self.benchmark_activation_functions()
        self.benchmark_memory_bandwidth()
        self.benchmark_data_types()
        
        print("\n🎉 All Beginner Lesson Benchmarks Complete!")

def main():
    """Main function to run beginner lesson benchmarks."""
    benchmark_suite = BenchmarkBeginnerLessons()
    benchmark_suite.run_all_benchmarks()

if __name__ == "__main__":
    main()
