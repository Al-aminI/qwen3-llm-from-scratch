"""
📊 Benchmark Suite for Intermediate Lessons

This module contains benchmarks for the intermediate-level Triton tutorials.
"""

import torch
import triton
import triton.language as tl
import time
import sys
import os

# Add the parent directory to the path to import lessons
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lessons.intermediate.lesson_04_matrix_operations import basic_matmul, optimized_matmul, batch_matmul, matrix_transpose
from lessons.intermediate.lesson_05_advanced_memory import shared_memory_matmul, cache_friendly_reduction
from lessons.intermediate.lesson_06_kernel_fusion import fused_add_multiply, fused_matmul_activation, fused_loop

class BenchmarkIntermediateLessons:
    """
    📊 BENCHMARK SUITE FOR INTERMEDIATE LESSONS
    
    Benchmarks for lessons 4-6 (Matrix Operations, Advanced Memory, Kernel Fusion)
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
        
        print(f"🚀 Initializing Intermediate Lessons Benchmark Suite")
        print(f"   Device: {self.device}")
        print(f"   Warmup runs: {self.warmup_runs}")
        print(f"   Benchmark runs: {self.benchmark_runs}")
    
    def benchmark_matrix_operations(self):
        """Benchmark matrix operations."""
        print("\n📊 Benchmarking Matrix Operations:")
        print("=" * 50)
        
        sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]
        
        for M, K, N in sizes:
            print(f"\n📈 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
            
            # Create test data
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = basic_matmul(a, b)
                _ = optimized_matmul(a, b)
                _ = torch.matmul(a, b)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark basic Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = basic_matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            basic_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark optimized Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = optimized_matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            optimized_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Calculate speedups
            basic_speedup = pytorch_time / basic_time if basic_time > 0 else 0
            optimized_speedup = pytorch_time / optimized_time if optimized_time > 0 else 0
            
            print(f"  Basic Triton:    {basic_time:.3f} ms")
            print(f"  Optimized Triton: {optimized_time:.3f} ms")
            print(f"  PyTorch:         {pytorch_time:.3f} ms")
            print(f"  Basic Speedup:   {basic_speedup:.2f}x")
            print(f"  Optimized Speedup: {optimized_speedup:.2f}x")
            
            # Verify correctness
            basic_result = basic_matmul(a, b)
            optimized_result = optimized_matmul(a, b)
            pytorch_result = torch.matmul(a, b)
            
            basic_correct = torch.allclose(basic_result, pytorch_result, rtol=1e-3, atol=1e-3)
            optimized_correct = torch.allclose(optimized_result, pytorch_result, rtol=1e-3, atol=1e-3)
            
            print(f"  Basic Correct:   {'✅' if basic_correct else '❌'}")
            print(f"  Optimized Correct: {'✅' if optimized_correct else '❌'}")
    
    def benchmark_batch_matrix_operations(self):
        """Benchmark batch matrix operations."""
        print("\n📊 Benchmarking Batch Matrix Operations:")
        print("=" * 50)
        
        batch_sizes = [2, 4, 8, 16]
        M, K, N = 256, 128, 192
        
        for batch_size in batch_sizes:
            print(f"\n📈 Batch Size: {batch_size}")
            
            # Create test data
            a = torch.randn(batch_size, M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(batch_size, K, N, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = batch_matmul(a, b)
                _ = torch.bmm(a, b)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = batch_matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.bmm(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            triton_result = batch_matmul(a, b)
            pytorch_result = torch.bmm(a, b)
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_matrix_transpose(self):
        """Benchmark matrix transpose operations."""
        print("\n📊 Benchmarking Matrix Transpose:")
        print("=" * 50)
        
        sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        
        for M, N in sizes:
            print(f"\n📈 Size: {M}x{N}")
            
            # Create test data
            a = torch.randn(M, N, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = matrix_transpose(a)
                _ = a.t()
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = matrix_transpose(a)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = a.t()
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            triton_result = matrix_transpose(a)
            pytorch_result = a.t()
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_shared_memory_operations(self):
        """Benchmark shared memory operations."""
        print("\n📊 Benchmarking Shared Memory Operations:")
        print("=" * 50)
        
        sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
        
        for M, K, N in sizes:
            print(f"\n📈 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
            
            # Create test data
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = shared_memory_matmul(a, b)
                _ = torch.matmul(a, b)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = shared_memory_matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            triton_result = shared_memory_matmul(a, b)
            pytorch_result = torch.matmul(a, b)
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_cache_friendly_operations(self):
        """Benchmark cache-friendly operations."""
        print("\n📊 Benchmarking Cache-Friendly Operations:")
        print("=" * 50)
        
        sizes = [1024, 4096, 16384, 65536]
        
        for size in sizes:
            print(f"\n📈 Size: {size:,} elements")
            
            # Create test data
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = cache_friendly_reduction(x)
                _ = torch.sum(x)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = cache_friendly_reduction(x)
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
            triton_result = cache_friendly_reduction(x)
            pytorch_result = torch.sum(x)
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-4, atol=1e-5)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_kernel_fusion(self):
        """Benchmark kernel fusion operations."""
        print("\n📊 Benchmarking Kernel Fusion:")
        print("=" * 50)
        
        # Test fused add-multiply
        print("\n📈 Fused Add-Multiply:")
        sizes = [1024, 4096, 16384, 65536]
        
        for size in sizes:
            print(f"\n  Size: {size:,} elements")
            
            # Create test data
            a = torch.randn(size, device=self.device, dtype=self.dtype)
            b = torch.randn(size, device=self.device, dtype=self.dtype)
            c = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = fused_add_multiply(a, b, c)
                _ = (a + b) * c
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark fused operation
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = fused_add_multiply(a, b, c)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            fused_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark separate operations
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = (a + b) * c
            if self.device == 'cuda':
                torch.cuda.synchronize()
            separate_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = separate_time / fused_time if fused_time > 0 else 0
            
            print(f"    Fused:    {fused_time:.3f} ms")
            print(f"    Separate: {separate_time:.3f} ms")
            print(f"    Speedup:  {speedup:.2f}x")
            
            # Verify correctness
            fused_result = fused_add_multiply(a, b, c)
            separate_result = (a + b) * c
            is_correct = torch.allclose(fused_result, separate_result, rtol=1e-5, atol=1e-6)
            print(f"    Correct:  {'✅' if is_correct else '❌'}")
        
        # Test fused matrix multiplication + activation
        print("\n📈 Fused Matrix Multiplication + Activation:")
        mat_sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
        activations = ["relu", "tanh", "sigmoid", "gelu"]
        
        for M, K, N in mat_sizes:
            print(f"\n  Size: {M}x{K} @ {K}x{N} = {M}x{N}")
            
            # Create test data
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            for activation in activations:
                # Warmup
                for _ in range(self.warmup_runs):
                    _ = fused_matmul_activation(a, b, activation)
                    if activation == "relu":
                        _ = torch.relu(torch.matmul(a, b))
                    elif activation == "tanh":
                        _ = torch.tanh(torch.matmul(a, b))
                    elif activation == "sigmoid":
                        _ = torch.sigmoid(torch.matmul(a, b))
                    elif activation == "gelu":
                        _ = torch.nn.functional.gelu(torch.matmul(a, b))
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark fused operation
                start_time = time.time()
                for _ in range(self.benchmark_runs):
                    _ = fused_matmul_activation(a, b, activation)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                fused_time = (time.time() - start_time) / self.benchmark_runs * 1000
                
                # Benchmark separate operations
                start_time = time.time()
                for _ in range(self.benchmark_runs):
                    matmul_result = torch.matmul(a, b)
                    if activation == "relu":
                        _ = torch.relu(matmul_result)
                    elif activation == "tanh":
                        _ = torch.tanh(matmul_result)
                    elif activation == "sigmoid":
                        _ = torch.sigmoid(matmul_result)
                    elif activation == "gelu":
                        _ = torch.nn.functional.gelu(matmul_result)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                separate_time = (time.time() - start_time) / self.benchmark_runs * 1000
                
                speedup = separate_time / fused_time if fused_time > 0 else 0
                
                print(f"    {activation.capitalize()}:")
                print(f"      Fused:    {fused_time:.3f} ms")
                print(f"      Separate: {separate_time:.3f} ms")
                print(f"      Speedup:  {speedup:.2f}x")
        
        # Test fused loop
        print("\n📈 Fused Loop:")
        loop_sizes = [1024, 4096, 16384]
        
        for size in loop_sizes:
            print(f"\n  Size: {size:,} elements")
            
            # Create test data
            x = torch.randn(size, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = fused_loop(x, num_iterations=5)
                # PyTorch reference
                pytorch_result = x
                for i in range(5):
                    pytorch_result = pytorch_result * 2.0 + 1.0
                    pytorch_result = torch.relu(pytorch_result)
                    pytorch_result = pytorch_result * 0.5
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark fused operation
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = fused_loop(x, num_iterations=5)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            fused_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark separate operations
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                pytorch_result = x
                for i in range(5):
                    pytorch_result = pytorch_result * 2.0 + 1.0
                    pytorch_result = torch.relu(pytorch_result)
                    pytorch_result = pytorch_result * 0.5
            if self.device == 'cuda':
                torch.cuda.synchronize()
            separate_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = separate_time / fused_time if fused_time > 0 else 0
            
            print(f"    Fused:    {fused_time:.3f} ms")
            print(f"    Separate: {separate_time:.3f} ms")
            print(f"    Speedup:  {speedup:.2f}x")
            
            # Verify correctness
            fused_result = fused_loop(x, num_iterations=5)
            pytorch_result = x
            for i in range(5):
                pytorch_result = pytorch_result * 2.0 + 1.0
                pytorch_result = torch.relu(pytorch_result)
                pytorch_result = pytorch_result * 0.5
            is_correct = torch.allclose(fused_result, pytorch_result, rtol=1e-4, atol=1e-4)
            print(f"    Correct:  {'✅' if is_correct else '❌'}")
    
    def run_all_benchmarks(self):
        """Run all intermediate lesson benchmarks."""
        print("🚀 Running All Intermediate Lesson Benchmarks")
        print("=" * 70)
        
        self.benchmark_matrix_operations()
        self.benchmark_batch_matrix_operations()
        self.benchmark_matrix_transpose()
        self.benchmark_shared_memory_operations()
        self.benchmark_cache_friendly_operations()
        self.benchmark_kernel_fusion()
        
        print("\n🎉 All Intermediate Lesson Benchmarks Complete!")

def main():
    """Main function to run intermediate lesson benchmarks."""
    benchmark_suite = BenchmarkIntermediateLessons()
    benchmark_suite.run_all_benchmarks()

if __name__ == "__main__":
    main()
