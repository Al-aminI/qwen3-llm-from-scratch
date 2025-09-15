"""
🚀 Performance Optimization Examples

This module demonstrates various performance optimization techniques using Triton.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List
from triton_tutorials.utils.benchmarking import BenchmarkSuite
from triton_tutorials.utils.profiling import PerformanceProfiler
from triton_tutorials.utils.performance_analysis import PerformanceAnalyzer

# ============================================================================
# 🧠 BASIC PERFORMANCE OPTIMIZATION
# ============================================================================

@triton.jit
def basic_vector_add_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🧠 BASIC VECTOR ADDITION KERNEL
    
    Basic vector addition implementation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    output = a + b
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def basic_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🧠 BASIC VECTOR ADDITION WRAPPER
    
    Wrapper function for basic vector addition.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape == b.shape, "Input tensors must have the same shape!"
    
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    basic_vector_add_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🚀 OPTIMIZED PERFORMANCE
# ============================================================================

@triton.jit
def optimized_vector_add_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🚀 OPTIMIZED VECTOR ADDITION KERNEL
    
    Optimized vector addition with:
    - Coalesced memory access
    - Optimal block size
    - Efficient memory patterns
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with coalesced access
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    output = a + b
    
    # Store result with coalesced access
    tl.store(output_ptr + offsets, output, mask=mask)

def optimized_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED VECTOR ADDITION WRAPPER
    
    Wrapper function for optimized vector addition.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape == b.shape, "Input tensors must have the same shape!"
    
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    # Define optimal block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    optimized_vector_add_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🔥 FUSED PERFORMANCE
# ============================================================================

@triton.jit
def fused_vector_operations_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🔥 FUSED VECTOR OPERATIONS KERNEL
    
    Fused vector operations:
    output = (a + b) * c + a * 2.0
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: (a + b) * c + a * 2.0
    intermediate1 = a + b
    intermediate2 = intermediate1 * c
    intermediate3 = a * 2.0
    output = intermediate2 + intermediate3
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_vector_operations(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    🔥 FUSED VECTOR OPERATIONS WRAPPER
    
    Wrapper function for fused vector operations.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda and c.is_cuda, "Input tensors must be on GPU!"
    assert a.shape == b.shape == c.shape, "Input tensors must have the same shape!"
    
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_vector_operations_kernel[grid](
        a, b, c, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🎯 AUTOTUNED PERFORMANCE
# ============================================================================

@triton.jit
def autotuned_vector_add_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    """
    🎯 AUTOTUNED VECTOR ADDITION KERNEL
    
    Autotuned vector addition with configurable parameters.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    output = a + b
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def autotuned_vector_add(a: torch.Tensor, b: torch.Tensor, 
                        block_size: int = 256, num_warps: int = 4) -> torch.Tensor:
    """
    🎯 AUTOTUNED VECTOR ADDITION WRAPPER
    
    Wrapper function for autotuned vector addition.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape == b.shape, "Input tensors must have the same shape!"
    
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, block_size),)
    
    # Launch kernel
    autotuned_vector_add_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=block_size,
        NUM_WARPS=num_warps
    )
    
    return output

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_performance_optimizations():
    """
    🧪 TEST PERFORMANCE OPTIMIZATIONS
    
    Tests various performance optimization techniques.
    """
    print("🧪 Testing Performance Optimizations:")
    print("=" * 50)
    
    # Test basic vector addition
    print("\n📊 Test: Basic Vector Addition")
    size = 1024
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    
    triton_result = basic_vector_add(a, b)
    pytorch_result = a + b
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test optimized vector addition
    print("\n📊 Test: Optimized Vector Addition")
    triton_result = optimized_vector_add(a, b)
    pytorch_result = a + b
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test fused vector operations
    print("\n📊 Test: Fused Vector Operations")
    c = torch.randn(size, device='cuda', dtype=torch.float32)
    
    triton_result = fused_vector_operations(a, b, c)
    pytorch_result = (a + b) * c + a * 2.0
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test autotuned vector addition
    print("\n📊 Test: Autotuned Vector Addition")
    triton_result = autotuned_vector_add(a, b, block_size=256, num_warps=4)
    pytorch_result = a + b
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_performance_optimizations():
    """
    📊 BENCHMARK PERFORMANCE OPTIMIZATIONS
    
    Benchmarks various performance optimization techniques.
    """
    print("\n📊 Benchmarking Performance Optimizations:")
    print("=" * 50)
    
    # Test different sizes
    sizes = [1024, 4096, 16384, 65536, 262144]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        # Create test data
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)
        c = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Benchmark basic vector addition
        print("\n  Basic Vector Addition:")
        try:
            # Warmup
            for _ in range(10):
                _ = basic_vector_add(a, b)
                _ = a + b
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = basic_vector_add(a, b)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = a + b
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        
        # Benchmark optimized vector addition
        print("\n  Optimized Vector Addition:")
        try:
            # Warmup
            for _ in range(10):
                _ = optimized_vector_add(a, b)
                _ = a + b
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = optimized_vector_add(a, b)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = a + b
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        
        # Benchmark fused vector operations
        print("\n  Fused Vector Operations:")
        try:
            # Warmup
            for _ in range(10):
                _ = fused_vector_operations(a, b, c)
                _ = (a + b) * c + a * 2.0
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = fused_vector_operations(a, b, c)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = (a + b) * c + a * 2.0
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        
        # Benchmark autotuned vector addition
        print("\n  Autotuned Vector Addition:")
        try:
            # Warmup
            for _ in range(10):
                _ = autotuned_vector_add(a, b, block_size=256, num_warps=4)
                _ = a + b
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = autotuned_vector_add(a, b, block_size=256, num_warps=4)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = a + b
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")

# ============================================================================
# 🔍 PROFILING AND ANALYSIS
# ============================================================================

def profile_performance_optimizations():
    """
    🔍 PROFILE PERFORMANCE OPTIMIZATIONS
    
    Profiles various performance optimization techniques.
    """
    print("\n🔍 Profiling Performance Optimizations:")
    print("=" * 50)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Test configuration
    size = 1024 * 1024  # 1M elements
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Profile basic vector addition
    print("\n📊 Profile: Basic Vector Addition")
    result = profiler.profile_function(
        basic_vector_add,
        "Basic Vector Addition",
        a, b
    )
    
    if result:
        print(f"  Execution Time: {result.execution_time:.3f} ms")
        print(f"  Memory Usage: {result.memory_usage:.3f} GB")
        print(f"  GPU Memory Usage: {result.gpu_memory_usage:.3f} GB")
        print(f"  Status: ✅")
    else:
        print(f"  Status: ❌")
    
    # Profile optimized vector addition
    print("\n📊 Profile: Optimized Vector Addition")
    result = profiler.profile_function(
        optimized_vector_add,
        "Optimized Vector Addition",
        a, b
    )
    
    if result:
        print(f"  Execution Time: {result.execution_time:.3f} ms")
        print(f"  Memory Usage: {result.memory_usage:.3f} GB")
        print(f"  GPU Memory Usage: {result.gpu_memory_usage:.3f} GB")
        print(f"  Status: ✅")
    else:
        print(f"  Status: ❌")
    
    # Profile fused vector operations
    print("\n📊 Profile: Fused Vector Operations")
    result = profiler.profile_function(
        fused_vector_operations,
        "Fused Vector Operations",
        a, b, c
    )
    
    if result:
        print(f"  Execution Time: {result.execution_time:.3f} ms")
        print(f"  Memory Usage: {result.memory_usage:.3f} GB")
        print(f"  GPU Memory Usage: {result.gpu_memory_usage:.3f} GB")
        print(f"  Status: ✅")
    else:
        print(f"  Status: ❌")
    
    # Profile autotuned vector addition
    print("\n📊 Profile: Autotuned Vector Addition")
    result = profiler.profile_function(
        lambda x, y: autotuned_vector_add(x, y, block_size=256, num_warps=4),
        "Autotuned Vector Addition",
        a, b
    )
    
    if result:
        print(f"  Execution Time: {result.execution_time:.3f} ms")
        print(f"  Memory Usage: {result.memory_usage:.3f} GB")
        print(f"  GPU Memory Usage: {result.gpu_memory_usage:.3f} GB")
        print(f"  Status: ✅")
    else:
        print(f"  Status: ❌")
    
    # Print profiling results
    profiler.print_results()
    
    # Save results
    profiler.save_results("performance_profiling_results.json")

# ============================================================================
# 📈 PERFORMANCE ANALYSIS
# ============================================================================

def analyze_performance_optimizations():
    """
    📈 ANALYZE PERFORMANCE OPTIMIZATIONS
    
    Analyzes various performance optimization techniques.
    """
    print("\n📈 Analyzing Performance Optimizations:")
    print("=" * 50)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Test different sizes
    sizes = [1024, 4096, 16384, 65536, 262144]
    
    for size in sizes:
        print(f"\n📊 Size: {size:,} elements")
        
        # Create test data
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Analyze basic vector addition
        result = analyzer.analyze_kernel(
            basic_vector_add,
            f"Basic Vector Addition ({size:,})",
            size,
            torch.float32,
            num_runs=100
        )
        
        if result:
            print(f"  Basic Vector Addition:")
            print(f"    Execution Time: {result.execution_time:.3f} ms")
            print(f"    Memory Bandwidth: {result.memory_bandwidth:.1f} GB/s")
            print(f"    Throughput: {result.throughput:.0f} elements/s")
            print(f"    Efficiency: {result.efficiency:.2f}")
        
        # Analyze optimized vector addition
        result = analyzer.analyze_kernel(
            optimized_vector_add,
            f"Optimized Vector Addition ({size:,})",
            size,
            torch.float32,
            num_runs=100
        )
        
        if result:
            print(f"  Optimized Vector Addition:")
            print(f"    Execution Time: {result.execution_time:.3f} ms")
            print(f"    Memory Bandwidth: {result.memory_bandwidth:.1f} GB/s")
            print(f"    Throughput: {result.throughput:.0f} elements/s")
            print(f"    Efficiency: {result.efficiency:.2f}")
    
    # Print analysis summary
    analyzer.print_summary()
    
    # Generate report
    analyzer.generate_report("performance_analysis_report.json")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the performance optimization examples.
    """
    print("🚀 PERFORMANCE OPTIMIZATION EXAMPLES")
    print("=" * 70)
    print("This module demonstrates various performance optimization techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the examples
    test_performance_optimizations()
    benchmark_performance_optimizations()
    profile_performance_optimizations()
    analyze_performance_optimizations()
    
    print("\n🎉 Performance Optimization Examples Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Basic performance optimization techniques")
    print("2. ✅ Optimized memory access patterns")
    print("3. ✅ Fused operations for better performance")
    print("4. ✅ Autotuned kernels for optimal performance")
    print("5. ✅ Performance profiling and analysis")
    print("6. ✅ Memory bandwidth optimization")
    print("7. ✅ Scalable performance optimization")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different optimization strategies")
    print("- Try optimizing for different hardware configurations")
    print("- Implement advanced autotuning techniques")
    print("- Add support for different data types and operations")

if __name__ == "__main__":
    main()
