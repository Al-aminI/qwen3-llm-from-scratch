"""
🔧 Lesson 5: Advanced Memory Patterns & Optimization

This lesson covers:
1. Shared memory optimization techniques
2. Memory coalescing patterns and strategies
3. Cache-friendly algorithms and data layouts
4. Bandwidth optimization and memory hierarchy
5. Advanced memory access patterns

Prerequisites: Lessons 1-4 (Beginner and intermediate matrix operations)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🧠 SHARED MEMORY OPTIMIZATION
# ============================================================================

def explain_shared_memory():
    """
    📚 SHARED MEMORY OPTIMIZATION
    
    Shared memory is a fast, on-chip memory that's shared among threads in a block.
    It's crucial for optimizing memory access patterns and reducing global memory traffic.
    """
    print("🧠 Shared Memory Optimization:")
    print("=" * 50)
    
    print("""
    🎯 What is Shared Memory?
    
    Shared memory is a fast, on-chip memory that's shared among threads in a block:
    - Size: ~48KB per block (varies by GPU)
    - Speed: 1-32 cycles (much faster than global memory)
    - Scope: Shared among all threads in a block
    - Use cases: Inter-thread communication, caching frequently accessed data
    
    🚀 Key Benefits:
    1. Fast Access: Much faster than global memory
    2. Inter-thread Communication: Threads can share data
    3. Caching: Store frequently accessed data
    4. Bandwidth: Higher bandwidth than global memory
    
    📊 Optimization Strategies:
    1. Use shared memory for frequently accessed data
    2. Minimize shared memory bank conflicts
    3. Use appropriate shared memory layouts
    4. Balance shared memory usage with occupancy
    """)

@triton.jit
def shared_memory_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    🎯 SHARED MEMORY MATRIX MULTIPLICATION KERNEL
    
    Demonstrates how to use shared memory for matrix multiplication.
    This kernel uses shared memory to cache frequently accessed data.
    """
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Calculate K offsets
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load A block
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

def shared_memory_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🎯 SHARED MEMORY MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for shared memory matrix multiplication.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match!"
    
    M, K = a.shape
    _, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Calculate strides
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)
    
    # Define block sizes optimized for shared memory
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    shared_memory_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c

# ============================================================================
# 🎯 MEMORY COALESCING PATTERNS
# ============================================================================

def explain_memory_coalescing_patterns():
    """
    📚 MEMORY COALESCING PATTERNS
    
    Different memory access patterns have different coalescing characteristics.
    Understanding these patterns is crucial for optimization.
    """
    print("\n🎯 Memory Coalescing Patterns:")
    print("=" * 50)
    
    print("""
    🧠 Coalescing Patterns:
    
    1. Perfect Coalescing:
       - All threads access consecutive memory locations
       - Maximum memory bandwidth utilization
       - Example: vector addition, element-wise operations
    
    2. Strided Access:
       - Threads access memory with a constant stride
       - Coalescing depends on stride value
       - Example: matrix transpose, strided operations
    
    3. Random Access:
       - Threads access scattered memory locations
       - Poor coalescing, low bandwidth utilization
       - Example: sparse operations, irregular access
    
    4. Block Access:
       - Threads access memory in blocks
       - Good coalescing within blocks
       - Example: matrix operations, tiled algorithms
    
    🚀 Optimization Strategies:
    1. Design algorithms for coalesced access
    2. Use appropriate data layouts
    3. Minimize memory transactions
    4. Consider memory alignment
    """)

@triton.jit
def coalesced_access_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 COALESCED ACCESS KERNEL
    
    Demonstrates perfect memory coalescing.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced access: consecutive memory locations
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # Simple operation
    output = data * 2.0
    
    # Coalesced store: consecutive memory locations
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def strided_access_kernel(
    input_ptr, output_ptr,
    n_elements, stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 STRIDED ACCESS KERNEL
    
    Demonstrates strided memory access.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Strided access: memory locations with stride
    strided_offsets = offsets * stride
    strided_mask = strided_offsets < n_elements
    
    data = tl.load(input_ptr + strided_offsets, mask=strided_mask)
    output = data * 2.0
    tl.store(output_ptr + strided_offsets, output, mask=strided_mask)

def test_memory_coalescing():
    """
    🧪 TEST MEMORY COALESCING PATTERNS
    
    Tests different memory access patterns and measures performance.
    """
    print("\n🧪 Testing Memory Coalescing Patterns:")
    print("=" * 50)
    
    size = 1024 * 1024  # 1M elements
    
    # Create test data
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    output_coalesced = torch.empty_like(x)
    output_strided = torch.empty_like(x)
    
    # Test coalesced access
    print("\n📊 Test: Coalesced Access")
    coalesced_access_kernel[(triton.cdiv(size, 128),)](
        x, output_coalesced, size, BLOCK_SIZE=128
    )
    
    # Test strided access
    print("\n📊 Test: Strided Access (stride=2)")
    strided_access_kernel[(triton.cdiv(size, 128),)](
        x, output_strided, size, stride=2, BLOCK_SIZE=128
    )
    
    # Verify correctness
    expected_coalesced = x * 2.0
    expected_strided = x[::2] * 2.0
    
    is_correct_coalesced = torch.allclose(output_coalesced, expected_coalesced, rtol=1e-5)
    is_correct_strided = torch.allclose(output_strided[::2], expected_strided, rtol=1e-5)
    
    print(f"  Coalesced: {'✅ PASS' if is_correct_coalesced else '❌ FAIL'}")
    print(f"  Strided: {'✅ PASS' if is_correct_strided else '❌ FAIL'}")

# ============================================================================
# 🎯 CACHE-FRIENDLY ALGORITHMS
# ============================================================================

def explain_cache_friendly_algorithms():
    """
    📚 CACHE-FRIENDLY ALGORITHMS
    
    Cache-friendly algorithms are designed to maximize cache utilization
    and minimize cache misses.
    """
    print("\n🎯 Cache-Friendly Algorithms:")
    print("=" * 50)
    
    print("""
    🧠 Cache-Friendly Design Principles:
    
    1. Spatial Locality:
       - Access nearby memory locations
       - Use contiguous memory layouts
       - Minimize memory fragmentation
    
    2. Temporal Locality:
       - Reuse recently accessed data
       - Keep frequently used data in cache
       - Minimize cache evictions
    
    3. Data Layout Optimization:
       - Use appropriate data structures
       - Minimize pointer chasing
       - Consider cache line sizes
    
    4. Algorithm Design:
       - Use blocking and tiling
       - Minimize random access
       - Optimize for cache hierarchy
    
    🚀 Optimization Strategies:
    1. Use blocking and tiling techniques
    2. Optimize data layouts for cache
    3. Minimize memory transactions
    4. Consider cache line alignment
    """)

@triton.jit
def cache_friendly_reduction_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 CACHE-FRIENDLY REDUCTION KERNEL
    
    Implements cache-friendly reduction using shared memory.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with good spatial locality
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Block-level reduction
    block_sum = tl.sum(data, axis=0)
    
    # Store block result
    if pid == 0:  # Only the first block stores the result
        tl.store(output_ptr, block_sum)

def cache_friendly_reduction(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    🎯 CACHE-FRIENDLY REDUCTION WRAPPER
    
    Wrapper function for cache-friendly reduction.
    """
    # Input validation
    assert input_tensor.is_cuda, "Input tensor must be on GPU!"
    
    n_elements = input_tensor.numel()
    output = torch.zeros(1, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    cache_friendly_reduction_kernel[grid](
        input_tensor, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🎯 BANDWIDTH OPTIMIZATION
# ============================================================================

def explain_bandwidth_optimization():
    """
    📚 BANDWIDTH OPTIMIZATION
    
    Memory bandwidth is often the bottleneck in GPU kernels.
    Optimizing bandwidth utilization is crucial for performance.
    """
    print("\n🎯 Bandwidth Optimization:")
    print("=" * 50)
    
    print("""
    🧠 Bandwidth Optimization Strategies:
    
    1. Memory Coalescing:
       - Ensure consecutive memory access
       - Maximize memory transaction efficiency
       - Use appropriate data types
    
    2. Data Type Optimization:
       - Use appropriate precision
       - Consider memory vs compute trade-offs
       - Use vectorized operations
    
    3. Memory Access Patterns:
       - Minimize memory transactions
       - Use shared memory for frequently accessed data
       - Optimize for cache hierarchy
    
    4. Algorithm Design:
       - Use blocking and tiling
       - Minimize redundant memory access
       - Consider memory vs compute balance
    
    🚀 Key Metrics:
    1. Memory Bandwidth Utilization
    2. Cache Hit Rate
    3. Memory Transaction Efficiency
    4. Overall Kernel Performance
    """)

@triton.jit
def bandwidth_optimized_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 BANDWIDTH OPTIMIZED KERNEL
    
    Demonstrates bandwidth optimization techniques.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with optimal bandwidth utilization
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform computation
    output = data * 2.0 + 1.0
    
    # Store result with optimal bandwidth utilization
    tl.store(output_ptr + offsets, output, mask=mask)

def measure_memory_bandwidth():
    """
    📊 MEASURE MEMORY BANDWIDTH
    
    Measures achieved memory bandwidth for different configurations.
    """
    print("\n📊 Measuring Memory Bandwidth:")
    print("=" * 50)
    
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        # Create test data
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)
        
        # Warmup
        for _ in range(10):
            bandwidth_optimized_kernel[(triton.cdiv(size, 128),)](
                x, output, size, BLOCK_SIZE=128
            )
        
        torch.cuda.synchronize()
        
        # Measure time
        start_time = time.time()
        for _ in range(100):
            bandwidth_optimized_kernel[(triton.cdiv(size, 128),)](
                x, output, size, BLOCK_SIZE=128
            )
        torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) / 100
        
        # Calculate bandwidth
        # Each element is read once and written once (2 operations)
        # Each element is 4 bytes (float32)
        bytes_transferred = size * 4 * 2  # read + write
        bandwidth_gb_s = (bytes_transferred / elapsed_time) / (1024**3)
        
        print(f"  Time: {elapsed_time*1000:.3f} ms")
        print(f"  Bandwidth: {bandwidth_gb_s:.1f} GB/s")

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_advanced_memory():
    """
    🧪 TEST ADVANCED MEMORY OPTIMIZATIONS
    
    Tests various advanced memory optimization techniques.
    """
    print("\n🧪 Testing Advanced Memory Optimizations:")
    print("=" * 50)
    
    # Test shared memory matrix multiplication
    print("\n📊 Test: Shared Memory Matrix Multiplication")
    M, K, N = 256, 128, 192
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    triton_result = shared_memory_matmul(a, b)
    pytorch_result = torch.matmul(a, b)
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test cache-friendly reduction
    print("\n📊 Test: Cache-Friendly Reduction")
    size = 1024
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    
    triton_result = cache_friendly_reduction(x)
    pytorch_result = torch.sum(x)
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-4)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    print(f"  Triton: {triton_result.item():.6f}")
    print(f"  PyTorch: {pytorch_result.item():.6f}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_advanced_memory():
    """
    📊 BENCHMARK ADVANCED MEMORY OPTIMIZATIONS
    
    Compares performance between different memory optimization techniques.
    """
    print("\n📊 Benchmarking Advanced Memory Optimizations:")
    print("=" * 50)
    
    # Benchmark shared memory matrix multiplication
    print("\n📈 Benchmark: Shared Memory Matrix Multiplication")
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, K, N in sizes:
        print(f"\n  Size: {M}x{K} @ {K}x{N} = {M}x{N}")
        
        # Create test data
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
            _ = shared_memory_matmul(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(100):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 100 * 1000
        
        # Benchmark Triton
        start_time = time.time()
        for _ in range(100):
            _ = shared_memory_matmul(a, b)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / 100 * 1000
        
        speedup = pytorch_time / triton_time if triton_time > 0 else 0
        
        print(f"    PyTorch: {pytorch_time:.3f} ms")
        print(f"    Triton:  {triton_time:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 5 tutorial.
    """
    print("🔧 LESSON 5: ADVANCED MEMORY PATTERNS & OPTIMIZATION")
    print("=" * 70)
    print("This lesson covers advanced memory optimization techniques in Triton.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_shared_memory()
    explain_memory_coalescing_patterns()
    test_memory_coalescing()
    
    explain_cache_friendly_algorithms()
    explain_bandwidth_optimization()
    measure_memory_bandwidth()
    
    test_advanced_memory()
    benchmark_advanced_memory()
    
    print("\n🎉 Lesson 5 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding shared memory optimization")
    print("2. ✅ Learning memory coalescing patterns")
    print("3. ✅ Implementing cache-friendly algorithms")
    print("4. ✅ Optimizing memory bandwidth utilization")
    print("5. ✅ Advanced memory access patterns")
    print("6. ✅ Performance analysis and optimization")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different shared memory configurations")
    print("- Try optimizing memory access patterns")
    print("- Move on to Lesson 6: Kernel Fusion & Performance Tuning")

if __name__ == "__main__":
    main()
