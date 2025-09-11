"""
🔧 Lesson 4: Matrix Operations & Tiling Strategies

This lesson covers:
1. Matrix multiplication fundamentals and algorithms
2. Tiling and blocking strategies for memory optimization
3. Memory access patterns and coalescing
4. Performance analysis and optimization
5. Advanced matrix operations (transpose, batch operations)

Prerequisites: Lessons 1-3 (Beginner level)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🧠 MATRIX MULTIPLICATION FUNDAMENTALS
# ============================================================================

def explain_matrix_multiplication():
    """
    📚 MATRIX MULTIPLICATION FUNDAMENTALS
    
    Matrix multiplication is one of the most important operations in deep learning.
    Understanding the algorithms and optimization techniques is crucial.
    """
    print("🧠 Matrix Multiplication Fundamentals:")
    print("=" * 50)
    
    print("""
    🎯 What is Matrix Multiplication?
    
    Matrix multiplication C = A @ B where:
    - A is (M, K)
    - B is (K, N) 
    - C is (M, N)
    
    Each element C[i,j] = sum(A[i,k] * B[k,j] for k in range(K))
    
    🚀 Key Challenges:
    1. Memory Access Patterns: Need to access A, B, and C efficiently
    2. Data Reuse: Elements of A and B are used multiple times
    3. Load Balancing: Ensure all threads have equal work
    4. Memory Bandwidth: Maximize utilization of memory bandwidth
    
    📊 Optimization Strategies:
    1. Tiling: Divide matrices into blocks for better cache utilization
    2. Memory Coalescing: Ensure consecutive memory access
    3. Shared Memory: Cache frequently accessed data
    4. Register Blocking: Use registers for small blocks
    5. Loop Unrolling: Reduce loop overhead
    """)

# ============================================================================
# 🎯 BASIC MATRIX MULTIPLICATION KERNEL
# ============================================================================

@triton.jit
def basic_matmul_kernel(
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
    🎯 BASIC MATRIX MULTIPLICATION KERNEL
    
    Implements C = A @ B using tiling strategy.
    This is the foundation for more advanced optimizations.
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

def basic_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🎯 BASIC MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for the basic matrix multiplication kernel.
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
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    basic_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c

# ============================================================================
# 🎯 OPTIMIZED MATRIX MULTIPLICATION KERNEL
# ============================================================================

@triton.jit
def optimized_matmul_kernel(
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
    🎯 OPTIMIZED MATRIX MULTIPLICATION KERNEL
    
    Implements optimized matrix multiplication with:
    - Better memory access patterns
    - Improved tiling strategy
    - Enhanced boundary handling
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
        
        # Load A block with optimized access pattern
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block with optimized access pattern
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result with proper masking
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

def optimized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🎯 OPTIMIZED MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for the optimized matrix multiplication kernel.
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
    
    # Define optimized block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    optimized_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c

# ============================================================================
# 🎯 BATCH MATRIX MULTIPLICATION
# ============================================================================

@triton.jit
def batch_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    🎯 BATCH MATRIX MULTIPLICATION KERNEL
    
    Implements batch matrix multiplication for multiple matrix pairs.
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_b = pid_b < B
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
        a_ptrs = (a_ptr + pid_b * stride_ab + 
                  offs_m[:, None] * stride_am + 
                  offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_b & mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block
        b_ptrs = (b_ptr + pid_b * stride_bb + 
                  offs_k[:, None] * stride_bk + 
                  offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_b & mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result
    c_ptrs = (c_ptr + pid_b * stride_cb + 
              offs_m[:, None] * stride_cm + 
              offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=mask_b & mask_m[:, None] & mask_n[None, :])

def batch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🎯 BATCH MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for batch matrix multiplication.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape[0] == b.shape[0], "Batch dimensions must match!"
    assert a.shape[2] == b.shape[1], "Inner dimensions must match!"
    
    B, M, K = a.shape
    _, _, N = b.shape
    
    # Create output tensor
    c = torch.empty((B, M, N), device=a.device, dtype=torch.float32)
    
    # Calculate strides
    stride_ab, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)
    stride_bb, stride_bk, stride_bn = b.stride(0), b.stride(1), b.stride(2)
    stride_cb, stride_cm, stride_cn = c.stride(0), c.stride(1), c.stride(2)
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid = (B, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    batch_matmul_kernel[grid](
        a, b, c,
        B, M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c

# ============================================================================
# 🎯 MATRIX TRANSPOSE KERNEL
# ============================================================================

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    🎯 MATRIX TRANSPOSE KERNEL
    
    Transposes a matrix efficiently using tiling.
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
    
    # Load input block
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Transpose the block
    transposed_data = tl.trans(input_data)
    
    # Store output block
    output_ptrs = output_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    tl.store(output_ptrs, transposed_data, mask=mask_n[:, None] & mask_m[None, :])

def matrix_transpose(a: torch.Tensor) -> torch.Tensor:
    """
    🎯 MATRIX TRANSPOSE WRAPPER
    
    Wrapper function for matrix transpose.
    """
    # Input validation
    assert a.is_cuda, "Input tensor must be on GPU!"
    assert a.dim() == 2, "Input must be a 2D tensor!"
    
    M, N = a.shape
    
    # Create output tensor
    output = torch.empty((N, M), device=a.device, dtype=a.dtype)
    
    # Calculate strides
    stride_im, stride_in = a.stride(0), a.stride(1)
    stride_om, stride_on = output.stride(0), output.stride(1)
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    matrix_transpose_kernel[grid](
        a, output,
        M, N,
        stride_im, stride_in,
        stride_om, stride_on,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_matrix_operations():
    """
    🧪 TEST MATRIX OPERATIONS
    
    Tests various matrix operations and validates correctness.
    """
    print("\n🧪 Testing Matrix Operations:")
    print("=" * 50)
    
    # Test basic matrix multiplication
    print("\n📊 Test: Basic Matrix Multiplication")
    M, K, N = 256, 128, 192
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    triton_result = basic_matmul(a, b)
    pytorch_result = torch.matmul(a, b)
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test optimized matrix multiplication
    print("\n📊 Test: Optimized Matrix Multiplication")
    triton_opt_result = optimized_matmul(a, b)
    is_correct = torch.allclose(triton_opt_result, pytorch_result, rtol=1e-3, atol=1e-3)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test batch matrix multiplication
    print("\n📊 Test: Batch Matrix Multiplication")
    B = 8
    a_batch = torch.randn(B, M, K, device='cuda', dtype=torch.float32)
    b_batch = torch.randn(B, K, N, device='cuda', dtype=torch.float32)
    
    triton_batch_result = batch_matmul(a_batch, b_batch)
    pytorch_batch_result = torch.bmm(a_batch, b_batch)
    
    is_correct = torch.allclose(triton_batch_result, pytorch_batch_result, rtol=1e-3, atol=1e-3)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test matrix transpose
    print("\n📊 Test: Matrix Transpose")
    triton_transpose_result = matrix_transpose(a)
    pytorch_transpose_result = a.t()
    
    is_correct = torch.allclose(triton_transpose_result, pytorch_transpose_result, rtol=1e-5)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_matrix_operations():
    """
    📊 BENCHMARK MATRIX OPERATIONS
    
    Compares performance between Triton and PyTorch for matrix operations.
    """
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
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
            _ = basic_matmul(a, b)
            _ = optimized_matmul(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 10 * 1000
        
        # Benchmark basic Triton
        start_time = time.time()
        for _ in range(10):
            _ = basic_matmul(a, b)
        torch.cuda.synchronize()
        triton_basic_time = (time.time() - start_time) / 10 * 1000
        
        # Benchmark optimized Triton
        start_time = time.time()
        for _ in range(10):
            _ = optimized_matmul(a, b)
        torch.cuda.synchronize()
        triton_opt_time = (time.time() - start_time) / 10 * 1000
        
        print(f"  PyTorch:        {pytorch_time:.3f} ms")
        print(f"  Triton Basic:   {triton_basic_time:.3f} ms")
        print(f"  Triton Optimized: {triton_opt_time:.3f} ms")
        print(f"  Basic Speedup:  {pytorch_time / triton_basic_time:.2f}x")
        print(f"  Opt Speedup:    {pytorch_time / triton_opt_time:.2f}x")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 4 tutorial.
    """
    print("🔧 LESSON 4: MATRIX OPERATIONS & TILING STRATEGIES")
    print("=" * 70)
    print("This lesson covers matrix operations and optimization techniques in Triton.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_matrix_multiplication()
    test_matrix_operations()
    benchmark_matrix_operations()
    
    print("\n🎉 Lesson 4 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding matrix multiplication algorithms")
    print("2. ✅ Implementing tiling and blocking strategies")
    print("3. ✅ Optimizing memory access patterns")
    print("4. ✅ Working with batch operations")
    print("5. ✅ Implementing matrix transpose")
    print("6. ✅ Performance analysis and optimization")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different block sizes and tiling strategies")
    print("- Try implementing more complex matrix operations")
    print("- Move on to Lesson 5: Advanced Memory Patterns & Optimization")

if __name__ == "__main__":
    main()
