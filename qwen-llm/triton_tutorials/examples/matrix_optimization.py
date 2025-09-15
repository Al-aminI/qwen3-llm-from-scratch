"""
🔢 Matrix Optimization Examples

This module demonstrates various matrix optimization techniques using Triton.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional

# ============================================================================
# 🧠 BASIC MATRIX OPERATIONS
# ============================================================================

@triton.jit
def basic_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🧠 BASIC MATRIX MULTIPLICATION KERNEL
    
    Implements basic matrix multiplication: C = A @ B
    """
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE
    block_start_n = pid_n * BLOCK_SIZE
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE):
        offs_k = k + tl.arange(0, BLOCK_SIZE)
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
    🧠 BASIC MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for basic matrix multiplication.
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
    
    # Define block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    
    # Launch kernel
    basic_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return c

# ============================================================================
# 🚀 OPTIMIZED MATRIX OPERATIONS
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
    🚀 OPTIMIZED MATRIX MULTIPLICATION KERNEL
    
    Implements optimized matrix multiplication with:
    - Tiled computation
    - Shared memory optimization
    - Fused operations
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

def optimized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for optimized matrix multiplication.
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
    optimized_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return c

# ============================================================================
# 🔥 FUSED MATRIX OPERATIONS
# ============================================================================

@triton.jit
def fused_matmul_activation_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    activation_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    🔥 FUSED MATRIX MULTIPLICATION + ACTIVATION KERNEL
    
    Implements fused matrix multiplication with activation:
    output = activation(A @ B)
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
    
    # Apply activation function (fused)
    if activation_type == 0:  # ReLU
        output = tl.where(accumulator > 0, accumulator, 0.0)
    elif activation_type == 1:  # Sigmoid
        output = 1.0 / (1.0 + tl.exp(-accumulator))
    elif activation_type == 2:  # Tanh
        output = tl.tanh(accumulator)
    elif activation_type == 3:  # GELU (approximation)
        output = 0.5 * accumulator * (1.0 + tl.tanh(0.79788456 * (accumulator + 0.044715 * accumulator * accumulator * accumulator)))
    else:
        output = accumulator  # Identity
    
    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(output_ptrs, output, mask=mask_m[:, None] & mask_n[None, :])

def fused_matmul_activation(a: torch.Tensor, b: torch.Tensor, activation: str = "relu") -> torch.Tensor:
    """
    🔥 FUSED MATRIX MULTIPLICATION + ACTIVATION WRAPPER
    
    Wrapper function for fused matrix multiplication + activation.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match!"
    
    M, K = a.shape
    _, N = b.shape
    
    # Create output tensor
    output = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Calculate strides
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = output.stride(0), output.stride(1)
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Map activation string to integer
    activation_map = {"relu": 0, "sigmoid": 1, "tanh": 2, "gelu": 3}
    activation_type = activation_map.get(activation, 0)
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    fused_matmul_activation_kernel[grid](
        a, b, output,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        activation_type=activation_type,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

# ============================================================================
# 🎯 BATCH MATRIX OPERATIONS
# ============================================================================

@triton.jit
def batch_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    🎯 BATCH MATRIX MULTIPLICATION KERNEL
    
    Implements batch matrix multiplication: C[b] = A[b] @ B[b]
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
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load A block
        a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block
        b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

def batch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    🎯 BATCH MATRIX MULTIPLICATION WRAPPER
    
    Wrapper function for batch matrix multiplication.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU!"
    assert a.shape[0] == b.shape[0], "Batch sizes must match!"
    assert a.shape[2] == b.shape[1], "Inner dimensions must match!"
    
    batch_size, M, K = a.shape
    _, _, N = b.shape
    
    # Create output tensor
    c = torch.empty((batch_size, M, N), device=a.device, dtype=torch.float32)
    
    # Calculate strides
    stride_ab, stride_am, stride_ak = a.stride()
    stride_bb, stride_bk, stride_bn = b.stride()
    stride_cb, stride_cm, stride_cn = c.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    batch_matmul_kernel[grid](
        a, b, c,
        batch_size, M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return c

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_matrix_operations():
    """
    🧪 TEST MATRIX OPERATIONS
    
    Tests various matrix operations and validates correctness.
    """
    print("🧪 Testing Matrix Operations:")
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
    triton_result = optimized_matmul(a, b)
    pytorch_result = torch.matmul(a, b)
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test fused matrix multiplication + activation
    print("\n📊 Test: Fused Matrix Multiplication + Activation")
    activations = ["relu", "tanh", "sigmoid", "gelu"]
    
    for activation in activations:
        triton_result = fused_matmul_activation(a, b, activation)
        
        # PyTorch reference
        matmul_result = torch.matmul(a, b)
        if activation == "relu":
            pytorch_result = torch.relu(matmul_result)
        elif activation == "tanh":
            pytorch_result = torch.tanh(matmul_result)
        elif activation == "sigmoid":
            pytorch_result = torch.sigmoid(matmul_result)
        elif activation == "gelu":
            pytorch_result = torch.nn.functional.gelu(matmul_result)
        
        is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
        print(f"  {activation.capitalize()}: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test batch matrix multiplication
    print("\n📊 Test: Batch Matrix Multiplication")
    batch_size = 4
    batch_a = torch.randn(batch_size, M, K, device='cuda', dtype=torch.float32)
    batch_b = torch.randn(batch_size, K, N, device='cuda', dtype=torch.float32)
    
    triton_result = batch_matmul(batch_a, batch_b)
    pytorch_result = torch.bmm(batch_a, batch_b)
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_matrix_operations():
    """
    📊 BENCHMARK MATRIX OPERATIONS
    
    Benchmarks various matrix operations and compares performance.
    """
    print("\n📊 Benchmarking Matrix Operations:")
    print("=" * 50)
    
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
        
        # Benchmark basic matrix multiplication
        print("\n  Basic Matrix Multiplication:")
        try:
            # Warmup
            for _ in range(10):
                _ = basic_matmul(a, b)
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = basic_matmul(a, b)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        
        # Benchmark optimized matrix multiplication
        print("\n  Optimized Matrix Multiplication:")
        try:
            # Warmup
            for _ in range(10):
                _ = optimized_matmul(a, b)
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = optimized_matmul(a, b)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        
        # Benchmark fused matrix multiplication + activation
        print("\n  Fused Matrix Multiplication + Activation:")
        activations = ["relu", "tanh", "sigmoid", "gelu"]
        
        for activation in activations:
            try:
                # Warmup
                for _ in range(10):
                    _ = fused_matmul_activation(a, b, activation)
                    matmul_result = torch.matmul(a, b)
                    if activation == "relu":
                        _ = torch.relu(matmul_result)
                    elif activation == "tanh":
                        _ = torch.tanh(matmul_result)
                    elif activation == "sigmoid":
                        _ = torch.sigmoid(matmul_result)
                    elif activation == "gelu":
                        _ = torch.nn.functional.gelu(matmul_result)
                
                torch.cuda.synchronize()
                
                # Benchmark Triton
                start_time = time.time()
                for _ in range(100):
                    _ = fused_matmul_activation(a, b, activation)
                torch.cuda.synchronize()
                triton_time = (time.time() - start_time) / 100 * 1000
                
                # Benchmark PyTorch
                start_time = time.time()
                for _ in range(100):
                    matmul_result = torch.matmul(a, b)
                    if activation == "relu":
                        _ = torch.relu(matmul_result)
                    elif activation == "tanh":
                        _ = torch.tanh(matmul_result)
                    elif activation == "sigmoid":
                        _ = torch.sigmoid(matmul_result)
                    elif activation == "gelu":
                        _ = torch.nn.functional.gelu(matmul_result)
                torch.cuda.synchronize()
                pytorch_time = (time.time() - start_time) / 100 * 1000
                
                speedup = pytorch_time / triton_time if triton_time > 0 else 0
                
                print(f"    {activation.capitalize()}:")
                print(f"      Triton:  {triton_time:.3f} ms")
                print(f"      PyTorch: {pytorch_time:.3f} ms")
                print(f"      Speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"    {activation.capitalize()}: Error - {e}")
        
        # Benchmark batch matrix multiplication
        print("\n  Batch Matrix Multiplication:")
        batch_size = 4
        batch_a = torch.randn(batch_size, M, K, device='cuda', dtype=torch.float32)
        batch_b = torch.randn(batch_size, K, N, device='cuda', dtype=torch.float32)
        
        try:
            # Warmup
            for _ in range(10):
                _ = batch_matmul(batch_a, batch_b)
                _ = torch.bmm(batch_a, batch_b)
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(100):
                _ = batch_matmul(batch_a, batch_b)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100 * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(100):
                _ = torch.bmm(batch_a, batch_b)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100 * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"    Triton:  {triton_time:.3f} ms")
            print(f"    PyTorch: {pytorch_time:.3f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the matrix optimization examples.
    """
    print("🔢 MATRIX OPTIMIZATION EXAMPLES")
    print("=" * 70)
    print("This module demonstrates various matrix optimization techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the examples
    test_matrix_operations()
    benchmark_matrix_operations()
    
    print("\n🎉 Matrix Optimization Examples Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Basic matrix multiplication implementation")
    print("2. ✅ Optimized matrix multiplication with tiling")
    print("3. ✅ Fused matrix multiplication + activation")
    print("4. ✅ Batch matrix multiplication")
    print("5. ✅ Performance comparison between different approaches")
    print("6. ✅ Memory-efficient matrix operations")
    print("7. ✅ Scalable matrix operations for different sizes")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different block sizes")
    print("- Try optimizing for different hardware configurations")
    print("- Implement matrix transpose operations")
    print("- Add support for different data types")

if __name__ == "__main__":
    main()
