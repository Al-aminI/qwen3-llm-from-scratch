"""
🔧 Lesson 6: Kernel Fusion & Performance Tuning

This lesson covers:
1. Kernel fusion techniques and strategies
2. Reducing memory traffic through fusion
3. Kernel launch overhead optimization
4. Performance profiling and analysis
5. Advanced optimization techniques

Prerequisites: Lessons 1-5 (All previous lessons)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🧠 KERNEL FUSION FUNDAMENTALS
# ============================================================================

def explain_kernel_fusion():
    """
    📚 KERNEL FUSION FUNDAMENTALS
    
    Kernel fusion combines multiple operations into a single kernel to reduce
    memory traffic and kernel launch overhead.
    """
    print("🧠 Kernel Fusion Fundamentals:")
    print("=" * 50)
    
    print("""
    🎯 What is Kernel Fusion?
    
    Kernel fusion combines multiple operations into a single kernel:
    - Reduces memory traffic between global memory and registers
    - Minimizes kernel launch overhead
    - Improves cache utilization
    - Increases arithmetic intensity
    
    🚀 Benefits of Kernel Fusion:
    1. Reduced Memory Traffic: Intermediate results stay in registers
    2. Lower Launch Overhead: Fewer kernel launches
    3. Better Cache Utilization: Data stays in cache longer
    4. Higher Arithmetic Intensity: More compute per memory access
    
    📊 Fusion Strategies:
    1. Horizontal Fusion: Combine operations on the same data
    2. Vertical Fusion: Combine operations in a pipeline
    3. Diagonal Fusion: Combine operations with data dependencies
    4. Loop Fusion: Combine operations in loops
    
    🎯 When to Use Fusion:
    1. Operations share data
    2. Memory bandwidth is the bottleneck
    3. Kernel launch overhead is significant
    4. Cache utilization can be improved
    """)

# ============================================================================
# 🎯 BASIC KERNEL FUSION
# ============================================================================

@triton.jit
def fused_add_multiply_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 FUSED ADD-MULTIPLY KERNEL
    
    Fuses addition and multiplication operations:
    output = (a + b) * c
    
    This demonstrates basic horizontal fusion.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Fused operations: (a + b) * c
    # Intermediate result (a + b) stays in registers
    intermediate = a + b
    output = intermediate * c
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_add_multiply(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    🎯 FUSED ADD-MULTIPLY WRAPPER
    
    Wrapper function for fused add-multiply kernel.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda and c.is_cuda, "Input tensors must be on GPU!"
    assert a.shape == b.shape == c.shape, "Input tensors must have the same shape!"
    
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_add_multiply_kernel[grid](
        a, b, c, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🎯 ADVANCED KERNEL FUSION
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
    🎯 FUSED MATRIX MULTIPLICATION + ACTIVATION KERNEL
    
    Fuses matrix multiplication with activation function:
    output = activation(A @ B)
    
    This demonstrates advanced vertical fusion.
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
    c_ptrs = output_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=mask_m[:, None] & mask_n[None, :])

def fused_matmul_activation(a: torch.Tensor, b: torch.Tensor, activation: str = "relu") -> torch.Tensor:
    """
    🎯 FUSED MATRIX MULTIPLICATION + ACTIVATION WRAPPER
    
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
# 🎯 LOOP FUSION
# ============================================================================

@triton.jit
def fused_loop_kernel(
    input_ptr, output_ptr,
    n_elements, num_iterations,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 FUSED LOOP KERNEL
    
    Demonstrates loop fusion by combining multiple operations in a loop.
    This reduces memory traffic and improves performance.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # Fused loop: multiple operations without storing intermediate results
    for i in range(num_iterations):
        # Multiple operations in the loop
        data = data * 2.0 + 1.0
        data = tl.where(data > 0, data, 0.0)  # ReLU
        data = data * 0.5
    
    # Store final result
    tl.store(output_ptr + offsets, data, mask=mask)

def fused_loop(input_tensor: torch.Tensor, num_iterations: int = 10) -> torch.Tensor:
    """
    🎯 FUSED LOOP WRAPPER
    
    Wrapper function for fused loop kernel.
    """
    # Input validation
    assert input_tensor.is_cuda, "Input tensor must be on GPU!"
    
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_loop_kernel[grid](
        input_tensor, output,
        n_elements, num_iterations,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🎯 PERFORMANCE PROFILING
# ============================================================================

def explain_performance_profiling():
    """
    📚 PERFORMANCE PROFILING
    
    Performance profiling is crucial for identifying bottlenecks and
    optimizing kernel performance.
    """
    print("\n🎯 Performance Profiling:")
    print("=" * 50)
    
    print("""
    🧠 Performance Profiling Techniques:
    
    1. Timing Analysis:
       - Measure kernel execution time
       - Compare different implementations
       - Identify performance bottlenecks
    
    2. Memory Bandwidth Analysis:
       - Measure memory bandwidth utilization
       - Identify memory-bound vs compute-bound kernels
       - Optimize memory access patterns
    
    3. Occupancy Analysis:
       - Measure GPU utilization
       - Identify resource limitations
       - Optimize block sizes and grid sizes
    
    4. Cache Analysis:
       - Measure cache hit rates
       - Identify cache misses
       - Optimize data access patterns
    
    🚀 Profiling Tools:
    1. CUDA Profiler (nvprof, nsight)
    2. PyTorch Profiler
    3. Custom timing utilities
    4. Memory bandwidth measurements
    
    📊 Key Metrics:
    1. Kernel execution time
    2. Memory bandwidth utilization
    3. GPU occupancy
    4. Cache hit rate
    5. Arithmetic intensity
    """)

def profile_kernel_performance():
    """
    📊 PROFILE KERNEL PERFORMANCE
    
    Profiles different kernel implementations and compares performance.
    """
    print("\n📊 Profiling Kernel Performance:")
    print("=" * 50)
    
    # Test different kernel implementations
    size = 1024 * 1024  # 1M elements
    
    # Create test data
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Test 1: Separate operations (no fusion)
    print("\n📈 Test 1: Separate Operations (No Fusion)")
    
    # Warmup
    for _ in range(10):
        _ = a + b
        _ = (a + b) * c
    
    torch.cuda.synchronize()
    
    # Time separate operations
    start_time = time.time()
    for _ in range(100):
        intermediate = a + b
        result = intermediate * c
    torch.cuda.synchronize()
    separate_time = (time.time() - start_time) / 100 * 1000
    
    # Test 2: Fused operations
    print("\n📈 Test 2: Fused Operations")
    
    # Warmup
    for _ in range(10):
        _ = fused_add_multiply(a, b, c)
    
    torch.cuda.synchronize()
    
    # Time fused operations
    start_time = time.time()
    for _ in range(100):
        _ = fused_add_multiply(a, b, c)
    torch.cuda.synchronize()
    fused_time = (time.time() - start_time) / 100 * 1000
    
    # Calculate speedup
    speedup = separate_time / fused_time if fused_time > 0 else 0
    
    print(f"  Separate Operations: {separate_time:.3f} ms")
    print(f"  Fused Operations:    {fused_time:.3f} ms")
    print(f"  Speedup:             {speedup:.2f}x")
    
    # Test 3: Matrix multiplication + activation
    print("\n📈 Test 3: Matrix Multiplication + Activation")
    
    M, K, N = 512, 256, 512
    a_mat = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b_mat = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(a_mat, b_mat)
        _ = torch.relu(torch.matmul(a_mat, b_mat))
        _ = fused_matmul_activation(a_mat, b_mat, "relu")
    
    torch.cuda.synchronize()
    
    # Time separate operations
    start_time = time.time()
    for _ in range(100):
        matmul_result = torch.matmul(a_mat, b_mat)
        activation_result = torch.relu(matmul_result)
    torch.cuda.synchronize()
    separate_matmul_time = (time.time() - start_time) / 100 * 1000
    
    # Time fused operations
    start_time = time.time()
    for _ in range(100):
        _ = fused_matmul_activation(a_mat, b_mat, "relu")
    torch.cuda.synchronize()
    fused_matmul_time = (time.time() - start_time) / 100 * 1000
    
    # Calculate speedup
    matmul_speedup = separate_matmul_time / fused_matmul_time if fused_matmul_time > 0 else 0
    
    print(f"  Separate MatMul+Act: {separate_matmul_time:.3f} ms")
    print(f"  Fused MatMul+Act:    {fused_matmul_time:.3f} ms")
    print(f"  Speedup:             {matmul_speedup:.2f}x")

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_kernel_fusion():
    """
    🧪 TEST KERNEL FUSION
    
    Tests various kernel fusion techniques and validates correctness.
    """
    print("\n🧪 Testing Kernel Fusion:")
    print("=" * 50)
    
    # Test fused add-multiply
    print("\n📊 Test: Fused Add-Multiply")
    size = 1024
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.randn(size, device='cuda', dtype=torch.float32)
    
    triton_result = fused_add_multiply(a, b, c)
    pytorch_result = (a + b) * c
    
    is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test fused matrix multiplication + activation
    print("\n📊 Test: Fused Matrix Multiplication + Activation")
    M, K, N = 256, 128, 192
    a_mat = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b_mat = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Test ReLU activation
    triton_relu = fused_matmul_activation(a_mat, b_mat, "relu")
    pytorch_relu = torch.relu(torch.matmul(a_mat, b_mat))
    
    is_correct_relu = torch.allclose(triton_relu, pytorch_relu, rtol=1e-3, atol=1e-3)
    print(f"  ReLU: {'✅ PASS' if is_correct_relu else '❌ FAIL'}")
    
    # Test Tanh activation
    triton_tanh = fused_matmul_activation(a_mat, b_mat, "tanh")
    pytorch_tanh = torch.tanh(torch.matmul(a_mat, b_mat))
    
    is_correct_tanh = torch.allclose(triton_tanh, pytorch_tanh, rtol=1e-3, atol=1e-3)
    print(f"  Tanh: {'✅ PASS' if is_correct_tanh else '❌ FAIL'}")
    
    # Test fused loop
    print("\n📊 Test: Fused Loop")
    size = 1024
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    
    triton_loop = fused_loop(x, num_iterations=5)
    
    # PyTorch reference
    pytorch_loop = x
    for i in range(5):
        pytorch_loop = pytorch_loop * 2.0 + 1.0
        pytorch_loop = torch.relu(pytorch_loop)
        pytorch_loop = pytorch_loop * 0.5
    
    is_correct_loop = torch.allclose(triton_loop, pytorch_loop, rtol=1e-4, atol=1e-4)
    print(f"  Result: {'✅ PASS' if is_correct_loop else '❌ FAIL'}")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 6 tutorial.
    """
    print("🔧 LESSON 6: KERNEL FUSION & PERFORMANCE TUNING")
    print("=" * 70)
    print("This lesson covers kernel fusion and performance optimization techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_kernel_fusion()
    explain_performance_profiling()
    
    test_kernel_fusion()
    profile_kernel_performance()
    
    print("\n🎉 Lesson 6 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding kernel fusion techniques")
    print("2. ✅ Implementing horizontal and vertical fusion")
    print("3. ✅ Reducing memory traffic through fusion")
    print("4. ✅ Performance profiling and analysis")
    print("5. ✅ Advanced optimization techniques")
    print("6. ✅ Real-world performance improvements")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different fusion strategies")
    print("- Profile your own kernels for optimization opportunities")
    print("- Move on to Advanced Level: Attention Mechanisms & FlashAttention")

if __name__ == "__main__":
    main()
