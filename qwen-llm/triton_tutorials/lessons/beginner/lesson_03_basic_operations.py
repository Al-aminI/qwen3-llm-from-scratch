"""
🎯 Lesson 3: Basic Operations & Kernels

This lesson covers:
1. Element-wise operations (add, multiply, divide, etc.)
2. Reduction operations (sum, max, min, etc.)
3. Broadcasting and masking techniques
4. Error handling and debugging
5. Performance optimization for basic operations

Prerequisites: Lesson 1 (GPU Fundamentals) and Lesson 2 (Memory Management)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🎯 ELEMENT-WISE OPERATIONS
# ============================================================================

def explain_element_wise_operations():
    """
    📚 ELEMENT-WISE OPERATIONS
    
    Element-wise operations are fundamental building blocks for more complex kernels.
    They operate on corresponding elements of tensors.
    """
    print("🎯 Element-wise Operations:")
    print("=" * 50)
    
    print("""
    🧠 What are Element-wise Operations?
    
    Element-wise operations perform the same operation on corresponding elements
    of input tensors. For example:
    
    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]
    
    A + B = [6, 8, 10, 12]  # Element-wise addition
    A * B = [5, 12, 21, 32] # Element-wise multiplication
    
    🚀 Common Element-wise Operations:
    - Addition: a + b
    - Subtraction: a - b  
    - Multiplication: a * b
    - Division: a / b
    - Power: a ** b
    - Comparison: a > b, a < b, a == b
    - Logical: a & b, a | b, ~a
    """)

@triton.jit
def element_wise_add_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 ELEMENT-WISE ADDITION KERNEL
    
    Adds corresponding elements of two tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Element-wise addition
    output = a + b
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def element_wise_multiply_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 ELEMENT-WISE MULTIPLICATION KERNEL
    
    Multiplies corresponding elements of two tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Element-wise multiplication
    output = a * b
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def element_wise_activation_kernel(
    input_ptr, output_ptr,
    n_elements,
    activation_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 ELEMENT-WISE ACTIVATION KERNEL
    
    Applies activation functions element-wise.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply activation function
    if activation_type == 0:  # ReLU
        output = tl.where(x > 0, x, 0.0)
    elif activation_type == 1:  # Sigmoid
        output = 1.0 / (1.0 + tl.exp(-x))
    elif activation_type == 2:  # Tanh
        output = tl.tanh(x)
    elif activation_type == 3:  # GELU (approximation)
        output = 0.5 * x * (1.0 + tl.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    else:
        output = x  # Identity
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def test_element_wise_operations():
    """
    🧪 TEST ELEMENT-WISE OPERATIONS
    
    Tests various element-wise operations and validates correctness.
    """
    print("\n🧪 Testing Element-wise Operations:")
    print("=" * 50)
    
    size = 1024
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Test addition
    print("\n📊 Test: Element-wise Addition")
    output_add = torch.empty_like(a)
    element_wise_add_kernel[(triton.cdiv(size, 128),)](
        a, b, output_add, size, BLOCK_SIZE=128
    )
    expected_add = a + b
    is_correct = torch.allclose(output_add, expected_add, rtol=1e-5)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test multiplication
    print("\n📊 Test: Element-wise Multiplication")
    output_mul = torch.empty_like(a)
    element_wise_multiply_kernel[(triton.cdiv(size, 128),)](
        a, b, output_mul, size, BLOCK_SIZE=128
    )
    expected_mul = a * b
    is_correct = torch.allclose(output_mul, expected_mul, rtol=1e-5)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test activations
    print("\n📊 Test: Activation Functions")
    activations = [
        (0, "ReLU", torch.relu),
        (1, "Sigmoid", torch.sigmoid),
        (2, "Tanh", torch.tanh),
        (3, "GELU", torch.nn.functional.gelu),
    ]
    
    for act_id, name, torch_func in activations:
        output_act = torch.empty_like(a)
        element_wise_activation_kernel[(triton.cdiv(size, 128),)](
            a, output_act, size, activation_type=act_id, BLOCK_SIZE=128
        )
        expected_act = torch_func(a)
        is_correct = torch.allclose(output_act, expected_act, rtol=1e-3, atol=1e-3)
        print(f"  {name}: {'✅ PASS' if is_correct else '❌ FAIL'}")

# ============================================================================
# 🎯 REDUCTION OPERATIONS
# ============================================================================

def explain_reduction_operations():
    """
    📚 REDUCTION OPERATIONS
    
    Reduction operations combine multiple elements into a single result.
    They are more complex than element-wise operations because they require
    communication between threads.
    """
    print("\n🎯 Reduction Operations:")
    print("=" * 50)
    
    print("""
    🧠 What are Reduction Operations?
    
    Reduction operations combine multiple elements into a single result:
    
    Sum: [1, 2, 3, 4] → 10
    Max: [1, 2, 3, 4] → 4
    Min: [1, 2, 3, 4] → 1
    Mean: [1, 2, 3, 4] → 2.5
    
    🚀 Challenges in Parallel Reduction:
    1. Thread coordination
    2. Memory access patterns
    3. Load balancing
    4. Numerical stability
    
    📊 Reduction Strategies:
    1. Sequential: Simple but slow
    2. Tree Reduction: Log(n) steps, good for small arrays
    3. Block Reduction: Combine with global reduction
    4. Warp Reduction: Use hardware primitives
    """)

@triton.jit
def sum_reduction_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 SUM REDUCTION KERNEL
    
    Computes the sum of all elements in the input tensor.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Block-level reduction
    block_sum = tl.sum(data, axis=0)
    
    # Store block result
    if pid == 0:  # Only the first block stores the result
        tl.store(output_ptr, block_sum)

@triton.jit
def max_reduction_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 MAX REDUCTION KERNEL
    
    Finds the maximum value in the input tensor.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    data = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Block-level reduction
    block_max = tl.max(data, axis=0)
    
    # Store block result
    if pid == 0:  # Only the first block stores the result
        tl.store(output_ptr, block_max)

@triton.jit
def mean_reduction_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 MEAN REDUCTION KERNEL
    
    Computes the mean of all elements in the input tensor.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Block-level reduction
    block_sum = tl.sum(data, axis=0)
    block_count = tl.sum(mask.to(tl.float32), axis=0)
    
    # Store block results
    if pid == 0:  # Only the first block stores the result
        tl.store(output_ptr, block_sum / block_count)

def test_reduction_operations():
    """
    🧪 TEST REDUCTION OPERATIONS
    
    Tests various reduction operations and validates correctness.
    """
    print("\n🧪 Testing Reduction Operations:")
    print("=" * 50)
    
    size = 1024
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Test sum reduction
    print("\n📊 Test: Sum Reduction")
    output_sum = torch.zeros(1, device='cuda', dtype=torch.float32)
    sum_reduction_kernel[(triton.cdiv(size, 128),)](
        x, output_sum, size, BLOCK_SIZE=128
    )
    expected_sum = torch.sum(x)
    is_correct = torch.allclose(output_sum, expected_sum, rtol=1e-4)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    print(f"  Triton: {output_sum.item():.6f}")
    print(f"  PyTorch: {expected_sum.item():.6f}")
    
    # Test max reduction
    print("\n📊 Test: Max Reduction")
    output_max = torch.zeros(1, device='cuda', dtype=torch.float32)
    max_reduction_kernel[(triton.cdiv(size, 128),)](
        x, output_max, size, BLOCK_SIZE=128
    )
    expected_max = torch.max(x)
    is_correct = torch.allclose(output_max, expected_max, rtol=1e-4)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    print(f"  Triton: {output_max.item():.6f}")
    print(f"  PyTorch: {expected_max.item():.6f}")
    
    # Test mean reduction
    print("\n📊 Test: Mean Reduction")
    output_mean = torch.zeros(1, device='cuda', dtype=torch.float32)
    mean_reduction_kernel[(triton.cdiv(size, 128),)](
        x, output_mean, size, BLOCK_SIZE=128
    )
    expected_mean = torch.mean(x)
    is_correct = torch.allclose(output_mean, expected_mean, rtol=1e-4)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    print(f"  Triton: {output_mean.item():.6f}")
    print(f"  PyTorch: {expected_mean.item():.6f}")

# ============================================================================
# 🎯 BROADCASTING AND MASKING
# ============================================================================

def explain_broadcasting():
    """
    📚 BROADCASTING IN TRITON
    
    Broadcasting allows operations between tensors of different shapes.
    It's a powerful feature that enables efficient operations.
    """
    print("\n🎯 Broadcasting in Triton:")
    print("=" * 50)
    
    print("""
    🧠 What is Broadcasting?
    
    Broadcasting allows operations between tensors of different shapes:
    
    A: [3, 1]    B: [1, 4]    A + B: [3, 4]
    [1]          [2]          [3]
    [2]    +     [3]    =     [4]
    [3]          [4]          [5]
    
    🚀 Broadcasting Rules:
    1. Dimensions are aligned from the right
    2. Size 1 dimensions are broadcast to match
    3. Missing dimensions are treated as size 1
    
    📊 Examples:
    - [5, 1] + [1, 3] → [5, 3]
    - [4] + [1, 3] → [4, 3]
    - [2, 1, 3] + [1, 4, 1] → [2, 4, 3]
    """)

@triton.jit
def broadcasting_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    🎯 BROADCASTING KERNEL
    
    Demonstrates broadcasting between tensors of different shapes.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load data with broadcasting
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    
    # Perform operation
    output = a + b
    
    # Store result
    c_ptrs = output_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=mask)

def test_broadcasting():
    """
    🧪 TEST BROADCASTING
    
    Tests broadcasting operations and validates correctness.
    """
    print("\n🧪 Testing Broadcasting:")
    print("=" * 50)
    
    # Test case 1: [3, 1] + [1, 4] = [3, 4]
    print("\n📊 Test: [3, 1] + [1, 4] = [3, 4]")
    a = torch.randn(3, 1, device='cuda', dtype=torch.float32)
    b = torch.randn(1, 4, device='cuda', dtype=torch.float32)
    output = torch.empty(3, 4, device='cuda', dtype=torch.float32)
    
    grid = (triton.cdiv(3, 32), triton.cdiv(4, 32))
    broadcasting_kernel[grid](
        a, b, output,
        3, 4,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    
    expected = a + b
    is_correct = torch.allclose(output, expected, rtol=1e-5)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")

# ============================================================================
# 🎯 ERROR HANDLING AND DEBUGGING
# ============================================================================

def explain_error_handling():
    """
    📚 ERROR HANDLING AND DEBUGGING
    
    Debugging Triton kernels can be challenging. Here are some strategies:
    """
    print("\n🎯 Error Handling and Debugging:")
    print("=" * 50)
    
    print("""
    🐛 Common Debugging Strategies:
    
    1. Validate Inputs:
       - Check tensor shapes and dtypes
       - Verify memory layout (contiguous, strides)
       - Ensure proper device placement
    
    2. Use Masks:
       - Always use masks for boundary conditions
       - Test with non-power-of-2 sizes
       - Verify mask logic
    
    3. Numerical Validation:
       - Compare with PyTorch reference
       - Check for NaN and Inf values
       - Verify numerical precision
    
    4. Performance Profiling:
       - Use CUDA profilers
       - Measure memory bandwidth
       - Check kernel launch overhead
    
    5. Gradual Development:
       - Start with simple kernels
       - Add complexity incrementally
       - Test at each step
    """)

@triton.jit
def debug_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 DEBUG KERNEL
    
    Demonstrates proper error handling and debugging techniques.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with proper masking
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform operation with error checking
    # Check for potential overflow/underflow
    output = tl.where(tl.abs(data) > 1e6, 0.0, data * 2.0)
    
    # Store result with masking
    tl.store(output_ptr + offsets, output, mask=mask)

def test_error_handling():
    """
    🧪 TEST ERROR HANDLING
    
    Tests error handling and edge cases.
    """
    print("\n🧪 Testing Error Handling:")
    print("=" * 50)
    
    # Test with edge cases
    test_cases = [
        (1, "Single element"),
        (127, "Non-power-of-2 size"),
        (1000, "Large size"),
    ]
    
    for size, description in test_cases:
        print(f"\n📊 Test: {description} (size={size})")
        
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)
        
        debug_kernel[(triton.cdiv(size, 128),)](
            x, output, size, BLOCK_SIZE=128
        )
        
        expected = torch.where(torch.abs(x) > 1e6, torch.zeros_like(x), x * 2.0)
        is_correct = torch.allclose(output, expected, rtol=1e-5)
        print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")

# ============================================================================
# 🎯 PERFORMANCE OPTIMIZATION
# ============================================================================

def explain_performance_optimization():
    """
    📚 PERFORMANCE OPTIMIZATION
    
    Key strategies for optimizing basic operations:
    """
    print("\n🎯 Performance Optimization:")
    print("=" * 50)
    
    print("""
    🚀 Optimization Strategies:
    
    1. Block Size Tuning:
       - Test different block sizes (32, 64, 128, 256)
       - Consider memory bandwidth vs occupancy
       - Use autotuning for optimal performance
    
    2. Memory Access Patterns:
       - Ensure coalesced memory access
       - Minimize memory transactions
       - Use appropriate data types
    
    3. Kernel Fusion:
       - Combine multiple operations
       - Reduce memory traffic
       - Minimize kernel launch overhead
    
    4. Numerical Precision:
       - Use float16 when precision allows
       - Consider mixed precision
       - Optimize for target hardware
    
    5. Load Balancing:
       - Ensure even work distribution
       - Handle boundary conditions efficiently
       - Use appropriate grid sizes
    """)

def benchmark_basic_operations():
    """
    📊 BENCHMARK BASIC OPERATIONS
    
    Compares performance between Triton and PyTorch for basic operations.
    """
    print("\n📊 Benchmarking Basic Operations:")
    print("=" * 50)
    
    sizes = [1024, 4096, 16384, 65536]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = a + b
            _ = a * b
            _ = torch.sum(a)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(100):
            _ = a + b
        torch.cuda.synchronize()
        pytorch_add_time = (time.time() - start_time) / 100 * 1000
        
        start_time = time.time()
        for _ in range(100):
            _ = a * b
        torch.cuda.synchronize()
        pytorch_mul_time = (time.time() - start_time) / 100 * 1000
        
        start_time = time.time()
        for _ in range(100):
            _ = torch.sum(a)
        torch.cuda.synchronize()
        pytorch_sum_time = (time.time() - start_time) / 100 * 1000
        
        # Benchmark Triton
        output_add = torch.empty_like(a)
        output_mul = torch.empty_like(a)
        output_sum = torch.zeros(1, device='cuda', dtype=torch.float32)
        
        start_time = time.time()
        for _ in range(100):
            element_wise_add_kernel[(triton.cdiv(size, 128),)](
                a, b, output_add, size, BLOCK_SIZE=128
            )
        torch.cuda.synchronize()
        triton_add_time = (time.time() - start_time) / 100 * 1000
        
        start_time = time.time()
        for _ in range(100):
            element_wise_multiply_kernel[(triton.cdiv(size, 128),)](
                a, b, output_mul, size, BLOCK_SIZE=128
            )
        torch.cuda.synchronize()
        triton_mul_time = (time.time() - start_time) / 100 * 1000
        
        start_time = time.time()
        for _ in range(100):
            sum_reduction_kernel[(triton.cdiv(size, 128),)](
                a, output_sum, size, BLOCK_SIZE=128
            )
        torch.cuda.synchronize()
        triton_sum_time = (time.time() - start_time) / 100 * 1000
        
        print(f"  Addition:")
        print(f"    PyTorch: {pytorch_add_time:.3f} ms")
        print(f"    Triton:  {triton_add_time:.3f} ms")
        print(f"    Speedup: {pytorch_add_time / triton_add_time:.2f}x")
        
        print(f"  Multiplication:")
        print(f"    PyTorch: {pytorch_mul_time:.3f} ms")
        print(f"    Triton:  {triton_mul_time:.3f} ms")
        print(f"    Speedup: {pytorch_mul_time / triton_mul_time:.2f}x")
        
        print(f"  Sum Reduction:")
        print(f"    PyTorch: {pytorch_sum_time:.3f} ms")
        print(f"    Triton:  {triton_sum_time:.3f} ms")
        print(f"    Speedup: {pytorch_sum_time / triton_sum_time:.2f}x")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 3 tutorial.
    """
    print("🎯 LESSON 3: BASIC OPERATIONS & KERNELS")
    print("=" * 70)
    print("This lesson covers basic operations and kernel development in Triton.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_element_wise_operations()
    test_element_wise_operations()
    
    explain_reduction_operations()
    test_reduction_operations()
    
    explain_broadcasting()
    test_broadcasting()
    
    explain_error_handling()
    test_error_handling()
    
    explain_performance_optimization()
    benchmark_basic_operations()
    
    print("\n🎉 Lesson 3 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding element-wise operations")
    print("2. ✅ Implementing reduction operations")
    print("3. ✅ Working with broadcasting and masking")
    print("4. ✅ Error handling and debugging techniques")
    print("5. ✅ Performance optimization strategies")
    print("6. ✅ Benchmarking and validation")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different block sizes and operations")
    print("- Try implementing more complex reduction operations")
    print("- Move on to Intermediate Level: Matrix Operations")

if __name__ == "__main__":
    main()
