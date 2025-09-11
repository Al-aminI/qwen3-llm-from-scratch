"""
🎯 Lesson 2: Memory Management & Data Types

This lesson covers:
1. Understanding memory coalescing and access patterns
2. Working with different data types (float32, float16, int32)
3. Memory bandwidth optimization
4. Bank conflicts and how to avoid them
5. Stride patterns and non-contiguous tensors

Prerequisites: Lesson 1 (GPU Fundamentals & Triton Basics)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🧠 MEMORY COALESCING FUNDAMENTALS
# ============================================================================

def explain_memory_coalescing():
    """
    📚 MEMORY COALESCING EXPLANATION
    
    Memory coalescing is crucial for achieving high memory bandwidth:
    
    ✅ Coalesced Access (Good):
    - Threads access consecutive memory locations
    - GPU can combine multiple requests into one
    - Achieves near-peak memory bandwidth
    
    ❌ Non-Coalesced Access (Bad):
    - Threads access scattered memory locations
    - Each access requires separate memory transaction
    - Poor memory bandwidth utilization
    """
    print("🧠 Memory Coalescing Fundamentals:")
    print("=" * 50)
    
    print("""
    🎯 What is Memory Coalescing?
    
    Memory coalescing is the GPU's ability to combine multiple memory requests
    from different threads into a single, more efficient memory transaction.
    
    📊 Example - Coalesced vs Non-Coalesced:
    
    ✅ Coalesced (Good):
    Thread 0: loads memory[0]
    Thread 1: loads memory[1]  
    Thread 2: loads memory[2]
    Thread 3: loads memory[3]
    → GPU combines into 1 transaction for memory[0:4]
    
    ❌ Non-Coalesced (Bad):
    Thread 0: loads memory[0]
    Thread 1: loads memory[100]
    Thread 2: loads memory[200]
    Thread 3: loads memory[300]
    → GPU needs 4 separate transactions
    
    🚀 Key Principles:
    1. Access consecutive memory locations when possible
    2. Use appropriate data types for your workload
    3. Minimize memory transactions
    4. Consider memory alignment
    """)

# ============================================================================
# 🎯 COALESCED MEMORY ACCESS KERNEL
# ============================================================================

@triton.jit
def coalesced_access_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 COALESCED MEMORY ACCESS KERNEL
    
    Demonstrates proper memory coalescing by accessing consecutive elements.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced access: consecutive memory locations
    input_data = tl.load(input_ptr + offsets * stride, mask=mask)
    
    # Simple operation (multiply by 2)
    output_data = input_data * 2.0
    
    # Coalesced store: consecutive memory locations
    tl.store(output_ptr + offsets * stride, output_data, mask=mask)

@triton.jit
def non_coalesced_access_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 NON-COALESCED MEMORY ACCESS KERNEL
    
    Demonstrates poor memory access patterns (for comparison).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Non-coalesced access: scattered memory locations
    # Each thread accesses memory[thread_id * 100] instead of memory[thread_id]
    scattered_offsets = offsets * 100
    scattered_mask = scattered_offsets < n_elements
    
    input_data = tl.load(input_ptr + scattered_offsets * stride, mask=scattered_mask)
    output_data = input_data * 2.0
    tl.store(output_ptr + scattered_offsets * stride, output_data, mask=scattered_mask)

def benchmark_memory_access():
    """
    📊 BENCHMARK COALESCED VS NON-COALESCED ACCESS
    
    Compares performance between coalesced and non-coalesced memory access.
    """
    print("\n📊 Benchmarking Memory Access Patterns:")
    print("=" * 50)
    
    sizes = [1024, 4096, 16384, 65536]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        # Create test data
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        output_coalesced = torch.empty_like(x)
        output_non_coalesced = torch.empty_like(x)
        
        # Warmup
        for _ in range(10):
            coalesced_access_kernel[(triton.cdiv(size, 128),)](
                x, output_coalesced, size, 1, BLOCK_SIZE=128
            )
            non_coalesced_access_kernel[(triton.cdiv(size, 128),)](
                x, output_non_coalesced, size, 1, BLOCK_SIZE=128
            )
        
        torch.cuda.synchronize()
        
        # Benchmark coalesced access
        start_time = time.time()
        for _ in range(100):
            coalesced_access_kernel[(triton.cdiv(size, 128),)](
                x, output_coalesced, size, 1, BLOCK_SIZE=128
            )
        torch.cuda.synchronize()
        coalesced_time = (time.time() - start_time) / 100 * 1000
        
        # Benchmark non-coalesced access
        start_time = time.time()
        for _ in range(100):
            non_coalesced_access_kernel[(triton.cdiv(size, 128),)](
                x, output_non_coalesced, size, 1, BLOCK_SIZE=128
            )
        torch.cuda.synchronize()
        non_coalesced_time = (time.time() - start_time) / 100 * 1000
        
        speedup = non_coalesced_time / coalesced_time if coalesced_time > 0 else 0
        
        print(f"  Coalesced:    {coalesced_time:.3f} ms")
        print(f"  Non-Coalesced: {non_coalesced_time:.3f} ms")
        print(f"  Speedup:      {speedup:.2f}x")

# ============================================================================
# 🎯 DATA TYPES AND PRECISION
# ============================================================================

def explain_data_types():
    """
    📚 DATA TYPES IN TRITON
    
    Different data types have different characteristics:
    - Memory usage
    - Precision
    - Performance
    - Hardware support
    """
    print("\n🎯 Data Types in Triton:")
    print("=" * 50)
    
    data_types = {
        "float32": {
            "size": "4 bytes",
            "precision": "~7 decimal digits",
            "range": "±3.4 × 10³⁸",
            "use_case": "General purpose, high precision"
        },
        "float16": {
            "size": "2 bytes", 
            "precision": "~3 decimal digits",
            "range": "±6.5 × 10⁴",
            "use_case": "Memory-constrained, acceptable precision loss"
        },
        "bfloat16": {
            "size": "2 bytes",
            "precision": "~3 decimal digits", 
            "range": "±3.4 × 10³⁸",
            "use_case": "Training, better range than float16"
        },
        "int32": {
            "size": "4 bytes",
            "precision": "Exact integers",
            "range": "±2.1 × 10⁹",
            "use_case": "Indices, counts, discrete values"
        },
        "int8": {
            "size": "1 byte",
            "precision": "Exact integers",
            "range": "±128",
            "use_case": "Quantization, memory-constrained"
        }
    }
    
    for dtype, info in data_types.items():
        print(f"\n{dtype}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

@triton.jit
def data_type_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 DATA TYPE DEMONSTRATION KERNEL
    
    Shows how to work with different data types in Triton.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data (Triton will handle the data type automatically)
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Perform operation
    # Note: Triton handles type promotion automatically
    output_data = input_data * 2.0 + 1.0
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)

def test_data_types():
    """
    🧪 TEST DIFFERENT DATA TYPES
    
    Tests the kernel with different data types and compares performance.
    """
    print("\n🧪 Testing Different Data Types:")
    print("=" * 50)
    
    data_types = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.int32, "int32"),
    ]
    
    size = 1024
    
    for dtype, name in data_types:
        print(f"\n📊 Testing {name}:")
        
        # Create test data
        if dtype in [torch.int32]:
            x = torch.randint(0, 100, (size,), device='cuda', dtype=dtype)
        else:
            x = torch.randn(size, device='cuda', dtype=dtype)
        
        output = torch.empty_like(x)
        
        # Test kernel
        data_type_kernel[(triton.cdiv(size, 128),)](
            x, output, size, BLOCK_SIZE=128
        )
        
        # Verify correctness
        if dtype in [torch.int32]:
            expected = x * 2 + 1
        else:
            expected = x * 2.0 + 1.0
        
        is_correct = torch.allclose(output.float(), expected.float(), rtol=1e-3)
        
        if is_correct:
            print(f"  ✅ PASS: {name} works correctly")
        else:
            print(f"  ❌ FAIL: {name} has incorrect results")

# ============================================================================
# 🎯 STRIDE PATTERNS AND NON-CONTIGUOUS TENSORS
# ============================================================================

def explain_strides():
    """
    📚 UNDERSTANDING STRIDES
    
    Strides determine how to access elements in multi-dimensional tensors:
    - stride[i] = number of elements to skip to get to next element in dimension i
    - Contiguous tensors have predictable stride patterns
    - Non-contiguous tensors require careful stride handling
    """
    print("\n🎯 Understanding Strides:")
    print("=" * 50)
    
    print("""
    📊 Stride Examples:
    
    1D Tensor [0, 1, 2, 3, 4]:
    - stride = [1] (move 1 element to get next)
    
    2D Tensor [[0, 1, 2], [3, 4, 5]]:
    - stride = [3, 1] (move 3 elements for next row, 1 for next column)
    
    3D Tensor shape (2, 3, 4):
    - stride = [12, 4, 1] (move 12 for next 2D slice, 4 for next row, 1 for next column)
    
    🔍 Non-Contiguous Tensors:
    - Transposed tensors have different stride patterns
    - Sliced tensors may not be contiguous
    - Reshaped tensors may not be contiguous
    """)

@triton.jit
def stride_aware_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 STRIDE-AWARE KERNEL
    
    Demonstrates how to handle different stride patterns.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Use stride to access elements
    input_data = tl.load(input_ptr + offsets * input_stride, mask=mask)
    output_data = input_data * 2.0
    tl.store(output_ptr + offsets * output_stride, output_data, mask=mask)

def test_strides():
    """
    🧪 TEST DIFFERENT STRIDE PATTERNS
    
    Tests the kernel with different stride patterns.
    """
    print("\n🧪 Testing Stride Patterns:")
    print("=" * 50)
    
    # Test 1: Contiguous tensor
    print("\n📊 Test 1: Contiguous Tensor")
    x = torch.randn(1024, device='cuda')
    output = torch.empty_like(x)
    
    stride_aware_kernel[(triton.cdiv(1024, 128),)](
        x, output, 1024, 1, 1, BLOCK_SIZE=128
    )
    
    expected = x * 2.0
    is_correct = torch.allclose(output, expected)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # Test 2: Transposed tensor
    print("\n📊 Test 2: Transposed Tensor")
    x = torch.randn(64, 16, device='cuda')
    x_t = x.t()  # Transpose
    output = torch.empty_like(x_t)
    
    stride_aware_kernel[(triton.cdiv(x_t.numel(), 128),)](
        x_t, output, x_t.numel(), x_t.stride(0), output.stride(0), BLOCK_SIZE=128
    )
    
    expected = x_t * 2.0
    is_correct = torch.allclose(output, expected)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    print(f"  Input stride: {x_t.stride()}")
    print(f"  Output stride: {output.stride()}")

# ============================================================================
# 🎯 MEMORY BANDWIDTH OPTIMIZATION
# ============================================================================

def explain_memory_bandwidth():
    """
    📚 MEMORY BANDWIDTH OPTIMIZATION
    
    Key strategies for maximizing memory bandwidth:
    1. Use appropriate data types
    2. Ensure memory coalescing
    3. Minimize memory transactions
    4. Use shared memory when beneficial
    5. Consider memory alignment
    """
    print("\n🎯 Memory Bandwidth Optimization:")
    print("=" * 50)
    
    print("""
    🚀 Optimization Strategies:
    
    1. Data Type Selection:
       - Use float16 for memory-constrained applications
       - Use int8 for quantization
       - Use float32 only when precision is critical
    
    2. Memory Coalescing:
       - Access consecutive memory locations
       - Use appropriate block sizes
       - Consider memory alignment
    
    3. Minimize Transactions:
       - Combine multiple operations
       - Use shared memory for frequently accessed data
       - Avoid redundant memory accesses
    
    4. Memory Alignment:
       - Align data to cache line boundaries
       - Use power-of-2 sizes when possible
       - Consider hardware-specific optimizations
    """)

@triton.jit
def bandwidth_test_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 BANDWIDTH TEST KERNEL
    
    Simple kernel to test memory bandwidth.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Simple memory copy (read + write)
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)

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
            bandwidth_test_kernel[(triton.cdiv(size, 128),)](
                x, output, size, BLOCK_SIZE=128
            )
        
        torch.cuda.synchronize()
        
        # Measure time
        start_time = time.time()
        for _ in range(100):
            bandwidth_test_kernel[(triton.cdiv(size, 128),)](
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
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 2 tutorial.
    """
    print("🎯 LESSON 2: MEMORY MANAGEMENT & DATA TYPES")
    print("=" * 70)
    print("This lesson covers memory optimization techniques in Triton.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_memory_coalescing()
    benchmark_memory_access()
    
    explain_data_types()
    test_data_types()
    
    explain_strides()
    test_strides()
    
    explain_memory_bandwidth()
    measure_memory_bandwidth()
    
    print("\n🎉 Lesson 2 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding memory coalescing and access patterns")
    print("2. ✅ Working with different data types effectively")
    print("3. ✅ Handling stride patterns and non-contiguous tensors")
    print("4. ✅ Optimizing memory bandwidth")
    print("5. ✅ Measuring and benchmarking memory performance")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different block sizes and data types")
    print("- Try optimizing memory access patterns")
    print("- Move on to Lesson 3: Basic Operations & Kernels")

if __name__ == "__main__":
    main()
