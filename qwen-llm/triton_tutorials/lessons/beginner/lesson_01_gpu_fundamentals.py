"""
🎯 Lesson 1: GPU Fundamentals & Triton Basics

This lesson covers:
1. Understanding GPU architecture and memory hierarchy
2. Introduction to Triton language
3. Your first kernel: Vector Addition
4. Understanding program IDs, blocks, and grids
5. Memory access patterns and coalescing

Prerequisites: Basic Python, familiarity with PyTorch tensors
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional

# ============================================================================
# 🧠 GPU FUNDAMENTALS
# ============================================================================

def explain_gpu_architecture():
    """
    📚 EXPLAINING GPU ARCHITECTURE
    
    GPUs are massively parallel processors designed for:
    - High throughput (many operations per second)
    - High memory bandwidth (fast data access)
    - Parallel execution (thousands of threads)
    
    Key Components:
    1. Streaming Multiprocessors (SMs) - Compute units
    2. CUDA Cores - Individual processing units
    3. Memory Hierarchy - Different types of memory
    4. Warps - Groups of 32 threads that execute together
    """
    print("🧠 GPU Architecture Overview:")
    print("=" * 50)
    
    # Get GPU information
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        print(f"GPU: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"CUDA Cores: {props.multi_processor_count * 64}")  # Approximate
        print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"Memory Bandwidth: ~{props.total_memory / 1024**3 * 100:.0f} GB/s")
    else:
        print("❌ CUDA not available. Please use a GPU-enabled environment.")
    
    print("\n📊 Memory Hierarchy (Fastest to Slowest):")
    print("1. Registers (per-thread, fastest)")
    print("2. Shared Memory (per-block, very fast)")
    print("3. L1 Cache (per-SM, fast)")
    print("4. L2 Cache (global, medium)")
    print("5. Global Memory (slowest, largest)")

def explain_memory_hierarchy():
    """
    📚 MEMORY HIERARCHY EXPLANATION
    
    Understanding memory hierarchy is crucial for writing efficient kernels:
    
    1. Registers: Fastest, private to each thread
    2. Shared Memory: Fast, shared among threads in a block
    3. L1 Cache: Fast, shared among threads in an SM
    4. L2 Cache: Medium speed, shared across SMs
    5. Global Memory: Slowest, accessible by all threads
    
    Key Principles:
    - Keep frequently accessed data in faster memory
    - Minimize global memory accesses
    - Use coalesced memory access patterns
    """
    print("\n💾 Memory Hierarchy Details:")
    print("=" * 50)
    
    memory_types = {
        "Registers": {
            "size": "~255 per thread",
            "speed": "1 cycle",
            "scope": "per-thread",
            "use_case": "local variables, loop counters"
        },
        "Shared Memory": {
            "size": "~48KB per block",
            "speed": "1-32 cycles",
            "scope": "per-block",
            "use_case": "inter-thread communication, caching"
        },
        "L1 Cache": {
            "size": "~128KB per SM",
            "speed": "20-200 cycles",
            "scope": "per-SM",
            "use_case": "automatic caching of global memory"
        },
        "L2 Cache": {
            "size": "~6MB total",
            "speed": "200-400 cycles",
            "scope": "global",
            "use_case": "shared cache across all SMs"
        },
        "Global Memory": {
            "size": "GBs",
            "speed": "400-800 cycles",
            "scope": "global",
            "use_case": "main data storage"
        }
    }
    
    for memory_type, info in memory_types.items():
        print(f"\n{memory_type}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

# ============================================================================
# 🚀 TRITON BASICS
# ============================================================================

def explain_triton_concepts():
    """
    📚 TRITON LANGUAGE CONCEPTS
    
    Triton is a Python-like language for writing CUDA kernels:
    
    Key Concepts:
    1. @triton.jit decorator - Marks functions as kernels
    2. tl.constexpr - Compile-time constants
    3. tl.program_id() - Gets the current program/block ID
    4. tl.arange() - Creates ranges of indices
    5. tl.load() / tl.store() - Memory operations
    6. Grid and Block concepts - Parallel execution model
    """
    print("\n🚀 Triton Language Concepts:")
    print("=" * 50)
    
    concepts = {
        "@triton.jit": "Decorator that compiles Python-like code to CUDA kernels",
        "tl.constexpr": "Compile-time constants for optimization",
        "tl.program_id()": "Gets the current program/block ID in the grid",
        "tl.arange()": "Creates ranges of indices (must be power of 2)",
        "tl.load()": "Loads data from global memory",
        "tl.store()": "Stores data to global memory",
        "Grid": "Collection of blocks that execute the kernel",
        "Block": "Group of threads that execute together",
        "Thread": "Individual execution unit (conceptual in Triton)"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept:20}: {explanation}")

# ============================================================================
# 🎯 YOUR FIRST KERNEL: VECTOR ADDITION
# ============================================================================

@triton.jit
def vector_add_kernel(
    # Input and output pointers
    x_ptr,           # Pointer to first input vector
    y_ptr,           # Pointer to second input vector  
    output_ptr,      # Pointer to output vector
    
    # Vector size
    n_elements,      # Total number of elements
    
    # Strides (how much to increment pointer for next element)
    x_stride: tl.constexpr,      # Stride for x vector
    y_stride: tl.constexpr,      # Stride for y vector
    output_stride: tl.constexpr, # Stride for output vector
    
    # Block size (compile-time constant)
    BLOCK_SIZE: tl.constexpr,    # Number of elements per block
):
    """
    🎯 VECTOR ADDITION KERNEL
    
    This kernel adds two vectors element-wise:
    output[i] = x[i] + y[i]
    
    Key Concepts Demonstrated:
    1. Program ID and block indexing
    2. Memory offset calculation
    3. Masking for boundary conditions
    4. Memory loading and storing
    """
    
    # Get the program ID (which block we're in)
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    # tl.arange(0, BLOCK_SIZE) gives [0, 1, 2, ..., BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks to handle cases where n_elements is not divisible by BLOCK_SIZE
    # This prevents accessing memory beyond the array bounds
    mask = offsets < n_elements
    
    # Load input vectors from global memory
    x = tl.load(x_ptr + offsets * x_stride, mask=mask)
    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    
    # Perform the addition
    output = x + y
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets * output_stride, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    🎯 VECTOR ADDITION WRAPPER
    
    This function:
    1. Validates inputs
    2. Creates output tensor
    3. Launches the kernel with appropriate grid size
    4. Returns the result
    
    Args:
        x: First input vector (must be on GPU)
        y: Second input vector (must be on GPU)
        
    Returns:
        output: Result of x + y
    """
    # Input validation
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU!"
    assert x.numel() == y.numel(), "Input tensors must have the same size!"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype!"
    
    # Create output tensor with same shape and device as input
    output = torch.empty_like(x)
    
    # Calculate grid size
    # triton.cdiv(n, m) returns ceiling(n/m)
    # We need enough blocks to cover all elements
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    vector_add_kernel[grid](
        x_ptr=x,
        y_ptr=y, 
        output_ptr=output,
        n_elements=x.numel(),
        x_stride=1,      # Contiguous tensors have stride 1
        y_stride=1,
        output_stride=1,
        BLOCK_SIZE=128,  # Process 128 elements per block
    )
    
    return output

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_vector_addition():
    """
    🧪 TEST VECTOR ADDITION KERNEL
    
    Tests the kernel with various input sizes and validates correctness.
    """
    print("\n🧪 Testing Vector Addition Kernel:")
    print("=" * 50)
    
    test_cases = [
        (1024, "Power of 2 size"),
        (1000, "Non-power of 2 size"),
        (1, "Single element"),
        (10000, "Large size"),
    ]
    
    for size, description in test_cases:
        print(f"\n📊 Test: {description} (size={size})")
        
        # Create test data
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Test Triton kernel
        triton_result = vector_add(x, y)
        
        # Test PyTorch reference
        pytorch_result = x + y
        
        # Validate correctness
        is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-6)
        
        if is_correct:
            print(f"✅ PASS: Results match PyTorch reference")
        else:
            print(f"❌ FAIL: Results don't match PyTorch reference")
            print(f"Max difference: {torch.max(torch.abs(triton_result - pytorch_result)):.2e}")

def benchmark_vector_addition():
    """
    📊 BENCHMARK VECTOR ADDITION
    
    Compares performance between Triton kernel and PyTorch.
    """
    print("\n📊 Benchmarking Vector Addition:")
    print("=" * 50)
    
    sizes = [1024, 4096, 16384, 65536, 262144]
    
    for size in sizes:
        print(f"\n📈 Size: {size:,} elements")
        
        # Create test data
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = vector_add(x, y)
            _ = x + y
        
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start_time = time.time()
        for _ in range(100):
            triton_result = vector_add(x, y)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / 100 * 1000  # Convert to ms
        
        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(100):
            pytorch_result = x + y
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 100 * 1000  # Convert to ms
        
        # Calculate speedup
        speedup = pytorch_time / triton_time if triton_time > 0 else 0
        
        print(f"  Triton:  {triton_time:.3f} ms")
        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

# ============================================================================
# 🎓 UNDERSTANDING PROGRAM IDS AND BLOCKS
# ============================================================================

def explain_program_ids():
    """
    📚 UNDERSTANDING PROGRAM IDS AND BLOCKS
    
    This is crucial for understanding how parallel execution works in Triton.
    """
    print("\n🎓 Understanding Program IDs and Blocks:")
    print("=" * 50)
    
    print("""
    🧠 Key Concepts:
    
    1. Grid: The entire collection of blocks that execute your kernel
    2. Block: A group of threads that execute together (conceptual in Triton)
    3. Program ID: Unique identifier for each block in the grid
    4. Thread: Individual execution unit (handled automatically by Triton)
    
    📊 Example with BLOCK_SIZE=128 and n_elements=1000:
    
    Grid size = ceil(1000 / 128) = 8 blocks
    
    Block 0: handles elements [0, 127]     (pid=0)
    Block 1: handles elements [128, 255]   (pid=1)  
    Block 2: handles elements [256, 383]   (pid=2)
    Block 3: handles elements [384, 511]   (pid=3)
    Block 4: handles elements [512, 639]   (pid=4)
    Block 5: handles elements [640, 767]   (pid=5)
    Block 6: handles elements [768, 895]   (pid=6)
    Block 7: handles elements [896, 999]   (pid=7)  # Only 104 elements!
    
    🔍 Block 7 demonstrates masking:
    - offsets = [896, 897, 898, ..., 1023]
    - mask = [True, True, ..., True, False, False, ...]  # Only first 104 are True
    - Only the first 104 elements are processed and stored
    """)

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 1 tutorial.
    """
    print("🎯 LESSON 1: GPU FUNDAMENTALS & TRITON BASICS")
    print("=" * 70)
    print("Welcome to your first Triton tutorial!")
    print("This lesson will teach you the fundamentals of GPU programming with Triton.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_gpu_architecture()
    explain_memory_hierarchy()
    explain_triton_concepts()
    explain_program_ids()
    
    # Test and benchmark
    test_vector_addition()
    benchmark_vector_addition()
    
    print("\n🎉 Lesson 1 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding GPU architecture and memory hierarchy")
    print("2. ✅ Learning Triton language basics")
    print("3. ✅ Writing your first kernel (vector addition)")
    print("4. ✅ Understanding program IDs, blocks, and grids")
    print("5. ✅ Memory access patterns and masking")
    print("6. ✅ Testing and benchmarking kernels")
    
    print("\n🚀 Next Steps:")
    print("- Try modifying the BLOCK_SIZE and see how it affects performance")
    print("- Experiment with different data types (float16, int32)")
    print("- Move on to Lesson 2: Memory Management & Data Types")

if __name__ == "__main__":
    main()
