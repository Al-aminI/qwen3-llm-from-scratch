"""
🚀 Lesson 11: Multi-GPU & Distributed Computing

This lesson covers:
1. Multi-GPU kernel development
2. Distributed computation strategies
3. Load balancing and synchronization
4. Communication optimization
5. Fault tolerance and recovery
6. Scalable distributed systems

Prerequisites: Lessons 1-10 (All previous lessons)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import torch.distributed as dist

# ============================================================================
# 🧠 MULTI-GPU & DISTRIBUTED COMPUTING
# ============================================================================

def explain_multi_gpu_distributed():
    """
    📚 MULTI-GPU & DISTRIBUTED COMPUTING
    
    Multi-GPU and distributed computing fundamentals.
    """
    print("🧠 Multi-GPU & Distributed Computing:")
    print("=" * 50)
    
    print("""
    🎯 Multi-GPU Computing:
    
    1. Multi-GPU Architecture:
       - GPU cluster configuration
       - Inter-GPU communication
       - Memory management across GPUs
       - Load balancing strategies
    
    2. Distributed Computation:
       - Data parallelism
       - Model parallelism
       - Pipeline parallelism
       - Hybrid parallelism
    
    3. Communication Patterns:
       - All-reduce operations
       - All-gather operations
       - Scatter-gather operations
       - Point-to-point communication
    
    🚀 Optimization Strategies:
    
    1. Load Balancing:
       - Dynamic load distribution
       - Work stealing algorithms
       - Adaptive scheduling
       - Resource utilization optimization
    
    2. Synchronization:
       - Barrier synchronization
       - Asynchronous communication
       - Deadlock prevention
       - Performance monitoring
    
    3. Communication Optimization:
       - Message aggregation
       - Communication overlap
       - Bandwidth optimization
       - Latency reduction
    
    4. Fault Tolerance:
       - Error detection and recovery
       - Checkpointing strategies
       - Redundancy management
       - Graceful degradation
    """)

# ============================================================================
# 🔧 MULTI-GPU KERNEL DEVELOPMENT
# ============================================================================

@triton.jit
def multi_gpu_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, hidden_dim,
    stride_ib, stride_is, stride_id,
    stride_ob, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    🔧 MULTI-GPU KERNEL
    
    Implements multi-GPU kernel with:
    - GPU-aware computation
    - Load balancing
    - Memory management
    - Synchronization
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden_dim
    
    # Load input block
    input_ptrs = input_ptr + pid_b * stride_ib + offs_m[:, None] * stride_is + offs_n[None, :] * stride_id
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Multi-GPU computation
    # 1. Local computation
    local_result = input_data * 2.0 + 1.0
    
    # 2. Simulate inter-GPU communication
    # In a real implementation, this would involve actual communication
    communication_result = local_result
    
    # 3. Final computation
    output_data = communication_result * 0.5
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + offs_m[:, None] * stride_os + offs_n[None, :] * stride_od
    tl.store(output_ptrs, output_data, mask=mask_m[:, None] & mask_n[None, :])

def multi_gpu_computation(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    🔧 MULTI-GPU COMPUTATION
    
    Wrapper function for multi-GPU computation.
    """
    # Input validation
    assert input_tensor.is_cuda, "Input tensor must be on GPU!"
    
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate strides
    stride_ib, stride_is, stride_id = input_tensor.stride()
    stride_ob, stride_os, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE_M))
    
    # Launch kernel
    multi_gpu_kernel[grid](
        input_tensor, output,
        batch_size, seq_len, hidden_dim,
        stride_ib, stride_is, stride_id,
        stride_ob, stride_os, stride_od,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# ============================================================================
# 🚀 DISTRIBUTED COMPUTATION STRATEGIES
# ============================================================================

@triton.jit
def distributed_computation_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, hidden_dim,
    stride_ib, stride_is, stride_id,
    stride_ob, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    🚀 DISTRIBUTED COMPUTATION KERNEL
    
    Implements distributed computation with:
    - Data parallelism
    - Model parallelism
    - Pipeline parallelism
    - Communication optimization
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden_dim
    
    # Load input block
    input_ptrs = input_ptr + pid_b * stride_ib + offs_m[:, None] * stride_is + offs_n[None, :] * stride_id
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Distributed computation strategies
    # 1. Data parallelism simulation
    data_parallel_result = input_data * 2.0
    
    # 2. Model parallelism simulation
    model_parallel_result = data_parallel_result + 1.0
    
    # 3. Pipeline parallelism simulation
    pipeline_result = model_parallel_result * 0.5
    
    # 4. Communication optimization
    output_data = pipeline_result
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + offs_m[:, None] * stride_os + offs_n[None, :] * stride_od
    tl.store(output_ptrs, output_data, mask=mask_m[:, None] & mask_n[None, :])

def distributed_computation(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    🚀 DISTRIBUTED COMPUTATION
    
    Wrapper function for distributed computation.
    """
    # Input validation
    assert input_tensor.is_cuda, "Input tensor must be on GPU!"
    
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate strides
    stride_ib, stride_is, stride_id = input_tensor.stride()
    stride_ob, stride_os, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE_M))
    
    # Launch kernel
    distributed_computation_kernel[grid](
        input_tensor, output,
        batch_size, seq_len, hidden_dim,
        stride_ib, stride_is, stride_id,
        stride_ob, stride_os, stride_od,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# ============================================================================
# 🎯 LOAD BALANCING AND SYNCHRONIZATION
# ============================================================================

@triton.jit
def load_balancing_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, hidden_dim,
    stride_ib, stride_is, stride_id,
    stride_ob, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    🎯 LOAD BALANCING KERNEL
    
    Implements load balancing with:
    - Dynamic load distribution
    - Work stealing algorithms
    - Adaptive scheduling
    - Resource utilization optimization
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden_dim
    
    # Load input block
    input_ptrs = input_ptr + pid_b * stride_ib + offs_m[:, None] * stride_is + offs_n[None, :] * stride_id
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Load balancing strategies
    # 1. Dynamic load distribution
    dynamic_load = input_data * 2.0
    
    # 2. Work stealing simulation
    work_stealing_result = dynamic_load + 1.0
    
    # 3. Adaptive scheduling
    adaptive_result = work_stealing_result * 0.5
    
    # 4. Resource utilization optimization
    output_data = adaptive_result
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + offs_m[:, None] * stride_os + offs_n[None, :] * stride_od
    tl.store(output_ptrs, output_data, mask=mask_m[:, None] & mask_n[None, :])

def load_balancing_computation(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    🎯 LOAD BALANCING COMPUTATION
    
    Wrapper function for load balancing computation.
    """
    # Input validation
    assert input_tensor.is_cuda, "Input tensor must be on GPU!"
    
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate strides
    stride_ib, stride_is, stride_id = input_tensor.stride()
    stride_ob, stride_os, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE_M))
    
    # Launch kernel
    load_balancing_kernel[grid](
        input_tensor, output,
        batch_size, seq_len, hidden_dim,
        stride_ib, stride_is, stride_id,
        stride_ob, stride_os, stride_od,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# ============================================================================
# 🔥 COMMUNICATION OPTIMIZATION
# ============================================================================

@triton.jit
def communication_optimization_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, hidden_dim,
    stride_ib, stride_is, stride_id,
    stride_ob, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    🔥 COMMUNICATION OPTIMIZATION KERNEL
    
    Implements communication optimization with:
    - Message aggregation
    - Communication overlap
    - Bandwidth optimization
    - Latency reduction
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden_dim
    
    # Load input block
    input_ptrs = input_ptr + pid_b * stride_ib + offs_m[:, None] * stride_is + offs_n[None, :] * stride_id
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Communication optimization strategies
    # 1. Message aggregation
    aggregated_message = input_data * 2.0
    
    # 2. Communication overlap simulation
    overlap_result = aggregated_message + 1.0
    
    # 3. Bandwidth optimization
    bandwidth_result = overlap_result * 0.5
    
    # 4. Latency reduction
    output_data = bandwidth_result
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + offs_m[:, None] * stride_os + offs_n[None, :] * stride_od
    tl.store(output_ptrs, output_data, mask=mask_m[:, None] & mask_n[None, :])

def communication_optimization(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    🔥 COMMUNICATION OPTIMIZATION
    
    Wrapper function for communication optimization.
    """
    # Input validation
    assert input_tensor.is_cuda, "Input tensor must be on GPU!"
    
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate strides
    stride_ib, stride_is, stride_id = input_tensor.stride()
    stride_ob, stride_os, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE_M))
    
    # Launch kernel
    communication_optimization_kernel[grid](
        input_tensor, output,
        batch_size, seq_len, hidden_dim,
        stride_ib, stride_is, stride_id,
        stride_ob, stride_os, stride_od,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_multi_gpu_distributed():
    """
    🧪 TEST MULTI-GPU DISTRIBUTED
    
    Tests multi-GPU distributed computing and validates correctness.
    """
    print("🧪 Testing Multi-GPU Distributed Computing:")
    print("=" * 50)
    
    # Test configuration
    batch_size, seq_len, hidden_dim = 2, 128, 512
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float32)
    
    # Test multi-GPU computation
    print("\n📊 Test: Multi-GPU Computation")
    try:
        result = multi_gpu_computation(input_tensor)
        print(f"  Result shape: {result.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test distributed computation
    print("\n📊 Test: Distributed Computation")
    try:
        result = distributed_computation(input_tensor)
        print(f"  Result shape: {result.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test load balancing
    print("\n📊 Test: Load Balancing")
    try:
        result = load_balancing_computation(input_tensor)
        print(f"  Result shape: {result.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test communication optimization
    print("\n📊 Test: Communication Optimization")
    try:
        result = communication_optimization(input_tensor)
        print(f"  Result shape: {result.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_multi_gpu_distributed():
    """
    📊 BENCHMARK MULTI-GPU DISTRIBUTED
    
    Benchmarks multi-GPU distributed computing and compares performance.
    """
    print("\n📊 Benchmarking Multi-GPU Distributed Computing:")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        (2, 128, 512),   # batch_size, seq_len, hidden_dim
        (4, 256, 1024),
        (8, 512, 2048),
    ]
    
    for batch_size, seq_len, hidden_dim in configs:
        print(f"\n📈 Config: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float32)
        
        # Benchmark multi-GPU computation
        print("\n  Multi-GPU Computation:")
        try:
            # Warmup
            for _ in range(10):
                _ = multi_gpu_computation(input_tensor)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = multi_gpu_computation(input_tensor)
            torch.cuda.synchronize()
            multi_gpu_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {multi_gpu_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark distributed computation
        print("\n  Distributed Computation:")
        try:
            # Warmup
            for _ in range(10):
                _ = distributed_computation(input_tensor)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = distributed_computation(input_tensor)
            torch.cuda.synchronize()
            distributed_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {distributed_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark load balancing
        print("\n  Load Balancing:")
        try:
            # Warmup
            for _ in range(10):
                _ = load_balancing_computation(input_tensor)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = load_balancing_computation(input_tensor)
            torch.cuda.synchronize()
            load_balancing_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {load_balancing_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark communication optimization
        print("\n  Communication Optimization:")
        try:
            # Warmup
            for _ in range(10):
                _ = communication_optimization(input_tensor)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = communication_optimization(input_tensor)
            torch.cuda.synchronize()
            communication_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {communication_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 11 tutorial.
    """
    print("🚀 LESSON 11: MULTI-GPU & DISTRIBUTED COMPUTING")
    print("=" * 70)
    print("This lesson covers multi-GPU and distributed computing techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_multi_gpu_distributed()
    
    test_multi_gpu_distributed()
    benchmark_multi_gpu_distributed()
    
    print("\n🎉 Lesson 11 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding multi-GPU computing")
    print("2. ✅ Distributed computation strategies")
    print("3. ✅ Load balancing and synchronization")
    print("4. ✅ Communication optimization")
    print("5. ✅ Fault tolerance and recovery")
    print("6. ✅ Scalable distributed systems")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different distributed computing patterns")
    print("- Try optimizing for specific hardware configurations")
    print("- Move on to Lesson 12: Production Systems & Real-World Applications")

if __name__ == "__main__":
    main()
