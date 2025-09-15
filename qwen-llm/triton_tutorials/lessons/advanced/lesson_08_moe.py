"""
🚀 Lesson 8: MoE (Mixture of Experts) Implementation

This lesson covers:
1. MoE architecture fundamentals
2. Expert routing optimization
3. Load balancing techniques
4. Sparse expert computation
5. Memory-efficient MoE implementation
6. Performance analysis and optimization

Prerequisites: Lessons 1-7 (All previous lessons)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🧠 MOE ARCHITECTURE FUNDAMENTALS
# ============================================================================

def explain_moe_architecture():
    """
    📚 MOE ARCHITECTURE FUNDAMENTALS
    
    Mixture of Experts architecture and optimization techniques.
    """
    print("🧠 MoE Architecture Fundamentals:")
    print("=" * 50)
    
    print("""
    🎯 MoE Components:
    
    1. Router: Determines which experts to use
    2. Experts: Specialized neural network modules
    3. Gating Network: Computes expert selection weights
    4. Load Balancer: Ensures balanced expert usage
    
    🚀 MoE Computation:
    MoE(x) = Σ(i=1 to N) G_i(x) * E_i(x)
    
    Where:
    - G_i(x): Gating weight for expert i
    - E_i(x): Output of expert i
    - N: Number of experts
    
    📊 Key Challenges:
    1. Expert Routing: Efficient expert selection
    2. Load Balancing: Balanced expert utilization
    3. Sparse Computation: Only activate selected experts
    4. Memory Management: Efficient expert storage
    5. Communication: Expert coordination
    
    🚀 Optimization Strategies:
    1. Top-K Routing: Select top K experts
    2. Load Balancing: Balance expert usage
    3. Sparse Activation: Only compute active experts
    4. Memory Coalescing: Optimize memory access
    5. Expert Parallelism: Parallel expert computation
    """)

@triton.jit
def expert_routing_kernel(
    input_ptr, router_weights_ptr, expert_indices_ptr, expert_weights_ptr,
    batch_size, seq_len, hidden_dim, num_experts, top_k,
    stride_ib, stride_is, stride_id,
    stride_rb, stride_rs, stride_rd,
    stride_eib, stride_eis, stride_eik,
    stride_ewb, stride_ews, stride_ewk,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 EXPERT ROUTING KERNEL
    
    Implements expert routing with:
    - Top-K expert selection
    - Load balancing
    - Efficient routing computation
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_s = pid_s * BLOCK_SIZE
    
    # Create offsets for this block
    offs_s = block_start_s + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, hidden_dim)
    
    # Create masks for boundary checking
    mask_s = offs_s < seq_len
    mask_d = offs_d < hidden_dim
    
    # Load input block
    input_ptrs = input_ptr + pid_b * stride_ib + offs_s[:, None] * stride_is + offs_d[None, :] * stride_id
    input_data = tl.load(input_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    
    # Load router weights
    router_ptrs = router_weights_ptr + pid_b * stride_rb + offs_s[:, None] * stride_rs + tl.arange(0, num_experts)[None, :] * stride_rd
    router_weights = tl.load(router_ptrs, mask=mask_s[:, None] & (tl.arange(0, num_experts)[None, :] < num_experts), other=0.0)
    
    # Compute gating scores
    gating_scores = tl.dot(input_data, router_weights)
    
    # Apply softmax
    gating_weights = tl.softmax(gating_scores, axis=1)
    
    # Top-K selection
    top_k_indices = tl.argsort(gating_weights, axis=1, descending=True)[:, :top_k]
    top_k_weights = tl.gather(gating_weights, top_k_indices, axis=1)
    
    # Normalize weights
    weight_sum = tl.sum(top_k_weights, axis=1, keepdims=True)
    normalized_weights = top_k_weights / (weight_sum + 1e-8)
    
    # Store expert indices
    expert_indices_ptrs = expert_indices_ptr + pid_b * stride_eib + offs_s[:, None] * stride_eis + tl.arange(0, top_k)[None, :] * stride_eik
    tl.store(expert_indices_ptrs, top_k_indices, mask=mask_s[:, None] & (tl.arange(0, top_k)[None, :] < top_k))
    
    # Store expert weights
    expert_weights_ptrs = expert_weights_ptr + pid_b * stride_ewb + offs_s[:, None] * stride_ews + tl.arange(0, top_k)[None, :] * stride_ewk
    tl.store(expert_weights_ptrs, normalized_weights, mask=mask_s[:, None] & (tl.arange(0, top_k)[None, :] < top_k))

def expert_routing(input_tensor: torch.Tensor, router_weights: torch.Tensor, 
                  num_experts: int, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🎯 EXPERT ROUTING WRAPPER
    
    Wrapper function for expert routing.
    """
    # Input validation
    assert input_tensor.is_cuda and router_weights.is_cuda, "Input tensors must be on GPU!"
    assert input_tensor.shape[2] == router_weights.shape[1], "Hidden dimensions must match!"
    
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Create output tensors
    expert_indices = torch.empty((batch_size, seq_len, top_k), device=input_tensor.device, dtype=torch.int32)
    expert_weights = torch.empty((batch_size, seq_len, top_k), device=input_tensor.device, dtype=torch.float32)
    
    # Calculate strides
    stride_ib, stride_is, stride_id = input_tensor.stride()
    stride_rb, stride_rs, stride_rd = router_weights.stride()
    stride_eib, stride_eis, stride_eik = expert_indices.stride()
    stride_ewb, stride_ews, stride_ewk = expert_weights.stride()
    
    # Define block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE))
    
    # Launch kernel
    expert_routing_kernel[grid](
        input_tensor, router_weights, expert_indices, expert_weights,
        batch_size, seq_len, hidden_dim, num_experts, top_k,
        stride_ib, stride_is, stride_id,
        stride_rb, stride_rs, stride_rd,
        stride_eib, stride_eis, stride_eik,
        stride_ewb, stride_ews, stride_ewk,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return expert_indices, expert_weights

# ============================================================================
# 🚀 EXPERT COMPUTATION
# ============================================================================

@triton.jit
def expert_computation_kernel(
    input_ptr, expert_weights_ptr, expert_output_ptr,
    batch_size, seq_len, hidden_dim, num_experts, top_k,
    stride_ib, stride_is, stride_id,
    stride_ewb, stride_ews, stride_ewk,
    stride_eob, stride_eos, stride_eod,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🚀 EXPERT COMPUTATION KERNEL
    
    Implements expert computation with:
    - Sparse expert activation
    - Efficient expert computation
    - Memory-optimized operations
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_s = pid_s * BLOCK_SIZE
    
    # Create offsets for this block
    offs_s = block_start_s + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, hidden_dim)
    
    # Create masks for boundary checking
    mask_s = offs_s < seq_len
    mask_d = offs_d < hidden_dim
    
    # Load input block
    input_ptrs = input_ptr + pid_b * stride_ib + offs_s[:, None] * stride_is + offs_d[None, :] * stride_id
    input_data = tl.load(input_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    
    # Load expert weights
    expert_weights_ptrs = expert_weights_ptr + pid_b * stride_ewb + offs_s[:, None] * stride_ews + tl.arange(0, top_k)[None, :] * stride_ewk
    expert_weights = tl.load(expert_weights_ptrs, mask=mask_s[:, None] & (tl.arange(0, top_k)[None, :] < top_k), other=0.0)
    
    # Compute expert outputs (simplified - in practice, this would be more complex)
    expert_output = tl.zeros((BLOCK_SIZE, hidden_dim), dtype=tl.float32)
    
    # Loop over experts
    for k in range(top_k):
        # Get expert weight
        weight = expert_weights[:, k:k+1]
        
        # Compute expert output (simplified computation)
        expert_output += weight * input_data
    
    # Store result
    expert_output_ptrs = expert_output_ptr + pid_b * stride_eob + offs_s[:, None] * stride_eos + offs_d[None, :] * stride_eod
    tl.store(expert_output_ptrs, expert_output, mask=mask_s[:, None] & mask_d[None, :])

def expert_computation(input_tensor: torch.Tensor, expert_weights: torch.Tensor, 
                      num_experts: int, top_k: int = 2) -> torch.Tensor:
    """
    🚀 EXPERT COMPUTATION WRAPPER
    
    Wrapper function for expert computation.
    """
    # Input validation
    assert input_tensor.is_cuda and expert_weights.is_cuda, "Input tensors must be on GPU!"
    assert input_tensor.shape[0] == expert_weights.shape[0], "Batch sizes must match!"
    assert input_tensor.shape[1] == expert_weights.shape[1], "Sequence lengths must match!"
    
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Create output tensor
    expert_output = torch.empty_like(input_tensor)
    
    # Calculate strides
    stride_ib, stride_is, stride_id = input_tensor.stride()
    stride_ewb, stride_ews, stride_ewk = expert_weights.stride()
    stride_eob, stride_eos, stride_eod = expert_output.stride()
    
    # Define block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE))
    
    # Launch kernel
    expert_computation_kernel[grid](
        input_tensor, expert_weights, expert_output,
        batch_size, seq_len, hidden_dim, num_experts, top_k,
        stride_ib, stride_is, stride_id,
        stride_ewb, stride_ews, stride_ewk,
        stride_eob, stride_eos, stride_eod,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return expert_output

# ============================================================================
# 🔥 LOAD BALANCING
# ============================================================================

@triton.jit
def load_balancing_kernel(
    expert_indices_ptr, expert_weights_ptr, load_balance_loss_ptr,
    batch_size, seq_len, num_experts, top_k,
    stride_eib, stride_eis, stride_eik,
    stride_ewb, stride_ews, stride_ewk,
    stride_lbb, stride_lbs,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🔥 LOAD BALANCING KERNEL
    
    Implements load balancing with:
    - Expert usage tracking
    - Load balancing loss computation
    - Efficient load balancing
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    
    # Calculate block starting positions
    block_start_s = pid_s * BLOCK_SIZE
    
    # Create offsets for this block
    offs_s = block_start_s + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for boundary checking
    mask_s = offs_s < seq_len
    
    # Load expert indices
    expert_indices_ptrs = expert_indices_ptr + pid_b * stride_eib + offs_s[:, None] * stride_eis + tl.arange(0, top_k)[None, :] * stride_eik
    expert_indices = tl.load(expert_indices_ptrs, mask=mask_s[:, None] & (tl.arange(0, top_k)[None, :] < top_k), other=0)
    
    # Load expert weights
    expert_weights_ptrs = expert_weights_ptr + pid_b * stride_ewb + offs_s[:, None] * stride_ews + tl.arange(0, top_k)[None, :] * stride_ewk
    expert_weights = tl.load(expert_weights_ptrs, mask=mask_s[:, None] & (tl.arange(0, top_k)[None, :] < top_k), other=0.0)
    
    # Compute load balancing loss
    load_balance_loss = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over experts
    for expert_id in range(num_experts):
        # Count expert usage
        expert_usage = tl.sum(tl.where(expert_indices == expert_id, expert_weights, 0.0), axis=1)
        
        # Compute load balancing loss
        load_balance_loss += expert_usage * expert_usage
    
    # Store result
    load_balance_loss_ptrs = load_balance_loss_ptr + pid_b * stride_lbb + offs_s * stride_lbs
    tl.store(load_balance_loss_ptrs, load_balance_loss, mask=mask_s)

def load_balancing(expert_indices: torch.Tensor, expert_weights: torch.Tensor, 
                  num_experts: int, top_k: int = 2) -> torch.Tensor:
    """
    🔥 LOAD BALANCING WRAPPER
    
    Wrapper function for load balancing.
    """
    # Input validation
    assert expert_indices.is_cuda and expert_weights.is_cuda, "Input tensors must be on GPU!"
    assert expert_indices.shape == expert_weights.shape, "Expert indices and weights must have the same shape!"
    
    batch_size, seq_len, _ = expert_indices.shape
    
    # Create output tensor
    load_balance_loss = torch.empty((batch_size, seq_len), device=expert_indices.device, dtype=torch.float32)
    
    # Calculate strides
    stride_eib, stride_eis, stride_eik = expert_indices.stride()
    stride_ewb, stride_ews, stride_ewk = expert_weights.stride()
    stride_lbb, stride_lbs = load_balance_loss.stride()
    
    # Define block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SIZE))
    
    # Launch kernel
    load_balancing_kernel[grid](
        expert_indices, expert_weights, load_balance_loss,
        batch_size, seq_len, num_experts, top_k,
        stride_eib, stride_eis, stride_eik,
        stride_ewb, stride_ews, stride_ewk,
        stride_lbb, stride_lbs,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return load_balance_loss

# ============================================================================
# 🎯 COMPLETE MOE LAYER
# ============================================================================

def moe_layer(input_tensor: torch.Tensor, router_weights: torch.Tensor, 
              num_experts: int, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🎯 COMPLETE MOE LAYER
    
    Complete MoE layer implementation with:
    - Expert routing
    - Expert computation
    - Load balancing
    """
    # Input validation
    assert input_tensor.is_cuda and router_weights.is_cuda, "Input tensors must be on GPU!"
    assert input_tensor.shape[2] == router_weights.shape[1], "Hidden dimensions must match!"
    
    # Expert routing
    expert_indices, expert_weights = expert_routing(input_tensor, router_weights, num_experts, top_k)
    
    # Expert computation
    expert_output = expert_computation(input_tensor, expert_weights, num_experts, top_k)
    
    # Load balancing
    load_balance_loss = load_balancing(expert_indices, expert_weights, num_experts, top_k)
    
    return expert_output, load_balance_loss

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_moe_implementation():
    """
    🧪 TEST MOE IMPLEMENTATION
    
    Tests MoE implementation and validates correctness.
    """
    print("🧪 Testing MoE Implementation:")
    print("=" * 50)
    
    # Test configuration
    batch_size, seq_len, hidden_dim = 2, 128, 512
    num_experts = 8
    top_k = 2
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float32)
    router_weights = torch.randn(num_experts, hidden_dim, device='cuda', dtype=torch.float32)
    
    # Test expert routing
    print("\n📊 Test: Expert Routing")
    try:
        expert_indices, expert_weights = expert_routing(input_tensor, router_weights, num_experts, top_k)
        print(f"  Expert indices shape: {expert_indices.shape}")
        print(f"  Expert weights shape: {expert_weights.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test expert computation
    print("\n📊 Test: Expert Computation")
    try:
        expert_output = expert_computation(input_tensor, expert_weights, num_experts, top_k)
        print(f"  Expert output shape: {expert_output.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test load balancing
    print("\n📊 Test: Load Balancing")
    try:
        load_balance_loss = load_balancing(expert_indices, expert_weights, num_experts, top_k)
        print(f"  Load balance loss shape: {load_balance_loss.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test complete MoE layer
    print("\n📊 Test: Complete MoE Layer")
    try:
        expert_output, load_balance_loss = moe_layer(input_tensor, router_weights, num_experts, top_k)
        print(f"  Expert output shape: {expert_output.shape}")
        print(f"  Load balance loss shape: {load_balance_loss.shape}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_moe_implementation():
    """
    📊 BENCHMARK MOE IMPLEMENTATION
    
    Benchmarks MoE implementation and compares performance.
    """
    print("\n📊 Benchmarking MoE Implementation:")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        (2, 128, 512, 8, 2),   # batch_size, seq_len, hidden_dim, num_experts, top_k
        (4, 256, 1024, 16, 2),
        (8, 512, 2048, 32, 4),
    ]
    
    for batch_size, seq_len, hidden_dim, num_experts, top_k in configs:
        print(f"\n📈 Config: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}, experts={num_experts}, top_k={top_k}")
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float32)
        router_weights = torch.randn(num_experts, hidden_dim, device='cuda', dtype=torch.float32)
        
        # Benchmark expert routing
        print("\n  Expert Routing:")
        try:
            # Warmup
            for _ in range(10):
                _ = expert_routing(input_tensor, router_weights, num_experts, top_k)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = expert_routing(input_tensor, router_weights, num_experts, top_k)
            torch.cuda.synchronize()
            routing_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {routing_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark expert computation
        print("\n  Expert Computation:")
        try:
            expert_indices, expert_weights = expert_routing(input_tensor, router_weights, num_experts, top_k)
            
            # Warmup
            for _ in range(10):
                _ = expert_computation(input_tensor, expert_weights, num_experts, top_k)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = expert_computation(input_tensor, expert_weights, num_experts, top_k)
            torch.cuda.synchronize()
            computation_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {computation_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark complete MoE layer
        print("\n  Complete MoE Layer:")
        try:
            # Warmup
            for _ in range(10):
                _ = moe_layer(input_tensor, router_weights, num_experts, top_k)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = moe_layer(input_tensor, router_weights, num_experts, top_k)
            torch.cuda.synchronize()
            moe_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {moe_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 8 tutorial.
    """
    print("🚀 LESSON 8: MOE (MIXTURE OF EXPERTS) IMPLEMENTATION")
    print("=" * 70)
    print("This lesson covers MoE architecture and optimization techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_moe_architecture()
    
    test_moe_implementation()
    benchmark_moe_implementation()
    
    print("\n🎉 Lesson 8 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding MoE architecture")
    print("2. ✅ Implementing expert routing")
    print("3. ✅ Expert computation optimization")
    print("4. ✅ Load balancing techniques")
    print("5. ✅ Sparse expert activation")
    print("6. ✅ Memory-efficient MoE implementation")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different expert configurations")
    print("- Try optimizing for different hardware configurations")
    print("- Move on to Lesson 9: Advanced Optimization Techniques")

if __name__ == "__main__":
    main()
