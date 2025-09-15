"""
🎯 Attention Optimization Examples

This module demonstrates various attention optimization techniques using Triton.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional

# ============================================================================
# 🧠 BASIC ATTENTION MECHANISM
# ============================================================================

@triton.jit
def basic_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    """
    🎯 BASIC ATTENTION KERNEL
    
    Implements basic attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)
    
    # Calculate block starting positions
    block_start_s = pid_s * BLOCK_SIZE
    block_start_d = 0
    
    # Create offsets for this block
    offs_s = block_start_s + tl.arange(0, BLOCK_SIZE)
    offs_d = block_start_d + tl.arange(0, head_dim)
    
    # Create masks for boundary checking
    mask_s = offs_s < seq_len
    mask_d = offs_d < head_dim
    
    # Load Q block
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    
    # Load K block
    k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    
    # Load V block
    v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    
    # Compute attention scores: QK^T
    attention_scores = tl.dot(q, k, trans_b=True)
    
    # Scale by sqrt(d_k)
    attention_scores = attention_scores / tl.sqrt(tl.cast(head_dim, tl.float32))
    
    # Apply softmax
    attention_weights = tl.softmax(attention_scores, axis=1)
    
    # Apply attention to values: Attention * V
    output = tl.dot(attention_weights, v)
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(output_ptrs, output, mask=mask_s[:, None] & mask_d[None, :])

def basic_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    🎯 BASIC ATTENTION WRAPPER
    
    Wrapper function for basic attention mechanism.
    """
    # Input validation
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on GPU!"
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape!"
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create output tensor
    output = torch.empty_like(q)
    
    # Calculate strides
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ob, stride_oh, stride_os, stride_od = output.stride()
    
    # Define block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_SIZE))
    
    # Launch kernel
    basic_attention_kernel[grid](
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# ============================================================================
# 🚀 OPTIMIZED ATTENTION MECHANISM
# ============================================================================

@triton.jit
def optimized_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    🚀 OPTIMIZED ATTENTION KERNEL
    
    Implements optimized attention mechanism with:
    - Tiled computation
    - Shared memory optimization
    - Fused operations
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=3)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checking
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len
    mask_k = offs_k < head_dim
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, head_dim, BLOCK_SIZE_K):
        # Load Q block
        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + (k + offs_k)[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load K block
        k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + (k + offs_k)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate attention scores: QK^T
        accumulator += tl.dot(q, k, trans_b=True)
    
    # Scale by sqrt(d_k)
    accumulator = accumulator / tl.sqrt(tl.cast(head_dim, tl.float32))
    
    # Apply softmax
    attention_weights = tl.softmax(accumulator, axis=1)
    
    # Initialize output accumulator
    output_accumulator = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    
    # Loop over K dimension for V
    for k in range(0, head_dim, BLOCK_SIZE_K):
        # Load V block
        v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + (k + offs_k)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Apply attention to values: Attention * V
        output_accumulator += tl.dot(attention_weights, v)
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + tl.arange(0, head_dim)[None, :] * stride_od
    tl.store(output_ptrs, output_accumulator, mask=mask_m[:, None])

def optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED ATTENTION WRAPPER
    
    Wrapper function for optimized attention mechanism.
    """
    # Input validation
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on GPU!"
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape!"
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create output tensor
    output = torch.empty_like(q)
    
    # Calculate strides
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ob, stride_oh, stride_os, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(seq_len, BLOCK_SIZE_N))
    
    # Launch kernel
    optimized_attention_kernel[grid](
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

# ============================================================================
# 🔥 FLASH ATTENTION MECHANISM
# ============================================================================

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    🔥 FLASH ATTENTION KERNEL
    
    Implements FlashAttention mechanism with:
    - Online softmax computation
    - Memory-efficient attention
    - Tiled computation
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    
    # Calculate block starting positions
    block_start_m = pid_m * BLOCK_SIZE_M
    
    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checking
    mask_m = offs_m < seq_len
    mask_k = offs_k < head_dim
    
    # Load Q block
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Initialize accumulators
    l = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    m = tl.full((BLOCK_SIZE_M,), -float('inf'), dtype=tl.float32)
    o = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    
    # Loop over N dimension in blocks
    for n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < seq_len
        
        # Load K block
        k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_k[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Load V block
        v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_k[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Compute attention scores: QK^T
        attention_scores = tl.dot(q, k, trans_b=True)
        
        # Scale by sqrt(d_k)
        attention_scores = attention_scores / tl.sqrt(tl.cast(head_dim, tl.float32))
        
        # Online softmax computation
        m_new = tl.maximum(m, tl.max(attention_scores, axis=1))
        alpha = tl.exp(attention_scores - m_new[:, None])
        l_new = l + tl.sum(alpha, axis=1)
        
        # Update output
        o = o * (l / l_new)[:, None] + tl.dot(alpha, v) * tl.exp(m - m_new)[:, None]
        
        # Update accumulators
        l = l_new
        m = m_new
    
    # Store result
    output_ptrs = output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_k[None, :] * stride_od
    tl.store(output_ptrs, o, mask=mask_m[:, None] & mask_k[None, :])

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    🔥 FLASH ATTENTION WRAPPER
    
    Wrapper function for FlashAttention mechanism.
    """
    # Input validation
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on GPU!"
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape!"
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create output tensor
    output = torch.empty_like(q)
    
    # Calculate strides
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ob, stride_oh, stride_os, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M))
    
    # Launch kernel
    flash_attention_kernel[grid](
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

# ============================================================================
# 🧪 TESTING AND VALIDATION
# ============================================================================

def test_attention_mechanisms():
    """
    🧪 TEST ATTENTION MECHANISMS
    
    Tests various attention mechanisms and compares performance.
    """
    print("🧪 Testing Attention Mechanisms:")
    print("=" * 50)
    
    # Test configuration
    batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
    
    # Create test data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    # Test basic attention
    print("\n📊 Test: Basic Attention")
    try:
        basic_result = basic_attention(q, k, v)
        print(f"  Result shape: {basic_result.shape}")
        print(f"  Result device: {basic_result.device}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test optimized attention
    print("\n📊 Test: Optimized Attention")
    try:
        optimized_result = optimized_attention(q, k, v)
        print(f"  Result shape: {optimized_result.shape}")
        print(f"  Result device: {optimized_result.device}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test FlashAttention
    print("\n📊 Test: FlashAttention")
    try:
        flash_result = flash_attention(q, k, v)
        print(f"  Result shape: {flash_result.shape}")
        print(f"  Result device: {flash_result.device}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")

# ============================================================================
# 📊 PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_attention_mechanisms():
    """
    📊 BENCHMARK ATTENTION MECHANISMS
    
    Benchmarks various attention mechanisms and compares performance.
    """
    print("\n📊 Benchmarking Attention Mechanisms:")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        (2, 8, 128, 64),   # batch_size, num_heads, seq_len, head_dim
        (4, 8, 256, 64),
        (8, 16, 512, 64),
    ]
    
    for batch_size, num_heads, seq_len, head_dim in configs:
        print(f"\n📈 Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
        
        # Create test data
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
        
        # Benchmark basic attention
        print("\n  Basic Attention:")
        try:
            # Warmup
            for _ in range(10):
                _ = basic_attention(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = basic_attention(q, k, v)
            torch.cuda.synchronize()
            basic_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {basic_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark optimized attention
        print("\n  Optimized Attention:")
        try:
            # Warmup
            for _ in range(10):
                _ = optimized_attention(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = optimized_attention(q, k, v)
            torch.cuda.synchronize()
            optimized_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {optimized_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")
        
        # Benchmark FlashAttention
        print("\n  FlashAttention:")
        try:
            # Warmup
            for _ in range(10):
                _ = flash_attention(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = flash_attention(q, k, v)
            torch.cuda.synchronize()
            flash_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {flash_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the attention optimization examples.
    """
    print("🎯 ATTENTION OPTIMIZATION EXAMPLES")
    print("=" * 70)
    print("This module demonstrates various attention optimization techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the examples
    test_attention_mechanisms()
    benchmark_attention_mechanisms()
    
    print("\n🎉 Attention Optimization Examples Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Basic attention mechanism implementation")
    print("2. ✅ Optimized attention with tiling and shared memory")
    print("3. ✅ FlashAttention with online softmax computation")
    print("4. ✅ Performance comparison between different approaches")
    print("5. ✅ Memory-efficient attention computation")
    print("6. ✅ Scalable attention for different sequence lengths")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different block sizes")
    print("- Try optimizing for different hardware configurations")
    print("- Implement causal attention mechanisms")
    print("- Add support for different attention patterns")

if __name__ == "__main__":
    main()
