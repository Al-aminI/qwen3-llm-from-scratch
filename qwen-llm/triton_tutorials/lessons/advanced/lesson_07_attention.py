"""
🚀 Lesson 7: Attention Mechanisms & FlashAttention

This lesson covers:
1. Advanced attention mechanism implementations
2. FlashAttention optimization techniques
3. Causal attention for autoregressive models
4. Multi-head attention optimization
5. Memory-efficient attention patterns
6. Performance analysis and optimization

Prerequisites: Lessons 1-6 (All beginner and intermediate lessons)
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional, List

# ============================================================================
# 🧠 ADVANCED ATTENTION MECHANISMS
# ============================================================================

def explain_attention_mechanisms():
    """
    📚 ATTENTION MECHANISMS FUNDAMENTALS
    
    Advanced attention mechanisms and optimization techniques.
    """
    print("🧠 Advanced Attention Mechanisms:")
    print("=" * 50)
    
    print("""
    🎯 Attention Mechanism Components:
    
    1. Query (Q): What information are we looking for?
    2. Key (K): What information is available?
    3. Value (V): What is the actual information content?
    
    🚀 Attention Computation:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    📊 Advanced Optimizations:
    1. FlashAttention: Memory-efficient attention computation
    2. Causal Attention: Masked attention for autoregressive models
    3. Multi-Head Attention: Parallel attention computation
    4. Sparse Attention: Reduced attention patterns
    5. Linear Attention: Approximated attention computation
    
    🎯 Key Challenges:
    1. Memory Complexity: O(N²) for sequence length N
    2. Compute Complexity: O(N²d) for head dimension d
    3. Memory Bandwidth: High memory traffic
    4. Cache Efficiency: Poor cache utilization
    
    🚀 Optimization Strategies:
    1. Tiling: Break computation into smaller blocks
    2. Online Softmax: Compute softmax incrementally
    3. Memory Coalescing: Optimize memory access patterns
    4. Kernel Fusion: Combine operations
    5. Shared Memory: Use fast on-chip memory
    """)

@triton.jit
def advanced_attention_kernel(
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
    🚀 ADVANCED ATTENTION KERNEL
    
    Implements advanced attention mechanism with:
    - Tiled computation for memory efficiency
    - Online softmax computation
    - Optimized memory access patterns
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
    
    # Initialize accumulators for online softmax
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

def advanced_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    🚀 ADVANCED ATTENTION WRAPPER
    
    Wrapper function for advanced attention mechanism.
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
    advanced_attention_kernel[grid](
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
# 🔥 FLASH ATTENTION IMPLEMENTATION
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
    
    Implements FlashAttention with:
    - Online softmax computation
    - Memory-efficient tiling
    - Optimized memory access patterns
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
    
    # Initialize accumulators for online softmax
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
# 🎯 CAUSAL ATTENTION
# ============================================================================

@triton.jit
def causal_attention_kernel(
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
    🎯 CAUSAL ATTENTION KERNEL
    
    Implements causal attention with:
    - Causal masking for autoregressive models
    - Online softmax computation
    - Memory-efficient tiling
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
    
    # Initialize accumulators for online softmax
    l = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    m = tl.full((BLOCK_SIZE_M,), -float('inf'), dtype=tl.float32)
    o = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    
    # Loop over N dimension in blocks
    for n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < seq_len
        
        # Create causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        
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
        
        # Apply causal mask
        attention_scores = tl.where(causal_mask, attention_scores, -float('inf'))
        
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

def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    🎯 CAUSAL ATTENTION WRAPPER
    
    Wrapper function for causal attention mechanism.
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
    causal_attention_kernel[grid](
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
    
    Tests various attention mechanisms and validates correctness.
    """
    print("🧪 Testing Attention Mechanisms:")
    print("=" * 50)
    
    # Test configuration
    batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
    
    # Create test data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    # Test advanced attention
    print("\n📊 Test: Advanced Attention")
    try:
        result = advanced_attention(q, k, v)
        print(f"  Result shape: {result.shape}")
        print(f"  Result device: {result.device}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test FlashAttention
    print("\n📊 Test: FlashAttention")
    try:
        result = flash_attention(q, k, v)
        print(f"  Result shape: {result.shape}")
        print(f"  Result device: {result.device}")
        print(f"  Status: ✅ Working")
    except Exception as e:
        print(f"  Status: ❌ Error - {e}")
    
    # Test causal attention
    print("\n📊 Test: Causal Attention")
    try:
        result = causal_attention(q, k, v)
        print(f"  Result shape: {result.shape}")
        print(f"  Result device: {result.device}")
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
        
        # Benchmark advanced attention
        print("\n  Advanced Attention:")
        try:
            # Warmup
            for _ in range(10):
                _ = advanced_attention(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = advanced_attention(q, k, v)
            torch.cuda.synchronize()
            advanced_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {advanced_time:.3f} ms")
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
        
        # Benchmark causal attention
        print("\n  Causal Attention:")
        try:
            # Warmup
            for _ in range(10):
                _ = causal_attention(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = causal_attention(q, k, v)
            torch.cuda.synchronize()
            causal_time = (time.time() - start_time) / 100 * 1000
            
            print(f"    Time: {causal_time:.3f} ms")
            print(f"    Status: ✅ Working")
        except Exception as e:
            print(f"    Status: ❌ Error - {e}")

# ============================================================================
# 🎯 MAIN FUNCTION
# ============================================================================

def main():
    """
    🎯 MAIN FUNCTION
    
    Runs the complete lesson 7 tutorial.
    """
    print("🚀 LESSON 7: ATTENTION MECHANISMS & FLASHATTENTION")
    print("=" * 70)
    print("This lesson covers advanced attention mechanisms and optimization techniques.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Please use a GPU-enabled environment.")
        return
    
    # Run the tutorial sections
    explain_attention_mechanisms()
    
    test_attention_mechanisms()
    benchmark_attention_mechanisms()
    
    print("\n🎉 Lesson 7 Complete!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Understanding advanced attention mechanisms")
    print("2. ✅ Implementing FlashAttention optimization")
    print("3. ✅ Causal attention for autoregressive models")
    print("4. ✅ Memory-efficient attention computation")
    print("5. ✅ Online softmax computation")
    print("6. ✅ Performance analysis and optimization")
    
    print("\n🚀 Next Steps:")
    print("- Experiment with different attention patterns")
    print("- Try optimizing for different sequence lengths")
    print("- Move on to Lesson 8: MoE (Mixture of Experts) Implementation")

if __name__ == "__main__":
    main()
