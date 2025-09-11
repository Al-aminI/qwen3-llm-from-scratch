"""
🧠 QWEN3 CORE COMPONENTS - DETAILED EXPLANATIONS

This file contains the core neural network components with extensive explanations
to help you become an expert in modern transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from dataclasses import dataclass

# =============================================================================
# 🔄 COMPONENT 1: ROTARY POSITIONAL EMBEDDINGS (RoPE)
# =============================================================================

class Rotary(nn.Module):
    """
    🔄 ROTARY POSITIONAL EMBEDDINGS (RoPE)
    
    This is one of the most important innovations in modern transformers!
    
    🎯 What RoPE does:
    - Encodes position information by ROTATING vectors
    - Unlike traditional positional embeddings that just ADD position info
    - Allows the model to understand relative positions naturally
    
    🧮 The Math:
    - For each position i, we compute rotation angles based on i
    - We rotate the query and key vectors by these angles
    - The dot product between rotated vectors encodes relative position
    
    🔍 Why it's better:
    - Extrapolates to longer sequences (can handle 100K+ tokens)
    - Relative position encoding (position 5 vs 10 is same as 15 vs 20)
    - No learned parameters (more efficient)
    - Better performance on long sequences
    
    📐 The rotation:
    - Split embedding into pairs: [x1, x2, x3, x4] → [[x1,x2], [x3,x4]]
    - Rotate each pair by angle θ_i = i / (10000^(2j/d))
    - This creates a spiral pattern in high-dimensional space
    """
    
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        
        # Compute angular frequencies for each dimension
        # Higher frequencies for higher dimensions (like musical harmonics)
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        
        # Mirror the frequencies (mathematical requirement)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        
        # Create position indices
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # Compute rotation angles for each position and dimension
        # This creates a 2D grid: positions × frequencies
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        
        # Precompute cos and sin values (efficiency optimization)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        """
        Apply RoPE to input tensor
        
        Args:
            x_BTHD: (batch, time, heads, dim) - query or key vectors
            
        Returns:
            Rotated vectors with position information encoded
        """
        assert self.cos.size(0) >= x_BTHD.size(-3)
        
        # Get cos and sin for current sequence length
        cos = self.cos[None, :x_BTHD.size(-3), None, :]  # Add batch and head dims
        sin = self.sin[None, :x_BTHD.size(-3), None, :]
        
        # Split into pairs for rotation
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        
        # Apply rotation: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        
        # Concatenate back
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

# =============================================================================
# 🎯 COMPONENT 2: GROUPED-QUERY ATTENTION (GQA)
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    🔑 GROUPED-QUERY ATTENTION HELPER
    
    This implements the key innovation in GQA:
    - Fewer Key-Value heads than Query heads
    - Each KV head is "shared" across multiple Query heads
    - Massive memory savings with minimal performance loss
    
    🧮 Example:
    - 8 Query heads, 2 KV heads
    - KV head 1 is used by Query heads 1,2,3,4
    - KV head 2 is used by Query heads 5,6,7,8
    - Memory reduction: 75% (8→2 KV heads)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    # Add dimension for repetition
    hidden_states = hidden_states[:, :, None, :, :]
    
    # Expand to repeat each KV head
    hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # Reshape to merge repetition dimension
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen3Attention(nn.Module):
    """
    🎯 GROUPED-QUERY ATTENTION (GQA) IMPLEMENTATION
    
    This is the heart of modern transformer attention mechanisms!
    
    🧠 Key Innovations:
    1. Grouped-Query Attention: Fewer KV heads than Query heads
    2. QK-Normalization: Normalizes queries and keys for stability
    3. RoPE: Rotary positional embeddings
    4. Scaled dot-product attention: The core attention mechanism
    
    🔍 How it works:
    1. Project input to Q, K, V (separate linear layers)
    2. Apply QK normalization (Qwen3 innovation)
    3. Apply RoPE for position encoding
    4. Repeat KV heads for GQA
    5. Compute attention scores and apply to values
    6. Project back to model dimension
    
    📊 Memory Efficiency:
    - Traditional: 8 Q heads + 8 K heads + 8 V heads = 24 heads
    - GQA: 8 Q heads + 2 K heads + 2 V heads = 12 heads
    - Savings: 50% reduction in attention memory!
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # QK-Normalization (Qwen3 innovation for training stability)
        self.q_norm = RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_k, eps=config.rms_norm_eps)

        # RoPE for position encoding
        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # 1. Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, n_heads * d_k)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * d_k)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * d_k)

        # 2. Reshape into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # 3. Apply QK-Normalization (Qwen3 innovation)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply RoPE (position encoding)
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # 5. Transpose for attention computation
        Q = q.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        K = k.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)
        V = v.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)

        # 6. Repeat K and V heads for GQA
        K = repeat_kv(K, self.n_kv_groups)
        V = repeat_kv(V, self.n_kv_groups)

        # 7. Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=True,  # Mask future tokens (language modeling)
            dropout_p=self.dropout if self.training else 0.0
        )

        # 8. Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

# =============================================================================
# 🔥 COMPONENT 3: SWIGLU FEED-FORWARD NETWORK
# =============================================================================

class SwiGLUFeedForward(nn.Module):
    """
    🔥 SWIGLU FEED-FORWARD NETWORK
    
    SwiGLU is a modern activation function that combines:
    - Swish activation: x * sigmoid(x) (smooth, non-monotonic)
    - GLU (Gated Linear Unit): element-wise multiplication with a gate
    
    🧮 The Math:
    SwiGLU(x) = Swish(W1(x)) ⊙ W2(x)
    Where:
    - W1(x) is the "gate" (controls information flow)
    - W2(x) is the "value" (the actual transformation)
    - ⊙ is element-wise multiplication
    - Swish(x) = x * sigmoid(x)
    
    🎯 Why SwiGLU is better:
    - More expressive than ReLU (can represent more complex functions)
    - Smooth gradients (better for training)
    - Gating mechanism (selective information flow)
    - Used in state-of-the-art models (PaLM, LLaMA, Qwen)
    
    🔍 The gating mechanism:
    - Gate controls how much information flows through
    - Value provides the actual transformation
    - Together they create selective, adaptive processing
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Three linear layers for SwiGLU
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # Output projection
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)    # Value projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        SwiGLU forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            SwiGLU output (batch, seq_len, d_model)
        """
        # Compute gate and value
        gate = self.gate_proj(x)    # Gate: controls information flow
        value = self.up_proj(x)     # Value: actual transformation
        
        # Apply Swish activation to gate, then multiply with value
        activated_gate = F.silu(gate)  # Swish activation
        gated_value = activated_gate * value  # Element-wise multiplication
        
        # Apply dropout and project back to model dimension
        return self.down_proj(self.dropout(gated_value))

# =============================================================================
# 🏗️ COMPONENT 4: TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    🏗️ TRANSFORMER BLOCK - THE BUILDING BLOCK OF MODERN LLMs
    
    This combines all the components into a complete transformer layer:
    
    🧠 Architecture:
    1. Pre-norm attention (RMSNorm before attention)
    2. Residual connection (x + attention(x))
    3. Pre-norm feed-forward (RMSNorm before SwiGLU)
    4. Residual connection (x + feedforward(x))
    
    🔍 Pre-norm vs Post-norm:
    - Pre-norm: norm(input) → attention → output + input
    - Post-norm: attention(input) → norm → output + input
    - Pre-norm is more stable for deep networks
    - Qwen3 uses pre-norm architecture
    
    📊 Why this works:
    - Residual connections prevent vanishing gradients
    - Pre-norm provides stable gradients
    - Dropout prevents overfitting
    - Each component is optimized for its specific task
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = Qwen3Attention(config)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        
        # Pre-norm architecture (RMSNorm before each sub-layer)
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Transformer block forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

# =============================================================================
# 🎯 COMPONENT 5: RMSNorm (Root Mean Square Normalization)
# =============================================================================

class RMSNorm(nn.Module):
    """
    📐 RMSNorm - ROOT MEAN SQUARE NORMALIZATION
    
    This is a modern alternative to LayerNorm that's more efficient:
    
    🧮 The Math:
    RMSNorm(x) = x / sqrt(mean(x²) + ε) * g
    
    Where:
    - x is the input
    - mean(x²) is the mean of squared values
    - ε is a small constant (1e-6)
    - g is a learnable scale parameter
    
    🎯 Why RMSNorm is better:
    - Simpler than LayerNorm (no centering)
    - More efficient (fewer operations)
    - Better numerical stability
    - Used in modern models (LLaMA, Qwen, etc.)
    
    🔍 Comparison with LayerNorm:
    - LayerNorm: (x - mean(x)) / std(x) * g + b
    - RMSNorm: x / sqrt(mean(x²)) * g
    - RMSNorm is simpler and often works better
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        RMSNorm forward pass
        
        Args:
            x: Input tensor (..., dim)
            
        Returns:
            Normalized tensor (..., dim)
        """
        # Compute mean of squared values
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        
        # Normalize
        x = x * torch.rsqrt(mean_sq + self.eps)
        
        # Apply learnable scale
        return self.weight * x

if __name__ == "__main__":
    print("🧠 Qwen3 Core Components Ready!")
    print("\nKey Components Explained:")
    print("1. 🔄 RoPE: Rotary Positional Embeddings for position encoding")
    print("2. 🎯 GQA: Grouped-Query Attention for memory efficiency")
    print("3. 🔥 SwiGLU: Modern activation function with gating")
    print("4. 🏗️ Transformer Block: Complete transformer layer")
    print("5. 📐 RMSNorm: Efficient normalization technique")
    print("\nEach component is optimized for modern transformer architectures!")
