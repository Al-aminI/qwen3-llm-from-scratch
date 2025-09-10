"""
Optimized attention layer with paged KV caching.

This module provides an advanced attention layer with paged KV caching for maximum performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.sampling import SamplingParams
    from ..cache.paged_cache import PagedKVCache


class OptimizedAttention(nn.Module):
    """
    🎯 OPTIMIZED ATTENTION WITH PAGED KV CACHE
    
    Advanced attention layer with paged KV caching for maximum performance.
    """
    
    def __init__(self, config, kv_cache: 'PagedKVCache'):
        """
        Initialize optimized attention layer.
        
        Args:
            config: Model configuration
            kv_cache: Paged KV cache instance
        """
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k
        self.kv_cache = kv_cache

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Normalization
        self.q_norm = nn.LayerNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.LayerNorm(self.d_k, eps=config.rms_norm_eps)

        # RoPE (simplified - you may want to import from your core components)
        self.rotary = self._create_rotary_embedding(config)
        self.dropout = config.dropout

    def _create_rotary_embedding(self, config):
        """Create rotary embedding (simplified version)."""
        # This is a placeholder - you should import your actual Rotary class
        class SimpleRotary:
            def __init__(self, dim, max_seq_len):
                self.dim = dim
                self.max_seq_len = max_seq_len
                
            def __call__(self, x):
                # Simplified rotary embedding - replace with actual implementation
                return x
        
        return SimpleRotary(config.d_k, config.max_seq_len)

    def forward(self, x: torch.Tensor, seq_id: int, positions: torch.Tensor, use_cache: bool = True):
        """
        Forward pass with paged KV caching.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            seq_id: Sequence ID for cache lookup
            positions: Position indices for cache
            use_cache: Whether to use cached KV values
            
        Returns:
            Attention output tensor
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Update KV cache
        if use_cache:
            self.kv_cache.update_cache(seq_id, positions, k[0], v[0])

        # Get cached K, V for attention
        if use_cache and seq_id in self.kv_cache.sequence_pages:
            # Retrieve cached values
            cached_k = self.kv_cache.k_cache[0, :, :positions.max() + 1, :]
            cached_v = self.kv_cache.v_cache[0, :, :positions.max() + 1, :]
            
            # Use cached values for attention
            K = cached_k.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, cached_len, d_k)
            V = cached_v.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, cached_len, d_k)
        else:
            K = k.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)
            V = v.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)

        # Repeat KV heads for GQA
        K = self._repeat_kv(K, self.n_kv_groups)
        V = self._repeat_kv(V, self.n_kv_groups)

        # Transpose Q for attention
        Q = q.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads for Grouped Query Attention.
        
        Args:
            hidden_states: KV tensor (batch, n_kv_heads, seq_len, head_dim)
            n_rep: Number of repetitions
            
        Returns:
            Repeated KV tensor
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
