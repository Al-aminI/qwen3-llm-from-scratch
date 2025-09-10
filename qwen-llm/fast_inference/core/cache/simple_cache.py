"""
Simple KV Cache implementation for fast inference.

This module provides a straightforward KV cache that stores key-value pairs
for each sequence position. Much simpler than paged attention but still effective.
"""

import torch
from typing import Tuple
from collections import defaultdict


class SimpleKVCache:
    """
    🎯 SIMPLE KV CACHE
    
    A straightforward KV cache implementation that stores key-value pairs
    for each sequence position. Much simpler than paged attention but still effective.
    """
    
    def __init__(self, max_seq_len: int, n_heads: int, head_dim: int, dtype: torch.dtype, device: str):
        """
        Initialize simple KV cache.
        
        Args:
            max_seq_len: Maximum sequence length
            n_heads: Number of attention heads
            head_dim: Dimension of each head
            dtype: Data type for cache tensors
            device: Device to store cache on
        """
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # Cache storage: [seq_id][layer_idx] -> (k_cache, v_cache)
        # k_cache, v_cache: (n_heads, max_seq_len, head_dim)
        self.cache = defaultdict(lambda: defaultdict(lambda: (
            torch.zeros(n_heads, max_seq_len, head_dim, dtype=dtype, device=device),
            torch.zeros(n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        )))
        
        # Track sequence lengths
        self.seq_lengths = defaultdict(int)
    
    def get_cache(self, seq_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached K, V for a sequence and layer.
        
        Args:
            seq_id: Sequence ID
            layer_idx: Layer index
            
        Returns:
            Tuple of (k_cache, v_cache) tensors
        """
        return self.cache[seq_id][layer_idx]
    
    def update_cache(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int):
        """
        Update cache with new K, V values.
        
        Args:
            seq_id: Sequence ID
            layer_idx: Layer index
            k: Key tensor (n_heads, seq_len, head_dim)
            v: Value tensor (n_heads, seq_len, head_dim)
            pos: Position to update
        """
        k_cache, v_cache = self.cache[seq_id][layer_idx]
        
        # Update cache at position
        k_cache[:, pos:pos + k.size(1), :] = k
        v_cache[:, pos:pos + v.size(1), :] = v
        
        # Update sequence length
        self.seq_lengths[seq_id] = max(self.seq_lengths[seq_id], pos + k.size(1))
    
    def get_cached_kv(self, seq_id: int, layer_idx: int, current_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached K, V up to current length.
        
        Args:
            seq_id: Sequence ID
            layer_idx: Layer index
            current_len: Current sequence length
            
        Returns:
            Tuple of (k_cache, v_cache) tensors up to current_len
        """
        k_cache, v_cache = self.cache[seq_id][layer_idx]
        return k_cache[:, :current_len, :], v_cache[:, :current_len, :]
    
    def clear_sequence(self, seq_id: int):
        """
        Clear cache for a sequence.
        
        Args:
            seq_id: Sequence ID to clear
        """
        if seq_id in self.cache:
            del self.cache[seq_id]
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]
    
    def get_sequence_length(self, seq_id: int) -> int:
        """
        Get the current length of a sequence.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Current sequence length
        """
        return self.seq_lengths.get(seq_id, 0)
    
    def has_sequence(self, seq_id: int) -> bool:
        """
        Check if a sequence exists in cache.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            True if sequence exists in cache
        """
        return seq_id in self.cache
    
    def get_memory_usage(self) -> dict:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        total_elements = 0
        total_sequences = len(self.cache)
        
        for seq_cache in self.cache.values():
            for k_cache, v_cache in seq_cache.values():
                total_elements += k_cache.numel() + v_cache.numel()
        
        # Estimate memory usage (assuming float16)
        memory_mb = total_elements * 2 / (1024 * 1024)  # 2 bytes per float16
        
        return {
            'total_sequences': total_sequences,
            'total_elements': total_elements,
            'memory_mb': memory_mb,
            'avg_sequence_length': sum(self.seq_lengths.values()) / max(total_sequences, 1)
        }
