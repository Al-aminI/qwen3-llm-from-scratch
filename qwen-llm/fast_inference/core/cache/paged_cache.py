"""
Paged KV Cache implementation for advanced inference.

This module provides a paged KV cache implementation inspired by vLLM's PagedAttention.
More sophisticated than simple cache but provides better memory management for multiple sequences.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


class PagedKVCache(nn.Module):
    """
    🧠 PAGED KV CACHE
    
    Efficient memory management for KV cache using page-based allocation.
    Inspired by vLLM's PagedAttention but simplified for our use case.
    """
    
    def __init__(self, n_pages: int, page_size: int, n_heads: int, head_dim: int, dtype: torch.dtype, device: str):
        """
        Initialize paged KV cache.
        
        Args:
            n_pages: Number of pages to allocate
            page_size: Size of each page in tokens
            n_heads: Number of attention heads
            head_dim: Dimension of each head
            dtype: Data type for cache tensors
            device: Device to store cache on
        """
        super().__init__()
        self.n_pages = n_pages
        self.page_size = page_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # Allocate cache memory in pages
        cache_shape = (1, n_heads, n_pages * page_size, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        
        # Page table: [batch_idx, logical_page] -> physical_page
        self.page_table = -torch.ones((256, n_pages), dtype=torch.long, device=device)  # Max 256 sequences
        self.free_pages = list(range(n_pages))
        self.sequence_pages = {}  # seq_id -> list of allocated pages
        
    def allocate_pages(self, seq_id: int, num_pages: int) -> List[int]:
        """
        Allocate pages for a sequence.
        
        Args:
            seq_id: Sequence ID
            num_pages: Number of pages to allocate
            
        Returns:
            List of allocated page indices
            
        Raises:
            RuntimeError: If not enough free pages available
        """
        if len(self.free_pages) < num_pages:
            raise RuntimeError(f"Not enough free pages. Need {num_pages}, have {len(self.free_pages)}")
        
        allocated_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self.sequence_pages[seq_id] = allocated_pages
        
        # Update page table
        for i, page in enumerate(allocated_pages):
            self.page_table[seq_id, i] = page
            
        return allocated_pages
    
    def deallocate_pages(self, seq_id: int):
        """
        Deallocate pages for a sequence.
        
        Args:
            seq_id: Sequence ID to deallocate
        """
        if seq_id in self.sequence_pages:
            self.free_pages.extend(self.sequence_pages[seq_id])
            self.page_table[seq_id, :] = -1
            del self.sequence_pages[seq_id]
    
    def get_physical_positions(self, seq_id: int, logical_positions: torch.Tensor) -> torch.Tensor:
        """
        Convert logical positions to physical cache positions.
        
        Args:
            seq_id: Sequence ID
            logical_positions: Logical position indices
            
        Returns:
            Physical position indices in cache
        """
        page_indices = logical_positions // self.page_size
        offsets = logical_positions % self.page_size
        
        physical_pages = self.page_table[seq_id, page_indices]
        physical_positions = physical_pages * self.page_size + offsets
        
        return physical_positions
    
    def update_cache(self, seq_id: int, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Update KV cache at specific positions.
        
        Args:
            seq_id: Sequence ID
            positions: Position indices to update
            k: Key tensor
            v: Value tensor
        """
        physical_positions = self.get_physical_positions(seq_id, positions)
        
        # Update cache
        self.k_cache[0, :, physical_positions, :] = k
        self.v_cache[0, :, physical_positions, :] = v
    
    def get_cached_kv(self, seq_id: int, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached K, V up to specified length.
        
        Args:
            seq_id: Sequence ID
            max_length: Maximum length to retrieve
            
        Returns:
            Tuple of (k_cache, v_cache) tensors
        """
        if seq_id not in self.sequence_pages:
            # Return empty cache if sequence not found
            empty_k = torch.zeros(1, self.n_heads, 0, self.head_dim, device=self.device)
            empty_v = torch.zeros(1, self.n_heads, 0, self.head_dim, device=self.device)
            return empty_k, empty_v
        
        # Get all positions up to max_length
        positions = torch.arange(max_length, device=self.device)
        physical_positions = self.get_physical_positions(seq_id, positions)
        
        # Extract cached values
        k_cache = self.k_cache[0, :, physical_positions, :].unsqueeze(0)
        v_cache = self.v_cache[0, :, physical_positions, :].unsqueeze(0)
        
        return k_cache, v_cache
    
    def get_memory_usage(self) -> dict:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        total_pages = self.n_pages
        used_pages = sum(len(pages) for pages in self.sequence_pages.values())
        free_pages = len(self.free_pages)
        
        # Calculate memory usage
        total_elements = total_pages * self.page_size * self.n_heads * self.head_dim
        used_elements = used_pages * self.page_size * self.n_heads * self.head_dim
        
        # Estimate memory usage (assuming float16)
        total_memory_mb = total_elements * 2 / (1024 * 1024)
        used_memory_mb = used_elements * 2 / (1024 * 1024)
        
        return {
            'total_pages': total_pages,
            'used_pages': used_pages,
            'free_pages': free_pages,
            'total_sequences': len(self.sequence_pages),
            'total_memory_mb': total_memory_mb,
            'used_memory_mb': used_memory_mb,
            'utilization': used_pages / total_pages if total_pages > 0 else 0
        }
    
    def clear_all(self):
        """Clear all cached data."""
        self.free_pages = list(range(self.n_pages))
        self.sequence_pages.clear()
        self.page_table.fill_(-1)
        self.k_cache.zero_()
        self.v_cache.zero_()
