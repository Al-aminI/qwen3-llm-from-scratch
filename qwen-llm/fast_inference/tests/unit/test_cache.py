"""
Unit tests for KV cache implementations.
"""

import pytest
import torch
from fast_inference.core.cache import SimpleKVCache, PagedKVCache


class TestSimpleKVCache:
    """Test cases for SimpleKVCache."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = SimpleKVCache(
            max_seq_len=100,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        assert cache.max_seq_len == 100
        assert cache.n_heads == 8
        assert cache.head_dim == 64
        assert cache.device == "cpu"
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = SimpleKVCache(
            max_seq_len=100,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        seq_id = 1
        layer_idx = 0
        
        # Test initial state
        assert not cache.has_sequence(seq_id)
        assert cache.get_sequence_length(seq_id) == 0
        
        # Test cache update
        k = torch.randn(8, 10, 64, dtype=torch.float16)
        v = torch.randn(8, 10, 64, dtype=torch.float16)
        
        cache.update_cache(seq_id, layer_idx, k, v, pos=0)
        
        assert cache.has_sequence(seq_id)
        assert cache.get_sequence_length(seq_id) == 10
        
        # Test cache retrieval
        cached_k, cached_v = cache.get_cached_kv(seq_id, layer_idx, 10)
        assert cached_k.shape == (8, 10, 64)
        assert cached_v.shape == (8, 10, 64)
        
        # Test cache clearing
        cache.clear_sequence(seq_id)
        assert not cache.has_sequence(seq_id)
        assert cache.get_sequence_length(seq_id) == 0
    
    def test_memory_usage(self):
        """Test memory usage tracking."""
        cache = SimpleKVCache(
            max_seq_len=100,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        # Add some data
        k = torch.randn(8, 20, 64, dtype=torch.float16)
        v = torch.randn(8, 20, 64, dtype=torch.float16)
        
        cache.update_cache(1, 0, k, v, pos=0)
        cache.update_cache(2, 0, k, v, pos=0)
        
        stats = cache.get_memory_usage()
        
        assert stats['total_sequences'] == 2
        assert stats['total_elements'] > 0
        assert stats['memory_mb'] > 0
        assert stats['avg_sequence_length'] == 20


class TestPagedKVCache:
    """Test cases for PagedKVCache."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = PagedKVCache(
            n_pages=10,
            page_size=128,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        assert cache.n_pages == 10
        assert cache.page_size == 128
        assert cache.n_heads == 8
        assert cache.head_dim == 64
        assert cache.device == "cpu"
    
    def test_page_allocation(self):
        """Test page allocation and deallocation."""
        cache = PagedKVCache(
            n_pages=10,
            page_size=128,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        seq_id = 1
        num_pages = 3
        
        # Test page allocation
        allocated_pages = cache.allocate_pages(seq_id, num_pages)
        assert len(allocated_pages) == num_pages
        assert seq_id in cache.sequence_pages
        
        # Test page deallocation
        cache.deallocate_pages(seq_id)
        assert seq_id not in cache.sequence_pages
    
    def test_insufficient_pages(self):
        """Test handling of insufficient pages."""
        cache = PagedKVCache(
            n_pages=2,
            page_size=128,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        # Allocate all pages
        cache.allocate_pages(1, 2)
        
        # Try to allocate more pages
        with pytest.raises(RuntimeError, match="Not enough free pages"):
            cache.allocate_pages(2, 1)
    
    def test_cache_operations(self):
        """Test cache operations with paged storage."""
        cache = PagedKVCache(
            n_pages=10,
            page_size=128,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        seq_id = 1
        cache.allocate_pages(seq_id, 2)
        
        # Test cache update
        positions = torch.tensor([0, 1, 2])
        k = torch.randn(8, 3, 64, dtype=torch.float16)
        v = torch.randn(8, 3, 64, dtype=torch.float16)
        
        cache.update_cache(seq_id, positions, k, v)
        
        # Test cache retrieval
        cached_k, cached_v = cache.get_cached_kv(seq_id, 3)
        assert cached_k.shape == (1, 8, 3, 64)
        assert cached_v.shape == (1, 8, 3, 64)
    
    def test_memory_usage(self):
        """Test memory usage tracking."""
        cache = PagedKVCache(
            n_pages=10,
            page_size=128,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        # Allocate some pages
        cache.allocate_pages(1, 3)
        cache.allocate_pages(2, 2)
        
        stats = cache.get_memory_usage()
        
        assert stats['total_pages'] == 10
        assert stats['used_pages'] == 5
        assert stats['free_pages'] == 5
        assert stats['total_sequences'] == 2
        assert stats['utilization'] == 0.5
    
    def test_clear_all(self):
        """Test clearing all cached data."""
        cache = PagedKVCache(
            n_pages=10,
            page_size=128,
            n_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device="cpu"
        )
        
        # Allocate some pages
        cache.allocate_pages(1, 3)
        cache.allocate_pages(2, 2)
        
        # Clear all
        cache.clear_all()
        
        stats = cache.get_memory_usage()
        assert stats['used_pages'] == 0
        assert stats['free_pages'] == 10
        assert stats['total_sequences'] == 0
