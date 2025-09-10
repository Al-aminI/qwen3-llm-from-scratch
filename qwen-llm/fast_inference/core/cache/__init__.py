"""
KV Cache implementations for fast inference.

This module provides different KV cache implementations:
- SimpleKVCache: Straightforward cache for single sequences
- PagedKVCache: Advanced paged cache for multiple sequences
"""

from .simple_cache import SimpleKVCache
from .paged_cache import PagedKVCache

__all__ = [
    "SimpleKVCache",
    "PagedKVCache"
]
