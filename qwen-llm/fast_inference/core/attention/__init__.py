"""
Optimized attention layers with KV caching.

This module provides attention implementations optimized for fast inference:
- CachedAttention: Simple attention with KV caching
- OptimizedAttention: Advanced attention with paged KV caching
"""

from .cached_attention import CachedAttention
from .optimized_attention import OptimizedAttention

__all__ = [
    "CachedAttention",
    "OptimizedAttention"
]
