"""
Utility modules for fast inference.

This package contains utility functions and classes:
- Sampling: Token sampling and generation utilities
- Benchmarking: Performance measurement and comparison tools
"""

from .sampling import SamplingParams, sample_tokens
from .benchmarking import benchmark_inference, compare_methods

__all__ = [
    "SamplingParams",
    "sample_tokens",
    "benchmark_inference", 
    "compare_methods"
]
