"""
🚀 Triton Tutorials Package

A comprehensive tutorial series for learning Triton from beginner to expert level.
This package provides structured lessons, examples, and benchmarks for CUDA kernel
optimization using Triton.

Structure:
- lessons/: Progressive tutorials from beginner to expert
- examples/: Practical examples and use cases
- benchmarks/: Performance comparison tools
- tests/: Unit tests and validation
- utils/: Helper functions and utilities
- docs/: Documentation and guides

Author: Qwen3 Triton Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Qwen3 Triton Team"
__email__ = "team@qwen3-triton.com"

# Import key components
from .lessons.beginner import *
from .lessons.intermediate import *
from .lessons.advanced import *
from .lessons.expert import *

__all__ = [
    "__version__",
    "__author__", 
    "__email__"
]
