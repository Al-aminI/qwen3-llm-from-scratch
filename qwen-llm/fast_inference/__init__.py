"""
🚀 Fast Inference Engine for Qwen3

A high-performance inference engine with KV caching for fast text generation.
Provides 10-100x speedup over naive inference methods.

Key Features:
- KV caching for memory efficiency
- Simple and advanced inference modes
- Comprehensive benchmarking tools
- Easy integration with existing models

Usage:
    from fast_inference import SimpleFastInference, create_simple_fast_inference
    
    # Create engine
    engine = create_simple_fast_inference("model.pt", "tokenizer")
    
    # Generate text
    result = engine.generate_single("Hello, world!", max_new_tokens=50)
"""

__version__ = "1.0.0"
__author__ = "Qwen3 Fast Inference Team"
__email__ = "team@qwen3-inference.com"

# Core imports
from .core.engine import SimpleFastInference, FastInferenceEngine
from .core.engine.universal_engine import UniversalFastInference, create_universal_fast_inference
from .core.engine.vllm_style_engine import VLLMStyleEngine, create_vllm_style_engine, SchedulerPolicy
from .core.engine.universal_vllm_engine import UniversalVLLMStyleEngine, create_universal_vllm_style_engine
from .core.cache import SimpleKVCache, PagedKVCache
from .core.attention import CachedAttention, OptimizedAttention
from .utils.sampling import SamplingParams, sample_tokens
from .utils.benchmarking import benchmark_inference, compare_methods

# Convenience functions
from .core.engine import create_simple_fast_inference, create_fast_inference_engine

__all__ = [
    # Core classes
    "SimpleFastInference",
    "FastInferenceEngine",
    "UniversalFastInference",
    "VLLMStyleEngine",
    "UniversalVLLMStyleEngine",
    "SimpleKVCache",
    "PagedKVCache",
    "CachedAttention",
    "OptimizedAttention",
    
    # Utilities
    "SamplingParams",
    "sample_tokens",
    "benchmark_inference",
    "compare_methods",
    "SchedulerPolicy",
    
    # Convenience functions
    "create_simple_fast_inference",
    "create_fast_inference_engine",
    "create_universal_fast_inference",
    "create_vllm_style_engine",
    "create_universal_vllm_style_engine",
    
    # Version info
    "__version__",
    "__author__",
    "__email__"
]
