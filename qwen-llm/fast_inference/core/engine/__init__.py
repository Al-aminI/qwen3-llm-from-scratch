"""
Fast Inference Engine Package

This package provides various inference engines for high-performance text generation:
- SimpleFastInference: Basic KV caching for single sequences
- FastInferenceEngine: Advanced paged attention with continuous batching
- UniversalFastInference: Universal model support (any model)
- UniversalVLLMStyleEngine: vLLM-style features with universal model support
- OpenAI-compatible API server for production deployment
"""

from .simple_engine import SimpleFastInference, create_simple_fast_inference
from .advanced_engine import FastInferenceEngine, create_fast_inference_engine
from .universal_engine import UniversalFastInference, create_universal_fast_inference
from .universal_vllm_engine import UniversalVLLMStyleEngine, create_universal_vllm_style_engine
from .vllm_style_engine import VLLMStyleEngine, create_vllm_style_engine

# OpenAI-compatible API components
from .openai_protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    EmbeddingRequest, EmbeddingResponse,
    TokenizeRequest, TokenizeResponse,
    DetokenizeRequest, DetokenizeResponse,
    HealthResponse, ModelList, ErrorResponse
)
from .openai_serving_engine import OpenAIServingEngine
from .openai_api_server import app as openai_api_app

__all__ = [
    # Core engines
    "SimpleFastInference",
    "FastInferenceEngine", 
    "UniversalFastInference",
    "UniversalVLLMStyleEngine",
    "VLLMStyleEngine",
    
    # Factory functions
    "create_simple_fast_inference",
    "create_fast_inference_engine",
    "create_universal_fast_inference", 
    "create_universal_vllm_style_engine",
    "create_vllm_style_engine",
    
    # OpenAI API components
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "CompletionRequest", 
    "CompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "TokenizeRequest",
    "TokenizeResponse", 
    "DetokenizeRequest",
    "DetokenizeResponse",
    "HealthResponse",
    "ModelList",
    "ErrorResponse",
    "OpenAIServingEngine",
    "openai_api_app"
]