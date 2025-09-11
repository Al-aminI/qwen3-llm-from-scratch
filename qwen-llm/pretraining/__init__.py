"""
🎯 Pretraining Package

This package contains all components for training Qwen3-style language models from scratch.
"""

from .core.model.minimal_llm import MinimalLLM
from .core.model.components import (
    Qwen3Attention, 
    SwiGLUFeedForward, 
    TransformerBlock, 
    RMSNorm, 
    Rotary
)
from .core.training.trainer import PretrainingTrainer
from .core.training.optimizer import Muon, setup_muon_optimizer
from .core.config.config import PretrainingConfig, set_seed
from .utils.data import load_and_cache_data, TextTokenDataset
from .utils.generation import generate_text

__version__ = "1.0.0"
__author__ = "Qwen3 Implementation"

__all__ = [
    # Model components
    "MinimalLLM",
    "Qwen3Attention",
    "SwiGLUFeedForward", 
    "TransformerBlock",
    "RMSNorm",
    "Rotary",
    
    # Training
    "PretrainingTrainer",
    "Muon",
    "setup_muon_optimizer",
    
    # Configuration
    "PretrainingConfig",
    "set_seed",
    
    # Utilities
    "load_and_cache_data",
    "TextTokenDataset", 
    "generate_text"
]
