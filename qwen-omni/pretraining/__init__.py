"""
🎯 Multimodal Pretraining Package

This package contains all components for training multimodal Qwen3-style language models
that can process both text and audio tokens using SNAC audio tokenization.
"""

from .core.model.minimal_llm import MultimodalLLM
from .core.model.components import (
    Qwen3Attention, 
    SwiGLUFeedForward, 
    TransformerBlock, 
    RMSNorm, 
    Rotary
)
from .core.training.multimodal_trainer import MultimodalPretrainingTrainer
from .core.training.optimizer import Muon, setup_muon_optimizer
from .core.config.config import MultimodalPretrainingConfig, set_seed
from .core.audio.snac_tokenizer import SNACTokenizer
from .core.dataset.multimodal_dataset import MultimodalDataset, LibriSpeechDataset, create_multimodal_dataloader
from .utils.data_simplified import TextTokenDataset, load_and_cache_librispeech_simplified
from .utils.generation import generate_text

__version__ = "1.0.0"
__author__ = "QWEN-OMNI Implementation"

__all__ = [
    # Model components
    "MultimodalLLM",
    "Qwen3Attention",
    "SwiGLUFeedForward", 
    "TransformerBlock",
    "RMSNorm",
    "Rotary",
    
    # Training
    "MultimodalPretrainingTrainer",
    "Muon",
    "setup_muon_optimizer",
    
    # Configuration
    "MultimodalPretrainingConfig",
    "set_seed",
    
    # Audio processing
    "SNACTokenizer",
    
    # Datasets
    "MultimodalDataset",
    "LibriSpeechDataset", 
    "create_multimodal_dataloader",
    
    # Utilities
    "load_and_cache_data",
    "TextTokenDataset", 
    "generate_text"
]
