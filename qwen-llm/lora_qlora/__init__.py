"""
🎯 LoRA & QLoRA Fine-tuning Package

A comprehensive package for efficient fine-tuning using LoRA (Low-Rank Adaptation)
and QLoRA (Quantized LoRA) techniques. Provides 1000x reduction in trainable parameters
and 8x memory reduction for training large language models.

Key Features:
- LoRA: Low-rank adaptation with minimal trainable parameters
- QLoRA: 4-bit quantization + LoRA for maximum efficiency
- Comprehensive quantization support (4-bit, 8-bit, 16-bit)
- Production-ready training and serving pipelines
- Extensive benchmarking and evaluation tools

Usage:
    from lora_qlora import LoRAManager, QLoRAManager, create_lora_trainer
    
    # LoRA fine-tuning
    lora_manager = LoRAManager(model, config)
    lora_manager.apply_lora()
    
    # QLoRA fine-tuning
    qlora_manager = QLoRAManager(model, config)
    qlora_manager.apply_qlora()
    
    # Training
    trainer = create_lora_trainer(model, config)
    trainer.train()
"""

__version__ = "1.0.0"
__author__ = "LoRA & QLoRA Fine-tuning Team"
__email__ = "team@lora-qlora.com"

# Core imports
from .core.quantization import QuantizationExpert, QuantizationConfig
from .core.lora import LoRALayer, LoRALinear, LoRAManager
from .core.qlora import QLoRALayer, QLoRALinear, QLoRAManager
from .core.training import (
    LoRATrainingConfig, QLoRATrainingConfig,
    LoRADataset, QLoRADataset,
    LoRATrainer, QLoRATrainer
)

# Utilities
from .utils.config import load_config, save_config, merge_configs
from .utils.data import load_data, preprocess_data, split_data
from .utils.serving import ModelServer, InferenceEngine

__all__ = [
    # Core classes
    "QuantizationExpert",
    "QuantizationConfig",
    "LoRALayer",
    "LoRALinear", 
    "LoRAManager",
    "QLoRALayer",
    "QLoRALinear",
    "QLoRAManager",
    # Training components
    "LoRATrainingConfig",
    "QLoRATrainingConfig",
    "LoRADataset",
    "QLoRADataset",
    "LoRATrainer",
    "QLoRATrainer",
    # Utilities
    "load_config",
    "save_config",
    "merge_configs",
    "load_data",
    "preprocess_data",
    "split_data",
    "ModelServer",
    "InferenceEngine",
    # Version info
    "__version__",
    "__author__",
    "__email__"
]
