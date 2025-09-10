"""
Training configuration classes.

This module provides configuration classes for LoRA and QLoRA training,
following the original fine-tuning approach.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class LoRATrainingConfig:
    """
    🎯 LORA TRAINING CONFIGURATION
    
    Configuration for LoRA fine-tuning following the original approach.
    """
    # Model parameters
    pretrained_model_path: str = "models/final_model1.pt"
    tokenizer_path: str = "HuggingFaceTB/SmolLM-135M"
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_seq_len: int = 256
    
    # Data parameters
    dataset_name: str = "imdb"
    num_samples: int = 1000  # For demo, use subset
    
    # Technical
    use_amp: bool = True
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    dataloader_num_workers: int = 2
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]


@dataclass
class QLoRATrainingConfig:
    """
    🎯 QLORA TRAINING CONFIGURATION
    
    Configuration for QLoRA fine-tuning following the original approach.
    """
    # Model parameters
    pretrained_model_path: str = "models/final_model1.pt"
    tokenizer_path: str = "HuggingFaceTB/SmolLM-135M"
    
    # QLoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    qlora_bits: int = 4  # 4-bit quantization
    target_modules: List[str] = None
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 2e-4  # Higher LR for QLoRA
    num_epochs: int = 3
    max_seq_len: int = 256
    
    # Data parameters
    dataset_name: str = "imdb"
    num_samples: int = 1000  # For demo, use subset
    
    # Technical
    use_amp: bool = True
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    dataloader_num_workers: int = 2
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]


def create_lora_config(
    pretrained_model_path: str = "models/final_model1.pt",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    batch_size: int = 8,
    num_samples: int = 1000
) -> LoRATrainingConfig:
    """
    Create LoRA training configuration with custom parameters.
    
    Args:
        pretrained_model_path: Path to pre-trained model
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_samples: Number of training samples
        
    Returns:
        LoRATrainingConfig instance
    """
    return LoRATrainingConfig(
        pretrained_model_path=pretrained_model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_samples=num_samples
    )


def create_qlora_config(
    pretrained_model_path: str = "models/final_model1.pt",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    qlora_bits: int = 4,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 8,
    num_samples: int = 1000
) -> QLoRATrainingConfig:
    """
    Create QLoRA training configuration with custom parameters.
    
    Args:
        pretrained_model_path: Path to pre-trained model
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        qlora_bits: QLoRA quantization bits
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_samples: Number of training samples
        
    Returns:
        QLoRATrainingConfig instance
    """
    return QLoRATrainingConfig(
        pretrained_model_path=pretrained_model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        qlora_bits=qlora_bits,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_samples=num_samples
    )