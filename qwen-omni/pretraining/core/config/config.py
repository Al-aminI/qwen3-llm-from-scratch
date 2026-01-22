"""
🎯 PRETRAINING CONFIGURATION

This module contains the configuration classes for pretraining.
"""

import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class MultimodalPretrainingConfig:
    """
    🎯 MULTIMODAL PRETRAINING CONFIGURATION
    
    This configuration is optimized for:
    - Multimodal training with text and audio
    - Fast training on CPU/limited GPU
    - Learning the concepts without waiting hours
    - Still demonstrating all key components
    """
    # Model architecture - MUCH SMALLER
    d_model: int = 128          # Reduced from 384 (3x smaller)
    n_heads: int = 4            # Reduced from 8 (2x smaller)
    n_layers: int = 3           # Reduced from 6 (2x smaller)
    d_ff: int = 512             # Reduced from 1536 (3x smaller)
    
    # Training parameters - FASTER
    batch_size: int = 8         # Reduced from 24 (3x smaller)
    max_steps: int = 1000        # Reduced from 2000 (4x smaller)
    gradient_accumulation_steps: int = 2  # Reduced from 4

    # Qwen3-like parameters
    n_kv_heads: int = 2         # For Grouped-Query Attention (GQA)
    sliding_window: int = 1024   # Smaller context window
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    # Training parameters
    muon_lr: float = 0.01

    # Data parameters - MUCH SMALLER DATASET
    max_seq_len: int = 256      # Reduced from 512 (2x smaller)
    num_documents: int = 2000    # Reduced from 2000 (10x smaller!)
    max_tokens: int = 50000     # Reduced from 500000 (10x smaller!)

    # Evaluation
    eval_every: int = 100       # More frequent evaluation
    eval_steps: int = 20        # Smaller eval steps

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = 50304  # GPT-2 vocab size + SNAC tokens (50257 + 47 for safety)
    
    # Multimodal parameters
    audio_sample_rate: int = 24000
    max_audio_duration: float = 10.0
    audio_vocab_size: int = 1024
    text_vocab_size: int = 50257
    
    # Special tokens (aligned with tts-snac-base.py)
    eos_token_id: int = 50256  # EOS token (same as tts-snac-base.py)
    pad_token_id: int = 50257  # PAD token (same as tts-snac-base.py)
    audio_start_token: int = 50258  # Audio start token (same as tts-snac-base.py)
    audio_end_token: int = 50256    # Audio end token (same as EOS in tts-snac-base.py)
    
    # Audio processing
    snac_model_name: str = "hubertsiuzdak/snac_24khz"
    audio_weight: float = 1.0
    text_weight: float = 0.1

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads
