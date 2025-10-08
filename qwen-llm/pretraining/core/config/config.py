"""
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
    print(f" Set all seeds to {seed}")

@dataclass
class PretrainingConfig:
    """
    This configuration is optimized for:
    - Fast training on CPU/limited GPU
    - Learning the concepts without waiting hours
    - Still demonstrating all key components
    """
    # Model architecture
    d_model: int = 128         
    n_heads: int = 4           
    n_layers: int = 3          
    d_ff: int = 512             
    
    # Training parameters
    batch_size: int = 8         
    max_steps: int = 1000       
    gradient_accumulation_steps: int = 2  

  
    n_kv_heads: int = 2         # For Grouped-Query Attention (GQA)
    sliding_window: int = 1024   # Smaller context window
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    # Training parameters
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 256     
    num_documents: int = 200    
    max_tokens: int = 50000    

    # Evaluation
    eval_every: int = 100       
    eval_steps: int = 20        

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads
