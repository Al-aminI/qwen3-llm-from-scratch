# -*- coding: utf-8 -*-
"""
Modified Qwen3 from Scratch - Small Configuration for Fast Training
This version uses smaller parameters and dataset for faster training and learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import math
import random
import numpy as np

from datasets import load_dataset
from tqdm import tqdm

import time
from transformers import AutoTokenizer

from dataclasses import dataclass
from typing import List, Optional

import warnings
import os
import pickle

warnings.filterwarnings('ignore')

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
class SmallModelConfig:
    """
    🎯 SMALL CONFIGURATION FOR FAST TRAINING
    
    This configuration is optimized for:
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
    max_steps: int = 500        # Reduced from 2000 (4x smaller)
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
    num_documents: int = 200    # Reduced from 2000 (10x smaller!)
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
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads

# =============================================================================
# 🧠 COMPONENT 1: GROUPED-QUERY ATTENTION (GQA) HELPER
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    🔑 KEY COMPONENT: Grouped-Query Attention (GQA)
    
    GQA is a memory-efficient attention mechanism where:
    - We have fewer Key-Value heads than Query heads
    - Each KV head is "repeated" to match the number of Query heads
    - This reduces memory usage while maintaining performance
    
    Example:
    - 4 Query heads, 2 KV heads → each KV head repeated 2 times
    - Memory savings: 50% reduction in KV cache memory
    - Performance: Nearly identical to full attention
    
    This function implements the "repetition" part of GQA.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    # Step 1: Add dimension for repetition
    # Shape: (batch, num_kv_heads, slen, head_dim) 
    #     → (batch, num_kv_heads, 1, slen, head_dim)
    hidden_states = hidden_states[:, :, None, :, :]
    
    # Step 2: Expand to repeat each KV head n_rep times
    # Shape: (batch, num_kv_heads, 1, slen, head_dim)
    #     → (batch, num_kv_heads, n_rep, slen, head_dim)
    hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # Step 3: Reshape to merge the repetition dimension
    # Shape: (batch, num_kv_heads, n_rep, slen, head_dim)
    #     → (batch, num_kv_heads * n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# =============================================================================
# 🚀 COMPONENT 2: MUON OPTIMIZER - THE SECRET SAUCE
# =============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    🔬 NEWTON-SCHULZ ORTHOGONALIZATION
    
    This is the mathematical heart of the Muon optimizer:
    
    🎯 What it does:
    - Takes a matrix G (gradients)
    - Makes it "orthogonal" (like rotating it to be perfectly aligned)
    - Uses Newton-Schulz iteration (a fast numerical method)
    
    🧮 Why orthogonalization helps:
    - Orthogonal matrices preserve vector lengths and angles
    - This prevents gradients from exploding or vanishing
    - Leads to more stable and faster training
    
    🔢 The math:
    - Newton-Schulz finds the "square root" of the identity matrix
    - It's like finding the "best rotation" for our gradients
    - Uses coefficients (a, b, c) that are mathematically optimized
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients
    X = G.bfloat16()  # Use bfloat16 for efficiency

    # Handle rectangular matrices
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize to prevent numerical issues
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz iteration (the magic happens here!)
    for _ in range(steps):
        A = X @ X.mT  # Compute X * X^T
        B = b * A + c * A @ A  # Polynomial combination
        X = a * X + B @ X  # Update X

    # Restore original orientation if needed
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """
    🚀 MUON OPTIMIZER: MomentUm Orthogonalized by Newton-schulz
    
    This is a revolutionary optimizer that combines:
    1. Momentum (like Adam) - remembers past gradients
    2. Orthogonalization (Newton-Schulz) - makes gradients "well-behaved"
    3. Adaptive learning rates - adjusts based on matrix dimensions
    
    🎯 Why Muon is special:
    - 30-50% faster convergence than Adam
    - More stable training (fewer gradient explosions)
    - Better generalization (works well on new data)
    - Particularly good for transformer models
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer (like Adam's first moment)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                
                # Update momentum: buf = momentum * buf + (1-momentum) * grad
                buf.lerp_(g, 1 - group["momentum"])
                
                # Apply Nesterov momentum (look ahead)
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # 🔥 THE MAGIC: Apply Newton-Schulz orthogonalization
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                
                # Update parameters with adaptive learning rate
                # Larger matrices get higher learning rates (scales with √(height/width))
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

# =============================================================================
# 📊 COMPONENT 3: DATA LOADING AND CACHING
# =============================================================================

def load_and_cache_data(config: SmallModelConfig, cache_dir: str = "data_cache"):
    """
    📦 SMART DATA LOADING WITH CACHING
    
    This function demonstrates modern ML data handling:
    
    🎯 Key features:
    1. Caching: Avoids reprocessing the same data
    2. Streaming: Loads large datasets without memory issues
    3. Tokenization: Converts text to numbers the model can understand
    4. Efficient storage: Uses pickle for fast loading
    
    🔄 The process:
    1. Check if we already processed this data (cache hit)
    2. If not, load from HuggingFace datasets
    3. Tokenize the text (convert words → numbers)
    4. Cache the result for next time
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check cache first (smart optimization!)
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer (the "dictionary" that converts text ↔ numbers)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (streaming = memory efficient)
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

    # Load only a small subset for fast training
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])  # Limit text length

    print(f"Loaded {len(texts)} documents")

    # Tokenize (convert text to numbers)
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache for next time
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    """
    📚 CUSTOM DATASET FOR LANGUAGE MODELING
    
    This creates training examples for our language model:
    
    🎯 What it does:
    - Takes a long sequence of tokens
    - Creates sliding windows of fixed length
    - Each example: input sequence + target sequence (shifted by 1)
    
    📖 Example:
    Original text: "The cat sat on the mat"
    Tokens: [1, 2, 3, 4, 5, 6]
    Window size: 4
    
    Example 1: input=[1,2,3,4], target=[2,3,4,5]
    Example 2: input=[2,3,4,5], target=[3,4,5,6]
    
    This teaches the model to predict the next token!
    """
    def __init__(self, tokens: List[int], seq_len: int = 256):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    print("🎯 Qwen3 Small Configuration Ready!")
    print("This configuration is optimized for fast training and learning.")
    print("Key changes:")
    print("- Model: 128d, 3L, 4H (vs 384d, 6L, 8H)")
    print("- Dataset: 200 docs, 50K tokens (vs 2000 docs, 500K tokens)")
    print("- Training: 500 steps (vs 2000 steps)")
    print("- All components preserved for learning!")
