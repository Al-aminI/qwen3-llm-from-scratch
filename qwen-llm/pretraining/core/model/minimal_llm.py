"""
This file contains the complete language model that combines all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from dataclasses import dataclass

from .components import TransformerBlock, RMSNorm

class MinimalLLM(nn.Module):
    """
    This is the full language model that combines all components:
    
    Architecture:
    1. Token Embedding: Convert token IDs to vectors
    2. Positional Dropout: Prevent overfitting on positions
    3. Transformer Blocks: Stack of attention + feed-forward layers
    4. Final Normalization: RMSNorm before output
    5. Language Modeling Head: Convert vectors back to token probabilities
    6. Weight Tying: Share weights between input and output embeddings
    
    Key Features:
    - Pre-norm architecture (more stable training)
    - Weight tying (reduces parameters, improves generalization)
    - Proper initialization (Xavier/He initialization)
    - Efficient memory usage (GQA, RMSNorm)
    
    Parameter Efficiency:
    - Weight tying: Input and output embeddings share weights
    - GQA: Reduces attention memory by 50-75%
    - RMSNorm: More efficient than LayerNorm
    - SwiGLU: More expressive than ReLU with similar cost
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding: converts token IDs to vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional dropout: prevents overfitting on position information
        self.position_dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization before output
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head: converts vectors to token probabilities
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # WEIGHT TYING: Share weights between input and output embeddings
        # This reduces parameters and improves generalization
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Good initialization is crucial for training stability:
        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - Biases: Zero initialization
        
        This follows the initialization scheme used in modern LLMs.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass through the complete model
        
        Args:
            x: Input token IDs (batch, seq_len)
            
        Returns:
            Logits for next token prediction (batch, seq_len, vocab_size)
        """
        # 1. Convert token IDs to embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        
        # 2. Apply positional dropout
        x = self.position_dropout(x)

        # 3. Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # 4. Final normalization
        x = self.norm(x)
        x = self.output_dropout(x)
        
        # 5. Convert to token probabilities
        logits = self.lm_head(x)
        return logits
