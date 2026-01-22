"""
🏗️ MULTIMODAL QWEN3 MODEL

This file contains the complete multimodal language model that can process
both text and audio tokens using SNAC audio tokenization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .components import TransformerBlock, RMSNorm

class MultimodalLLM(nn.Module):
    """
    🏗️ MULTIMODAL QWEN3-STYLE LANGUAGE MODEL
    
    This is the full multimodal language model that can process both text and audio:
    
    🧠 Architecture:
    1. Token Embedding: Convert token IDs to vectors (text + audio)
    2. Positional Dropout: Prevent overfitting on positions
    3. Transformer Blocks: Stack of attention + feed-forward layers
    4. Final Normalization: RMSNorm before output
    5. Language Modeling Head: Convert vectors back to token probabilities
    6. Weight Tying: Share weights between input and output embeddings
    
    🎯 Key Features:
    - Pre-norm architecture (more stable training)
    - Weight tying (reduces parameters, improves generalization)
    - Proper initialization (Xavier/He initialization)
    - Efficient memory usage (GQA, RMSNorm)
    - Multimodal processing (text + audio tokens)
    
    📊 Parameter Efficiency:
    - Weight tying: Input and output embeddings share weights
    - GQA: Reduces attention memory by 50-75%
    - RMSNorm: More efficient than LayerNorm
    - SwiGLU: More expressive than ReLU with similar cost
    - Shared vocabulary for text and audio tokens
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding: converts token IDs to vectors (text + audio)
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
        
        # 🔑 WEIGHT TYING: Share weights between input and output embeddings
        # This reduces parameters and improves generalization
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        🎯 PROPER WEIGHT INITIALIZATION
        
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

    def forward(self, x, attention_mask=None):
        """
        Forward pass through the complete multimodal model
        
        Args:
            x: Input token IDs (batch, seq_len) - can contain both text and audio tokens
            attention_mask: Optional attention mask (batch, seq_len)
            
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
    
    def generate_audio(self, text_tokens, audio_tokenizer, max_length=1024, temperature=0.8, top_p=0.95):
        """
        Generate audio tokens from text tokens
        
        Args:
            text_tokens: Input text token IDs
            audio_tokenizer: SNAC audio tokenizer
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated audio tokens
        """
        self.eval()
        with torch.no_grad():
            # Start with text tokens
            input_ids = text_tokens.clone()
            
            # Add audio start token
            audio_start_token = audio_tokenizer.get_special_tokens()['audio_start']
            input_ids = torch.cat([input_ids, torch.tensor([[audio_start_token]])], dim=1)
            
            # Generate audio tokens
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token (EOS token 50256 like tts-snac-base.py)
                if next_token.item() == 50256:
                    break
            
            return input_ids
    
    def generate_text(self, input_ids, max_length=1024, temperature=0.8, top_p=0.95):
        """
        Generate text tokens from input
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text tokens
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token
                if next_token.item() == self.config.eos_token_id:
                    break
            
            return input_ids
