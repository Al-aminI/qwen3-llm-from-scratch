"""
Simple fast inference engine with basic KV caching.

This module provides a straightforward but fast inference engine that adds
KV caching to existing models for significant speedup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.sampling import SamplingParams
    from ..cache.simple_cache import SimpleKVCache
    from ..attention.cached_attention import CachedAttention


class SimpleFastInference:
    """
    🚀 SIMPLE FAST INFERENCE ENGINE
    
    A simplified but fast inference engine that adds KV caching to your existing model.
    Much easier to understand and integrate than full vLLM-style implementations.
    """
    
    def __init__(self, model: nn.Module, tokenizer, config, max_seq_len: int = 2048):
        """
        Initialize simple fast inference engine.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            config: Model configuration
            max_seq_len: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
        
        # Initialize KV cache
        self.kv_cache = SimpleKVCache(
            max_seq_len=max_seq_len,
            n_heads=config.n_kv_heads,
            head_dim=config.d_k,
            dtype=model.dtype if hasattr(model, 'dtype') else torch.float16,
            device=self.device
        )
        
        # Replace transformer blocks with cached versions
        self._replace_attention_layers()
        
        # Sequence tracking
        self.next_seq_id = 0
        
    def _replace_attention_layers(self):
        """Replace standard attention with cached attention."""
        for i, block in enumerate(self.model.transformer_blocks):
            cached_block = CachedTransformerBlock(self.config, self.kv_cache)
            
            # Copy weights from original block
            cached_block.attention.q_proj.load_state_dict(block.attention.q_proj.state_dict())
            cached_block.attention.k_proj.load_state_dict(block.attention.k_proj.state_dict())
            cached_block.attention.v_proj.load_state_dict(block.attention.v_proj.state_dict())
            cached_block.attention.w_o.load_state_dict(block.attention.w_o.state_dict())
            cached_block.attention.q_norm.load_state_dict(block.attention.q_norm.state_dict())
            cached_block.attention.k_norm.load_state_dict(block.attention.k_norm.state_dict())
            
            cached_block.feed_forward.load_state_dict(block.feed_forward.state_dict())
            cached_block.norm1.load_state_dict(block.norm1.state_dict())
            cached_block.norm2.load_state_dict(block.norm2.state_dict())
            
            self.model.transformer_blocks[i] = cached_block
    
    def generate_single(self, prompt: str, max_new_tokens: int = 100, 
                       temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        """
        Generate text for a single prompt with KV caching.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated text
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        input_length = input_ids.size(1)
        
        # Clear cache for this sequence
        self.kv_cache.clear_sequence(seq_id)
        
        # Prefill phase: process the entire prompt
        with torch.no_grad():
            x = self.model.token_embedding(input_ids) * math.sqrt(self.config.d_model)
            x = self.model.position_dropout(x)
            
            # Process through transformer blocks with caching
            for layer_idx, block in enumerate(self.model.transformer_blocks):
                x = block(x, seq_id, layer_idx, use_cache=True)
            
            # Get logits for last position
            x = self.model.norm(x)
            logits = self.model.lm_head(x[:, -1:, :])  # Only last position
        
        # Generate tokens one by one
        generated_tokens = []
        current_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Sample next token
            next_token = self._sample_token(logits, temperature, top_k, top_p)
            generated_tokens.append(next_token)
            
            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Prepare next input (only the new token)
            next_input = torch.tensor([[next_token]], device=self.device)
            
            # Forward pass for single token
            with torch.no_grad():
                x = self.model.token_embedding(next_input) * math.sqrt(self.config.d_model)
                x = self.model.position_dropout(x)
                
                # Process through transformer blocks with caching
                for layer_idx, block in enumerate(self.model.transformer_blocks):
                    x = block(x, seq_id, layer_idx, use_cache=True)
                
                # Get logits for next token
                x = self.model.norm(x)
                logits = self.model.lm_head(x)
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up cache
        self.kv_cache.clear_sequence(seq_id)
        
        return generated_text
    
    def _sample_token(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
        """Sample a token from logits."""
        logits = logits.squeeze(0)  # Remove batch dimension
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[top_k_indices] = top_k_logits
            logits = filtered_logits
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample token
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        
        return token
    
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 100, 
                      temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> List[str]:
        """
        Generate text for multiple prompts (sequential processing).
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate_single(prompt, max_new_tokens, temperature, top_k, top_p)
            results.append(result)
        return results


class CachedTransformerBlock(nn.Module):
    """
    🏗️ CACHED TRANSFORMER BLOCK
    
    Transformer block with cached attention.
    """
    
    def __init__(self, config, kv_cache: 'SimpleKVCache'):
        super().__init__()
        self.attention = CachedAttention(config, kv_cache)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        
        # Pre-norm architecture
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, seq_id: int, layer_idx: int, use_cache: bool = True):
        """Forward pass with KV caching."""
        # Pre-norm attention with residual connection
        attn_out = self.attention(self.norm1(x), seq_id, layer_idx, use_cache)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


def create_simple_fast_inference(model_path: str, tokenizer_path: str, 
                                max_seq_len: int = 2048) -> SimpleFastInference:
    """
    Create a simple fast inference engine from saved model.
    
    Args:
        model_path: Path to saved model
        tokenizer_path: Path to tokenizer
        max_seq_len: Maximum sequence length
        
    Returns:
        SimpleFastInference instance
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = SmallModelConfig()
    config.vocab_size = checkpoint.get('vocab_size', 32000)  # Default vocab size
    
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create engine
    engine = SimpleFastInference(
        model=model,
        tokenizer=tokenizer,
        config=config,
        max_seq_len=max_seq_len
    )
    
    return engine
