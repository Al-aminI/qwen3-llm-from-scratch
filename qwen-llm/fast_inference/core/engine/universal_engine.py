"""
🌐 Universal Fast Inference Engine

This module provides a universal inference engine that can work with ANY model,
not just the custom MinimalLLM. It automatically detects model architecture
and applies appropriate optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

if TYPE_CHECKING:
    from ...utils.sampling import SamplingParams
    from ..cache.simple_cache import SimpleKVCache
    from ..attention.cached_attention import CachedAttention


class UniversalFastInference:
    """
    🌐 UNIVERSAL FAST INFERENCE ENGINE
    
    A universal inference engine that can work with ANY model:
    - HuggingFace models (BERT, RoBERTa, GPT, LLaMA, etc.)
    - Custom models (like your MinimalLLM)
    - Any PyTorch model with transformer architecture
    
    Key Features:
    - Automatic model detection
    - Universal KV caching
    - Model-agnostic optimization
    - Easy integration
    """
    
    def __init__(self, model: Union[nn.Module, str], tokenizer: Union[Any, str], 
                 max_seq_len: int = 2048, model_type: str = "auto"):
        """
        Initialize universal fast inference engine.
        
        Args:
            model: Model instance, model path, or HuggingFace model name
            tokenizer: Tokenizer instance, tokenizer path, or HuggingFace tokenizer name
            max_seq_len: Maximum sequence length
            model_type: Model type ("auto", "huggingface", "custom", "minimal_llm")
        """
        self.max_seq_len = max_seq_len
        self.model_type = model_type
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.config = self._load_model_and_tokenizer(model, tokenizer)
        self.device = next(self.model.parameters()).device
        
        # Detect model architecture
        self.architecture_info = self._detect_architecture()
        
        # Initialize KV cache based on detected architecture
        self.kv_cache = self._create_kv_cache()
        
        # Replace attention layers with cached versions
        self._replace_attention_layers()
        
        # Sequence tracking
        self.next_seq_id = 0
        
    def _load_model_and_tokenizer(self, model, tokenizer):
        """Load model and tokenizer from various sources."""
        # Load tokenizer
        if isinstance(tokenizer, str):
            print(f"📦 Loading tokenizer: {tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        if isinstance(model, str):
            print(f"📦 Loading model: {model}")
            if self.model_type == "huggingface" or "huggingface" in model.lower():
                # HuggingFace model
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                config = model.config
            else:
                # Custom model checkpoint
                checkpoint = torch.load(model, map_location='cpu')
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    # Import your custom model
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
                    from pretraining import MinimalLLM
                    model = MinimalLLM(config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    raise ValueError("Invalid model checkpoint format")
        else:
            # Model instance provided
            config = getattr(model, 'config', None)
            if config is None:
                raise ValueError("Model must have a config attribute")
        
        model.eval()
        return model, tokenizer, config
    
    def _detect_architecture(self) -> Dict[str, Any]:
        """Detect model architecture automatically."""
        info = {
            'model_type': 'unknown',
            'has_attention': False,
            'has_transformer_blocks': False,
            'attention_heads': 0,
            'hidden_size': 0,
            'num_layers': 0,
            'vocab_size': 0
        }
        
        # Try to detect from config
        if hasattr(self.config, 'n_heads'):
            # Your MinimalLLM
            info.update({
                'model_type': 'minimal_llm',
                'has_attention': True,
                'has_transformer_blocks': True,
                'attention_heads': self.config.n_heads,
                'hidden_size': self.config.d_model,
                'num_layers': self.config.n_layers,
                'vocab_size': self.config.vocab_size
            })
        elif hasattr(self.config, 'num_attention_heads'):
            # HuggingFace models
            info.update({
                'model_type': 'huggingface',
                'has_attention': True,
                'has_transformer_blocks': hasattr(self.model, 'transformer') or hasattr(self.model, 'model'),
                'attention_heads': self.config.num_attention_heads,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_hidden_layers,
                'vocab_size': self.config.vocab_size
            })
        else:
            # Try to detect from model structure
            if hasattr(self.model, 'transformer_blocks'):
                info['has_transformer_blocks'] = True
                info['num_layers'] = len(self.model.transformer_blocks)
                if hasattr(self.model.transformer_blocks[0], 'attention'):
                    info['has_attention'] = True
                    # Try to get attention heads
                    attn = self.model.transformer_blocks[0].attention
                    if hasattr(attn, 'n_heads'):
                        info['attention_heads'] = attn.n_heads
                    elif hasattr(attn, 'num_heads'):
                        info['attention_heads'] = attn.num_heads
        
        print(f"🔍 Detected architecture: {info['model_type']}")
        print(f"   Attention heads: {info['attention_heads']}")
        print(f"   Hidden size: {info['hidden_size']}")
        print(f"   Layers: {info['num_layers']}")
        
        return info
    
    def _create_kv_cache(self):
        """Create appropriate KV cache based on detected architecture."""
        from ..cache.simple_cache import SimpleKVCache
        
        # Determine cache parameters
        if self.architecture_info['model_type'] == 'minimal_llm':
            n_heads = getattr(self.config, 'n_kv_heads', self.config.n_heads)
            head_dim = self.config.d_k
        elif self.architecture_info['model_type'] == 'huggingface':
            n_heads = getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads)
            head_dim = self.config.hidden_size // self.config.num_attention_heads
        else:
            # Default values
            n_heads = self.architecture_info['attention_heads']
            head_dim = self.architecture_info['hidden_size'] // n_heads if n_heads > 0 else 64
        
        return SimpleKVCache(
            max_seq_len=self.max_seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=self.device
        )
    
    def _replace_attention_layers(self):
        """Replace attention layers with cached versions."""
        if self.architecture_info['model_type'] == 'minimal_llm':
            self._replace_minimal_llm_attention()
        elif self.architecture_info['model_type'] == 'huggingface':
            self._replace_huggingface_attention()
        else:
            print("⚠️ Unknown model type, skipping attention replacement")
    
    def _replace_minimal_llm_attention(self):
        """Replace attention layers in MinimalLLM."""
        from ..attention.cached_attention import CachedAttention
        from pretraining.core.model.components import SwiGLUFeedForward, RMSNorm
        
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
    
    def _replace_huggingface_attention(self):
        """Replace attention layers in HuggingFace models."""
        # This is more complex and model-specific
        # For now, we'll use a generic approach
        print("⚠️ HuggingFace attention replacement not fully implemented yet")
        print("   Using standard attention (no caching)")
    
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
        
        # Generate based on model type
        if self.architecture_info['model_type'] == 'minimal_llm':
            return self._generate_minimal_llm(seq_id, input_ids, max_new_tokens, temperature, top_k, top_p)
        elif self.architecture_info['model_type'] == 'huggingface':
            return self._generate_huggingface(seq_id, input_ids, max_new_tokens, temperature, top_k, top_p)
        else:
            return self._generate_generic(seq_id, input_ids, max_new_tokens, temperature, top_k, top_p)
    
    def _generate_minimal_llm(self, seq_id: int, input_ids: torch.Tensor, 
                             max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
        """Generate text using MinimalLLM architecture."""
        with torch.no_grad():
            # Prefill phase
            x = self.model.token_embedding(input_ids) * math.sqrt(self.config.d_model)
            x = self.model.position_dropout(x)
            
            # Process through transformer blocks with caching
            for layer_idx, block in enumerate(self.model.transformer_blocks):
                x = block(x, seq_id, layer_idx, use_cache=True)
            
            # Get logits for last position
            x = self.model.norm(x)
            logits = self.model.lm_head(x[:, -1:, :])
        
        # Generate tokens
        generated_tokens = []
        for _ in range(max_new_tokens):
            next_token = self._sample_token(logits, temperature, top_k, top_p)
            generated_tokens.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Forward pass for single token
            next_input = torch.tensor([[next_token]], device=self.device)
            with torch.no_grad():
                x = self.model.token_embedding(next_input) * math.sqrt(self.config.d_model)
                x = self.model.position_dropout(x)
                
                for layer_idx, block in enumerate(self.model.transformer_blocks):
                    x = block(x, seq_id, layer_idx, use_cache=True)
                
                x = self.model.norm(x)
                logits = self.model.lm_head(x)
        
        # Clean up cache
        self.kv_cache.clear_sequence(seq_id)
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    def _generate_huggingface(self, seq_id: int, input_ids: torch.Tensor, 
                             max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
        """Generate text using HuggingFace model."""
        # Use standard HuggingFace generation (no caching for now)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
        return generated_text
    
    def _generate_generic(self, seq_id: int, input_ids: torch.Tensor, 
                         max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
        """Generate text using generic model."""
        # Fallback to standard generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
        return generated_text
    
    def _sample_token(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
        """Sample a token from logits."""
        logits = logits.squeeze(0)
        
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'architecture': self.architecture_info,
            'device': str(self.device),
            'max_seq_len': self.max_seq_len,
            'model_type': self.architecture_info['model_type'],
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'cache_info': self.kv_cache.get_memory_usage() if hasattr(self.kv_cache, 'get_memory_usage') else {}
        }


class CachedTransformerBlock(nn.Module):
    """
    🏗️ CACHED TRANSFORMER BLOCK
    
    Universal transformer block with cached attention.
    """
    
    def __init__(self, config, kv_cache: 'SimpleKVCache'):
        super().__init__()
        from ..attention.cached_attention import CachedAttention
        from pretraining.core.model.components import SwiGLUFeedForward, RMSNorm
        
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


def create_universal_fast_inference(model: Union[nn.Module, str], tokenizer: Union[Any, str], 
                                   max_seq_len: int = 2048, model_type: str = "auto") -> UniversalFastInference:
    """
    Create a universal fast inference engine.
    
    Args:
        model: Model instance, model path, or HuggingFace model name
        tokenizer: Tokenizer instance, tokenizer path, or HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        model_type: Model type ("auto", "huggingface", "custom", "minimal_llm")
        
    Returns:
        UniversalFastInference instance
    """
    return UniversalFastInference(model, tokenizer, max_seq_len, model_type)
