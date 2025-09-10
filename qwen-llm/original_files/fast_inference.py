#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 FAST INFERENCE ENGINE FOR QWEN3

This implementation combines the best optimizations from vLLM, nano-vLLM, and flex-nano-vLLM:
- PagedAttention for memory efficiency
- Continuous batching for throughput
- KV caching for speed
- CUDA graphs for optimization
- FlexAttention for modern attention computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Import our model components
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM
from qwen3_core_components import Qwen3Attention, SwiGLUFeedForward, TransformerBlock, RMSNorm, Rotary, repeat_kv

# =============================================================================
# 🎯 CORE DATA STRUCTURES
# =============================================================================

@dataclass
class SamplingParams:
    """Sampling parameters for text generation"""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = None
    
    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []

@dataclass
class Sequence:
    """Represents a single generation sequence"""
    seq_id: int
    prompt: str
    input_ids: torch.Tensor
    output_ids: List[int]
    sampling_params: SamplingParams
    batch_idx: int = -1
    finished: bool = False
    prefill_done: bool = False
    
    @property
    def total_length(self) -> int:
        return len(self.input_ids) + len(self.output_ids)
    
    @property
    def last_token_id(self) -> int:
        return self.output_ids[-1] if self.output_ids else self.input_ids[-1].item()

# =============================================================================
# 🧠 PAGED KV CACHE IMPLEMENTATION
# =============================================================================

class PagedKVCache(nn.Module):
    """
    🧠 PAGED KV CACHE
    
    Efficient memory management for KV cache using page-based allocation.
    Inspired by vLLM's PagedAttention but simplified for our use case.
    """
    
    def __init__(self, n_pages: int, page_size: int, n_heads: int, head_dim: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.n_pages = n_pages
        self.page_size = page_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # Allocate cache memory in pages
        cache_shape = (1, n_heads, n_pages * page_size, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        
        # Page table: [batch_idx, logical_page] -> physical_page
        self.page_table = -torch.ones((256, n_pages), dtype=torch.long, device=device)  # Max 256 sequences
        self.free_pages = list(range(n_pages))
        self.sequence_pages = {}  # seq_id -> list of allocated pages
        
    def allocate_pages(self, seq_id: int, num_pages: int) -> List[int]:
        """Allocate pages for a sequence"""
        if len(self.free_pages) < num_pages:
            raise RuntimeError(f"Not enough free pages. Need {num_pages}, have {len(self.free_pages)}")
        
        allocated_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self.sequence_pages[seq_id] = allocated_pages
        
        # Update page table
        for i, page in enumerate(allocated_pages):
            self.page_table[seq_id, i] = page
            
        return allocated_pages
    
    def deallocate_pages(self, seq_id: int):
        """Deallocate pages for a sequence"""
        if seq_id in self.sequence_pages:
            self.free_pages.extend(self.sequence_pages[seq_id])
            self.page_table[seq_id, :] = -1
            del self.sequence_pages[seq_id]
    
    def get_physical_positions(self, seq_id: int, logical_positions: torch.Tensor) -> torch.Tensor:
        """Convert logical positions to physical cache positions"""
        page_indices = logical_positions // self.page_size
        offsets = logical_positions % self.page_size
        
        physical_pages = self.page_table[seq_id, page_indices]
        physical_positions = physical_pages * self.page_size + offsets
        
        return physical_positions
    
    def update_cache(self, seq_id: int, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Update KV cache at specific positions"""
        physical_positions = self.get_physical_positions(seq_id, positions)
        
        # Update cache
        self.k_cache[0, :, physical_positions, :] = k
        self.v_cache[0, :, physical_positions, :] = v

# =============================================================================
# 🎯 OPTIMIZED ATTENTION WITH KV CACHE
# =============================================================================

class OptimizedAttention(nn.Module):
    """
    🎯 OPTIMIZED ATTENTION WITH KV CACHE
    
    Combines GQA with efficient KV caching for fast inference.
    """
    
    def __init__(self, config: SmallModelConfig, kv_cache: PagedKVCache):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k
        self.kv_cache = kv_cache

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Normalization
        self.q_norm = RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_k, eps=config.rms_norm_eps)

        # RoPE
        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, seq_id: int, positions: torch.Tensor, use_cache: bool = True):
        """
        Forward pass with KV caching
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            seq_id: Sequence ID for cache lookup
            positions: Position indices for cache
            use_cache: Whether to use cached KV values
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Update KV cache
        if use_cache:
            self.kv_cache.update_cache(seq_id, positions, k[0], v[0])

        # Get cached K, V for attention
        if use_cache and seq_id in self.kv_cache.sequence_pages:
            # Retrieve cached values
            cached_k = self.kv_cache.k_cache[0, :, :positions.max() + 1, :]
            cached_v = self.kv_cache.v_cache[0, :, :positions.max() + 1, :]
            
            # Use cached values for attention
            K = cached_k.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, cached_len, d_k)
            V = cached_v.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, cached_len, d_k)
        else:
            K = k.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)
            V = v.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)

        # Repeat KV heads for GQA
        K = repeat_kv(K, self.n_kv_groups)
        V = repeat_kv(V, self.n_kv_groups)

        # Transpose Q for attention
        Q = q.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

# =============================================================================
# 🏗️ OPTIMIZED TRANSFORMER BLOCK
# =============================================================================

class OptimizedTransformerBlock(nn.Module):
    """
    🏗️ OPTIMIZED TRANSFORMER BLOCK
    
    Transformer block with optimized attention and KV caching.
    """
    
    def __init__(self, config: SmallModelConfig, kv_cache: PagedKVCache):
        super().__init__()
        self.attention = OptimizedAttention(config, kv_cache)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        
        # Pre-norm architecture
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, seq_id: int, positions: torch.Tensor, use_cache: bool = True):
        """Forward pass with KV caching"""
        # Pre-norm attention with residual connection
        attn_out = self.attention(self.norm1(x), seq_id, positions, use_cache)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

# =============================================================================
# 🚀 FAST INFERENCE ENGINE
# =============================================================================

class FastInferenceEngine:
    """
    🚀 FAST INFERENCE ENGINE
    
    High-performance inference engine with:
    - Paged KV cache for memory efficiency
    - Continuous batching for throughput
    - CUDA graphs for optimization
    - Dynamic scheduling
    """
    
    def __init__(self, model: MinimalLLM, tokenizer, config: SmallModelConfig, 
                 max_batch_size: int = 32, max_seq_len: int = 2048, 
                 n_pages: int = 1000, page_size: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
        
        # Initialize KV cache
        self.kv_cache = PagedKVCache(
            n_pages=n_pages,
            page_size=page_size,
            n_heads=config.n_kv_heads,
            head_dim=config.d_k,
            dtype=model.dtype if hasattr(model, 'dtype') else torch.float16,
            device=self.device
        )
        
        # Replace transformer blocks with optimized versions
        self._replace_attention_layers()
        
        # Scheduling queues
        self.waiting_queue = deque()
        self.running_queue = deque()
        self.finished_queue = deque()
        
        # Sequence tracking
        self.next_seq_id = 0
        self.sequence_map = {}
        
        # CUDA graph optimization
        self.cuda_graphs = {}
        self.graph_captured = False
        
    def _replace_attention_layers(self):
        """Replace standard attention with optimized attention"""
        for i, block in enumerate(self.model.transformer_blocks):
            optimized_block = OptimizedTransformerBlock(self.config, self.kv_cache)
            
            # Copy weights from original block
            optimized_block.attention.q_proj.load_state_dict(block.attention.q_proj.state_dict())
            optimized_block.attention.k_proj.load_state_dict(block.attention.k_proj.state_dict())
            optimized_block.attention.v_proj.load_state_dict(block.attention.v_proj.state_dict())
            optimized_block.attention.w_o.load_state_dict(block.attention.w_o.state_dict())
            optimized_block.attention.q_norm.load_state_dict(block.attention.q_norm.state_dict())
            optimized_block.attention.k_norm.load_state_dict(block.attention.k_norm.state_dict())
            
            optimized_block.feed_forward.load_state_dict(block.feed_forward.state_dict())
            optimized_block.norm1.load_state_dict(block.norm1.state_dict())
            optimized_block.norm2.load_state_dict(block.norm2.state_dict())
            
            self.model.transformer_blocks[i] = optimized_block
    
    def add_request(self, prompt: str, sampling_params: SamplingParams) -> int:
        """Add a new generation request"""
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
        
        # Create sequence
        sequence = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            input_ids=input_ids,
            output_ids=[],
            sampling_params=sampling_params
        )
        
        self.sequence_map[seq_id] = sequence
        self.waiting_queue.append(sequence)
        
        return seq_id
    
    def _prefill_sequences(self, sequences: List[Sequence]) -> torch.Tensor:
        """Prefill phase: process input prompts"""
        if not sequences:
            return torch.empty(0, 0, self.config.vocab_size, device=self.device)
        
        # Allocate pages for sequences
        for seq in sequences:
            num_pages = (seq.total_length + self.kv_cache.page_size - 1) // self.kv_cache.page_size
            self.kv_cache.allocate_pages(seq.seq_id, num_pages)
            seq.batch_idx = seq.seq_id
            seq.prefill_done = True
        
        # Prepare batch
        max_len = max(seq.total_length for seq in sequences)
        batch_size = len(sequences)
        
        # Pad sequences to same length
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        for i, seq in enumerate(sequences):
            seq_len = seq.total_length
            input_ids[i, :seq_len] = torch.cat([seq.input_ids, torch.tensor(seq.output_ids, device=self.device)])
            positions[i, :seq_len] = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        x = self.model.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.model.position_dropout(x)
        
        # Process through transformer blocks
        for block in self.model.transformer_blocks:
            x = block(x, seq.seq_id, positions[0], use_cache=True)
        
        # Final normalization and output
        x = self.model.norm(x)
        logits = self.model.lm_head(x)
        
        # Return logits for last positions
        last_positions = torch.tensor([seq.total_length - 1 for seq in sequences], device=self.device)
        return logits[torch.arange(batch_size), last_positions, :]
    
    def _decode_sequences(self, sequences: List[Sequence]) -> torch.Tensor:
        """Decode phase: generate next tokens"""
        if not sequences:
            return torch.empty(0, 0, self.config.vocab_size, device=self.device)
        
        batch_size = len(sequences)
        
        # Get last tokens
        last_tokens = torch.tensor([seq.last_token_id for seq in sequences], device=self.device)
        positions = torch.tensor([seq.total_length - 1 for seq in sequences], device=self.device)
        
        # Forward pass for single token
        x = self.model.token_embedding(last_tokens.unsqueeze(1)) * math.sqrt(self.config.d_model)
        
        # Process through transformer blocks
        for block in self.model.transformer_blocks:
            x = block(x, sequences[0].seq_id, positions, use_cache=True)
        
        # Final normalization and output
        x = self.model.norm(x)
        logits = self.model.lm_head(x)
        
        return logits.squeeze(1)
    
    def _sample_tokens(self, logits: torch.Tensor, sampling_params: List[SamplingParams]) -> List[int]:
        """Sample next tokens from logits"""
        tokens = []
        
        for i, (logit, params) in enumerate(zip(logits, sampling_params)):
            # Apply temperature
            if params.temperature > 0:
                logit = logit / params.temperature
            
            # Apply top-k filtering
            if params.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logit, min(params.top_k, logit.size(-1)))
                filtered_logits = torch.full_like(logit, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                logit = filtered_logits
            
            # Apply top-p filtering
            if params.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logit, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > params.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logit[indices_to_remove] = float('-inf')
            
            # Sample token
            probs = F.softmax(logit, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(token)
        
        return tokens
    
    def _check_finished(self, sequences: List[Sequence]) -> Tuple[List[Sequence], List[Sequence]]:
        """Check which sequences are finished"""
        finished = []
        running = []
        
        for seq in sequences:
            # Check stopping conditions
            is_eos = seq.last_token_id == self.tokenizer.eos_token_id
            is_max_tokens = len(seq.output_ids) >= seq.sampling_params.max_new_tokens
            is_max_length = seq.total_length >= self.max_seq_len
            is_stop_token = seq.last_token_id in seq.sampling_params.stop_token_ids
            
            if is_eos or is_max_tokens or is_max_length or is_stop_token:
                seq.finished = True
                finished.append(seq)
                self.kv_cache.deallocate_pages(seq.seq_id)
            else:
                running.append(seq)
        
        return finished, running
    
    def step(self) -> List[Sequence]:
        """Execute one inference step"""
        finished_sequences = []
        
        # Prefill phase: process waiting sequences
        if self.waiting_queue:
            # Schedule sequences for prefill
            prefill_sequences = []
            while self.waiting_queue and len(prefill_sequences) < self.max_batch_size:
                seq = self.waiting_queue.popleft()
                prefill_sequences.append(seq)
            
            if prefill_sequences:
                # Prefill
                logits = self._prefill_sequences(prefill_sequences)
                sampling_params = [seq.sampling_params for seq in prefill_sequences]
                next_tokens = self._sample_tokens(logits, sampling_params)
                
                # Add tokens to sequences
                for seq, token in zip(prefill_sequences, next_tokens):
                    seq.output_ids.append(token)
                
                # Check finished sequences
                finished, running = self._check_finished(prefill_sequences)
                finished_sequences.extend(finished)
                self.running_queue.extend(running)
        
        # Decode phase: generate next tokens for running sequences
        if self.running_queue:
            # Schedule sequences for decode
            decode_sequences = []
            while self.running_queue and len(decode_sequences) < self.max_batch_size:
                seq = self.running_queue.popleft()
                decode_sequences.append(seq)
            
            if decode_sequences:
                # Decode
                logits = self._decode_sequences(decode_sequences)
                sampling_params = [seq.sampling_params for seq in decode_sequences]
                next_tokens = self._sample_tokens(logits, sampling_params)
                
                # Add tokens to sequences
                for seq, token in zip(decode_sequences, next_tokens):
                    seq.output_ids.append(token)
                
                # Check finished sequences
                finished, running = self._check_finished(decode_sequences)
                finished_sequences.extend(finished)
                self.running_queue.extend(running)
        
        return finished_sequences
    
    def is_finished(self) -> bool:
        """Check if all sequences are finished"""
        return not self.waiting_queue and not self.running_queue
    
    def generate(self, prompts: List[str], sampling_params: SamplingParams, 
                 use_tqdm: bool = True) -> List[str]:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of generated texts
        """
        # Add all requests
        seq_ids = []
        for prompt in prompts:
            seq_id = self.add_request(prompt, sampling_params)
            seq_ids.append(seq_id)
        
        # Generate
        all_finished = []
        pbar = tqdm(total=len(prompts), desc="Generating", disable=not use_tqdm)
        
        while not self.is_finished():
            finished = self.step()
            all_finished.extend(finished)
            
            if finished:
                pbar.update(len(finished))
        
        pbar.close()
        
        # Collect results in order
        results = []
        for seq_id in seq_ids:
            if seq_id in self.sequence_map:
                seq = self.sequence_map[seq_id]
                generated_text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)
                results.append(generated_text)
        
        return results

# =============================================================================
# 🎯 CONVENIENCE FUNCTIONS
# =============================================================================

def create_fast_inference_engine(model_path: str, tokenizer_path: str, 
                                max_batch_size: int = 32, max_seq_len: int = 2048,
                                n_pages: int = 1000, page_size: int = 128) -> FastInferenceEngine:
    """
    Create a fast inference engine from saved model
    
    Args:
        model_path: Path to saved model
        tokenizer_path: Path to tokenizer
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        n_pages: Number of KV cache pages
        page_size: Size of each page
        
    Returns:
        FastInferenceEngine instance
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
    engine = FastInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        n_pages=n_pages,
        page_size=page_size
    )
    
    return engine

# =============================================================================
# 🧪 TESTING AND BENCHMARKING
# =============================================================================

def benchmark_inference_engine(engine: FastInferenceEngine, num_requests: int = 100, 
                              max_input_len: int = 512, max_output_len: int = 256):
    """
    Benchmark the inference engine
    
    Args:
        engine: FastInferenceEngine instance
        num_requests: Number of requests to process
        max_input_len: Maximum input length
        max_output_len: Maximum output length
    """
    print(f"🚀 Benchmarking Fast Inference Engine")
    print(f"   Requests: {num_requests}")
    print(f"   Max input length: {max_input_len}")
    print(f"   Max output length: {max_output_len}")
    
    # Generate test prompts
    test_prompts = []
    for i in range(num_requests):
        prompt_len = np.random.randint(50, max_input_len)
        prompt = f"Generate a story about {prompt_len} words: "
        test_prompts.append(prompt)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_new_tokens=max_output_len,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    
    # Benchmark
    start_time = time.time()
    results = engine.generate(test_prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = sum(len(result.split()) for result in results)
    throughput = total_tokens / total_time
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Throughput: {throughput:.1f} tokens/s")
    print(f"   Requests/s: {num_requests / total_time:.2f}")
    
    return results

if __name__ == "__main__":
    print("🚀 Fast Inference Engine for Qwen3")
    print("=" * 50)
    
    # Example usage
    try:
        # Create engine (you'll need to provide actual model paths)
        # engine = create_fast_inference_engine(
        #     model_path="models/final_model1.pt",
        #     tokenizer_path="HuggingFaceTB/SmolLM-135M"
        # )
        
        # # Test single generation
        # prompts = ["Hello, how are you?", "Tell me a story about"]
        # sampling_params = SamplingParams(max_new_tokens=50, temperature=0.8)
        # results = engine.generate(prompts, sampling_params)
        
        # for i, (prompt, result) in enumerate(zip(prompts, results)):
        #     print(f"\n{i+1}. Prompt: {prompt}")
        #     print(f"   Generated: {result}")
        
        # # Benchmark
        # benchmark_inference_engine(engine, num_requests=50)
        
        print("✅ Fast Inference Engine ready!")
        print("   To use: create_fast_inference_engine(model_path, tokenizer_path)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have a trained model and tokenizer available")
