"""
Advanced fast inference engine with paged attention.

This module provides an advanced inference engine with paged attention,
continuous batching, and other vLLM-style optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import deque
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ...utils.sampling import SamplingParams
    from ..cache.paged_cache import PagedKVCache
    from ..attention.optimized_attention import OptimizedAttention


@dataclass
class Sequence:
    """Represents a single generation sequence."""
    seq_id: int
    prompt: str
    input_ids: torch.Tensor
    output_ids: List[int]
    sampling_params: 'SamplingParams'
    batch_idx: int = -1
    finished: bool = False
    prefill_done: bool = False
    
    @property
    def total_length(self) -> int:
        return len(self.input_ids) + len(self.output_ids)
    
    @property
    def last_token_id(self) -> int:
        return self.output_ids[-1] if self.output_ids else self.input_ids[-1].item()


class FastInferenceEngine:
    """
    🚀 FAST INFERENCE ENGINE
    
    High-performance inference engine with:
    - Paged KV cache for memory efficiency
    - Continuous batching for throughput
    - CUDA graphs for optimization
    - Dynamic scheduling
    """
    
    def __init__(self, model: nn.Module, tokenizer, config, 
                 max_batch_size: int = 32, max_seq_len: int = 2048, 
                 n_pages: int = 1000, page_size: int = 128):
        """
        Initialize fast inference engine.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            config: Model configuration
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            n_pages: Number of KV cache pages
            page_size: Size of each page
        """
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
        """Replace standard attention with optimized attention."""
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
    
    def add_request(self, prompt: str, sampling_params: 'SamplingParams') -> int:
        """Add a new generation request."""
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
        """Prefill phase: process input prompts."""
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
        """Decode phase: generate next tokens."""
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
    
    def _sample_tokens(self, logits: torch.Tensor, sampling_params: List['SamplingParams']) -> List[int]:
        """Sample next tokens from logits."""
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
        """Check which sequences are finished."""
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
        """Execute one inference step."""
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
        """Check if all sequences are finished."""
        return not self.waiting_queue and not self.running_queue
    
    def generate(self, prompts: List[str], sampling_params: 'SamplingParams', 
                 use_tqdm: bool = True) -> List[str]:
        """
        Generate text for multiple prompts.
        
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


class OptimizedTransformerBlock(nn.Module):
    """
    🏗️ OPTIMIZED TRANSFORMER BLOCK
    
    Transformer block with optimized attention and KV caching.
    """
    
    def __init__(self, config, kv_cache: 'PagedKVCache'):
        super().__init__()
        self.attention = OptimizedAttention(config, kv_cache)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        
        # Pre-norm architecture
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, seq_id: int, positions: torch.Tensor, use_cache: bool = True):
        """Forward pass with KV caching."""
        # Pre-norm attention with residual connection
        attn_out = self.attention(self.norm1(x), seq_id, positions, use_cache)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


def create_fast_inference_engine(model_path: str, tokenizer_path: str, 
                                max_batch_size: int = 32, max_seq_len: int = 2048,
                                n_pages: int = 1000, page_size: int = 128) -> FastInferenceEngine:
    """
    Create a fast inference engine from saved model.
    
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
