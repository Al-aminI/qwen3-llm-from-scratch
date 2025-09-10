"""
Token sampling utilities for fast inference.

This module provides utilities for sampling tokens from model outputs
with various sampling strategies like temperature, top-k, and top-p.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class SamplingParams:
    """
    Sampling parameters for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
        top_k: Number of top tokens to consider (0 = no limit)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = no limit)
        repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
        stop_token_ids: List of token IDs to stop generation at
    """
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []
        
        # Validate parameters
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


def sample_tokens(logits: torch.Tensor, sampling_params: SamplingParams, 
                 previous_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sample tokens from logits using the specified sampling parameters.
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        sampling_params: Sampling parameters
        previous_tokens: Previously generated tokens for repetition penalty
        
    Returns:
        Sampled token IDs (batch_size,)
    """
    batch_size, vocab_size = logits.shape
    
    # Apply temperature scaling
    if sampling_params.temperature > 0:
        logits = logits / sampling_params.temperature
    else:
        # Greedy sampling (temperature = 0)
        return logits.argmax(dim=-1)
    
    # Apply repetition penalty
    if sampling_params.repetition_penalty != 1.0 and previous_tokens is not None:
        logits = apply_repetition_penalty(logits, previous_tokens, sampling_params.repetition_penalty)
    
    # Apply top-k filtering
    if sampling_params.top_k > 0:
        logits = apply_top_k_filtering(logits, sampling_params.top_k)
    
    # Apply top-p (nucleus) filtering
    if sampling_params.top_p < 1.0:
        logits = apply_top_p_filtering(logits, sampling_params.top_p)
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return tokens


def apply_repetition_penalty(logits: torch.Tensor, previous_tokens: torch.Tensor, 
                           penalty: float) -> torch.Tensor:
    """
    Apply repetition penalty to logits.
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        previous_tokens: Previously generated tokens
        penalty: Repetition penalty factor
        
    Returns:
        Logits with repetition penalty applied
    """
    if penalty == 1.0:
        return logits
    
    # Create penalty mask
    penalty_mask = torch.ones_like(logits)
    
    # Apply penalty to previously generated tokens
    for token_id in previous_tokens.unique():
        if token_id < logits.size(-1):
            penalty_mask[:, token_id] = penalty
    
    # Apply penalty
    logits = logits * penalty_mask
    
    return logits


def apply_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Apply top-k filtering to logits.
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        top_k: Number of top tokens to keep
        
    Returns:
        Logits with top-k filtering applied
    """
    if top_k <= 0:
        return logits
    
    # Get top-k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
    
    # Create filtered logits
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
    
    return filtered_logits


def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits.
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        top_p: Cumulative probability threshold
        
    Returns:
        Logits with top-p filtering applied
    """
    if top_p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # Find indices to remove
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Get indices to remove
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    
    # Apply filtering
    filtered_logits = logits.clone()
    filtered_logits.scatter_(-1, indices_to_remove, float('-inf'))
    
    return filtered_logits


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling (always pick the most likely token).
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        
    Returns:
        Greedily sampled token IDs (batch_size,)
    """
    return logits.argmax(dim=-1)


def sample_random(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Random sampling with temperature.
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        temperature: Sampling temperature
        
    Returns:
        Randomly sampled token IDs (batch_size,)
    """
    if temperature > 0:
        logits = logits / temperature
    
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return tokens


def sample_beam_search(logits: torch.Tensor, beam_size: int = 4, 
                      length_penalty: float = 1.0) -> List[torch.Tensor]:
    """
    Beam search sampling (simplified version).
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        beam_size: Number of beams to maintain
        length_penalty: Length penalty factor
        
    Returns:
        List of beam sequences
    """
    # This is a simplified implementation
    # For a full beam search, you'd need to maintain multiple sequences
    # and handle the search across multiple time steps
    
    batch_size, vocab_size = logits.shape
    
    # Get top-k candidates
    top_k_logits, top_k_indices = torch.topk(logits, beam_size, dim=-1)
    
    # Apply length penalty (simplified)
    if length_penalty != 1.0:
        top_k_logits = top_k_logits / (length_penalty ** 0.5)
    
    # Return top candidates
    beams = []
    for i in range(beam_size):
        beam = top_k_indices[:, i]
        beams.append(beam)
    
    return beams
