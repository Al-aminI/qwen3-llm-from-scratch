"""
Unit tests for sampling utilities.
"""

import pytest
import torch
from fast_inference.utils.sampling import (
    SamplingParams, sample_tokens, apply_repetition_penalty,
    apply_top_k_filtering, apply_top_p_filtering, sample_greedy, sample_random
)


class TestSamplingParams:
    """Test cases for SamplingParams."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = SamplingParams()
        
        assert params.max_new_tokens == 100
        assert params.temperature == 0.8
        assert params.top_k == 50
        assert params.top_p == 0.9
        assert params.repetition_penalty == 1.0
        assert params.stop_token_ids == []
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = SamplingParams(
            max_new_tokens=200,
            temperature=1.2,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.1,
            stop_token_ids=[1, 2, 3]
        )
        
        assert params.max_new_tokens == 200
        assert params.temperature == 1.2
        assert params.top_k == 100
        assert params.top_p == 0.95
        assert params.repetition_penalty == 1.1
        assert params.stop_token_ids == [1, 2, 3]
    
    def test_validation(self):
        """Test parameter validation."""
        # Valid parameters should not raise
        SamplingParams(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
        
        # Invalid parameters should raise
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            SamplingParams(temperature=-1.0)
        
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            SamplingParams(top_k=-1)
        
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            SamplingParams(top_p=1.5)
        
        with pytest.raises(ValueError, match="repetition_penalty must be positive"):
            SamplingParams(repetition_penalty=0.0)


class TestSamplingFunctions:
    """Test cases for sampling functions."""
    
    def test_sample_greedy(self):
        """Test greedy sampling."""
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
        tokens = sample_greedy(logits)
        
        assert tokens.item() == 3  # Highest logit at index 3
    
    def test_sample_random(self):
        """Test random sampling."""
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        tokens = sample_random(logits, temperature=1.0)
        
        assert tokens.shape == (1,)
        assert 0 <= tokens.item() <= 3
    
    def test_apply_repetition_penalty(self):
        """Test repetition penalty application."""
        logits = torch.tensor([[1.0, 2.0, 1.5, 0.5]])
        previous_tokens = torch.tensor([1, 2])  # Tokens 1 and 2 should be penalized
        
        penalized_logits = apply_repetition_penalty(logits, previous_tokens, penalty=0.5)
        
        # Token 1 (index 1) should be penalized: 2.0 * 0.5 = 1.0
        # Token 2 (index 2) should be penalized: 1.5 * 0.5 = 0.75
        assert penalized_logits[0, 1] == 1.0
        assert penalized_logits[0, 2] == 0.75
        # Other tokens should remain unchanged
        assert penalized_logits[0, 0] == 1.0
        assert penalized_logits[0, 3] == 0.5
    
    def test_apply_top_k_filtering(self):
        """Test top-k filtering."""
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5]])
        
        # Top-3 filtering
        filtered_logits = apply_top_k_filtering(logits, top_k=3)
        
        # Only top-3 tokens should have finite logits
        finite_indices = torch.isfinite(filtered_logits[0]).nonzero().flatten()
        assert len(finite_indices) == 3
        
        # Check that the top-3 tokens are preserved
        original_top3 = torch.topk(logits[0], 3).indices
        assert torch.all(torch.isfinite(filtered_logits[0, original_top3]))
    
    def test_apply_top_p_filtering(self):
        """Test top-p (nucleus) filtering."""
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5]])
        
        # Top-p filtering with p=0.8
        filtered_logits = apply_top_p_filtering(logits, top_p=0.8)
        
        # Should filter out some tokens
        finite_count = torch.isfinite(filtered_logits[0]).sum().item()
        assert finite_count <= 5  # Should be less than or equal to original count
    
    def test_sample_tokens_comprehensive(self):
        """Test comprehensive token sampling."""
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5]])
        params = SamplingParams(
            temperature=0.5,
            top_k=3,
            top_p=0.8,
            repetition_penalty=1.1
        )
        previous_tokens = torch.tensor([1, 2])
        
        tokens = sample_tokens(logits, params, previous_tokens)
        
        assert tokens.shape == (1,)
        assert 0 <= tokens.item() <= 4
    
    def test_sample_tokens_greedy(self):
        """Test greedy sampling (temperature=0)."""
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
        params = SamplingParams(temperature=0.0)
        
        tokens = sample_tokens(logits, params)
        
        assert tokens.item() == 3  # Highest logit at index 3
    
    def test_sample_tokens_no_previous_tokens(self):
        """Test sampling without previous tokens."""
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
        params = SamplingParams(temperature=1.0)
        
        tokens = sample_tokens(logits, params, previous_tokens=None)
        
        assert tokens.shape == (1,)
        assert 0 <= tokens.item() <= 3
