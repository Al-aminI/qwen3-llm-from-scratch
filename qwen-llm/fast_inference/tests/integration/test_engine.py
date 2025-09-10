"""
Integration tests for inference engines.

These tests require actual model files and are marked as slow.
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch

from fast_inference.core.engine import SimpleFastInference
from fast_inference.utils.sampling import SamplingParams


class TestSimpleFastInference:
    """Integration tests for SimpleFastInference."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.parameters.return_value = [torch.tensor([1.0])]
        model.dtype = torch.float16
        model.transformer_blocks = [Mock() for _ in range(2)]
        
        # Mock transformer block components
        for block in model.transformer_blocks:
            block.attention = Mock()
            block.attention.q_proj = Mock()
            block.attention.k_proj = Mock()
            block.attention.v_proj = Mock()
            block.attention.w_o = Mock()
            block.attention.q_norm = Mock()
            block.attention.k_norm = Mock()
            block.feed_forward = Mock()
            block.norm1 = Mock()
            block.norm2 = Mock()
            
            # Mock state_dict methods
            for component in [block.attention.q_proj, block.attention.k_proj, 
                            block.attention.v_proj, block.attention.w_o,
                            block.attention.q_norm, block.attention.k_norm,
                            block.feed_forward, block.norm1, block.norm2]:
                component.load_state_dict = Mock()
                component.state_dict = Mock(return_value={})
        
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.tensor([1, 2, 3, 4, 5]))
        tokenizer.decode = Mock(return_value="Generated text")
        tokenizer.eos_token_id = 2
        tokenizer.pad_token = None
        return tokenizer
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock()
        config.d_model = 512
        config.n_kv_heads = 8
        config.d_k = 64
        config.rms_norm_eps = 1e-6
        config.attention_bias = False
        config.dropout = 0.1
        config.d_ff = 2048
        return config
    
    def test_engine_initialization(self, mock_model, mock_tokenizer, mock_config):
        """Test engine initialization."""
        engine = SimpleFastInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
            max_seq_len=1024
        )
        
        assert engine.model == mock_model
        assert engine.tokenizer == mock_tokenizer
        assert engine.config == mock_config
        assert engine.max_seq_len == 1024
        assert engine.next_seq_id == 0
    
    def test_generate_single_basic(self, mock_model, mock_tokenizer, mock_config):
        """Test basic single text generation."""
        # Mock the model's forward pass
        mock_model.token_embedding = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.position_dropout = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.norm = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.lm_head = Mock(return_value=torch.randn(1, 5, 1000))
        
        # Mock transformer blocks
        for block in mock_model.transformer_blocks:
            block.return_value = torch.randn(1, 5, 512)
        
        engine = SimpleFastInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
            max_seq_len=1024
        )
        
        # Mock the sampling function
        with patch.object(engine, '_sample_token', return_value=2):
            result = engine.generate_single(
                prompt="Hello, world!",
                max_new_tokens=10,
                temperature=0.8
            )
        
        assert result == "Generated text"
        mock_tokenizer.encode.assert_called_once()
        mock_tokenizer.decode.assert_called_once()
    
    def test_generate_batch_basic(self, mock_model, mock_tokenizer, mock_config):
        """Test basic batch text generation."""
        # Mock the model's forward pass
        mock_model.token_embedding = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.position_dropout = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.norm = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.lm_head = Mock(return_value=torch.randn(1, 5, 1000))
        
        # Mock transformer blocks
        for block in mock_model.transformer_blocks:
            block.return_value = torch.randn(1, 5, 512)
        
        engine = SimpleFastInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
            max_seq_len=1024
        )
        
        # Mock the sampling function
        with patch.object(engine, '_sample_token', return_value=2):
            prompts = ["Hello", "World", "Test"]
            results = engine.generate_batch(
                prompts=prompts,
                max_new_tokens=10,
                temperature=0.8
            )
        
        assert len(results) == 3
        assert all(result == "Generated text" for result in results)
        assert mock_tokenizer.encode.call_count == 3
        assert mock_tokenizer.decode.call_count == 3
    
    def test_sampling_parameters(self, mock_model, mock_tokenizer, mock_config):
        """Test different sampling parameters."""
        # Mock the model's forward pass
        mock_model.token_embedding = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.position_dropout = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.norm = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.lm_head = Mock(return_value=torch.randn(1, 5, 1000))
        
        # Mock transformer blocks
        for block in mock_model.transformer_blocks:
            block.return_value = torch.randn(1, 5, 512)
        
        engine = SimpleFastInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
            max_seq_len=1024
        )
        
        # Test different sampling parameters
        test_cases = [
            {"temperature": 0.0, "top_k": 0, "top_p": 1.0},  # Greedy
            {"temperature": 1.0, "top_k": 50, "top_p": 0.9},  # Balanced
            {"temperature": 1.5, "top_k": 100, "top_p": 0.95},  # Creative
        ]
        
        for params in test_cases:
            with patch.object(engine, '_sample_token', return_value=2):
                result = engine.generate_single(
                    prompt="Test prompt",
                    max_new_tokens=5,
                    **params
                )
            
            assert result == "Generated text"
    
    def test_sequence_id_tracking(self, mock_model, mock_tokenizer, mock_config):
        """Test sequence ID tracking."""
        engine = SimpleFastInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
            max_seq_len=1024
        )
        
        # Mock the model's forward pass
        mock_model.token_embedding = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.position_dropout = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.norm = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.lm_head = Mock(return_value=torch.randn(1, 5, 1000))
        
        # Mock transformer blocks
        for block in mock_model.transformer_blocks:
            block.return_value = torch.randn(1, 5, 512)
        
        with patch.object(engine, '_sample_token', return_value=2):
            # Generate multiple sequences
            result1 = engine.generate_single("Prompt 1", max_new_tokens=5)
            result2 = engine.generate_single("Prompt 2", max_new_tokens=5)
            result3 = engine.generate_single("Prompt 3", max_new_tokens=5)
        
        # Check that sequence IDs are tracked
        assert engine.next_seq_id == 3
        assert all(result == "Generated text" for result in [result1, result2, result3])
    
    @pytest.mark.slow
    def test_memory_cleanup(self, mock_model, mock_tokenizer, mock_config):
        """Test memory cleanup after generation."""
        engine = SimpleFastInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
            max_seq_len=1024
        )
        
        # Mock the model's forward pass
        mock_model.token_embedding = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.position_dropout = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.norm = Mock(return_value=torch.randn(1, 5, 512))
        mock_model.lm_head = Mock(return_value=torch.randn(1, 5, 1000))
        
        # Mock transformer blocks
        for block in mock_model.transformer_blocks:
            block.return_value = torch.randn(1, 5, 512)
        
        with patch.object(engine, '_sample_token', return_value=2):
            # Generate a sequence
            result = engine.generate_single("Test prompt", max_new_tokens=5)
        
        # Check that cache is cleaned up
        assert result == "Generated text"
        # The cache should be empty after generation
        assert len(engine.kv_cache.cache) == 0
