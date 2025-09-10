"""
Pytest configuration and fixtures for fast inference tests.
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock


@pytest.fixture
def device():
    """Get the device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    config = Mock()
    config.d_model = 512
    config.n_heads = 8
    config.n_kv_heads = 8
    config.n_kv_groups = 1
    config.d_k = 64
    config.d_ff = 2048
    config.rms_norm_eps = 1e-6
    config.attention_bias = False
    config.dropout = 0.1
    config.max_seq_len = 2048
    config.vocab_size = 32000
    return config


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=torch.tensor([1, 2, 3, 4, 5]))
    tokenizer.decode = Mock(return_value="Generated text")
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = None
    return tokenizer


@pytest.fixture
def sample_logits():
    """Create sample logits for testing."""
    return torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5]])


@pytest.fixture
def sample_kv_tensors():
    """Create sample K and V tensors for testing."""
    k = torch.randn(8, 10, 64, dtype=torch.float16)
    v = torch.randn(8, 10, 64, dtype=torch.float16)
    return k, v


@pytest.fixture
def sample_prompts():
    """Create sample prompts for testing."""
    return [
        "Hello, how are you?",
        "Tell me a joke about programming",
        "Write a short story about",
        "Explain the concept of machine learning",
        "What is the meaning of life?"
    ]


@pytest.fixture
def sample_sampling_params():
    """Create sample sampling parameters."""
    from fast_inference.utils.sampling import SamplingParams
    return SamplingParams(
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0
    )


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "cpu: marks tests that run on CPU only")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might take time
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add GPU marker to tests that might need GPU
        if "cuda" in item.name or "gpu" in item.name:
            item.add_marker(pytest.mark.gpu)
