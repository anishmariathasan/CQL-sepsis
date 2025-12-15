"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def set_seed():
    """Fixture to set random seeds for reproducibility."""
    def _set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _set_seed


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 32
    return {
        'states': np.random.randn(batch_size, 1).astype(np.float32),
        'actions': np.random.randint(0, 25, (batch_size, 1)),
        'rewards': np.random.randn(batch_size, 1).astype(np.float32),
        'next_states': np.random.randn(batch_size, 1).astype(np.float32),
        'dones': np.random.randint(0, 2, (batch_size, 1)).astype(np.float32),
    }


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    length = 20
    return {
        'states': [np.array([i], dtype=np.float32) for i in range(length + 1)],
        'actions': [np.random.randint(0, 25) for _ in range(length)],
        'rewards': [0.0] * (length - 1) + [1.0],  # Sparse reward
        'dones': [False] * (length - 1) + [True],
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
