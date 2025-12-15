"""
Unit tests for replay buffer implementation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.replay_buffer import OfflineReplayBuffer


class TestOfflineReplayBuffer:
    """Tests for the OfflineReplayBuffer."""
    
    @pytest.fixture
    def buffer(self):
        """Create a replay buffer for testing."""
        return OfflineReplayBuffer(
            state_dim=1,
            buffer_size=1000,
        )
    
    @pytest.fixture
    def filled_buffer(self, buffer):
        """Create a buffer with some data."""
        for i in range(500):
            buffer.add(
                state=np.array([i], dtype=np.float32),
                action=i % 25,
                reward=1.0 if i % 10 == 0 else 0.0,
                next_state=np.array([i + 1], dtype=np.float32),
                done=i % 50 == 49,
            )
        return buffer
    
    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert len(buffer) == 0
        assert buffer.buffer_size == 1000
        assert buffer.state_dim == 1
    
    def test_add_transition(self, buffer):
        """Test adding transitions."""
        buffer.add(
            state=np.array([0.0]),
            action=0,
            reward=1.0,
            next_state=np.array([1.0]),
            done=False,
        )
        
        assert len(buffer) == 1
    
    def test_add_multiple_transitions(self, buffer):
        """Test adding multiple transitions."""
        for i in range(100):
            buffer.add(
                state=np.array([float(i)]),
                action=i % 25,
                reward=0.0,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
        
        assert len(buffer) == 100
    
    def test_buffer_overflow(self, buffer):
        """Test buffer overflow behavior."""
        # Fill beyond capacity
        for i in range(1500):
            buffer.add(
                state=np.array([float(i)]),
                action=0,
                reward=0.0,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
        
        assert len(buffer) == 1000  # Should not exceed capacity
    
    def test_sample_batch(self, filled_buffer):
        """Test sampling a batch."""
        batch = filled_buffer.sample(batch_size=32)
        
        assert 'states' in batch
        assert 'actions' in batch
        assert 'rewards' in batch
        assert 'next_states' in batch
        assert 'dones' in batch
        
        assert batch['states'].shape == (32, 1)
        assert batch['actions'].shape == (32, 1)
        assert batch['rewards'].shape == (32, 1)
        assert batch['next_states'].shape == (32, 1)
        assert batch['dones'].shape == (32, 1)
    
    def test_sample_batch_sizes(self, filled_buffer):
        """Test sampling different batch sizes."""
        for batch_size in [1, 16, 64, 128]:
            batch = filled_buffer.sample(batch_size=batch_size)
            assert batch['states'].shape[0] == batch_size
    
    def test_sample_larger_than_buffer(self, buffer):
        """Test sampling when batch size > buffer size."""
        # Add small amount of data
        for i in range(10):
            buffer.add(
                state=np.array([float(i)]),
                action=0,
                reward=0.0,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
        
        # Sample more than available (should work with replacement)
        batch = buffer.sample(batch_size=32)
        assert batch['states'].shape[0] == 32
    
    def test_data_types(self, filled_buffer):
        """Test that data types are correct."""
        batch = filled_buffer.sample(batch_size=32)
        
        assert batch['states'].dtype == np.float32
        assert batch['rewards'].dtype == np.float32
        assert batch['next_states'].dtype == np.float32
        assert batch['dones'].dtype == np.float32
    
    def test_compute_statistics(self, filled_buffer):
        """Test computing buffer statistics."""
        stats = filled_buffer.compute_statistics()
        
        assert 'n_transitions' in stats
        assert stats['n_transitions'] == 500
    
    def test_save_load(self, filled_buffer):
        """Test saving and loading buffer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "buffer.pkl"
            
            # Save
            filled_buffer.save(str(save_path))
            assert save_path.exists()
            
            # Load
            loaded_buffer = OfflineReplayBuffer.load(str(save_path))
            
            assert len(loaded_buffer) == len(filled_buffer)
            
            # Check data matches
            for i in range(min(100, len(filled_buffer))):
                assert np.allclose(filled_buffer.states[i], loaded_buffer.states[i])
    
    def test_empty_buffer_sample(self, buffer):
        """Test that sampling from empty buffer raises error."""
        with pytest.raises(ValueError):
            buffer.sample(batch_size=32)
    
    def test_state_normalization(self, filled_buffer):
        """Test state normalization if implemented."""
        batch = filled_buffer.sample(batch_size=32)
        states = batch['states']
        
        # States should be finite
        assert np.isfinite(states).all()
    
    def test_done_flags(self, filled_buffer):
        """Test that done flags are binary."""
        batch = filled_buffer.sample(batch_size=100)
        dones = batch['dones']
        
        assert np.all((dones == 0) | (dones == 1))
    
    def test_action_range(self, filled_buffer):
        """Test that actions are in valid range."""
        batch = filled_buffer.sample(batch_size=100)
        actions = batch['actions']
        
        assert np.all(actions >= 0)
        assert np.all(actions < 25)


class TestBufferEpisodeTracking:
    """Tests for episode tracking in buffer."""
    
    def test_episode_boundaries(self):
        """Test tracking episode boundaries."""
        buffer = OfflineReplayBuffer(state_dim=1, buffer_size=1000)
        
        # Add two episodes
        # Episode 1
        for i in range(10):
            buffer.add(
                state=np.array([float(i)]),
                action=0,
                reward=0.0,
                next_state=np.array([float(i + 1)]),
                done=(i == 9),
            )
        
        # Episode 2
        for i in range(5):
            buffer.add(
                state=np.array([float(i + 100)]),
                action=0,
                reward=1.0,
                next_state=np.array([float(i + 101)]),
                done=(i == 4),
            )
        
        assert len(buffer) == 15
        
        # Check done flags
        assert buffer.dones[9] == 1.0
        assert buffer.dones[14] == 1.0


class TestBufferPerformance:
    """Performance tests for buffer."""
    
    def test_large_buffer_performance(self):
        """Test performance with large buffer."""
        buffer = OfflineReplayBuffer(state_dim=1, buffer_size=100000)
        
        # Fill buffer
        for i in range(100000):
            buffer.add(
                state=np.array([float(i)]),
                action=i % 25,
                reward=0.0,
                next_state=np.array([float(i + 1)]),
                done=i % 100 == 99,
            )
        
        assert len(buffer) == 100000
        
        # Test sampling speed (should be fast)
        import time
        start = time.time()
        for _ in range(1000):
            buffer.sample(batch_size=256)
        elapsed = time.time() - start
        
        # Should sample 1000 batches in under 2 seconds
        assert elapsed < 2.0, f"Sampling too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
