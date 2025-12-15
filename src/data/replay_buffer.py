"""
Offline Replay Buffer for batch reinforcement learning.

This module provides efficient storage and sampling of offline datasets
for training offline RL algorithms like CQL.

Features:
- Efficient numpy-based storage
- Batch sampling with replacement
- Dataset statistics computation
- Support for loading/saving datasets
- Memory-efficient circular buffer implementation
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class OfflineReplayBuffer:
    """
    Manages offline datasets for batch reinforcement learning.
    
    This buffer stores transitions (s, a, r, s', done) from an offline dataset
    and provides efficient batch sampling for training.
    
    Features:
        - Efficient numpy-based storage
        - Random batch sampling with replacement
        - Dataset statistics computation
        - Save/load functionality for persistence
        - Memory-efficient implementation
    
    Args:
        capacity: Maximum number of transitions to store
        state_dtype: Data type for states (default: np.int32 for discrete states)
        action_dtype: Data type for actions (default: np.int32)
    
    Example:
        >>> buffer = OfflineReplayBuffer(capacity=100000)
        >>> buffer.add(state=10, action=5, reward=0.0, next_state=11, done=False)
        >>> batch = buffer.sample(batch_size=256)
        >>> states, actions, rewards, next_states, dones = batch
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        state_dtype: np.dtype = np.int32,
        action_dtype: np.dtype = np.int32,
    ):
        self.capacity = capacity
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros(capacity, dtype=state_dtype)
        self.actions = np.zeros(capacity, dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros(capacity, dtype=state_dtype)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Current size and position
        self.size = 0
        self.position = 0
        
        # Episode tracking for statistics
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_episode_return = 0.0
        self._current_episode_length = 0
        
        logger.info(f"Initialized OfflineReplayBuffer with capacity {capacity}")
    
    def add(
        self,
        state: Union[int, np.ndarray],
        action: int,
        reward: float,
        next_state: Union[int, np.ndarray],
        done: bool,
    ) -> None:
        """
        Add a single transition to the buffer.
        
        Args:
            state: Current state (integer index or array)
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode terminated
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Track episode statistics
        self._current_episode_return += reward
        self._current_episode_length += 1
        
        if done:
            self.episode_returns.append(self._current_episode_return)
            self.episode_lengths.append(self._current_episode_length)
            self._current_episode_return = 0.0
            self._current_episode_length = 0
    
    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Add a batch of transitions to the buffer.
        
        Args:
            states: Array of states
            actions: Array of actions
            rewards: Array of rewards
            next_states: Array of next states
            dones: Array of done flags
        """
        batch_size = len(states)
        
        for i in range(batch_size):
            self.add(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=bool(dones[i]),
            )
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample random indices with replacement
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def get_all_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data in the buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        return (
            self.states[:self.size].copy(),
            self.actions[:self.size].copy(),
            self.rewards[:self.size].copy(),
            self.next_states[:self.size].copy(),
            self.dones[:self.size].copy(),
        )
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.
        
        Returns:
            Dictionary containing:
                - size: Number of transitions
                - n_episodes: Number of complete episodes
                - avg_return: Average episode return
                - std_return: Standard deviation of returns
                - avg_episode_length: Average episode length
                - survival_rate: Fraction of episodes with positive return
                - action_distribution: Frequency of each action
                - state_coverage: Number of unique states visited
                - reward_statistics: Min, max, mean, std of rewards
        """
        if self.size == 0:
            return {"size": 0, "error": "Empty buffer"}
        
        stats = {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
        }
        
        # Episode statistics
        if len(self.episode_returns) > 0:
            returns = np.array(self.episode_returns)
            lengths = np.array(self.episode_lengths)
            
            stats["n_episodes"] = len(returns)
            stats["avg_return"] = float(np.mean(returns))
            stats["std_return"] = float(np.std(returns))
            stats["min_return"] = float(np.min(returns))
            stats["max_return"] = float(np.max(returns))
            stats["avg_episode_length"] = float(np.mean(lengths))
            stats["std_episode_length"] = float(np.std(lengths))
            
            # Survival rate (positive return = survival)
            stats["survival_rate"] = float(np.mean(returns > 0))
        
        # Action distribution
        actions = self.actions[:self.size]
        unique_actions, counts = np.unique(actions, return_counts=True)
        action_dist = {int(a): int(c) for a, c in zip(unique_actions, counts)}
        stats["action_distribution"] = action_dist
        stats["n_unique_actions"] = len(unique_actions)
        
        # State coverage
        unique_states = np.unique(self.states[:self.size])
        stats["n_unique_states"] = len(unique_states)
        stats["state_coverage"] = list(unique_states[:100])  # First 100 unique states
        
        # Reward statistics
        rewards = self.rewards[:self.size]
        stats["reward_mean"] = float(np.mean(rewards))
        stats["reward_std"] = float(np.std(rewards))
        stats["reward_min"] = float(np.min(rewards))
        stats["reward_max"] = float(np.max(rewards))
        
        # Terminal state statistics
        dones = self.dones[:self.size]
        stats["terminal_fraction"] = float(np.mean(dones))
        
        return stats
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the buffer to disk.
        
        Args:
            filepath: Path to save the buffer (will add .pkl extension if needed)
        """
        filepath = Path(filepath)
        if filepath.suffix != ".pkl":
            filepath = filepath.with_suffix(".pkl")
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "states": self.states[:self.size],
            "actions": self.actions[:self.size],
            "rewards": self.rewards[:self.size],
            "next_states": self.next_states[:self.size],
            "dones": self.dones[:self.size],
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "metadata": {
                "size": self.size,
                "capacity": self.capacity,
                "state_dtype": str(self.state_dtype),
                "action_dtype": str(self.action_dtype),
            },
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved buffer ({self.size} transitions) to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load buffer data from disk.
        
        Args:
            filepath: Path to the saved buffer file
        """
        filepath = Path(filepath)
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Load data
        loaded_size = len(data["states"])
        
        if loaded_size > self.capacity:
            logger.warning(
                f"Loaded data size ({loaded_size}) exceeds capacity ({self.capacity}). "
                f"Truncating to capacity."
            )
            loaded_size = self.capacity
        
        self.states[:loaded_size] = data["states"][:loaded_size]
        self.actions[:loaded_size] = data["actions"][:loaded_size]
        self.rewards[:loaded_size] = data["rewards"][:loaded_size]
        self.next_states[:loaded_size] = data["next_states"][:loaded_size]
        self.dones[:loaded_size] = data["dones"][:loaded_size]
        
        self.size = loaded_size
        self.position = loaded_size % self.capacity
        
        self.episode_returns = data.get("episode_returns", [])
        self.episode_lengths = data.get("episode_lengths", [])
        
        logger.info(f"Loaded buffer ({self.size} transitions) from {filepath}")
    
    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        capacity: Optional[int] = None,
    ) -> "OfflineReplayBuffer":
        """
        Create a buffer from a saved file.
        
        Args:
            filepath: Path to the saved buffer file
            capacity: Buffer capacity (default: auto-determine from file)
        
        Returns:
            Loaded OfflineReplayBuffer
        """
        filepath = Path(filepath)
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        loaded_size = len(data["states"])
        
        if capacity is None:
            capacity = loaded_size
        
        buffer = cls(capacity=max(capacity, loaded_size))
        buffer.load(filepath)
        
        return buffer
    
    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return self.size
    
    def __repr__(self) -> str:
        return (
            f"OfflineReplayBuffer(size={self.size}, capacity={self.capacity}, "
            f"n_episodes={len(self.episode_returns)})"
        )


class EpisodeBuffer:
    """
    Buffer that stores complete episodes for analysis.
    
    Useful for computing episode-level statistics and visualizations.
    """
    
    def __init__(self):
        self.episodes: List[Dict[str, np.ndarray]] = []
    
    def add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a complete episode."""
        self.episodes.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "length": len(states),
            "return": float(np.sum(rewards)),
        })
    
    def get_returns(self) -> np.ndarray:
        """Get all episode returns."""
        return np.array([ep["return"] for ep in self.episodes])
    
    def get_lengths(self) -> np.ndarray:
        """Get all episode lengths."""
        return np.array([ep["length"] for ep in self.episodes])
    
    def to_replay_buffer(self, capacity: Optional[int] = None) -> OfflineReplayBuffer:
        """Convert to OfflineReplayBuffer."""
        total_transitions = sum(ep["length"] for ep in self.episodes)
        
        if capacity is None:
            capacity = total_transitions
        
        buffer = OfflineReplayBuffer(capacity=capacity)
        
        for ep in self.episodes:
            for i in range(ep["length"]):
                next_state = ep["states"][i + 1] if i < ep["length"] - 1 else ep["states"][i]
                buffer.add(
                    state=ep["states"][i],
                    action=ep["actions"][i],
                    reward=ep["rewards"][i],
                    next_state=next_state,
                    done=bool(ep["dones"][i]),
                )
        
        return buffer
    
    def __len__(self) -> int:
        return len(self.episodes)


if __name__ == "__main__":
    # Test the replay buffer
    logging.basicConfig(level=logging.INFO)
    
    # Create buffer
    buffer = OfflineReplayBuffer(capacity=10000)
    
    # Add some dummy data
    for episode in range(10):
        episode_length = np.random.randint(5, 20)
        for t in range(episode_length):
            state = np.random.randint(0, 716)
            action = np.random.randint(0, 25)
            reward = 1.0 if t == episode_length - 1 else 0.0
            next_state = np.random.randint(0, 716)
            done = (t == episode_length - 1)
            buffer.add(state, action, reward, next_state, done)
    
    print(buffer)
    print("\nStatistics:", buffer.compute_statistics())
    
    # Test sampling
    batch = buffer.sample(batch_size=32)
    print("\nSampled batch shapes:")
    print(f"  States: {batch[0].shape}")
    print(f"  Actions: {batch[1].shape}")
    print(f"  Rewards: {batch[2].shape}")
    
    # Test save/load
    buffer.save("test_buffer.pkl")
    buffer2 = OfflineReplayBuffer.from_file("test_buffer.pkl")
    print(f"\nLoaded buffer: {buffer2}")
    
    # Clean up
    import os
    os.remove("test_buffer.pkl")
    
    print("\nReplay buffer tests passed!")
