"""
Data collection utilities for generating offline datasets.

This module provides functions for collecting data from the ICU-Sepsis
environment using various behavior policies.

Functions:
    - collect_episodes: Collect episodes using a given policy
    - collect_dataset: Collect a full dataset for offline RL
    - create_behavior_policy: Create different behavior policies
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from src.data.replay_buffer import OfflineReplayBuffer, EpisodeBuffer

logger = logging.getLogger(__name__)


def create_behavior_policy(
    policy_type: str,
    env: gym.Env,
    epsilon: float = 0.1,
    **kwargs,
) -> Callable[[Any, Optional[List[int]]], int]:
    """
    Create a behavior policy for data collection.
    
    Args:
        policy_type: Type of policy:
            - "random": Uniformly random actions
            - "behavior": Default environment behavior policy
            - "epsilon_greedy": Epsilon-greedy with given base policy
            - "softmax": Softmax policy with temperature
        env: Environment instance
        epsilon: Exploration rate for epsilon-greedy
        **kwargs: Additional arguments for specific policy types
    
    Returns:
        Policy function that takes (state, admissible_actions) and returns action
    """
    if policy_type == "random":
        def random_policy(state, admissible_actions=None):
            if admissible_actions is not None and len(admissible_actions) > 0:
                return np.random.choice(admissible_actions)
            return env.action_space.sample()
        return random_policy
    
    elif policy_type == "behavior":
        # Use the environment's built-in behavior policy if available
        def behavior_policy(state, admissible_actions=None):
            # ICU-Sepsis has an underlying behavior policy
            # We approximate it with uniform random over admissible actions
            if admissible_actions is not None and len(admissible_actions) > 0:
                return np.random.choice(admissible_actions)
            return env.action_space.sample()
        return behavior_policy
    
    elif policy_type == "epsilon_greedy":
        base_policy = kwargs.get("base_policy", None)
        
        def epsilon_greedy_policy(state, admissible_actions=None):
            if np.random.random() < epsilon:
                if admissible_actions is not None and len(admissible_actions) > 0:
                    return np.random.choice(admissible_actions)
                return env.action_space.sample()
            else:
                if base_policy is not None:
                    return base_policy(state, admissible_actions)
                return env.action_space.sample()
        return epsilon_greedy_policy
    
    elif policy_type == "softmax":
        temperature = kwargs.get("temperature", 1.0)
        q_values = kwargs.get("q_values", None)
        
        def softmax_policy(state, admissible_actions=None):
            if q_values is not None and state in q_values:
                q = q_values[state]
                if admissible_actions is not None:
                    q = q[admissible_actions]
                probs = np.exp(q / temperature)
                probs = probs / probs.sum()
                if admissible_actions is not None:
                    return np.random.choice(admissible_actions, p=probs)
                return np.random.choice(len(q), p=probs)
            else:
                if admissible_actions is not None and len(admissible_actions) > 0:
                    return np.random.choice(admissible_actions)
                return env.action_space.sample()
        return softmax_policy
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def collect_episode(
    env: gym.Env,
    policy: Callable[[Any, Optional[List[int]]], int],
    max_steps: int = 1000,
    seed: Optional[int] = None,
    use_action_masking: bool = True,
) -> Tuple[List[int], List[int], List[float], List[bool]]:
    """
    Collect a single episode from the environment.
    
    Args:
        env: Gymnasium environment
        policy: Policy function (state, admissible_actions) -> action
        max_steps: Maximum steps per episode
        seed: Random seed for episode
        use_action_masking: Whether to use admissible action masking
    
    Returns:
        Tuple of (states, actions, rewards, dones) lists
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # Reset environment
    if seed is not None:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()
    
    for step in range(max_steps):
        # Get admissible actions if available
        admissible_actions = None
        if use_action_masking:
            try:
                admissible_actions = env.get_admissible_actions()
            except AttributeError:
                pass
        
        # Select action
        action = policy(state, admissible_actions)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        if done:
            break
        
        state = next_state
    
    return states, actions, rewards, dones


def collect_episodes(
    env: gym.Env,
    policy: Callable[[Any, Optional[List[int]]], int],
    n_episodes: int,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    use_action_masking: bool = True,
    verbose: bool = True,
) -> EpisodeBuffer:
    """
    Collect multiple episodes from the environment.
    
    Args:
        env: Gymnasium environment
        policy: Policy function
        n_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        seed: Base random seed
        use_action_masking: Whether to use admissible action masking
        verbose: Whether to print progress
    
    Returns:
        EpisodeBuffer containing all collected episodes
    """
    episode_buffer = EpisodeBuffer()
    
    returns = []
    lengths = []
    
    for ep in range(n_episodes):
        # Set seed for reproducibility
        ep_seed = seed + ep if seed is not None else None
        
        # Collect episode
        states, actions, rewards, dones = collect_episode(
            env=env,
            policy=policy,
            max_steps=max_steps,
            seed=ep_seed,
            use_action_masking=use_action_masking,
        )
        
        # Store episode
        episode_buffer.add_episode(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
        )
        
        # Track statistics
        ep_return = sum(rewards)
        ep_length = len(states)
        returns.append(ep_return)
        lengths.append(ep_length)
        
        # Progress logging
        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            avg_return = np.mean(returns[-100:])
            avg_length = np.mean(lengths[-100:])
            logger.info(
                f"Episode {ep + 1}/{n_episodes}: "
                f"Avg Return (last 100) = {avg_return:.3f}, "
                f"Avg Length = {avg_length:.1f}"
            )
    
    if verbose:
        logger.info(f"Collection complete:")
        logger.info(f"  Total episodes: {len(episode_buffer)}")
        logger.info(f"  Avg return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
        logger.info(f"  Survival rate: {np.mean(np.array(returns) > 0):.1%}")
    
    return episode_buffer


def collect_dataset(
    env: gym.Env,
    policy_type: str = "behavior",
    n_episodes: int = 5000,
    max_steps: int = 1000,
    seed: Optional[int] = 42,
    buffer_capacity: Optional[int] = None,
    use_action_masking: bool = True,
    verbose: bool = True,
    **policy_kwargs,
) -> OfflineReplayBuffer:
    """
    Collect a complete offline dataset from the environment.
    
    This is the main function for generating offline datasets for CQL training.
    
    Args:
        env: Gymnasium environment
        policy_type: Type of behavior policy ("random", "behavior", "epsilon_greedy")
        n_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        buffer_capacity: Replay buffer capacity (default: auto)
        use_action_masking: Whether to use admissible action masking
        verbose: Whether to print progress
        **policy_kwargs: Additional arguments for policy creation
    
    Returns:
        OfflineReplayBuffer containing the collected dataset
    
    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('Sepsis-v0')
        >>> dataset = collect_dataset(env, n_episodes=5000, seed=42)
        >>> dataset.save('data/offline_datasets/behavior_50k.pkl')
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Create behavior policy
    policy = create_behavior_policy(
        policy_type=policy_type,
        env=env,
        **policy_kwargs,
    )
    
    if verbose:
        logger.info(f"Collecting {n_episodes} episodes with {policy_type} policy...")
    
    # Collect episodes
    episode_buffer = collect_episodes(
        env=env,
        policy=policy,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        use_action_masking=use_action_masking,
        verbose=verbose,
    )
    
    # Convert to replay buffer
    if buffer_capacity is None:
        # Auto-determine capacity based on collected data
        total_transitions = sum(ep["length"] for ep in episode_buffer.episodes)
        buffer_capacity = int(total_transitions * 1.1)  # 10% buffer
    
    replay_buffer = episode_buffer.to_replay_buffer(capacity=buffer_capacity)
    
    if verbose:
        stats = replay_buffer.compute_statistics()
        logger.info(f"Dataset statistics:")
        logger.info(f"  Transitions: {stats['size']}")
        logger.info(f"  Episodes: {stats.get('n_episodes', 'N/A')}")
        logger.info(f"  Avg return: {stats.get('avg_return', 'N/A'):.3f}")
        logger.info(f"  Survival rate: {stats.get('survival_rate', 'N/A'):.1%}")
    
    return replay_buffer


def load_or_collect_dataset(
    filepath: str,
    env: gym.Env,
    n_episodes: int = 5000,
    policy_type: str = "behavior",
    seed: int = 42,
    force_collect: bool = False,
    **kwargs,
) -> OfflineReplayBuffer:
    """
    Load dataset from file or collect if not available.
    
    Args:
        filepath: Path to dataset file
        env: Environment for collection if needed
        n_episodes: Number of episodes to collect
        policy_type: Behavior policy type
        seed: Random seed
        force_collect: Force collection even if file exists
        **kwargs: Additional arguments for collect_dataset
    
    Returns:
        OfflineReplayBuffer with dataset
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    
    if filepath.exists() and not force_collect:
        logger.info(f"Loading existing dataset from {filepath}")
        return OfflineReplayBuffer.from_file(filepath)
    else:
        logger.info(f"Collecting new dataset...")
        buffer = collect_dataset(
            env=env,
            policy_type=policy_type,
            n_episodes=n_episodes,
            seed=seed,
            **kwargs,
        )
        
        # Save for future use
        filepath.parent.mkdir(parents=True, exist_ok=True)
        buffer.save(filepath)
        
        return buffer


if __name__ == "__main__":
    # Test data collection
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple mock environment for testing
    class MockSepsisEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'sample': lambda: np.random.randint(0, 25), 'n': 25})()
            self.observation_space = type('obj', (object,), {'n': 716})()
            self.state = 0
            self.step_count = 0
        
        def reset(self, seed=None):
            if seed is not None:
                np.random.seed(seed)
            self.state = np.random.randint(0, 716)
            self.step_count = 0
            return self.state, {}
        
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randint(0, 716)
            
            # Episode ends with 10% probability or after 20 steps
            done = np.random.random() < 0.1 or self.step_count >= 20
            
            # Reward: 1 for survival at end, 0 otherwise
            reward = 1.0 if done and np.random.random() > 0.3 else 0.0
            
            self.state = next_state
            return next_state, reward, done, False, {}
        
        def get_admissible_actions(self):
            # Return subset of actions as admissible
            return list(range(0, 25, 2))  # Even actions only
    
    # Test with mock environment
    env = MockSepsisEnv()
    
    # Collect small dataset
    dataset = collect_dataset(
        env=env,
        policy_type="behavior",
        n_episodes=100,
        seed=42,
        verbose=True,
    )
    
    print(f"\nCollected dataset: {dataset}")
    print(f"Statistics: {dataset.compute_statistics()}")
    
    print("\nData collection tests passed!")
