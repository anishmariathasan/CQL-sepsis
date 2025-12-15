#!/usr/bin/env python
"""
Collect offline dataset from ICU-Sepsis environment using behavior policy.

This script generates offline datasets for training CQL and other offline RL
algorithms. The data is collected using a behavior policy that approximates
historical clinician decisions.

Usage:
    python scripts/02_collect_offline_data.py --n_episodes 5000 --save_path data/offline_datasets/behavior_50k.pkl
    
    # Quick test with fewer episodes
    python scripts/02_collect_offline_data.py --n_episodes 100 --save_path data/offline_datasets/test.pkl

Options:
    --n_episodes: Number of episodes to collect (default: 5000)
    --policy_type: Type of behavior policy (default: behavior)
    --save_path: Path to save the dataset
    --seed: Random seed for reproducibility
    --max_steps: Maximum steps per episode
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.data.replay_buffer import OfflineReplayBuffer
from src.data.data_collection import collect_dataset, create_behavior_policy
from src.environments.icu_sepsis_wrapper import create_sepsis_env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect offline dataset from ICU-Sepsis environment"
    )
    
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5000,
        help="Number of episodes to collect (default: 5000)",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="behavior",
        choices=["random", "behavior", "epsilon_greedy"],
        help="Type of behavior policy (default: behavior)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy policy (default: 0.1)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/offline_datasets/behavior_policy.pkl",
        help="Path to save the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100)",
    )
    parser.add_argument(
        "--use_action_masking",
        action="store_true",
        default=True,
        help="Use admissible action masking",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress during collection",
    )
    
    return parser.parse_args()


def main():
    """Main data collection routine."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Offline Data Collection for CQL-Sepsis")
    logger.info("=" * 60)
    
    # Create environment
    logger.info("Creating ICU-Sepsis environment...")
    env = create_sepsis_env(
        use_action_masking=args.use_action_masking,
        log_episodes=False,
    )
    
    # Verify we're using the real ICU-Sepsis environment
    env_type = type(env.env).__name__
    if "Mock" in env_type:
        logger.error("=" * 60)
        logger.error("WARNING: Using MOCK environment, not real ICU-Sepsis!")
        logger.error("Install icu-sepsis: pip install icu-sepsis")
        logger.error("=" * 60)
        raise RuntimeError("Mock environment detected. Install icu-sepsis package.")
    else:
        logger.info(f"  Environment: {env_type} (real ICU-Sepsis)")
    
    logger.info(f"  State space: {env.n_states} discrete states")
    logger.info(f"  Action space: {env.n_actions} discrete actions")
    
    # Create save directory
    save_path = Path(project_root) / args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect data
    logger.info(f"\nCollecting {args.n_episodes} episodes...")
    logger.info(f"  Policy type: {args.policy_type}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Max steps per episode: {args.max_steps}")
    
    buffer = collect_dataset(
        env=env,
        policy_type=args.policy_type,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        use_action_masking=args.use_action_masking,
        verbose=args.verbose,
        epsilon=args.epsilon,
    )
    
    # Save dataset
    logger.info(f"\nSaving dataset to {save_path}...")
    buffer.save(save_path)
    
    # Print statistics
    stats = buffer.compute_statistics()
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"  Total transitions: {stats['size']:,}")
    logger.info(f"  Total episodes: {stats.get('n_episodes', 'N/A')}")
    logger.info(f"  Average return: {stats.get('avg_return', 0):.3f} ± {stats.get('std_return', 0):.3f}")
    logger.info(f"  Survival rate: {stats.get('survival_rate', 0):.1%}")
    logger.info(f"  Average episode length: {stats.get('avg_episode_length', 0):.1f}")
    logger.info(f"  Unique states visited: {stats.get('n_unique_states', 0)}")
    logger.info(f"  Unique actions used: {stats.get('n_unique_actions', 0)}")
    
    # Action distribution
    if 'action_distribution' in stats:
        logger.info("\nAction distribution (top 5):")
        action_dist = stats['action_distribution']
        sorted_actions = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        for action, count in sorted_actions:
            pct = count / stats['size'] * 100
            logger.info(f"    Action {action}: {count:,} ({pct:.1f}%)")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Dataset saved to: {save_path}")
    logger.info("=" * 60)
    
    return buffer


if __name__ == "__main__":
    main()
