#!/usr/bin/env python
"""
Train baseline algorithms (Behavior Cloning, DQN) for comparison with CQL.

This script trains baseline algorithms on the same offline dataset used for CQL,
enabling fair comparison of different offline RL approaches.

Usage:
    python scripts/04_train_baselines.py --dataset data/offline_datasets/behavior_policy.pkl
    
    # Train specific algorithms
    python scripts/04_train_baselines.py --algorithms bc dqn
    
    # Quick test
    python scripts/04_train_baselines.py --n_iterations 1000

"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.algorithms.bc import BehaviorCloning
from src.algorithms.dqn import DQN
from src.data.replay_buffer import OfflineReplayBuffer
from src.environments.icu_sepsis_wrapper import create_sepsis_env
from src.utils.evaluation import evaluate_policy
from src.utils.logging import setup_logging, create_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train baseline algorithms for comparison"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bc_baseline.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/offline_datasets/behavior_policy.pkl",
        help="Path to offline dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baselines",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["bc", "dqn"],
        choices=["bc", "dqn", "random"],
        help="Algorithms to train",
    )
    
    # Training parameters
    parser.add_argument("--n_iterations", type=int, default=50000)
    parser.add_argument("--eval_frequency", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    
    # Evaluation
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()


def train_behavior_cloning(
    buffer: OfflineReplayBuffer,
    eval_env,
    output_dir: Path,
    config: dict,
    device: str = "auto",
) -> dict:
    """Train Behavior Cloning baseline."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("Training Behavior Cloning")
    logger.info("=" * 60)
    
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create BC agent
    agent = BehaviorCloning(
        state_dim=716,
        action_dim=25,
        hidden_dim=config.get("hidden_dim", 256),
        lr=config.get("lr", 3e-4),
        device=device,
    )
    
    n_iterations = config.get("n_iterations", 50000)
    eval_frequency = config.get("eval_frequency", 5000)
    batch_size = config.get("batch_size", 256)
    n_eval_episodes = config.get("n_eval_episodes", 100)
    
    # Training
    training_metrics = {"iterations": [], "loss": [], "accuracy": []}
    eval_metrics = {"iterations": [], "survival_rate": [], "mean_return": []}
    best_survival_rate = 0.0
    
    pbar = tqdm(range(n_iterations), desc="BC Training")
    
    for iteration in pbar:
        batch = buffer.sample(batch_size)
        metrics = agent.update(batch)
        
        if iteration % 100 == 0:
            training_metrics["iterations"].append(iteration)
            training_metrics["loss"].append(metrics["bc_loss"])
            training_metrics["accuracy"].append(metrics["accuracy"])
            
            pbar.set_postfix({
                "loss": f"{metrics['bc_loss']:.3f}",
                "acc": f"{metrics['accuracy']:.1%}",
            })
        
        if (iteration + 1) % eval_frequency == 0:
            eval_results = evaluate_policy(
                env=eval_env,
                policy=agent,
                n_episodes=n_eval_episodes,
                seed=seed,
            )
            
            eval_metrics["iterations"].append(iteration + 1)
            eval_metrics["survival_rate"].append(eval_results["survival_rate"])
            eval_metrics["mean_return"].append(eval_results["mean_return"])
            
            logger.info(f"Iteration {iteration + 1}: "
                       f"Survival = {eval_results['survival_rate']:.1%}")
            
            if eval_results["survival_rate"] > best_survival_rate:
                best_survival_rate = eval_results["survival_rate"]
                agent.save(str(output_dir / "bc_best.pt"))
    
    # Final eval
    final_results = evaluate_policy(
        env=eval_env,
        policy=agent,
        n_episodes=n_eval_episodes * 2,
        seed=seed,
    )
    
    agent.save(str(output_dir / "bc_final.pt"))
    
    results = {
        "algorithm": "bc",
        "training_metrics": training_metrics,
        "eval_metrics": eval_metrics,
        "final_results": {
            "survival_rate": final_results["survival_rate"],
            "mean_return": final_results["mean_return"],
            "std_return": final_results["std_return"],
        },
        "best_survival_rate": best_survival_rate,
    }
    
    logger.info(f"BC Best Survival Rate: {best_survival_rate:.1%}")
    
    return results


def train_dqn(
    buffer: OfflineReplayBuffer,
    eval_env,
    output_dir: Path,
    config: dict,
    device: str = "auto",
) -> dict:
    """Train DQN baseline (offline, without CQL penalty)."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("Training DQN (Offline)")
    logger.info("=" * 60)
    
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create DQN agent
    agent = DQN(
        state_dim=716,
        action_dim=25,
        hidden_dim=config.get("hidden_dim", 256),
        lr=config.get("lr", 3e-4),
        gamma=config.get("gamma", 0.99),
        tau=config.get("tau", 0.005),
        device=device,
    )
    
    n_iterations = config.get("n_iterations", 50000)
    eval_frequency = config.get("eval_frequency", 5000)
    batch_size = config.get("batch_size", 256)
    n_eval_episodes = config.get("n_eval_episodes", 100)
    
    # Training
    training_metrics = {"iterations": [], "loss": [], "q_values_mean": []}
    eval_metrics = {"iterations": [], "survival_rate": [], "mean_return": []}
    best_survival_rate = 0.0
    
    pbar = tqdm(range(n_iterations), desc="DQN Training")
    
    for iteration in pbar:
        batch = buffer.sample(batch_size)
        metrics = agent.update(batch)
        
        if iteration % 100 == 0:
            training_metrics["iterations"].append(iteration)
            training_metrics["loss"].append(metrics["td_loss"])
            training_metrics["q_values_mean"].append(metrics["q_values_mean"])
            
            pbar.set_postfix({
                "loss": f"{metrics['td_loss']:.3f}",
                "Q": f"{metrics['q_values_mean']:.2f}",
            })
        
        if (iteration + 1) % eval_frequency == 0:
            eval_results = evaluate_policy(
                env=eval_env,
                policy=agent,
                n_episodes=n_eval_episodes,
                seed=seed,
            )
            
            eval_metrics["iterations"].append(iteration + 1)
            eval_metrics["survival_rate"].append(eval_results["survival_rate"])
            eval_metrics["mean_return"].append(eval_results["mean_return"])
            
            logger.info(f"Iteration {iteration + 1}: "
                       f"Survival = {eval_results['survival_rate']:.1%}")
            
            if eval_results["survival_rate"] > best_survival_rate:
                best_survival_rate = eval_results["survival_rate"]
                agent.save(str(output_dir / "dqn_best.pt"))
    
    # Final eval
    final_results = evaluate_policy(
        env=eval_env,
        policy=agent,
        n_episodes=n_eval_episodes * 2,
        seed=seed,
    )
    
    agent.save(str(output_dir / "dqn_final.pt"))
    
    results = {
        "algorithm": "dqn",
        "training_metrics": training_metrics,
        "eval_metrics": eval_metrics,
        "final_results": {
            "survival_rate": final_results["survival_rate"],
            "mean_return": final_results["mean_return"],
            "std_return": final_results["std_return"],
        },
        "best_survival_rate": best_survival_rate,
    }
    
    logger.info(f"DQN Best Survival Rate: {best_survival_rate:.1%}")
    
    return results


def evaluate_random_policy(
    eval_env,
    config: dict,
) -> dict:
    """Evaluate random policy baseline."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Random Policy")
    logger.info("=" * 60)
    
    seed = config.get("seed", 42)
    n_eval_episodes = config.get("n_eval_episodes", 100) * 2
    
    def random_policy(state, admissible_actions=None):
        if admissible_actions is not None and len(admissible_actions) > 0:
            return np.random.choice(admissible_actions)
        return np.random.randint(0, 25)
    
    eval_results = evaluate_policy(
        env=eval_env,
        policy=random_policy,
        n_episodes=n_eval_episodes,
        seed=seed,
    )
    
    results = {
        "algorithm": "random",
        "final_results": {
            "survival_rate": eval_results["survival_rate"],
            "mean_return": eval_results["mean_return"],
            "std_return": eval_results["std_return"],
        },
    }
    
    logger.info(f"Random Policy Survival Rate: {eval_results['survival_rate']:.1%}")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_dir="results/logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Baseline Training for Sepsis Treatment")
    logger.info("=" * 60)
    
    # Config
    config = {
        "n_iterations": args.n_iterations,
        "eval_frequency": args.eval_frequency,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_eval_episodes": args.n_eval_episodes,
        "seed": args.seed,
        "gamma": 0.99,
        "tau": 0.005,
    }
    
    # Create output directory
    output_dir = Path(project_root) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    dataset_path = Path(project_root) / args.dataset
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please run: python scripts/02_collect_offline_data.py first")
        sys.exit(1)
    
    buffer = OfflineReplayBuffer.from_file(dataset_path)
    logger.info(f"Loaded {len(buffer):,} transitions")
    
    # Create evaluation environment
    eval_env = create_sepsis_env(use_action_masking=True)
    
    # Train each algorithm
    all_results = {}
    
    if "bc" in args.algorithms:
        results = train_behavior_cloning(
            buffer=buffer,
            eval_env=eval_env,
            output_dir=output_dir,
            config=config,
            device=args.device,
        )
        all_results["bc"] = results
    
    if "dqn" in args.algorithms:
        results = train_dqn(
            buffer=buffer,
            eval_env=eval_env,
            output_dir=output_dir,
            config=config,
            device=args.device,
        )
        all_results["dqn"] = results
    
    if "random" in args.algorithms:
        results = evaluate_random_policy(
            eval_env=eval_env,
            config=config,
        )
        all_results["random"] = results
    
    # Save all results
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete - Summary")
    logger.info("=" * 60)
    
    for algo, results in all_results.items():
        survival = results["final_results"]["survival_rate"]
        logger.info(f"  {algo.upper()}: Survival Rate = {survival:.1%}")
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
