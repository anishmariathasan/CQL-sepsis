#!/usr/bin/env python
"""
Evaluate trained policies on the ICU-Sepsis environment.

This script loads trained model checkpoints and runs comprehensive evaluation,
including survival rate, safety metrics, and action distribution analysis.

Usage:
    python scripts/05_evaluate_policies.py --checkpoint results/cql_default/checkpoints/best_model.pt
    
    # Evaluate multiple models
    python scripts/05_evaluate_policies.py --checkpoints_dir results/
    
    # More evaluation episodes
    python scripts/05_evaluate_policies.py --checkpoint model.pt --n_episodes 500
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.algorithms.cql import CQL
from src.algorithms.bc import BehaviorCloning
from src.algorithms.dqn import DQN
from src.environments.icu_sepsis_wrapper import create_sepsis_env
from src.utils.evaluation import (
    evaluate_policy,
    analyze_safety_metrics,
    compute_confidence_intervals,
)
from src.utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies on ICU-Sepsis"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint to evaluate",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Directory containing multiple checkpoints to evaluate",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="cql",
        choices=["cql", "bc", "dqn"],
        help="Algorithm type for loading checkpoint",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--include_random",
        action="store_true",
        help="Include random policy baseline",
    )
    parser.add_argument(
        "--safety_analysis",
        action="store_true",
        default=True,
        help="Run safety analysis",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference",
    )
    
    return parser.parse_args()


def load_agent(checkpoint_path: str, algorithm: str, device: str = "auto"):
    """Load trained agent from checkpoint."""
    if algorithm == "cql":
        return CQL.from_checkpoint(checkpoint_path, device=device)
    elif algorithm == "bc":
        return BehaviorCloning.from_checkpoint(checkpoint_path, device=device)
    elif algorithm == "dqn":
        return DQN.from_checkpoint(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_single_checkpoint(
    checkpoint_path: str,
    algorithm: str,
    env,
    n_episodes: int,
    seeds: list,
    run_safety: bool = True,
    device: str = "auto",
) -> dict:
    """Evaluate a single checkpoint across multiple seeds."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\nEvaluating: {checkpoint_path}")
    logger.info(f"  Algorithm: {algorithm}")
    logger.info(f"  Episodes per seed: {n_episodes}")
    logger.info(f"  Seeds: {seeds}")
    
    # Load agent
    agent = load_agent(checkpoint_path, algorithm, device)
    
    # Evaluate across seeds
    all_returns = []
    all_survival_rates = []
    all_lengths = []
    
    for seed in seeds:
        results = evaluate_policy(
            env=env,
            policy=agent,
            n_episodes=n_episodes,
            seed=seed,
            verbose=False,
        )
        
        all_returns.extend(results["all_returns"])
        all_survival_rates.append(results["survival_rate"])
        all_lengths.extend(results["all_lengths"])
    
    all_returns = np.array(all_returns)
    all_survival_rates = np.array(all_survival_rates)
    all_lengths = np.array(all_lengths)
    
    # Compute statistics with confidence intervals
    mean_return, return_lower, return_upper = compute_confidence_intervals(all_returns)
    mean_survival = np.mean(all_survival_rates)
    std_survival = np.std(all_survival_rates)
    
    results = {
        "checkpoint": str(checkpoint_path),
        "algorithm": algorithm,
        "n_episodes_total": len(all_returns),
        "n_seeds": len(seeds),
        
        # Return metrics
        "mean_return": float(mean_return),
        "std_return": float(np.std(all_returns)),
        "return_ci_lower": float(return_lower),
        "return_ci_upper": float(return_upper),
        
        # Survival metrics
        "mean_survival_rate": float(mean_survival),
        "std_survival_rate": float(std_survival),
        "survival_rate_per_seed": all_survival_rates.tolist(),
        
        # Episode length
        "mean_episode_length": float(np.mean(all_lengths)),
        "std_episode_length": float(np.std(all_lengths)),
    }
    
    # Safety analysis
    if run_safety:
        logger.info("  Running safety analysis...")
        safety = analyze_safety_metrics(
            policy=agent,
            env=env,
            n_episodes=n_episodes,
            seed=seeds[0],
        )
        results["safety_metrics"] = safety
    
    logger.info(f"  Survival Rate: {mean_survival:.1%} ± {std_survival:.1%}")
    logger.info(f"  Mean Return: {mean_return:.3f} (95% CI: [{return_lower:.3f}, {return_upper:.3f}])")
    
    return results


def evaluate_random_baseline(
    env,
    n_episodes: int,
    seeds: list,
) -> dict:
    """Evaluate random policy baseline."""
    logger = logging.getLogger(__name__)
    logger.info("\nEvaluating: Random Policy")
    
    def random_policy(state, admissible_actions=None):
        if admissible_actions is not None and len(admissible_actions) > 0:
            return np.random.choice(admissible_actions)
        return np.random.randint(0, 25)
    
    all_returns = []
    all_survival_rates = []
    
    for seed in seeds:
        results = evaluate_policy(
            env=env,
            policy=random_policy,
            n_episodes=n_episodes,
            seed=seed,
        )
        all_returns.extend(results["all_returns"])
        all_survival_rates.append(results["survival_rate"])
    
    all_returns = np.array(all_returns)
    mean_survival = np.mean(all_survival_rates)
    std_survival = np.std(all_survival_rates)
    
    logger.info(f"  Survival Rate: {mean_survival:.1%} ± {std_survival:.1%}")
    
    return {
        "algorithm": "random",
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_survival_rate": float(mean_survival),
        "std_survival_rate": float(std_survival),
    }


def find_checkpoints(checkpoints_dir: str) -> list:
    """Find all checkpoint files in directory."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints = []
    
    # Look for .pt files
    for pattern in ["**/*best*.pt", "**/*final*.pt", "**/checkpoint_*.pt"]:
        checkpoints.extend(checkpoints_dir.glob(pattern))
    
    return list(set(checkpoints))


def main():
    """Main evaluation routine."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_dir="results/logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Policy Evaluation for ICU-Sepsis")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(project_root) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create evaluation environment
    env = create_sepsis_env(use_action_masking=True)
    
    # Verify we're using the real ICU-Sepsis environment
    env_type = type(env.env).__name__
    if "Mock" in env_type:
        logger.error("WARNING: Using MOCK environment, not real ICU-Sepsis!")
        raise RuntimeError("Mock environment detected. Install icu-sepsis package.")
    logger.info(f"Environment: {env_type} (real ICU-Sepsis)")
    
    all_results = {}
    
    # Evaluate single checkpoint
    if args.checkpoint:
        checkpoint_path = Path(project_root) / args.checkpoint
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        results = evaluate_single_checkpoint(
            checkpoint_path=str(checkpoint_path),
            algorithm=args.algorithm,
            env=env,
            n_episodes=args.n_episodes,
            seeds=args.seeds,
            run_safety=args.safety_analysis,
            device=args.device,
        )
        
        all_results[args.algorithm] = results
    
    # Evaluate multiple checkpoints
    elif args.checkpoints_dir:
        checkpoints_dir = Path(project_root) / args.checkpoints_dir
        checkpoints = find_checkpoints(checkpoints_dir)
        
        if not checkpoints:
            logger.warning(f"No checkpoints found in {checkpoints_dir}")
        
        for checkpoint in checkpoints:
            # Infer algorithm from path
            if "cql" in str(checkpoint).lower():
                algorithm = "cql"
            elif "bc" in str(checkpoint).lower():
                algorithm = "bc"
            elif "dqn" in str(checkpoint).lower():
                algorithm = "dqn"
            else:
                algorithm = args.algorithm
            
            try:
                results = evaluate_single_checkpoint(
                    checkpoint_path=str(checkpoint),
                    algorithm=algorithm,
                    env=env,
                    n_episodes=args.n_episodes,
                    seeds=args.seeds,
                    run_safety=args.safety_analysis,
                    device=args.device,
                )
                
                # Generate unique key using parent directory and filename
                # This prevents overwrites when multiple experiments have same checkpoint names
                parent_dir = checkpoint.parent.parent.name  # e.g., "cql_default"
                key = f"{parent_dir}_{checkpoint.stem}"
                
                # If key still exists, add a counter
                base_key = key
                counter = 1
                while key in all_results:
                    key = f"{base_key}_{counter}"
                    counter += 1
                
                all_results[key] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate {checkpoint}: {e}")
    
    # Random baseline
    if args.include_random:
        random_results = evaluate_random_baseline(
            env=env,
            n_episodes=args.n_episodes,
            seeds=args.seeds,
        )
        all_results["random"] = random_results
    
    # Save results
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    
    for name, results in all_results.items():
        survival = results.get("mean_survival_rate", results.get("survival_rate", 0))
        std = results.get("std_survival_rate", results.get("std_return", 0))
        logger.info(f"  {name}: {survival:.1%} ± {std:.1%}")
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
