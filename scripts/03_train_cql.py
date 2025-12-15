#!/usr/bin/env python
"""
Train Conservative Q-Learning (CQL) on offline sepsis treatment data.

This is the main training script for CQL. It loads an offline dataset,
trains a CQL agent, and periodically evaluates performance.

Usage:
    python scripts/03_train_cql.py --config configs/cql_default.yaml --dataset data/offline_datasets/behavior_policy.pkl
    
    # With custom alpha
    python scripts/03_train_cql.py --alpha 1.0 --dataset data/offline_datasets/behavior_policy.pkl
    
    # Quick test
    python scripts/03_train_cql.py --n_iterations 1000 --eval_frequency 500

Options:
    --config: Path to YAML configuration file
    --dataset: Path to offline dataset
    --output_dir: Directory for outputs
    --alpha: CQL conservatism coefficient
    --seed: Random seed
    --n_iterations: Number of training iterations
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

from src.algorithms.cql import CQL
from src.data.replay_buffer import OfflineReplayBuffer
from src.environments.icu_sepsis_wrapper import create_sepsis_env
from src.utils.evaluation import evaluate_policy
from src.utils.logging import setup_logging, create_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CQL on offline sepsis treatment data"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cql_default.yaml",
        help="Path to YAML configuration file",
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
        default=None,
        help="Directory for outputs (default: results/cql_<timestamp>)",
    )
    
    # Hyperparameters (override config)
    parser.add_argument("--alpha", type=float, default=None, help="CQL alpha coefficient")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--tau", type=float, default=None, help="Target update rate")
    
    # Training
    parser.add_argument("--n_iterations", type=int, default=None, help="Training iterations")
    parser.add_argument("--eval_frequency", type=int, default=None, help="Evaluation frequency")
    parser.add_argument("--checkpoint_frequency", type=int, default=None, help="Checkpoint frequency")
    parser.add_argument("--log_frequency", type=int, default=None, help="Log frequency")
    
    # Evaluation
    parser.add_argument("--n_eval_episodes", type=int, default=None, help="Evaluation episodes")
    
    # Misc
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="cql-sepsis", help="W&B project")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(project_root) / config_path
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    else:
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}


def merge_configs(base_config: dict, args: argparse.Namespace) -> dict:
    """Merge command line arguments with config file."""
    config = base_config.copy()
    
    # Ensure nested dicts exist
    config.setdefault("algorithm", {})
    config.setdefault("training", {})
    config.setdefault("evaluation", {})
    config.setdefault("logging", {})
    
    # Override with command line arguments
    if args.alpha is not None:
        config["algorithm"]["alpha"] = args.alpha
    if args.lr is not None:
        config["algorithm"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["algorithm"]["batch_size"] = args.batch_size
    if args.hidden_dim is not None:
        config["algorithm"]["hidden_dim"] = args.hidden_dim
    if args.gamma is not None:
        config["algorithm"]["gamma"] = args.gamma
    if args.tau is not None:
        config["algorithm"]["tau"] = args.tau
    
    if args.n_iterations is not None:
        config["training"]["n_iterations"] = args.n_iterations
    if args.eval_frequency is not None:
        config["training"]["eval_frequency"] = args.eval_frequency
    if args.checkpoint_frequency is not None:
        config["training"]["checkpoint_frequency"] = args.checkpoint_frequency
    if args.log_frequency is not None:
        config["training"]["log_frequency"] = args.log_frequency
    
    if args.n_eval_episodes is not None:
        config["evaluation"]["n_episodes"] = args.n_eval_episodes
    
    if args.seed is not None:
        config["seed"] = args.seed
    
    if args.use_wandb:
        config["logging"]["use_wandb"] = True
        config["logging"]["wandb_project"] = args.wandb_project
    
    # Set defaults
    config["algorithm"].setdefault("alpha", 1.0)
    config["algorithm"].setdefault("learning_rate", 3e-4)
    config["algorithm"].setdefault("batch_size", 256)
    config["algorithm"].setdefault("hidden_dim", 256)
    config["algorithm"].setdefault("num_layers", 2)
    config["algorithm"].setdefault("gamma", 0.99)
    config["algorithm"].setdefault("tau", 0.005)
    config["algorithm"].setdefault("grad_clip", 1.0)
    config["algorithm"].setdefault("use_double_dqn", True)
    
    # Ensure numeric types (YAML can sometimes load as strings)
    config["algorithm"]["alpha"] = float(config["algorithm"]["alpha"])
    config["algorithm"]["learning_rate"] = float(config["algorithm"]["learning_rate"])
    config["algorithm"]["batch_size"] = int(config["algorithm"]["batch_size"])
    config["algorithm"]["hidden_dim"] = int(config["algorithm"]["hidden_dim"])
    config["algorithm"]["num_layers"] = int(config["algorithm"]["num_layers"])
    config["algorithm"]["gamma"] = float(config["algorithm"]["gamma"])
    config["algorithm"]["tau"] = float(config["algorithm"]["tau"])
    config["algorithm"]["grad_clip"] = float(config["algorithm"]["grad_clip"])
    
    config["training"].setdefault("n_iterations", 100000)
    config["training"].setdefault("eval_frequency", 5000)
    config["training"].setdefault("checkpoint_frequency", 10000)
    config["training"].setdefault("log_frequency", 100)
    
    config["evaluation"].setdefault("n_episodes", 100)
    
    config.setdefault("seed", 42)
    config["logging"].setdefault("use_wandb", False)
    config["logging"].setdefault("use_tensorboard", True)
    
    return config


def train_cql(
    config: dict,
    dataset_path: str,
    output_dir: str,
    device: str = "auto",
) -> dict:
    """
    Main CQL training loop.
    
    Args:
        config: Training configuration
        dataset_path: Path to offline dataset
        output_dir: Directory for outputs
        device: Device to use
    
    Returns:
        Dictionary of training results
    """
    logger = logging.getLogger(__name__)
    
    # Set seed
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset_path = Path(project_root) / dataset_path
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please run: python scripts/02_collect_offline_data.py first")
        sys.exit(1)
    
    buffer = OfflineReplayBuffer.from_file(dataset_path)
    logger.info(f"Loaded {len(buffer):,} transitions")
    
    # Create environment for evaluation
    logger.info("Creating evaluation environment...")
    eval_env = create_sepsis_env(use_action_masking=True)
    
    # Extract hyperparameters
    algo_config = config["algorithm"]
    train_config = config["training"]
    eval_config = config["evaluation"]
    
    # Create CQL agent
    logger.info("Creating CQL agent...")
    agent = CQL(
        state_dim=716,  # ICU-Sepsis state space
        action_dim=25,   # ICU-Sepsis action space
        hidden_dim=algo_config["hidden_dim"],
        num_layers=algo_config.get("num_layers", 2),
        lr=algo_config["learning_rate"],
        alpha=algo_config["alpha"],
        gamma=algo_config["gamma"],
        tau=algo_config["tau"],
        use_double_dqn=algo_config.get("use_double_dqn", True),
        grad_clip=algo_config.get("grad_clip", 1.0),
        device=device,
    )
    
    logger.info(f"  Alpha (conservatism): {algo_config['alpha']}")
    logger.info(f"  Learning rate: {algo_config['learning_rate']}")
    logger.info(f"  Batch size: {algo_config['batch_size']}")
    logger.info(f"  Hidden dim: {algo_config['hidden_dim']}")
    logger.info(f"  Device: {agent.device}")
    
    # Setup experiment logger
    exp_logger = create_logger(
        use_wandb=config["logging"].get("use_wandb", False),
        use_tensorboard=config["logging"].get("use_tensorboard", True),
        wandb_project=config["logging"].get("wandb_project", "cql-sepsis"),
        tensorboard_dir=str(output_dir / "logs"),
        run_name=f"cql_alpha_{algo_config['alpha']}_seed_{seed}",
        config=config,
    )
    
    # Training loop
    logger.info(f"\nStarting training for {train_config['n_iterations']:,} iterations...")
    
    n_iterations = train_config["n_iterations"]
    eval_frequency = train_config["eval_frequency"]
    checkpoint_frequency = train_config["checkpoint_frequency"]
    log_frequency = train_config["log_frequency"]
    batch_size = algo_config["batch_size"]
    
    # Track metrics
    training_metrics = {
        "iterations": [],
        "td_loss": [],
        "cql_penalty": [],
        "total_loss": [],
        "q_values_mean": [],
    }
    eval_metrics = {
        "iterations": [],
        "survival_rate": [],
        "mean_return": [],
    }
    
    best_survival_rate = 0.0
    
    # Progress bar
    pbar = tqdm(range(n_iterations), desc="Training")
    
    for iteration in pbar:
        # Sample batch and update
        batch = buffer.sample(batch_size)
        metrics = agent.update(batch)
        
        # Log training metrics
        if iteration % log_frequency == 0:
            training_metrics["iterations"].append(iteration)
            training_metrics["td_loss"].append(metrics["td_loss"])
            training_metrics["cql_penalty"].append(metrics["cql_penalty"])
            training_metrics["total_loss"].append(metrics["total_loss"])
            training_metrics["q_values_mean"].append(metrics["q_values_mean"])
            
            exp_logger.log_metrics({
                "train/td_loss": metrics["td_loss"],
                "train/cql_penalty": metrics["cql_penalty"],
                "train/total_loss": metrics["total_loss"],
                "train/q_values_mean": metrics["q_values_mean"],
                "train/q_values_std": metrics.get("q_values_std", 0),
            }, step=iteration)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['total_loss']:.3f}",
                "Q": f"{metrics['q_values_mean']:.2f}",
            })
        
        # Evaluate
        if (iteration + 1) % eval_frequency == 0:
            logger.info(f"\nEvaluating at iteration {iteration + 1}...")
            
            eval_results = evaluate_policy(
                env=eval_env,
                policy=agent,
                n_episodes=eval_config["n_episodes"],
                seed=seed,
                verbose=False,
            )
            
            survival_rate = eval_results["survival_rate"]
            mean_return = eval_results["mean_return"]
            
            eval_metrics["iterations"].append(iteration + 1)
            eval_metrics["survival_rate"].append(survival_rate)
            eval_metrics["mean_return"].append(mean_return)
            
            exp_logger.log_metrics({
                "eval/survival_rate": survival_rate,
                "eval/mean_return": mean_return,
                "eval/std_return": eval_results["std_return"],
                "eval/mean_episode_length": eval_results["mean_episode_length"],
            }, step=iteration + 1)
            
            logger.info(f"  Survival rate: {survival_rate:.1%}")
            logger.info(f"  Mean return: {mean_return:.3f}")
            
            # Save best model
            if survival_rate > best_survival_rate:
                best_survival_rate = survival_rate
                agent.save(str(output_dir / "checkpoints" / "best_model.pt"))
                logger.info(f"  New best model saved! (survival: {best_survival_rate:.1%})")
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_frequency == 0:
            checkpoint_path = output_dir / "checkpoints" / f"checkpoint_{iteration + 1}.pt"
            agent.save(str(checkpoint_path))
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    final_results = evaluate_policy(
        env=eval_env,
        policy=agent,
        n_episodes=eval_config["n_episodes"] * 2,  # More episodes for final eval
        seed=seed,
        verbose=True,
    )
    
    # Save final model
    agent.save(str(output_dir / "checkpoints" / "final_model.pt"))
    
    # Save results
    results = {
        "config": config,
        "training_metrics": training_metrics,
        "eval_metrics": eval_metrics,
        "final_results": {
            "survival_rate": final_results["survival_rate"],
            "mean_return": final_results["mean_return"],
            "std_return": final_results["std_return"],
            "mean_episode_length": final_results["mean_episode_length"],
        },
        "best_survival_rate": best_survival_rate,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    exp_logger.finish()
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best survival rate: {best_survival_rate:.1%}")
    logger.info(f"Final survival rate: {final_results['survival_rate']:.1%}")
    logger.info(f"Results saved to: {output_dir}")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_dir="results/logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("CQL Training for Sepsis Treatment")
    logger.info("=" * 60)
    
    # Load and merge config
    base_config = load_config(args.config)
    config = merge_configs(base_config, args)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alpha = config["algorithm"]["alpha"]
        seed = config.get("seed", 42)
        output_dir = f"results/cql_alpha_{alpha}_seed_{seed}_{timestamp}"
    
    # Train
    results = train_cql(
        config=config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        device=args.device,
    )
    
    return results


if __name__ == "__main__":
    main()
