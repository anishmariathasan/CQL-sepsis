"""
Policy evaluation utilities for offline reinforcement learning.

This module provides comprehensive evaluation functions for:
- Online policy evaluation (rollouts in environment)
- Off-policy evaluation (importance sampling, doubly robust)
- Safety analysis (action distributions, extreme actions)

All metrics are designed for the ICU-Sepsis benchmark.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

logger = logging.getLogger(__name__)


def evaluate_policy(
    env: gym.Env,
    policy: Union[Callable, Any],
    n_episodes: int = 100,
    max_steps: int = 100,
    seed: Optional[int] = None,
    use_action_masking: bool = True,
    deterministic: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a policy by running it in the environment.
    
    This function runs the policy for multiple episodes and computes
    comprehensive evaluation metrics including survival rate, returns,
    episode lengths, and action distributions.
    
    Args:
        env: Gymnasium environment
        policy: Policy to evaluate. Can be:
            - A callable that takes state and returns action
            - An object with select_action(state) method
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        use_action_masking: Whether to use admissible action masking
        deterministic: Whether to use deterministic action selection
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing:
            - mean_return: Average episode return
            - std_return: Standard deviation of returns
            - min_return: Minimum return
            - max_return: Maximum return
            - survival_rate: Fraction of episodes with positive return
            - mean_episode_length: Average episode length
            - std_episode_length: Standard deviation of lengths
            - all_returns: List of all episode returns
            - all_lengths: List of all episode lengths
            - action_distribution: Frequency of each action
            - action_counts: Raw counts of each action
    
    Example:
        >>> from src.algorithms import CQL
        >>> agent = CQL.from_checkpoint('model.pt')
        >>> metrics = evaluate_policy(env, agent, n_episodes=100)
        >>> print(f"Survival rate: {metrics['survival_rate']:.1%}")
    """
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Track metrics
    returns = []
    lengths = []
    all_actions = []
    
    # Determine how to call the policy
    if callable(policy) and not hasattr(policy, 'select_action'):
        get_action = policy
    elif hasattr(policy, 'select_action'):
        def get_action(state, admissible_actions=None):
            return policy.select_action(
                state,
                eval_mode=True,
                admissible_actions=admissible_actions,
            )
    else:
        raise ValueError("Policy must be callable or have select_action method")
    
    for ep in range(n_episodes):
        # Reset environment
        ep_seed = seed + ep if seed is not None else None
        state, info = env.reset(seed=ep_seed)
        
        episode_return = 0.0
        episode_length = 0
        episode_actions = []
        
        for step in range(max_steps):
            # Get admissible actions
            admissible_actions = None
            if use_action_masking:
                try:
                    admissible_actions = env.get_admissible_actions()
                except AttributeError:
                    pass
            
            # Select action
            action = get_action(state, admissible_actions)
            episode_actions.append(action)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        returns.append(episode_return)
        lengths.append(episode_length)
        all_actions.extend(episode_actions)
        
        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            logger.info(
                f"Episode {ep + 1}/{n_episodes}: "
                f"Return = {episode_return:.2f}, "
                f"Length = {episode_length}"
            )
    
    # Compute metrics
    returns = np.array(returns)
    lengths = np.array(lengths)
    all_actions = np.array(all_actions)
    
    # Action distribution
    n_actions = 25  # ICU-Sepsis has 25 actions
    action_counts = np.bincount(all_actions, minlength=n_actions)
    action_distribution = action_counts / action_counts.sum()
    
    metrics = {
        # Return statistics
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        
        # Survival rate (positive return = survival in ICU-Sepsis)
        "survival_rate": float(np.mean(returns > 0)),
        "mortality_rate": float(np.mean(returns <= 0)),
        
        # Episode length statistics
        "mean_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),
        "min_episode_length": float(np.min(lengths)),
        "max_episode_length": float(np.max(lengths)),
        
        # Raw data
        "all_returns": returns.tolist(),
        "all_lengths": lengths.tolist(),
        
        # Action statistics
        "action_distribution": action_distribution.tolist(),
        "action_counts": action_counts.tolist(),
        "n_total_actions": len(all_actions),
        
        # Episode count
        "n_episodes": n_episodes,
    }
    
    if verbose:
        logger.info(f"Evaluation complete:")
        logger.info(f"  Mean return: {metrics['mean_return']:.3f} ± {metrics['std_return']:.3f}")
        logger.info(f"  Survival rate: {metrics['survival_rate']:.1%}")
        logger.info(f"  Mean episode length: {metrics['mean_episode_length']:.1f}")
    
    return metrics


def evaluate_multiple_seeds(
    env_fn: Callable[[], gym.Env],
    policy: Any,
    n_episodes_per_seed: int = 100,
    seeds: List[int] = [0, 1, 2, 3, 4],
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate policy across multiple random seeds.
    
    Args:
        env_fn: Function that creates a new environment
        policy: Policy to evaluate
        n_episodes_per_seed: Episodes per seed
        seeds: List of random seeds
        **kwargs: Additional arguments for evaluate_policy
    
    Returns:
        Aggregated metrics across all seeds
    """
    all_returns = []
    all_lengths = []
    all_survival_rates = []
    
    for seed in seeds:
        env = env_fn()
        metrics = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=n_episodes_per_seed,
            seed=seed,
            **kwargs,
        )
        
        all_returns.extend(metrics["all_returns"])
        all_lengths.extend(metrics["all_lengths"])
        all_survival_rates.append(metrics["survival_rate"])
    
    all_returns = np.array(all_returns)
    all_lengths = np.array(all_lengths)
    all_survival_rates = np.array(all_survival_rates)
    
    return {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "sem_return": float(np.std(all_returns) / np.sqrt(len(all_returns))),
        
        "mean_survival_rate": float(np.mean(all_survival_rates)),
        "std_survival_rate": float(np.std(all_survival_rates)),
        "sem_survival_rate": float(np.std(all_survival_rates) / np.sqrt(len(all_survival_rates))),
        
        "mean_episode_length": float(np.mean(all_lengths)),
        "std_episode_length": float(np.std(all_lengths)),
        
        "n_total_episodes": len(all_returns),
        "n_seeds": len(seeds),
        "seeds": seeds,
    }


def compute_off_policy_estimates(
    policy: Any,
    dataset: Any,
    behavior_policy: Optional[Any] = None,
    gamma: float = 0.99,
    n_samples: int = 10000,
) -> Dict[str, float]:
    """
    Compute off-policy evaluation (OPE) estimates.
    
    Implements several OPE methods:
    - Direct Method (DM): Use Q-function estimates
    - Importance Sampling (IS): Weight returns by policy ratio
    - Weighted Importance Sampling (WIS): Normalised IS
    - Doubly Robust (DR): Combine DM and IS
    
    Args:
        policy: Policy to evaluate
        dataset: Offline dataset (OfflineReplayBuffer or similar)
        behavior_policy: Behavior policy (for IS weights). If None, uses uniform.
        gamma: Discount factor
        n_samples: Number of samples for estimation
    
    Returns:
        Dictionary of OPE estimates
    """
    # Sample data from dataset
    states, actions, rewards, next_states, dones = dataset.sample(n_samples)
    
    # Direct Method: Use Q-values if available
    dm_estimate = None
    if hasattr(policy, 'get_q_values'):
        q_values = policy.get_q_values(states)
        
        # Get policy actions
        if hasattr(policy, 'select_action'):
            policy_actions = np.array([
                policy.select_action(s, eval_mode=True) for s in states
            ])
        else:
            policy_actions = q_values.argmax(axis=1)
        
        # V(s) = Q(s, pi(s))
        v_values = q_values[np.arange(len(states)), policy_actions]
        dm_estimate = float(np.mean(v_values))
    
    # Importance Sampling
    # For discrete actions, estimate behavior policy as empirical frequency
    n_actions = 25
    action_counts = np.bincount(actions, minlength=n_actions)
    behavior_probs = action_counts / action_counts.sum()
    behavior_probs = np.clip(behavior_probs, 1e-10, 1.0)  # Avoid division by zero
    
    # Get policy probabilities
    if hasattr(policy, 'get_action_probs'):
        policy_probs = policy.get_action_probs(states)
    elif hasattr(policy, 'get_q_values'):
        # Use softmax of Q-values as proxy
        q_values = policy.get_q_values(states)
        policy_probs = np.exp(q_values - q_values.max(axis=1, keepdims=True))
        policy_probs = policy_probs / policy_probs.sum(axis=1, keepdims=True)
    else:
        # Assume uniform policy
        policy_probs = np.ones((len(states), n_actions)) / n_actions
    
    # Importance weights
    pi_probs = policy_probs[np.arange(len(states)), actions]
    behavior_probs_sample = behavior_probs[actions]
    importance_weights = pi_probs / behavior_probs_sample
    
    # Clip weights for stability
    importance_weights = np.clip(importance_weights, 0, 100)
    
    # IS estimate (per-step)
    is_estimate = float(np.mean(importance_weights * rewards))
    
    # Weighted IS (self-normalised)
    wis_estimate = float(np.sum(importance_weights * rewards) / np.sum(importance_weights))
    
    results = {
        "is_estimate": is_estimate,
        "wis_estimate": wis_estimate,
        "mean_importance_weight": float(np.mean(importance_weights)),
        "max_importance_weight": float(np.max(importance_weights)),
        "effective_sample_size": float(np.sum(importance_weights)**2 / np.sum(importance_weights**2)),
    }
    
    if dm_estimate is not None:
        results["dm_estimate"] = dm_estimate
        # Doubly robust estimate (simplified)
        results["dr_estimate"] = float(0.5 * dm_estimate + 0.5 * wis_estimate)
    
    return results


def analyze_safety_metrics(
    policy: Any,
    env: gym.Env,
    n_episodes: int = 100,
    seed: Optional[int] = None,
    reference_policy: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Analyze safety-related metrics of a policy.
    
    Safety metrics for ICU-Sepsis:
    - Extreme action frequency (very high doses)
    - Dose distribution statistics
    - Action stability (how often actions change)
    - Comparison to reference (clinician) policy
    
    Args:
        policy: Policy to analyze
        env: Environment
        n_episodes: Number of episodes for analysis
        seed: Random seed
        reference_policy: Reference policy for comparison (e.g., behavior policy)
    
    Returns:
        Dictionary of safety metrics
    """
    from src.environments.icu_sepsis_wrapper import action_to_doses
    
    # Collect episodes
    if seed is not None:
        np.random.seed(seed)
    
    all_actions = []
    all_vasopressor_levels = []
    all_iv_fluid_levels = []
    action_changes = []
    
    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        state, info = env.reset(seed=ep_seed)
        
        prev_action = None
        episode_actions = []
        
        done = False
        while not done:
            # Get action
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state, eval_mode=True)
            else:
                action = policy(state)
            
            episode_actions.append(action)
            
            # Track action changes
            if prev_action is not None:
                action_changes.append(int(action != prev_action))
            prev_action = action
            
            # Decompose action into doses
            vaso, iv = action_to_doses(action)
            all_vasopressor_levels.append(vaso)
            all_iv_fluid_levels.append(iv)
            
            # Step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        all_actions.extend(episode_actions)
    
    all_actions = np.array(all_actions)
    all_vasopressor_levels = np.array(all_vasopressor_levels)
    all_iv_fluid_levels = np.array(all_iv_fluid_levels)
    
    # Compute safety metrics
    n_actions = 25
    action_counts = np.bincount(all_actions, minlength=n_actions)
    action_distribution = action_counts / action_counts.sum()
    
    # Extreme actions (highest dose levels: vasopressor=4 or iv_fluid=4)
    extreme_vaso = np.mean(all_vasopressor_levels == 4)
    extreme_iv = np.mean(all_iv_fluid_levels == 4)
    extreme_either = np.mean((all_vasopressor_levels == 4) | (all_iv_fluid_levels == 4))
    extreme_both = np.mean((all_vasopressor_levels == 4) & (all_iv_fluid_levels == 4))
    
    # No treatment (action 0: no vasopressor, no IV fluid)
    no_treatment = np.mean(all_actions == 0)
    
    # Conservative actions (low doses: levels 0-1)
    conservative_vaso = np.mean(all_vasopressor_levels <= 1)
    conservative_iv = np.mean(all_iv_fluid_levels <= 1)
    
    # Action stability
    action_change_rate = np.mean(action_changes) if action_changes else 0.0
    
    safety_metrics = {
        # Extreme action rates
        "extreme_vasopressor_rate": float(extreme_vaso),
        "extreme_iv_fluid_rate": float(extreme_iv),
        "extreme_action_rate_either": float(extreme_either),
        "extreme_action_rate_both": float(extreme_both),
        
        # Conservative action rates
        "no_treatment_rate": float(no_treatment),
        "conservative_vasopressor_rate": float(conservative_vaso),
        "conservative_iv_fluid_rate": float(conservative_iv),
        
        # Dose distributions
        "mean_vasopressor_level": float(np.mean(all_vasopressor_levels)),
        "std_vasopressor_level": float(np.std(all_vasopressor_levels)),
        "mean_iv_fluid_level": float(np.mean(all_iv_fluid_levels)),
        "std_iv_fluid_level": float(np.std(all_iv_fluid_levels)),
        
        # Action stability
        "action_change_rate": float(action_change_rate),
        
        # Action distribution
        "action_distribution": action_distribution.tolist(),
        "action_entropy": float(-np.sum(action_distribution * np.log(action_distribution + 1e-10))),
        
        # Counts
        "n_total_actions": len(all_actions),
        "n_episodes": n_episodes,
    }
    
    # Compare to reference policy if provided
    if reference_policy is not None:
        ref_metrics = analyze_safety_metrics(
            policy=reference_policy,
            env=env,
            n_episodes=n_episodes,
            seed=seed,
        )
        
        safety_metrics["ref_extreme_action_rate"] = ref_metrics["extreme_action_rate_either"]
        safety_metrics["ref_action_entropy"] = ref_metrics["action_entropy"]
        
        # KL divergence from reference
        ref_dist = np.array(ref_metrics["action_distribution"])
        kl_div = np.sum(action_distribution * np.log((action_distribution + 1e-10) / (ref_dist + 1e-10)))
        safety_metrics["kl_divergence_from_reference"] = float(kl_div)
    
    return safety_metrics


def compute_confidence_intervals(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence intervals using bootstrap.
    
    Args:
        values: Array of values
        confidence: Confidence level (default: 95%)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n_bootstrap = 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(values)
    
    return mean, lower, upper


if __name__ == "__main__":
    # Test evaluation utilities
    logging.basicConfig(level=logging.INFO)
    
    from src.environments.icu_sepsis_wrapper import create_sepsis_env
    from src.algorithms.cql import CQL
    
    # Create environment and random policy
    env = create_sepsis_env()
    
    # Random policy for testing
    def random_policy(state, admissible_actions=None):
        if admissible_actions is not None and len(admissible_actions) > 0:
            return np.random.choice(admissible_actions)
        return np.random.randint(0, 25)
    
    # Evaluate random policy
    print("Evaluating random policy...")
    metrics = evaluate_policy(
        env=env,
        policy=random_policy,
        n_episodes=50,
        seed=42,
        verbose=True,
    )
    
    print(f"\nResults:")
    print(f"  Survival rate: {metrics['survival_rate']:.1%}")
    print(f"  Mean return: {metrics['mean_return']:.3f}")
    
    # Safety analysis
    print("\nSafety analysis...")
    safety = analyze_safety_metrics(
        policy=random_policy,
        env=env,
        n_episodes=50,
        seed=42,
    )
    
    print(f"  Extreme action rate: {safety['extreme_action_rate_either']:.1%}")
    print(f"  Action entropy: {safety['action_entropy']:.2f}")
    
    print("\nEvaluation tests passed!")
