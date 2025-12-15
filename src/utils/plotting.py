"""
Publication-quality plotting utilities for CQL-Sepsis experiments.

This module provides visualisation functions for:
- Learning curves
- Performance comparisons
- Hyperparameter sensitivity analysis
- Action distributions
- Q-value visualisations
- Safety analysis

All plots follow publication standards with:
- Clean, minimal styling
- Appropriate font sizes
- Colour-blind friendly palettes
- Vector format support (PDF, SVG)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logging.warning("matplotlib/seaborn not installed. Plotting disabled.")

logger = logging.getLogger(__name__)

# Colour palette (colour-blind friendly)
COLORS = {
    "cql": "#2E86AB",        # Blue
    "bc": "#A23B72",         # Magenta
    "dqn": "#F18F01",        # Orange
    "random": "#C73E1D",     # Red
    "clinician": "#3B1F2B",  # Dark
    "optimal": "#95C623",    # Green
}

# Seaborn color palette
PALETTE = list(COLORS.values())


def set_publication_style():
    """
    Set matplotlib style for publication-quality figures.
    
    This configures:
    - Clean, minimal style
    - Appropriate font sizes
    - White background
    - Thin lines and axes
    """
    if not HAS_PLOTTING:
        return
    
    # Use seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # Additional matplotlib settings
    plt.rcParams.update({
        # Figure
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.autolayout": True,
        
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        
        # Lines
        "lines.linewidth": 2,
        "lines.markersize": 6,
        
        # Axes
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
        
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_learning_curves(
    results_dict: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
    xlabel: str = "Training Iterations",
    ylabel: str = "Average Return",
    show_std: bool = True,
    smooth_window: int = 10,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """
    Plot learning curves comparing multiple algorithms.
    
    Args:
        results_dict: Dictionary mapping algorithm names to their results.
            Each value should be a dict with 'iterations' and 'returns' keys,
            where 'returns' can be a list of values or list of lists (multiple seeds).
        save_path: Path to save the figure (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_std: Whether to show standard deviation bands
        smooth_window: Window size for smoothing
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> results = {
        ...     'CQL': {'iterations': list(range(100)), 'returns': [...]},
        ...     'BC': {'iterations': list(range(100)), 'returns': [...]},
        ... }
        >>> plot_learning_curves(results, save_path='learning_curves.png')
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available")
        return None
    
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    for algo_name, data in results_dict.items():
        iterations = np.array(data["iterations"])
        returns = np.array(data["returns"])
        
        # Handle multiple seeds
        if returns.ndim == 2:
            mean_returns = np.mean(returns, axis=0)
            std_returns = np.std(returns, axis=0)
        else:
            mean_returns = returns
            std_returns = None
        
        # Smooth the curve
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            mean_returns = np.convolve(mean_returns, kernel, mode='valid')
            if std_returns is not None:
                std_returns = np.convolve(std_returns, kernel, mode='valid')
            iterations = iterations[:len(mean_returns)]
        
        # Get color
        color = COLORS.get(algo_name.lower(), None)
        
        # Plot mean
        line, = ax.plot(
            iterations, mean_returns,
            label=algo_name,
            color=color,
            linewidth=2,
        )
        
        # Plot std band
        if show_std and std_returns is not None:
            ax.fill_between(
                iterations,
                mean_returns - std_returns,
                mean_returns + std_returns,
                alpha=0.2,
                color=line.get_color(),
            )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning curves to {save_path}")
    
    return fig


def plot_survival_rate_comparison(
    algorithms: List[str],
    survival_rates: List[float],
    std_errors: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Survival Rate Comparison",
    ylabel: str = "Survival Rate (%)",
    highlight_best: bool = True,
    reference_lines: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """
    Create bar chart comparing survival rates across algorithms.
    
    Args:
        algorithms: List of algorithm names
        survival_rates: List of survival rates (0-1 scale)
        std_errors: Optional standard errors for error bars
        save_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        highlight_best: Whether to highlight the best performer
        reference_lines: Dict of reference line names and values to add
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        return None
    
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to percentage
    survival_rates = np.array(survival_rates) * 100
    if std_errors is not None:
        std_errors = np.array(std_errors) * 100
    
    # Get colors
    colors = [COLORS.get(algo.lower(), PALETTE[i % len(PALETTE)]) 
              for i, algo in enumerate(algorithms)]
    
    # Highlight best
    if highlight_best:
        best_idx = np.argmax(survival_rates)
        edge_colors = ['gold' if i == best_idx else 'none' for i in range(len(algorithms))]
        edge_widths = [3 if i == best_idx else 0 for i in range(len(algorithms))]
    else:
        edge_colors = ['none'] * len(algorithms)
        edge_widths = [0] * len(algorithms)
    
    # Create bars
    x = np.arange(len(algorithms))
    bars = ax.bar(
        x, survival_rates,
        color=colors,
        edgecolor=edge_colors,
        linewidth=edge_widths,
        alpha=0.8,
    )
    
    # Add error bars
    if std_errors is not None:
        ax.errorbar(
            x, survival_rates, yerr=std_errors,
            fmt='none', color='black', capsize=5, capthick=2,
        )
    
    # Add value labels on bars
    for bar, rate in zip(bars, survival_rates):
        height = bar.get_height()
        ax.annotate(
            f'{rate:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )
    
    # Add reference lines
    if reference_lines:
        for name, value in reference_lines.items():
            ax.axhline(y=value * 100, color='gray', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=name)
    
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    
    if reference_lines:
        ax.legend(loc='upper right')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved survival rate comparison to {save_path}")
    
    return fig


def plot_alpha_sensitivity(
    alpha_values: List[float],
    survival_rates: List[float],
    std_errors: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "CQL Alpha Sensitivity",
    xlabel: str = "Alpha (Conservatism Coefficient)",
    ylabel: str = "Survival Rate (%)",
    highlight_best: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """
    Plot survival rate as a function of CQL alpha parameter.
    
    Args:
        alpha_values: List of alpha values tested
        survival_rates: Corresponding survival rates
        std_errors: Optional standard errors
        save_path: Path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        highlight_best: Whether to highlight optimal alpha
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        return None
    
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    alpha_values = np.array(alpha_values)
    survival_rates = np.array(survival_rates) * 100
    if std_errors is not None:
        std_errors = np.array(std_errors) * 100
    
    # Plot line with markers
    ax.plot(alpha_values, survival_rates, 'o-', color=COLORS['cql'], 
            linewidth=2, markersize=10)
    
    # Add error bands
    if std_errors is not None:
        ax.fill_between(
            alpha_values,
            survival_rates - std_errors,
            survival_rates + std_errors,
            alpha=0.2,
            color=COLORS['cql'],
        )
    
    # Highlight best
    if highlight_best:
        best_idx = np.argmax(survival_rates)
        ax.scatter(
            [alpha_values[best_idx]], [survival_rates[best_idx]],
            s=200, facecolors='none', edgecolors='gold', linewidths=3,
            zorder=5, label=f'Best α={alpha_values[best_idx]}'
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Use log scale if alpha spans orders of magnitude
    if max(alpha_values) / (min(alpha_values) + 1e-6) > 10:
        ax.set_xscale('log')
    
    if highlight_best:
        ax.legend(loc='lower right')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved alpha sensitivity plot to {save_path}")
    
    return fig


def plot_action_distributions(
    action_distributions: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Action Distribution Comparison",
    figsize: Tuple[int, int] = (14, 6),
) -> Optional[plt.Figure]:
    """
    Plot action distributions for multiple policies.
    
    The 25 actions are visualised as a 5x5 grid (vasopressor x IV fluid).
    
    Args:
        action_distributions: Dict mapping policy names to action distributions
            (arrays of shape (25,) representing probability of each action)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        return None
    
    set_publication_style()
    
    n_policies = len(action_distributions)
    fig, axes = plt.subplots(1, n_policies, figsize=figsize)
    
    if n_policies == 1:
        axes = [axes]
    
    for ax, (name, dist) in zip(axes, action_distributions.items()):
        dist = np.array(dist).reshape(5, 5)  # 5x5 grid
        
        im = ax.imshow(dist, cmap='Blues', aspect='auto', vmin=0)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability')
        
        # Labels
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['None', 'Low', 'Med', 'High', 'V.High'], fontsize=8)
        ax.set_yticklabels(['None', 'Low', 'Med', 'High', 'V.High'], fontsize=8)
        ax.set_xlabel('IV Fluid Level')
        ax.set_ylabel('Vasopressor Level')
        ax.set_title(name)
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f'{dist[i, j]:.2f}',
                              ha='center', va='center', fontsize=7,
                              color='white' if dist[i, j] > 0.1 else 'black')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved action distribution plot to {save_path}")
    
    return fig


def plot_q_value_heatmap(
    q_values: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Q-Value Heatmap",
    n_states_to_show: int = 50,
    figsize: Tuple[int, int] = (14, 8),
) -> Optional[plt.Figure]:
    """
    Create heatmap of Q-values across states and actions.
    
    Args:
        q_values: Q-values array of shape (n_states, n_actions)
        save_path: Path to save figure
        title: Plot title
        n_states_to_show: Number of states to display
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        return None
    
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select subset of states
    if q_values.shape[0] > n_states_to_show:
        # Select evenly spaced states
        indices = np.linspace(0, q_values.shape[0] - 1, n_states_to_show, dtype=int)
        q_values = q_values[indices]
    
    # Create heatmap
    im = ax.imshow(q_values, aspect='auto', cmap='RdYlGn')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Q-Value')
    
    # Labels
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(title)
    
    # Action labels
    n_actions = q_values.shape[1]
    if n_actions == 25:
        ax.set_xticks(range(0, 25, 5))
        ax.set_xticklabels([f'A{i}' for i in range(0, 25, 5)])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Q-value heatmap to {save_path}")
    
    return fig


def plot_data_efficiency(
    dataset_sizes: List[int],
    performance: List[float],
    std_errors: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Data Efficiency",
    xlabel: str = "Dataset Size (episodes)",
    ylabel: str = "Survival Rate (%)",
    baseline_performance: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """
    Plot performance as a function of dataset size.
    
    Args:
        dataset_sizes: List of dataset sizes
        performance: Corresponding performance values
        std_errors: Optional standard errors
        save_path: Path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        baseline_performance: Optional baseline to show
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        return None
    
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    dataset_sizes = np.array(dataset_sizes)
    performance = np.array(performance) * 100
    if std_errors is not None:
        std_errors = np.array(std_errors) * 100
    
    # Plot line
    ax.plot(dataset_sizes, performance, 'o-', color=COLORS['cql'],
            linewidth=2, markersize=10, label='CQL')
    
    # Add error bands
    if std_errors is not None:
        ax.fill_between(
            dataset_sizes,
            performance - std_errors,
            performance + std_errors,
            alpha=0.2,
            color=COLORS['cql'],
        )
    
    # Baseline
    if baseline_performance is not None:
        ax.axhline(y=baseline_performance * 100, color=COLORS['clinician'],
                  linestyle='--', linewidth=2, label='Clinician Baseline')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved data efficiency plot to {save_path}")
    
    return fig


def plot_safety_analysis(
    safety_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Safety Analysis",
    figsize: Tuple[int, int] = (12, 5),
) -> Optional[plt.Figure]:
    """
    Create safety analysis visualization.
    
    Shows extreme action rates and dose distributions for multiple policies.
    
    Args:
        safety_metrics: Dict mapping policy names to their safety metrics
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        return None
    
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    policies = list(safety_metrics.keys())
    n_policies = len(policies)
    x = np.arange(n_policies)
    
    # Plot 1: Extreme action rates
    ax1 = axes[0]
    extreme_rates = [safety_metrics[p]['extreme_action_rate_either'] * 100 
                     for p in policies]
    colors = [COLORS.get(p.lower(), PALETTE[i % len(PALETTE)]) 
              for i, p in enumerate(policies)]
    
    ax1.bar(x, extreme_rates, color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)
    ax1.set_ylabel('Extreme Action Rate (%)')
    ax1.set_title('Extreme Action Frequency')
    
    # Add value labels
    for i, rate in enumerate(extreme_rates):
        ax1.annotate(f'{rate:.1f}%', xy=(i, rate), xytext=(0, 3),
                    textcoords="offset points", ha='center', fontsize=9)
    
    # Plot 2: Mean dose levels
    ax2 = axes[1]
    width = 0.35
    
    vaso_means = [safety_metrics[p]['mean_vasopressor_level'] for p in policies]
    iv_means = [safety_metrics[p]['mean_iv_fluid_level'] for p in policies]
    
    ax2.bar(x - width/2, vaso_means, width, label='Vasopressor', color=COLORS['cql'])
    ax2.bar(x + width/2, iv_means, width, label='IV Fluid', color=COLORS['bc'])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies)
    ax2.set_ylabel('Mean Dose Level (0-4)')
    ax2.set_title('Average Treatment Intensity')
    ax2.legend()
    ax2.set_ylim(0, 4)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved safety analysis plot to {save_path}")
    
    return fig


def create_all_figures(
    results: Dict[str, Any],
    output_dir: str = "results/figures",
) -> Dict[str, plt.Figure]:
    """
    Create all publication figures from experimental results.
    
    Args:
        results: Dictionary containing all experimental results
        output_dir: Directory to save figures
    
    Returns:
        Dictionary of generated figures
    """
    if not HAS_PLOTTING:
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Learning curves
    if "learning_curves" in results:
        fig = plot_learning_curves(
            results["learning_curves"],
            save_path=output_dir / "learning_curves.png",
        )
        figures["learning_curves"] = fig
    
    # Survival rate comparison
    if "survival_rates" in results:
        fig = plot_survival_rate_comparison(
            algorithms=results["survival_rates"]["algorithms"],
            survival_rates=results["survival_rates"]["rates"],
            std_errors=results["survival_rates"].get("std_errors"),
            save_path=output_dir / "survival_rate_comparison.png",
        )
        figures["survival_rate_comparison"] = fig
    
    # Alpha sensitivity
    if "alpha_sweep" in results:
        fig = plot_alpha_sensitivity(
            alpha_values=results["alpha_sweep"]["alphas"],
            survival_rates=results["alpha_sweep"]["rates"],
            std_errors=results["alpha_sweep"].get("std_errors"),
            save_path=output_dir / "alpha_sensitivity.png",
        )
        figures["alpha_sensitivity"] = fig
    
    # Action distributions
    if "action_distributions" in results:
        fig = plot_action_distributions(
            results["action_distributions"],
            save_path=output_dir / "action_distributions.png",
        )
        figures["action_distributions"] = fig
    
    # Data efficiency
    if "data_efficiency" in results:
        fig = plot_data_efficiency(
            dataset_sizes=results["data_efficiency"]["sizes"],
            performance=results["data_efficiency"]["performance"],
            save_path=output_dir / "data_efficiency.png",
        )
        figures["data_efficiency"] = fig
    
    # Safety analysis
    if "safety_metrics" in results:
        fig = plot_safety_analysis(
            results["safety_metrics"],
            save_path=output_dir / "safety_analysis.png",
        )
        figures["safety_analysis"] = fig
    
    logger.info(f"Created {len(figures)} figures in {output_dir}")
    
    return figures


if __name__ == "__main__":
    # Test plotting utilities
    logging.basicConfig(level=logging.INFO)
    
    if not HAS_PLOTTING:
        print("Plotting libraries not available")
        exit(0)
    
    set_publication_style()
    
    # Test learning curves
    results = {
        "CQL": {
            "iterations": list(range(100)),
            "returns": np.random.randn(5, 100).cumsum(axis=1) / 10 + 0.8,
        },
        "BC": {
            "iterations": list(range(100)),
            "returns": np.random.randn(5, 100).cumsum(axis=1) / 15 + 0.7,
        },
    }
    
    fig = plot_learning_curves(results, save_path="test_learning_curves.png")
    print("Created learning curves plot")
    
    # Test survival rate comparison
    fig = plot_survival_rate_comparison(
        algorithms=["Random", "BC", "CQL", "Optimal"],
        survival_rates=[0.22, 0.77, 0.85, 0.90],
        std_errors=[0.02, 0.01, 0.01, 0.01],
        save_path="test_survival_comparison.png",
    )
    print("Created survival rate comparison plot")
    
    # Test alpha sensitivity
    fig = plot_alpha_sensitivity(
        alpha_values=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        survival_rates=[0.81, 0.83, 0.84, 0.85, 0.84, 0.82, 0.79],
        std_errors=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
        save_path="test_alpha_sensitivity.png",
    )
    print("Created alpha sensitivity plot")
    
    # Test action distribution
    fig = plot_action_distributions(
        {
            "Clinician": np.random.dirichlet(np.ones(25)),
            "CQL": np.random.dirichlet(np.ones(25) * 2),
        },
        save_path="test_action_dist.png",
    )
    print("Created action distribution plot")
    
    print("\nAll plotting tests passed!")
    
    # Cleanup
    import os
    for f in ["test_learning_curves.png", "test_survival_comparison.png",
              "test_alpha_sensitivity.png", "test_action_dist.png"]:
        if os.path.exists(f):
            os.remove(f)
