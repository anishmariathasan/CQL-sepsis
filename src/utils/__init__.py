"""
Utility functions for CQL-Sepsis project.

This package provides:
- Logging utilities (WandB, TensorBoard integration)
- Evaluation utilities (policy evaluation, off-policy estimation)
- Plotting utilities (publication-quality figures)
"""

from src.utils.logging import (
    setup_logging,
    WandbLogger,
    TensorBoardLogger,
)
from src.utils.evaluation import (
    evaluate_policy,
    compute_off_policy_estimates,
    analyze_safety_metrics,
)
from src.utils.plotting import (
    plot_learning_curves,
    plot_survival_rate_comparison,
    plot_alpha_sensitivity,
    plot_action_distributions,
    plot_q_value_heatmap,
    plot_data_efficiency,
    set_publication_style,
)

__all__ = [
    # Logging
    "setup_logging",
    "WandbLogger",
    "TensorBoardLogger",
    # Evaluation
    "evaluate_policy",
    "compute_off_policy_estimates",
    "analyze_safety_metrics",
    # Plotting
    "plot_learning_curves",
    "plot_survival_rate_comparison",
    "plot_alpha_sensitivity",
    "plot_action_distributions",
    "plot_q_value_heatmap",
    "plot_data_efficiency",
    "set_publication_style",
]
