"""
Logging utilities for experiment tracking.

This module provides logging integrations with:
- Weights & Biases (WandB)
- TensorBoard
- Standard Python logging

Supports tracking of:
- Training metrics (loss, Q-values, etc.)
- Evaluation metrics (survival rate, returns)
- Hyperparameters
- Model checkpoints
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: str = "results/logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
    filename: str = "training.log",
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to also log to file
        filename: Log filename
    
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(log_path / filename)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured. Log directory: {log_dir}")
    
    return root_logger


class BaseLogger:
    """Base class for experiment loggers."""
    
    def __init__(self, name: str = "experiment"):
        self.name = name
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dictionary of metrics."""
        raise NotImplementedError
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        raise NotImplementedError
    
    def log_artifact(self, filepath: str, artifact_type: str = "model"):
        """Log an artifact (e.g., model checkpoint)."""
        raise NotImplementedError
    
    def finish(self):
        """Cleanup and finalise logging."""
        pass


class WandbLogger(BaseLogger):
    """
    Weights & Biases logger for experiment tracking.
    
    Features:
        - Automatic metric logging
        - Hyperparameter tracking
        - Model checkpoint saving
        - Visualization dashboards
    
    Args:
        project: W&B project name
        name: Run name
        config: Configuration dictionary
        entity: W&B entity (username or team)
        tags: List of tags for the run
        notes: Run notes
        mode: W&B mode ('online', 'offline', 'disabled')
    """
    
    def __init__(
        self,
        project: str = "cql-sepsis",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
    ):
        super().__init__(name or "cql-run")
        
        self.enabled = True
        
        try:
            import wandb
            self.wandb = wandb
            
            # Initialise W&B run
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                tags=tags,
                notes=notes,
                mode=mode,
                reinit=True,
            )
            
            logger.info(f"Initialised W&B logger: {project}/{name}")
            
        except ImportError:
            logger.warning("wandb not installed. W&B logging disabled.")
            self.enabled = False
            self.run = None
        except Exception as e:
            logger.warning(f"Failed to initialise W&B: {e}")
            self.enabled = False
            self.run = None
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to W&B."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.wandb.log(metrics, step=self.step)
        self.step += 1
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters to W&B config."""
        if not self.enabled:
            return
        
        self.wandb.config.update(hparams)
    
    def log_artifact(
        self,
        filepath: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
    ):
        """Log artifact (e.g., model checkpoint) to W&B."""
        if not self.enabled:
            return
        
        artifact_name = name or Path(filepath).stem
        artifact = self.wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(filepath)
        self.run.log_artifact(artifact)
    
    def log_figure(
        self,
        figure,
        name: str,
    ):
        """Log matplotlib figure to W&B."""
        if not self.enabled:
            return
        
        self.wandb.log({name: self.wandb.Image(figure)})
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run is not None:
            self.run.finish()


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger for experiment tracking.
    
    Features:
        - Scalar metric logging
        - Histogram logging
        - Hyperparameter logging
        - Graph visualization
    
    Args:
        log_dir: Directory for TensorBoard logs
        name: Run name
        comment: Comment to append to run name
    """
    
    def __init__(
        self,
        log_dir: str = "results/logs/tensorboard",
        name: Optional[str] = None,
        comment: str = "",
    ):
        super().__init__(name or "cql-run")
        
        self.enabled = True
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # Create log directory
            if name:
                log_dir = os.path.join(log_dir, name)
            
            self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
            self.log_dir = log_dir
            
            logger.info(f"Initialized TensorBoard logger: {log_dir}")
            
        except ImportError:
            logger.warning("tensorboard not installed. TensorBoard logging disabled.")
            self.enabled = False
            self.writer = None
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log scalar metrics to TensorBoard."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.step)
        
        self.step += 1
    
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log hyperparameters to TensorBoard."""
        if not self.enabled:
            return
        
        # Convert non-scalar values to strings
        hparams_cleaned = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                hparams_cleaned[k] = v
            else:
                hparams_cleaned[k] = str(v)
        
        if metrics is None:
            metrics = {}
        
        self.writer.add_hparams(hparams_cleaned, metrics)
    
    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: Optional[int] = None,
    ):
        """Log histogram to TensorBoard."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.step
        
        self.writer.add_histogram(name, values, step)
    
    def log_figure(
        self,
        name: str,
        figure,
        step: Optional[int] = None,
    ):
        """Log matplotlib figure to TensorBoard."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.step
        
        self.writer.add_figure(name, figure, step)
    
    def log_artifact(
        self,
        filepath: str,
        artifact_type: str = "model",
    ):
        """Log artifact path (TensorBoard doesn't store artifacts directly)."""
        if not self.enabled:
            return
        
        self.writer.add_text(
            f"artifacts/{artifact_type}",
            f"Saved to: {filepath}",
            self.step,
        )
    
    def finish(self):
        """Close TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()


class CompositeLogger(BaseLogger):
    """
    Composite logger that writes to multiple backends.
    
    Args:
        loggers: List of logger instances
    """
    
    def __init__(self, loggers: List[BaseLogger]):
        super().__init__("composite")
        self.loggers = loggers
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log to all loggers."""
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters to all loggers."""
        for logger in self.loggers:
            logger.log_hyperparameters(hparams)
    
    def log_artifact(self, filepath: str, artifact_type: str = "model"):
        """Log artifact to all loggers."""
        for logger in self.loggers:
            logger.log_artifact(filepath, artifact_type)
    
    def finish(self):
        """Finish all loggers."""
        for logger in self.loggers:
            logger.finish()


def create_logger(
    use_wandb: bool = False,
    use_tensorboard: bool = True,
    wandb_project: str = "cql-sepsis",
    wandb_entity: Optional[str] = None,
    tensorboard_dir: str = "results/logs/tensorboard",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BaseLogger:
    """
    Create appropriate logger based on configuration.
    
    Args:
        use_wandb: Whether to use W&B
        use_tensorboard: Whether to use TensorBoard
        wandb_project: W&B project name
        wandb_entity: W&B entity
        tensorboard_dir: TensorBoard log directory
        run_name: Name for the run
        config: Configuration dictionary
    
    Returns:
        Configured logger instance
    """
    loggers = []
    
    if use_wandb:
        loggers.append(WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config,
        ))
    
    if use_tensorboard:
        loggers.append(TensorBoardLogger(
            log_dir=tensorboard_dir,
            name=run_name,
        ))
    
    if len(loggers) == 0:
        # Return a dummy logger that does nothing
        return BaseLogger()
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)


if __name__ == "__main__":
    # Test logging utilities
    setup_logging(log_dir="test_logs")
    
    # Test TensorBoard logger
    tb_logger = TensorBoardLogger(log_dir="test_logs/tensorboard", name="test_run")
    
    for i in range(100):
        tb_logger.log_metrics({
            "loss": 1.0 / (i + 1),
            "accuracy": i / 100.0,
        }, step=i)
    
    tb_logger.finish()
    
    print("Logging tests passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_logs", ignore_errors=True)
