"""
Experiment Tracking Interface
=============================
Centralized experiment tracking to avoid repetitive logging checks.

WHAT IS EXPERIMENT TRACKING?
----------------------------
Recording your experiments (metrics, parameters, results) so you can:
- Compare different runs
- Track what worked and what didn't
- Share results with team members
- Reproduce experiments later
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import wandb


class ExperimentTracker(ABC):
    """
    Abstract interface for experiment tracking.

    Any tracker must implement:
    1. init() - Initialize tracking session
    2. log() - Log metrics/data
    3. log_artifact() - Log files (models, plots, etc.)
    4. finish() - Close tracking session
    """

    @abstractmethod
    def init(self, project: str, name: str, config: Dict):
        """Initialize tracking session."""
        pass

    @abstractmethod
    def log(self, data: Dict):
        """Log metrics or data."""
        pass

    @abstractmethod
    def log_artifact(self, filepath: str):
        """Log a file artifact."""
        pass

    @abstractmethod
    def finish(self):
        """Finish tracking session."""
        pass


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self, enabled: bool = True):
        """
        Initialize W&B tracker.

        Args:
            enabled: Whether tracking is enabled (if False, becomes no-op)
        """
        self.enabled = enabled
        self.run = None

    def init(self, project: str, name: str, config: Dict):
        """
        Initialize W&B run.

        Args:
            project: Project name (e.g., "materials-bandgap")
            name: Run name (e.g., "xgboost-20250101")
            config: Configuration dictionary
        """
        if not self.enabled:
            print("⚠ Experiment tracking disabled")
            return

        self.run = wandb.init(
            project=project,
            name=name,
            config=config
        )
        print(f"✓ W&B initialized - Track at: {wandb.run.url}")

    def log(self, data: Dict):
        """
        Log metrics or data to W&B.

        Args:
            data: Dictionary of metrics to log
        """
        if not self.enabled:
            return

        wandb.log(data)

    def log_artifact(self, filepath: str):
        """
        Log a file artifact to W&B.

        Args:
            filepath: Path to file to save
        """
        if not self.enabled:
            return

        # Special handling for image files
        if filepath.endswith(('.png', '.jpg', '.jpeg')):
            wandb.log({filepath.replace('.png', '').replace('.jpg', '').replace('.jpeg', ''): wandb.Image(filepath)})

        wandb.save(filepath)

    def finish(self):
        """Finish W&B run."""
        if not self.enabled:
            return

        wandb.finish()
        print("✓ Results uploaded to W&B")


class NoOpTracker(ExperimentTracker):
    """
    No-operation tracker that does nothing.

    Useful for:
    - Running experiments without tracking
    - Testing code without W&B dependency
    - Quick local experiments
    """

    def init(self, project: str, name: str, config: Dict):
        """Initialize (no-op)."""
        print("⚠ Running without experiment tracking")

    def log(self, data: Dict):
        """Log (no-op)."""
        pass

    def log_artifact(self, filepath: str):
        """Log artifact (no-op)."""
        pass

    def finish(self):
        """Finish (no-op)."""
        pass


def create_tracker(use_wandb: bool = True) -> ExperimentTracker:
    """
    Factory function to create appropriate tracker.

    Args:
        use_wandb: Whether to use W&B tracking

    Returns:
        ExperimentTracker instance (either WandB or NoOp)

    Example:
        tracker = create_tracker(use_wandb=True)
        tracker.init(project="my-project", name="run-1", config={...})
        tracker.log({"accuracy": 0.95})
        tracker.finish()
    """
    if use_wandb:
        return WandBTracker(enabled=True)
    else:
        return NoOpTracker()


# ============================================================================
# EXAMPLE: How to add other tracking platforms
# ============================================================================
#
# To add MLflow, TensorBoard, or custom tracking:
#
# class MLflowTracker(ExperimentTracker):
#     """MLflow experiment tracker."""
#
#     def __init__(self, enabled: bool = True):
#         self.enabled = enabled
#
#     def init(self, project: str, name: str, config: Dict):
#         if not self.enabled:
#             return
#         import mlflow
#         mlflow.start_run(run_name=name)
#         mlflow.log_params(config)
#
#     def log(self, data: Dict):
#         if not self.enabled:
#             return
#         import mlflow
#         mlflow.log_metrics(data)
#
#     def log_artifact(self, filepath: str):
#         if not self.enabled:
#             return
#         import mlflow
#         mlflow.log_artifact(filepath)
#
#     def finish(self):
#         if not self.enabled:
#             return
#         import mlflow
#         mlflow.end_run()
#
# Usage:
#     tracker = MLflowTracker(enabled=True)
#     tracker.init("project", "run-1", config)
# ============================================================================
