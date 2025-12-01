"""
Visualization Interface and Implementations
==========================================
Abstract interface for creating visualizations of model results with concrete
implementations for regression analysis plots.

WHAT IS VISUALIZATION?
----------------------
Creating charts and graphs to understand model performance visually.
Makes it easy to spot patterns, errors, and areas for improvement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer(ABC):
    """
    Abstract interface for visualization.

    Any visualizer must implement:
    1. create_plots() - Generate visualizations
    2. save() - Save plots to file
    """

    @abstractmethod
    def create_plots(self, results: Dict, **kwargs) -> Any:
        """
        Create visualizations from results.

        Args:
            results: Dictionary containing data to visualize
            **kwargs: Additional visualization parameters

        Returns:
            Figure object or other visualization artifact
        """
        pass

    @abstractmethod
    def save(self, filepath: str, **kwargs):
        """
        Save visualization to file.

        Args:
            filepath: Path to save the visualization
            **kwargs: Additional save parameters (dpi, format, etc.)
        """
        pass


class RegressionVisualizer(Visualizer):
    """
    Comprehensive visualizations for regression problems.

    Creates a multi-panel figure showing:
    1. Predicted vs Actual scatter plot
    2. Residual plot
    3. Error distribution
    4. Feature importance
    5. Target distribution
    6. Classification accuracy (for band gap types)
    7. Experiment summary
    """

    def __init__(self, figsize: tuple = (16, 12)):
        """
        Initialize regression visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None

    def create_plots(self, results: Dict, verbose: bool = True, **kwargs) -> plt.Figure:
        """
        Create comprehensive regression visualization.

        Args:
            results: Dictionary with keys:
                - y_test: True test values
                - y_pred_test: Predicted test values
                - y_train: True training values (optional)
                - y_pred_train: Predicted training values (optional)
                - metrics: Dict with mae, rmse, r2
                - feature_importance: Array of feature importances
                - feature_names: List of feature names
                - dataset_size: Total dataset size
                - training_time: Training time in seconds
                - cv_mae: Cross-validation MAE
                - cv_std: Cross-validation std
            verbose: Print progress messages
            **kwargs: Additional parameters

        Returns:
            matplotlib Figure object
        """
        if verbose:
            print("\n[6/6] Creating visualizations...")
        # Extract data
        y_test = results['y_test']
        y_pred_test = results['y_pred_test']
        metrics = results['metrics']
        feature_importance = results.get('feature_importance', None)
        feature_names = results.get('feature_names', None)

        # Create figure with subplots
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Predicted vs Actual
        self._plot_predicted_vs_actual(gs, y_test, y_pred_test, metrics)

        # 2. Residuals
        self._plot_residuals(gs, y_test, y_pred_test)

        # 3. Error Distribution
        self._plot_error_distribution(gs, y_test, y_pred_test, metrics)

        # 4. Feature Importance (if available)
        if feature_importance is not None:
            self._plot_feature_importance(gs, feature_importance, feature_names)

        # 5. Band Gap Distribution
        self._plot_target_distribution(gs, y_test, y_pred_test)

        # 6. Material Type Accuracy (band gap specific)
        self._plot_classification_accuracy(gs, y_test, y_pred_test)

        # 7. Experiment Summary
        self._plot_summary(gs, results)

        # Overall title
        plt.suptitle('Materials Project: Band Gap Prediction Results',
                     fontsize=14, fontweight='bold')

        return self.fig

    def _plot_predicted_vs_actual(self, gs, y_test, y_pred_test, metrics):
        """Plot predicted vs actual values."""
        ax = self.fig.add_subplot(gs[0, 0])
        ax.scatter(y_test, y_pred_test, alpha=0.4, s=20, c='steelblue', edgecolors='none')
        ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, alpha=0.8)
        ax.set_xlabel('Actual Band Gap (eV)', fontsize=11)
        ax.set_ylabel('Predicted Band Gap (eV)', fontsize=11)
        ax.set_title(f'Test Set: Predicted vs Actual\nMAE={metrics["test_mae"]:.3f} eV, R²={metrics["test_r2"]:.3f}',
                     fontsize=11)
        ax.grid(True, alpha=0.3)

    def _plot_residuals(self, gs, y_test, y_pred_test):
        """Plot residuals (errors)."""
        ax = self.fig.add_subplot(gs[0, 1])
        residuals = y_test - y_pred_test
        ax.scatter(y_pred_test, residuals, alpha=0.4, s=20, c='green', edgecolors='none')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Band Gap (eV)', fontsize=11)
        ax.set_ylabel('Residual (eV)', fontsize=11)
        ax.set_title('Residual Plot', fontsize=11)
        ax.grid(True, alpha=0.3)

    def _plot_error_distribution(self, gs, y_test, y_pred_test, metrics):
        """Plot distribution of prediction errors."""
        ax = self.fig.add_subplot(gs[0, 2])
        residuals = y_test - y_pred_test
        abs_errors = np.abs(residuals)
        ax.hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(x=metrics['test_mae'], color='red', linestyle='--', lw=2,
                   label=f'MAE={metrics["test_mae"]:.3f}')
        ax.set_xlabel('Absolute Error (eV)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Error Distribution', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_feature_importance(self, gs, importances, feature_names):
        """Plot top N most important features."""
        ax = self.fig.add_subplot(gs[1, :])
        n_features = min(20, len(importances))
        indices = np.argsort(importances)[-n_features:]

        # Clean up feature names
        if feature_names:
            top_features = [feature_names[i].replace('MagpieData ', '').replace('mean ', '')
                           for i in indices]
        else:
            top_features = [f'Feature {i}' for i in indices]

        ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title(f'Top {n_features} Most Important Features', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_target_distribution(self, gs, y_test, y_pred_test):
        """Plot distribution of actual vs predicted band gaps."""
        ax = self.fig.add_subplot(gs[2, 0])
        ax.hist(y_test, bins=50, alpha=0.5, label='Actual', color='blue', edgecolor='black')
        ax.hist(y_pred_test, bins=50, alpha=0.5, label='Predicted', color='red', edgecolor='black')
        ax.axvline(x=0, color='green', linestyle='--', lw=1, alpha=0.5, label='Conductors')
        ax.axvline(x=4, color='orange', linestyle='--', lw=1, alpha=0.5, label='Insulators')
        ax.set_xlabel('Band Gap (eV)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Band Gap Distribution', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_classification_accuracy(self, gs, y_test, y_pred_test):
        """Plot accuracy for different material types (conductor, semiconductor, insulator)."""
        ax = self.fig.add_subplot(gs[2, 1])

        # Define categories
        categories = ['Conductors\n(gap=0)', 'Semiconductors\n(0-4 eV)', 'Insulators\n(>4 eV)']

        # Count actual samples in each category
        actual_counts = [
            (y_test == 0).sum(),
            ((y_test > 0) & (y_test <= 4)).sum(),
            (y_test > 4).sum()
        ]

        # Count correct predictions in each category
        pred_correct = [
            ((y_test == 0) & (np.abs(y_pred_test) < 0.5)).sum(),
            ((y_test > 0) & (y_test <= 4) & (y_pred_test > 0) & (y_pred_test <= 4)).sum(),
            ((y_test > 4) & (y_pred_test > 4)).sum()
        ]

        # Calculate accuracy
        accuracy = [pred_correct[i]/actual_counts[i]*100 if actual_counts[i] > 0 else 0
                    for i in range(3)]

        # Plot
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, accuracy, color=['gold', 'skyblue', 'lightcoral'],
                      alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel('Classification Accuracy (%)', fontsize=11)
        ax.set_title('Accuracy by Material Type', fontsize=11)
        ax.set_ylim(0, 105)

        # Add text labels
        for i, (bar, acc) in enumerate(zip(bars, accuracy)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%\n(n={actual_counts[i]})', ha='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_summary(self, gs, results):
        """Plot text summary of experiment."""
        ax = self.fig.add_subplot(gs[2, 2])
        ax.axis('off')

        metrics = results['metrics']
        dataset_size = results.get('dataset_size', 'N/A')
        n_features = results.get('n_features', 'N/A')
        training_time = results.get('training_time', 0)
        cv_mae = results.get('cv_mae', 0)
        cv_std = results.get('cv_std', 0)

        summary_text = f"""
EXPERIMENT SUMMARY
{'='*30}

Dataset: {dataset_size} materials
Features: {n_features}
Training time: {training_time:.1f}s

Test Performance:
  MAE:  {metrics['test_mae']:.3f} eV
  RMSE: {metrics['test_rmse']:.3f} eV
  R²:   {metrics['test_r2']:.3f}

5-Fold CV:
  MAE: {cv_mae:.3f} ± {cv_std:.3f} eV

Top 3 Features:
"""
        # Add top 3 features if available
        if 'feature_importance' in results and 'feature_names' in results:
            importances = results['feature_importance']
            feature_names = results['feature_names']
            top_3_idx = np.argsort(importances)[-3:][::-1]
            for idx in top_3_idx:
                fname = feature_names[idx].replace('MagpieData ', '').replace('mean ', '')
                summary_text += f"  • {fname[:25]}\n"

        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def save(self, filepath: str, dpi: int = 300, **kwargs):
        """
        Save the figure to file.

        Args:
            filepath: Path to save file
            dpi: Resolution (dots per inch)
            **kwargs: Additional matplotlib savefig parameters
        """
        if self.fig is None:
            raise ValueError("No figure to save! Call create_plots() first.")

        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"✓ Saved visualization: {filepath}")

    def show(self):
        """Display the figure."""
        if self.fig is None:
            raise ValueError("No figure to show! Call create_plots() first.")

        plt.show()


# ============================================================================
# EXAMPLE: How to add custom visualizations
# ============================================================================
#
# To add new visualization types, implement the Visualizer interface:
#
# class ClassificationVisualizer(Visualizer):
#     """Visualizer for classification problems."""
#
#     def __init__(self):
#         self.fig = None
#
#     def create_plots(self, results: Dict, **kwargs):
#         # Create confusion matrix
#         # Create ROC curves
#         # Create precision-recall curves
#         pass
#
#     def save(self, filepath: str, **kwargs):
#         self.fig.savefig(filepath, **kwargs)
#
# Usage:
#     viz = ClassificationVisualizer()
#     fig = viz.create_plots(results)
#     viz.save('classification_results.png')
# ============================================================================
