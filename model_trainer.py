"""
Model Training Interface and Implementations
===========================================
Abstract interface for training machine learning models with concrete
implementations for various regression models.

WHAT IS MODEL TRAINING?
-----------------------
Teaching a computer to predict band gaps by showing it examples of
materials with known band gaps. The model learns patterns like:
"When atomic mass is high and electronegativity is low, band gap is usually 0"
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# Suppress XGBoost device mismatch warning (we handle it properly via DMatrix)
warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')


class ModelTrainer(ABC):
    """
    Abstract interface for model training.

    Any model trainer must implement:
    1. train() - Train the model on data
    2. predict() - Make predictions
    3. evaluate() - Calculate performance metrics
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray = None, y_test: np.ndarray = None,
              verbose: bool = True) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Optional test features for validation
            y_test: Optional test targets for validation
            verbose: Print progress

        Returns:
            Dictionary with training statistics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted values
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True target values

        Returns:
            Dictionary with metrics (MAE, RMSE, R²)
        """
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """Get the underlying model object."""
        pass


class XGBoostTrainer(ModelTrainer):
    """
    XGBoost model trainer for regression tasks.

    XGBoost = eXtreme Gradient Boosting
    - Fast and accurate machine learning algorithm
    - Works by combining many simple decision trees
    - Popular for tabular data (like our material features)
    """

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 max_depth: int = 8, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, use_gpu: bool = True,
                 random_state: int = 42):
        """
        Initialize XGBoost trainer.

        Args:
            n_estimators: Number of trees to build (more = better but slower)
            learning_rate: How fast the model learns (lower = more careful)
            max_depth: How complex each tree can be (higher = more complex)
            subsample: Fraction of samples to use per tree (prevents overfitting)
            colsample_bytree: Fraction of features to use per tree
            use_gpu: Use GPU acceleration if available
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray = None, y_test: np.ndarray = None,
              verbose: bool = True) -> Dict:
        """
        Train XGBoost model.

        Example:
            trainer = XGBoostTrainer()
            stats = trainer.train(X_train, y_train, X_test, y_test)
            print(f"Training took {stats['training_time']:.2f}s")

        Args:
            X_train: Training features (2400 samples × 132 features)
            y_train: Training targets (2400 band gap values)
            X_test: Test features (600 samples × 132 features)
            y_test: Test targets (600 band gap values)
            verbose: Print progress

        Returns:
            Dictionary with training_time, cv_mae, cv_std
        """
        if verbose:
            print("\n[4/6] Training XGBoost model...")
            print(f"✓ Training set:   {len(X_train)} samples")
            if X_test is not None:
                print(f"✓ Test set:       {len(X_test)} samples")
                print(f"✓ Train/Test ratio: {len(X_train)/len(X_test):.1f}:1")
            print(f"Device: {'GPU' if self.use_gpu else 'CPU'}")

        start_time = time.time()

        # Initialize model
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method='hist',
            device='cuda:0' if self.use_gpu else 'cpu',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Prepare evaluation set
        eval_set = []
        if X_test is not None and y_test is not None:
            eval_set = [(X_train, y_train), (X_test, y_test)]

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if eval_set else None,
            verbose=False
        )

        training_time = time.time() - start_time

        if verbose:
            print(f"✓ Training complete ({training_time:.2f}s)")

        # Cross-validation with fresh model instance
        stats = {'training_time': training_time}

        if verbose:
            print("\nRunning cross-validation...")

        # Create fresh model for CV - use CPU to avoid device mismatch warnings
        # (cross_val_score clones models internally and doesn't handle GPU well)
        cv_model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method='hist',
            device='cpu',  # Use CPU for CV to avoid device mismatch
            random_state=self.random_state,
            n_jobs=-1
        )

        cv_scores = cross_val_score(
            cv_model, X_train, y_train,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        stats['cv_mae'] = -cv_scores.mean()
        stats['cv_std'] = cv_scores.std()

        if verbose:
            print(f"✓ 5-Fold CV MAE: {stats['cv_mae']:.3f} ± {stats['cv_std']:.3f} eV")

        return stats

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Example:
            predictions = trainer.predict(X_test)
            # predictions = [0.0, 2.3, 1.5, ...] (predicted band gaps)

        Args:
            X: Features to predict on

        Returns:
            Predicted band gap values
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")

        # Use GPU for prediction via DMatrix to avoid device mismatch warning
        if self.use_gpu:
            try:
                import xgboost as xgb
                # Create DMatrix on GPU device to match model device
                dmatrix = xgb.DMatrix(X, device='cuda:0')
                predictions = self.model.get_booster().predict(dmatrix)
                # Ensure we return a numpy array
                return np.asarray(predictions)
            except Exception as e:
                # Fall back to standard prediction if GPU DMatrix fails
                print(f"GPU prediction failed ({e}), falling back to CPU")
                return self.model.predict(X)
        else:
            return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """
        Evaluate model performance.

        Calculates three metrics:
        - MAE (Mean Absolute Error): Average prediction error
        - RMSE (Root Mean Squared Error): Penalizes large errors more
        - R² Score: How well predictions match reality (1.0 = perfect)

        Args:
            X: Features
            y: True target values
            verbose: Print metrics

        Returns:
            Dictionary with mae, rmse, r2
        """
        y_pred = self.predict(X)

        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }

        if verbose:
            print(f"\nMetrics:")
            print(f"  MAE:  {metrics['mae']:.3f} eV")
            print(f"  RMSE: {metrics['rmse']:.3f} eV")
            print(f"  R²:   {metrics['r2']:.3f}")

        return metrics

    def get_model(self) -> XGBRegressor:
        """Get the underlying XGBoost model object."""
        return self.model

    def get_feature_importance(self, feature_names: list = None) -> np.ndarray:
        """
        Get feature importances.

        Shows which features the model thinks are most important for predictions.

        Args:
            feature_names: Optional list of feature names

        Returns:
            Array of importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")

        return self.model.feature_importances_

    def display_evaluation(self, metrics: Dict, verbose: bool = True):
        """
        Display evaluation metrics in a formatted table.

        Args:
            metrics: Dictionary with train/test metrics (mae, rmse, r2)
            verbose: Print results
        """
        if not verbose:
            return

        print(f"\n[5/6] Evaluating model...")
        print(f"\n{'='*70}")
        print("MODEL PERFORMANCE")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {'Training':<15} {'Test':<15} {'Difference':<15}")
        print(f"{'-'*70}")
        print(f"{'MAE (eV)':<20} {metrics['train_mae']:<15.3f} {metrics['test_mae']:<15.3f} {abs(metrics['train_mae']-metrics['test_mae']):<15.3f}")
        print(f"{'RMSE (eV)':<20} {metrics['train_rmse']:<15.3f} {metrics['test_rmse']:<15.3f} {abs(metrics['train_rmse']-metrics['test_rmse']):<15.3f}")
        print(f"{'R² Score':<20} {metrics['train_r2']:<15.3f} {metrics['test_r2']:<15.3f} {abs(metrics['train_r2']-metrics['test_r2']):<15.3f}")
        print(f"{'='*70}")

        # Overfitting check
        overfit_check = abs(metrics['train_mae'] - metrics['test_mae'])
        if overfit_check < 0.05:
            print("✓ Excellent generalization (low overfitting)")
        elif overfit_check < 0.15:
            print("✓ Good generalization")
        else:
            print("⚠ Warning: Possible overfitting detected")

        print(f"{'='*70}")


# ============================================================================
# EXAMPLE: How to add other model types
# ============================================================================
#
# To add a different ML algorithm, implement the ModelTrainer interface:
#
# from sklearn.ensemble import RandomForestRegressor
#
# class RandomForestTrainer(ModelTrainer):
#     """Random Forest model trainer."""
#
#     def __init__(self, n_estimators: int = 100, max_depth: int = 10):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.model = None
#
#     def train(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
#         self.model = RandomForestRegressor(
#             n_estimators=self.n_estimators,
#             max_depth=self.max_depth,
#             random_state=42
#         )
#         self.model.fit(X_train, y_train)
#         return {'training_time': 10.5}
#
#     def predict(self, X):
#         return self.model.predict(X)
#
#     def evaluate(self, X, y, verbose=True):
#         y_pred = self.predict(X)
#         return {
#             'mae': mean_absolute_error(y, y_pred),
#             'rmse': np.sqrt(mean_squared_error(y, y_pred)),
#             'r2': r2_score(y, y_pred)
#         }
#
#     def get_model(self):
#         return self.model
#
# Usage:
#     trainer = RandomForestTrainer(n_estimators=200)
#     trainer.train(X_train, y_train)
#     predictions = trainer.predict(X_test)
# ============================================================================
