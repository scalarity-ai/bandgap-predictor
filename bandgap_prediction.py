"""
Materials Project: Band Gap Prediction from Composition
========================================================
Predict material band gaps using only chemical composition (no structure needed)

Prerequisites:
    pip install wandb xgboost mp-api matminer scikit-learn pandas matplotlib seaborn

Setup:
    1. Get free API key from https://next-gen.materialsproject.org/
    2. Run: wandb login (get key from https://wandb.ai/authorize)
    3. Replace API_KEY below
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import wandb
import time
from datetime import datetime

# Import our modular components
from data_fetchers import MaterialsProjectFetcher
from feature_engineering import CompositionFeatureEngineer
from model_trainer import XGBoostTrainer
from visualization import RegressionVisualizer
from experiment_tracker import create_tracker

# ============================================================================
# CONFIGURATION
# ============================================================================
API_KEY = "eXF7FfK8NjzokZ2ofiBwIccSTixJehn8"  # Get from materialsproject.org
USE_WANDB = True               # Set False to run without W&B
USE_GPU = True                 # Set False for CPU-only
N_SAMPLES = 3000               # Number of materials to download

# Experiment config
CONFIG = {
    'n_samples': N_SAMPLES,
    'learning_rate': 0.05,
    'max_depth': 8,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'device': 'gpu' if USE_GPU else 'cpu'
}

# ============================================================================
# INITIALIZE EXPERIMENT TRACKING
# ============================================================================
tracker = create_tracker(USE_WANDB)
tracker.init(
    project="materials-bandgap",
    name=f"xgboost-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config=CONFIG
)

print("=" * 70)
print("MATERIALS PROJECT: BAND GAP PREDICTION")
print("=" * 70)
print(f"Device: {'GPU (CUDA)' if USE_GPU else 'CPU'}")
print(f"Target samples: {N_SAMPLES}")
print(f"W&B tracking: {'Enabled' if USE_WANDB else 'Disabled'}")
print("=" * 70)

# ============================================================================
# STEP 1: DATA COLLECTION
# ============================================================================
# Initialize fetcher and retrieve data
fetcher = MaterialsProjectFetcher(api_key=API_KEY, num_elements=(1, 6))
df, stats = fetcher.fetch(n_samples=N_SAMPLES, verbose=True)

# Log to tracker
tracker.log({
    "dataset/total_samples": stats['total'],
    "dataset/conductors": stats['conductors'],
    "dataset/semiconductors": stats['semiconductors'],
    "dataset/insulators": stats['insulators'],
    "dataset/mean_bandgap": stats['mean_bandgap'],
})

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
# Initialize feature engineer and generate features
engineer = CompositionFeatureEngineer(
    use_magpie=True,
    use_stoichiometry=True,
    use_valence=True
)
X, y, feature_cols, feature_stats = engineer.generate_features(df, verbose=True)

tracker.log({
    "features/total": feature_stats['total_features'],
    "features/clean_samples": feature_stats['clean_samples']
})

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=(y == 0)  # Stratify by conductor/non-conductor
)

# ============================================================================
# STEP 4: MODEL TRAINING & CROSS-VALIDATION
# ============================================================================
# Initialize trainer
trainer = XGBoostTrainer(
    n_estimators=CONFIG['n_estimators'],
    learning_rate=CONFIG['learning_rate'],
    max_depth=CONFIG['max_depth'],
    subsample=CONFIG['subsample'],
    colsample_bytree=CONFIG['colsample_bytree'],
    use_gpu=USE_GPU,
    random_state=42
)

# Train model (includes cross-validation)
training_stats = trainer.train(X_train, y_train, X_test, y_test, verbose=True)

tracker.log({
    "training/time_seconds": training_stats['training_time'],
    "cv/mae_mean": training_stats['cv_mae'],
    "cv/mae_std": training_stats['cv_std']
})

# ============================================================================
# STEP 5: EVALUATION
# ============================================================================
# Get predictions and metrics
y_pred_train = trainer.predict(X_train)
y_pred_test = trainer.predict(X_test)

metrics = {
    'train_mae': mean_absolute_error(y_train, y_pred_train),
    'test_mae': mean_absolute_error(y_test, y_pred_test),
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'train_r2': r2_score(y_train, y_pred_train),
    'test_r2': r2_score(y_test, y_pred_test),
}

# Display evaluation results
trainer.display_evaluation(metrics, verbose=True)

tracker.log(metrics)

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
# Prepare results dictionary for visualizer
viz_results = {
    'y_test': y_test,
    'y_pred_test': y_pred_test,
    'metrics': metrics,
    'feature_importance': trainer.get_feature_importance(),
    'feature_names': feature_cols,
    'dataset_size': len(df),
    'n_features': len(feature_cols),
    'training_time': training_stats['training_time'],
    'cv_mae': training_stats['cv_mae'],
    'cv_std': training_stats['cv_std']
}

# Create and save visualizations
visualizer = RegressionVisualizer(figsize=(16, 12))
fig = visualizer.create_plots(viz_results, verbose=True)
visualizer.save('bandgap_prediction_results.png', dpi=300)

# Log to tracker
tracker.log_artifact('bandgap_prediction_results.png')

# Display
visualizer.show()

# ============================================================================
# SAVE MODEL & FINISH
# ============================================================================
model_filename = 'bandgap_xgboost_model.json'
trainer.get_model().save_model(model_filename)

tracker.log_artifact(model_filename)
tracker.finish()

print("\n✓ Experiment complete! Check W&B dashboard for results.")
