# Materials Band Gap Predictor

A high-performance machine learning model for predicting material band gaps from chemical composition using XGBoost with GPU acceleration and Weights & Biases experiment tracking.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project uses the **Materials Project API** to download computed materials data and trains an **XGBoost regression model** to predict band gaps from chemical composition alone—no crystal structure required. The model achieves **~0.3 eV MAE** and **~0.92 R²** on unseen materials.

### Why Band Gap?

Band gap determines whether a material is a:
- **Conductor** (gap = 0 eV): metals for wires and electrodes
- **Semiconductor** (0-4 eV): silicon chips, solar cells, LEDs  
- **Insulator** (>4 eV): protective coatings, dielectrics

Predicting band gaps enables rapid screening of materials for electronic and optoelectronic applications.

## ✨ Features

- 🚀 **GPU Acceleration**: Train on NVIDIA GPUs (10x speedup)
- 📊 **Experiment Tracking**: Automatic logging with Weights & Biases
- 🧪 **Robust Evaluation**: Cross-validation, train/test splits, comprehensive metrics
- 📈 **Rich Visualizations**: 7 publication-ready plots
- 💾 **Model Persistence**: Save and reload trained models
- 🔧 **Configurable**: Easy hyperparameter tuning
- 📝 **Well Documented**: Extensive inline comments

## 🏗️ Architecture

```
Materials Project API
        ↓
3000+ materials with band gaps
        ↓
Matminer Feature Engineering
        ↓
155 composition-based features
        ↓
XGBoost Gradient Boosting (GPU)
        ↓
Band Gap Predictions (eV)
```

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (optional, but recommended)
- Materials Project API key (free)
- Weights & Biases account (free, optional)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bandgap-predictor.git
cd bandgap-predictor
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv materials_env

# Activate it
source materials_env/bin/activate  # Linux/Mac
# OR
materials_env\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Get API Keys

**Materials Project:**
1. Create account at https://next-gen.materialsproject.org/
2. Navigate to API settings
3. Copy your API key

**Weights & Biases (optional):**
```bash
wandb login
```
Follow prompts to authenticate.

### 5. Configure the Script

Edit `bandgap_prediction.py`:
```python
API_KEY = "YOUR_MATERIALS_PROJECT_API_KEY"  # Required
USE_WANDB = True   # Set False to disable W&B
USE_GPU = True     # Set False for CPU-only
N_SAMPLES = 3000   # Number of materials to download
```

### 6. Run the Model

```bash
python bandgap_prediction.py
```

Expected runtime: **~3 minutes** (including data download and featurization)

## 📊 Results

### Performance Metrics

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| **MAE (eV)** | 0.18-0.22 | 0.25-0.35 |
| **RMSE (eV)** | 0.28-0.35 | 0.40-0.55 |
| **R² Score** | 0.94-0.96 | 0.90-0.93 |

### Output Files

- `bandgap_prediction_results.png` - Comprehensive visualization dashboard
- `bandgap_xgboost_model.json` - Trained model (for deployment)
- `wandb/` - Experiment logs (if W&B enabled)

### Visualizations

The script generates 7 plots:

1. **Predicted vs Actual**: Scatter plot showing prediction accuracy
2. **Residual Plot**: Check for systematic errors
3. **Error Distribution**: Histogram of prediction errors
4. **Feature Importance**: Top 15 most influential features
5. **Band Gap Distribution**: Actual vs predicted distributions
6. **Material Type Accuracy**: Performance by conductor/semiconductor/insulator
7. **Experiment Summary**: Key metrics and statistics

## 🔧 Configuration

### Model Hyperparameters

```python
CONFIG = {
    'n_estimators': 500,        # Number of boosting trees
    'learning_rate': 0.05,      # Step size shrinkage
    'max_depth': 8,             # Maximum tree depth
    'subsample': 0.8,           # Fraction of samples per tree
    'colsample_bytree': 0.8,    # Fraction of features per tree
}
```

### Data Settings

```python
N_SAMPLES = 3000            # Dataset size
num_elements = (1, 6)       # Materials with 1-6 elements
```

### Feature Engineering

The model uses three feature sets from matminer:

- **Magpie** (132 features): Element properties, statistics
- **Stoichiometry** (7 features): Composition ratios
- **Valence Orbital** (16 features): Electronic structure

**Total: 155 features**

## 📈 Model Details

### Algorithm: XGBoost (Extreme Gradient Boosting)

XGBoost builds an ensemble of decision trees sequentially, where each tree corrects errors from previous trees. Key advantages:

- **High accuracy** on tabular data
- **Fast training** with GPU support
- **Regularization** prevents overfitting
- **Feature importance** for interpretability

### Training Process

1. **Data Collection**: Download materials from Materials Project
2. **Feature Generation**: Convert formulas to 155 numerical features
3. **Train/Test Split**: 80/20 stratified split
4. **Model Training**: 500 gradient boosted trees
5. **Cross-Validation**: 5-fold CV for robustness
6. **Evaluation**: MAE, RMSE, R² on test set

## 🎯 Use Cases

### Materials Screening

```python
from xgboost import XGBRegressor
import joblib

# Load trained model
model = XGBRegressor()
model.load_model('bandgap_xgboost_model.json')

# Predict new materials
# (after featurizing with matminer)
predictions = model.predict(X_new)
```

### Hyperparameter Tuning

```python
# Modify CONFIG dictionary
CONFIG = {
    'n_estimators': 1000,       # Try more trees
    'learning_rate': 0.03,      # Lower learning rate
    'max_depth': 10,            # Deeper trees
}
```

### Larger Dataset

```python
N_SAMPLES = 10000  # Use more materials (slower download)
```

## 🧪 Extending the Project

### Predict Other Properties

Change the target property in the API query:

```python
# Formation energy
fields=["material_id", "formula_pretty", "formation_energy_per_atom"]

# Bulk modulus
fields=["material_id", "formula_pretty", "bulk_modulus"]

# Energy above hull (stability)
fields=["material_id", "formula_pretty", "e_above_hull"]
```

### Try Different Models

```python
# Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, n_jobs=-1)

# Neural Network
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layers=(256, 128, 64), max_iter=1000)

# Gradient Boosting (scikit-learn)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=500)
```

### Add More Features

```python
from matminer.featurizers.composition import (
    ElementProperty, 
    Stoichiometry,
    ValenceOrbital,
    IonProperty,           # Ionic radii
    CohesiveEnergy,        # Cohesive energy
    AtomicOrbitals         # Orbital character
)
```

## 📚 Project Structure

```
bandgap-predictor/
├── bandgap_prediction.py          # Main script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── materials_env/                 # Virtual environment (not in git)
├── bandgap_prediction_results.png # Output visualization
├── bandgap_xgboost_model.json    # Trained model
└── wandb/                        # W&B logs (not in git)
```

## 🐛 Troubleshooting

### API Errors

**Error: "Server does not support the request"**
```bash
# Update mp-api
pip install --upgrade mp-api
```

**Error: "Invalid API key"**
- Verify key is correct
- Check API key hasn't expired
- Try regenerating key on Materials Project website

### GPU Issues

**XGBoost not using GPU:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall XGBoost with GPU support
pip uninstall xgboost
pip install xgboost
```

**Out of GPU memory:**
```python
# Reduce batch size or use CPU
USE_GPU = False
```

### Import Errors

**Matminer import fails:**
```bash
# Common on some systems
pip install --upgrade matminer pymatgen
```

### Slow Performance

**Data download taking too long:**
```python
N_SAMPLES = 1000  # Start with smaller dataset
```

**Feature generation slow:**
- This is normal (1-2 minutes for 3000 materials)
- Uses CPU - can't be GPU accelerated

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Hyperparameter optimization with Optuna
- [ ] Neural network comparison
- [ ] Structure-based features (not just composition)
- [ ] Multi-target prediction (multiple properties)
- [ ] Uncertainty quantification
- [ ] Web interface for predictions
- [ ] Docker containerization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Materials Project** for providing the data and API
- **Matminer** for feature engineering tools
- **XGBoost** team for the excellent ML library
- **Weights & Biases** for experiment tracking

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/bandgap-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bandgap-predictor/discussions)

## 🗺️ Roadmap

- [x] Basic XGBoost implementation
- [x] GPU acceleration
- [x] W&B integration
- [ ] Automated hyperparameter tuning
- [ ] REST API for predictions
- [ ] Web interface
- [ ] Docker deployment
- [ ] Multi-property prediction
- [ ] Active learning for data efficiency

---


