#!/bin/bash

# Create and activate environment
python3 -m venv materials_env
source materials_env/bin/activate # Change for Windows

# Upgrade pip
pip install --upgrade pip

# Install all packages
pip install \
  wandb \
  xgboost \
  mp-api \
  matminer \
  scikit-learn \
  pandas \
  matplotlib \
  seaborn

# Verify GPU support
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"

echo "✓ Environment ready!"
echo "To activate: source materials_env/bin/activate"
echo "To deactivate: deactivate"
