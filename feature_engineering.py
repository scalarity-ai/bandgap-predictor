"""
Feature Engineering Interface and Implementations
================================================
Abstract interface for generating features from materials data with concrete
implementations for composition-based featurizers.

WHAT IS FEATURE ENGINEERING?
----------------------------
Converting raw material formulas (like "Ag2S") into numbers that machine
learning models can understand. Think of it as translating chemistry into
math that computers can learn from.

Example:
    "Ag2S" → [82.6, 1.93, 7.33, ...] (132 numbers)

These numbers represent properties like atomic mass, electronegativity, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import time
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital


class FeatureEngineer(ABC):
    """
    Abstract interface for feature engineering.

    Any feature engineer must implement:
    1. generate_features() - Convert raw data to numbers
    2. get_feature_names() - Return what each number means
    """

    @abstractmethod
    def generate_features(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Generate features from materials data.

        Args:
            df: DataFrame with 'formula' and 'band_gap' columns
            verbose: Print progress information

        Returns:
            Tuple of:
                - X: Feature matrix (samples × features)
                - y: Target values (band gaps)
                - feature_names: List of feature names
                - stats: Statistics dictionary
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (what each number represents)."""
        pass


class CompositionFeatureEngineer(FeatureEngineer):
    """
    Feature engineer for composition-based features using matminer.

    Generates features based purely on chemical composition without requiring
    crystal structure information.

    Three types of features:
    1. Magpie: Element properties (atomic mass, electronegativity, etc.)
    2. Stoichiometry: Ratios and counts (how many atoms, what ratios)
    3. Valence: Electron configuration (which orbitals are filled)
    """

    def __init__(self, use_magpie: bool = True, use_stoichiometry: bool = True,
                 use_valence: bool = True):
        """
        Initialize composition-based feature engineer.

        Args:
            use_magpie: Use Magpie element property features (132 features)
            use_stoichiometry: Use stoichiometry features (7 features)
            use_valence: Use valence orbital features (16 features)
        """
        self.use_magpie = use_magpie
        self.use_stoichiometry = use_stoichiometry
        self.use_valence = use_valence
        self.feature_names_ = []

        # Initialize featurizers with impute_nan=True for future compatibility
        self.featurizers = {}
        if use_magpie:
            # Create MagpieData source with impute_nan=True to avoid warnings
            from matminer.utils.data import MagpieData
            magpie_data = MagpieData(impute_nan=True)

            # Get the preset configuration for features and stats
            magpie_preset = ElementProperty.from_preset("magpie")

            # Create ElementProperty with the impute_nan-enabled data source
            magpie = ElementProperty(
                data_source=magpie_data,
                features=magpie_preset.features,
                stats=magpie_preset.stats,
                impute_nan=True
            )
            magpie.set_n_jobs(1)  # Avoid threading issues
            self.featurizers['magpie'] = magpie
        if use_stoichiometry:
            self.featurizers['stoichiometry'] = Stoichiometry()
        if use_valence:
            self.featurizers['valence'] = ValenceOrbital(impute_nan=True)

    def generate_features(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Generate composition-based features.

        Example:
            Input:  DataFrame with formula="Ag2S", band_gap=0.0
            Output: X=[82.6, 1.93, 7.33, ...], y=[0.0], names=["avg_mass", ...]

        Args:
            df: DataFrame with 'formula' and 'band_gap' columns
            verbose: Print progress information

        Returns:
            Tuple of (X, y, feature_names, stats)
        """
        if verbose:
            print("\n[2/6] Generating composition-based features...")
            print("This will take 1-2 minutes for 3000 materials...")

        start_time = time.time()

        # Create a copy for featurization
        feature_df = df.copy()

        # Convert formulas to Composition objects
        # "Ag2S" (string) → Composition object that matminer understands
        feature_df['composition'] = feature_df['formula'].apply(lambda x: Composition(x))

        # Apply each featurizer
        for name, featurizer in self.featurizers.items():
            if verbose:
                print(f"  - Applying {name} featurizer...")
            feature_df = featurizer.featurize_dataframe(
                feature_df,
                col_id='composition',
                ignore_errors=True
            )

        # Extract feature columns
        feature_cols = []
        if self.use_magpie:
            feature_cols += [col for col in feature_df.columns if col.startswith('MagpieData')]
        if self.use_stoichiometry:
            feature_cols += [col for col in feature_df.columns if col.startswith('Stoichiometry')]
        if self.use_valence:
            feature_cols += [col for col in feature_df.columns if col.startswith('ValenceOrbital')]

        # Remove rows with NaN features (materials we couldn't featurize)
        clean_df = feature_df.dropna(subset=feature_cols)

        # Extract X (features) and y (target)
        X = clean_df[feature_cols].values
        y = clean_df['band_gap'].values

        # Store feature names
        self.feature_names_ = feature_cols

        # Calculate statistics
        feature_time = time.time() - start_time
        stats = {
            'total_features': len(feature_cols),
            'clean_samples': len(X),
            'removed_samples': len(df) - len(X),
            'feature_time': feature_time,
            'magpie_features': sum(1 for c in feature_cols if 'Magpie' in c),
            'stoichiometry_features': sum(1 for c in feature_cols if 'Stoichiometry' in c),
            'valence_features': sum(1 for c in feature_cols if 'ValenceOrbital' in c),
        }

        if verbose:
            self._print_statistics(stats)

        return X, y, feature_cols, stats

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names_

    def _print_statistics(self, stats: Dict):
        """Print feature generation statistics."""
        print(f"✓ Generated {stats['total_features']} features ({stats['feature_time']:.1f}s)")
        print(f"✓ Clean samples: {stats['clean_samples']} (removed {stats['removed_samples']} with missing features)")
        print(f"\nFeature breakdown:")
        print(f"  - Magpie (element properties):    {stats['magpie_features']}")
        print(f"  - Stoichiometry:                  {stats['stoichiometry_features']}")
        print(f"  - Valence Orbitals:               {stats['valence_features']}")


# ============================================================================
# EXAMPLE: How to add custom feature engineering
# ============================================================================
#
# To add new types of features, implement the FeatureEngineer interface:
#
# class StructureFeatureEngineer(FeatureEngineer):
#     """
#     Feature engineer that uses crystal structure information.
#     Requires 3D atomic positions, not just chemical formula.
#     """
#
#     def __init__(self, structure_featurizers: List):
#         self.featurizers = structure_featurizers
#
#     def generate_features(self, df: pd.DataFrame, verbose: bool = True):
#         # Convert crystal structures to features
#         # Would require 'structure' column in DataFrame
#         # Could generate features like:
#         # - Bond angles
#         # - Atomic distances
#         # - Coordination numbers
#         # - Crystal symmetry
#         pass
#
#     def get_feature_names(self):
#         return ["bond_angle_avg", "nearest_neighbor_dist", ...]
#
# Usage:
#     engineer = StructureFeatureEngineer(featurizers=[...])
#     X, y, names, stats = engineer.generate_features(df)
# ============================================================================
