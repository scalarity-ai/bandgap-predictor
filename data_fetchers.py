"""
Data Fetcher Interface and Implementations
==========================================
Abstract interface for fetching materials data with concrete implementations
for various materials databases.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
import time
from mp_api.client import MPRester


class MaterialsDataFetcher(ABC):
    """Abstract interface for fetching materials data."""

    @abstractmethod
    def fetch(self, n_samples: int, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch materials data.

        Args:
            n_samples: Number of materials to retrieve
            verbose: Print progress information

        Returns:
            Tuple of (DataFrame with materials data, statistics dict)
        """
        pass


class MaterialsProjectFetcher(MaterialsDataFetcher):
    """Concrete implementation for Materials Project API."""

    def __init__(self, api_key: str, num_elements: Tuple[int, int] = (1, 6)):
        """
        Initialize Materials Project fetcher.

        Args:
            api_key: Materials Project API key
            num_elements: Range of elements per material (min, max)
        """
        self.api_key = api_key
        self.num_elements = num_elements
        self.fields = ["material_id", "formula_pretty", "band_gap", "nsites", "volume"]

    def fetch(self, n_samples: int, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch materials from Materials Project API.

        Args:
            n_samples: Number of materials to retrieve
            verbose: Print progress information

        Returns:
            Tuple of (DataFrame, statistics dict)
        """
        if verbose:
            print(f"\n[1/6] Fetching materials data from Materials Project...")

        start_time = time.time()

        # Query Materials Project API
        with MPRester(self.api_key) as mpr:
            docs = mpr.materials.summary.search(
                num_elements=self.num_elements,
                fields=self.fields
            )

        # Convert to DataFrame
        data = []
        for i, doc in enumerate(docs):
            if i >= n_samples:
                break
            if doc.band_gap is not None:
                data.append({
                    'material_id': str(doc.material_id),
                    'formula': doc.formula_pretty,
                    'band_gap': doc.band_gap,
                    'nsites': doc.nsites if doc.nsites else 0,
                    'volume': doc.volume if doc.volume else 0
                })

        df = pd.DataFrame(data)
        fetch_time = time.time() - start_time

        # Calculate statistics
        conductors = (df['band_gap'] == 0).sum()
        semiconductors = ((df['band_gap'] > 0) & (df['band_gap'] <= 4)).sum()
        insulators = (df['band_gap'] > 4).sum()

        stats = {
            'total': len(df),
            'conductors': int(conductors),
            'semiconductors': int(semiconductors),
            'insulators': int(insulators),
            'mean_bandgap': float(df['band_gap'].mean()),
            'median_bandgap': float(df['band_gap'].median()),
            'min_bandgap': float(df['band_gap'].min()),
            'max_bandgap': float(df['band_gap'].max()),
            'fetch_time': fetch_time
        }

        if verbose:
            self._print_statistics(df, stats)

        return df, stats

    def _print_statistics(self, df: pd.DataFrame, stats: Dict):
        """Print dataset statistics."""
        print(f"✓ Retrieved {len(df)} materials ({stats['fetch_time']:.1f}s)")
        print(f"\nDataset Preview:")
        print(df.head(10))
        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print(f"{'='*70}")
        print(f"Total materials: {stats['total']}")
        print(f"Band gap range: {stats['min_bandgap']:.2f} - {stats['max_bandgap']:.2f} eV")
        print(f"Mean band gap: {stats['mean_bandgap']:.2f} eV")
        print(f"Median band gap: {stats['median_bandgap']:.2f} eV")
        print(f"\nMaterial Types:")
        print(f"  Conductors (gap = 0):       {stats['conductors']:4d} ({stats['conductors']/stats['total']*100:.1f}%)")
        print(f"  Semiconductors (0-4 eV):    {stats['semiconductors']:4d} ({stats['semiconductors']/stats['total']*100:.1f}%)")
        print(f"  Insulators (>4 eV):         {stats['insulators']:4d} ({stats['insulators']/stats['total']*100:.1f}%)")
        print(f"{'='*70}")


# ============================================================================
# EXAMPLE: How to add additional data sources
# ============================================================================
#
# To add a new materials database, simply implement the MaterialsDataFetcher interface:
#
# class OpenQuantumMaterialsFetcher(MaterialsDataFetcher):
#     def __init__(self, api_key: str, **kwargs):
#         self.api_key = api_key
#         # Add source-specific configuration
#
#     def fetch(self, n_samples: int, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
#         # Implement fetching logic for this source
#         # Must return (DataFrame, stats_dict) with same structure
#         pass
#
# Usage:
#     fetcher = OpenQuantumMaterialsFetcher(api_key=KEY)
#     df, stats = fetcher.fetch(n_samples=1000)
# ============================================================================
