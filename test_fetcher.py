"""
Test and Demonstration Script for MaterialsProjectFetcher
=========================================================
This script demonstrates how to use the MaterialsProjectFetcher interface
to retrieve materials data from the Materials Project API.

Usage:
    python test_fetcher.py
"""

from data_fetchers import MaterialsProjectFetcher
import pandas as pd

# Configuration
API_KEY = "eXF7FfK8NjzokZ2ofiBwIccSTixJehn8"  # Materials Project API key


def test_basic_fetch():
    """Test basic fetching with default parameters."""
    print("=" * 70)
    print("TEST 1: Basic Fetch (50 materials)")
    print("=" * 70)

    # Create fetcher instance
    fetcher = MaterialsProjectFetcher(api_key=API_KEY)

    # Fetch small sample
    df, stats = fetcher.fetch(n_samples=50, verbose=True)

    print("\n📊 Returned Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return df, stats


def test_custom_elements():
    """Test fetching with custom element range."""
    print("\n" + "=" * 70)
    print("TEST 2: Binary Compounds Only (num_elements=(2, 2))")
    print("=" * 70)

    # Create fetcher for binary compounds only
    fetcher = MaterialsProjectFetcher(
        api_key=API_KEY,
        num_elements=(2, 2)  # Only materials with exactly 2 elements
    )

    df, stats = fetcher.fetch(n_samples=30, verbose=True)

    # Show formula examples
    print("\n🧪 Sample Binary Compound Formulas:")
    print(df[['formula', 'band_gap']].head(15).to_string(index=False))

    return df, stats


def test_silent_fetch():
    """Test fetching without verbose output."""
    print("\n" + "=" * 70)
    print("TEST 3: Silent Fetch (verbose=False)")
    print("=" * 70)

    fetcher = MaterialsProjectFetcher(api_key=API_KEY)

    print("Fetching 20 materials silently...")
    df, stats = fetcher.fetch(n_samples=20, verbose=False)

    print(f"✓ Done! Retrieved {stats['total']} materials")
    print(f"  Conductors: {stats['conductors']}")
    print(f"  Semiconductors: {stats['semiconductors']}")
    print(f"  Insulators: {stats['insulators']}")

    return df, stats


def test_data_analysis():
    """Demonstrate basic data analysis on fetched data."""
    print("\n" + "=" * 70)
    print("TEST 4: Data Analysis Example")
    print("=" * 70)

    fetcher = MaterialsProjectFetcher(api_key=API_KEY, num_elements=(1, 3))
    df, stats = fetcher.fetch(n_samples=100, verbose=False)

    print(f"\n📈 Analysis of {len(df)} materials:\n")

    # Find materials with specific band gap ranges
    semiconductors = df[(df['band_gap'] > 1.0) & (df['band_gap'] < 3.0)]
    print(f"Materials with band gap 1-3 eV (good for solar cells): {len(semiconductors)}")

    if len(semiconductors) > 0:
        print("\nTop 5 Solar Cell Candidates:")
        print(semiconductors[['formula', 'band_gap', 'nsites']].head().to_string(index=False))

    # Find the largest unit cells
    print(f"\n🏗️  Largest Unit Cell: {df['formula'].iloc[df['volume'].idxmax()]}")
    print(f"   Volume: {df['volume'].max():.2f} Ų")

    # Find the smallest structures
    print(f"\n⚛️  Smallest Structure: {df['formula'].iloc[df['nsites'].idxmin()]}")
    print(f"   Sites: {df['nsites'].min()}")

    return df, stats


def test_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 70)
    print("TEST 5: Interface Consistency Check")
    print("=" * 70)

    fetcher = MaterialsProjectFetcher(api_key=API_KEY)
    df, stats = fetcher.fetch(n_samples=10, verbose=False)

    # Verify interface contract
    print("\n✓ Checking interface contract...")

    # Check DataFrame structure
    required_columns = ['material_id', 'formula', 'band_gap', 'nsites', 'volume']
    assert all(col in df.columns for col in required_columns), "Missing required columns!"
    print(f"  ✓ DataFrame has all required columns: {required_columns}")

    # Check stats structure
    required_stats = ['total', 'conductors', 'semiconductors', 'insulators',
                      'mean_bandgap', 'median_bandgap', 'min_bandgap',
                      'max_bandgap', 'fetch_time']
    assert all(key in stats for key in required_stats), "Missing required stats!"
    print(f"  ✓ Stats dict has all required keys")

    # Check data types
    assert isinstance(df, pd.DataFrame), "Return type must be DataFrame!"
    assert isinstance(stats, dict), "Stats must be a dictionary!"
    print(f"  ✓ Return types are correct (DataFrame, Dict)")

    # Check value ranges
    assert stats['total'] == len(df), "Total count mismatch!"
    assert stats['conductors'] + stats['semiconductors'] + stats['insulators'] == stats['total']
    print(f"  ✓ Statistics are internally consistent")

    print("\n✅ All interface checks passed!")

    return True


def main():
    """Run all tests."""
    print("\n" + "🔬" * 35)
    print("MaterialsProjectFetcher Test Suite")
    print("🔬" * 35 + "\n")

    try:
        # Run all tests
        test_basic_fetch()
        test_custom_elements()
        test_silent_fetch()
        test_data_analysis()
        test_error_handling()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe MaterialsProjectFetcher is working correctly and ready to use!")
        print("\n💡 Usage in your own code:")
        print("   from data_fetchers import MaterialsProjectFetcher")
        print("   fetcher = MaterialsProjectFetcher(api_key=YOUR_KEY)")
        print("   df, stats = fetcher.fetch(n_samples=1000)")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
