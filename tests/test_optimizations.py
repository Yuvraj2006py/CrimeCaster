"""
Test script to verify Bronze and Silver layer optimizations work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest
from ingestion.bronze.csv_loader import (
    detect_encoding,
    load_csv,
)
from ingestion.silver.cleaner import (
    build_column_mapping as silver_build_column_mapping,
    validate_coordinates_vectorized,
    standardize_crime_types_fast,
    remove_duplicates_fast,
    transform_to_schema_single_pass,
)


def test_encoding_detection():
    """Test encoding detection function."""
    # Create a test CSV file
    test_file = project_root / "data" / "raw" / "test_encoding.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write test data with UTF-8
    test_data = pd.DataFrame({"col1": ["test", "data"], "col2": [1, 2]})
    test_data.to_csv(test_file, index=False, encoding="utf-8")
    
    try:
        encoding = detect_encoding(test_file)
        assert encoding is not None
        assert isinstance(encoding, str)
        print("[PASS] Encoding detection test passed")
    finally:
        if test_file.exists():
            test_file.unlink()


def test_column_mapping():
    """Test column mapping function."""
    # Create test DataFrame with various column names
    df = pd.DataFrame({
        "OCC_DATE": ["2024-01-01", "2024-01-02"],
        "OCC_HOUR": [10, 15],
        "LAT_WGS84": [43.7, 43.8],
        "LONG_WGS84": [-79.4, -79.5],
        "MCI_CATEGORY": ["Assault", "Robbery"],
        "NEIGHBOURHOOD_158": ["Downtown", "Uptown"],
        "PREMISES_TYPE": ["Apartment", "House"],
        "EVENT_UNIQUE_ID": ["1", "2"],
    })
    
    col_map = silver_build_column_mapping(df)
    
    assert col_map["date"] == "OCC_DATE"
    assert col_map["hour"] == "OCC_HOUR"
    assert col_map["lat"] == "LAT_WGS84"
    assert col_map["lon"] == "LONG_WGS84"
    assert col_map["crime"] == "MCI_CATEGORY"
    assert col_map["neighbourhood"] is not None
    assert "NEIGHBOURHOOD" in col_map["neighbourhood"].upper()
    assert col_map["premise"] == "PREMISES_TYPE"
    assert col_map["unique_id"] == "EVENT_UNIQUE_ID"
    
    print("[PASS] Column mapping test passed")


def test_vectorized_coordinate_validation():
    """Test vectorized coordinate validation."""
    df = pd.DataFrame({
        "lat": [43.7, 43.8, 50.0, None, 43.6],  # 50.0 and None are invalid
        "lon": [-79.4, -79.5, -80.0, -79.3, None],  # -80.0 and None are invalid
    })
    
    valid_mask = validate_coordinates_vectorized(df, "lat", "lon")
    
    # Should have 2 valid rows (indices 0 and 1)
    assert valid_mask.sum() == 2
    assert valid_mask.iloc[0] == True
    assert valid_mask.iloc[1] == True
    assert valid_mask.iloc[2] == False  # Out of bounds
    assert valid_mask.iloc[3] == False  # None
    assert valid_mask.iloc[4] == False  # None
    
    print("[PASS] Vectorized coordinate validation test passed")


def test_standardize_crime_types_fast():
    """Test optimized crime type standardization."""
    df = pd.DataFrame({
        "MCI_CATEGORY": [
            "assault",
            "robbery",
            "break and enter",
            "auto theft",
            "theft over",
            "unknown",
        ]
    })
    
    result = standardize_crime_types_fast(df, "MCI_CATEGORY")
    
    assert result.iloc[0] == "Assault"
    assert result.iloc[1] == "Robbery"
    assert result.iloc[2] == "Break and Enter"
    assert result.iloc[3] == "Auto Theft"
    assert result.iloc[4] == "Theft Over"
    
    print("[PASS] Crime type standardization test passed")


def test_remove_duplicates_fast():
    """Test optimized duplicate removal."""
    df = pd.DataFrame({
        "id": [1, 2, 2, 3, 1, 4],
        "value": ["a", "b", "c", "d", "e", "f"],
    })
    
    result = remove_duplicates_fast(df, "id")
    
    # Should have 4 unique rows (ids: 1, 2, 3, 4)
    assert len(result) == 4
    assert set(result["id"].values) == {1, 2, 3, 4}
    
    print("[PASS] Duplicate removal test passed")


def test_single_pass_transformation():
    """Test single-pass transformation."""
    df = pd.DataFrame({
        "OCC_DATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "OCC_HOUR": [10, 15, 20],
        "LAT_WGS84": [43.7, 43.8, 50.0],  # 50.0 is invalid
        "LONG_WGS84": [-79.4, -79.5, -79.6],
        "MCI_CATEGORY": ["Assault", "Robbery", "Break and Enter"],
        "NEIGHBOURHOOD_158": ["Downtown", "Uptown", "Midtown"],
        "PREMISES_TYPE": ["Apartment", "House", "Store"],
    })
    
    result = transform_to_schema_single_pass(df, "test_file.csv")
    
    # Should have 2 valid rows (third row has invalid coordinates)
    assert len(result) == 2
    assert "occurred_at" in result.columns
    assert "latitude" in result.columns
    assert "longitude" in result.columns
    assert "crime_type" in result.columns
    assert "source_file" in result.columns
    assert "dataset_type" in result.columns
    
    print("[PASS] Single-pass transformation test passed")


def run_all_tests():
    """Run all optimization tests."""
    print("=" * 60)
    print("Testing Bronze and Silver Layer Optimizations")
    print("=" * 60)
    print()
    
    tests = [
        test_encoding_detection,
        test_column_mapping,
        test_vectorized_coordinate_validation,
        test_standardize_crime_types_fast,
        test_remove_duplicates_fast,
        test_single_pass_transformation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

