"""Quick verification script for Silver layer output."""

import pandas as pd
from pathlib import Path

silver_dir = Path("data/silver")
latest = max(silver_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)

print("=" * 60)
print("SILVER LAYER VERIFICATION")
print("=" * 60)
print(f"\nFile: {latest.name}")

df = pd.read_csv(latest, nrows=5000)

print(f"\nTotal rows sampled: {len(df):,}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nNull counts:")
print(df.isnull().sum())
print(f"\nCrime types: {sorted(df['crime_type'].unique())}")
print(f"\nDate range: {df['occurred_at'].min()} to {df['occurred_at'].max()}")
print(f"\nCoordinate ranges:")
print(f"  Lat: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
print(f"  Lon: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
print(f"\nSample rows:")
print(df.head(2))

# Validation checks
print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

checks = {
    "No nulls in occurred_at": df["occurred_at"].isnull().sum() == 0,
    "No nulls in latitude": df["latitude"].isnull().sum() == 0,
    "No nulls in longitude": df["longitude"].isnull().sum() == 0,
    "No nulls in crime_type": df["crime_type"].isnull().sum() == 0,
    "Valid coordinates (lat 43.5-43.9)": (
        (df["latitude"] >= 43.5) & (df["latitude"] <= 43.9)
    ).all(),
    "Valid coordinates (lon -79.8 to -79.0)": (
        (df["longitude"] >= -79.8) & (df["longitude"] <= -79.0)
    ).all(),
    "Valid crime types": all(
        ct in ["Assault", "Robbery", "Break and Enter", "Auto Theft", "Theft Over"]
        for ct in df["crime_type"].unique()
    ),
}

for check, passed in checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status}: {check}")

all_passed = all(checks.values())
print("\n" + "=" * 60)
if all_passed:
    print("[PASS] ALL CHECKS PASSED - Silver layer output is valid!")
else:
    print("[FAIL] SOME CHECKS FAILED - Review output")
print("=" * 60)

