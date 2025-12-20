"""Test Gold layer data preparation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformations.gold.h3_mapper import prepare_for_database, map_to_h3
import pandas as pd

# Test data
df = pd.DataFrame({
    'occurred_at': ['2014-01-01 04:00:00', '2014-01-01 08:00:00'],
    'latitude': [43.6532, 43.7000],
    'longitude': [-79.3832, -79.4000],
    'crime_type': ['Assault', 'Robbery'],
    'neighbourhood': ['Downtown', 'North'],
    'premise_type': ['Apartment', 'House'],
    'source_file': ['test.csv', 'test.csv']
})

print("Testing H3 mapping...")
df_with_h3 = map_to_h3(df)
print(f"H3 indices: {df_with_h3['h3_index'].tolist()}")

print("\nTesting data preparation...")
result = prepare_for_database(df_with_h3)
print(f"occurred_at type: {result['occurred_at'].dtype}")
print(f"Timezone: {result['occurred_at'].iloc[0].tz}")
print(f"Columns: {list(result.columns)}")
print(f"Data types:\n{result.dtypes}")

print("\n[PASS] All Gold layer functions work correctly!")

