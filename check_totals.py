#!/usr/bin/env python3
"""Check if total race population is available when breakdown is suppressed."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

# Load 2022 data
df = pd.read_csv(DATA_DIR / "R50089698_SL050.csv", skiprows=1, low_memory=False)
df['county_fips'] = df['Geo__geoid_'].astype(str).str.zfill(5)

race_detail_cols = [f'SE_A04001_{i:03d}' for i in range(3, 10)]
race_total_col = 'SE_A04001_001'
race_nonhispanic_col = 'SE_A04001_002'

print("="*70)
print("CHECKING RACE TOTALS VS BREAKDOWN")
print("="*70)

# Find counties with NaN in breakdown but check if totals exist
nan_with_total = 0
nan_without_total = 0

for idx, row in df.iterrows():
    detail_vals = [row[c] for c in race_detail_cols]
    has_nan_detail = any(pd.isna(v) for v in detail_vals)
    
    if has_nan_detail:
        total_val = row.get(race_total_col, np.nan)
        nonhisp_val = row.get(race_nonhispanic_col, np.nan)
        
        if pd.notna(total_val) or pd.notna(nonhisp_val):
            nan_with_total += 1
        else:
            nan_without_total += 1

print(f"\nCounties with NaN race breakdown:")
print(f"  But HAVE total population data: {nan_with_total}")
print(f"  And NO total population data: {nan_without_total}")

# Show examples
print("\n" + "="*70)
print("EXAMPLES")
print("="*70)

examples = 0
for idx, row in df.iterrows():
    detail_vals = [row[c] for c in race_detail_cols]
    has_nan_detail = any(pd.isna(v) for v in detail_vals)
    
    if has_nan_detail and examples < 5:
        total_val = row.get(race_total_col, 'N/A')
        nonhisp_val = row.get(race_nonhispanic_col, 'N/A')
        
        print(f"\nCounty {row['county_fips']}: {row.get('Geo_NAME', 'Unknown')}")
        print(f"  Race detail (003-009): All NaN")
        print(f"  A04001_001 (Total): {total_val}")
        print(f"  A04001_002 (Non-Hispanic): {nonhisp_val}")
        
        # Also check age total for comparison
        age_total = row.get('SE_A01001_001', 'N/A')
        print(f"  A01001_001 (Age Total): {age_total}")
        
        examples += 1

print("\n" + "="*70)
print("SOLUTION")
print("="*70)
print("""
The issue: compute_eta.py requires ALL 3 dimensions to have data:
  - Age: A01001_002-013 (12 categories)
  - Education: A12001_002-008 (7 categories)
  - Race: A04001_003-009 (7 categories)

When race is suppressed, the county is excluded entirely.

Better approach:
- Allow counties with missing race data
- Compute eta_age and eta_education from all counties
- Compute eta_race only from counties with race data
- This increases MSA coverage from 136 to ~380+
""")
