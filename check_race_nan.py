#!/usr/bin/env python3
"""Check why race data has NaN values."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

# Load 2022 data
df = pd.read_csv(DATA_DIR / "R50089698_SL050.csv", skiprows=1, low_memory=False)
df['county_fips'] = df['Geo__geoid_'].astype(str).str.zfill(5)

race_cols = [f'SE_A04001_{i:03d}' for i in range(3, 10)]

print("="*70)
print("CHECKING RACE DATA FOR COUNTIES WITH NaN")
print("="*70)

# Find counties with any NaN in race
nan_counties = []
for idx, row in df.iterrows():
    race_vals = [row[c] for c in race_cols]
    if any(pd.isna(v) for v in race_vals):
        nan_counties.append({
            'fips': row['county_fips'],
            'name': row.get('Geo_NAME', 'Unknown'),
            'values': race_vals
        })

print(f"\nCounties with NaN in race: {len(nan_counties)}")

# Show first 10 examples
print("\nFirst 10 counties with NaN race data:")
for i, county in enumerate(nan_counties[:10]):
    print(f"\n{i+1}. County {county['fips']}: {county['name']}")
    print(f"   Race values: {county['values']}")
    nan_cols = [race_cols[j] for j, v in enumerate(county['values']) if pd.isna(v)]
    print(f"   NaN columns: {nan_cols}")

# Check if there's a pattern (e.g., small counties)
print("\n" + "="*70)
print("CHECKING FOR PATTERNS")
print("="*70)

# Get total population for these counties (from age total)
age_total_col = 'SE_A01001_001'
if age_total_col in df.columns:
    print("\nTotal population for counties with NaN race data:")
    for county in nan_counties[:10]:
        matching = df[df['county_fips'] == county['fips']]
        if len(matching) > 0:
            row = matching.iloc[0]
            total_pop = row.get(age_total_col, 'N/A')
            print(f"  {county['fips']}: {total_pop}")
        else:
            print(f"  {county['fips']}: NOT FOUND")

# Check what columns actually exist for race
print("\n" + "="*70)
print("ALL A04001 COLUMNS IN DATASET")
print("="*70)
a04001_cols = [c for c in df.columns if 'A04001' in c]
print(f"Found {len(a04001_cols)} A04001 columns:")
for c in sorted(a04001_cols)[:20]:
    print(f"  {c}")
if len(a04001_cols) > 20:
    print(f"  ... and {len(a04001_cols)-20} more")
