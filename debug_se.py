#!/usr/bin/env python3
"""Debug Social Explorer file formats."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

# Check files - pick early, middle, and late years
test_files = [
    ("R50089713_SL050.csv", "2006"),
    ("R50089704_SL050.csv", "2015"), 
    ("R50089698_SL050.csv", "2022"),
]

for filename, year in test_files:
    filepath = DATA_DIR / filename
    print(f"\n{'='*50}")
    print(f"File: {filename} (Year {year})")
    print('='*50)
    
    if not filepath.exists():
        print("  FILE NOT FOUND")
        continue
    
    # Read first 5 lines raw
    with open(filepath, 'r') as f:
        lines = [f.readline().strip() for _ in range(5)]
    
    print("\nFirst 3 lines (raw):")
    for i, line in enumerate(lines[:3]):
        print(f"  Line {i}: {line[:100]}...")
    
    # Read with pandas
    df = pd.read_csv(filepath, skiprows=1, nrows=3)
    print(f"\nColumn names (first 10):")
    for col in list(df.columns)[:10]:
        print(f"  - {col}")
    
    # Check for FIPS columns
    fips_cols = [c for c in df.columns if 'fips' in c.lower() or 'geo' in c.lower()]
    print(f"\nFIPS/GEO columns: {fips_cols}")
    
    for col in fips_cols:
        print(f"  {col}: {df[col].iloc[0]}")

print("\n" + "="*50)
