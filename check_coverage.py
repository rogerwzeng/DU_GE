#!/usr/bin/env python3
"""Check data coverage across MSAs."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

print("="*70)
print("MSA DATA COVERAGE DIAGNOSTICS")
print("="*70)

# Step 1: Load CBSA delineation
print("\nStep 1: Loading CBSA delineation...")
cbsa_df = load_cbsa_delineation()
print(f"  Total CBSA codes in file: {cbsa_df['cbsa_code'].nunique()}")

# Step 2: Load demographic data
print("\nStep 2: Loading demographic data...")
all_distributions = []

for year in STUDY_YEARS:
    print(f"  Loading {year}...")
    df = load_demographic_data(year)
    
    if df is not None:
        distributions = extract_demographic_distributions(df, year)
        all_distributions.extend(distributions)
        print(f"    {len(distributions)} counties")

print(f"\n  Total: {len(all_distributions)} county-year records")

# Step 3: Aggregate to MSA level
print("\nStep 3: Aggregating to MSA level...")
msa_distributions = aggregate_to_msa_distributions(all_distributions, cbsa_df)
print(f"    {len(msa_distributions)} MSA-year distributions")

# Step 4: Analyze coverage
print("\n" + "="*70)
print("COVERAGE ANALYSIS")
print("="*70)

# Get unique MSAs
all_cbsas = set(k[0] for k in msa_distributions.keys())
print(f"\nMSAs with ANY data: {len(all_cbsas)}")

# Count valid years per MSA
coverage = {}
for cbsa_code in all_cbsas:
    valid_years = []
    for year in STUDY_YEARS:
        key = (cbsa_code, year)
        if key in msa_distributions:
            dist = msa_distributions[key]
            if (len(dist['age']) > 0 and dist['age'].sum() > 0 and
                len(dist['education']) > 0 and dist['education'].sum() > 0 and
                len(dist['race']) > 0 and dist['race'].sum() > 0):
                valid_years.append(year)
    coverage[cbsa_code] = len(valid_years)

# Distribution of coverage
coverage_counts = pd.Series(coverage.values()).value_counts().sort_index()

print(f"\nDistribution of valid years per MSA:")
print(f"{'Years':<10} {'Count':<10} {'Cumulative':<10}")
print("-"*30)
cumulative = 0
for years, count in coverage_counts.items():
    cumulative += count
    print(f"{years:<10} {count:<10} {cumulative:<10}")

print(f"\nMSAs with 10+ years (current threshold): {sum(1 for v in coverage.values() if v >= 10)}")
print(f"MSAs with 5+ years: {sum(1 for v in coverage.values() if v >= 5)}")
print(f"MSAs with 1+ years: {sum(1 for v in coverage.values() if v >= 1)}")

# Show some examples
print("\n" + "="*70)
print("EXAMPLE MSAs")
print("="*70)

# Get CBSA titles
cbsa_titles = cbsa_df[['cbsa_code', 'cbsa_title']].drop_duplicates().set_index('cbsa_code')['cbsa_title'].to_dict()

# MSAs with 16 years
full_coverage = [cbsa for cbsa, years in coverage.items() if years == 16]
print(f"\nMSAs with full coverage (16 years): {len(full_coverage)}")
for cbsa in sorted(full_coverage)[:5]:
    print(f"  {cbsa}: {cbsa_titles.get(cbsa, 'Unknown')}")

# MSAs with 10-15 years
high_coverage = [cbsa for cbsa, years in coverage.items() if 10 <= years < 16]
print(f"\nMSAs with high coverage (10-15 years): {len(high_coverage)}")

# MSAs with 5-9 years  
medium_coverage = [cbsa for cbsa, years in coverage.items() if 5 <= years < 10]
print(f"\nMSAs with medium coverage (5-9 years): {len(medium_coverage)}")

# MSAs with <5 years
low_coverage = [cbsa for cbsa, years in coverage.items() if years < 5]
print(f"\nMSAs with low coverage (<5 years): {len(low_coverage)}")

print("\n" + "="*70)
