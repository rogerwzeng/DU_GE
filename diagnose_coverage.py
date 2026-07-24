#!/usr/bin/env python3
"""Detailed diagnostic of MSA data coverage."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")
STUDY_YEARS = [y for y in range(2006, 2023) if y != 2020]

SE_FILE_MAPPING = {
    2006: "R50089713_SL050.csv", 2007: "R50089714_SL050.csv", 
    2008: "R50089715_SL050.csv", 2009: "R50089710_SL050.csv",
    2010: "R50089711_SL050.csv", 2011: "R50089712_SL050.csv",
    2012: "R50089707_SL050.csv", 2013: "R50089708_SL050.csv",
    2014: "R50089709_SL050.csv", 2015: "R50089704_SL050.csv",
    2016: "R50089705_SL050.csv", 2017: "R50089706_SL050.csv",
    2018: "R50089717_SL050.csv", 2019: "R50089718_SL050.csv",
    2021: "R50089719_SL050.csv", 2022: "R50089698_SL050.csv",
}

print("="*80)
print("DETAILED MSA COVERAGE DIAGNOSTIC")
print("="*80)

# Load CBSA file
cbsa = pd.read_csv(DATA_DIR / "reference" / "cbsa_county_crosswalk.csv")
cbsa['county_fips'] = cbsa['State FIPS'].astype(str).str.zfill(2) + cbsa['County FIPS'].astype(str).str.zfill(3)
cbsa_titles = cbsa[['CBSA Code', 'CBSA Title']].drop_duplicates().set_index('CBSA Code')['CBSA Title'].to_dict()

print(f"\nStudy period: {STUDY_YEARS[0]}-{STUDY_YEARS[-1]} ({len(STUDY_YEARS)} years, excluding 2020)")
print(f"Total MSAs in CBSA file: {cbsa['CBSA Code'].nunique()}")

# Build year-by-year coverage for each MSA
print("\n" + "="*80)
print("CHECKING YEAR-BY-YEAR COVERAGE...")
print("="*80)

msa_coverage = {}  # {cbsa_code: {year: n_counties}}

for year in STUDY_YEARS:
    filename = SE_FILE_MAPPING[year]
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        print(f"  {year}: FILE NOT FOUND")
        continue
    
    # Load SE data
    df = pd.read_csv(filepath, skiprows=1, low_memory=False)
    
    # Extract FIPS
    if 'Geo_FIPS' in df.columns:
        df['county_fips'] = df['Geo_FIPS'].astype(str).str.zfill(5)
    elif 'Geo__geoid_' in df.columns:
        df['county_fips'] = df['Geo__geoid_'].astype(str).str.zfill(5)
    
    se_counties = set(df['county_fips'])
    
    # Check which MSAs have coverage this year
    for cbsa_code in cbsa['CBSA Code'].unique():
        if cbsa_code not in msa_coverage:
            msa_coverage[cbsa_code] = {}
        
        # Get counties for this MSA
        msa_counties = set(cbsa[cbsa['CBSA Code'] == cbsa_code]['county_fips'])
        
        # Count how many counties have SE data
        overlap = msa_counties & se_counties
        msa_coverage[cbsa_code][year] = len(overlap)

# Analyze results
print("\n" + "="*80)
print("COVERAGE SUMMARY BY MSA")
print("="*80)

results = []
for cbsa_code, year_data in sorted(msa_coverage.items()):
    valid_years = [y for y, count in year_data.items() if count > 0]
    n_valid = len(valid_years)
    
    # Get MSA counties
    msa_counties = set(cbsa[cbsa['CBSA Code'] == cbsa_code]['county_fips'])
    total_counties = len(msa_counties)
    
    results.append({
        'cbsa_code': cbsa_code,
        'cbsa_title': cbsa_titles.get(cbsa_code, 'Unknown'),
        'n_valid_years': n_valid,
        'total_counties': total_counties,
        'valid_years': valid_years,
        'missing_years': [y for y in STUDY_YEARS if y not in valid_years]
    })

results_df = pd.DataFrame(results)

# Show distribution
print("\nDistribution of valid years per MSA:")
coverage_dist = results_df['n_valid_years'].value_counts().sort_index()
for years, count in coverage_dist.items():
    pct = count / len(results_df) * 100
    bar = "█" * int(count / 5)
    print(f"  {years:2d} years: {count:3d} MSAs ({pct:5.1f}%) {bar}")

# Show MSAs with <10 years
low_coverage = results_df[results_df['n_valid_years'] < 10].sort_values('n_valid_years')
print(f"\n\n{'='*80}")
print(f"MSAs WITH <10 YEARS OF DATA (n={len(low_coverage)})")
print(f"{'='*80}")

for _, row in low_coverage.iterrows():
    print(f"\n{row['cbsa_code']}: {row['cbsa_title']}")
    print(f"  Valid years: {row['n_valid_years']}/16")
    print(f"  Years present: {row['valid_years']}")
    print(f"  Years missing: {row['missing_years']}")
    print(f"  Total counties in MSA: {row['total_counties']}")

print(f"\n\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"MSAs with 16 years (complete): {len(results_df[results_df['n_valid_years'] == 16])}")
print(f"MSAs with 10-15 years: {len(results_df[(results_df['n_valid_years'] >= 10) & (results_df['n_valid_years'] < 16)])}")
print(f"MSAs with 5-9 years: {len(results_df[(results_df['n_valid_years'] >= 5) & (results_df['n_valid_years'] < 10)])}")
print(f"MSAs with 1-4 years: {len(results_df[(results_df['n_valid_years'] >= 1) & (results_df['n_valid_years'] < 5)])}")
print(f"MSAs with 0 years: {len(results_df[results_df['n_valid_years'] == 0])}")

# Show some specific examples
print(f"\n\n{'='*80}")
print(f"EXAMPLES: MSAs with GAPS in coverage")
print(f"{'='*80}")

# Find MSAs with gaps
for _, row in low_coverage.head(20).iterrows():
    print(f"\n{row['cbsa_code']}: {row['cbsa_title']}")
    print(f"  Valid years: {row['n_valid_years']}/16")
    print(f"  Present: {row['valid_years']}")
    print(f"  Missing: {row['missing_years']}")
