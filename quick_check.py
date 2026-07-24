#!/usr/bin/env python3
"""Quick check of data coverage."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

print("="*60)
print("DATA COVERAGE CHECK")
print("="*60)

# Load CBSA file
cbsa = pd.read_csv(DATA_DIR / "reference" / "cbsa_county_crosswalk.csv")
cbsa['county_fips'] = cbsa['State FIPS'].astype(str).str.zfill(2) + cbsa['County FIPS'].astype(str).str.zfill(3)
print(f"\nCBSA file: {cbsa['CBSA Code'].nunique()} unique MSAs")
print(f"CBSA file: {cbsa['county_fips'].nunique()} unique counties")

# Check 2022 SE file
se2022 = pd.read_csv(DATA_DIR / "R50089698_SL050.csv", skiprows=1, low_memory=False)
se2022['county_fips'] = se2022['Geo__geoid_'].astype(str).str.zfill(5)
print(f"\nSE 2022: {len(se2022)} counties")

# Check overlap with CBSA
se2022_counties = set(se2022['county_fips'])
cbsa_counties = set(cbsa['county_fips'])
overlap_2022 = se2022_counties & cbsa_counties
print(f"SE 2022 overlap with CBSA: {len(overlap_2022)} counties")

# Count MSAs that have at least one county in SE
cbsa_with_2022 = cbsa[cbsa['county_fips'].isin(se2022_counties)]['CBSA Code'].nunique()
print(f"\nMSAs with at least one county in SE 2022: {cbsa_with_2022}")

# Check which MSAs are NOT covered
all_cbsa_codes = set(cbsa['CBSA Code'].unique())
cbsa_2022_codes = set(cbsa[cbsa['county_fips'].isin(se2022_counties)]['CBSA Code'].unique())
missing_cbsa = all_cbsa_codes - cbsa_2022_codes

print(f"\nMSAs missing from SE 2022: {len(missing_cbsa)}")
print("\nExamples of missing MSAs (first 15):")
cbsa_titles = cbsa[['CBSA Code', 'CBSA Title']].drop_duplicates().set_index('CBSA Code')['CBSA Title'].to_dict()
for code in sorted(list(missing_cbsa))[:15]:
    title = cbsa_titles.get(code, 'Unknown')
    if title and pd.notna(title):
        title_str = str(title)
        if len(title_str) > 40:
            title_str = title_str[:37] + "..."
    else:
        title_str = 'Unknown'
    print(f"  {code}: {title_str}")
