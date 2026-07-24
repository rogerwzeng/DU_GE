#!/usr/bin/env python3
"""Test script to verify compute_eta.py will run successfully."""

import sys
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

print("="*50)
print("Testing compute_eta.py prerequisites")
print("="*50)

# Check all required Social Explorer files
required_files = {
    2006: "R50089713_SL050.csv",
    2007: "R50089714_SL050.csv",
    2008: "R50089715_SL050.csv",
    2009: "R50089710_SL050.csv",
    2010: "R50089711_SL050.csv",
    2011: "R50089712_SL050.csv",
    2012: "R50089707_SL050.csv",
    2013: "R50089708_SL050.csv",
    2014: "R50089709_SL050.csv",
    2015: "R50089704_SL050.csv",
    2016: "R50089705_SL050.csv",
    2017: "R50089706_SL050.csv",
    2018: "R50089717_SL050.csv",
    2019: "R50089718_SL050.csv",
    2021: "R50089719_SL050.csv",
    2022: "R50089698_SL050.csv",
}

print("\nChecking Social Explorer files:")
missing = []
for year, filename in required_files.items():
    filepath = DATA_DIR / filename
    status = "✓" if filepath.exists() else "✗"
    print(f"  {status} {year}: {filename}")
    if not filepath.exists():
        missing.append((year, filename))

# Check CBSA file
print("\nChecking CBSA delineation file:")
cbsa_file = DATA_DIR / "reference" / "cbsa_county_crosswalk.csv"
if cbsa_file.exists():
    print(f"  ✓ {cbsa_file.name}")
else:
    print(f"  ✗ {cbsa_file.name} NOT FOUND")
    missing.append(("CBSA", str(cbsa_file)))

# Summary
print("\n" + "="*50)
if missing:
    print(f"✗ MISSING {len(missing)} FILES:")
    for item in missing:
        print(f"  - {item}")
else:
    print("✓ ALL REQUIRED FILES PRESENT")
print("="*50)
