#!/usr/bin/env python3
"""Systematically verify column mappings for ALL years 2006-2022."""

from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

# Year -> (ACS txt file, R50089 CSV file) mapping
YEAR_FILES = {
    2006: ("ACS2006_R50089713.txt", "R50089713_SL050.csv"),
    2007: ("ACS2007_R50089714.txt", "R50089714_SL050.csv"),
    2008: ("ACS2008_R50089715.txt", "R50089715_SL050.csv"),
    2009: ("ACS2009_R50089710.txt", "R50089710_SL050.csv"),
    2010: ("ACS2010_R50089711.txt", "R50089711_SL050.csv"),
    2011: ("ACS2011_R50089712.txt", "R50089712_SL050.csv"),
    2012: ("ACS2012_R50089707.txt", "R50089707_SL050.csv"),
    2013: ("ACS2013_R50089708.txt", "R50089708_SL050.csv"),
    2014: ("ACS2014_R50089709.txt", "R50089709_SL050.csv"),
    2015: ("ACS2015_R50089704.txt", "R50089704_SL050.csv"),
    2016: ("ACS2016_R50089705.txt", "R50089705_SL050.csv"),
    2017: ("ACS2017_R50089706.txt", "R50089706_SL050.csv"),
    2018: ("ACS2018_R50089717.txt", "R50089717_SL050.csv"),
    2019: ("ACS2019_R50089718.txt", "R50089718_SL050.csv"),
    2021: ("ACS2021_R50089719.txt", "R50089719_SL050.csv"),
    2022: ("ACS2022_R50089698.txt", "R50089698_SL050.csv"),
}

print("="*90)
print("YEAR-BY-YEAR COLUMN MAPPING VERIFICATION (2006-2022)")
print("="*90)

all_results = []

for year in sorted(YEAR_FILES.keys()):
    txt_file, csv_file = YEAR_FILES[year]
    txt_path = DATA_DIR / txt_file
    csv_path = DATA_DIR / csv_file
    
    print(f"\n{'='*90}")
    print(f"YEAR {year}")
    print(f"  TXT: {txt_file}")
    print(f"  CSV: {csv_file}")
    print('='*90)
    
    result = {'year': year, 'txt_file': txt_file, 'csv_file': csv_file}
    
    # Check if files exist
    if not txt_path.exists():
        print(f"  ✗ TXT file NOT FOUND")
        continue
    if not csv_path.exists():
        print(f"  ✗ CSV file NOT FOUND")
        continue
    
    # Read ACS metadata
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Extract variables from metadata
    age_vars = re.findall(r'\s+(A01001_\d+):\s+(.+)', content)
    age_vars = sorted([(v, d.strip()) for v, d in age_vars])
    
    edu_vars = re.findall(r'\s+(A12001_\d+):\s+(.+)', content)
    edu_vars = sorted([(v, d.strip()) for v, d in edu_vars])
    
    race_vars = re.findall(r'\s+(A04001_\d+):\s+(.+)', content)
    race_vars = sorted([(v, d.strip()) for v, d in race_vars])
    
    print(f"\n  METADATA (ACS{year}):")
    print(f"    Age (A01001): {len(age_vars)} variables")
    for v, d in age_vars[:3]:
        print(f"      {v}: {d[:50]}")
    if len(age_vars) > 3:
        print(f"      ... and {len(age_vars)-3} more")
    
    print(f"\n    Education (A12001): {len(edu_vars)} variables")
    for v, d in edu_vars[:3]:
        print(f"      {v}: {d[:50]}")
    if len(edu_vars) > 3:
        print(f"      ... and {len(edu_vars)-3} more")
        
    print(f"\n    Race (A04001): {len(race_vars)} variables")
    for v, d in race_vars[:3]:
        print(f"      {v}: {d[:50]}")
    if len(race_vars) > 3:
        print(f"      ... and {len(race_vars)-3} more")
    
    # Read CSV and verify columns exist
    try:
        df = pd.read_csv(csv_path, skiprows=1, nrows=5)
        csv_cols = list(df.columns)
        
        print(f"\n  CSV VERIFICATION:")
        print(f"    Total columns in CSV: {len(csv_cols)}")
        
        # Check FIPS column
        fips_col = None
        if 'Geo_FIPS' in csv_cols:
            fips_col = 'Geo_FIPS'
        elif 'Geo__geoid_' in csv_cols:
            fips_col = 'Geo__geoid_'
        
        if fips_col:
            print(f"    ✓ FIPS column: {fips_col}")
            print(f"      Sample value: {df[fips_col].iloc[0]}")
        else:
            print(f"    ✗ NO FIPS COLUMN FOUND")
            print(f"      Available Geo columns: {[c for c in csv_cols if 'Geo' in c][:5]}")
        
        # Check age columns (we need 002-013)
        age_needed = [v for v, d in age_vars if 2 <= int(v.split('_')[1]) <= 13]
        age_csv_cols = [f"SE_{v}" for v in age_needed if f"SE_{v}" in csv_cols]
        age_missing = [f"SE_{v}" for v in age_needed if f"SE_{v}" not in csv_cols]
        print(f"\n    Age columns in CSV (002-013): {len(age_csv_cols)}/{len(age_needed)}")
        if age_missing:
            print(f"    ✗ Missing: {age_missing}")
        else:
            print(f"    ✓ All age columns present")
        
        # Check education columns (we need 002-008)
        edu_needed = [v for v, d in edu_vars if 2 <= int(v.split('_')[1]) <= 8]
        edu_csv_cols = [f"SE_{v}" for v in edu_needed if f"SE_{v}" in csv_cols]
        edu_missing = [f"SE_{v}" for v in edu_needed if f"SE_{v}" not in csv_cols]
        print(f"\n    Education columns in CSV (002-008): {len(edu_csv_cols)}/{len(edu_needed)}")
        if edu_missing:
            print(f"    ✗ Missing: {edu_missing}")
        else:
            print(f"    ✓ All education columns present")
        
        # Check race columns (we need 003-009 for non-Hispanic)
        race_needed = [v for v, d in race_vars if 3 <= int(v.split('_')[1]) <= 9]
        race_csv_cols = [f"SE_{v}" for v in race_needed if f"SE_{v}" in csv_cols]
        race_missing = [f"SE_{v}" for v in race_needed if f"SE_{v}" not in csv_cols]
        print(f"\n    Race columns in CSV (003-009): {len(race_csv_cols)}/{len(race_needed)}")
        if race_missing:
            print(f"    ✗ Missing: {race_missing}")
        else:
            print(f"    ✓ All race columns present")
        
        result.update({
            'age_meta': len(age_vars),
            'age_csv': len(age_csv_cols),
            'edu_meta': len(edu_vars),
            'edu_csv': len(edu_csv_cols),
            'race_meta': len(race_vars),
            'race_csv': len(race_csv_cols),
            'fips_col': fips_col
        })
        
    except Exception as e:
        print(f"\n  ✗ ERROR reading CSV: {e}")
        result['error'] = str(e)
    
    all_results.append(result)

# Summary table
print("\n\n" + "="*90)
print("SUMMARY TABLE")
print("="*90)
print(f"{'Year':<6} {'FIPS Column':<15} {'Age':<10} {'Edu':<10} {'Race':<10} {'Status':<10}")
print("-"*90)

for r in all_results:
    year = r['year']
    fips = r.get('fips_col', 'N/A')
    age = f"{r.get('age_csv', 0)}/{r.get('age_meta', 0)}"
    edu = f"{r.get('edu_csv', 0)}/{r.get('edu_meta', 0)}"
    race = f"{r.get('race_csv', 0)}/{r.get('race_meta', 0)}"
    
    # Determine status
    if 'error' in r:
        status = "ERROR"
    elif r.get('age_csv', 0) == 0 or r.get('edu_csv', 0) == 0 or r.get('race_csv', 0) == 0:
        status = "INCOMPLETE"
    elif r.get('age_csv') == r.get('age_meta') and r.get('edu_csv') == r.get('edu_meta') and r.get('race_csv') == r.get('race_meta'):
        status = "✓ OK"
    else:
        status = "PARTIAL"
    
    print(f"{year:<6} {fips:<15} {age:<10} {edu:<10} {race:<10} {status:<10}")

print("="*90)
