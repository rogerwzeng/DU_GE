#!/usr/bin/env python3
"""Debug demographic data extraction."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

# Load 2022 data as example
df = pd.read_csv(DATA_DIR / "R50089698_SL050.csv", skiprows=1, low_memory=False)
df['county_fips'] = df['Geo__geoid_'].astype(str).str.zfill(5)

print("="*70)
print("DEBUG: Demographic Data Extraction for 2022")
print("="*70)

print(f"\nTotal rows: {len(df)}")
print(f"Sample FIPS: {df['county_fips'].iloc[0]}")

# Check age columns
age_cols = [f'SE_A01001_{i:03d}' for i in range(2, 14)]
age_present = [c for c in age_cols if c in df.columns]
age_missing = [c for c in age_cols if c not in df.columns]
print(f"\nAge columns: {len(age_present)}/12 present")
if age_missing:
    print(f"  Missing: {age_missing}")
else:
    print(f"  All present!")
    # Check sample values
    sample = df.iloc[0][age_cols]
    print(f"  Sample values: {sample.values}")
    print(f"  Sum: {sample.sum()}")

# Check education columns
edu_cols = [f'SE_A12001_{i:03d}' for i in range(2, 9)]
edu_present = [c for c in edu_cols if c in df.columns]
edu_missing = [c for c in edu_cols if c not in df.columns]
print(f"\nEducation columns: {len(edu_present)}/7 present")
if edu_missing:
    print(f"  Missing: {edu_missing}")
else:
    print(f"  All present!")
    sample = df.iloc[0][edu_cols]
    print(f"  Sample values: {sample.values}")
    print(f"  Sum: {sample.sum()}")

# Check race columns
race_cols = [f'SE_A04001_{i:03d}' for i in range(3, 10)]
race_present = [c for c in race_cols if c in df.columns]
race_missing = [c for c in race_cols if c not in df.columns]
print(f"\nRace columns: {len(race_present)}/7 present")
if race_missing:
    print(f"  Missing: {race_missing}")
else:
    print(f"  All present!")
    sample = df.iloc[0][race_cols]
    print(f"  Sample values: {sample.values}")
    print(f"  Sum: {sample.sum()}")

# Count how many counties have valid data for all three
print("\n" + "="*70)
print("CHECKING VALID DATA COUNTS")
print("="*70)

valid_count = 0
invalid_reasons = {'age': 0, 'education': 0, 'race': 0, 'multiple': 0}

for idx, row in df.iterrows():
    has_age = all(pd.notna(row[c]) for c in age_present)
    has_edu = all(pd.notna(row[c]) for c in edu_present)
    has_race = all(pd.notna(row[c]) for c in race_present)
    
    if has_age and has_edu and has_race:
        valid_count += 1
    else:
        issues = []
        if not has_age:
            issues.append('age')
        if not has_edu:
            issues.append('education')
        if not has_race:
            issues.append('race')
        
        if len(issues) > 1:
            invalid_reasons['multiple'] += 1
        else:
            invalid_reasons[issues[0]] += 1

print(f"\nCounties with all 3 dimensions valid (no NaN): {valid_count}/{len(df)}")
print(f"Counties with NaN values:")
for reason, count in invalid_reasons.items():
    print(f"  {reason}: {count}")

# Show some examples of counties
print("\n" + "="*70)
print("EXAMPLES OF COUNTIES")
print("="*70)

for idx, row in df.head(10).iterrows():
    fips = row['county_fips']
    age_vals = [row[c] for c in age_cols]
    edu_vals = [row[c] for c in edu_cols]
    race_vals = [row[c] for c in race_cols]
    
    # Check for NaN values (allow zero values)
    age_valid = all(pd.notna(v) for v in age_vals)
    edu_valid = all(pd.notna(v) for v in edu_vals)
    race_valid = all(pd.notna(v) for v in race_vals)
    
    status = "✓" if (age_valid and edu_valid and race_valid) else "✗"
    age_sum = sum(v for v in age_vals if pd.notna(v))
    edu_sum = sum(v for v in edu_vals if pd.notna(v))
    race_sum = sum(v for v in race_vals if pd.notna(v))
    
    print(f"\n{status} County {fips}:")
    print(f"  Age: sum={age_sum:.0f}, valid={age_valid}")
    print(f"  Edu: sum={edu_sum:.0f}, valid={edu_valid}")
    print(f"  Race: sum={race_sum:.0f}, valid={race_valid}")
