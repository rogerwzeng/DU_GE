#!/usr/bin/env python3
"""Check demographic variable consistency across all years."""

from pathlib import Path
import re

DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")

# All ACS metadata files
acs_files = sorted(DATA_DIR.glob("ACS20*.txt"))

print("="*80)
print("DEMOGRAPHIC VARIABLE MAPPING ACROSS YEARS")
print("="*80)

# Track consistency
all_age = set()
all_edu = set()
all_race = set()

year_data = {}

for acs_file in acs_files:
    year = acs_file.stem[3:7]
    
    with open(acs_file, 'r') as f:
        content = f.read()
    
    # Extract age variables (A01001) - categories 002-013
    age_vars = re.findall(r'\s+(A01001_\d+):\s+(.+)', content)
    age_vars = sorted([(v, d.strip()) for v, d in age_vars if 2 <= int(v.split('_')[1]) <= 13])
    
    # Extract education variables (A12001) - categories 002-008
    edu_vars = re.findall(r'\s+(A12001_\d+):\s+(.+)', content)
    edu_vars = sorted([(v, d.strip()) for v, d in edu_vars if 2 <= int(v.split('_')[1]) <= 8])
    
    # Extract race variables (A04001) - categories 003-009 (non-Hispanic)
    race_vars = re.findall(r'\s+(A04001_\d+):\s+(.+)', content)
    race_vars = sorted([(v, d.strip()) for v, d in race_vars if 3 <= int(v.split('_')[1]) <= 9])
    
    year_data[year] = {
        'age': [v for v, d in age_vars],
        'edu': [v for v, d in edu_vars],
        'race': [v for v, d in race_vars]
    }
    
    all_age.add(tuple(year_data[year]['age']))
    all_edu.add(tuple(year_data[year]['edu']))
    all_race.add(tuple(year_data[year]['race']))

# Summary
print("\nCONSISTENCY SUMMARY:")
print(f"  Age categories: {len(all_age)} unique pattern(s) (expected: 12)")
print(f"  Education categories: {len(all_edu)} unique pattern(s) (expected: 7)")
print(f"  Race categories: {len(all_race)} unique pattern(s) (expected: 7)")

if len(all_age) == 1:
    print("\n  ✓ Age: CONSISTENT across all years")
    print(f"    Variables: {list(all_age)[0]}")
else:
    print("\n  ✗ Age: INCONSISTENT")
    for pattern in all_age:
        years = [y for y, d in year_data.items() if tuple(d['age']) == pattern]
        print(f"    Pattern in {years}: {pattern}")

if len(all_edu) == 1:
    print("\n  ✓ Education: CONSISTENT across all years")
    print(f"    Variables: {list(all_edu)[0]}")
else:
    print("\n  ✗ Education: INCONSISTENT")
    for pattern in all_edu:
        years = [y for y, d in year_data.items() if tuple(d['edu']) == pattern]
        print(f"    Pattern in {years}: {pattern}")

if len(all_race) == 1:
    print("\n  ✓ Race: CONSISTENT across all years")
    print(f"    Variables: {list(all_race)[0]}")
else:
    print("\n  ✗ Race: INCONSISTENT")
    for pattern in all_race:
        years = [y for y, d in year_data.items() if tuple(d['race']) == pattern]
        print(f"    Pattern in {years}: {pattern}")

print("\n" + "="*80)
