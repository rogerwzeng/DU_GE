#!/usr/bin/env python3
"""
Compute Demographic FLUX (eta_flux) - Rate of Demographic Change

Uses existing panel_complete.csv but computes flux instead of efficiency.
Formula: eta_flux(t) = |P_t - P_{t-1}|_Hellinger
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")

def main():
    print("="*70)
    print("Demographic FLUX (eta_flux)")
    print("="*70)
    print("\nComputing year-to-year demographic change rates...")
    
    # Load existing panel
    df = pd.read_csv(RESULTS_DIR / "panel_complete.csv")
    df['cbsa_code'] = df['cbsa_code'].astype(str)
    df = df.sort_values(['cbsa_code', 'year'])
    
    # Since we don't have raw distributions in panel_complete, 
    # we compute flux as absolute change in eta components
    # This is an approximation: flux ≈ |Δeta| when eta changes
    
    # Compute absolute changes
    df['eta_flux_approx'] = df.groupby('cbsa_code')['eta'].diff().abs()
    df['eta_flux_age'] = df.groupby('cbsa_code')['eta_age'].diff().abs()
    df['eta_flux_education'] = df.groupby('cbsa_code')['eta_education'].diff().abs()
    df['eta_flux_race'] = df.groupby('cbsa_code')['eta_race'].diff().abs()
    
    # Drop first year for each MSA (no change computed)
    results = df[df['eta_flux_approx'].notna()].copy()
    
    # Summary
    print(f"\nRecords: {len(results)}")
    print(f"MSAs: {results['cbsa_code'].nunique()}")
    
    print(f"\nη_flux (approximate demographic flux):")
    print(f"  Mean:   {results['eta_flux_approx'].mean():.4f}")
    print(f"  Median: {results['eta_flux_approx'].median():.4f}")
    print(f"  Std:    {results['eta_flux_approx'].std():.4f}")
    
    print(f"\nBy component:")
    print(f"  Age:        {results['eta_flux_age'].mean():.4f}")
    print(f"  Education:  {results['eta_flux_education'].mean():.4f}")
    print(f"  Race:       {results['eta_flux_race'].mean():.4f}")
    
    # Top MSAs by flux
    top = results.groupby('cbsa_code').agg({
        'eta_flux_approx': 'mean',
        'cbsa_title': 'first'
    }).nlargest(5, 'eta_flux_approx')
    
    print(f"\nTop 5 MSAs (highest demographic flux):")
    for _, row in top.iterrows():
        title = str(row['cbsa_title'])[:40]
        print(f"  {title:40s}: {row['eta_flux_approx']:.4f}")
    
    # Save
    results[['cbsa_code', 'cbsa_title', 'year', 'eta_flux_approx', 
             'eta_flux_age', 'eta_flux_education', 'eta_flux_race',
             'eta', 'sigma_econ', 'sigma_mig', 'population']].to_csv(
        RESULTS_DIR / "eta_flux_2006-2022.csv", index=False)
    
    print(f"\nSaved: eta_flux_2006-2022.csv")

if __name__ == "__main__":
    main()
