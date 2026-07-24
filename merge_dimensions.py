#!/usr/bin/env python3
"""
Merge three dimensions (η, σ_econ, σ_mig) into master dataset.

This script:
1. Loads results from all three computation scripts
2. Merges them on CBSA code
3. Calculates intersection (MSAs with all three dimensions)
4. Saves master dataset for analysis

Usage:
    python merge_dimensions.py

Output:
    results/master_2006-2022.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")

# Years for analysis
STUDY_YEARS = [y for y in range(2006, 2023) if y != 2020]  # 2006-2019, 2021-2022


def load_results():
    """Load all three dimension results."""
    print("\nLoading dimension results...")
    
    # Load eta (demographic structure)
    eta_path = RESULTS_DIR / "eta_2006-2022.csv"
    if eta_path.exists():
        eta_df = pd.read_csv(eta_path)
        eta_df = eta_df.rename(columns={
            'eta': 'eta_overall',
            'eta_age': 'eta_age',
            'eta_education': 'eta_education', 
            'eta_race': 'eta_race'
        })
        eta_df = eta_df[['cbsa_code', 'cbsa_title', 'eta_overall', 'eta_age', 'eta_education', 
                         'eta_race', 'n_years', 'n_years_race', 'years']]
        print(f"  η (demographic): {len(eta_df)} MSAs")
    else:
        print(f"  ERROR: {eta_path} not found")
        return None, None, None
    
    # Load sigma_econ (economic flux)
    sigma_econ_path = RESULTS_DIR / "sigma_econ_2006-2022.csv"
    if sigma_econ_path.exists():
        sigma_econ_df = pd.read_csv(sigma_econ_path)
        sigma_econ_df = sigma_econ_df[['cbsa_code', 'sigma_econ', 'sigma_econ_median',
                                       'completeness', 'n_valid_transitions', 'n_counties',
                                       'gdp_2006', 'gdp_2019', 'gdp_2021', 'gdp_2024']]
        print(f"  σ_econ (economic): {len(sigma_econ_df)} MSAs")
    else:
        print(f"  ERROR: {sigma_econ_path} not found")
        return None, None, None
    
    # Load sigma_mig (migration flux)
    sigma_mig_path = RESULTS_DIR / "sigma_mig_2006-2022.csv"
    if sigma_mig_path.exists():
        sigma_mig_df = pd.read_csv(sigma_mig_path)
        sigma_mig_df = sigma_mig_df[['cbsa_code', 'sigma_mig', 'n_valid_years']]
        print(f"  σ_mig (migration): {len(sigma_mig_df)} MSAs")
    else:
        print(f"  ERROR: {sigma_mig_path} not found")
        return None, None, None
    
    return eta_df, sigma_econ_df, sigma_mig_df


def merge_dimensions(eta_df, sigma_econ_df, sigma_mig_df):
    """Merge all three dimensions on CBSA code."""
    print("\nMerging dimensions...")
    
    # Start with eta (has CBSA titles)
    master = eta_df.copy()
    
    # Merge sigma_econ
    master = master.merge(
        sigma_econ_df,
        on='cbsa_code',
        how='outer',
        indicator='_merge_econ'
    )
    
    # Merge sigma_mig
    master = master.merge(
        sigma_mig_df,
        on='cbsa_code',
        how='outer',
        indicator='_merge_mig'
    )
    
    # Get CBSA titles from eta for MSAs that only have sigma_econ or sigma_mig
    cbsa_titles = eta_df[['cbsa_code', 'cbsa_title']].drop_duplicates()
    
    # For rows without cbsa_title, try to fill from eta
    missing_title = master['cbsa_title'].isna()
    if missing_title.sum() > 0:
        master.loc[missing_title, 'cbsa_title'] = master.loc[missing_title, 'cbsa_code'].map(
            cbsa_titles.set_index('cbsa_code')['cbsa_title']
        )
    
    print(f"  Total merged: {len(master)} MSAs")
    
    # Count matches
    has_eta = master['eta_overall'].notna()
    has_econ = master['sigma_econ'].notna()
    has_mig = master['sigma_mig'].notna()
    
    print(f"\n  Match breakdown:")
    print(f"    η only:         {(has_eta & ~has_econ & ~has_mig).sum()}")
    print(f"    σ_econ only:    {(~has_eta & has_econ & ~has_mig).sum()}")
    print(f"    σ_mig only:     {(~has_eta & ~has_econ & has_mig).sum()}")
    print(f"    η + σ_econ:     {(has_eta & has_econ & ~has_mig).sum()}")
    print(f"    η + σ_mig:      {(has_eta & ~has_econ & has_mig).sum()}")
    print(f"    σ_econ + σ_mig: {(~has_eta & has_econ & has_mig).sum()}")
    print(f"    All three:      {(has_eta & has_econ & has_mig).sum()}")
    
    return master


def analyze_correlations(master_df):
    """Analyze correlations between dimensions."""
    print(f"\n{'='*50}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*50}")
    
    # Filter to MSAs with all three dimensions
    complete = master_df[
        master_df['eta_overall'].notna() & 
        master_df['sigma_econ'].notna() & 
        master_df['sigma_mig'].notna()
    ].copy()
    
    if len(complete) == 0:
        print("\n  No MSAs have all three dimensions!")
        return None
    
    print(f"\n  Complete cases: {len(complete)} MSAs")
    
    # Core correlations
    print("\n  Pearson correlations:")
    corr_eta_econ = complete['eta_overall'].corr(complete['sigma_econ'])
    corr_eta_mig = complete['eta_overall'].corr(complete['sigma_mig'])
    corr_econ_mig = complete['sigma_econ'].corr(complete['sigma_mig'])
    
    print(f"    η vs σ_econ:   {corr_eta_econ:+.4f}")
    print(f"    η vs σ_mig:    {corr_eta_mig:+.4f}")
    print(f"    σ_econ vs σ_mig: {corr_econ_mig:+.4f}")
    
    # Rank correlations (Spearman)
    print("\n  Spearman rank correlations:")
    spearman_eta_econ = complete['eta_overall'].corr(complete['sigma_econ'], method='spearman')
    spearman_eta_mig = complete['eta_overall'].corr(complete['sigma_mig'], method='spearman')
    spearman_econ_mig = complete['sigma_econ'].corr(complete['sigma_mig'], method='spearman')
    
    print(f"    η vs σ_econ:   {spearman_eta_econ:+.4f}")
    print(f"    η vs σ_mig:    {spearman_eta_mig:+.4f}")
    print(f"    σ_econ vs σ_mig: {spearman_econ_mig:+.4f}")
    
    # Summary statistics
    print("\n  Summary statistics:")
    print(f"\n    η (demographic coherence):")
    print(f"      Mean:  {complete['eta_overall'].mean():.4f}")
    print(f"      Std:   {complete['eta_overall'].std():.4f}")
    print(f"      Range: [{complete['eta_overall'].min():.4f}, {complete['eta_overall'].max():.4f}]")
    
    print(f"\n    σ_econ (economic volatility):")
    print(f"      Mean:  {complete['sigma_econ'].mean():.4f}")
    print(f"      Std:   {complete['sigma_econ'].std():.4f}")
    print(f"      Range: [{complete['sigma_econ'].min():.4f}, {complete['sigma_econ'].max():.4f}]")
    
    print(f"\n    σ_mig (migration flux):")
    print(f"      Mean:  {complete['sigma_mig'].mean():.4f}")
    print(f"      Std:   {complete['sigma_mig'].std():.4f}")
    print(f"      Range: [{complete['sigma_mig'].min():.4f}, {complete['sigma_mig'].max():.4f}]")
    
    # Top/bottom MSAs by each dimension
    print("\n  Top 5 by η (most coherent):")
    top5_eta = complete.nlargest(5, 'eta_overall')[['cbsa_title', 'eta_overall', 'sigma_econ', 'sigma_mig']]
    for i, (_, row) in enumerate(top5_eta.iterrows(), 1):
        title = row['cbsa_title'][:30] + "..." if len(row['cbsa_title']) > 30 else row['cbsa_title']
        print(f"    {i}. {title}")
        print(f"       η={row['eta_overall']:.4f}, σ_econ={row['sigma_econ']:.4f}, σ_mig={row['sigma_mig']:.4f}")
    
    print("\n  Top 5 by σ_econ (most volatile):")
    top5_econ = complete.nlargest(5, 'sigma_econ')[['cbsa_title', 'eta_overall', 'sigma_econ', 'sigma_mig']]
    for i, (_, row) in enumerate(top5_econ.iterrows(), 1):
        title = row['cbsa_title'][:30] + "..." if len(row['cbsa_title']) > 30 else row['cbsa_title']
        print(f"    {i}. {title}")
        print(f"       η={row['eta_overall']:.4f}, σ_econ={row['sigma_econ']:.4f}, σ_mig={row['sigma_mig']:.4f}")
    
    print("\n  Top 5 by σ_mig (highest flux):")
    top5_mig = complete.nlargest(5, 'sigma_mig')[['cbsa_title', 'eta_overall', 'sigma_econ', 'sigma_mig']]
    for i, (_, row) in enumerate(top5_mig.iterrows(), 1):
        title = row['cbsa_title'][:30] + "..." if len(row['cbsa_title']) > 30 else row['cbsa_title']
        print(f"    {i}. {title}")
        print(f"       η={row['eta_overall']:.4f}, σ_econ={row['sigma_econ']:.4f}, σ_mig={row['sigma_mig']:.4f}")
    
    return complete


def main():
    """Main merge pipeline."""
    print("="*50)
    print("THREE-DIMENSION ANALYSIS")
    print("(η demographic, σ_econ economic, σ_mig migration)")
    print("="*50)
    print(f"Period: {STUDY_YEARS[0]}-{STUDY_YEARS[-1]} (16 years, excl. 2020)")
    
    # Load results
    eta_df, sigma_econ_df, sigma_mig_df = load_results()
    
    if eta_df is None or sigma_econ_df is None or sigma_mig_df is None:
        print("\n  ERROR: Could not load all results!")
        return None
    
    # Merge dimensions
    master = merge_dimensions(eta_df, sigma_econ_df, sigma_mig_df)
    
    # Analyze correlations
    complete = analyze_correlations(master)
    
    # Clean up merge indicators
    master = master.drop(columns=['_merge_econ', '_merge_mig'], errors='ignore')
    
    # Add completeness flag
    master['has_all_three'] = (
        master['eta_overall'].notna() & 
        master['sigma_econ'].notna() & 
        master['sigma_mig'].notna()
    )
    
    # Reorder columns
    col_order = ['cbsa_code', 'cbsa_title', 'has_all_three',
                 'eta_overall', 'eta_age', 'eta_education', 'eta_race',
                 'sigma_econ', 'sigma_econ_median', 'sigma_mig',
                 'n_years', 'n_years_race', 'n_valid_transitions', 'n_valid_years',
                 'completeness', 'n_counties',
                 'gdp_2006', 'gdp_2019', 'gdp_2021', 'gdp_2024', 'years']
    
    # Keep only columns that exist
    col_order = [c for c in col_order if c in master.columns]
    master = master[col_order]
    
    # Save master dataset
    output_path = RESULTS_DIR / "master_2006-2022.csv"
    master.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print("OUTPUT FILES")
    print(f"{'='*50}")
    print(f"  Total MSAs: {len(master)}")
    print(f"  With all 3 dimensions: {master['has_all_three'].sum()}")
    print(f"\n  Files saved to results/:")
    print(f"    - eta_2006-2022.csv")
    print(f"    - sigma_econ_2006-2022.csv")  
    print(f"    - sigma_mig_2006-2022.csv")
    print(f"    - master_2006-2022.csv")
    
    print(f"\n{'='*50}")
    print("COMPLETE")
    print(f"{'='*50}")
    
    return master


if __name__ == "__main__":
    main()
