#!/usr/bin/env python3
"""
Merge three manifold metrics into unified panel dataset.

This script:
1. Reads eta, sigma_econ, and sigma_mig output files
2. Merges on (cbsa_code, year)
3. Creates balanced panel with all three metrics
4. Outputs descriptive statistics and coverage analysis

Usage:
    python merge_panel.py

Output:
    results/panel_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")


def load_data():
    """Load all three metric files."""
    print("Loading data files...")
    
    # Load eta (demographic efficiency)
    eta_file = RESULTS_DIR / "eta_2006-2022.csv"
    if eta_file.exists():
        eta_df = pd.read_csv(eta_file)
        eta_df['cbsa_code'] = eta_df['cbsa_code'].astype(str)
        eta_df['year'] = eta_df['year'].astype(int)
        print(f"  eta: {len(eta_df)} records")
    else:
        print(f"  ERROR: {eta_file} not found")
        return None, None, None
    
    # Load sigma_econ (economic flux)
    sigma_econ_file = RESULTS_DIR / "sigma_econ_2006-2022.csv"
    if sigma_econ_file.exists():
        sigma_econ_df = pd.read_csv(sigma_econ_file)
        sigma_econ_df['cbsa_code'] = sigma_econ_df['cbsa_code'].astype(str)
        sigma_econ_df['year'] = sigma_econ_df['year'].astype(int)
        # Keep only necessary columns
        sigma_econ_df = sigma_econ_df[['cbsa_code', 'year', 'sigma_econ', 'gdp_current', 'gdp_previous']]
        print(f"  sigma_econ: {len(sigma_econ_df)} records")
    else:
        print(f"  ERROR: {sigma_econ_file} not found")
        return None, None, None
    
    # Load sigma_mig (migration flux)
    sigma_mig_file = RESULTS_DIR / "sigma_mig_2006-2022.csv"
    if sigma_mig_file.exists():
        sigma_mig_df = pd.read_csv(sigma_mig_file)
        sigma_mig_df['cbsa_code'] = sigma_mig_df['cbsa_code'].astype(str)
        sigma_mig_df['year'] = sigma_mig_df['year'].astype(int)
        # Keep only necessary columns
        sigma_mig_df = sigma_mig_df[['cbsa_code', 'year', 'sigma_mig', 'gross_migration', 
                                     'population', 'total_in', 'total_out']]
        print(f"  sigma_mig: {len(sigma_mig_df)} records")
    else:
        print(f"  ERROR: {sigma_mig_file} not found")
        return None, None, None
    
    return eta_df, sigma_econ_df, sigma_mig_df


def merge_panel(eta_df, sigma_econ_df, sigma_mig_df):
    """Merge all three datasets on (cbsa_code, year)."""
    print("\nMerging datasets...")
    
    # Start with eta as base (most restrictive)
    panel = eta_df.copy()
    
    # Merge sigma_econ
    panel = panel.merge(sigma_econ_df, on=['cbsa_code', 'year'], how='outer')
    print(f"  After merging sigma_econ: {len(panel)} records")
    
    # Merge sigma_mig
    panel = panel.merge(sigma_mig_df, on=['cbsa_code', 'year'], how='outer')
    print(f"  After merging sigma_mig: {len(panel)} records")
    
    # Sort for readability
    panel = panel.sort_values(['cbsa_code', 'year']).reset_index(drop=True)
    
    return panel


def analyze_coverage(panel):
    """Analyze data coverage across the three metrics."""
    print("\n" + "="*60)
    print("PANEL COVERAGE ANALYSIS")
    print("="*60)
    
    total_records = len(panel)
    print(f"\nTotal (cbsa_code, year) pairs: {total_records}")
    
    # Coverage by metric
    eta_valid = panel['eta'].notna().sum()
    sigma_econ_valid = panel['sigma_econ'].notna().sum()
    sigma_mig_valid = panel['sigma_mig'].notna().sum()
    
    print(f"\nCoverage by metric:")
    print(f"  eta:          {eta_valid:5d} / {total_records} ({100*eta_valid/total_records:.1f}%)")
    print(f"  sigma_econ:   {sigma_econ_valid:5d} / {total_records} ({100*sigma_econ_valid/total_records:.1f}%)")
    print(f"  sigma_mig:    {sigma_mig_valid:5d} / {total_records} ({100*sigma_mig_valid/total_records:.1f}%)")
    
    # Complete cases (all three metrics)
    complete_cases = panel[['eta', 'sigma_econ', 'sigma_mig']].notna().all(axis=1).sum()
    print(f"\nComplete cases (all 3 metrics): {complete_cases} ({100*complete_cases/total_records:.1f}%)")
    
    # At least two metrics
    at_least_two = (panel[['eta', 'sigma_econ', 'sigma_mig']].notna().sum(axis=1) >= 2).sum()
    print(f"At least 2 metrics: {at_least_two} ({100*at_least_two/total_records:.1f}%)")
    
    # MSAs with at least one year of complete data
    msa_complete_counts = panel.groupby('cbsa_code').apply(
        lambda x: x[['eta', 'sigma_econ', 'sigma_mig']].notna().all(axis=1).sum()
    )
    msas_with_complete = (msa_complete_counts > 0).sum()
    total_msas = panel['cbsa_code'].nunique()
    print(f"\nMSAs with at least 1 complete year: {msas_with_complete} / {total_msas}")
    
    # Year coverage
    print(f"\nYear coverage:")
    for year in sorted(panel['year'].unique()):
        year_data = panel[panel['year'] == year]
        eta_n = year_data['eta'].notna().sum()
        se_n = year_data['sigma_econ'].notna().sum()
        sm_n = year_data['sigma_mig'].notna().sum()
        print(f"  {year}: eta={eta_n:3d}, sigma_econ={se_n:3d}, sigma_mig={sm_n:3d}")
    
    return complete_cases, msas_with_complete


def compute_summary_stats(panel):
    """Compute summary statistics for the merged panel."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Complete cases only for joint statistics
    complete = panel[panel[['eta', 'sigma_econ', 'sigma_mig']].notna().all(axis=1)]
    
    print(f"\nBased on {len(complete)} complete observations:")
    
    print("\n  η (Demographic Efficiency):")
    print(f"    Mean:   {complete['eta'].mean():.4f}")
    print(f"    Std:    {complete['eta'].std():.4f}")
    print(f"    Range:  [{complete['eta'].min():.4f}, {complete['eta'].max():.4f}]")
    
    print("\n  σ_econ (Economic Flux / GDP Growth Rate):")
    print(f"    Mean:   {complete['sigma_econ'].mean():.4f} ({complete['sigma_econ'].mean()*100:.2f}%)")
    print(f"    Std:    {complete['sigma_econ'].std():.4f}")
    print(f"    Range:  [{complete['sigma_econ'].min():.4f}, {complete['sigma_econ'].max():.4f}]")
    
    print("\n  σ_mig (Migration Rate):")
    print(f"    Mean:   {complete['sigma_mig'].mean():.4f} ({complete['sigma_mig'].mean()*100:.2f}%)")
    print(f"    Std:    {complete['sigma_mig'].std():.4f}")
    print(f"    Range:  [{complete['sigma_mig'].min():.4f}, {complete['sigma_mig'].max():.4f}]")
    
    # Correlations
    print("\n  Pairwise Correlations (complete cases):")
    corr_eta_se = complete['eta'].corr(complete['sigma_econ'])
    corr_eta_sm = complete['eta'].corr(complete['sigma_mig'])
    corr_se_sm = complete['sigma_econ'].corr(complete['sigma_mig'])
    
    print(f"    η vs σ_econ:   {corr_eta_se:+.4f}")
    print(f"    η vs σ_mig:    {corr_eta_sm:+.4f}")
    print(f"    σ_econ vs σ_mig: {corr_se_sm:+.4f}")
    
    # Top MSAs by each metric
    print("\n  Top 5 MSAs by η (demographic coherence):")
    top_eta = complete.groupby('cbsa_code').agg({'eta': 'mean', 'cbsa_title': 'first'}).nlargest(5, 'eta')
    for _, row in top_eta.iterrows():
        title = row['cbsa_title'] if pd.notna(row['cbsa_title']) else 'Unknown'
        print(f"    {str(title)[:40]:40s}: {row['eta']:.4f}")
    
    print("\n  Top 5 MSAs by σ_econ (economic volatility):")
    top_se = complete.groupby('cbsa_code')['sigma_econ'].mean().nlargest(5)
    for cbsa_code, val in top_se.items():
        title = complete[complete['cbsa_code'] == cbsa_code]['cbsa_title'].iloc[0]
        print(f"    {title[:40]:40s}: {val:.4f}")
    
    print("\n  Top 5 MSAs by σ_mig (migration rate):")
    top_sm = complete.groupby('cbsa_code')['sigma_mig'].mean().nlargest(5)
    for cbsa_code, val in top_sm.items():
        title = complete[complete['cbsa_code'] == cbsa_code]['cbsa_title'].iloc[0]
        print(f"    {title[:40]:40s}: {val:.4f}")


def main():
    """Main merge pipeline."""
    print("="*60)
    print("PANEL DATA MERGE - Three Manifold Framework")
    print("="*60)
    print("\nMerging: η (demographic) + σ_econ (economic) + σ_mig (migration)")
    
    # Load data
    eta_df, sigma_econ_df, sigma_mig_df = load_data()
    
    if eta_df is None:
        print("\nERROR: Could not load required data files.")
        return
    
    # Merge
    panel = merge_panel(eta_df, sigma_econ_df, sigma_mig_df)
    
    # Analyze coverage
    complete_cases, msas_with_complete = analyze_coverage(panel)
    
    # Summary stats
    compute_summary_stats(panel)
    
    # Save panel
    output_file = RESULTS_DIR / "panel_data.csv"
    panel.to_csv(output_file, index=False)
    print(f"\n  Saved: panel_data.csv ({len(panel)} records)")
    
    # Also save complete cases only
    complete_panel = panel[panel[['eta', 'sigma_econ', 'sigma_mig']].notna().all(axis=1)].copy()
    complete_file = RESULTS_DIR / "panel_complete.csv"
    complete_panel.to_csv(complete_file, index=False)
    print(f"  Saved: panel_complete.csv ({len(complete_panel)} records, all 3 metrics)")
    
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    
    return panel


if __name__ == "__main__":
    main()
