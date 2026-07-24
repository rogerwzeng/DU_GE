#!/usr/bin/env python3
"""
Compute economic flux (sigma_econ) from BEA CAGDP9 data.

This script:
1. Reads county-level GDP data from CAGDP9
2. Maps counties to MSAs using CBSA delineation
3. Aggregates county GDP to MSA level
4. Computes year-to-year log changes
5. Calculates mean absolute log-change = sigma_econ

Usage:
    python compute_sigma_econ.py

Output:
    results/sigma_econ_2006-2022.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")

# Create results directory if needed
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Years for analysis
YEARS = list(range(2001, 2025))  # 2001-2024 (what's available in CAGDP9)
STUDY_YEARS = [y for y in range(2006, 2023) if y != 2020]  # 2006-2019, 2021-2022 (our study period)


def load_cagdp9_data():
    """Load BEA CAGDP9 county-level GDP data."""
    filepath = DATA_DIR / "CAGDP9__ALL_AREAS_2001_2024.csv"
    
    print(f"  Reading: {filepath.name}")
    
    # Read CSV - handle potential formatting issues
    df = pd.read_csv(filepath, low_memory=False)
    
    # Filter to total GDP (LineCode = 1)
    df = df[df['LineCode'] == 1].copy()
    print(f"    Records: {len(df)} counties")
    
    # Clean FIPS codes (remove quotes and pad to 5 digits)
    df['GeoFIPS'] = df['GeoFIPS'].astype(str).str.replace('"', '').str.strip()
    df['GeoFIPS'] = df['GeoFIPS'].str.zfill(5)
    
    # Parse state and county name
    df['State'] = df['GeoName'].str.extract(r', ([A-Z]{2})$')[0]
    df['County'] = df['GeoName'].str.replace(r', [A-Z]{2}$', '', regex=True)
    
    return df


def find_cbsa_delineation_files():
    """Find all CBSA delineation files in the data directory."""
    # Check official reference location
    reference_dir = DATA_DIR / "reference"
    if reference_dir.exists():
        cbsa_files = list(reference_dir.glob('*cbsa*')) + list(reference_dir.glob('*CBSA*'))
        return sorted(set(cbsa_files))
    return []


def load_cbsa_delineation():
    """Load official OMB CBSA delineation file to map counties to MSAs.
    
    Uses data/reference/cbsa_county_crosswalk.csv with preprocessing
    to construct full 5-digit county FIPS from State FIPS + County FIPS.
    """
    filepath = DATA_DIR / "reference" / "cbsa_county_crosswalk.csv"
    
    if not filepath.exists():
        print(f"  ERROR: CBSA file not found: {filepath}")
        return None
    
    print(f"  Loading: {filepath.name}")
    try:
        cbsa = pd.read_csv(filepath)
        
        # Construct full 5-digit county FIPS from State FIPS (2-digit) + County FIPS (3-digit)
        # Example: State=48 (TX) + County=59 (Callahan) → 48059
        cbsa['county_fips'] = (
            cbsa['State FIPS'].astype(str).str.zfill(2) + 
            cbsa['County FIPS'].astype(str).str.zfill(3)
        )
        
        # Rename and prepare columns
        cbsa = cbsa.rename(columns={
            'CBSA Code': 'cbsa_code',
            'CBSA Title': 'cbsa_title'
        })
        
        # Ensure string types
        cbsa['cbsa_code'] = cbsa['cbsa_code'].astype(str)
        cbsa['county_fips'] = cbsa['county_fips'].astype(str)
        
        print(f"    {len(cbsa)} counties -> {cbsa['cbsa_code'].nunique()} MSAs")
        
        return cbsa[['cbsa_code', 'cbsa_title', 'county_fips', 'County Name', 'State']]
        
    except Exception as e:
        print(f"  ERROR loading CBSA file: {e}")
        return None


def create_county_to_cbsa_mapping(gdp_df):
    """Create a basic county-to-CBSA mapping using state information."""
    # Get unique counties from GDP data
    counties = gdp_df[['GeoFIPS', 'State', 'County']].drop_duplicates()
    
    # Create a simple placeholder mapping
    # In reality, we'd use proper CBSA delineation files
    counties['cbsa_code'] = counties['State'] + counties['GeoFIPS'].str[:3]
    counties = counties.rename(columns={'GeoFIPS': 'county_fips'})
    
    print(f"  Created placeholder mapping for {len(counties)} counties")
    print("  NOTE: This is a placeholder. Use proper CBSA delineation for accurate results.")
    
    return counties[['county_fips', 'cbsa_code']]


def aggregate_to_msa(gdp_df, cbsa_df):
    """Aggregate county-level GDP to MSA level."""
    # Merge GDP data with CBSA mapping
    merged = gdp_df.merge(
        cbsa_df,
        left_on='GeoFIPS',
        right_on='county_fips',
        how='left'
    )
    
    # Check merge success
    unmatched = merged[merged['cbsa_code'].isna()]
    matched_count = len(merged) - len(unmatched)
    if len(unmatched) > 0:
        print(f"    Warning: {len(unmatched)} counties not matched")
        print(f"    Matched: {matched_count}/{len(merged)} ({matched_count/len(merged)*100:.1f}%)")
    else:
        print(f"    Matched: all {len(merged)} counties")
    
    # Get year columns
    year_cols = [str(y) for y in YEARS if str(y) in merged.columns]
    print(f"    Years: {year_cols[0]}-{year_cols[-1]}")
    
    # Convert GDP values to numeric (handle '(D)' and other non-numeric)
    for col in year_cols:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    
    # Aggregate by CBSA code
    msa_gdp = merged.groupby('cbsa_code')[year_cols].sum(min_count=1).reset_index()
    
    # Count counties per MSA and add CBSA title
    counties_per_msa = merged.groupby('cbsa_code').size().reset_index(name='n_counties')
    msa_gdp = msa_gdp.merge(counties_per_msa, on='cbsa_code', how='left')
    
    # Add CBSA title (take first occurrence for each MSA)
    cbsa_titles = merged.groupby('cbsa_code')['cbsa_title'].first().reset_index()
    msa_gdp = msa_gdp.merge(cbsa_titles, on='cbsa_code', how='left')
    
    print(f"    {len(msa_gdp)} MSAs created")
    
    return msa_gdp


def compute_sigma_econ(msa_gdp):
    """Compute sigma_econ as year-by-year absolute log-change.
    
    For longitudinal analysis, outputs instantaneous flux at each year t:
    σ_econ(t) = |ln(GDP[t]) - ln(GDP[t-1])|
    
    Excludes COVID transitions (2019-20, 2020-21) but includes 2021-22.
    """
    results = []
    study_year_cols = [str(y) for y in STUDY_YEARS]
    
    for _, row in msa_gdp.iterrows():
        cbsa_code = row['cbsa_code']
        cbsa_title = row.get('cbsa_title', '')
        n_counties = row.get('n_counties', 0)
        
        # Extract GDP values for study years
        gdp_values = []
        for year_col in study_year_cols:
            val = row.get(year_col, np.nan)
            gdp_values.append(val)
        
        gdp_values = np.array(gdp_values)
        
        # Compute year-by-year sigma_econ
        for i in range(1, len(STUDY_YEARS)):
            year = STUDY_YEARS[i]
            prev_year = STUDY_YEARS[i-1]
            
            # Skip if this is a COVID transition
            if (prev_year == 2019 and year == 2020) or (prev_year == 2020 and year == 2021):
                continue
            
            val_curr = gdp_values[i]
            val_prev = gdp_values[i-1]
            
            # Compute instantaneous sigma_econ
            if not np.isnan(val_curr) and not np.isnan(val_prev) and val_curr > 0 and val_prev > 0:
                sigma_econ = np.abs(np.log(val_curr) - np.log(val_prev))
            else:
                sigma_econ = np.nan
            
            results.append({
                'cbsa_code': cbsa_code,
                'cbsa_title': cbsa_title,
                'year': year,
                'sigma_econ': sigma_econ,
                'gdp_current': val_curr,
                'gdp_previous': val_prev,
                'n_counties': n_counties
            })
    
    return pd.DataFrame(results)


def main():
    """Main computation pipeline."""
    print("="*50)
    print("Economic Flux (sigma_econ) - LONGITUDINAL")
    print("="*50)
    print(f"Period: {STUDY_YEARS[0]}-{STUDY_YEARS[-1]} (16 years, excl. 2020)")
    
    # Step 1: Load GDP data
    print("\nStep 1: Loading GDP data...")
    gdp_df = load_cagdp9_data()
    
    print("\nStep 2: Loading CBSA delineation...")
    cbsa_df = load_cbsa_delineation()
    if cbsa_df is None:
        print("  ERROR: CBSA file not found!")
        return None
    
    print("\nStep 3: Aggregating to MSAs...")
    msa_gdp = aggregate_to_msa(gdp_df, cbsa_df)
    if msa_gdp is None or len(msa_gdp) == 0:
        print("  ERROR: No MSA-level data!")
        return None
    
    print("\nStep 4: Computing sigma_econ time-series...")
    results_df = compute_sigma_econ(msa_gdp)
    
    # Filter to valid results
    valid_results = results_df[results_df['sigma_econ'].notna()].copy()
    
    # Summary by MSA
    msa_summary = valid_results.groupby('cbsa_code').agg({
        'sigma_econ': ['mean', 'std', 'count'],
        'cbsa_title': 'first'
    }).reset_index()
    msa_summary.columns = ['cbsa_code', 'sigma_mean', 'sigma_std', 'n_years', 'cbsa_title']
    n_msas = msa_summary['cbsa_code'].nunique()
    
    print(f"\n  Valid sigma_econ: {len(valid_results)} MSA-year records")
    print(f"  Unique MSAs: {n_msas}")
    
    if len(valid_results) == 0:
        print("\n  WARNING: No valid sigma_econ values!")
    
    # Save year-by-year GDP values
    gdp_output_path = RESULTS_DIR / "msa_gdp_2006-2022.csv"
    msa_gdp.to_csv(gdp_output_path, index=False)
    print(f"\n  Saved GDP time series: {len(msa_gdp)} MSAs")
    
    # Summary statistics
    if len(valid_results) > 0:
        print(f"\n{'='*50}")
        print("SUMMARY (Longitudinal)")
        print(f"{'='*50}")
        print(f"  MSA-year records: {len(valid_results)}")
        print(f"  Unique MSAs:      {n_msas}")
        print(f"  Years per MSA:    {msa_summary['n_years'].mean():.1f} (avg)")
        print(f"\n  σ_econ (across all observations):")
        print(f"    Mean:    {valid_results['sigma_econ'].mean():.4f}")
        print(f"    Median:  {valid_results['sigma_econ'].median():.4f}")
        print(f"    Std:     {valid_results['sigma_econ'].std():.4f}")
        print(f"    Range:   [{valid_results['sigma_econ'].min():.4f}, {valid_results['sigma_econ'].max():.4f}]")
        
        # Show top 5 MSAs with highest average volatility
        print("\n  Top 5 MSAs (most volatile on average):")
        top5 = msa_summary.nlargest(5, 'sigma_mean')[['cbsa_title', 'sigma_mean', 'n_years']]
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            title = row['cbsa_title'][:30] + "..." if len(row['cbsa_title']) > 30 else row['cbsa_title']
            print(f"    {i}. {title}: {row['sigma_mean']:.4f} (n={int(row['n_years'])})")
    
    # Save results
    output_path = RESULTS_DIR / "sigma_econ_2006-2022.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved: sigma_econ_2006-2022.csv")
    print(f"         msa_gdp_2006-2022.csv")
    print(f"\n{'='*50}")
    print("COMPLETE")
    print(f"{'='*50}")
    
    return results_df


if __name__ == "__main__":
    main()
