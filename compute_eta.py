#!/usr/bin/env python3
"""
Compute demographic structure (eta) using Fisher-Rao geodesic efficiency.

This script:
1. Reads county-level demographic data from Social Explorer R50089 files (2006-2022)
2. Maps counties to MSAs using CBSA delineation
3. Aggregates to MSA-level probability distributions
4. Computes Fisher-Rao geodesic efficiency for age, income, and race dimensions

Usage:
    python compute_eta.py

Output:
    results/eta_2006-2022.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/data")
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")

# Create results directory if needed
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Years for analysis: 2006-2022, excluding 2020
STUDY_YEARS = [y for y in range(2006, 2023) if y != 2020]  # 2006-2019, 2021-2022

# Social Explorer file mapping: Year -> file suffix
# R50089698 = 2006, R50089699 = 2007, ..., R50089730 = 2024
# Mapping from ACS20??_R50089???.txt files (non-linear)
SE_FILE_MAPPING = {
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
    2023: "R50089699_SL050.csv",
    2024: "R50089700_SL050.csv",
}


def fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Fisher-Rao distance between two probability vectors.
    
    d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors (must sum to ~1, all positive)
    eps : float
        Small constant to avoid log(0)
        
    Returns
    -------
    float
        Fisher-Rao distance
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Clip to avoid issues
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    
    # Clip for numerical stability
    bc = np.clip(bc, -1, 1)
    
    # Fisher-Rao distance
    return 2 * np.arccos(bc)


def compute_geodesic_efficiency(prob_matrix: np.ndarray) -> float:
    """
    Compute geodesic efficiency η for a single demographic dimension.
    
    η = d_FR(P[0], P[T]) / sum(d_FR(P[t], P[t+1]))
    
    Parameters
    ----------
    prob_matrix : np.ndarray
        Array of shape (T, K) where T is time points and K is categories
        Each row should sum to ~1.
        
    Returns
    -------
    float
        Geodesic efficiency η ∈ [0, 1]
    """
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    T = prob_matrix.shape[0]
    
    if T < 2:
        return np.nan
    
    # Normalize each row to sum to 1
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    prob_matrix = prob_matrix / row_sums
    
    # Numerator: direct distance from first to last
    p_first = prob_matrix[0]
    p_last = prob_matrix[-1]
    numerator = fisher_rao_distance(p_first, p_last)
    
    # Denominator: sum of incremental distances
    denominator = 0.0
    for t in range(T - 1):
        p_t = prob_matrix[t]
        p_t1 = prob_matrix[t + 1]
        denominator += fisher_rao_distance(p_t, p_t1)
    
    # Handle edge cases
    if denominator == 0:
        if numerator == 0:
            return 1.0  # No change at all
        else:
            return np.nan
    
    eta = numerator / denominator
    
    # Numerical protection
    return float(np.clip(eta, 0.0, 1.0))


def load_cbsa_delineation():
    """Load official OMB CBSA delineation file."""
    filepath = DATA_DIR / "reference" / "cbsa_county_crosswalk.csv"
    
    if not filepath.exists():
        print(f"  ERROR: CBSA file not found: {filepath}")
        return None
    
    print(f"  Loading: {filepath.name}")
    cbsa = pd.read_csv(filepath)
    
    # Construct full 5-digit county FIPS
    cbsa['county_fips'] = (
        cbsa['State FIPS'].astype(str).str.zfill(2) + 
        cbsa['County FIPS'].astype(str).str.zfill(3)
    )
    
    cbsa = cbsa.rename(columns={
        'CBSA Code': 'cbsa_code',
        'CBSA Title': 'cbsa_title'
    })
    
    cbsa['cbsa_code'] = cbsa['cbsa_code'].astype(str)
    cbsa['county_fips'] = cbsa['county_fips'].astype(str)
    
    print(f"    {len(cbsa)} counties -> {cbsa['cbsa_code'].nunique()} MSAs")
    
    return cbsa[['cbsa_code', 'cbsa_title', 'county_fips']]


def load_demographic_data(year):
    """Load demographic data for a specific year from Social Explorer."""
    filename = SE_FILE_MAPPING.get(year)
    if not filename:
        return None
    
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"    Warning: {filename} not found")
        return None
    
    # Read CSV - skip first row (headers), use second row as column names
    df = pd.read_csv(filepath, skiprows=1, low_memory=False)
    
    # Extract FIPS based on year-specific column naming
    # 2006-2018: Geo_FIPS, 2019+: Geo__geoid_
    if 'Geo_FIPS' in df.columns:
        # 2006-2018 format
        df['county_fips'] = df['Geo_FIPS'].astype(str).str.zfill(5)
    elif 'Geo__geoid_' in df.columns:
        # 2019+ format (note double underscore)
        df['county_fips'] = df['Geo__geoid_'].astype(str).str.zfill(5)
    else:
        print(f"    ERROR: No FIPS column found. Available: {list(df.columns)[:5]}...")
        return None
    
    return df


def extract_demographic_distributions(df, year):
    """Extract age, education, and race distributions from demographic data.
    
    Variable definitions (consistent across 2006-2022):
    - Age: A01001_002 to A01001_013 (12 categories)
    - Education: A12001_002 to A12001_008 (7 categories)  
    - Race: A04001_003 to A04001_009 (7 non-Hispanic categories)
    
    Note: Race data may be suppressed for privacy in some counties.
    We include these counties with race_values=None and handle at aggregation.
    """
    distributions = []
    
    for _, row in df.iterrows():
        county_fips = row.get('county_fips')
        if pd.isna(county_fips):
            continue
        
        # Age distribution: SE_A01001_002 to SE_A01001_013 (12 categories)
        age_cols = [f'SE_A01001_{i:03d}' for i in range(2, 14)]
        age_cols = [c for c in age_cols if c in df.columns]
        
        # Education distribution: SE_A12001_002 to SE_A12001_008 (7 categories)
        education_cols = [f'SE_A12001_{i:03d}' for i in range(2, 9)]
        education_cols = [c for c in education_cols if c in df.columns]
        
        # Race distribution: SE_A04001_003 to SE_A04001_009 (7 non-Hispanic categories)
        race_cols = [f'SE_A04001_{i:03d}' for i in range(3, 10)]
        race_cols = [c for c in race_cols if c in df.columns]
        
        try:
            age_values = row[age_cols].values.astype(float) if age_cols else np.array([])
            education_values = row[education_cols].values.astype(float) if education_cols else np.array([])
            race_values = row[race_cols].values.astype(float) if race_cols else np.array([])
            
            # Skip if age or education is missing (required)
            if len(age_values) == 0 or len(education_values) == 0:
                continue
            
            # Check if race data is valid (not all NaN)
            has_race = len(race_values) > 0 and not all(pd.isna(race_values)) and race_values.sum() > 0
            
            distributions.append({
                'county_fips': str(county_fips).zfill(5),
                'year': year,
                'age_values': age_values,
                'education_values': education_values,
                'race_values': race_values if has_race else None
            })
        except:
            continue
    
    return distributions


def extract_population_data(df, year):
    """Extract total population (SE_A01001_001) for each county.
    
    SE_A01001_001 = Total Population from ACS Table A01001 (Age)
    """
    populations = []
    pop_col = 'SE_A01001_001'
    
    if pop_col not in df.columns:
        print(f"    Warning: Population column {pop_col} not found")
        return []
    
    for _, row in df.iterrows():
        county_fips = row.get('county_fips')
        if pd.isna(county_fips):
            continue
        
        try:
            pop = float(row[pop_col])
            if pop > 0:  # Valid population count
                populations.append({
                    'county_fips': str(county_fips).zfill(5),
                    'year': year,
                    'population': pop
                })
        except:
            continue
    
    return populations


def aggregate_to_msa_population(all_populations, cbsa_df):
    """Aggregate county populations to MSA level."""
    pop_df = pd.DataFrame(all_populations)
    
    # Merge with CBSA mapping
    merged = pop_df.merge(cbsa_df, on='county_fips', how='inner')
    
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Sum population by MSA and year
    msa_pop = merged.groupby(['cbsa_code', 'year', 'cbsa_title'])['population'].sum().reset_index()
    
    return msa_pop


def aggregate_to_msa_distributions(all_distributions, cbsa_df):
    """Aggregate county distributions to MSA level."""
    # Convert to DataFrame for merging
    dist_df = pd.DataFrame(all_distributions)
    
    # Merge with CBSA mapping
    merged = dist_df.merge(cbsa_df, on='county_fips', how='inner')
    
    if len(merged) == 0:
        return {}
    
    # Group by MSA and year
    msa_distributions = {}
    
    for (cbsa_code, year), group in merged.groupby(['cbsa_code', 'year']):
        # Sum across counties for age and education (all counties)
        age_sum = np.sum([v for v in group['age_values'] if len(v) > 0], axis=0)
        education_sum = np.sum([v for v in group['education_values'] if len(v) > 0], axis=0)
        
        # Sum race only from counties with valid race data
        race_values_list = [v for v in group['race_values'] if v is not None and len(v) > 0]
        race_sum = np.sum(race_values_list, axis=0) if race_values_list else np.array([])
        
        # Count how many counties contributed to each dimension
        n_counties_age = len(group)
        n_counties_race = len(race_values_list)
        
        msa_distributions[(cbsa_code, year)] = {
            'age': age_sum,
            'education': education_sum,
            'race': race_sum,
            'n_counties_age': n_counties_age,
            'n_counties_race': n_counties_race
        }
    
    return msa_distributions


def compute_eta_for_all_msas(msa_distributions, cbsa_df, min_window=3):
    """Compute η time-series for all MSAs using rolling windows.
    
    For longitudinal analysis, we compute η_t for each year t using
    a rolling window from max(start, t-min_window+1) to t.
    
    Parameters
    ----------
    msa_distributions : dict
        MSA-year distributions
    cbsa_df : DataFrame
        CBSA metadata
    min_window : int
        Minimum window size for computing efficiency (default 3)
        
    Returns
    -------
    DataFrame with columns: cbsa_code, year, eta, eta_age, eta_education, eta_race, window_size
    """
    results = []
    
    # Get unique MSAs
    all_cbsas = set(k[0] for k in msa_distributions.keys())
    
    for cbsa_code in all_cbsas:
        # Build full time series for this MSA
        age_series = []
        education_series = []
        race_series = []
        valid_years = []
        
        for year in STUDY_YEARS:
            key = (cbsa_code, year)
            if key in msa_distributions:
                dist = msa_distributions[key]
                
                # Include if age and education have data (required)
                if (len(dist['age']) > 0 and dist['age'].sum() > 0 and
                    len(dist['education']) > 0 and dist['education'].sum() > 0):
                    
                    age_series.append(dist['age'])
                    education_series.append(dist['education'])
                    valid_years.append(year)
                    
                    # Include race only if available for this year
                    if len(dist['race']) > 0 and dist['race'].sum() > 0:
                        race_series.append(dist['race'])
                    else:
                        race_series.append(None)
        
        if len(valid_years) < min_window:
            continue
        
        # Compute rolling window efficiency for each year
        for i, year in enumerate(valid_years):
            # Window from max(0, i-min_window+1) to i, but at least 2 points
            start_idx = max(0, i - min_window + 1)
            window_len = i - start_idx + 1
            
            if window_len < 2:
                continue  # Need at least 2 points for efficiency
            
            window_age = np.array(age_series[start_idx:i+1])
            window_edu = np.array(education_series[start_idx:i+1])
            
            eta_age = compute_geodesic_efficiency(window_age)
            eta_edu = compute_geodesic_efficiency(window_edu)
            
            # Race: only use available years in window
            window_race = [r for r in race_series[start_idx:i+1] if r is not None]
            if len(window_race) >= 2:
                eta_race = compute_geodesic_efficiency(np.array(window_race))
            else:
                eta_race = np.nan
            
            # Overall η is mean across available dimensions
            eta_overall = np.nanmean([eta_age, eta_edu, eta_race])
            
            results.append({
                'cbsa_code': cbsa_code,
                'year': year,
                'eta': eta_overall,
                'eta_age': eta_age,
                'eta_education': eta_edu,
                'eta_race': eta_race,
                'window_size': window_len
            })
    
    return pd.DataFrame(results)


def main():
    """Main computation pipeline."""
    print("="*50)
    print("Demographic Structure (eta) - LONGITUDINAL")
    print("="*50)
    print(f"Period: {STUDY_YEARS[0]}-{STUDY_YEARS[-1]} (16 years, excl. 2020)")
    print()
    print("Method: Fisher-Rao geodesic efficiency")
    print("  - Age: 12 categories")
    print("  - Education: 7 categories")
    print("  - Race: 7 categories")
    
    # Step 1: Load CBSA delineation
    print("\nStep 1: Loading CBSA delineation...")
    cbsa_df = load_cbsa_delineation()
    if cbsa_df is None:
        print("  ERROR: CBSA file not found!")
        return None
    
    # Step 2: Load demographic data for all years
    print("\nStep 2: Loading demographic data...")
    all_distributions = []
    all_populations = []
    
    for year in STUDY_YEARS:
        print(f"  Loading {year}...")
        df = load_demographic_data(year)
        
        if df is not None:
            # Extract demographic distributions for eta calculation
            distributions = extract_demographic_distributions(df, year)
            all_distributions.extend(distributions)
            
            # Extract population data for MSA-level aggregation
            populations = extract_population_data(df, year)
            all_populations.extend(populations)
            
            print(f"    {len(distributions)} counties, {len(populations)} with population")
        else:
            print(f"    File not found")
    
    if len(all_distributions) == 0:
        print("\n  ERROR: No demographic data loaded!")
        return None
    
    print(f"\n  Total: {len(all_distributions)} county-year records")
    
    # Step 3a: Aggregate population to MSA level and save
    print("\nStep 3a: Aggregating population to MSA level...")
    msa_pop_df = aggregate_to_msa_population(all_populations, cbsa_df)
    if len(msa_pop_df) > 0:
        pop_output_path = RESULTS_DIR / "msa_pop_2006-2022.csv"
        msa_pop_df.to_csv(pop_output_path, index=False)
        print(f"    Saved MSA population data: {len(msa_pop_df)} records -> msa_pop.csv")
    else:
        print("    Warning: No population data to save")
    
    # Step 3b: Aggregate distributions to MSA level
    print("\nStep 3b: Aggregating distributions to MSA level...")
    msa_distributions = aggregate_to_msa_distributions(all_distributions, cbsa_df)
    
    print(f"    {len(msa_distributions)} MSA-year distributions")
    
    # Step 4: Compute η time-series
    print(f"\nStep 4: Computing eta time-series...")
    results_df = compute_eta_for_all_msas(msa_distributions, cbsa_df)
    
    # Add CBSA titles
    cbsa_titles = cbsa_df[['cbsa_code', 'cbsa_title']].drop_duplicates()
    results_df = results_df.merge(cbsa_titles, on='cbsa_code', how='left')
    
    # Filter to valid results
    valid_results = results_df[results_df['eta'].notna()].copy()
    
    # Summary by MSA
    msa_summary = valid_results.groupby('cbsa_code').agg({
        'eta': ['mean', 'std', 'count'],
        'cbsa_title': 'first'
    }).reset_index()
    msa_summary.columns = ['cbsa_code', 'eta_mean', 'eta_std', 'n_years', 'cbsa_title']
    n_msas = len(msa_summary)
    
    print(f"\n  Valid eta: {len(valid_results)} MSA-year records")
    print(f"  Unique MSAs: {n_msas}")
    
    # Summary statistics
    if len(valid_results) > 0:
        print(f"\n{'='*50}")
        print("SUMMARY (Longitudinal)")
        print(f"{'='*50}")
        print(f"  MSA-year records: {len(valid_results)}")
        print(f"  Unique MSAs:      {n_msas}")
        print(f"  Years per MSA:    {msa_summary['n_years'].mean():.1f} (avg)")
        print(f"\n  η (across all observations):")
        print(f"    Mean:  {valid_results['eta'].mean():.4f}")
        print(f"    Std:   {valid_results['eta'].std():.4f}")
        print(f"\n  η_age:        {valid_results['eta_age'].mean():.4f}")
        print(f"  η_education:  {valid_results['eta_education'].mean():.4f}")
        print(f"  η_race:       {valid_results['eta_race'].mean():.4f}")
        print()
        
        # Show top 5 MSAs with highest average η (most coherent)
        print("  Top 5 MSAs (most coherent on average):")
        top5 = msa_summary.nlargest(5, 'eta_mean')[['cbsa_title', 'eta_mean', 'n_years']]
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            title = row['cbsa_title'][:30] + "..." if len(row['cbsa_title']) > 30 else row['cbsa_title']
            print(f"    {i}. {title}: {row['eta_mean']:.4f} (n={int(row['n_years'])})")
    
    # Save results
    output_path = RESULTS_DIR / "eta_2006-2022.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved: eta_2006-2022.csv")
    print(f"\n{'='*50}")
    print("COMPLETE")
    print(f"{'='*50}")
    
    return results_df


if __name__ == "__main__":
    main()
