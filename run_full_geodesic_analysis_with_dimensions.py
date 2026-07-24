#!/usr/bin/env python3
"""
Full Geodesic Solver Analysis with Per-Dimension Breakdown

This script runs the complete geodesic solver analysis on all 386 MSAs
and saves per-dimension geodesic efficiency values (age, income, race)
in addition to the aggregate metric.

Key differences from the original:
- Computes geodesic efficiency for each demographic dimension separately
- Saves η_age, η_income, η_race as separate columns
- Helps identify which dimension dominates demographic change
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys
import warnings
import time
from typing import Dict, List, Tuple, Optional

# Add src to path
HOME = Path.home()
sys.path.insert(0, str(HOME / 'DissipativeUrbanism/src'))

from geometry.demographic_manifold import DemographicManifold
from geometry.geodesic_validation import GeodesicValidator, GeodesicTestResult
from geometry.fisher_metric import FisherMetric
from analysis.geodesic_framework import compute_geodesic_efficiency_all_dimensions


# Paths
BASE_DIR = HOME / 'DissipativeUrbanism'
DATA_DIR = BASE_DIR / 'results/data'
OUTPUT_DIR = BASE_DIR / 'results/geodesic_solver_full'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_probability_data() -> Dict[int, Dict]:
    """
    Load raw probability distributions for age, income, and race.
    
    Returns:
        Dictionary mapping msa_code to:
        {
            'msa_name': str,
            'age_probs': np.array of shape (T, 18),
            'income_probs': np.array of shape (T, 10),
            'race_probs': np.array of shape (T, 7),
            'years': list of years
        }
    """
    print("Loading raw probability data...")
    raw_file = DATA_DIR / 'msa_demographics_raw_annual.csv'
    df = pd.read_csv(raw_file)
    
    print(f"  Loaded {len(df)} records for {df['msa_code'].nunique()} MSAs")
    
    # Age cohort columns (18 cohorts)
    age_cols = ['age_0_4', 'age_5_9', 'age_10_14', 'age_15_17', 'age_18_19',
                'age_20_24', 'age_25_29', 'age_30_34', 'age_35_44', 'age_45_54',
                'age_55_59', 'age_60_64', 'age_65_74', 'age_75_84', 'age_85_plus']
    
    # Race columns (7 categories)
    race_cols = ['race_white', 'race_black', 'race_asian', 'race_aian', 
                 'race_nhpi', 'race_other', 'race_hispanic']
    
    # Income decile columns (10 deciles)
    income_cols = [f'income_decile_{i}' for i in range(1, 11)]
    
    msa_data = {}
    
    for msa_code, group in df.groupby('msa_code'):
        msa_code = int(msa_code)
        msa_name = group['msa_name'].iloc[0]
        group = group.sort_values('year')
        
        years = group['year'].tolist()
        
        # Extract probability distributions for each year
        age_probs_list = []
        income_probs_list = []
        race_probs_list = []
        
        for _, row in group.iterrows():
            # Age probabilities
            age_counts = np.array([row.get(col, 0) for col in age_cols], dtype=float)
            age_total = age_counts.sum()
            if age_total > 0:
                age_probs = age_counts / age_total
            else:
                age_probs = np.ones(len(age_cols)) / len(age_cols)
            age_probs_list.append(age_probs)
            
            # Income probabilities
            income_counts = np.array([row.get(col, 0) for col in income_cols], dtype=float)
            income_total = income_counts.sum()
            if income_total > 0:
                income_probs = income_counts / income_total
            else:
                income_probs = np.ones(len(income_cols)) / len(income_cols)
            income_probs_list.append(income_probs)
            
            # Race probabilities
            race_counts = np.array([row.get(col, 0) for col in race_cols], dtype=float)
            race_total = race_counts.sum()
            if race_total > 0:
                race_probs = race_counts / race_total
            else:
                race_probs = np.ones(len(race_cols)) / len(race_cols)
            race_probs_list.append(race_probs)
        
        msa_data[msa_code] = {
            'msa_name': msa_name,
            'age_probs': np.array(age_probs_list),
            'income_probs': np.array(income_probs_list),
            'race_probs': np.array(race_probs_list),
            'years': years
        }
    
    print(f"  Processed {len(msa_data)} MSAs")
    return msa_data


def run_geodesic_analysis_with_dimensions(df: pd.DataFrame, 
                                          prob_data: Dict[int, Dict],
                                          n_permutations: int = 100) -> pd.DataFrame:
    """
    Run full geodesic solver analysis with per-dimension breakdown.
    
    Args:
        df: DataFrame with theta coordinates
        prob_data: Dictionary with probability distributions per MSA
        n_permutations: Number of permutations for null model testing
        
    Returns:
        DataFrame with geodesic efficiency results including per-dimension values
    """
    print("\n" + "="*70)
    print("RUNNING FULL GEODESIC SOLVER ANALYSIS WITH DIMENSIONS")
    print("="*70)
    
    # Create manifold
    print("\nConstructing demographic manifold...")
    manifold = DemographicManifold(df)
    print(f"  MSAs: {len(manifold.msa_codes)}")
    print(f"  Years: {manifold.years[0]} - {manifold.years[-1]}")
    
    # Create validator
    validator = GeodesicValidator(manifold)
    
    # Process all MSAs
    results = []
    msa_codes = manifold.msa_codes
    n_msas = len(msa_codes)
    
    print(f"\nProcessing {n_msas} MSAs with {n_permutations} permutations each...")
    print("(This may take significant time due to geodesic solving)\n")
    
    start_time = time.time()
    
    for idx, msa_code in enumerate(msa_codes):
        msa_code = int(msa_code)
        msa_name = manifold.msa_names.get(msa_code, 'Unknown')
        
        # Progress
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_msas - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{n_msas}] {msa_name[:40]:40s} "
                  f"(rate: {rate:.2f} MSA/s, ETA: {eta/60:.1f} min)")
        
        try:
            # Run geodesic hypothesis test (aggregate)
            result = validator.test_geodesic_hypothesis(
                msa_code=msa_code,
                null_model='randomized',
                n_permutations=n_permutations
            )
            
            # Compute per-dimension geodesic efficiency
            if msa_code in prob_data:
                prob = prob_data[msa_code]
                mean_eta, eta_by_dim = compute_geodesic_efficiency_all_dimensions(
                    prob['age_probs'],
                    prob['income_probs'],
                    prob['race_probs']
                )
                eta_age = eta_by_dim['age']
                eta_income = eta_by_dim['income']
                eta_race = eta_by_dim['race']
            else:
                eta_age = np.nan
                eta_income = np.nan
                eta_race = np.nan
            
            results.append({
                'msa_code': result.msa_code,
                'msa_name': result.msa_name,
                'geodesic_efficiency': result.geodesic_efficiency,
                'geodesic_efficiency_age': eta_age,
                'geodesic_efficiency_income': eta_income,
                'geodesic_efficiency_race': eta_race,
                'p_value': result.p_value,
                'is_geodesic': result.is_geodesic,
                'geodesic_deviation': result.geodesic_deviation,
                'actual_path_length': result.actual_path_length,
                'geodesic_distance': result.geodesic_distance
            })
            
        except Exception as e:
            warnings.warn(f"Failed to analyze MSA {msa_code}: {e}")
            results.append({
                'msa_code': msa_code,
                'msa_name': msa_name,
                'geodesic_efficiency': np.nan,
                'geodesic_efficiency_age': np.nan,
                'geodesic_efficiency_income': np.nan,
                'geodesic_efficiency_race': np.nan,
                'p_value': np.nan,
                'is_geodesic': False,
                'geodesic_deviation': np.nan,
                'actual_path_length': np.nan,
                'geodesic_distance': np.nan,
                'error': str(e)
            })
    
    elapsed_total = time.time() - start_time
    print(f"\nCompleted {n_msas} MSAs in {elapsed_total/60:.2f} minutes")
    print(f"  Average: {elapsed_total/n_msas:.2f} seconds per MSA")
    
    return pd.DataFrame(results)


def identify_dominant_dimension(row: pd.Series) -> str:
    """Identify which demographic dimension has the highest geodesic efficiency."""
    eta_age = row.get('geodesic_efficiency_age', np.nan)
    eta_income = row.get('geodesic_efficiency_income', np.nan)
    eta_race = row.get('geodesic_efficiency_race', np.nan)
    
    etas = {
        'age': eta_age,
        'income': eta_income,
        'race': eta_race
    }
    
    # Filter out NaN values
    valid_et = {k: v for k, v in etas.items() if not np.isnan(v)}
    
    if not valid_et:
        return 'unknown'
    
    return max(valid_et, key=valid_et.get)


def generate_dimension_report(results_df: pd.DataFrame) -> str:
    """Generate report on per-dimension geodesic efficiency."""
    
    report = []
    report.append("="*80)
    report.append("PER-DIMENSION GEODESIC EFFICIENCY ANALYSIS")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    for dim in ['age', 'income', 'race']:
        col = f'geodesic_efficiency_{dim}'
        if col in results_df.columns:
            vals = results_df[col].dropna()
            report.append(f"\n{dim.upper()} DIMENSION:")
            report.append(f"  Mean η: {vals.mean():.4f}")
            report.append(f"  Std:    {vals.std():.4f}")
            report.append(f"  Min:    {vals.min():.4f}")
            report.append(f"  Max:    {vals.max():.4f}")
            report.append(f"  Median: {vals.median():.4f}")
    
    # Identify dominant dimension for each MSA
    results_df = results_df.copy()
    results_df['dominant_dimension'] = results_df.apply(identify_dominant_dimension, axis=1)
    
    report.append("\n" + "="*80)
    report.append("DOMINANT DIMENSION DISTRIBUTION")
    report.append("="*80)
    
    dim_counts = results_df['dominant_dimension'].value_counts()
    for dim, count in dim_counts.items():
        pct = 100 * count / len(results_df)
        report.append(f"  {dim.capitalize():10s}: {count:3d} MSAs ({pct:5.1f}%)")
    
    # Top 20 MSAs by overall efficiency - show their dominant dimension
    report.append("\n" + "="*80)
    report.append("TOP 20 MSAs BY OVERALL GEODESIC EFFICIENCY")
    report.append("="*80)
    report.append(f"{'Rank':<5} {'MSA Name':<50} {'η':<8} {'Dominant':<10} {'η_age':<8} {'η_inc':<8} {'η_race':<8}")
    report.append("-"*80)
    
    top20 = results_df.nlargest(20, 'geodesic_efficiency')
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        name = row['msa_name'][:48]
        eta = row['geodesic_efficiency']
        dom = row['dominant_dimension']
        eta_age = row.get('geodesic_efficiency_age', np.nan)
        eta_inc = row.get('geodesic_efficiency_income', np.nan)
        eta_race = row.get('geodesic_efficiency_race', np.nan)
        
        eta_age_str = f"{eta_age:.3f}" if not np.isnan(eta_age) else "N/A"
        eta_inc_str = f"{eta_inc:.3f}" if not np.isnan(eta_inc) else "N/A"
        eta_race_str = f"{eta_race:.3f}" if not np.isnan(eta_race) else "N/A"
        
        report.append(f"{rank:<5} {name:<50} {eta:.4f}   {dom:<10} {eta_age_str:<8} {eta_inc_str:<8} {eta_race_str:<8}")
    
    # Analysis by dimension for high-efficiency MSAs
    report.append("\n" + "="*80)
    report.append("DIMENSION ANALYSIS FOR HIGH-EFFICIENCY MSAs (η > 0.9)")
    report.append("="*80)
    
    high_eta = results_df[results_df['geodesic_efficiency'] > 0.9]
    report.append(f"\nTotal high-efficiency MSAs: {len(high_eta)}")
    
    for dim in ['age', 'income', 'race']:
        col = f'geodesic_efficiency_{dim}'
        if col in high_eta.columns:
            dim_high = high_eta[high_eta[col] > 0.9]
            report.append(f"\n{dim.capitalize()} η > 0.9: {len(dim_high)} MSAs")
            report.append(f"  Mean {dim} η: {dim_high[col].mean():.4f}")
            
            # Show top 5 for this dimension
            top5 = dim_high.nlargest(5, col)[['msa_name', col, 'geodesic_efficiency']]
            report.append(f"  Top 5 by {dim} efficiency:")
            for _, row in top5.iterrows():
                report.append(f"    {row['msa_name'][:45]:<45} η_{dim}={row[col]:.4f} (overall η={row['geodesic_efficiency']:.4f})")
    
    return "\n".join(report)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("FULL GEODESIC SOLVER ANALYSIS WITH PER-DIMENSION BREAKDOWN")
    print("Processing all MSAs with numerical geodesic integration")
    print("="*80)
    
    # Load theta coordinate data
    print("\nLoading theta coordinate data...")
    theta_file = DATA_DIR / 'msa_demographics_theta.csv'
    if theta_file.exists():
        df = pd.read_csv(theta_file)
    else:
        # Fallback: process from raw data
        from run_full_geodesic_analysis import load_and_process_raw_data
        df = load_and_process_raw_data()
    
    # Load raw probability data
    prob_data = load_raw_probability_data()
    
    # Run geodesic analysis with dimensions
    results_df = run_geodesic_analysis_with_dimensions(df, prob_data, n_permutations=100)
    
    # Add dominant dimension column
    results_df['dominant_dimension'] = results_df.apply(identify_dominant_dimension, axis=1)
    
    # Save main results
    output_file = OUTPUT_DIR / 'msa_geodesic_efficiency_with_dimensions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # Generate and save dimension report
    report = generate_dimension_report(results_df)
    report_file = OUTPUT_DIR / 'dimension_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Saved dimension report to: {report_file}")
    
    # Print report to console
    print("\n" + report)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - msa_geodesic_efficiency_with_dimensions.csv: Main results with per-dimension η")
    print(f"  - dimension_analysis_report.txt: Detailed dimension analysis")


if __name__ == '__main__':
    main()
