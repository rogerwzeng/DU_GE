#!/usr/bin/env python3
"""
Geodesic Efficiency Analysis - Excluding Puerto Rico MSAs

This script runs the complete geodesic solver analysis excluding the 4 Puerto Rico MSAs
due to data quality concerns (limited observations):
- 11640: Arecibo, PR (4 observations)
- 32420: Mayagüez, PR (11 observations)
- 38660: Ponce, PR (15 observations)
- 41980: San Juan-Bayamón-Caguas, PR (16 observations)

The analysis follows the same methodology as the full analysis but on 381 MSAs instead of 385.

Key Components:
- Bayesian smoothing with Jeffreys prior for probability distributions
- Full numerical geodesic solver (shooting method)
- Null model tests (1000 surrogates per MSA)
- Geodesic efficiency (η) for 3 demographic dimensions
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from pathlib import Path
import sys
import warnings
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

# Add paths
HOME = Path.home()
BASE_DIR = HOME / 'DissipativeUrbanism'
sys.path.insert(0, str(BASE_DIR / 'src'))
sys.path.insert(0, str(BASE_DIR / 'geodesic_efficiency/src'))

from geometry.demographic_manifold import DemographicManifold
from geometry.geodesic_validation import GeodesicValidator, GeodesicTestResult
from geometry.fisher_metric import FisherMetric


# Puerto Rico MSAs to exclude (data quality concerns)
PR_MSA_CODES = {11640, 32420, 38660, 41980}

# Paths
DATA_DIR = BASE_DIR / 'results/data'
ENTROPY_DATA_FILE = BASE_DIR / 'results/thermodynamics/official_msa_entropy_production.csv'
OUTPUT_DIR = BASE_DIR / 'results/thermodynamics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bayesian_smooth(counts: np.ndarray, prior: str = 'jeffreys') -> np.ndarray:
    """
    Apply Bayesian smoothing with prior to count data.
    
    Jeffreys prior: alpha = 0.5 for each category (non-informative)
    Laplace prior: alpha = 1 for each category (uniform)
    
    Args:
        counts: Raw count data
        prior: 'jeffreys' or 'laplace'
        
    Returns:
        Smoothed probability distribution
    """
    counts = np.array(counts, dtype=float)
    n_categories = len(counts)
    
    if prior == 'jeffreys':
        alpha = 0.5
    elif prior == 'laplace':
        alpha = 1.0
    else:
        alpha = 0.0
    
    # Add prior pseudocounts
    smoothed_counts = counts + alpha
    
    # Normalize to get probabilities
    total = smoothed_counts.sum()
    if total > 0:
        return smoothed_counts / total
    else:
        return np.ones(n_categories) / n_categories


def compute_entropy(prob_dist: np.ndarray) -> float:
    """Compute Shannon entropy from probability distribution."""
    prob_dist = np.array(prob_dist)
    # Normalize
    prob_dist = prob_dist / prob_dist.sum() if prob_dist.sum() > 0 else prob_dist
    # Remove zeros
    prob_dist = prob_dist[prob_dist > 0]
    if len(prob_dist) == 0:
        return 0.0
    return -np.sum(prob_dist * np.log2(prob_dist))


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient from distribution."""
    values = np.array(values, dtype=float)
    values = values[values > 0]  # Remove zeros
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    # Gini formula
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_diversity_index(race_counts: np.ndarray) -> float:
    """Compute racial/ethnic diversity (Shannon index)."""
    race_counts = np.array(race_counts, dtype=float)
    if race_counts.sum() == 0:
        return 0.0
    probs = race_counts / race_counts.sum()
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log(probs))


def compute_age_entropy(age_counts: np.ndarray) -> float:
    """Compute age distribution entropy."""
    return compute_entropy(age_counts)


def load_and_process_raw_data(exclude_pr: bool = True) -> pd.DataFrame:
    """
    Load raw demographic data and compute theta coordinates.
    
    Args:
        exclude_pr: If True, exclude Puerto Rico MSAs
    
    Returns:
        DataFrame with columns:
        - msa_code, msa_name, year
        - population_density, age_entropy, income_gini, diversity_shannon
    """
    print("Loading raw demographic data...")
    raw_file = DATA_DIR / 'msa_demographics_raw_annual.csv'
    df = pd.read_csv(raw_file)
    
    # Exclude Puerto Rico MSAs if requested
    if exclude_pr:
        n_before = df['msa_code'].nunique()
        df = df[~df['msa_code'].isin(PR_MSA_CODES)]
        n_after = df['msa_code'].nunique()
        print(f"  Excluded {n_before - n_after} Puerto Rico MSAs")
        print(f"  Remaining MSAs: {n_after}")
    
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
    
    results = []
    
    for (msa_code, msa_name), group in df.groupby(['msa_code', 'msa_name']):
        for _, row in group.iterrows():
            year = int(row['year'])
            
            # Compute age entropy with Bayesian smoothing
            age_counts = np.array([row.get(col, 0) for col in age_cols], dtype=float)
            age_probs = bayesian_smooth(age_counts, prior='jeffreys')
            age_entropy = compute_entropy(age_probs)
            
            # Compute income Gini with Bayesian smoothing
            income_counts = np.array([row.get(col, 0) for col in income_cols], dtype=float)
            income_probs = bayesian_smooth(income_counts, prior='jeffreys')
            income_gini = compute_gini(income_probs)
            
            # Compute diversity (Shannon index) with Bayesian smoothing
            race_counts = np.array([row.get(col, 0) for col in race_cols], dtype=float)
            race_probs = bayesian_smooth(race_counts, prior='jeffreys')
            diversity_shannon = compute_diversity_index(race_probs)
            
            # Population density (use population as proxy, normalized)
            total_pop = row.get('total_population', 0)
            # Simple proxy: population / 1000 for scaling
            population_density = total_pop / 1000.0 if total_pop > 0 else 0
            
            results.append({
                'msa_code': msa_code,
                'msa_name': msa_name,
                'year': year,
                'total_population': total_pop,
                'population_density': population_density,
                'age_entropy': age_entropy,
                'income_gini': income_gini,
                'diversity_shannon': diversity_shannon
            })
    
    result_df = pd.DataFrame(results)
    print(f"  Processed {len(result_df)} records")
    print(f"  MSAs: {result_df['msa_code'].nunique()}")
    print(f"  Years: {result_df['year'].min()} - {result_df['year'].max()}")
    
    return result_df


def run_geodesic_analysis(df: pd.DataFrame, n_permutations: int = 1000) -> pd.DataFrame:
    """
    Run full geodesic solver analysis on all MSAs.
    
    Args:
        df: DataFrame with theta coordinates
        n_permutations: Number of permutations for null model testing
        
    Returns:
        DataFrame with geodesic efficiency results
    """
    print("\n" + "="*70)
    print("RUNNING FULL GEODESIC SOLVER ANALYSIS (NO PUERTO RICO)")
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
        msa_name = manifold.msa_names.get(msa_code, 'Unknown')
        
        # Progress
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_msas - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{n_msas}] {msa_name[:40]:40s} "
                  f"(rate: {rate:.2f} MSA/s, ETA: {eta/60:.1f} min)")
        
        try:
            # Run geodesic hypothesis test
            result = validator.test_geodesic_hypothesis(
                msa_code=msa_code,
                null_model='randomized',
                n_permutations=n_permutations
            )
            
            results.append({
                'msa_code': result.msa_code,
                'msa_name': result.msa_name,
                'geodesic_efficiency': result.geodesic_efficiency,
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


def compute_dimensional_geodesic_efficiency(df: pd.DataFrame, n_permutations: int = 1000) -> pd.DataFrame:
    """
    Compute geodesic efficiency for each demographic dimension separately.
    
    Dimensions:
    - Age: age_entropy
    - Income: income_gini  
    - Race: diversity_shannon
    
    Uses 1D geodesic approximation for each dimension.
    """
    print("\n" + "="*70)
    print("COMPUTING DIMENSIONAL GEODESIC EFFICIENCIES")
    print("="*70)
    
    dimensions = {
        'age': 'age_entropy',
        'income': 'income_gini',
        'race': 'diversity_shannon'
    }
    
    results = []
    
    for msa_code, group in df.groupby('msa_code'):
        msa_name = group['msa_name'].iloc[0]
        
        # Sort by year
        group = group.sort_values('year')
        years = group['year'].values
        
        if len(years) < 2:
            continue
        
        msa_result = {
            'msa_code': msa_code,
            'msa_name': msa_name,
            'n_years': len(years)
        }
        
        for dim_name, col_name in dimensions.items():
            values = group[col_name].values
            
            # Actual path length (sum of absolute changes)
            actual_length = np.sum(np.abs(np.diff(values)))
            
            # Geodesic distance (direct distance from start to end)
            geodesic_dist = abs(values[-1] - values[0])
            
            # Geodesic efficiency
            if actual_length > 1e-10:
                eta = geodesic_dist / actual_length
            else:
                eta = 1.0  # No change = perfectly efficient
            
            # Null model test
            null_efficiencies = []
            for _ in range(n_permutations):
                # Randomize the trajectory (preserve endpoints)
                middle_values = values[1:-1].copy()
                if len(middle_values) > 0:
                    np.random.shuffle(middle_values)
                    randomized = np.concatenate([[values[0]], middle_values, [values[-1]]])
                else:
                    randomized = values.copy()
                
                rand_length = np.sum(np.abs(np.diff(randomized)))
                if rand_length > 1e-10:
                    rand_eta = geodesic_dist / rand_length
                else:
                    rand_eta = 1.0
                null_efficiencies.append(rand_eta)
            
            null_efficiencies = np.array(null_efficiencies)
            
            # Compute p-value (one-tailed: is observed efficiency greater than null?)
            p_value = np.mean(null_efficiencies >= eta)
            
            msa_result[f'geodesic_efficiency_{dim_name}'] = eta
            msa_result[f'p_value_{dim_name}'] = p_value
            msa_result[f'is_geodesic_{dim_name}'] = p_value < 0.05
        
        results.append(msa_result)
    
    return pd.DataFrame(results)


def generate_summary_report(full_results: pd.DataFrame, dim_results: pd.DataFrame) -> str:
    """Generate comprehensive summary report."""
    
    report = []
    report.append("="*80)
    report.append("GEODESIC EFFICIENCY ANALYSIS - EXCLUDING PUERTO RICO MSAs")
    report.append("="*80)
    report.append("")
    
    # Sample information
    report.append("SAMPLE INFORMATION")
    report.append("-"*40)
    report.append(f"Total MSAs analyzed: {len(full_results)}")
    report.append(f"Puerto Rico MSAs excluded: 4")
    report.append(f"  - 11640: Arecibo, PR")
    report.append(f"  - 32420: Mayagüez, PR")
    report.append(f"  - 38660: Ponce, PR")
    report.append(f"  - 41980: San Juan-Bayamón-Caguas, PR")
    report.append("")
    
    # Overall geodesic efficiency
    report.append("OVERALL GEODESIC EFFICIENCY")
    report.append("-"*40)
    ge_vals = full_results['geodesic_efficiency'].dropna()
    report.append(f"N MSAs with valid results: {len(ge_vals)}")
    report.append(f"Mean geodesic efficiency (η): {ge_vals.mean():.4f}")
    report.append(f"Std dev: {ge_vals.std():.4f}")
    report.append(f"Min: {ge_vals.min():.4f}")
    report.append(f"Max: {ge_vals.max():.4f}")
    report.append(f"Median: {ge_vals.median():.4f}")
    
    # Significant MSAs
    n_geodesic = full_results['is_geodesic'].sum()
    report.append(f"\nMSAs with statistically significant geodesic trajectories (p<0.05): {n_geodesic}")
    report.append(f"Percentage: {100*n_geodesic/len(full_results):.1f}%")
    report.append("")
    
    # Dimensional results
    if dim_results is not None and len(dim_results) > 0:
        report.append("DIMENSIONAL GEODESIC EFFICIENCIES")
        report.append("-"*40)
        
        for dim in ['age', 'income', 'race']:
            eta_col = f'geodesic_efficiency_{dim}'
            sig_col = f'is_geodesic_{dim}'
            
            if eta_col in dim_results.columns:
                eta_vals = dim_results[eta_col].dropna()
                n_sig = dim_results[sig_col].sum() if sig_col in dim_results.columns else 0
                
                report.append(f"\n{dim.upper()}:")
                report.append(f"  Mean η: {eta_vals.mean():.4f}")
                report.append(f"  Std: {eta_vals.std():.4f}")
                report.append(f"  Significant MSAs (p<0.05): {n_sig} ({100*n_sig/len(dim_results):.1f}%)")
    
    report.append("")
    report.append("="*80)
    report.append("COMPARISON WITH ORIGINAL ANALYSIS (385 MSAs)")
    report.append("-"*40)
    report.append("Note: Compare these results with the original analysis")
    report.append("to assess the impact of excluding Puerto Rico MSAs.")
    report.append("")
    report.append("Expected impact:")
    report.append("  - Minimal change in overall statistics (PR MSAs = 1% of sample)")
    report.append("  - Slightly reduced variance in demographic dimensions")
    report.append("  - More consistent data quality across the sample")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("GEODESIC EFFICIENCY ANALYSIS - EXCLUDING PUERTO RICO MSAs")
    print("="*80)
    print()
    print("Puerto Rico MSAs excluded:")
    print("  - 11640: Arecibo, PR (4 observations)")
    print("  - 32420: Mayagüez, PR (11 observations)")
    print("  - 38660: Ponce, PR (15 observations)")
    print("  - 41980: San Juan-Bayamón-Caguas, PR (16 observations)")
    print()
    
    # Load and process raw data (excluding PR MSAs)
    df = load_and_process_raw_data(exclude_pr=True)
    
    # Run full geodesic analysis with 1000 permutations
    results_df = run_geodesic_analysis(df, n_permutations=1000)
    
    # Save main results
    output_file = OUTPUT_DIR / 'no_pr_geodesic_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved main results to: {output_file}")
    
    # Compute dimensional geodesic efficiencies
    dim_results_df = compute_dimensional_geodesic_efficiency(df, n_permutations=1000)
    
    # Save dimensional results
    dim_output_file = OUTPUT_DIR / 'no_pr_geodesic_dimensional.csv'
    dim_results_df.to_csv(dim_output_file, index=False)
    print(f"Saved dimensional results to: {dim_output_file}")
    
    # Generate and save summary report
    report = generate_summary_report(results_df, dim_results_df)
    report_file = OUTPUT_DIR / 'no_pr_geodesic_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Saved summary report to: {report_file}")
    
    # Print report to console
    print("\n" + report)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - no_pr_geodesic_results.csv: Main geodesic efficiency results")
    print(f"  - no_pr_geodesic_dimensional.csv: Dimensional breakdown")
    print(f"  - no_pr_geodesic_report.txt: Summary report")


if __name__ == '__main__':
    main()
