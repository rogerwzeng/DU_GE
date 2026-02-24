#!/usr/bin/env python3
"""
Full Geodesic Solver Analysis for All MSAs

This script runs the complete geodesic solver analysis on all 386 MSAs
using the GeodesicValidator class from geodesic_validation.py.

Key differences from Fisher-Rao approximation:
- Uses numerical geodesic integration (shooting method)
- More accurate but computationally slower
- Computes actual geodesic paths between endpoints
- Full Riemannian geometry framework
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

# Add src to path
sys.path.insert(0, '/home/roger/DissipativeUrbanism/src')

from geometry.demographic_manifold import DemographicManifold
from geometry.geodesic_validation import GeodesicValidator, GeodesicTestResult
from geometry.fisher_metric import FisherMetric


# Paths
DATA_DIR = Path('/home/roger/DissipativeUrbanism/results/data')
OUTPUT_DIR = Path('/home/roger/DissipativeUrbanism/results/geodesic_solver_full')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_and_process_raw_data() -> pd.DataFrame:
    """
    Load raw demographic data and compute theta coordinates.
    
    Returns DataFrame with columns:
    - msa_code, msa_name, year
    - population_density, age_entropy, income_gini, diversity_shannon
    """
    print("Loading raw demographic data...")
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
    
    results = []
    
    for (msa_code, msa_name), group in df.groupby(['msa_code', 'msa_name']):
        for _, row in group.iterrows():
            year = int(row['year'])
            
            # Compute age entropy
            age_counts = np.array([row.get(col, 0) for col in age_cols], dtype=float)
            age_entropy = compute_age_entropy(age_counts)
            
            # Compute income Gini
            income_counts = np.array([row.get(col, 0) for col in income_cols], dtype=float)
            income_gini = compute_gini(income_counts)
            
            # Compute diversity (Shannon index)
            race_counts = np.array([row.get(col, 0) for col in race_cols], dtype=float)
            diversity_shannon = compute_diversity_index(race_counts)
            
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


def run_geodesic_analysis(df: pd.DataFrame, n_permutations: int = 100) -> pd.DataFrame:
    """
    Run full geodesic solver analysis on all MSAs.
    
    Args:
        df: DataFrame with theta coordinates
        n_permutations: Number of permutations for null model testing
        
    Returns:
        DataFrame with geodesic efficiency results
    """
    print("\n" + "="*70)
    print("RUNNING FULL GEODESIC SOLVER ANALYSIS")
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


def load_fisher_rao_results() -> pd.DataFrame:
    """Load Fisher-Rao approximation results for comparison."""
    fisher_file = Path('/home/roger/DissipativeUrbanism/results/analysis_real/msa_metrics_all_386.csv')
    if not fisher_file.exists():
        print(f"Warning: Fisher-Rao results not found at {fisher_file}")
        return None
    
    df = pd.read_csv(fisher_file)
    return df[['msa_code', 'msa_name', 'geodesic_efficiency', 
               'geodesic_efficiency_age', 'geodesic_efficiency_income', 
               'geodesic_efficiency_race', 'null_model_p_value', 
               'null_model_significant']]


def generate_comparison_report(full_solver_df: pd.DataFrame, 
                               fisher_rao_df: pd.DataFrame) -> str:
    """Generate comparison report between full solver and Fisher-Rao."""
    
    report = []
    report.append("="*80)
    report.append("FULL GEODESIC SOLVER vs FISHER-RAO APPROXIMATION")
    report.append("Comparison Report")
    report.append("="*80)
    report.append("")
    
    # Full solver statistics
    report.append("FULL SOLVER RESULTS")
    report.append("-"*40)
    ge_vals = full_solver_df['geodesic_efficiency'].dropna()
    report.append(f"  N MSAs analyzed: {len(ge_vals)}")
    report.append(f"  Mean geodesic efficiency: {ge_vals.mean():.4f}")
    report.append(f"  Std dev: {ge_vals.std():.4f}")
    report.append(f"  Min: {ge_vals.min():.4f}")
    report.append(f"  Max: {ge_vals.max():.4f}")
    report.append(f"  Median: {ge_vals.median():.4f}")
    
    # Count geodesic MSAs
    n_geodesic = full_solver_df['is_geodesic'].sum()
    report.append(f"\n  MSAs classified as geodesic (p<0.05): {n_geodesic} ({100*n_geodesic/len(full_solver_df):.1f}%)")
    
    # Comparison with Fisher-Rao
    if fisher_rao_df is not None:
        report.append("\n" + "="*80)
        report.append("FISHER-RAO APPROXIMATION RESULTS")
        report.append("-"*40)
        fr_vals = fisher_rao_df['geodesic_efficiency'].dropna()
        report.append(f"  N MSAs: {len(fr_vals)}")
        report.append(f"  Mean geodesic efficiency: {fr_vals.mean():.4f}")
        report.append(f"  Std dev: {fr_vals.std():.4f}")
        report.append(f"  Min: {fr_vals.min():.4f}")
        report.append(f"  Max: {fr_vals.max():.4f}")
        report.append(f"  Median: {fr_vals.median():.4f}")
        
        # Merge for correlation
        merged = full_solver_df.merge(fisher_rao_df, on='msa_code', suffixes=('_full', '_fra'))
        
        if len(merged) > 0:
            report.append("\n" + "="*80)
            report.append("COMPARISON: FULL SOLVER vs FISHER-RAO")
            report.append("-"*40)
            
            # Correlation
            valid = merged[['geodesic_efficiency_full', 'geodesic_efficiency_fra']].dropna()
            if len(valid) > 10:
                corr, pval = stats.pearsonr(valid['geodesic_efficiency_full'], 
                                           valid['geodesic_efficiency_fra'])
                report.append(f"  Pearson correlation: {corr:.4f} (p={pval:.4e})")
                
                # Spearman rank correlation
                scorr, spval = stats.spearmanr(valid['geodesic_efficiency_full'], 
                                               valid['geodesic_efficiency_fra'])
                report.append(f"  Spearman rank correlation: {scorr:.4f} (p={spval:.4e})")
            
            # Mean absolute difference
            valid['abs_diff'] = np.abs(valid['geodesic_efficiency_full'] - valid['geodesic_efficiency_fra'])
            mad = valid['abs_diff'].mean()
            report.append(f"\n  Mean absolute difference: {mad:.4f}")
            report.append(f"  Std dev of difference: {valid['abs_diff'].std():.4f}")
            
            # Bias
            valid['diff'] = valid['geodesic_efficiency_full'] - valid['geodesic_efficiency_fra']
            bias = valid['diff'].mean()
            report.append(f"  Bias (Full - Fisher-Rao): {bias:.4f}")
            report.append(f"    Positive = Full solver gives higher efficiency")
            report.append(f"    Negative = Full solver gives lower efficiency")
            
            # Systematic differences by range
            report.append("\n  Systematic differences by efficiency range:")
            for threshold in [0.1, 0.3, 0.5, 0.7]:
                subset = valid[valid['geodesic_efficiency_fra'] >= threshold]
                if len(subset) > 5:
                    mean_diff = subset['diff'].mean()
                    report.append(f"    Efficiency >= {threshold}: bias = {mean_diff:.4f} (n={len(subset)})")
    
    # Summary statistics table
    report.append("\n" + "="*80)
    report.append("SUMMARY STATISTICS TABLE")
    report.append("-"*40)
    report.append(f"{'Statistic':<25} {'Full Solver':>15} {'Fisher-Rao':>15}")
    report.append("-"*60)
    report.append(f"{'N MSAs':<25} {len(ge_vals):>15} {len(fr_vals) if fisher_rao_df is not None else 'N/A':>15}")
    report.append(f"{'Mean':<25} {ge_vals.mean():>15.4f} {fr_vals.mean() if fisher_rao_df is not None else float('nan'):>15.4f}")
    report.append(f"{'Std Dev':<25} {ge_vals.std():>15.4f} {fr_vals.std() if fisher_rao_df is not None else float('nan'):>15.4f}")
    report.append(f"{'Min':<25} {ge_vals.min():>15.4f} {fr_vals.min() if fisher_rao_df is not None else float('nan'):>15.4f}")
    report.append(f"{'Max':<25} {ge_vals.max():>15.4f} {fr_vals.max() if fisher_rao_df is not None else float('nan'):>15.4f}")
    report.append(f"{'Median':<25} {ge_vals.median():>15.4f} {fr_vals.median() if fisher_rao_df is not None else float('nan'):>15.4f}")
    
    # Technical notes
    report.append("\n" + "="*80)
    report.append("TECHNICAL NOTES")
    report.append("-"*40)
    report.append("""
Full Geodesic Solver:
- Uses numerical integration (shooting method) to solve geodesic equations
- Computes actual geodesic paths on the Fisher information manifold
- Geodesic efficiency = D_geodesic / D_actual (ratio of geodesic distance to actual path length)
- More accurate but computationally intensive

Fisher-Rao Approximation:
- Uses closed-form Fisher-Rao distance formula
- Approximates geodesic distance directly without path integration
- Faster computation but may differ from true geodesic distance
- Used for large-scale screening analysis

Key Differences:
1. Full solver explicitly finds geodesic curves; Fisher-Rao uses metric distance
2. Full solver handles manifold curvature more accurately
3. Fisher-Rao is a first-order approximation
4. Computational cost: Full solver ~100x slower per MSA
""")
    
    return "\n".join(report)


def save_summary_comparison(full_solver_df: pd.DataFrame, 
                           fisher_rao_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary comparison CSV."""
    
    summary = {
        'metric': [],
        'full_solver': [],
        'fisher_rao': [],
        'difference': []
    }
    
    # Compute statistics for full solver
    ge_full = full_solver_df['geodesic_efficiency'].dropna()
    
    metrics = [
        ('n_msas', len),
        ('mean', lambda x: x.mean()),
        ('std', lambda x: x.std()),
        ('min', lambda x: x.min()),
        ('max', lambda x: x.max()),
        ('median', lambda x: x.median()),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
    ]
    
    for name, func in metrics:
        summary['metric'].append(name)
        summary['full_solver'].append(func(ge_full))
        
        if fisher_rao_df is not None:
            ge_fra = fisher_rao_df['geodesic_efficiency'].dropna()
            summary['fisher_rao'].append(func(ge_fra))
            summary['difference'].append(func(ge_full) - func(ge_fra))
        else:
            summary['fisher_rao'].append(np.nan)
            summary['difference'].append(np.nan)
    
    return pd.DataFrame(summary)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("FULL GEODESIC SOLVER ANALYSIS")
    print("Processing all MSAs with numerical geodesic integration")
    print("="*80)
    
    # Load and process raw data
    df = load_and_process_raw_data()
    
    # Run geodesic analysis
    results_df = run_geodesic_analysis(df, n_permutations=100)
    
    # Save main results
    output_file = OUTPUT_DIR / 'msa_geodesic_efficiency_full_solver.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # Load Fisher-Rao results for comparison
    fisher_rao_df = load_fisher_rao_results()
    
    # Generate and save comparison report
    report = generate_comparison_report(results_df, fisher_rao_df)
    report_file = OUTPUT_DIR / 'comparison_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Saved comparison report to: {report_file}")
    
    # Save summary comparison CSV
    summary_df = save_summary_comparison(results_df, fisher_rao_df)
    summary_file = OUTPUT_DIR / 'summary_comparison.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary comparison to: {summary_file}")
    
    # Print report to console
    print("\n" + report)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - msa_geodesic_efficiency_full_solver.csv: Main results")
    print(f"  - summary_comparison.csv: Statistical comparison")
    print(f"  - comparison_report.txt: Detailed report")


if __name__ == '__main__':
    main()
