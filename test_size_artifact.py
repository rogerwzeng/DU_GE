#!/usr/bin/env python3
"""
Test for Population Size Artifact in η Stability

The concern: Large cities appear more stable simply because 
absolute changes are diluted across more people.

This script tests:
1. Is η stability just a function of population size?
2. Do we see proportional vs. absolute change effects?
3. Can we partial out population effects?

Usage:
    python test_size_artifact.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
FIGURES_DIR = RESULTS_DIR / "figures"


def load_data():
    """Load data with absolute and proportional changes."""
    df = pd.read_csv(RESULTS_DIR / "panel_complete.csv")
    df['cbsa_code'] = df['cbsa_code'].astype(str)
    df = df.sort_values(['cbsa_code', 'year'])
    
    # Compute year-to-year changes
    df['eta_change_abs'] = df.groupby('cbsa_code')['eta'].diff().abs()
    df['eta_change_prop'] = df.groupby('cbsa_code')['eta'].pct_change().abs()
    
    df['sigma_econ_change_abs'] = df.groupby('cbsa_code')['sigma_econ'].diff().abs()
    df['sigma_econ_change_prop'] = df.groupby('cbsa_code')['sigma_econ'].pct_change().abs()
    
    return df


def analyze_population_effects(df):
    """Test whether population size drives apparent stability."""
    print("="*70)
    print("TESTING FOR POPULATION SIZE ARTIFACT")
    print("="*70)
    
    # Compute MSA-level statistics
    print("\n1. Computing MSA-level change statistics...")
    
    msa_stats = df.groupby('cbsa_code').agg({
        'population': 'mean',
        'eta': ['mean', 'std'],
        'eta_change_abs': 'mean',
        'eta_change_prop': 'mean',
        'cbsa_title': 'first'
    }).reset_index()
    
    msa_stats.columns = ['cbsa_code', 'population', 'eta_mean', 'eta_std',
                        'eta_change_abs', 'eta_change_prop', 'cbsa_title']
    
    # Coefficient of variation
    msa_stats['eta_cv'] = msa_stats['eta_std'] / msa_stats['eta_mean']
    
    print(f"   Computed for {len(msa_stats)} MSAs")
    
    # Test 1: Is absolute change related to population?
    print("\n" + "-"*70)
    print("TEST 1: Absolute change vs. Population size")
    print("-"*70)
    
    pop = msa_stats['population'].values
    abs_change = msa_stats['eta_change_abs'].fillna(0).values
    prop_change = msa_stats['eta_change_prop'].fillna(0).values
    cv = msa_stats['eta_cv'].fillna(0).values
    
    # Log-log regression
    def log_fit(x, y, label):
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        lx = np.log(x[valid])
        ly = np.log(y[valid])
        if len(lx) < 10:
            return None, None, None
        slope, intercept, r, p, _ = stats.linregress(lx, ly)
        return slope, r**2, p
    
    beta_abs, r2_abs, p_abs = log_fit(pop, abs_change, 'abs_change')
    beta_prop, r2_prop, p_prop = log_fit(pop, prop_change, 'prop_change')
    beta_cv, r2_cv, p_cv = log_fit(pop, cv, 'cv')
    
    print(f"\nAbsolute change scaling: β = {beta_abs:.4f}, R² = {r2_abs:.4f}, p = {p_abs:.4f}")
    print(f"Proportional change scaling: β = {beta_prop:.4f}, R² = {r2_prop:.4f}, p = {p_prop:.4f}")
    print(f"CV scaling: β = {beta_cv:.4f}, R² = {r2_cv:.4f}, p = {p_cv:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if beta_abs and beta_abs < 0:
        print("  → Larger cities have SMALLER absolute changes in η")
    if beta_prop and beta_prop > -0.05 and beta_prop < 0.05:
        print("  → Proportional changes in η are scale-INVARIANT")
    if beta_cv and beta_cv < 0:
        print("  → Larger cities have LOWER variability (CV)")
    
    # Test 2: Partial correlation controlling for population
    print("\n" + "-"*70)
    print("TEST 2: Partial correlations controlling for log(population)")
    print("-"*70)
    
    log_pop = np.log(msa_stats['population'].values)
    
    # Residualize each variable
    def residualize(y, x):
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return y - (intercept + slope * x)
    
    eta_std_resid = residualize(msa_stats['eta_std'].fillna(0).values, log_pop)
    eta_cv_resid = residualize(cv, log_pop)
    eta_mean_resid = residualize(msa_stats['eta_mean'].fillna(0).values, log_pop)
    
    # Now correlate residuals
    corr_std_mean = stats.pearsonr(eta_std_resid, eta_mean_resid)
    corr_cv_mean = stats.pearsonr(eta_cv_resid, eta_mean_resid)
    
    print(f"\nAfter controlling for log(pop):")
    print(f"  Corr(η_std, η_mean): r = {corr_std_mean[0]:.4f}, p = {corr_std_mean[1]:.4f}")
    print(f"  Corr(η_CV, η_mean): r = {corr_cv_mean[0]:.4f}, p = {corr_cv_mean[1]:.4f}")
    
    # Test 3: Compare small vs. large cities directly
    print("\n" + "-"*70)
    print("TEST 3: Small vs. Large MSAs (median split)")
    print("-"*70)
    
    median_pop = msa_stats['population'].median()
    small_msas = msa_stats[msa_stats['population'] < median_pop]
    large_msas = msa_stats[msa_stats['population'] >= median_pop]
    
    print(f"\nSmall MSAs (n={len(small_msas)}, pop < {median_pop:,.0f}):")
    print(f"  Mean η: {small_msas['eta_mean'].mean():.4f} ± {small_msas['eta_mean'].std():.4f}")
    print(f"  Mean η_std: {small_msas['eta_std'].mean():.4f}")
    print(f"  Mean η_CV: {small_msas['eta_cv'].mean():.4f}")
    print(f"  Mean absolute change: {small_msas['eta_change_abs'].mean():.4f}")
    print(f"  Mean proportional change: {small_msas['eta_change_prop'].mean():.4f}")
    
    print(f"\nLarge MSAs (n={len(large_msas)}, pop >= {median_pop:,.0f}):")
    print(f"  Mean η: {large_msas['eta_mean'].mean():.4f} ± {large_msas['eta_mean'].std():.4f}")
    print(f"  Mean η_std: {large_msas['eta_std'].mean():.4f}")
    print(f"  Mean η_CV: {large_msas['eta_cv'].mean():.4f}")
    print(f"  Mean absolute change: {large_msas['eta_change_abs'].mean():.4f}")
    print(f"  Mean proportional change: {large_msas['eta_change_prop'].mean():.4f}")
    
    # T-tests
    try:
        from scipy.stats import ttest_ind
    except:
        pass
    
    t_eta_mean, p_eta_mean = ttest_ind(small_msas['eta_mean'], large_msas['eta_mean'])
    t_eta_std, p_eta_std = ttest_ind(small_msas['eta_std'], large_msas['eta_std'])
    t_eta_cv, p_eta_cv = ttest_ind(small_msas['eta_cv'], large_msas['eta_cv'])
    
    print(f"\nT-test results (small vs. large):")
    print(f"  η mean: t = {t_eta_mean:.3f}, p = {p_eta_mean:.4f}")
    print(f"  η std: t = {t_eta_std:.3f}, p = {p_eta_std:.4f}")
    print(f"  η CV: t = {t_eta_cv:.3f}, p = {p_eta_cv:.4f}")
    
    # Test 4: Fisher information interpretation
    print("\n" + "-"*70)
    print("TEST 4: Statistical Efficiency Argument")
    print("-"*70)
    print("\nIf η is an estimate based on demographic counts, larger cities")
    print("should have lower standard error due to larger sample size.")
    print("\nExpected SE scaling: SE ~ 1/√N")
    print("Observed η_std scaling should be compared to 1/√N...")
    
    # Compute theoretical SE scaling
    theoretical_se = 1 / np.sqrt(pop)
    observed_cv = cv
    
    # Correlation between observed and theoretical
    valid = (pop > 0) & np.isfinite(theoretical_se) & np.isfinite(observed_cv)
    if valid.sum() > 10:
        corr_theo_obs = stats.pearsonr(np.log(theoretical_se[valid]), 
                                       np.log(observed_cv[valid]))
        print(f"\nCorr(log(theoretical_SE), log(observed_CV)): r = {corr_theo_obs[0]:.4f}")
        
        if abs(corr_theo_obs[0]) > 0.5:
            print("  → OBSERVED SCALING CONSISTENT WITH STATISTICAL ARTIFACT")
        else:
            print("  → OBSERVED SCALING DIFFERENT FROM STATISTICAL ARTIFACT")
    
    return msa_stats


def create_diagnostic_plots(msa_stats):
    """Create plots to visualize the artifact test."""
    print("\n" + "="*70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Testing for Population Size Artifact in η Stability', 
                 fontsize=16, fontweight='bold')
    
    pop = msa_stats['population'].values / 1e6  # Millions
    
    # Row 1: Raw relationships
    # Plot 1: η mean vs. population
    ax = axes[0, 0]
    ax.scatter(pop, msa_stats['eta_mean'], alpha=0.5, s=30)
    ax.set_xscale('log')
    ax.set_xlabel('Population (millions)')
    ax.set_ylabel('Mean η')
    ax.set_title('Mean η vs. Population')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    valid = (pop > 0) & np.isfinite(msa_stats['eta_mean'])
    z = np.polyfit(np.log(pop[valid]), msa_stats['eta_mean'][valid], 1)
    p = np.poly1d(z)
    x_line = np.logspace(np.log10(pop.min()), np.log10(pop.max()), 100)
    ax.plot(x_line, p(np.log(x_line)), 'r--', linewidth=2)
    
    # Plot 2: η std vs. population
    ax = axes[0, 1]
    ax.scatter(pop, msa_stats['eta_std'], alpha=0.5, s=30)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Population (millions)')
    ax.set_ylabel('Std(η)')
    ax.set_title('Std(η) vs. Population')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: η CV vs. population
    ax = axes[0, 2]
    cv = msa_stats['eta_cv'].fillna(0).values
    ax.scatter(pop, cv, alpha=0.5, s=30)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Population (millions)')
    ax.set_ylabel('CV(η)')
    ax.set_title('CV(η) vs. Population')
    ax.grid(True, alpha=0.3)
    
    # Theoretical line: CV ~ 1/√N
    theoretical_cv = 1 / np.sqrt(pop * 1e6) * 100  # Scale for visibility
    ax.plot(pop, theoretical_cv, 'r--', linewidth=2, label='1/√N scaling')
    ax.legend()
    
    # Row 2: Change metrics
    # Plot 4: Absolute change vs. population
    ax = axes[1, 0]
    abs_change = msa_stats['eta_change_abs'].fillna(0).values
    ax.scatter(pop, abs_change, alpha=0.5, s=30, color='orange')
    ax.set_xscale('log')
    ax.set_xlabel('Population (millions)')
    ax.set_ylabel('Mean Absolute Change in η')
    ax.set_title('Absolute Change vs. Population')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Proportional change vs. population
    ax = axes[1, 1]
    prop_change = msa_stats['eta_change_prop'].fillna(0).values
    ax.scatter(pop, prop_change, alpha=0.5, s=30, color='green')
    ax.set_xscale('log')
    ax.set_xlabel('Population (millions)')
    ax.set_ylabel('Mean Proportional Change in η')
    ax.set_title('Proportional Change vs. Population')
    ax.grid(True, alpha=0.3)
    ax.axhline(np.median(prop_change), color='red', linestyle='--', 
              label=f'Median = {np.median(prop_change):.4f}')
    ax.legend()
    
    # Plot 6: Binned comparison
    ax = axes[1, 2]
    
    # Create population bins
    log_pop = np.log10(msa_stats['population'])
    bins = np.percentile(log_pop, [0, 25, 50, 75, 100])
    labels = ['Q1\n(Smallest)', 'Q2', 'Q3', 'Q4\n(Largest)']
    
    msa_stats['pop_quartile'] = pd.cut(log_pop, bins=bins, labels=labels, include_lowest=True)
    
    quartile_means = msa_stats.groupby('pop_quartile').agg({
        'eta_mean': 'mean',
        'eta_std': 'mean',
        'eta_cv': 'mean',
        'eta_change_abs': 'mean',
        'eta_change_prop': 'mean'
    })
    
    x = np.arange(len(labels))
    width = 0.15
    
    ax.bar(x - 2*width, quartile_means['eta_mean'], width, label='Mean η', alpha=0.8)
    ax.bar(x - width, quartile_means['eta_std'], width, label='Std η', alpha=0.8)
    ax.bar(x, quartile_means['eta_cv'], width, label='CV η', alpha=0.8)
    ax.bar(x + width, quartile_means['eta_change_abs'] * 10, width, label='Abs Change (×10)', alpha=0.8)
    ax.bar(x + 2*width, quartile_means['eta_change_prop'], width, label='Prop Change', alpha=0.8)
    
    ax.set_xlabel('Population Quartile')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics by Population Quartile')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'size_artifact_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: size_artifact_test.png")
    
    # Print quartile table
    print("\n" + "-"*70)
    print("QUARTILE COMPARISON:")
    print("-"*70)
    print(quartile_means.round(4))


def main():
    """Main analysis."""
    print("="*70)
    print("TESTING FOR POPULATION SIZE ARTIFACT IN η STABILITY")
    print("="*70)
    
    df = load_data()
    msa_stats = analyze_population_effects(df)
    create_diagnostic_plots(msa_stats)
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("\nKey question: Is η stability in large cities just a statistical artifact")
    print("of having more people (slower proportional change)?")
    print("\nLook for:")
    print("  - If proportional change ~ scale-invariant → ARTIFACT")
    print("  - If absolute change decreases with size → ARTIFACT")
    print("  - If η_std tracks 1/√N → STATISTICAL NOISE EFFECT")
    print("  - If differences persist after residualizing → REAL EFFECT")


if __name__ == "__main__":
    main()
