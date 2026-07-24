#!/usr/bin/env python3
"""
Urban Scaling Analysis - Power Law Relationships

Tests whether three manifold metrics follow power laws with MSA population:
    Y = Y_0 * N^β
    
Where N = population, β = scaling exponent

Expected urban scaling (Bettencourt et al.):
- Superlinear (β > 1): Social/economic outputs (innovation, crime, GDP)
- Linear (β = 1): Individual-level outcomes (employment, household consumption)
- Sublinear (β < 1): Infrastructure (roads, cables)

For our metrics:
- η (efficiency): Unknown scaling
- σ_econ (flux): May scale with economic diversity (superlinear?)
- σ_mig (migration): May scale with network connectivity

Usage:
    python scaling_analysis.py

Output:
    Scaling exponents and visualization
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
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    """Load panel data with population."""
    print("Loading data...")
    df = pd.read_csv(RESULTS_DIR / "panel_complete.csv")
    df['cbsa_code'] = df['cbsa_code'].astype(str)
    
    # Filter to valid observations
    df = df[(df['eta'].notna()) & 
            (df['sigma_econ'].notna()) & 
            (df['sigma_mig'].notna()) &
            (df['population'].notna()) &
            (df['population'] > 0)]
    
    print(f"  Loaded {len(df)} observations")
    return df


def fit_power_law(x, y, metric_name):
    """Fit power law: y = a * x^b, estimate b and CI."""
    # Log-log regression: ln(y) = ln(a) + b*ln(x)
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Remove infinities/nans
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    log_x = log_x[valid]
    log_y = log_y[valid]
    
    if len(log_x) < 10:
        return None, None, None, None, None
    
    # OLS in log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # 95% CI
    ci = 1.96 * std_err
    
    # Prefactor (a = exp(intercept))
    prefactor = np.exp(intercept)
    
    return slope, prefactor, r_value**2, p_value, ci


def analyze_cross_sectional_scaling(df):
    """Analyze scaling at each time point (cross-sectional)."""
    print("\n" + "="*70)
    print("CROSS-SECTIONAL SCALING ANALYSIS")
    print("="*70)
    print("\nTesting: Y ~ Population^β for each year")
    
    results = []
    
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        
        if len(year_data) < 50:
            continue
        
        pop = year_data['population'].values
        
        row = {'year': year, 'n': len(year_data)}
        
        # η scaling
        beta, a, r2, p, ci = fit_power_law(pop, year_data['eta'].values, 'eta')
        if beta is not None:
            row.update({
                'eta_beta': beta, 'eta_r2': r2, 'eta_p': p, 'eta_ci': ci
            })
        
        # σ_econ scaling
        beta, a, r2, p, ci = fit_power_law(pop, year_data['sigma_econ'].values, 'sigma_econ')
        if beta is not None:
            row.update({
                'sigma_econ_beta': beta, 'sigma_econ_r2': r2, 'sigma_econ_p': p, 'sigma_econ_ci': ci
            })
        
        # σ_mig scaling
        beta, a, r2, p, ci = fit_power_law(pop, year_data['sigma_mig'].values, 'sigma_mig')
        if beta is not None:
            row.update({
                'sigma_mig_beta': beta, 'sigma_mig_r2': r2, 'sigma_mig_p': p, 'sigma_mig_ci': ci
            })
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\nScaling Exponents by Year:")
    print("-" * 90)
    print(f"{'Year':<6} {'η β':<8} {'η R²':<8} {'σ_econ β':<10} {'σ_econ R²':<10} {'σ_mig β':<10} {'σ_mig R²':<10}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        year = int(row['year'])
        eta_b = f"{row.get('eta_beta', 0):.3f}" if 'eta_beta' in row else "N/A"
        eta_r = f"{row.get('eta_r2', 0):.3f}" if 'eta_r2' in row else "N/A"
        se_b = f"{row.get('sigma_econ_beta', 0):.3f}" if 'sigma_econ_beta' in row else "N/A"
        se_r = f"{row.get('sigma_econ_r2', 0):.3f}" if 'sigma_econ_r2' in row else "N/A"
        sm_b = f"{row.get('sigma_mig_beta', 0):.3f}" if 'sigma_mig_beta' in row else "N/A"
        sm_r = f"{row.get('sigma_mig_r2', 0):.3f}" if 'sigma_mig_r2' in row else "N/A"
        
        print(f"{year:<6} {eta_b:<8} {eta_r:<8} {se_b:<10} {se_r:<10} {sm_b:<10} {sm_r:<10}")
    
    # Average across years
    print("\n" + "="*70)
    print("AVERAGE SCALING EXPONENTS (across years):")
    print("="*70)
    
    for metric in ['eta', 'sigma_econ', 'sigma_mig']:
        beta_col = f"{metric}_beta"
        if beta_col in results_df.columns:
            mean_beta = results_df[beta_col].mean()
            std_beta = results_df[beta_col].std()
            mean_r2 = results_df[f"{metric}_r2"].mean()
            print(f"\n{metric}:")
            print(f"  β = {mean_beta:.3f} ± {std_beta:.3f}")
            print(f"  R² = {mean_r2:.3f}")
            
            # Interpretation
            if mean_beta > 0.05:
                print(f"  → SUPERLINEAR: Larger cities have HIGHER {metric}")
            elif mean_beta < -0.05:
                print(f"  → SUBLINEAR: Larger cities have LOWER {metric}")
            else:
                print(f"  → LINEAR/SCALE-INVARIANT: {metric} independent of size")
    
    return results_df


def analyze_longitudinal_scaling(df):
    """Analyze scaling of temporal dynamics."""
    print("\n" + "="*70)
    print("LONGITUDINAL SCALING ANALYSIS")
    print("="*70)
    
    # For each MSA, compute temporal statistics
    print("\nComputing MSA-level temporal statistics...")
    
    msa_stats = df.groupby('cbsa_code').agg({
        'eta': ['mean', 'std', 'min', 'max'],
        'sigma_econ': ['mean', 'std', 'min', 'max'],
        'sigma_mig': ['mean', 'std', 'min', 'max'],
        'population': 'mean',
        'cbsa_title': 'first'
    }).reset_index()
    
    # Flatten column names
    msa_stats.columns = ['cbsa_code', 
                         'eta_mean', 'eta_std', 'eta_min', 'eta_max',
                         'sigma_econ_mean', 'sigma_econ_std', 'sigma_econ_min', 'sigma_econ_max',
                         'sigma_mig_mean', 'sigma_mig_std', 'sigma_mig_min', 'sigma_mig_max',
                         'population_mean', 'cbsa_title']
    
    # Compute coefficient of variation (CV = std/mean)
    msa_stats['eta_cv'] = msa_stats['eta_std'] / msa_stats['eta_mean']
    msa_stats['sigma_econ_cv'] = msa_stats['sigma_econ_std'] / msa_stats['sigma_econ_mean']
    msa_stats['sigma_mig_cv'] = msa_stats['sigma_mig_std'] / msa_stats['sigma_mig_mean']
    
    # Range (max - min)
    msa_stats['eta_range'] = msa_stats['eta_max'] - msa_stats['eta_min']
    msa_stats['sigma_econ_range'] = msa_stats['sigma_econ_max'] - msa_stats['sigma_econ_min']
    msa_stats['sigma_mig_range'] = msa_stats['sigma_mig_max'] - msa_stats['sigma_mig_min']
    
    print(f"  Computed statistics for {len(msa_stats)} MSAs")
    
    # Test scaling of temporal variability
    print("\n" + "-"*70)
    print("SCALING OF TEMPORAL VARIABILITY:")
    print("-"*70)
    print("\nDo larger cities have more/less variable dynamics?")
    
    pop = msa_stats['population_mean'].values
    
    for metric in ['eta', 'sigma_econ', 'sigma_mig']:
        # CV scaling
        cv_col = f"{metric}_cv"
        beta_cv, a_cv, r2_cv, p_cv, ci_cv = fit_power_law(
            pop, msa_stats[cv_col].fillna(0).values, f"{metric}_cv"
        )
        
        # Range scaling
        range_col = f"{metric}_range"
        beta_r, a_r, r2_r, p_r, ci_r = fit_power_law(
            pop, msa_stats[range_col].fillna(0).values, f"{metric}_range"
        )
        
        print(f"\n{metric}:")
        if beta_cv is not None:
            sig_cv = "***" if p_cv < 0.001 else "**" if p_cv < 0.01 else "*" if p_cv < 0.05 else ""
            print(f"  CV scaling:     β = {beta_cv:+.3f} ± {ci_cv:.3f} (R²={r2_cv:.3f}) {sig_cv}")
        if beta_r is not None:
            sig_r = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else ""
            print(f"  Range scaling:  β = {beta_r:+.3f} ± {ci_r:.3f} (R²={r2_r:.3f}) {sig_r}")
    
    return msa_stats


def create_scaling_plots(df, msa_stats):
    """Create comprehensive scaling visualizations."""
    print("\n" + "="*70)
    print("CREATING SCALING VISUALIZATIONS")
    print("="*70)
    
    # Plot 1: Cross-sectional scaling for recent year (2019)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Cross-Sectional Scaling: Three Metrics vs. Population (2019)', 
                 fontsize=14, fontweight='bold')
    
    year_data = df[df['year'] == 2019]
    pop = year_data['population'].values / 1e6  # Convert to millions
    
    metrics = [
        ('eta', 'η (Demographic Efficiency)', 'blue'),
        ('sigma_econ', 'σ_econ (Economic Flux)', 'magenta'),
        ('sigma_mig', 'σ_mig (Migration Rate)', 'orange')
    ]
    
    for i, (metric, title, color) in enumerate(metrics):
        ax = axes[i]
        y = year_data[metric].values
        
        # Scatter
        ax.scatter(pop, y, alpha=0.4, s=30, color=color)
        
        # Fit line
        beta, a, r2, p, ci = fit_power_law(pop * 1e6, y, metric)
        if beta is not None:
            x_line = np.logspace(np.log10(pop.min()), np.log10(pop.max()), 100)
            y_line = a * (x_line * 1e6) ** beta
            ax.plot(x_line, y_line, 'r-', linewidth=2, 
                   label=f'β = {beta:.3f}, R² = {r2:.3f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Population (millions)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scaling_cross_section_2019.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: scaling_cross_section_2019.png")
    
    # Plot 2: Longitudinal variability scaling
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Scaling of Temporal Variability (CV vs. Mean Population)', 
                 fontsize=14, fontweight='bold')
    
    pop = msa_stats['population_mean'].values / 1e6
    
    for i, (metric, title, color) in enumerate(metrics):
        ax = axes[i]
        y = msa_stats[f"{metric}_cv"].fillna(0).values
        
        ax.scatter(pop, y, alpha=0.4, s=30, color=color)
        
        beta, a, r2, p, ci = fit_power_law(pop * 1e6, y, f"{metric}_cv")
        if beta is not None:
            x_line = np.logspace(np.log10(pop.min()), np.log10(pop.max()), 100)
            y_line = a * (x_line * 1e6) ** beta
            ax.plot(x_line, y_line, 'r-', linewidth=2,
                   label=f'β = {beta:.3f}, R² = {r2:.3f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Mean Population (millions)', fontsize=11)
        ax.set_ylabel(f'CV of {metric}', fontsize=11)
        ax.set_title(f'Variability Scaling: {title}', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scaling_temporal_variability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: scaling_temporal_variability.png")
    
    # Plot 3: Time evolution of scaling exponents
    print("\n  Computing time evolution...")
    years = sorted(df['year'].unique())
    eta_betas = []
    se_betas = []
    sm_betas = []
    valid_years = []
    
    for year in years:
        year_data = df[df['year'] == year]
        if len(year_data) < 50:
            continue
        
        pop = year_data['population'].values
        
        beta_eta, _, _, _, _ = fit_power_law(pop, year_data['eta'].values, 'eta')
        beta_se, _, _, _, _ = fit_power_law(pop, year_data['sigma_econ'].values, 'se')
        beta_sm, _, _, _, _ = fit_power_law(pop, year_data['sigma_mig'].values, 'sm')
        
        if beta_eta is not None and beta_se is not None and beta_sm is not None:
            eta_betas.append(beta_eta)
            se_betas.append(beta_se)
            sm_betas.append(beta_sm)
            valid_years.append(year)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valid_years, eta_betas, 'o-', label='η', color='blue', linewidth=2)
    ax.plot(valid_years, se_betas, 's-', label='σ_econ', color='magenta', linewidth=2)
    ax.plot(valid_years, sm_betas, '^-', label='σ_mig', color='orange', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, label='Scale-invariant (β=0)')
    ax.axhline(1, color='gray', linestyle=':', alpha=0.5, label='Linear (β=1)')
    ax.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Scaling Exponent β', fontsize=12)
    ax.set_title('Evolution of Scaling Exponents Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scaling_exponent_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: scaling_exponent_evolution.png")


def main():
    """Main scaling analysis pipeline."""
    print("="*70)
    print("URBAN SCALING ANALYSIS - Power Law Relationships")
    print("="*70)
    print("\nTesting: Y = Y_0 × Population^β")
    
    df = load_data()
    
    # Cross-sectional scaling
    cs_results = analyze_cross_sectional_scaling(df)
    
    # Longitudinal scaling
    msa_stats = analyze_longitudinal_scaling(df)
    
    # Create plots
    create_scaling_plots(df, msa_stats)
    
    print("\n" + "="*70)
    print("SCALING ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
