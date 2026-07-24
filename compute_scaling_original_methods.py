"""
Compute temporal scaling using ORIGINAL implementations from DissipativeUrbanism/src.
Uses proper Fisher-Rao metric (with factor of 2) and entropy production formula.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add original src to path
sys.path.insert(0, str(Path.home() / 'DissipativeUrbanism/src'))

from analysis.geodesic_framework import (
    compute_geodesic_efficiency,
    fisher_rao_distance,
    compute_entropy_production,
    compute_shannon_entropy
)

# Paths
TEMP_DIR = Path("/home/roger/DissipativeUrbanism/results/temp")
OUTPUT_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_metrics_original(year_files):
    """
    Compute η and σ using ORIGINAL methods from geodesic_framework.py.
    
    η: Uses proper Fisher-Rao distance with factor of 2
    σ: Uses |dlnP/dt| + |dH/dt| formula
    """
    # Sort files by year
    year_files = sorted(year_files, key=lambda x: int(x.stem.split('_')[-1]))
    
    # Load all data
    all_data = {}
    for f in year_files:
        year = int(f.stem.split('_')[-1])
        df = pd.read_csv(f)
        all_data[year] = df
    
    years = sorted(all_data.keys())
    print(f"Loaded data for years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Get column names
    sample_df = all_data[years[0]]
    age_cols = [c for c in sample_df.columns if c.startswith('age_')]
    race_cols = [c for c in sample_df.columns if c.startswith('race_')]
    income_cols = [c for c in sample_df.columns if c.startswith('income_decile_')]
    
    print(f"Dimensions: {len(age_cols)} age + {len(race_cols)} race + {len(income_cols)} income")
    
    # Get all MSAs
    all_msas = set()
    for year, df in all_data.items():
        all_msas.update(df['msa_code'].unique())
    
    print(f"Total unique MSAs: {len(all_msas)}")
    
    results = []
    
    for msa_code in all_msas:
        # Collect data for this MSA
        age_probs_list = []
        race_probs_list = []
        income_probs_list = []
        populations = []
        msa_years = []
        msa_name = None
        
        for year in years:
            df = all_data[year]
            msa_row = df[df['msa_code'] == msa_code]
            if len(msa_row) > 0:
                row = msa_row.iloc[0]
                if msa_name is None:
                    msa_name = row['msa_name']
                
                # Get counts
                age_counts = np.array([row.get(c, 0) for c in age_cols], dtype=float)
                race_counts = np.array([row.get(c, 0) for c in race_cols], dtype=float)
                income_counts = np.array([row.get(c, 0) for c in income_cols], dtype=float)
                total_pop = row.get('total_population', 0)
                
                # Normalize to probabilities
                age_total = age_counts.sum()
                race_total = race_counts.sum()
                income_total = income_counts.sum()
                
                if age_total > 0:
                    age_probs = age_counts / age_total
                else:
                    age_probs = np.ones(len(age_cols)) / len(age_cols)
                
                if race_total > 0:
                    race_probs = race_counts / race_total
                else:
                    race_probs = np.ones(len(race_cols)) / len(race_cols)
                
                if income_total > 0:
                    income_probs = income_counts / income_total
                else:
                    income_probs = np.ones(len(income_cols)) / len(income_cols)
                
                age_probs_list.append(age_probs)
                race_probs_list.append(race_probs)
                income_probs_list.append(income_probs)
                populations.append(total_pop)
                msa_years.append(year)
        
        if len(age_probs_list) < 2:
            continue
        
        # Convert to arrays
        age_probs_arr = np.array(age_probs_list)
        race_probs_arr = np.array(race_probs_list)
        income_probs_arr = np.array(income_probs_list)
        populations_arr = np.array(populations)
        
        # Compute η for each dimension using ORIGINAL function
        try:
            eta_age = compute_geodesic_efficiency(age_probs_arr)
            eta_race = compute_geodesic_efficiency(race_probs_arr)
            eta_income = compute_geodesic_efficiency(income_probs_arr)
            
            # Overall η is mean across dimensions
            eta = np.mean([eta_age, eta_race, eta_income])
            eta = np.clip(eta, 0.0, 1.0)
        except Exception as e:
            print(f"  Warning: Could not compute η for MSA {msa_code}: {e}")
            continue
        
        # Compute σ using ORIGINAL function (generic version)
        try:
            # Compute total entropy for each time point
            entropy_series = np.zeros(len(populations_arr))
            for t in range(len(populations_arr)):
                h_age = compute_shannon_entropy(age_probs_arr[t])
                h_race = compute_shannon_entropy(race_probs_arr[t])
                h_income = compute_shannon_entropy(income_probs_arr[t])
                entropy_series[t] = h_age + h_race + h_income
            
            # Compute entropy production: |dlnP/dt| + |dH/dt|
            sigma_series = compute_entropy_production(populations_arr, entropy_series, dt=1.0)
            mean_sigma = np.mean(sigma_series)
        except Exception as e:
            print(f"  Warning: Could not compute σ for MSA {msa_code}: {e}")
            continue
        
        results.append({
            'msa_code': msa_code,
            'msa_name': msa_name,
            'eta': eta,
            'eta_age': eta_age,
            'eta_race': eta_race,
            'eta_income': eta_income,
            'sigma': mean_sigma,
            'sigma_std': np.std(sigma_series) if len(sigma_series) > 1 else 0,
            'n_years': len(msa_years),
        })
    
    return pd.DataFrame(results)


def analyze_scaling_law(df):
    """Analyze the scaling relationship between η and σ."""
    
    # Filter valid data
    valid = df[(df['eta'] > 0) & (df['eta'] <= 1) & 
               (df['sigma'] > 0) & np.isfinite(df['sigma'])].copy()
    
    print(f"\n{'='*70}")
    print("TEMPORAL SCALING ANALYSIS (ORIGINAL METHODS)")
    print(f"{'='*70}")
    print(f"Valid MSAs: {len(valid)}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  η: mean={valid['eta'].mean():.4f}, std={valid['eta'].std():.4f}, "
          f"median={valid['eta'].median():.4f}")
    print(f"  σ: mean={valid['sigma'].mean():.4f}, std={valid['sigma'].std():.4f}, "
          f"median={valid['sigma'].median():.4f}")
    
    eta = valid['eta'].values
    sigma = valid['sigma'].values
    
    # Power Law fit
    log_eta = np.log(eta)
    log_sigma = np.log(sigma)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eta, log_sigma)
    
    beta = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    print(f"\n{'='*70}")
    print("POWER LAW: σ = a × η^β")
    print(f"{'='*70}")
    print(f"  β = {beta:.4f} (scaling exponent)")
    print(f"  a = {a:.4f} (prefactor)")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Std error = {std_err:.4f}")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if beta > 1.05:
        print(f"    → SUPERLINEAR scaling (β > 1)")
        print(f"    → Efficiency gains produce MORE than proportional entropy increase")
    elif beta < 0.95:
        print(f"    → SUBLINEAR scaling (β < 1)")
        print(f"    → Efficiency gains produce LESS than proportional entropy increase")
        print(f"    → 'Infrastructure-like' scaling with diminishing returns")
    else:
        print(f"    → LINEAR scaling (β ≈ 1)")
        print(f"    → Proportional relationship")
    
    # Linear comparison
    slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(eta, sigma)
    print(f"\nLinear fit: σ = {slope_lin:.4f}×η + {intercept_lin:.4f}, R² = {r_lin**2:.4f}")
    
    # Bootstrap CI
    print(f"\n{'='*70}")
    print("BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
    print(f"{'='*70}")
    np.random.seed(42)
    n_bootstrap = 1000
    beta_bootstrap = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(valid), size=len(valid), replace=True)
        sample_eta = eta[idx]
        sample_sigma = sigma[idx]
        if len(sample_eta) > 1:
            s, _, _, _, _ = stats.linregress(np.log(sample_eta), np.log(sample_sigma))
            beta_bootstrap.append(s)
    
    beta_ci = np.percentile(beta_bootstrap, [2.5, 97.5])
    print(f"  β = {beta:.4f}")
    print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
    print(f"  Bootstrap std: {np.std(beta_bootstrap):.4f}")
    
    return {
        'beta': beta,
        'a': a,
        'r_squared': r_squared,
        'p_value': p_value,
        'beta_ci': beta_ci,
        'beta_std': np.std(beta_bootstrap),
        'std_err': std_err,
        'n_observations': len(valid),
        'df': valid
    }


def create_visualization(results, output_path):
    """Create visualization."""
    df = results['df']
    beta = results['beta']
    a = results['a']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    eta = df['eta'].values
    sigma = df['sigma'].values
    
    # Panel 1: Linear scale
    ax1 = axes[0, 0]
    ax1.scatter(eta, sigma, alpha=0.5, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
    eta_range = np.linspace(eta.min(), eta.max(), 200)
    sigma_pred = a * eta_range ** beta
    ax1.plot(eta_range, sigma_pred, 'r-', linewidth=2.5, 
            label=f'σ = {a:.2f} × η^{beta:.3f}')
    ax1.set_xlabel('Geodesic Efficiency (η)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entropy Production (σ)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Linear Scale', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Log-log
    ax2 = axes[0, 1]
    ax2.scatter(eta, sigma, alpha=0.5, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
    ax2.plot(eta_range, sigma_pred, 'r-', linewidth=2.5, label=f'β = {beta:.3f}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Geodesic Efficiency (η) [log]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Entropy Production (σ) [log]', fontsize=12, fontweight='bold')
    ax2.set_title('B. Log-Log Scale', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Residuals
    ax3 = axes[1, 0]
    sigma_fitted = a * eta ** beta
    residuals = sigma - sigma_fitted
    ax3.scatter(eta, residuals, alpha=0.5, s=40, c='coral', edgecolors='white', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Geodesic Efficiency (η)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residuals (σ - fitted)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Power Law Residuals', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Scaling comparison
    ax4 = axes[1, 1]
    categories = ['Infrastructure\n(β≈0.85)', 'Linear\n(β=1.0)', 'Socioeconomic\n(β≈1.15)', 
                  'Our Finding\n(Original)']
    betas = [0.85, 1.0, 1.15, beta]
    colors = ['blue', 'gray', 'green', 'red']
    bars = ax4.bar(categories, betas, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_ylabel('Scaling Exponent (β)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Scaling Regime Comparison', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 1.5)
    for bar, b in zip(bars, betas):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'β={b:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    interp = "SUBLINEAR" if beta < 0.95 else "SUPERLINEAR" if beta > 1.05 else "LINEAR"
    color = 'blue' if beta < 0.95 else 'green' if beta > 1.05 else 'gray'
    ax4.text(0.5, 0.95, f'Result: {interp}', 
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Temporal Scaling (Original Methods)\nn={len(df)} MSAs, β={beta:.3f}', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("TEMPORAL SCALING FROM RAW DATA (ORIGINAL METHODS)")
    print("="*70)
    print(f"Data: {TEMP_DIR}")
    print(f"Using ORIGINAL implementations from DissipativeUrbanism/src")
    print()
    
    year_files = list(TEMP_DIR.glob("msa_demographics_*.csv"))
    print(f"Found {len(year_files)} year files")
    
    print("\nComputing metrics using original methods...")
    df = compute_metrics_original(year_files)
    print(f"Computed for {len(df)} MSAs")
    
    # Save
    metrics_path = OUTPUT_DIR / "raw_metrics_original_methods.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    
    # Analyze
    results = analyze_scaling_law(df)
    
    # Save results
    results_path = OUTPUT_DIR / "raw_scaling_original_methods.txt"
    with open(results_path, 'w') as f:
        f.write(f"TEMPORAL SCALING (ORIGINAL METHODS)\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"N MSAs: {results['n_observations']}\n")
        f.write(f"η mean: {results['df']['eta'].mean():.4f}\n")
        f.write(f"σ mean: {results['df']['sigma'].mean():.4f}\n\n")
        f.write(f"Power Law: σ = {results['a']:.4f} × η^{results['beta']:.4f}\n")
        f.write(f"β = {results['beta']:.4f} [{results['beta_ci'][0]:.4f}, {results['beta_ci'][1]:.4f}]\n")
        f.write(f"R² = {results['r_squared']:.4f}, p = {results['p_value']:.2e}\n")
    
    print(f"\nResults saved: {results_path}")
    
    # Plot
    plot_path = OUTPUT_DIR / "raw_scaling_original_methods.png"
    create_visualization(results, plot_path)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
