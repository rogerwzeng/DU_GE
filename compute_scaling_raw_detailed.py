"""
Compute temporal scaling law using RAW MSA data from results/temp/
Properly computes η (geodesic efficiency) and σ (entropy production) from scratch.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
TEMP_DIR = Path("/home/roger/DissipativeUrbanism/results/temp")
OUTPUT_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
OUTPUT_DIR.mkdir(exist_ok=True)

def compute_shannon_entropy(counts):
    """Compute Shannon entropy from counts (natural log)."""
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]  # Remove zeros
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))

def compute_state_vector(row, age_cols, race_cols, income_cols):
    """
    Compute demographic state vector for a given year.
    Returns normalized probability distribution across all categories.
    """
    # Get counts for each dimension
    age_counts = np.array([row[c] for c in age_cols], dtype=float)
    race_counts = np.array([row[c] for c in race_cols], dtype=float)
    income_counts = np.array([row[c] for c in income_cols], dtype=float)
    
    # Normalize each dimension separately
    age_probs = age_counts / age_counts.sum() if age_counts.sum() > 0 else age_counts
    race_probs = race_counts / race_counts.sum() if race_counts.sum() > 0 else race_counts
    income_probs = income_counts / income_counts.sum() if income_counts.sum() > 0 else income_counts
    
    # Concatenate into single state vector
    state = np.concatenate([age_probs, race_probs, income_probs])
    
    # Remove any NaN values
    state = np.nan_to_num(state, nan=0.0)
    
    return state

def compute_metrics_from_raw(year_files):
    """
    Compute η (geodesic efficiency) and σ (entropy production) from raw data.
    
    η = geodesic_distance / actual_path_length
    σ = mean rate of entropy change over time
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
    
    # Get demographic column names from first file
    sample_df = all_data[years[0]]
    age_cols = [c for c in sample_df.columns if c.startswith('age_')]
    race_cols = [c for c in sample_df.columns if c.startswith('race_')]
    income_cols = [c for c in sample_df.columns if c.startswith('income_decile_')]
    
    print(f"Demographic dimensions: {len(age_cols)} age + {len(race_cols)} race + {len(income_cols)} income = {len(age_cols)+len(race_cols)+len(income_cols)} categories")
    
    # Get all MSAs
    all_msas = set()
    for year, df in all_data.items():
        all_msas.update(df['msa_code'].unique())
    
    print(f"Total unique MSAs: {len(all_msas)}")
    
    results = []
    
    for msa_code in all_msas:
        # Collect data for this MSA across all years
        msa_states = []
        msa_entropies = []
        msa_years = []
        msa_name = None
        
        for year in years:
            df = all_data[year]
            msa_row = df[df['msa_code'] == msa_code]
            if len(msa_row) > 0:
                row = msa_row.iloc[0]
                if msa_name is None:
                    msa_name = row['msa_name']
                
                # Compute state vector
                state = compute_state_vector(row, age_cols, race_cols, income_cols)
                msa_states.append(state)
                
                # Compute total entropy for this year
                age_counts = np.array([row[c] for c in age_cols], dtype=float)
                race_counts = np.array([row[c] for c in race_cols], dtype=float)
                income_counts = np.array([row[c] for c in income_cols], dtype=float)
                
                h_age = compute_shannon_entropy(age_counts)
                h_race = compute_shannon_entropy(race_counts)
                h_income = compute_shannon_entropy(income_counts)
                
                total_entropy = h_age + h_race + h_income
                msa_entropies.append(total_entropy)
                msa_years.append(year)
        
        if len(msa_states) < 2:
            continue  # Need at least 2 years
        
        msa_states = np.array(msa_states)
        msa_entropies = np.array(msa_entropies)
        msa_years = np.array(msa_years)
        
        # Compute η (geodesic efficiency)
        # η = geodesic_distance / actual_path_length
        # where geodesic_distance = Euclidean distance from start to end
        # and actual_path_length = sum of Euclidean distances between consecutive years
        
        # Actual path length (sum of step distances)
        actual_path = 0.0
        for i in range(len(msa_states) - 1):
            dist = np.linalg.norm(msa_states[i+1] - msa_states[i])
            actual_path += dist
        
        # Geodesic distance (straight line from start to end)
        geodesic_dist = np.linalg.norm(msa_states[-1] - msa_states[0])
        
        # Geodesic efficiency
        if actual_path > 1e-10:
            eta = geodesic_dist / actual_path
        else:
            eta = 1.0  # No change = perfect efficiency
        
        # Ensure η is in [0, 1]
        eta = np.clip(eta, 0.0, 1.0)
        
        # Compute σ (entropy production)
        # σ = rate of entropy change over time
        # Fit linear regression: entropy = slope * year + intercept
        # σ = |slope| (absolute rate of change)
        
        if len(msa_entropies) >= 2:
            slope, intercept, r_val, p_val, se = stats.linregress(msa_years, msa_entropies)
            sigma = abs(slope)  # Absolute rate of entropy change
            
            # Alternative: use std of entropy changes
            # entropy_changes = np.diff(msa_entropies)
            # sigma = np.std(entropy_changes)
        else:
            continue
        
        results.append({
            'msa_code': msa_code,
            'msa_name': msa_name,
            'eta': eta,
            'sigma': sigma,
            'n_years': len(msa_years),
            'entropy_start': msa_entropies[0],
            'entropy_end': msa_entropies[-1],
            'entropy_change': msa_entropies[-1] - msa_entropies[0],
            'actual_path': actual_path,
            'geodesic_dist': geodesic_dist,
            'entropy_slope': slope,
            'entropy_r2': r_val**2
        })
    
    return pd.DataFrame(results)

def analyze_scaling_law(df):
    """Analyze the scaling relationship between η and σ."""
    
    # Filter valid data
    valid = df[(df['eta'] > 0) & (df['eta'] <= 1) & 
               (df['sigma'] > 0) & np.isfinite(df['sigma'])].copy()
    
    print(f"\n{'='*70}")
    print("TEMPORAL SCALING ANALYSIS (COMPUTED FROM RAW DATA)")
    print(f"{'='*70}")
    print(f"Valid MSAs for analysis: {len(valid)}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  η (geodesic efficiency): mean={valid['eta'].mean():.4f}, std={valid['eta'].std():.4f}")
    print(f"                           median={valid['eta'].median():.4f}, range=[{valid['eta'].min():.4f}, {valid['eta'].max():.4f}]")
    print(f"  σ (entropy production):  mean={valid['sigma'].mean():.4f}, std={valid['sigma'].std():.4f}")
    print(f"                           median={valid['sigma'].median():.4f}, range=[{valid['sigma'].min():.4f}, {valid['sigma'].max():.4f}]")
    
    eta = valid['eta'].values
    sigma = valid['sigma'].values
    
    # 1. Power Law fit in log-log space: σ = a × η^β
    log_eta = np.log(eta)
    log_sigma = np.log(sigma)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eta, log_sigma)
    
    beta = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    print(f"\n{'='*70}")
    print("POWER LAW FIT: σ = a × η^β")
    print(f"{'='*70}")
    print(f"  β (scaling exponent) = {beta:.4f}")
    print(f"  a (prefactor)        = {a:.4f}")
    print(f"  R²                   = {r_squared:.4f}")
    print(f"  p-value              = {p_value:.2e}")
    print(f"  Std error (β)        = {std_err:.4f}")
    
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
        print(f"    → Proportional relationship between efficiency and entropy production")
    
    # 2. Linear fit for comparison
    slope_lin, intercept_lin, r_val_lin, p_val_lin, se_lin = stats.linregress(eta, sigma)
    print(f"\nLinear fit (comparison): σ = {slope_lin:.4f}×η + {intercept_lin:.4f}")
    print(f"  R² = {r_val_lin**2:.4f}, p = {p_val_lin:.2e}")
    
    # 3. Compare models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  Power Law:  R² = {r_squared:.4f}, β = {beta:.4f}")
    print(f"  Linear:     R² = {r_val_lin**2:.4f}, slope = {slope_lin:.4f}")
    
    if r_squared > r_val_lin**2:
        print(f"  → Power law provides better fit")
    else:
        print(f"  → Linear provides better fit")
    
    # 4. Bootstrap confidence intervals for β
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
    beta_std = np.std(beta_bootstrap)
    
    print(f"  β = {beta:.4f}")
    print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
    print(f"  Bootstrap std: {beta_std:.4f}")
    
    # Return results
    return {
        'beta': beta,
        'a': a,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'beta_ci': beta_ci,
        'beta_std': beta_std,
        'n_observations': len(valid),
        'df': valid
    }

def create_visualization(results, output_path):
    """Create comprehensive visualization of scaling relationship."""
    df = results['df']
    beta = results['beta']
    a = results['a']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    eta = df['eta'].values
    sigma = df['sigma'].values
    
    # Panel 1: Linear scale with power law fit
    ax1 = axes[0, 0]
    ax1.scatter(eta, sigma, alpha=0.5, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
    
    eta_range = np.linspace(eta.min(), eta.max(), 200)
    sigma_pred = a * eta_range ** beta
    ax1.plot(eta_range, sigma_pred, 'r-', linewidth=2.5, 
            label=f'σ = {a:.3f} × η^{beta:.3f}')
    
    ax1.set_xlabel('Geodesic Efficiency (η)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entropy Production (σ)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Linear Scale', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Log-log scale
    ax2 = axes[0, 1]
    ax2.scatter(eta, sigma, alpha=0.5, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
    ax2.plot(eta_range, sigma_pred, 'r-', linewidth=2.5, label=f'β = {beta:.3f}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Geodesic Efficiency (η) [log]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Entropy Production (σ) [log]', fontsize=12, fontweight='bold')
    ax2.set_title('B. Log-Log Scale (Power Law = Linear)', fontsize=13, fontweight='bold')
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
    
    # Panel 4: Scaling exponent interpretation
    ax4 = axes[1, 1]
    
    categories = ['Infrastructure\n(β≈0.85)', 'Linear\n(β=1.0)', 'Socioeconomic\n(β≈1.15)', 
                  'Our Finding\n(Raw Data)']
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
    
    # Interpretation
    if beta < 0.95:
        interp = "SUBLINEAR\n(Infrastructure-like)"
        color = 'blue'
    elif beta > 1.05:
        interp = "SUPERLINEAR\n(Socioeconomic-like)"
        color = 'green'
    else:
        interp = "LINEAR\n(Proportional)"
        color = 'gray'
    
    ax4.text(0.5, 0.95, f'Result: {interp}', 
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Temporal Scaling Law (n={len(df)} MSAs, Raw Data)\nσ = {a:.3f} × η^{beta:.3f}', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

def main():
    print("="*70)
    print("TEMPORAL SCALING ANALYSIS FROM RAW MSA DATA")
    print("="*70)
    print(f"Data source: {TEMP_DIR}")
    print()
    
    # Get all year files
    year_files = list(TEMP_DIR.glob("msa_demographics_*.csv"))
    print(f"Found {len(year_files)} yearly data files")
    
    # Compute η and σ from raw data
    print("\nComputing geodesic efficiency (η) and entropy production (σ)...")
    df = compute_metrics_from_raw(year_files)
    print(f"Computed metrics for {len(df)} MSAs")
    
    # Save computed metrics
    metrics_path = OUTPUT_DIR / "raw_metrics_detailed.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Saved computed metrics to: {metrics_path}")
    
    # Analyze scaling law
    results = analyze_scaling_law(df)
    
    # Save detailed results
    results_path = OUTPUT_DIR / "raw_scaling_detailed.txt"
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEMPORAL SCALING ANALYSIS (COMPUTED FROM RAW DATA)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data source: {TEMP_DIR}\n")
        f.write(f"Number of MSAs: {results['n_observations']}\n\n")
        f.write("COMPUTED METRICS:\n")
        f.write(f"  η (geodesic efficiency): mean={results['df']['eta'].mean():.4f}\n")
        f.write(f"  σ (entropy production):  mean={results['df']['sigma'].mean():.4f}\n\n")
        f.write("POWER LAW: σ = a × η^β\n")
        f.write(f"  β (scaling exponent) = {results['beta']:.4f}\n")
        f.write(f"  a (prefactor)        = {results['a']:.4f}\n")
        f.write(f"  R²                   = {results['r_squared']:.4f}\n")
        f.write(f"  p-value              = {results['p_value']:.2e}\n")
        f.write(f"  95% CI for β: [{results['beta_ci'][0]:.4f}, {results['beta_ci'][1]:.4f}]\n")
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create visualization
    plot_path = OUTPUT_DIR / "raw_scaling_detailed.png"
    create_visualization(results, plot_path)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
