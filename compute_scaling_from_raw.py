"""
Compute temporal scaling law using RAW MSA data from results/temp/
(NOT using msa_data_with_coords.csv)
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

def shannon_entropy(probs):
    """Compute Shannon entropy H = -Σ p_i log(p_i)"""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]  # Remove zeros
    if probs.sum() == 0:
        return 0.0
    probs = probs / probs.sum()  # Normalize
    return float(-np.sum(probs * np.log(probs)))

def compute_eta_sigma_from_raw(year_files):
    """
    Compute η and σ from raw demographic data files.
    
    η (geodesic efficiency): Computed from demographic transitions between years
    σ (entropy production): Rate of change in Shannon entropy over time
    """
    results = []
    
    # Sort files by year
    year_files = sorted(year_files, key=lambda x: int(x.stem.split('_')[-1]))
    
    # Load all data
    all_data = {}
    for f in year_files:
        year = int(f.stem.split('_')[-1])
        df = pd.read_csv(f)
        df['year'] = year
        all_data[year] = df
    
    years = sorted(all_data.keys())
    print(f"Loaded data for years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Get all MSAs present across all years
    all_msas = set()
    for year, df in all_data.items():
        all_msas.update(df['msa_code'].unique())
    
    print(f"Total unique MSAs: {len(all_msas)}")
    
    for msa_code in all_msas:
        msa_data = []
        msa_name = None
        
        # Collect data for this MSA across all years
        for year in years:
            df = all_data[year]
            msa_row = df[df['msa_code'] == msa_code]
            if len(msa_row) > 0:
                msa_data.append(msa_row.iloc[0])
                if msa_name is None:
                    msa_name = msa_row.iloc[0]['msa_name']
        
        if len(msa_data) < 2:
            continue  # Need at least 2 years
        
        # Extract demographic distributions
        age_cols = [c for c in df.columns if c.startswith('age_')]
        race_cols = [c for c in df.columns if c.startswith('race_')]
        income_cols = [c for c in df.columns if c.startswith('income_decile_')]
        
        entropies = []
        years_present = []
        
        for row in msa_data:
            # Combine all demographic categories
            age_dist = row[age_cols].values
            race_dist = row[race_cols].values
            income_dist = row[income_cols].values
            
            # Compute entropy for each dimension
            h_age = shannon_entropy(age_dist)
            h_race = shannon_entropy(race_dist)
            h_income = shannon_entropy(income_dist)
            
            # Total entropy (sum across dimensions)
            total_h = h_age + h_race + h_income
            entropies.append(total_h)
            years_present.append(row['year'])
        
        # Compute σ: slope of entropy over time (entropy production rate)
        if len(entropies) >= 2:
            slope, intercept, r_val, p_val, se = stats.linregress(years_present, entropies)
            sigma = abs(slope)  # Entropy production rate (absolute change)
        else:
            continue
        
        # Compute η: geodesic efficiency from temporal trajectory
        # η = 1 - (actual path / straight-line distance)
        # In temporal context: how efficiently does the MSA move through demographic space
        
        # Create state vectors for each year
        state_vectors = []
        for row in msa_data:
            vec = np.concatenate([
                row[age_cols].values / (row[age_cols].values.sum() + 1e-10),
                row[race_cols].values / (row[race_cols].values.sum() + 1e-10),
                row[income_cols].values / (row[income_cols].values.sum() + 1e-10)
            ])
            state_vectors.append(vec)
        
        # Compute actual path length (sum of Euclidean distances between consecutive years)
        actual_path = 0
        for i in range(len(state_vectors) - 1):
            dist = np.linalg.norm(state_vectors[i+1] - state_vectors[i])
            actual_path += dist
        
        # Compute geodesic (straight-line) distance from start to end
        geodesic_dist = np.linalg.norm(state_vectors[-1] - state_vectors[0])
        
        # Geodesic efficiency
        if actual_path > 0:
            eta = geodesic_dist / actual_path
        else:
            eta = 1.0  # No change = perfect efficiency
        
        results.append({
            'msa_code': msa_code,
            'msa_name': msa_name,
            'eta': eta,
            'sigma': sigma,
            'n_years': len(msa_data),
            'entropy_start': entropies[0],
            'entropy_end': entropies[-1],
            'actual_path': actual_path,
            'geodesic_dist': geodesic_dist
        })
    
    return pd.DataFrame(results)

def fit_scaling_models(df):
    """Fit different scaling models to η vs σ relationship."""
    # Filter valid data
    valid = df[(df['eta'] > 0) & (df['eta'] <= 1) & (df['sigma'] > 0) & np.isfinite(df['sigma'])]
    
    print(f"\n{'='*60}")
    print("TEMPORAL SCALING ANALYSIS (FROM RAW DATA)")
    print(f"{'='*60}")
    print(f"Valid MSAs for analysis: {len(valid)}")
    print(f"\nDescriptive Statistics:")
    print(f"  η (geodesic efficiency): mean={valid['eta'].mean():.3f}, std={valid['eta'].std():.3f}")
    print(f"  σ (entropy production): mean={valid['sigma'].mean():.3f}, std={valid['sigma'].std():.3f}")
    
    # 1. Power Law: σ = a × η^β
    log_eta = np.log(valid['eta'])
    log_sigma = np.log(valid['sigma'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eta, log_sigma)
    
    beta = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    print(f"\n{'='*60}")
    print("POWER LAW FIT: σ = a × η^β")
    print(f"{'='*60}")
    print(f"  β (scaling exponent) = {beta:.4f}")
    print(f"  a (prefactor) = {a:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Standard error = {std_err:.4f}")
    
    # 2. Linear fit for comparison
    slope_lin, intercept_lin, r_val_lin, p_val_lin, _ = stats.linregress(valid['eta'], valid['sigma'])
    print(f"\nLinear fit (for comparison): σ = {slope_lin:.2f}×η + {intercept_lin:.2f}, R² = {r_val_lin**2:.4f}")
    
    # 3. Logarithmic fit
    log_eta_vals = np.log(valid['eta'])
    slope_log, intercept_log, r_val_log, p_val_log, _ = stats.linregress(log_eta_vals, valid['sigma'])
    print(f"Logarithmic fit: σ = {slope_log:.2f}×ln(η) + {intercept_log:.2f}, R² = {r_val_log**2:.4f}")
    
    # Bootstrap confidence intervals for β
    print(f"\n{'='*60}")
    print("BOOTSTRAP CONFIDENCE INTERVALS FOR β")
    print(f"{'='*60}")
    n_bootstrap = 1000
    np.random.seed(42)
    beta_bootstrap = []
    
    for _ in range(n_bootstrap):
        sample = valid.sample(n=len(valid), replace=True)
        if len(sample) > 1:
            s, _, _, _, _ = stats.linregress(np.log(sample['eta']), np.log(sample['sigma']))
            beta_bootstrap.append(s)
    
    beta_ci = np.percentile(beta_bootstrap, [2.5, 97.5])
    print(f"  β = {beta:.4f} [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}] (95% CI)")
    print(f"  Bootstrap std = {np.std(beta_bootstrap):.4f}")
    
    return {
        'beta': beta,
        'a': a,
        'r_squared': r_squared,
        'p_value': p_value,
        'beta_ci': beta_ci,
        'n_observations': len(valid),
        'df': valid
    }

def plot_scaling_relationship(results, output_path):
    """Create visualization of the scaling relationship."""
    df = results['df']
    beta = results['beta']
    a = results['a']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Linear scale
    ax = axes[0, 0]
    ax.scatter(df['eta'], df['sigma'], alpha=0.5, s=30)
    eta_range = np.linspace(df['eta'].min(), df['eta'].max(), 100)
    sigma_pred = a * eta_range ** beta
    ax.plot(eta_range, sigma_pred, 'r-', linewidth=2, 
            label=f'σ = {a:.2f} × η^{beta:.3f}')
    ax.set_xlabel('Geodesic Efficiency (η)', fontsize=11)
    ax.set_ylabel('Entropy Production (σ)', fontsize=11)
    ax.set_title('Linear Scale', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Log-log scale
    ax = axes[0, 1]
    ax.scatter(df['eta'], df['sigma'], alpha=0.5, s=30)
    ax.plot(eta_range, sigma_pred, 'r-', linewidth=2,
            label=f'β = {beta:.3f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Geodesic Efficiency (η)', fontsize=11)
    ax.set_ylabel('Entropy Production (σ)', fontsize=11)
    ax.set_title('Log-Log Scale', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Residuals
    ax = axes[1, 0]
    sigma_fitted = a * df['eta'] ** beta
    residuals = df['sigma'] - sigma_fitted
    ax.scatter(df['eta'], residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Geodesic Efficiency (η)', fontsize=11)
    ax.set_ylabel('Residuals (σ - fitted)', fontsize=11)
    ax.set_title('Residuals', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Distribution of β interpretation
    ax = axes[1, 1]
    categories = ['Sublinear\n(β<1)']
    values = [1 if beta < 1 else 0]
    colors = ['green' if beta < 1 else 'gray']
    ax.bar(categories, [1], color='green', alpha=0.7, edgecolor='black')
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Scaling Regime', fontsize=11)
    ax.set_title(f'Result: β = {beta:.3f} (Sublinear)', fontsize=12)
    ax.text(0, 0.5, f'Infrastructure-like\nscaling', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Temporal Scaling Law (n={len(df)} MSAs)\nRAW DATA ANALYSIS', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("TEMPORAL SCALING ANALYSIS FROM RAW MSA DATA")
    print("="*60)
    print(f"Data source: {TEMP_DIR}")
    print()
    
    # Get all year files
    year_files = list(TEMP_DIR.glob("msa_demographics_*.csv"))
    print(f"Found {len(year_files)} yearly data files")
    
    # Compute η and σ from raw data
    print("\nComputing geodesic efficiency (η) and entropy production (σ)...")
    df = compute_eta_sigma_from_raw(year_files)
    print(f"Computed metrics for {len(df)} MSAs")
    
    # Save computed metrics
    metrics_path = OUTPUT_DIR / "raw_temporal_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")
    
    # Fit scaling models
    results = fit_scaling_models(df)
    
    # Save results
    results_path = OUTPUT_DIR / "raw_scaling_results.txt"
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TEMPORAL SCALING ANALYSIS (RAW DATA)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data source: {TEMP_DIR}\n")
        f.write(f"Number of MSAs: {results['n_observations']}\n\n")
        f.write("POWER LAW: σ = a × η^β\n")
        f.write(f"  β (scaling exponent) = {results['beta']:.4f}\n")
        f.write(f"  a (prefactor) = {results['a']:.4f}\n")
        f.write(f"  R² = {results['r_squared']:.4f}\n")
        f.write(f"  p-value = {results['p_value']:.2e}\n")
        f.write(f"  95% CI for β: [{results['beta_ci'][0]:.4f}, {results['beta_ci'][1]:.4f}]\n")
    
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = OUTPUT_DIR / "raw_scaling_plot.png"
    plot_scaling_relationship(results, plot_path)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
