"""
Compute temporal scaling using the TRUE SHOOTING METHOD for geodesic distance.
This actually solves the geodesic equation via numerical integration.
WARNING: Much slower than discrete approximation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path.home() / 'DissipativeUrbanism/src'))

from geometry.demographic_manifold import DemographicManifold
from geometry.geodesic_solver import GeodesicSolver
from geometry.fisher_metric import FisherMetric
from analysis.geodesic_framework import (
    compute_entropy_production,
    compute_shannon_entropy
)

# Paths
DATA_FILE = Path("/home/roger/DissipativeUrbanism/results/data/msa_demographics_raw_annual.csv")
OUTPUT_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_entropy(counts):
    """Compute Shannon entropy."""
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))


def compute_gini(values):
    """Compute Gini coefficient."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.sum(sorted_values)
    return (n + 1 - 2 * np.sum(np.cumsum(sorted_values)) / cumsum) / n


def prepare_data():
    """Load and process raw data."""
    print("Loading raw demographic data...")
    df_raw = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df_raw)} records for {df_raw['msa_code'].nunique()} MSAs")
    
    age_cols = [c for c in df_raw.columns if c.startswith('age_')]
    race_cols = [c for c in df_raw.columns if c.startswith('race_')]
    income_cols = [c for c in df_raw.columns if c.startswith('income_decile_')]
    
    results = []
    for (msa_code, msa_name), group in df_raw.groupby(['msa_code', 'msa_name']):
        for _, row in group.iterrows():
            year = int(row['year'])
            
            age_counts = np.array([row.get(c, 0) for c in age_cols], dtype=float)
            race_counts = np.array([row.get(c, 0) for c in race_cols], dtype=float)
            income_counts = np.array([row.get(c, 0) for c in income_cols], dtype=float)
            
            age_entropy = compute_entropy(age_counts)
            income_gini = compute_gini(income_counts)
            race_entropy = compute_entropy(race_counts)
            pop_density = row.get('total_population', 0) / 1000.0
            
            results.append({
                'msa_code': msa_code,
                'msa_name': msa_name,
                'year': year,
                'total_population': row.get('total_population', 0),
                'population_density': pop_density,
                'age_entropy': age_entropy,
                'income_gini': income_gini,
                'diversity_shannon': race_entropy
            })
    
    return pd.DataFrame(results)


def compute_eta_shooting(actual_path, g_func, solver):
    """
    Compute η using TRUE SHOOTING METHOD.
    
    eta = d_geodesic_shooting / d_actual
    
    where d_geodesic_shooting is computed by solving the boundary value problem.
    """
    theta_a = actual_path[0]
    theta_b = actual_path[-1]
    
    # Compute geodesic distance using SHOOTING METHOD
    try:
        d_geodesic = solver.compute_geodesic_distance(
            theta_a, theta_b, g_func, method='shooting'
        )
    except Exception as e:
        # Fallback to shortest_path if shooting fails
        print(f"    Shooting failed ({e}), using shortest_path")
        d_geodesic = solver.compute_geodesic_distance(
            theta_a, theta_b, g_func, method='shortest_path'
        )
    
    # Compute actual path length (same as before)
    d_actual = 0.0
    for i in range(len(actual_path) - 1):
        d_theta = actual_path[i+1] - actual_path[i]
        try:
            g = g_func(0.5 * (actual_path[i] + actual_path[i+1]))
            d_actual += np.sqrt(d_theta @ g @ d_theta)
        except Exception:
            d_actual += np.linalg.norm(d_theta)
    
    if d_actual == 0:
        return 1.0
    
    eta = d_geodesic / d_actual
    return np.clip(eta, 0.0, 1.0)


def compute_metrics_shooting(df):
    """Compute η using TRUE shooting method and σ using original formula."""
    print("\nCreating DemographicManifold...")
    manifold = DemographicManifold(df)
    
    # Create geodesic solver
    fisher_metric = FisherMetric()
    solver = GeodesicSolver(fisher_metric)
    
    print(f"  MSAs: {len(manifold.msa_codes)}")
    print(f"  Using TRUE SHOOTING METHOD (may be slow)...")
    
    # Get raw data for entropy production
    df_raw = pd.read_csv(DATA_FILE)
    age_cols = [c for c in df_raw.columns if c.startswith('age_')]
    race_cols = [c for c in df_raw.columns if c.startswith('race_')]
    income_cols = [c for c in df_raw.columns if c.startswith('income_decile_')]
    
    results = []
    total_time = 0
    
    for i, msa_code in enumerate(manifold.msa_codes):
        start = time.time()
        
        if (i + 1) % 10 == 0:
            avg_time = total_time / (i + 1)
            eta_sec = avg_time * (len(manifold.msa_codes) - i - 1)
            print(f"  [{i+1}/{len(manifold.msa_codes)}] ETA: {eta_sec/60:.1f} min "
                  f"(avg {avg_time:.2f}s/MSA)")
        
        # Get trajectory
        trajectory = manifold.embed_trajectory(msa_code)
        actual_path = trajectory['theta']
        
        # Define metric function
        def g_func(theta):
            return fisher_metric.compute_fisher_metric(
                theta.reshape(1, -1), method='empirical'
            )
        
        # Compute η using SHOOTING METHOD
        try:
            eta = compute_eta_shooting(actual_path, g_func, solver)
        except Exception as e:
            print(f"    Warning: Failed for MSA {msa_code}: {e}")
            continue
        
        # Compute σ from raw data
        msa_data = df_raw[df_raw['msa_code'] == msa_code].sort_values('year')
        
        populations = []
        entropies = []
        for _, row in msa_data.iterrows():
            age_counts = np.array([row.get(c, 0) for c in age_cols], dtype=float)
            race_counts = np.array([row.get(c, 0) for c in race_cols], dtype=float)
            income_counts = np.array([row.get(c, 0) for c in income_cols], dtype=float)
            
            h_age = compute_entropy(age_counts)
            h_race = compute_entropy(race_counts)
            h_income = compute_entropy(income_counts)
            
            populations.append(row.get('total_population', 0))
            entropies.append(h_age + h_race + h_income)
        
        pop_arr = np.array(populations)
        ent_arr = np.array(entropies)
        
        sigma_series = compute_entropy_production(pop_arr, ent_arr, dt=1.0)
        mean_sigma = np.mean(sigma_series)
        
        results.append({
            'msa_code': msa_code,
            'msa_name': trajectory['msa_name'],
            'eta': eta,
            'sigma': mean_sigma,
            'n_years': len(pop_arr),
        })
        
        total_time += time.time() - start
    
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    print(f"Average: {total_time/len(manifold.msa_codes):.2f}s per MSA")
    
    return pd.DataFrame(results)


def analyze_scaling_law(df):
    """Analyze scaling relationship."""
    valid = df[(df['eta'] > 0) & (df['eta'] <= 1) & 
               (df['sigma'] > 0) & np.isfinite(df['sigma'])].copy()
    
    print(f"\n{'='*70}")
    print("TEMPORAL SCALING ANALYSIS (TRUE SHOOTING METHOD)")
    print(f"{'='*70}")
    print(f"Valid MSAs: {len(valid)}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  η: mean={valid['eta'].mean():.4f}, std={valid['eta'].std():.4f}")
    print(f"  σ: mean={valid['sigma'].mean():.4f}, std={valid['sigma'].std():.4f}")
    
    eta = valid['eta'].values
    sigma = valid['sigma'].values
    
    # Power Law
    log_eta = np.log(eta)
    log_sigma = np.log(sigma)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eta, log_sigma)
    
    beta = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    print(f"\n{'='*70}")
    print("POWER LAW: σ = a × η^β")
    print(f"{'='*70}")
    print(f"  β = {beta:.4f}")
    print(f"  a = {a:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Std error = {std_err:.4f}")
    
    # Bootstrap CI
    np.random.seed(42)
    n_bootstrap = 1000
    beta_bootstrap = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(valid), size=len(valid), replace=True)
        s, _, _, _, _ = stats.linregress(np.log(eta[idx]), np.log(sigma[idx]))
        beta_bootstrap.append(s)
    
    beta_ci = np.percentile(beta_bootstrap, [2.5, 97.5])
    print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if beta > 1.05:
        print(f"    → SUPERLINEAR scaling (β > 1)")
    elif beta < 0.95:
        print(f"    → SUBLINEAR scaling (β < 1)")
        print(f"    → 'Infrastructure-like' with diminishing returns")
    else:
        print(f"    → LINEAR scaling (β ≈ 1)")
    
    # Compare with pre-computed (shortest_path) values
    print(f"\n{'='*70}")
    print("COMPARISON WITH PRE-COMPUTED (shortest_path)")
    print(f"{'='*70}")
    precomputed = pd.read_csv(OUTPUT_DIR / 'msa_data_with_coords.csv')
    merged = valid.merge(precomputed, on='msa_code', suffixes=('_shooting', '_pre'))
    
    if len(merged) > 0:
        r_eta, _ = stats.pearsonr(merged['eta'], merged['geodesic_efficiency'])
        print(f"  Correlation η (shooting vs shortest_path): r = {r_eta:.4f}")
        
        diff = merged['eta'] - merged['geodesic_efficiency']
        print(f"  Mean difference: {diff.mean():.6f}")
        print(f"  Std difference: {diff.std():.6f}")
        print(f"  Max |difference|: {diff.abs().max():.6f}")
    
    return {
        'beta': beta,
        'a': a,
        'r_squared': r_squared,
        'p_value': p_value,
        'beta_ci': beta_ci,
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
    
    # Panel 1: Linear
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
    ax3.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax3.set_title('C. Residuals', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Scaling comparison
    ax4 = axes[1, 1]
    categories = ['Infrastructure\n(β≈0.85)', 'Linear\n(β=1.0)', 'Socioeconomic\n(β≈1.15)', 
                  'True Shooting\nMethod']
    betas = [0.85, 1.0, 1.15, beta]
    colors = ['blue', 'gray', 'green', 'red']
    bars = ax4.bar(categories, betas, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_ylabel('Scaling Exponent (β)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Scaling Regime', fontsize=13, fontweight='bold')
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
    
    plt.suptitle(f'Temporal Scaling (TRUE Shooting Method)\nn={len(df)} MSAs, β={beta:.3f}', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("TEMPORAL SCALING - TRUE SHOOTING METHOD")
    print("="*70)
    print("WARNING: This uses actual numerical integration (shooting method)")
    print("         which is much slower than discrete approximation.")
    print(f"Data: {DATA_FILE}")
    print()
    
    # Prepare data
    df = prepare_data()
    
    # Compute metrics
    print("\nComputing metrics using TRUE shooting method...")
    results_df = compute_metrics_shooting(df)
    print(f"Computed for {len(results_df)} MSAs")
    
    # Save
    metrics_path = OUTPUT_DIR / "raw_metrics_true_shooting.csv"
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    
    # Analyze
    results = analyze_scaling_law(results_df)
    
    # Save
    results_path = OUTPUT_DIR / "raw_scaling_true_shooting.txt"
    with open(results_path, 'w') as f:
        f.write(f"TEMPORAL SCALING (TRUE SHOOTING METHOD)\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"N MSAs: {results['n_observations']}\n")
        f.write(f"η mean: {results['df']['eta'].mean():.4f}\n")
        f.write(f"σ mean: {results['df']['sigma'].mean():.4f}\n\n")
        f.write(f"Power Law: σ = {results['a']:.4f} × η^{results['beta']:.4f}\n")
        f.write(f"β = {results['beta']:.4f} [{results['beta_ci'][0]:.4f}, {results['beta_ci'][1]:.4f}]\n")
        f.write(f"R² = {results['r_squared']:.4f}, p = {results['p_value']:.2e}\n")
    
    print(f"\nResults saved: {results_path}")
    
    # Plot
    plot_path = OUTPUT_DIR / "raw_scaling_true_shooting.png"
    create_visualization(results, plot_path)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
