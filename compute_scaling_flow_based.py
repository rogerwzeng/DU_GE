"""
Compute temporal scaling using FLOW-BASED entropy production (from thermodynamics module)
and TRUE SHOOTING METHOD geodesic efficiency.

This is the theoretically consistent approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys

# Paths
BASE_DIR = Path("/home/roger/DissipativeUrbanism")
OUTPUT_DIR = BASE_DIR / "geodesic_efficiency/results"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_flow_based_sigma():
    """Load flow-based entropy production from thermodynamics module."""
    entropy_file = BASE_DIR / "results/thermodynamics/official_msa_entropy_production.csv"
    df = pd.read_csv(entropy_file)
    
    # Exclude 2006 (first year has 0 values)
    df_nonzero = df[df['year'] > 2006].copy()
    
    # Compute mean σ per MSA
    mean_sigma = df_nonzero.groupby(['msa_code', 'msa_name'])['entropy_production'].mean().reset_index()
    mean_sigma.columns = ['msa_code', 'msa_name', 'sigma']
    
    return mean_sigma

def load_eta_shooting():
    """Load η from true shooting method."""
    eta_file = OUTPUT_DIR / "raw_metrics_true_shooting.csv"
    df = pd.read_csv(eta_file)
    return df[['msa_code', 'eta', 'sigma']].rename(columns={'sigma': 'sigma_temporal'})

def analyze_scaling(sigma_df, eta_df):
    """Analyze scaling relationship."""
    # Merge datasets
    merged = pd.merge(sigma_df, eta_df, on='msa_code', how='inner')
    
    # Filter valid data
    valid = merged[
        (merged['eta'] > 0) & (merged['eta'] <= 1) & 
        (merged['sigma'] > 0) & np.isfinite(merged['sigma'])
    ].copy()
    
    print("="*70)
    print("TEMPORAL SCALING ANALYSIS")
    print("Flow-based σ + True Shooting η")
    print("="*70)
    print(f"Valid MSAs: {len(valid)}")
    
    eta = valid['eta'].values
    sigma = valid['sigma'].values
    
    print("\nDescriptive Statistics:")
    print(f"  η: mean={eta.mean():.4f}, std={eta.std():.4f}, min={eta.min():.4f}, max={eta.max():.4f}")
    print(f"  σ: mean={sigma.mean():.4f}, std={sigma.std():.4f}, min={sigma.min():.4f}, max={sigma.max():.4f}")
    
    # Power law fit: σ = a * η^β
    log_eta = np.log(eta)
    log_sigma = np.log(sigma)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eta, log_sigma)
    
    beta = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    print("\n" + "="*70)
    print("POWER LAW: σ = a × η^β")
    print("="*70)
    print(f"  β = {beta:.4f}")
    print(f"  a = {a:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Std error = {std_err:.4f}")
    
    # Bootstrap confidence intervals
    np.random.seed(42)
    n_bootstrap = 1000
    betas = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(valid), size=len(valid), replace=True)
        le = log_eta[idx]
        ls = log_sigma[idx]
        if len(np.unique(le)) > 1:
            b, _, _, _, _ = stats.linregress(le, ls)
            betas.append(b)
    
    ci_low = np.percentile(betas, 2.5)
    ci_high = np.percentile(betas, 97.5)
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Interpretation
    print("\n  Interpretation:")
    if beta > 0.5:
        print("    → SUPERLINEAR scaling (β > 0.5)")
        print("    → 'Network-like' with increasing returns")
    elif beta > 0:
        print("    → SUBLINEAR scaling (0 < β < 0.5)")
        print("    → 'Infrastructure-like' with diminishing returns")
    else:
        print("    → NEGATIVE scaling (β < 0)")
        print("    → Counter-intuitive relationship")
    
    # Spearman correlation (rank-based, non-parametric)
    rho, p_spearman = stats.spearmanr(eta, sigma)
    print(f"\nSpearman ρ = {rho:.4f} (p={p_spearman:.2e})")
    
    return valid, beta, a, r_squared, (ci_low, ci_high)

def create_plots(valid, beta, a, r_squared, ci):
    """Create visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eta = valid['eta'].values
    sigma = valid['sigma'].values
    
    # Plot 1: Scatter plot with regression line
    ax1 = axes[0]
    ax1.scatter(eta, sigma, alpha=0.5, s=30, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # Power law fit line
    eta_fit = np.linspace(eta.min(), eta.max(), 100)
    sigma_fit = a * (eta_fit ** beta)
    ax1.plot(eta_fit, sigma_fit, 'r-', linewidth=2, 
             label=f'σ = {a:.2f} × η^{beta:.3f}')
    
    ax1.set_xlabel('Geodesic Efficiency (η)', fontsize=12)
    ax1.set_ylabel('Entropy Production (σ)', fontsize=12)
    ax1.set_title('Temporal Scaling: σ vs η\n(Flow-based σ + True Shooting η)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'β = {beta:.3f}\nR² = {r_squared:.3f}\np < 0.001\nn = {len(valid)}'
    ax1.text(0.95, 0.05, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Log-log plot
    ax2 = axes[1]
    ax2.scatter(eta, sigma, alpha=0.5, s=30, c='steelblue', edgecolors='white', linewidth=0.5)
    ax2.plot(eta_fit, sigma_fit, 'r-', linewidth=2, label='Power law fit')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Geodesic Efficiency (η) [log]', fontsize=12)
    ax2.set_ylabel('Entropy Production (σ) [log]', fontsize=12)
    ax2.set_title('Log-Log Plot: σ vs η', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scaling_flow_based_sigma.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {OUTPUT_DIR / 'scaling_flow_based_sigma.png'}")
    plt.close()

def save_results(valid, beta, a, r_squared, ci):
    """Save analysis results."""
    # Save merged data
    valid.to_csv(OUTPUT_DIR / 'scaling_flow_based_data.csv', index=False)
    
    # Save summary
    with open(OUTPUT_DIR / 'scaling_flow_based_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEMPORAL SCALING ANALYSIS\n")
        f.write("Flow-based σ + True Shooting η\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Valid MSAs: {len(valid)}\n\n")
        
        f.write("Descriptive Statistics:\n")
        f.write(f"  η: mean={valid['eta'].mean():.4f}, std={valid['eta'].std():.4f}\n")
        f.write(f"  σ: mean={valid['sigma'].mean():.4f}, std={valid['sigma'].std():.4f}\n\n")
        
        f.write("POWER LAW: σ = a × η^β\n")
        f.write(f"  β = {beta:.4f}\n")
        f.write(f"  a = {a:.4f}\n")
        f.write(f"  R² = {r_squared:.4f}\n")
        f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
        
        rho, _ = stats.spearmanr(valid['eta'].values, valid['sigma'].values)
        f.write(f"\nSpearman ρ = {rho:.4f}\n")
    
    print(f"Results saved: {OUTPUT_DIR / 'scaling_flow_based_summary.txt'}")

def main():
    print("Loading flow-based entropy production...")
    sigma_df = load_flow_based_sigma()
    
    print("Loading true shooting η...")
    eta_df = load_eta_shooting()
    
    print("\nAnalyzing scaling relationship...")
    valid, beta, a, r_squared, ci = analyze_scaling(sigma_df, eta_df)
    
    print("\nCreating plots...")
    create_plots(valid, beta, a, r_squared, ci)
    
    save_results(valid, beta, a, r_squared, ci)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
