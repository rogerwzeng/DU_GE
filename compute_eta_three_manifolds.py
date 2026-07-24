"""
Compute η using THREE separate statistical manifolds as described in the paper:
- Age manifold: 18 categories (5-year cohorts)
- Income manifold: 10 categories (deciles)
- Race manifold: 7 categories

η = mean([η_age, η_income, η_race])
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys

# Paths
BASE_DIR = Path("/home/roger/DissipativeUrbanism")
DATA_FILE = BASE_DIR / "results/data/msa_demographics_raw_annual.csv"
OUTPUT_DIR = BASE_DIR / "geodesic_efficiency/results"
OUTPUT_DIR.mkdir(exist_ok=True)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Fisher-Rao distance between two probability vectors.
    
    d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors (must sum to ~1, all positive)
    eps : float
        Small constant to avoid log(0)
        
    Returns
    -------
    float
        Fisher-Rao distance
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Clip to avoid issues
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    
    # Clip for numerical stability
    bc = np.clip(bc, -1, 1)
    
    # Fisher-Rao distance
    return 2 * np.arccos(bc)


def compute_geodesic_efficiency(prob_matrix: np.ndarray) -> float:
    """
    Compute geodesic efficiency η for a single demographic dimension.
    
    η = d_FR(P[0], P[T]) / sum(d_FR(P[t], P[t+1]))
    
    Parameters
    ----------
    prob_matrix : np.ndarray
        Array of shape (T, K) where T is time points and K is categories
        Each row should sum to ~1.
        
    Returns
    -------
    float
        Geodesic efficiency η
    """
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    T = prob_matrix.shape[0]
    
    if T < 2:
        return np.nan
    
    # Normalize each row to sum to 1
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    prob_matrix = prob_matrix / row_sums
    
    # Numerator: direct distance from first to last
    p_first = prob_matrix[0]
    p_last = prob_matrix[-1]
    numerator = fisher_rao_distance(p_first, p_last)
    
    # Denominator: sum of incremental distances
    denominator = 0.0
    for t in range(T - 1):
        p_t = prob_matrix[t]
        p_t1 = prob_matrix[t + 1]
        denominator += fisher_rao_distance(p_t, p_t1)
    
    # Handle edge cases
    if denominator == 0:
        if numerator == 0:
            return 1.0  # No change at all
        else:
            return np.nan  # Should not occur
    
    eta = numerator / denominator
    
    # Numerical protection
    return float(np.clip(eta, 0.0, 1.0))


def compute_geodesic_efficiency_all_dimensions(
    age_probs: np.ndarray,
    income_probs: np.ndarray,
    race_probs: np.ndarray
) -> tuple:
    """
    Compute geodesic efficiency averaged across all three demographic dimensions.
    
    Returns
    -------
    tuple (mean_eta, {'age': eta_age, 'income': eta_income, 'race': eta_race})
    """
    eta_age = compute_geodesic_efficiency(age_probs)
    eta_income = compute_geodesic_efficiency(income_probs)
    eta_race = compute_geodesic_efficiency(race_probs)
    
    # Arithmetic mean across dimensions (per Section 3.4.4)
    mean_eta = np.nanmean([eta_age, eta_income, eta_race])
    
    return mean_eta, {
        'age': eta_age,
        'income': eta_income,
        'race': eta_race
    }


def compute_eta_three_manifolds():
    """Compute η using three separate statistical manifolds."""
    print("="*70)
    print("THREE MANIFOLDS APPROACH")
    print("="*70)
    print()
    print("Manifold 1: Age (18 categories)")
    print("Manifold 2: Income (10 categories)")
    print("Manifold 3: Race/Ethnicity (7 categories)")
    print()
    
    # Load raw data
    print("Loading raw demographic data...")
    df_raw = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df_raw)} records for {df_raw['msa_code'].nunique()} MSAs")
    
    # Get column names
    age_cols = [c for c in df_raw.columns if c.startswith('age_')]
    race_cols = [c for c in df_raw.columns if c.startswith('race_')]
    income_cols = [c for c in df_raw.columns if c.startswith('income_decile_')]
    
    print(f"  Age categories: {len(age_cols)}")
    print(f"  Race categories: {len(race_cols)}")
    print(f"  Income categories: {len(income_cols)}")
    print()
    
    results = []
    
    for (msa_code, msa_name), group in df_raw.groupby(['msa_code', 'msa_name']):
        group = group.sort_values('year')
        
        # Extract probability matrices for each dimension
        age_probs = group[age_cols].values
        race_probs = group[race_cols].values
        income_probs = group[income_cols].values
        
        # Compute η for each manifold
        eta_overall, eta_by_dim = compute_geodesic_efficiency_all_dimensions(
            age_probs, income_probs, race_probs
        )
        
        results.append({
            'msa_code': msa_code,
            'msa_name': msa_name,
            'eta': eta_overall,
            'eta_age': eta_by_dim['age'],
            'eta_income': eta_by_dim['income'],
            'eta_race': eta_by_dim['race'],
            'n_years': len(group)
        })
    
    results_df = pd.DataFrame(results)
    
    print("Results:")
    print(f"  N MSAs: {len(results_df)}")
    print()
    print("  η (overall):     mean={:.4f}, std={:.4f}".format(
        results_df['eta'].mean(), results_df['eta'].std()))
    print("  η_age:           mean={:.4f}, std={:.4f}".format(
        results_df['eta_age'].mean(), results_df['eta_age'].std()))
    print("  η_income:        mean={:.4f}, std={:.4f}".format(
        results_df['eta_income'].mean(), results_df['eta_income'].std()))
    print("  η_race:          mean={:.4f}, std={:.4f}".format(
        results_df['eta_race'].mean(), results_df['eta_race'].std()))
    
    return results_df


def analyze_scaling_with_flow_sigma(eta_df):
    """Analyze scaling using flow-based σ and three-manifold η."""
    print()
    print("="*70)
    print("SCALING ANALYSIS: Flow-based σ + Three-Manifold η")
    print("="*70)
    
    # Load flow-based entropy production
    entropy_df = pd.read_csv(BASE_DIR / "results/thermodynamics/official_msa_entropy_production.csv")
    entropy_nonzero = entropy_df[entropy_df['year'] > 2006]
    mean_sigma = entropy_nonzero.groupby(['msa_code', 'msa_name'])['entropy_production'].mean().reset_index()
    mean_sigma.columns = ['msa_code', 'msa_name', 'sigma']
    
    # Merge
    merged = pd.merge(eta_df[['msa_code', 'eta']], mean_sigma[['msa_code', 'sigma']], on='msa_code')
    
    # Filter valid
    valid = merged[(merged['eta'] > 0) & (merged['eta'] <= 1) & 
                   (merged['sigma'] > 0) & np.isfinite(merged['sigma'])].copy()
    
    print(f"Valid MSAs: {len(valid)}")
    print()
    print("Descriptive Statistics:")
    print(f"  η: mean={valid['eta'].mean():.4f}, std={valid['eta'].std():.4f}")
    print(f"  σ: mean={valid['sigma'].mean():.4f}, std={valid['sigma'].std():.4f}")
    
    # Power law fit
    eta = valid['eta'].values
    sigma = valid['sigma'].values
    
    log_eta = np.log(eta)
    log_sigma = np.log(sigma)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eta, log_sigma)
    
    beta = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    print()
    print("POWER LAW: σ = a × η^β")
    print(f"  β = {beta:.4f}")
    print(f"  a = {a:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Std error = {std_err:.4f}")
    
    # Bootstrap CI
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
    
    print()
    if beta > 0.5:
        print("  Interpretation: SUPERLINEAR scaling (β > 0.5)")
    elif beta > 0:
        print("  Interpretation: SUBLINEAR scaling (0 < β < 0.5)")
    else:
        print("  Interpretation: NEGATIVE scaling (β < 0)")
    
    # Spearman
    rho, p_spearman = stats.spearmanr(eta, sigma)
    print(f"\nSpearman ρ = {rho:.4f} (p={p_spearman:.2e})")
    
    return valid, beta, a, r_squared, (ci_low, ci_high)


def create_comparison_plot(eta_df):
    """Create comparison with original 4D approach."""
    import matplotlib.pyplot as plt
    
    # Load original 4D η
    original = pd.read_csv(OUTPUT_DIR / "raw_metrics_true_shooting.csv")
    
    merged = pd.merge(
        eta_df[['msa_code', 'eta']].rename(columns={'eta': 'eta_three'}),
        original[['msa_code', 'eta']].rename(columns={'eta': 'eta_four'}),
        on='msa_code'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(merged['eta_four'], merged['eta_three'], alpha=0.5, s=30, 
                c='steelblue', edgecolors='white', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', label='y=x (perfect agreement)')
    ax1.set_xlabel('4D Manifold η', fontsize=12)
    ax1.set_ylabel('3-Manifold η', fontsize=12)
    ax1.set_title('Comparison: 3-Manifold vs 4D Manifold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Statistics
    r = np.corrcoef(merged['eta_four'], merged['eta_three'])[0, 1]
    diff = (merged['eta_three'] - merged['eta_four']).abs()
    stats_text = f'r = {r:.4f}\nMean |diff| = {diff.mean():.4f}\nMax |diff| = {diff.max():.4f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogram comparison
    ax2 = axes[1]
    ax2.hist(merged['eta_four'], bins=30, alpha=0.5, label='4D Manifold', color='blue')
    ax2.hist(merged['eta_three'], bins=30, alpha=0.5, label='3-Manifold', color='red')
    ax2.set_xlabel('Geodesic Efficiency (η)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution Comparison', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_three_vs_four_manifolds.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: {OUTPUT_DIR / 'comparison_three_vs_four_manifolds.png'}")


def main():
    # Compute η using three manifolds
    eta_df = compute_eta_three_manifolds()
    
    # Save results
    eta_df.to_csv(OUTPUT_DIR / 'eta_three_manifolds.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'eta_three_manifolds.csv'}")
    
    # Analyze scaling
    valid, beta, a, r_squared, ci = analyze_scaling_with_flow_sigma(eta_df)
    
    # Create comparison plot
    create_comparison_plot(eta_df)
    
    # Save scaling results
    with open(OUTPUT_DIR / 'scaling_three_manifolds_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("THREE MANIFOLDS APPROACH - SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("Manifold Structure:\n")
        f.write("  - Age: 18 categories\n")
        f.write("  - Income: 10 categories\n")
        f.write("  - Race: 7 categories\n")
        f.write("  - η = mean([η_age, η_income, η_race])\n\n")
        f.write(f"η Statistics:\n")
        f.write(f"  Mean: {eta_df['eta'].mean():.4f}\n")
        f.write(f"  Std:  {eta_df['eta'].std():.4f}\n\n")
        f.write(f"Scaling: σ = a × η^β\n")
        f.write(f"  β = {beta:.4f}\n")
        f.write(f"  a = {a:.4f}\n")
        f.write(f"  R² = {r_squared:.4f}\n")
        f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
    
    print(f"\nSummary saved: {OUTPUT_DIR / 'scaling_three_manifolds_summary.txt'}")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
