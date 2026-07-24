"""
Option 2: Continuous analysis of η × σ interaction
Drops arbitrary quadrant thresholds, uses full (η, σ) space
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def load_data():
    """Load MSA data."""
    df = pd.read_csv(Path(__file__).parent.parent / 'results' / 'msa_data_with_coords.csv')
    return df

def continuous_analysis():
    """Analyze η and σ as continuous predictors."""
    df = load_data()
    
    print("=" * 80)
    print("OPTION 2: CONTINUOUS ANALYSIS (No Quadrant Thresholds)")
    print("=" * 80)
    
    # Basic correlations
    print("\n1. CORRELATION ANALYSIS")
    print("-" * 40)
    
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    
    corr, p = stats.pearsonr(eta, sigma)
    print(f"η vs σ correlation: r = {corr:.3f}, p = {p:.4f}")
    
    corr_spearman, p_spearman = stats.spearmanr(eta, sigma)
    print(f"η vs σ Spearman: ρ = {corr_spearman:.3f}, p = {p_spearman:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if abs(corr) < 0.1:
        print("  - η and σ are essentially uncorrelated")
        print("  - They capture distinct dimensions of urban dynamics")
        print("  - This justifies treating them as separate predictors")
    elif corr > 0:
        print("  - Positive correlation: high efficiency tends to co-occur with high entropy")
        print("  - This is theoretically expected for dissipative structures")
    else:
        print("  - Negative correlation: trade-off between efficiency and entropy production")
    
    # Non-linear relationship exploration
    print("\n2. NON-LINEAR RELATIONSHIPS")
    print("-" * 40)
    
    # Polynomial terms
    eta_squared = eta ** 2
    sigma_squared = sigma ** 2
    eta_sigma_interaction = eta * sigma
    
    # Correlation with interaction term
    corr_int, p_int = stats.pearsonr(eta_sigma_interaction, sigma)
    print(f"η×σ vs σ: r = {corr_int:.3f}")
    corr_int2, p_int2 = stats.pearsonr(eta_sigma_interaction, eta)
    print(f"η×σ vs η: r = {corr_int2:.3f}")
    
    # Distribution characteristics
    print("\n3. DISTRIBUTION CHARACTERISTICS")
    print("-" * 40)
    print(f"η - Skewness: {stats.skew(eta):.3f}, Kurtosis: {stats.kurtosis(eta):.3f}")
    print(f"σ - Skewness: {stats.skew(sigma):.3f}, Kurtosis: {stats.kurtosis(sigma):.3f}")
    
    # Test for bimodality (are there natural clusters?)
    from scipy.stats import kurtosis
    print("\n  Kurtosis interpretation:")
    print(f"    η: {stats.kurtosis(eta):.2f} ({'leptokurtic' if stats.kurtosis(eta) > 0 else 'platykurtic'})")
    print(f"    σ: {stats.kurtosis(sigma):.2f} ({'leptokurtic' if stats.kurtosis(sigma) > 0 else 'platykurtic'})")
    
    # Grid analysis
    print("\n4. GRID ANALYSIS (Continuous Quadrant Space)")
    print("-" * 40)
    
    # Create fine grid
    eta_bins = np.percentile(eta, [0, 25, 50, 75, 100])
    sigma_bins = np.percentile(sigma, [0, 25, 50, 75, 100])
    
    print("\nPercentile-based grid (natural data partitions):")
    print(f"η quartiles: {eta_bins}")
    print(f"σ quartiles: {sigma_bins}")
    
    # Count in each cell
    print("\nCell counts (η quartile × σ quartile):")
    print("        σ-Q1    σ-Q2    σ-Q3    σ-Q4")
    for i in range(4):
        row = f"η-Q{i+1}:  "
        for j in range(4):
            mask = ((eta >= eta_bins[i]) & (eta < eta_bins[i+1]) & 
                   (sigma >= sigma_bins[j]) & (sigma < sigma_bins[j+1]))
            if i == 3:  # Last bin includes upper bound
                mask = ((eta >= eta_bins[i]) & (eta <= eta_bins[i+1]) & 
                       (sigma >= sigma_bins[j]) & (sigma < sigma_bins[j+1]))
            if j == 3:
                mask = ((eta >= eta_bins[i]) & (eta < eta_bins[i+1]) & 
                       (sigma >= sigma_bins[j]) & (sigma <= sigma_bins[j+1]))
            if i == 3 and j == 3:
                mask = ((eta >= eta_bins[i]) & (eta <= eta_bins[i+1]) & 
                       (sigma >= sigma_bins[j]) & (sigma <= sigma_bins[j+1]))
            count = mask.sum()
            row += f"{count:6d}  "
        print(row)
    
    # What would "dissipative-like" look like in continuous space?
    print("\n5. CONTINUOUS 'DISSIPATIVENESS' SCORE")
    print("-" * 40)
    
    # Option A: Simple product (both high = high score)
    df['dissipativeness_A'] = (eta * sigma) / (eta.max() * sigma.max())
    
    # Option B: Standardized product (z-scores)
    eta_z = (eta - eta.mean()) / eta.std()
    sigma_z = (sigma - sigma.mean()) / sigma.std()
    df['dissipativeness_B'] = (eta_z * sigma_z)
    
    # Option C: Minimum of normalized values
    eta_norm = (eta - eta.min()) / (eta.max() - eta.min())
    sigma_norm = (sigma - sigma.min()) / (sigma.max() - sigma.min())
    df['dissipativeness_C'] = np.minimum(eta_norm, sigma_norm)
    
    print("\nTop 10 MSAs by 'dissipativeness' (Option A - simple product):")
    top_diss = df.nlargest(10, 'dissipativeness_A')[['msa_name', 'geodesic_efficiency', 
                                                      'mean_entropy_production', 'dissipativeness_A']]
    for _, row in top_diss.iterrows():
        print(f"  {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:6.1f} score={row['dissipativeness_A']:.3f}")
    
    print("\nTop 10 MSAs by 'dissipativeness' (Option C - minimum):")
    top_diss_c = df.nlargest(10, 'dissipativeness_C')[['msa_name', 'geodesic_efficiency', 
                                                        'mean_entropy_production', 'dissipativeness_C']]
    for _, row in top_diss_c.iterrows():
        print(f"  {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:6.1f} score={row['dissipativeness_C']:.3f}")
    
    # Create visualizations
    create_continuous_plots(df, eta, sigma, eta_bins, sigma_bins)
    
    return df

def create_continuous_plots(df, eta, sigma, eta_bins, sigma_bins):
    """Create plots showing continuous relationships."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Scatter plot with density
    ax1 = axes[0, 0]
    scatter = ax1.scatter(sigma, eta, c=eta * sigma, cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel('Entropy Production (σ)', fontsize=12)
    ax1.set_ylabel('Geodesic Efficiency (η)', fontsize=12)
    ax1.set_title('Continuous (η, σ) Space', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='η × σ')
    
    # Add percentile lines
    for q in [25, 50, 75]:
        ax1.axhline(np.percentile(eta, q), color='red', linestyle='--', alpha=0.3)
        ax1.axvline(np.percentile(sigma, q), color='red', linestyle='--', alpha=0.3)
    
    # Plot 2: Heatmap of density
    ax2 = axes[0, 1]
    from scipy.stats import gaussian_kde
    
    # Create grid
    xi = np.linspace(sigma.min(), sigma.max(), 100)
    yi = np.linspace(eta.min(), eta.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Kernel density estimation
    positions = np.vstack([xi_grid.ravel(), yi_grid.ravel()])
    values = np.vstack([sigma, eta])
    kernel = gaussian_kde(values)
    zi = np.reshape(kernel(positions).T, xi_grid.shape)
    
    im = ax2.imshow(zi, origin='lower', extent=[sigma.min(), sigma.max(), eta.min(), eta.max()],
                   cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Entropy Production (σ)', fontsize=12)
    ax2.set_ylabel('Geodesic Efficiency (η)', fontsize=12)
    ax2.set_title('Density Heatmap (KDE)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Density')
    
    # Plot 3: Dissipativeness scores
    ax3 = axes[1, 0]
    dissip_scores = np.minimum(
        (eta - eta.min()) / (eta.max() - eta.min()),
        (sigma - sigma.min()) / (sigma.max() - sigma.min())
    )
    scatter = ax3.scatter(sigma, eta, c=dissip_scores, cmap='RdYlGn', alpha=0.6, s=50)
    ax3.set_xlabel('Entropy Production (σ)', fontsize=12)
    ax3.set_ylabel('Geodesic Efficiency (η)', fontsize=12)
    ax3.set_title('Dissipativeness Score (min of normalized)', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Score')
    
    # Plot 4: Contour plot
    ax4 = axes[1, 1]
    contours = ax4.contour(xi_grid, yi_grid, zi, levels=10, colors='black', alpha=0.5)
    ax4.clabel(contours, inline=True, fontsize=8)
    ax4.scatter(sigma, eta, alpha=0.3, s=20)
    ax4.set_xlabel('Entropy Production (σ)', fontsize=12)
    ax4.set_ylabel('Geodesic Efficiency (η)', fontsize=12)
    ax4.set_title('Density Contours', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'continuous_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'continuous_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n\nContinuous analysis plots saved to: {output_dir / 'continuous_analysis.pdf'}")
    
    plt.close()

def compare_quadrant_vs_continuous():
    """Compare classification results: quadrants vs continuous."""
    df = load_data()
    
    print("\n" + "=" * 80)
    print("COMPARISON: QUADRANTS vs CONTINUOUS APPROACH")
    print("=" * 80)
    
    # Current quadrant classification
    eta_threshold = 0.7
    sigma_threshold = 30
    
    df['quadrant'] = 'Unknown'
    df.loc[(df['mean_entropy_production'] >= sigma_threshold) & 
           (df['geodesic_efficiency'] >= eta_threshold), 'quadrant'] = 'Dissipative'
    df.loc[(df['mean_entropy_production'] < sigma_threshold) & 
           (df['geodesic_efficiency'] >= eta_threshold), 'quadrant'] = 'Stable'
    df.loc[(df['mean_entropy_production'] >= sigma_threshold) & 
           (df['geodesic_efficiency'] < eta_threshold), 'quadrant'] = 'Forced'
    df.loc[(df['mean_entropy_production'] < sigma_threshold) & 
           (df['geodesic_efficiency'] < eta_threshold), 'quadrant'] = 'Stagnant'
    
    # Dissipativeness score (continuous)
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    eta_norm = (eta - eta.min()) / (eta.max() - eta.min())
    sigma_norm = (sigma - sigma.min()) / (sigma.max() - sigma.min())
    df['dissip_score'] = np.minimum(eta_norm, sigma_norm)
    
    # Compare top dissipative MSAs
    print("\nTop 15 'Dissipative' MSAs by QUADRANT classification:")
    top_quad = df[df['quadrant'] == 'Dissipative'].nlargest(15, 'geodesic_efficiency')
    for _, row in top_quad.iterrows():
        print(f"  {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:6.1f}")
    
    print("\nTop 15 'Dissipative' MSAs by CONTINUOUS score:")
    top_cont = df.nlargest(15, 'dissip_score')
    for _, row in top_cont.iterrows():
        quad = row['quadrant']
        marker = " ✓" if quad == 'Dissipative' else f" ({quad})"
        print(f"  {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:6.1f} score={row['dissip_score']:.3f}{marker}")
    
    # Agreement analysis
    top_20_quad = set(df[df['quadrant'] == 'Dissipative'].nlargest(20, 'geodesic_efficiency')['msa_name'])
    top_20_cont = set(df.nlargest(20, 'dissip_score')['msa_name'])
    
    agreement = len(top_20_quad & top_20_cont)
    print(f"\nAgreement between methods (top 20):")
    print(f"  {agreement}/20 MSAs appear in both lists ({100*agreement/20:.0f}% overlap)")
    
    # Which MSAs are in continuous top 20 but NOT quadrant dissipative?
    only_continuous = top_20_cont - top_20_quad
    if only_continuous:
        print(f"\nMSAs ranked high by CONTINUOUS score but NOT quadrant Dissipative:")
        for msa in only_continuous:
            row = df[df['msa_name'] == msa].iloc[0]
            print(f"  - {msa}: η={row['geodesic_efficiency']:.3f}, σ={row['mean_entropy_production']:.1f}, quadrant={row['quadrant']}")
    
    # Which MSAs are quadrant dissipative but NOT in continuous top 20?
    only_quadrant = top_20_quad - top_20_cont
    if only_quadrant:
        print(f"\nMSAs classified as Dissipative but NOT in continuous top 20:")
        for msa in only_quadrant:
            row = df[df['msa_name'] == msa].iloc[0]
            rank = df[df['msa_name'] == msa].index[0]
            cont_rank = (df['dissip_score'] > row['dissip_score']).sum() + 1
            print(f"  - {msa}: η={row['geodesic_efficiency']:.3f}, σ={row['mean_entropy_production']:.1f}, continuous rank={cont_rank}")

if __name__ == '__main__':
    import sys
    output_path = Path(__file__).parent.parent / 'results' / 'continuous_analysis_output.txt'
    output_path.parent.mkdir(exist_ok=True)
    
    class DualWriter:
        def __init__(self, stdout, file):
            self.stdout = stdout
            self.file = file
        def write(self, s):
            self.stdout.write(s)
            self.file.write(s)
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    
    f = open(output_path, 'w')
    old = sys.stdout
    sys.stdout = DualWriter(old, f)
    
    continuous_analysis()
    compare_quadrant_vs_continuous()
    
    sys.stdout = old
    f.close()
    
    print(f"\nFull output saved to: {output_path}")
