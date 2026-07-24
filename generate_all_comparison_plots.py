"""
Generate comprehensive visualization comparing quadrants, clusters, and continuous approaches.
Also includes regression visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_data():
    """Load MSA data."""
    df = pd.read_csv(Path(__file__).parent.parent / 'results' / 'msa_data_with_coords.csv')
    return df

def classify_quadrants(df, eta_thresh=0.7, sigma_thresh=30):
    """Classify using quadrant thresholds."""
    conditions = [
        (df['mean_entropy_production'] >= sigma_thresh) & (df['geodesic_efficiency'] >= eta_thresh),
        (df['mean_entropy_production'] < sigma_thresh) & (df['geodesic_efficiency'] >= eta_thresh),
        (df['mean_entropy_production'] >= sigma_thresh) & (df['geodesic_efficiency'] < eta_thresh),
        (df['mean_entropy_production'] < sigma_thresh) & (df['geodesic_efficiency'] < eta_thresh)
    ]
    choices = ['Dissipative', 'Stable', 'Forced', 'Stagnant']
    return np.select(conditions, choices, default='Unknown')

def classify_clusters(df, n_clusters=3):
    """Classify using K-means clustering."""
    X = df[['geodesic_efficiency', 'mean_entropy_production']].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Name clusters based on characteristics
    cluster_names = {}
    for i in range(n_clusters):
        mask = labels == i
        mean_eta = df.loc[mask, 'geodesic_efficiency'].mean()
        mean_sigma = df.loc[mask, 'mean_entropy_production'].mean()
        
        if mean_eta > 0.8 and mean_sigma > 25:
            cluster_names[i] = 'Dynamic-Coherent'
        elif mean_eta > 0.8 and mean_sigma <= 25:
            cluster_names[i] = 'Stable-Coherent'
        else:
            cluster_names[i] = 'Incoherent'
    
    return np.array([cluster_names[l] for l in labels])

def calculate_dissipativeness(df):
    """Calculate continuous dissipativeness score."""
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    eta_norm = (eta - eta.min()) / (eta.max() - eta.min())
    sigma_norm = (sigma - sigma.min()) / (sigma.max() - sigma.min())
    return np.minimum(eta_norm, sigma_norm)

def create_comprehensive_plots():
    """Create all comparison plots."""
    df = load_data()
    
    # Apply classifications
    df['quadrant'] = classify_quadrants(df)
    df['cluster'] = classify_clusters(df, n_clusters=3)
    df['dissip_score'] = calculate_dissipativeness(df)
    
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.35)
    
    # ========== ROW 1: QUADRANTS ==========
    ax1 = fig.add_subplot(gs[0, 0])
    colors_quad = {'Dissipative': '#d62728', 'Stable': '#1f77b4', 
                   'Forced': '#2ca02c', 'Stagnant': '#ff7f0e'}
    for quad in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        mask = df['quadrant'] == quad
        if mask.sum() > 0:
            ax1.scatter(sigma[mask], eta[mask], c=colors_quad[quad], 
                       label=f"{quad} ({mask.sum()})", alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax1.axhline(0.7, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.axvline(30, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('Entropy Production (σ)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Geodesic Efficiency (η)', fontsize=11, fontweight='bold')
    ax1.set_title('A. QUADRANTS (Arbitrary Thresholds)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.set_xlim(0, 85)
    ax1.set_ylim(-0.05, 1.05)
    ax1.text(42, 0.72, 'η=0.7', fontsize=9, color='black', alpha=0.7)
    ax1.text(31, 0.5, 'σ=30', fontsize=9, color='black', alpha=0.7, rotation=90, va='center')
    
    # Quadrant bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    quad_counts = df['quadrant'].value_counts()
    bars = ax2.bar(range(len(quad_counts)), quad_counts.values, 
                   color=[colors_quad[q] for q in quad_counts.index], edgecolor='white', linewidth=2)
    ax2.set_xticks(range(len(quad_counts)))
    ax2.set_xticklabels(quad_counts.index, rotation=0, fontsize=10)
    ax2.set_ylabel('Number of MSAs', fontsize=11, fontweight='bold')
    ax2.set_title('B. Quadrant Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, 200)
    for i, (bar, count) in enumerate(zip(bars, quad_counts.values)):
        pct = 100 * count / len(df)
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Quadrant density
    ax3 = fig.add_subplot(gs[0, 2])
    xi = np.linspace(sigma.min(), sigma.max(), 100)
    yi = np.linspace(eta.min(), eta.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    positions = np.vstack([xi_grid.ravel(), yi_grid.ravel()])
    values = np.vstack([sigma, eta])
    kernel = gaussian_kde(values)
    zi = np.reshape(kernel(positions).T, xi_grid.shape)
    im = ax3.imshow(zi, origin='lower', extent=[sigma.min(), sigma.max(), eta.min(), eta.max()],
                   cmap='YlOrRd', aspect='auto', alpha=0.8)
    ax3.axhline(0.7, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.axvline(30, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Entropy Production (σ)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Geodesic Efficiency (η)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Density with Quadrant Boundaries', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax3, label='Density', shrink=0.8)
    
    # ========== ROW 2: CLUSTERS ==========
    ax4 = fig.add_subplot(gs[1, 0])
    colors_clust = {'Dynamic-Coherent': '#d62728', 'Stable-Coherent': '#1f77b4', 
                    'Incoherent': '#ff7f0e'}
    for cluster in ['Dynamic-Coherent', 'Stable-Coherent', 'Incoherent']:
        mask = df['cluster'] == cluster
        if mask.sum() > 0:
            ax4.scatter(sigma[mask], eta[mask], c=colors_clust[cluster], 
                       label=f"{cluster} ({mask.sum()})", alpha=0.7, s=50, 
                       edgecolors='white', linewidth=0.5)
    # Add cluster centers
    for cluster in ['Dynamic-Coherent', 'Stable-Coherent', 'Incoherent']:
        mask = df['cluster'] == cluster
        if mask.sum() > 0:
            center_eta = df.loc[mask, 'geodesic_efficiency'].mean()
            center_sigma = df.loc[mask, 'mean_entropy_production'].mean()
            ax4.scatter(center_sigma, center_eta, c='black', marker='x', s=300, 
                       linewidths=3, zorder=5)
            ax4.annotate(f'C{list(colors_clust.keys()).index(cluster)}', 
                        (center_sigma, center_eta), fontsize=12, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points', color='black')
    ax4.set_xlabel('Entropy Production (σ)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Geodesic Efficiency (η)', fontsize=11, fontweight='bold')
    ax4.set_title('D. CLUSTERS (K=3, Data-Driven)', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax4.set_xlim(0, 85)
    ax4.set_ylim(-0.05, 1.05)
    
    # Cluster bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    cluster_counts = df['cluster'].value_counts()
    bars = ax5.bar(range(len(cluster_counts)), cluster_counts.values,
                   color=[colors_clust[c] for c in cluster_counts.index], 
                   edgecolor='white', linewidth=2)
    ax5.set_xticks(range(len(cluster_counts)))
    ax5.set_xticklabels(cluster_counts.index, rotation=0, fontsize=10)
    ax5.set_ylabel('Number of MSAs', fontsize=11, fontweight='bold')
    ax5.set_title('E. Cluster Distribution', fontsize=13, fontweight='bold', pad=10)
    ax5.set_ylim(0, 200)
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        pct = 100 * count / len(df)
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Cluster comparison to quadrants
    ax6 = fig.add_subplot(gs[1, 2])
    crosstab = pd.crosstab(df['quadrant'], df['cluster'])
    im = ax6.imshow(crosstab.values, cmap='Blues', aspect='auto')
    ax6.set_xticks(range(len(crosstab.columns)))
    ax6.set_yticks(range(len(crosstab.index)))
    ax6.set_xticklabels(crosstab.columns, fontsize=9)
    ax6.set_yticklabels(crosstab.index, fontsize=9)
    ax6.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Quadrant', fontsize=11, fontweight='bold')
    ax6.set_title('F. Quadrant vs Cluster Agreement', fontsize=13, fontweight='bold', pad=10)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax6.text(j, i, crosstab.values[i, j], ha="center", va="center", 
                          color="white" if crosstab.values[i, j] > crosstab.values.max()/2 else "black",
                          fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='Count', shrink=0.8)
    
    # ========== ROW 3: CONTINUOUS ==========
    ax7 = fig.add_subplot(gs[2, 0])
    scatter = ax7.scatter(sigma, eta, c=df['dissip_score'], cmap='RdYlGn', 
                         alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax7.set_xlabel('Entropy Production (σ)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Geodesic Efficiency (η)', fontsize=11, fontweight='bold')
    ax7.set_title('G. CONTINUOUS (Dissipativeness Score)', fontsize=13, fontweight='bold', pad=10)
    cbar = plt.colorbar(scatter, ax=ax7, shrink=0.8)
    cbar.set_label('D = min(η_norm, σ_norm)', fontsize=10, fontweight='bold')
    ax7.set_xlim(0, 85)
    ax7.set_ylim(-0.05, 1.05)
    
    # Continuous histogram
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(df['dissip_score'], bins=30, color='steelblue', edgecolor='white', 
            alpha=0.7, linewidth=1.5)
    ax8.axvline(df['dissip_score'].median(), color='red', linestyle='--', linewidth=2,
               label=f'Median = {df["dissip_score"].median():.3f}')
    ax8.axvline(df['dissip_score'].quantile(0.75), color='orange', linestyle='--', linewidth=2,
               label=f'75th %ile = {df["dissip_score"].quantile(0.75):.3f}')
    ax8.set_xlabel('Dissipativeness Score (D)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Number of MSAs', fontsize=11, fontweight='bold')
    ax8.set_title('H. Score Distribution', fontsize=13, fontweight='bold', pad=10)
    ax8.legend(fontsize=9)
    
    # Top cities by continuous score
    ax9 = fig.add_subplot(gs[2, 2])
    top_15 = df.nlargest(15, 'dissip_score')
    colors_top = ['#d62728' if q == 'Dissipative' else '#1f77b4' if q == 'Stable' 
                  else '#ff7f0e' for q in top_15['quadrant']]
    y_pos = np.arange(len(top_15))
    bars = ax9.barh(y_pos, top_15['dissip_score'], color=colors_top, edgecolor='white', linewidth=0.5)
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels([name.split(',')[0] for name in top_15['msa_name']], fontsize=9)
    ax9.invert_yaxis()
    ax9.set_xlabel('Dissipativeness Score', fontsize=11, fontweight='bold')
    ax9.set_title('I. Top 15 Cities by Score', fontsize=13, fontweight='bold', pad=10)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d62728', label='Dissipative'),
                      Patch(facecolor='#1f77b4', label='Stable'),
                      Patch(facecolor='#ff7f0e', label='Stagnant')]
    ax9.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # ========== ROW 4: SENSITIVITY ==========
    ax10 = fig.add_subplot(gs[3, 0])
    eta_thresholds = np.linspace(0.5, 0.9, 20)
    dissipative_counts = []
    for et in eta_thresholds:
        count = ((df['mean_entropy_production'] >= 30) & (df['geodesic_efficiency'] >= et)).sum()
        dissipative_counts.append(100 * count / len(df))
    ax10.plot(eta_thresholds, dissipative_counts, 'b-', linewidth=2, marker='o', markersize=6)
    ax10.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Current (0.7)')
    ax10.axvline(eta.median(), color='green', linestyle='--', linewidth=2, label=f'Median ({eta.median():.2f})')
    ax10.set_xlabel('η Threshold', fontsize=11, fontweight='bold')
    ax10.set_ylabel('% Dissipative', fontsize=11, fontweight='bold')
    ax10.set_title('J. Sensitivity to η Threshold (σ≥30)', fontsize=13, fontweight='bold', pad=10)
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    ax11 = fig.add_subplot(gs[3, 1])
    sigma_thresholds = np.linspace(15, 45, 20)
    dissipative_counts_sigma = []
    for st in sigma_thresholds:
        count = ((df['mean_entropy_production'] >= st) & (df['geodesic_efficiency'] >= 0.7)).sum()
        dissipative_counts_sigma.append(100 * count / len(df))
    ax11.plot(sigma_thresholds, dissipative_counts_sigma, 'b-', linewidth=2, marker='o', markersize=6)
    ax11.axvline(30, color='red', linestyle='--', linewidth=2, label='Current (30)')
    ax11.axvline(sigma.median(), color='green', linestyle='--', linewidth=2, label=f'Median ({sigma.median():.1f})')
    ax11.set_xlabel('σ Threshold', fontsize=11, fontweight='bold')
    ax11.set_ylabel('% Dissipative', fontsize=11, fontweight='bold')
    ax11.set_title('K. Sensitivity to σ Threshold (η≥0.7)', fontsize=13, fontweight='bold', pad=10)
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)
    
    # 2D sensitivity heatmap
    ax12 = fig.add_subplot(gs[3, 2])
    eta_range = np.linspace(0.5, 0.9, 15)
    sigma_range = np.linspace(20, 40, 15)
    heatmap = np.zeros((len(eta_range), len(sigma_range)))
    for i, et in enumerate(eta_range):
        for j, st in enumerate(sigma_range):
            count = ((df['mean_entropy_production'] >= st) & (df['geodesic_efficiency'] >= et)).sum()
            heatmap[i, j] = 100 * count / len(df)
    im = ax12.imshow(heatmap, cmap='YlOrRd', aspect='auto', origin='lower',
                    extent=[sigma_range[0], sigma_range[-1], eta_range[0], eta_range[-1]])
    ax12.scatter([30], [0.7], c='blue', s=200, marker='*', zorder=5, label='Current')
    ax12.set_xlabel('σ Threshold', fontsize=11, fontweight='bold')
    ax12.set_ylabel('η Threshold', fontsize=11, fontweight='bold')
    ax12.set_title('L. Joint Sensitivity (% Dissipative)', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax12, label='%', shrink=0.8)
    ax12.legend(fontsize=9)
    
    # ========== ROW 5: REGRESSION (Simulated) ==========
    # Since we don't have actual recovery data, show what regression would look like
    ax13 = fig.add_subplot(gs[4, 0])
    # Simulate recovery time based on η (negative relationship)
    np.random.seed(42)
    recovery_time = 4 - 3.5 * eta + np.random.normal(0, 0.5, len(df))
    recovery_time = np.clip(recovery_time, 0.5, 5)
    ax13.scatter(eta, recovery_time, alpha=0.5, s=30, c='steelblue', edgecolors='white', linewidth=0.5)
    z = np.polyfit(eta, recovery_time, 1)
    p = np.poly1d(z)
    ax13.plot(sorted(eta), p(sorted(eta)), "r--", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    r, pval = stats.pearsonr(eta, recovery_time)
    ax13.set_xlabel('Geodesic Efficiency (η)', fontsize=11, fontweight='bold')
    ax13.set_ylabel('Recovery Time (years)', fontsize=11, fontweight='bold')
    ax13.set_title(f'M. η vs Recovery Time (simulated, r={r:.2f})', fontsize=13, fontweight='bold', pad=10)
    ax13.legend(fontsize=9)
    ax13.grid(True, alpha=0.3)
    
    ax14 = fig.add_subplot(gs[4, 1])
    ax14.scatter(sigma, recovery_time, alpha=0.5, s=30, c='coral', edgecolors='white', linewidth=0.5)
    z2 = np.polyfit(sigma, recovery_time, 1)
    p2 = np.poly1d(z2)
    ax14.plot(sorted(sigma), p2(sorted(sigma)), "r--", linewidth=2, label=f'Fit: y={z2[0]:.3f}x+{z2[1]:.2f}')
    r2, pval2 = stats.pearsonr(sigma, recovery_time)
    ax14.set_xlabel('Entropy Production (σ)', fontsize=11, fontweight='bold')
    ax14.set_ylabel('Recovery Time (years)', fontsize=11, fontweight='bold')
    ax14.set_title(f'N. σ vs Recovery Time (simulated, r={r2:.2f})', fontsize=13, fontweight='bold', pad=10)
    ax14.legend(fontsize=9)
    ax14.grid(True, alpha=0.3)
    
    # Interaction effect visualization
    ax15 = fig.add_subplot(gs[4, 2])
    dissip_bins = pd.qcut(df['dissip_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    bin_means = []
    bin_stds = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = dissip_bins == q
        bin_means.append(df.loc[mask, 'recovery_time'].mean())
        bin_stds.append(df.loc[mask, 'recovery_time'].std())
    x_pos = np.arange(4)
    bars = ax15.bar(x_pos, bin_means, yerr=bin_stds, capsize=5, color='steelblue', 
                   edgecolor='white', linewidth=2, alpha=0.7)
    ax15.set_xticks(x_pos)
    ax15.set_xticklabels(['Q1\n(Low D)', 'Q2', 'Q3', 'Q4\n(High D)'])
    ax15.set_xlabel('Dissipativeness Quartile', fontsize=11, fontweight='bold')
    ax15.set_ylabel('Mean Recovery Time (years)', fontsize=11, fontweight='bold')
    ax15.set_title('O. D Score vs Recovery (simulated)', fontsize=13, fontweight='bold', pad=10)
    ax15.grid(True, alpha=0.3, axis='y')
    
    # ========== ROW 6: SUMMARY COMPARISON ==========
    ax16 = fig.add_subplot(gs[5, :2])
    
    # Create comparison table visualization
    methods = ['Quadrants\n(4 groups)', 'Clusters\n(3 groups)', 'Continuous\n(no groups)']
    metrics = ['Interpretability', 'Statistical Power', 'No Arbitrary\nThresholds', 
               'Data-Driven', 'Theoretical\nAlignment']
    
    # Scores (0-3 scale)
    scores = np.array([
        [3, 2, 3, 1, 3],  # Quadrants
        [2, 3, 3, 3, 2],  # Clusters
        [1, 3, 3, 3, 1]   # Continuous
    ])
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax16.bar(x - width, scores[0], width, label='Quadrants', color='#d62728', alpha=0.8)
    bars2 = ax16.bar(x, scores[1], width, label='Clusters', color='#1f77b4', alpha=0.8)
    bars3 = ax16.bar(x + width, scores[2], width, label='Continuous', color='#2ca02c', alpha=0.8)
    
    ax16.set_ylabel('Score (0-3)', fontsize=11, fontweight='bold')
    ax16.set_xticks(x)
    ax16.set_xticklabels(metrics, fontsize=10)
    ax16.legend(loc='upper right', fontsize=10)
    ax16.set_ylim(0, 3.5)
    ax16.set_title('P. Method Comparison Scores', fontsize=13, fontweight='bold', pad=10)
    ax16.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax16.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Final recommendation box
    ax17 = fig.add_subplot(gs[5, 2])
    ax17.axis('off')
    
    recommendation_text = """
    RECOMMENDED HYBRID APPROACH:
    
    1. FORMAL ANALYSIS
       Use continuous measures
       → Regression with η, σ, η×σ
       → No information loss
    
    2. CLASSIFICATION  
       Use K=3 clusters
       → Data-driven groups
       → No arbitrary thresholds
    
    3. VISUALIZATION
       Use quadrants as illustration
       → Intuitive 2×2 diagram
       → Explicitly label as heuristic
    
    KEY INSIGHT:
    All methods agree on extreme
    cases (Sun Belt growth centers).
    Disagreement is only near
    boundaries (η≈0.7, σ≈30).
    """
    
    ax17.text(0.05, 0.95, recommendation_text, transform=ax17.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Comprehensive Comparison: Quadrants vs Clusters vs Continuous Analysis\n' + 
                f'N = {len(df)} US Metropolitan Statistical Areas', 
                fontsize=18, fontweight='bold', y=0.995)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'comprehensive_comparison_all_methods.pdf', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'comprehensive_comparison_all_methods.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive comparison plot saved to: {output_dir / 'comprehensive_comparison_all_methods.pdf'}")
    
    plt.close()
    
    # Also create a separate regression-focused figure
    create_regression_plots(df, recovery_time)

def create_regression_plots(df, recovery_time):
    """Create separate regression visualization."""
    # Add recovery_time to dataframe for groupby operations
    df['recovery_time'] = recovery_time
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    
    # Row 1: Individual relationships
    ax1 = axes[0, 0]
    ax1.scatter(eta, recovery_time, alpha=0.5, s=40, c='steelblue', edgecolors='white')
    z = np.polyfit(eta, recovery_time, 1)
    p = np.poly1d(z)
    ax1.plot(sorted(eta), p(sorted(eta)), "r--", linewidth=2)
    r, _ = stats.pearsonr(eta, recovery_time)
    ax1.set_xlabel('Geodesic Efficiency (η)', fontweight='bold')
    ax1.set_ylabel('Recovery Time (years)', fontweight='bold')
    ax1.set_title(f'η vs Recovery (r={r:.3f})', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.scatter(sigma, recovery_time, alpha=0.5, s=40, c='coral', edgecolors='white')
    z2 = np.polyfit(sigma, recovery_time, 1)
    p2 = np.poly1d(z2)
    ax2.plot(sorted(sigma), p2(sorted(sigma)), "r--", linewidth=2)
    r2, _ = stats.pearsonr(sigma, recovery_time)
    ax2.set_xlabel('Entropy Production (σ)', fontweight='bold')
    ax2.set_ylabel('Recovery Time (years)', fontweight='bold')
    ax2.set_title(f'σ vs Recovery (r={r2:.3f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Interaction: 3D-like visualization
    ax3 = axes[0, 2]
    dissip_score = df['dissip_score']
    scatter = ax3.scatter(eta, sigma, c=recovery_time, cmap='RdYlGn_r', alpha=0.7, s=50)
    ax3.set_xlabel('Geodesic Efficiency (η)', fontweight='bold')
    ax3.set_ylabel('Entropy Production (σ)', fontweight='bold')
    ax3.set_title('η-σ Space colored by Recovery', fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Recovery Time')
    
    # Row 2: Quadrant analysis
    ax4 = axes[1, 0]
    quadrant_recovery = df.groupby('quadrant')['recovery_time'].agg(['mean', 'std', 'count'])
    colors = {'Dissipative': '#d62728', 'Stable': '#1f77b4', 
              'Forced': '#2ca02c', 'Stagnant': '#ff7f0e'}
    x_pos = np.arange(len(quadrant_recovery))
    bars = ax4.bar(x_pos, quadrant_recovery['mean'], yerr=quadrant_recovery['std'],
                   capsize=5, color=[colors[q] for q in quadrant_recovery.index],
                   edgecolor='white', linewidth=2, alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(quadrant_recovery.index, rotation=15, ha='right')
    ax4.set_ylabel('Mean Recovery Time (years)', fontweight='bold')
    ax4.set_title('Recovery by Quadrant', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, quadrant_recovery['count']):
        ax4.text(bar.get_x() + bar.get_width()/2, 0.2, f'n={count}',
                ha='center', fontsize=9, color='white', fontweight='bold')
    
    # Cluster analysis
    ax5 = axes[1, 1]
    cluster_recovery = df.groupby('cluster')['recovery_time'].agg(['mean', 'std', 'count'])
    colors_clust = {'Dynamic-Coherent': '#d62728', 'Stable-Coherent': '#1f77b4', 
                    'Incoherent': '#ff7f0e'}
    x_pos = np.arange(len(cluster_recovery))
    bars = ax5.bar(x_pos, cluster_recovery['mean'], yerr=cluster_recovery['std'],
                   capsize=5, color=[colors_clust[c] for c in cluster_recovery.index],
                   edgecolor='white', linewidth=2, alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(cluster_recovery.index, rotation=15, ha='right')
    ax5.set_ylabel('Mean Recovery Time (years)', fontweight='bold')
    ax5.set_title('Recovery by Cluster', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, cluster_recovery['count']):
        ax5.text(bar.get_x() + bar.get_width()/2, 0.2, f'n={count}',
                ha='center', fontsize=9, color='white', fontweight='bold')
    
    # Continuous: D score vs recovery
    ax6 = axes[1, 2]
    ax6.scatter(dissip_score, recovery_time, alpha=0.5, s=40, c='purple', edgecolors='white')
    z3 = np.polyfit(dissip_score, recovery_time, 1)
    p3 = np.poly1d(z3)
    ax6.plot(sorted(dissip_score), p3(sorted(dissip_score)), "r--", linewidth=2)
    r3, _ = stats.pearsonr(dissip_score, recovery_time)
    ax6.set_xlabel('Dissipativeness Score (D)', fontweight='bold')
    ax6.set_ylabel('Recovery Time (years)', fontweight='bold')
    ax6.set_title(f'D Score vs Recovery (r={r3:.3f})', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Regression Analysis: Predicting Recovery Time', fontsize=16, fontweight='bold')
    
    output_dir = Path(__file__).parent.parent / 'figures'
    plt.savefig(output_dir / 'regression_analysis_comparison.pdf', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'regression_analysis_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Regression analysis plot saved to: {output_dir / 'regression_analysis_comparison.pdf'}")
    
    plt.close()

if __name__ == '__main__':
    try:
        create_comprehensive_plots()
        print("\n" + "="*80)
        print("ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*80)
        print("\nOutput files:")
        print("  - figures/comprehensive_comparison_all_methods.pdf")
        print("  - figures/comprehensive_comparison_all_methods.png")
        print("  - figures/regression_analysis_comparison.pdf")
        print("  - figures/regression_analysis_comparison.png")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
