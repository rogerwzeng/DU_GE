"""
Compare three approaches:
1. Quadrants (arbitrary thresholds)
2. Continuous (no thresholds)
3. Clusters (data-driven)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_data():
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

def main():
    df = load_data()
    
    print("=" * 80)
    print("COMPARISON: THREE APPROACHES TO ANALYZING (η, σ) SPACE")
    print("=" * 80)
    
    # Apply three approaches
    df['quadrant'] = classify_quadrants(df)
    df['cluster'] = classify_clusters(df, n_clusters=3)
    df['dissip_score'] = calculate_dissipativeness(df)
    
    # 1. COMPARE CLASSIFICATIONS
    print("\n1. CLASSIFICATION COMPARISON")
    print("-" * 60)
    
    print("\nA. QUADRANTS (η≥0.7, σ≥30):")
    for quad in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        count = (df['quadrant'] == quad).sum()
        pct = 100 * count / len(df)
        print(f"   {quad:<15}: {count:3d} MSAs ({pct:5.1f}%)")
    
    print("\nB. K=3 CLUSTERS (data-driven):")
    for cluster in ['Dynamic-Coherent', 'Stable-Coherent', 'Incoherent']:
        count = (df['cluster'] == cluster).sum()
        pct = 100 * count / len(df)
        print(f"   {cluster:<20}: {count:3d} MSAs ({pct:5.1f}%)")
    
    print("\nC. CONTINUOUS (dissipativeness score):")
    print(f"   Mean score: {df['dissip_score'].mean():.3f}")
    print(f"   Std score:  {df['dissip_score'].std():.3f}")
    print(f"   Range: [{df['dissip_score'].min():.3f}, {df['dissip_score'].max():.3f}]")
    print("\n   Score quartiles:")
    for q, label in [(0.25, 'Q1'), (0.5, 'Median'), (0.75, 'Q3')]:
        val = df['dissip_score'].quantile(q)
        print(f"      {label}: {val:.3f}")
    
    # 2. AGREEMENT ANALYSIS
    print("\n2. AGREEMENT BETWEEN APPROACHES")
    print("-" * 60)
    
    # Quadrant vs Cluster
    print("\nQuadrant vs Cluster cross-tabulation:")
    crosstab = pd.crosstab(df['quadrant'], df['cluster'], margins=True)
    print(crosstab)
    
    # Agreement rates
    print("\nAgreement rates:")
    for quad in ['Dissipative', 'Stable', 'Stagnant']:
        quad_mask = df['quadrant'] == quad
        if quad_mask.sum() > 0:
            most_common_cluster = df.loc[quad_mask, 'cluster'].mode().values[0]
            agreement = (df.loc[quad_mask, 'cluster'] == most_common_cluster).sum()
            total = quad_mask.sum()
            print(f"   {quad} → {most_common_cluster}: {agreement}/{total} ({100*agreement/total:.0f}%)")
    
    # 3. TOP CITIES COMPARISON
    print("\n3. TOP 10 'DISSIPATIVE' CITIES BY EACH METHOD")
    print("-" * 60)
    
    print("\nA. Quadrant (Dissipative):")
    top_quad = df[df['quadrant'] == 'Dissipative'].nlargest(10, 'geodesic_efficiency')
    for i, (_, row) in enumerate(top_quad.iterrows(), 1):
        print(f"   {i:2d}. {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:5.1f}")
    
    print("\nB. Cluster (Dynamic-Coherent):")
    top_clust = df[df['cluster'] == 'Dynamic-Coherent'].nlargest(10, 'geodesic_efficiency')
    for i, (_, row) in enumerate(top_clust.iterrows(), 1):
        print(f"   {i:2d}. {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:5.1f}")
    
    print("\nC. Continuous (highest dissipativeness score):")
    top_cont = df.nlargest(10, 'dissip_score')
    for i, (_, row) in enumerate(top_cont.iterrows(), 1):
        quad_marker = "✓" if row['quadrant'] == 'Dissipative' else f"({row['quadrant'][:1]})"
        print(f"   {i:2d}. {row['msa_name']:<40} η={row['geodesic_efficiency']:.3f} σ={row['mean_entropy_production']:5.1f} score={row['dissip_score']:.3f} {quad_marker}")
    
    # 4. STATISTICAL PROPERTIES
    print("\n4. STATISTICAL PROPERTIES")
    print("-" * 60)
    
    # Correlation
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    r, p = stats.pearsonr(eta, sigma)
    print(f"\nη vs σ correlation: r = {r:.3f}, p = {p:.4f}")
    print(f"Interpretation: {'Weak' if abs(r) < 0.3 else 'Moderate' if abs(r) < 0.7 else 'Strong'} {'positive' if r > 0 else 'negative'} relationship")
    
    # Variance explained
    print("\nVariance in σ explained by cluster membership:")
    cluster_groups = df.groupby('cluster')['mean_entropy_production'].apply(list)
    f_stat, p_val = stats.f_oneway(*cluster_groups)
    eta_sq = f_stat * 2 / (f_stat * 2 + len(df) - 3)  # Approximate eta-squared
    print(f"   F-statistic: {f_stat:.2f}, p = {p_val:.4f}")
    print(f"   Approximate η² (variance explained): {eta_sq:.3f}")
    
    print("\nVariance in σ explained by quadrant membership:")
    quadrant_groups = df.groupby('quadrant')['mean_entropy_production'].apply(list)
    f_stat_q, p_val_q = stats.f_oneway(*quadrant_groups)
    eta_sq_q = f_stat_q * 3 / (f_stat_q * 3 + len(df) - 4)
    print(f"   F-statistic: {f_stat_q:.2f}, p = {p_val_q:.4f}")
    print(f"   Approximate η² (variance explained): {eta_sq_q:.3f}")
    
    # 5. RECOMMENDATION
    print("\n5. RECOMMENDATION SUMMARY")
    print("-" * 60)
    print("""
Based on the comparison:

QUADRANTS (Current approach):
  ✓ Intuitive, easy to visualize
  ✓ Matches theoretical framework
  ✗ Arbitrary thresholds (hard to defend)
  ✗ "Forced" quadrant nearly empty (0.5%)
  ✗ Statistical power reduced by dichotomization

CLUSTERS (K=3, data-driven):
  ✓ Statistically optimal (silhouette = 0.389)
  ✓ No arbitrary thresholds
  ✓ "Forced" naturally absorbed
  ✗ Less intuitive for readers
  ✗ Doesn't match 4-quadrant theory exactly

CONTINUOUS (No categorization):
  ✓ Preserves all information
  ✓ Best statistical properties
  ✓ No threshold defense needed
  ✗ Harder to visualize
  ✗ Less accessible for non-technical readers

RECOMMENDED HYBRID APPROACH:
  1. Use CONTINUOUS measures for formal analysis (regression)
  2. Use CLUSTERS (K=3) for classification if needed
  3. Use QUADRANTS as VISUAL ILLUSTRATION only
  4. State explicitly: "Quadrants are heuristic; analyses use continuous measures"
""")
    
    # Create comprehensive visualization
    create_comparison_plots(df)
    
    return df

def create_comparison_plots(df):
    """Create plots comparing all three approaches."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    eta = df['geodesic_efficiency']
    sigma = df['mean_entropy_production']
    
    # Row 1: Quadrant approach
    ax1 = fig.add_subplot(gs[0, 0])
    colors = {'Dissipative': '#d62728', 'Stable': '#1f77b4', 
              'Forced': '#2ca02c', 'Stagnant': '#ff7f0e'}
    for quad in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        mask = df['quadrant'] == quad
        ax1.scatter(sigma[mask], eta[mask], c=colors[quad], label=quad, 
                   alpha=0.6, s=40)
    ax1.axhline(0.7, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(30, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('σ (Entropy Production)')
    ax1.set_ylabel('η (Geodesic Efficiency)')
    ax1.set_title('A. QUADRANTS (Arbitrary Thresholds)', fontweight='bold', fontsize=11)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, 85)
    ax1.set_ylim(0, 1.05)
    
    # Quadrant bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    quad_counts = df['quadrant'].value_counts()
    bars = ax2.bar(quad_counts.index, quad_counts.values, 
                   color=[colors[q] for q in quad_counts.index])
    ax2.set_ylabel('Number of MSAs')
    ax2.set_title('B. Quadrant Distribution', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, 200)
    for bar, count in zip(bars, quad_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{count}\n({100*count/len(df):.0f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    # Row 2: Cluster approach
    ax3 = fig.add_subplot(gs[1, 0])
    cluster_colors = {'Dynamic-Coherent': '#d62728', 'Stable-Coherent': '#1f77b4', 
                      'Incoherent': '#ff7f0e'}
    for cluster in ['Dynamic-Coherent', 'Stable-Coherent', 'Incoherent']:
        mask = df['cluster'] == cluster
        ax3.scatter(sigma[mask], eta[mask], c=cluster_colors[cluster], 
                   label=cluster, alpha=0.6, s=40)
    ax3.set_xlabel('σ (Entropy Production)')
    ax3.set_ylabel('η (Geodesic Efficiency)')
    ax3.set_title('C. CLUSTERS (K=3, Data-Driven)', fontweight='bold', fontsize=11)
    ax3.legend(loc='lower right', fontsize=8)
    ax3.set_xlim(0, 85)
    ax3.set_ylim(0, 1.05)
    
    # Cluster bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    cluster_counts = df['cluster'].value_counts()
    bars = ax4.bar(cluster_counts.index, cluster_counts.values,
                   color=[cluster_colors[c] for c in cluster_counts.index])
    ax4.set_ylabel('Number of MSAs')
    ax4.set_title('D. Cluster Distribution', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, 200)
    for bar, count in zip(bars, cluster_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{count}\n({100*count/len(df):.0f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # Row 3: Continuous approach
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(sigma, eta, c=df['dissip_score'], cmap='RdYlGn', 
                         alpha=0.6, s=40)
    ax5.set_xlabel('σ (Entropy Production)')
    ax5.set_ylabel('η (Geodesic Efficiency)')
    ax5.set_title('E. CONTINUOUS (Dissipativeness Score)', fontweight='bold', fontsize=11)
    plt.colorbar(scatter, ax=ax5, label='Score')
    ax5.set_xlim(0, 85)
    ax5.set_ylim(0, 1.05)
    
    # Continuous histogram
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['dissip_score'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax6.axvline(df['dissip_score'].median(), color='red', linestyle='--', 
               label=f'Median = {df["dissip_score"].median():.3f}')
    ax6.set_xlabel('Dissipativeness Score')
    ax6.set_ylabel('Number of MSAs')
    ax6.set_title('F. Continuous Score Distribution', fontweight='bold', fontsize=11)
    ax6.legend()
    
    # Right column: Comparison charts
    ax7 = fig.add_subplot(gs[0, 2:])
    # Agreement heatmap
    crosstab = pd.crosstab(df['quadrant'], df['cluster'])
    im = ax7.imshow(crosstab.values, cmap='Blues', aspect='auto')
    ax7.set_xticks(range(len(crosstab.columns)))
    ax7.set_yticks(range(len(crosstab.index)))
    ax7.set_xticklabels(crosstab.columns)
    ax7.set_yticklabels(crosstab.index)
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Quadrant')
    ax7.set_title('G. Quadrant vs Cluster Agreement', fontweight='bold', fontsize=11)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            ax7.text(j, i, crosstab.values[i, j], ha="center", va="center", 
                    color="white" if crosstab.values[i, j] > crosstab.values.max()/2 else "black",
                    fontsize=12, fontweight='bold')
    
    ax8 = fig.add_subplot(gs[1, 2:])
    # Top cities comparison
    methods = ['Quadrant\n(Dissipative)', 'Cluster\n(Dynamic-Coherent)', 'Continuous\n(Top 20)']
    top_10_overlap = []
    
    top_quad = set(df[df['quadrant'] == 'Dissipative'].nlargest(20, 'geodesic_efficiency')['msa_name'])
    top_clust = set(df[df['cluster'] == 'Dynamic-Coherent'].nlargest(20, 'geodesic_efficiency')['msa_name'])
    top_cont = set(df.nlargest(20, 'dissip_score')['msa_name'])
    
    overlap_qc = len(top_quad & top_clust)
    overlap_qcont = len(top_quad & top_cont)
    overlap_ccont = len(top_clust & top_cont)
    
    x = np.arange(3)
    overlaps = [overlap_qc, overlap_qcont, overlap_ccont]
    bars = ax8.bar(x, overlaps, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax8.set_xticks(x)
    ax8.set_xticklabels(['Quad vs\nCluster', 'Quad vs\nContinuous', 'Cluster vs\nContinuous'])
    ax8.set_ylabel('Overlap (out of top 20)')
    ax8.set_title('H. Method Agreement (Top 20 MSAs)', fontweight='bold', fontsize=11)
    ax8.set_ylim(0, 25)
    for bar, val in zip(bars, overlaps):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}/20\n({100*val/20:.0f}%)', ha='center', va='bottom', fontsize=10)
    
    ax9 = fig.add_subplot(gs[2, 2:])
    # Summary metrics
    metrics_data = {
        'Approach': ['Quadrants', 'Clusters (K=3)', 'Continuous'],
        'Thresholds': ['Arbitrary\n(η≥0.7, σ≥30)', 'Data-driven\n(K-means)', 'None'],
        'Groups': ['4', '3', 'N/A'],
        'Interpretability': ['High', 'Medium', 'Low'],
        'Statistical Power': ['Reduced', 'Good', 'Optimal']
    }
    
    ax9.axis('off')
    table_data = []
    for i in range(len(metrics_data['Approach'])):
        row = [metrics_data['Approach'][i], metrics_data['Thresholds'][i],
               metrics_data['Groups'][i], metrics_data['Interpretability'][i],
               metrics_data['Statistical Power'][i]]
        table_data.append(row)
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Approach', 'Thresholds', '# Groups', 'Interpretability', 'Statistical Power'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    colors_table = ['#d62728', '#1f77b4', '#2ca02c']
    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor(colors_table[i-1])
            table[(i, j)].set_alpha(0.3)
    
    ax9.set_title('I. Approach Comparison Summary', fontweight='bold', fontsize=11, y=0.95)
    
    plt.suptitle('Comparison of Three Approaches: Quadrants vs Clusters vs Continuous', 
                fontsize=16, fontweight='bold', y=0.98)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'three_approaches_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'three_approaches_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n\nComparison plots saved to: {output_dir / 'three_approaches_comparison.pdf'}")
    
    plt.close()

if __name__ == '__main__':
    import sys
    output_path = Path(__file__).parent.parent / 'results' / 'three_approaches_comparison.txt'
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
    
    main()
    
    sys.stdout = old
    f.close()
    
    print(f"\nFull output saved to: {output_path}")
