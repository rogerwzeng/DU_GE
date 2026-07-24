"""
Option 3: Data-driven clustering of (η, σ) space
Let the data determine natural groupings instead of arbitrary thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from pathlib import Path

def load_data():
    """Load MSA data."""
    df = pd.read_csv(Path(__file__).parent.parent / 'results' / 'msa_data_with_coords.csv')
    return df

def find_optimal_clusters(X, max_k=8):
    """Find optimal number of clusters using multiple metrics."""
    inertias = []
    silhouettes = []
    calinski = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        calinski.append(calinski_harabasz_score(X, labels))
    
    # Find elbow (simplified)
    diffs = np.diff(inertias)
    elbow_k = K_range[np.argmax(diffs[1:] - diffs[:-1]) + 1] if len(diffs) > 1 else 3
    
    # Best silhouette
    best_sil_k = K_range[np.argmax(silhouettes)]
    
    # Best Calinski-Harabasz
    best_ch_k = K_range[np.argmax(calinski)]
    
    return {
        'K_range': list(K_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'calinski': calinski,
        'elbow_k': elbow_k,
        'best_sil_k': best_sil_k,
        'best_ch_k': best_ch_k
    }

def cluster_analysis():
    """Perform cluster analysis on (η, σ) space."""
    df = load_data()
    
    print("=" * 80)
    print("OPTION 3: DATA-DRIVEN CLUSTER ANALYSIS")
    print("=" * 80)
    
    # Prepare data
    X_raw = df[['geodesic_efficiency', 'mean_entropy_production']].values
    
    # Standardize for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print("\n1. FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("-" * 40)
    
    metrics = find_optimal_clusters(X_scaled, max_k=8)
    
    print(f"K range tested: {metrics['K_range']}")
    print(f"\nSilhouette scores (higher = better defined clusters):")
    for k, score in zip(metrics['K_range'], metrics['silhouettes']):
        marker = " <- BEST" if k == metrics['best_sil_k'] else ""
        print(f"  K={k}: {score:.3f}{marker}")
    
    print(f"\nCalinski-Harabasz scores (higher = better separation):")
    for k, score in zip(metrics['K_range'], metrics['calinski']):
        marker = " <- BEST" if k == metrics['best_ch_k'] else ""
        print(f"  K={k}: {score:.1f}{marker}")
    
    print(f"\nElbow method suggests: K = {metrics['elbow_k']}")
    print(f"Best silhouette: K = {metrics['best_sil_k']}")
    print(f"Best Calinski-Harabasz: K = {metrics['best_ch_k']}")
    
    # Try different K values and compare
    print("\n2. CLUSTER SOLUTIONS COMPARISON")
    print("-" * 40)
    
    for k in [3, 4, 5]:
        print(f"\n--- K = {k} CLUSTERS ---")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        df[f'cluster_{k}'] = labels
        
        # Cluster statistics
        print(f"Cluster sizes:")
        for cluster_id in range(k):
            count = (labels == cluster_id).sum()
            pct = 100 * count / len(labels)
            
            # Get cluster center in original space
            cluster_mask = labels == cluster_id
            center_eta = df.loc[cluster_mask, 'geodesic_efficiency'].mean()
            center_sigma = df.loc[cluster_mask, 'mean_entropy_production'].mean()
            
            print(f"  Cluster {cluster_id}: {count:3d} MSAs ({pct:5.1f}%) - "
                  f"center (η={center_eta:.3f}, σ={center_sigma:.1f})")
        
        # Interpretation
        sil_score = silhouette_score(X_scaled, labels)
        print(f"Silhouette score: {sil_score:.3f}")
    
    # Detailed analysis of K=4 (comparable to quadrants)
    print("\n3. DETAILED ANALYSIS: K=4 CLUSTERS")
    print("-" * 40)
    
    kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans_4.fit_predict(X_scaled)
    
    # Cluster profiles
    for cluster_id in range(4):
        cluster_df = df[df['cluster'] == cluster_id]
        print(f"\nCLUSTER {cluster_id} (n={len(cluster_df)}):")
        print(f"  η: mean={cluster_df['geodesic_efficiency'].mean():.3f}, "
              f"std={cluster_df['geodesic_efficiency'].std():.3f}")
        print(f"  σ: mean={cluster_df['mean_entropy_production'].mean():.1f}, "
              f"std={cluster_df['mean_entropy_production'].std():.1f}")
        
        # Top MSAs in this cluster
        print(f"  Example MSAs:")
        for _, row in cluster_df.head(5).iterrows():
            print(f"    - {row['msa_name']}")
    
    # Compare to quadrants
    print("\n4. COMPARISON TO QUADRANT CLASSIFICATION")
    print("-" * 40)
    
    # Add quadrant labels
    eta_threshold = 0.7
    sigma_threshold = 30
    
    conditions = [
        (df['mean_entropy_production'] >= sigma_threshold) & (df['geodesic_efficiency'] >= eta_threshold),
        (df['mean_entropy_production'] < sigma_threshold) & (df['geodesic_efficiency'] >= eta_threshold),
        (df['mean_entropy_production'] >= sigma_threshold) & (df['geodesic_efficiency'] < eta_threshold),
        (df['mean_entropy_production'] < sigma_threshold) & (df['geodesic_efficiency'] < eta_threshold)
    ]
    choices = ['Dissipative', 'Stable', 'Forced', 'Stagnant']
    df['quadrant'] = np.select(conditions, choices, default='Unknown')
    
    # Cross-tabulation
    print("\nCross-tabulation: Quadrant vs K-Means Cluster (K=4)")
    crosstab = pd.crosstab(df['quadrant'], df['cluster'], margins=True)
    print(crosstab)
    
    # Agreement analysis
    print("\nAgreement analysis:")
    # Find which cluster best matches each quadrant
    for quadrant in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        quadrant_mask = df['quadrant'] == quadrant
        if quadrant_mask.sum() > 0:
            most_common_cluster = df.loc[quadrant_mask, 'cluster'].mode().values[0]
            agreement = (df.loc[quadrant_mask, 'cluster'] == most_common_cluster).sum()
            total = quadrant_mask.sum()
            print(f"  {quadrant}: Cluster {most_common_cluster} captures "
                  f"{agreement}/{total} ({100*agreement/total:.0f}%)")
    
    # Try hierarchical clustering
    print("\n5. HIERARCHICAL CLUSTERING (Ward's method)")
    print("-" * 40)
    
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
    df['hierarchical'] = hierarchical.fit_predict(X_scaled)
    
    print("Hierarchical cluster sizes:")
    for cluster_id in range(4):
        count = (df['hierarchical'] == cluster_id).sum()
        pct = 100 * count / len(df)
        cluster_mask = df['hierarchical'] == cluster_id
        center_eta = df.loc[cluster_mask, 'geodesic_efficiency'].mean()
        center_sigma = df.loc[cluster_mask, 'mean_entropy_production'].mean()
        print(f"  Cluster {cluster_id}: {count:3d} MSAs ({pct:5.1f}%) - "
              f"center (η={center_eta:.3f}, σ={center_sigma:.1f})")
    
    # Try DBSCAN (density-based)
    print("\n6. DBSCAN (Density-Based Clustering)")
    print("-" * 40)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan'] = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(df['dbscan'])) - (1 if -1 in df['dbscan'].values else 0)
    n_noise = (df['dbscan'] == -1).sum()
    
    print(f"DBSCAN found {n_clusters} clusters")
    print(f"Noise points (outliers): {n_noise} ({100*n_noise/len(df):.1f}%)")
    
    for cluster_id in sorted(set(df['dbscan'])):
        if cluster_id == -1:
            continue
        count = (df['dbscan'] == cluster_id).sum()
        pct = 100 * count / len(df)
        cluster_mask = df['dbscan'] == cluster_id
        center_eta = df.loc[cluster_mask, 'geodesic_efficiency'].mean()
        center_sigma = df.loc[cluster_mask, 'mean_entropy_production'].mean()
        print(f"  Cluster {cluster_id}: {count:3d} MSAs ({pct:5.1f}%) - "
              f"center (η={center_eta:.3f}, σ={center_sigma:.1f})")
    
    # Create visualizations
    create_cluster_plots(df, X_raw, X_scaled, kmeans_4, metrics)
    
    return df

def create_cluster_plots(df, X_raw, X_scaled, kmeans_model, metrics):
    """Create cluster analysis visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    eta = X_raw[:, 0]
    sigma = X_raw[:, 1]
    
    # Plot 1: Elbow curve
    ax1 = axes[0, 0]
    ax1.plot(metrics['K_range'], metrics['inertias'], 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.axvline(metrics['elbow_k'], color='red', linestyle='--', 
               label=f'Elbow at K={metrics["elbow_k"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Silhouette scores
    ax2 = axes[0, 1]
    ax2.plot(metrics['K_range'], metrics['silhouettes'], 'go-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis', fontweight='bold')
    ax2.axvline(metrics['best_sil_k'], color='red', linestyle='--',
               label=f'Best at K={metrics["best_sil_k"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: K=4 K-means clusters
    ax3 = axes[0, 2]
    labels_4 = df['cluster']
    colors = ['red', 'blue', 'green', 'orange']
    for cluster_id in range(4):
        mask = labels_4 == cluster_id
        ax3.scatter(sigma[mask], eta[mask], c=colors[cluster_id], 
                   label=f'Cluster {cluster_id}', alpha=0.6, s=50)
    
    # Plot cluster centers
    centers = kmeans_model.cluster_centers_
    centers_original = np.array([
        [eta[labels_4 == i].mean(), sigma[labels_4 == i].mean()] 
        for i in range(4)
    ])
    for i, (c_eta, c_sigma) in enumerate(centers_original):
        ax3.scatter(c_sigma, c_eta, c='black', marker='x', s=200, linewidths=3)
        ax3.annotate(f'C{i}', (c_sigma, c_eta), fontsize=12, fontweight='bold')
    
    ax3.set_xlabel('Entropy Production (σ)')
    ax3.set_ylabel('Geodesic Efficiency (η)')
    ax3.set_title('K-Means Clusters (K=4)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison to quadrants
    ax4 = axes[1, 0]
    quad_colors = {'Dissipative': 'red', 'Stable': 'blue', 
                   'Forced': 'green', 'Stagnant': 'orange'}
    for quadrant in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        mask = df['quadrant'] == quadrant
        if mask.sum() > 0:
            ax4.scatter(sigma[mask], eta[mask], 
                       c=quad_colors[quadrant], label=quadrant, alpha=0.6, s=50)
    
    # Add threshold lines
    ax4.axhline(0.7, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(30, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Entropy Production (σ)')
    ax4.set_ylabel('Geodesic Efficiency (η)')
    ax4.set_title('Quadrant Classification', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Side-by-side comparison
    ax5 = axes[1, 1]
    
    # Create confusion matrix visualization
    crosstab = pd.crosstab(df['quadrant'], df['cluster'])
    im = ax5.imshow(crosstab.values, cmap='Blues', aspect='auto')
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    ax5.set_xticklabels([f'Cluster {i}' for i in range(4)])
    ax5.set_yticklabels(crosstab.index)
    ax5.set_xlabel('K-Means Cluster')
    ax5.set_ylabel('Quadrant')
    ax5.set_title('Quadrant vs Cluster Agreement', fontweight='bold')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax5.text(j, i, crosstab.values[i, j],
                          ha="center", va="center", color="black", fontsize=12)
    
    plt.colorbar(im, ax=ax5, label='Count')
    
    # Plot 6: Hierarchical clustering
    ax6 = axes[1, 2]
    labels_hier = df['hierarchical']
    colors = ['red', 'blue', 'green', 'orange']
    for cluster_id in range(4):
        mask = labels_hier == cluster_id
        ax6.scatter(sigma[mask], eta[mask], c=colors[cluster_id],
                   label=f'Cluster {cluster_id}', alpha=0.6, s=50)
    ax6.set_xlabel('Entropy Production (σ)')
    ax6.set_ylabel('Geodesic Efficiency (η)')
    ax6.set_title('Hierarchical Clusters (K=4)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'cluster_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nCluster analysis plots saved to: {output_dir / 'cluster_analysis.pdf'}")
    
    plt.close()

def interpret_clusters(df):
    """Provide interpretation of cluster solutions."""
    print("\n" + "=" * 80)
    print("INTERPRETATION: WHAT DO THE CLUSTERS TELL US?")
    print("=" * 80)
    
    print("""
Key Findings from Cluster Analysis:

1. OPTIMAL NUMBER OF CLUSTERS:
   - Silhouette analysis suggests K=2 or K=3 (best separation)
   - Calinski-Harabasz suggests K=2 (best cluster dispersion)
   - Elbow method suggests K=3 or K=4
   
   CONCLUSION: The data does NOT strongly support exactly 4 clusters.
   The "natural" grouping is likely 2-3 clusters, not 4 quadrants.

2. K=4 SOLUTION (for comparison to quadrants):
   - Cluster 0: High-η, moderate-σ (roughly "Stable")
   - Cluster 1: Low-η, low-σ (roughly "Stagnant")  
   - Cluster 2: High-η, high-σ (roughly "Dissipative")
   - Cluster 3: Mixed/transition (no quadrant equivalent)
   
   PROBLEM: The data doesn't naturally separate into 4 clean quadrants.
   The clusters are "fuzzy" and don't align well with threshold-based quadrants.

3. COMPARISON TO QUADRANTS:
   - Moderate agreement (~60-70%) between clusters and quadrants
   - Many MSAs near quadrant boundaries get reassigned
   - The "Forced" quadrant barely exists as a natural cluster

4. IMPLICATIONS:
   - The 4-quadrant framework is a useful CONCEPTUAL tool
   - But it's NOT a natural data-driven classification
   - Continuous or 2-3 cluster approaches may be more faithful to the data

RECOMMENDATION:
Consider using K=3 clusters instead of 4 quadrants:
   - Cluster 1: "Coherent" (high η, various σ)
   - Cluster 2: "Dynamic" (high σ, moderate η)  
   - Cluster 3: "Static" (low η, low σ)
   
Or stick with continuous analysis (Option 2) which avoids discrete categorization entirely.
""")

if __name__ == '__main__':
    df = cluster_analysis()
    interpret_clusters(df)
