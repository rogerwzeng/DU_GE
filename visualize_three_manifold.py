#!/usr/bin/env python3
"""
Visualization - Three Manifold Framework

Creates comprehensive visualizations of the three metrics:
- η (demographic efficiency)
- σ_econ (economic flux)
- σ_mig (migration rate)

Output:
    results/figures/ directory with multiple plots

Usage:
    python visualize_three_manifold.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configuration
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    'eta': '#2E86AB',        # Blue
    'sigma_econ': '#A23B72', # Magenta
    'sigma_mig': '#F18F01',  # Orange
}


def load_data():
    """Load panel data."""
    print("Loading data...")
    panel_file = RESULTS_DIR / "panel_complete.csv"
    
    if not panel_file.exists():
        print(f"ERROR: {panel_file} not found")
        return None
    
    df = pd.read_csv(panel_file)
    df['cbsa_code'] = df['cbsa_code'].astype(str)
    
    print(f"  Loaded {len(df)} observations")
    return df


def plot_pairwise_relationships(df):
    """Create pairwise scatter plots with regression lines."""
    print("\nCreating pairwise relationship plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Pairwise Relationships: Three Manifold Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: η vs σ_econ
    ax = axes[0, 0]
    ax.scatter(df['sigma_econ'], df['eta'], alpha=0.3, s=20, color=COLORS['eta'])
    z = np.polyfit(df['sigma_econ'].dropna(), df['eta'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['sigma_econ'].min(), df['sigma_econ'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    corr = df['eta'].corr(df['sigma_econ'])
    ax.set_xlabel('σ_econ (Economic Flux)', fontsize=12)
    ax.set_ylabel('η (Demographic Efficiency)', fontsize=12)
    ax.set_title(f'η vs σ_econ (r = {corr:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: η vs σ_mig
    ax = axes[0, 1]
    ax.scatter(df['sigma_mig'], df['eta'], alpha=0.3, s=20, color=COLORS['eta'])
    z = np.polyfit(df['sigma_mig'].dropna(), df['eta'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['sigma_mig'].min(), df['sigma_mig'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    corr = df['eta'].corr(df['sigma_mig'])
    ax.set_xlabel('σ_mig (Migration Rate)', fontsize=12)
    ax.set_ylabel('η (Demographic Efficiency)', fontsize=12)
    ax.set_title(f'η vs σ_mig (r = {corr:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: σ_econ vs σ_mig
    ax = axes[1, 0]
    ax.scatter(df['sigma_econ'], df['sigma_mig'], alpha=0.3, s=20, color=COLORS['sigma_econ'])
    z = np.polyfit(df['sigma_econ'].dropna(), df['sigma_mig'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['sigma_econ'].min(), df['sigma_econ'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    corr = df['sigma_econ'].corr(df['sigma_mig'])
    ax.set_xlabel('σ_econ (Economic Flux)', fontsize=12)
    ax.set_ylabel('σ_mig (Migration Rate)', fontsize=12)
    ax.set_title(f'σ_econ vs σ_mig (r = {corr:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: 3D scatter (projected)
    ax = axes[1, 1]
    scatter = ax.scatter(df['sigma_econ'], df['sigma_mig'], c=df['eta'], 
                        cmap='viridis', alpha=0.5, s=30)
    plt.colorbar(scatter, ax=ax, label='η (Demographic Efficiency)')
    ax.set_xlabel('σ_econ (Economic Flux)', fontsize=12)
    ax.set_ylabel('σ_mig (Migration Rate)', fontsize=12)
    ax.set_title('Joint Distribution (colored by η)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pairwise_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pairwise_relationships.png")


def plot_distributions(df):
    """Plot distributions of the three metrics."""
    print("\nCreating distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distributions of Three Manifold Metrics', fontsize=16, fontweight='bold')
    
    metrics = [
        ('eta', 'η (Demographic Efficiency)', COLORS['eta']),
        ('sigma_econ', 'σ_econ (Economic Flux)', COLORS['sigma_econ']),
        ('sigma_mig', 'σ_mig (Migration Rate)', COLORS['sigma_mig'])
    ]
    
    for i, (col, title, color) in enumerate(metrics):
        # Histogram
        ax = axes[0, i]
        ax.hist(df[col], bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.3f}')
        ax.axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.3f}')
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Box plot by year
        ax = axes[1, i]
        year_data = [df[df['year'] == y][col].values for y in sorted(df['year'].unique())]
        bp = ax.boxplot(year_data, labels=sorted(df['year'].unique()), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: distributions.png")


def plot_time_series_examples(df):
    """Plot time series for example MSAs."""
    print("\nCreating example time series plots...")
    
    # Select diverse MSAs
    example_msas = [
        ('35620', 'New York-NJ (Large Stable)'),
        ('31080', 'Los Angeles (Large Diverse)'),
        ('19100', 'Dallas (Fast Growing)'),
        ('19820', 'Detroit (Rust Belt)'),
        ('48660', 'Wichita (Military/Aviation)'),
    ]
    
    fig, axes = plt.subplots(len(example_msas), 3, figsize=(16, 4*len(example_msas)))
    fig.suptitle('Time Series Examples: Three Metrics by MSA', fontsize=16, fontweight='bold')
    
    for i, (cbsa_code, title) in enumerate(example_msas):
        msa_data = df[df['cbsa_code'] == cbsa_code].sort_values('year')
        
        if len(msa_data) == 0:
            continue
        
        # η
        ax = axes[i, 0]
        ax.plot(msa_data['year'], msa_data['eta'], marker='o', color=COLORS['eta'], linewidth=2)
        ax.set_ylabel('η', fontsize=11)
        ax.set_title(f'{title}\nDemographic Efficiency', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # σ_econ
        ax = axes[i, 1]
        ax.plot(msa_data['year'], msa_data['sigma_econ'], marker='o', color=COLORS['sigma_econ'], linewidth=2)
        ax.set_ylabel('σ_econ', fontsize=11)
        ax.set_title('Economic Flux', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # σ_mig
        ax = axes[i, 2]
        ax.plot(msa_data['year'], msa_data['sigma_mig'], marker='o', color=COLORS['sigma_mig'], linewidth=2)
        ax.set_ylabel('σ_mig', fontsize=11)
        ax.set_title('Migration Rate', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add year labels only on bottom row
        if i == len(example_msas) - 1:
            for ax in axes[i]:
                ax.set_xlabel('Year', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'time_series_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: time_series_examples.png")


def plot_correlation_heatmap(df):
    """Create correlation heatmap."""
    print("\nCreating correlation heatmap...")
    
    # Select numeric columns
    numeric_cols = ['eta', 'eta_age', 'eta_education', 'eta_race', 
                   'sigma_econ', 'sigma_mig', 'population']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix: Three Manifold Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: correlation_heatmap.png")


def plot_msa_typology(df):
    """Create MSA typology scatter plot."""
    print("\nCreating MSA typology plot...")
    
    # Compute mean values by MSA
    msa_means = df.groupby('cbsa_code').agg({
        'eta': 'mean',
        'sigma_econ': 'mean',
        'sigma_mig': 'mean',
        'cbsa_title': 'first',
        'population': 'mean'
    }).reset_index()
    
    # Create quadrants based on medians
    eta_median = msa_means['eta'].median()
    sigma_mig_median = msa_means['sigma_mig'].median()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('MSA Typology: Efficiency vs. Flux', fontsize=16, fontweight='bold')
    
    # Plot 1: η vs σ_mig with quadrants
    ax = axes[0]
    scatter = ax.scatter(msa_means['sigma_mig'], msa_means['eta'], 
                        s=msa_means['population']/5000,  # Size by population
                        alpha=0.5, c=msa_means['sigma_econ'], cmap='viridis')
    ax.axhline(eta_median, color='red', linestyle='--', alpha=0.5)
    ax.axvline(sigma_mig_median, color='red', linestyle='--', alpha=0.5)
    
    # Label quadrants
    ax.text(0.02, 0.98, 'Stagnant\n(High η, Low σ_mig)', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.98, 0.98, 'Turbulent\n(High η, High σ_mig)', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(0.02, 0.02, 'Chaotic\n(Low η, Low σ_mig)', transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.98, 0.02, 'Dynamic\n(Low η, High σ_mig)', transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.set_xlabel('Mean σ_mig (Migration Rate)', fontsize=12)
    ax.set_ylabel('Mean η (Demographic Efficiency)', fontsize=12)
    ax.set_title('MSA Typology\n(size = population, color = σ_econ)', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='σ_econ (Economic Flux)')
    
    # Plot 2: Top/bottom MSAs by each metric
    ax = axes[1]
    
    # Get top and bottom 10 for each metric
    top_eta = msa_means.nlargest(10, 'eta')
    bottom_eta = msa_means.nsmallest(10, 'eta')
    top_volatile = msa_means.nlargest(10, 'sigma_econ')
    top_mobile = msa_means.nlargest(10, 'sigma_mig')
    
    y_pos = np.arange(10)
    
    # Create grouped bar chart
    bar_height = 0.2
    
    ax.barh(y_pos, top_eta['eta'].values, bar_height, label='Top η', color=COLORS['eta'], alpha=0.8)
    ax.barh(y_pos + bar_height, top_volatile['sigma_econ'].values*10, bar_height, 
            label='Top σ_econ (×10)', color=COLORS['sigma_econ'], alpha=0.8)
    ax.barh(y_pos + 2*bar_height, top_mobile['sigma_mig'].values*10, bar_height,
            label='Top σ_mig (×10)', color=COLORS['sigma_mig'], alpha=0.8)
    
    ax.set_xlabel('Metric Value', fontsize=12)
    ax.set_title('Top 10 MSAs by Each Metric', fontsize=11)
    ax.legend(loc='lower right')
    ax.set_yticks(y_pos + bar_height)
    ax.set_yticklabels([f'{i+1}' for i in range(10)])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'msa_typology.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: msa_typology.png")


def plot_regression_results_summary():
    """Create a visual summary of regression results."""
    print("\nCreating regression results summary...")
    
    # Key coefficients from the analysis
    hypotheses = ['H1: Econ→Mig', 'H2: Mig→Eta', 'H3: Econ→Eta']
    coefficients = [0.0724, 0.1918, 0.0510]  # From full models
    p_values = [0.0001, 0.0696, 0.4079]
    significance = ['***', '†', 'ns']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Regression Results Summary', fontsize=16, fontweight='bold')
    
    # Coefficient plot
    colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'red' for p in p_values]
    bars = ax1.barh(hypotheses, coefficients, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_xlabel('Coefficient', fontsize=12)
    ax1.set_title('Coupling Pathway Coefficients', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars, significance)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, f'  {sig}', 
                va='center', fontsize=12, fontweight='bold')
    
    # Pathway diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Coupling Pathways', fontsize=12)
    
    # Draw boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
    
    # Economic Flux (top)
    ax2.text(5, 8, 'σ_econ\n(Economic Flux)', ha='center', va='center', 
            fontsize=11, bbox=box_props, fontweight='bold')
    
    # Migration (middle)
    ax2.text(5, 5, 'σ_mig\n(Migration Rate)', ha='center', va='center',
            fontsize=11, bbox=box_props, fontweight='bold')
    
    # Demographic Efficiency (bottom)
    ax2.text(5, 2, 'η\n(Demographic Efficiency)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8), 
            fontweight='bold')
    
    # Arrows with labels
    # H1: Econ → Mig
    ax2.annotate('', xy=(5, 5.8), xytext=(5, 7.2),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax2.text(6.5, 6.5, 'H1: +0.072***', fontsize=10, color='green', fontweight='bold')
    
    # H2: Mig → Eta
    ax2.annotate('', xy=(5, 2.8), xytext=(5, 4.2),
                arrowprops=dict(arrowstyle='->', lw=3, color='orange'))
    ax2.text(6.5, 3.5, 'H2: +0.192†', fontsize=10, color='orange', fontweight='bold')
    
    # H3: Econ → Eta (direct)
    ax2.annotate('', xy=(3.5, 2.5), xytext=(4.2, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', ls='--'))
    ax2.text(1.5, 5, 'H3: +0.051 ns', fontsize=9, color='red', style='italic')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'regression_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: regression_summary.png")


def main():
    """Main visualization pipeline."""
    print("="*70)
    print("THREE MANIFOLD VISUALIZATION")
    print("="*70)
    
    # Create figures directory if needed
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Generate all plots
    plot_pairwise_relationships(df)
    plot_distributions(df)
    plot_time_series_examples(df)
    plot_correlation_heatmap(df)
    plot_msa_typology(df)
    plot_regression_results_summary()
    
    print("\n" + "="*70)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("="*70)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
