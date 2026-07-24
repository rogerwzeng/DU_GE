"""
Sensitivity analysis for quadrant classification thresholds.
Tests how MSA classifications change with different η and σ thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data():
    """Load MSA data with geodesic efficiency and entropy production."""
    # Load data with entropy production
    results_path = Path(__file__).parent.parent / 'results' / 'msa_data_with_coords.csv'
    df = pd.read_csv(results_path)
    return df

def get_column_names(df):
    """Get the column names for sigma and eta."""
    if 'mean_entropy_production' in df.columns:
        sigma_col = 'mean_entropy_production'
    elif 'entropy_production' in df.columns:
        sigma_col = 'entropy_production'
    else:
        raise ValueError("No entropy production column found")
    
    if 'geodesic_efficiency' in df.columns:
        eta_col = 'geodesic_efficiency'
    elif 'eta_full_solver' in df.columns:
        eta_col = 'eta_full_solver'
    else:
        raise ValueError("No geodesic efficiency column found")
    
    return sigma_col, eta_col

def classify_quadrants(df, eta_threshold, sigma_threshold):
    """Classify MSAs into quadrants based on thresholds."""
    sigma_col, eta_col = get_column_names(df)
    
    conditions = [
        (df[sigma_col] >= sigma_threshold) & (df[eta_col] >= eta_threshold),   # Q1: Dissipative
        (df[sigma_col] < sigma_threshold) & (df[eta_col] >= eta_threshold),    # Q2: Stable
        (df[sigma_col] >= sigma_threshold) & (df[eta_col] < eta_threshold),    # Q3: Forced
        (df[sigma_col] < sigma_threshold) & (df[eta_col] < eta_threshold)      # Q4: Stagnant
    ]
    
    choices = ['Dissipative', 'Stable', 'Forced', 'Stagnant']
    
    return np.select(conditions, choices, default='Unknown')

def run_sensitivity_analysis():
    """Run sensitivity analysis for different threshold combinations."""
    df = load_data()
    
    # Get data stats
    if 'mean_entropy_production' in df.columns:
        sigma_col = 'mean_entropy_production'
    elif 'entropy_production' in df.columns:
        sigma_col = 'entropy_production'
    else:
        raise ValueError("No entropy production column found")
    
    if 'geodesic_efficiency' in df.columns:
        eta_col = 'geodesic_efficiency'
    elif 'eta_full_solver' in df.columns:
        eta_col = 'eta_full_solver'
    else:
        raise ValueError("No geodesic efficiency column found")
    
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: QUADRANT CLASSIFICATION THRESHOLDS")
    print("=" * 80)
    print(f"\nData: {len(df)} MSAs")
    print(f"\nσ (entropy production) statistics:")
    print(f"  Mean: {df[sigma_col].mean():.2f}")
    print(f"  Median: {df[sigma_col].median():.2f}")
    print(f"  Std: {df[sigma_col].std():.2f}")
    print(f"  Min: {df[sigma_col].min():.2f}")
    print(f"  Max: {df[sigma_col].max():.2f}")
    print(f"  25th percentile: {df[sigma_col].quantile(0.25):.2f}")
    print(f"  75th percentile: {df[sigma_col].quantile(0.75):.2f}")
    
    print(f"\nη (geodesic efficiency) statistics:")
    print(f"  Mean: {df[eta_col].mean():.3f}")
    print(f"  Median: {df[eta_col].median():.3f}")
    print(f"  Std: {df[eta_col].std():.3f}")
    print(f"  Min: {df[eta_col].min():.3f}")
    print(f"  Max: {df[eta_col].max():.3f}")
    print(f"  25th percentile: {df[eta_col].quantile(0.25):.3f}")
    print(f"  75th percentile: {df[eta_col].quantile(0.75):.3f}")
    
    # Baseline (current thresholds)
    baseline_eta = 0.7
    baseline_sigma = 30
    
    print(f"\n{'='*80}")
    print(f"BASELINE CLASSIFICATION (η ≥ {baseline_eta}, σ ≥ {baseline_sigma})")
    print(f"{'='*80}")
    
    df['quadrant'] = classify_quadrants(df, baseline_eta, baseline_sigma)
    baseline_counts = df['quadrant'].value_counts()
    for quad in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        count = baseline_counts.get(quad, 0)
        pct = 100 * count / len(df)
        print(f"  {quad:15s}: {count:3d} MSAs ({pct:5.1f}%)")
    
    # Test η sensitivity
    print(f"\n{'='*80}")
    print("SENSITIVITY TO η THRESHOLD (keeping σ = 30)")
    print(f"{'='*80}")
    print(f"{'η thresh':<10} {'Dissipative':>12} {'Stable':>12} {'Forced':>12} {'Stagnant':>12}")
    print("-" * 60)
    
    eta_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    eta_results = []
    
    for eta in eta_values:
        df['quadrant'] = classify_quadrants(df, eta, baseline_sigma)
        counts = df['quadrant'].value_counts()
        q1 = counts.get('Dissipative', 0)
        q2 = counts.get('Stable', 0)
        q3 = counts.get('Forced', 0)
        q4 = counts.get('Stagnant', 0)
        eta_results.append({
            'eta': eta, 'sigma': baseline_sigma,
            'Dissipative': q1, 'Stable': q2, 'Forced': q3, 'Stagnant': q4
        })
        print(f"{eta:<10.1f} {q1:>12} ({100*q1/len(df):4.1f}%) {q2:>12} ({100*q2/len(df):4.1f}%) {q3:>12} ({100*q3/len(df):4.1f}%) {q4:>12} ({100*q4/len(df):4.1f}%)")
    
    # Test σ sensitivity
    print(f"\n{'='*80}")
    print("SENSITIVITY TO σ THRESHOLD (keeping η = 0.7)")
    print(f"{'='*80}")
    print(f"{'σ thresh':<10} {'Dissipative':>12} {'Stable':>12} {'Forced':>12} {'Stagnant':>12}")
    print("-" * 60)
    
    sigma_values = [20, 25, 30, 35, 40]
    sigma_results = []
    
    for sigma in sigma_values:
        df['quadrant'] = classify_quadrants(df, baseline_eta, sigma)
        counts = df['quadrant'].value_counts()
        q1 = counts.get('Dissipative', 0)
        q2 = counts.get('Stable', 0)
        q3 = counts.get('Forced', 0)
        q4 = counts.get('Stagnant', 0)
        sigma_results.append({
            'eta': baseline_eta, 'sigma': sigma,
            'Dissipative': q1, 'Stable': q2, 'Forced': q3, 'Stagnant': q4
        })
        print(f"{sigma:<10.0f} {q1:>12} ({100*q1/len(df):4.1f}%) {q2:>12} ({100*q2/len(df):4.1f}%) {q3:>12} ({100*q3/len(df):4.1f}%) {q4:>12} ({100*q4/len(df):4.1f}%)")
    
    # Joint sensitivity (2D table)
    print(f"\n{'='*80}")
    print("JOINT SENSITIVITY: DISSIPATIVE (Q1) CLASSIFICATION %")
    print(f"{'='*80}")
    print("       σ=20   σ=25   σ=30   σ=35   σ=40")
    print("-" * 50)
    
    for eta in eta_values:
        row = f"η={eta:.1f}  "
        for sigma in sigma_values:
            df['quadrant'] = classify_quadrants(df, eta, sigma)
            q1_pct = 100 * (df['quadrant'] == 'Dissipative').sum() / len(df)
            row += f"{q1_pct:5.1f}% "
        print(row)
    
    print(f"\n{'='*80}")
    print("JOINT SENSITIVITY: FORCED (Q3) CLASSIFICATION %")
    print(f"{'='*80}")
    print("       σ=20   σ=25   σ=30   σ=35   σ=40")
    print("-" * 50)
    
    for eta in eta_values:
        row = f"η={eta:.1f}  "
        for sigma in sigma_values:
            df['quadrant'] = classify_quadrants(df, eta, sigma)
            q3_pct = 100 * (df['quadrant'] == 'Forced').sum() / len(df)
            row += f"{q3_pct:5.1f}% "
        print(row)
    
    # Calculate threshold stability
    print(f"\n{'='*80}")
    print("THRESHOLD STABILITY ANALYSIS")
    print(f"{'='*80}")
    
    # How many MSAs change classification with ±10% threshold variation?
    df['baseline'] = classify_quadrants(df, baseline_eta, baseline_sigma)
    
    # Test η ± 0.1
    df['eta_plus'] = classify_quadrants(df, baseline_eta + 0.1, baseline_sigma)
    df['eta_minus'] = classify_quadrants(df, baseline_eta - 0.1, baseline_sigma)
    
    eta_changes_plus = (df['baseline'] != df['eta_plus']).sum()
    eta_changes_minus = (df['baseline'] != df['eta_minus']).sum()
    
    print(f"\nChanging η from {baseline_eta} to {baseline_eta + 0.1}:")
    print(f"  {eta_changes_plus} MSAs ({100*eta_changes_plus/len(df):.1f}%) change quadrant")
    print(f"Changing η from {baseline_eta} to {baseline_eta - 0.1}:")
    print(f"  {eta_changes_minus} MSAs ({100*eta_changes_minus/len(df):.1f}%) change quadrant")
    
    # Test σ ± 5
    df['sigma_plus'] = classify_quadrants(df, baseline_eta, baseline_sigma + 5)
    df['sigma_minus'] = classify_quadrants(df, baseline_eta, baseline_sigma - 5)
    
    sigma_changes_plus = (df['baseline'] != df['sigma_plus']).sum()
    sigma_changes_minus = (df['baseline'] != df['sigma_minus']).sum()
    
    print(f"\nChanging σ from {baseline_sigma} to {baseline_sigma + 5}:")
    print(f"  {sigma_changes_plus} MSAs ({100*sigma_changes_plus/len(df):.1f}%) change quadrant")
    print(f"Changing σ from {baseline_sigma} to {baseline_sigma - 5}:")
    print(f"  {sigma_changes_minus} MSAs ({100*sigma_changes_minus/len(df):.1f}%) change quadrant")
    
    # Test joint ±10%
    df['joint_plus'] = classify_quadrants(df, baseline_eta + 0.1, baseline_sigma + 5)
    df['joint_minus'] = classify_quadrants(df, baseline_eta - 0.1, baseline_sigma - 5)
    
    joint_changes_plus = (df['baseline'] != df['joint_plus']).sum()
    joint_changes_minus = (df['baseline'] != df['joint_minus']).sum()
    
    print(f"\nChanging (η, σ) from ({baseline_eta}, {baseline_sigma}) to ({baseline_eta + 0.1}, {baseline_sigma + 5}):")
    print(f"  {joint_changes_plus} MSAs ({100*joint_changes_plus/len(df):.1f}%) change quadrant")
    print(f"Changing (η, σ) from ({baseline_eta}, {baseline_sigma}) to ({baseline_eta - 0.1}, {baseline_sigma - 5}):")
    print(f"  {joint_changes_minus} MSAs ({100*joint_changes_minus/len(df):.1f}%) change quadrant")
    
    # Create visualization
    create_sensitivity_plots(df, eta_values, sigma_values, baseline_eta, baseline_sigma)
    
    return df


def save_results_to_file(output_text, filename='sensitivity_analysis_output.txt'):
    """Save analysis results to file."""
    output_path = Path(__file__).parent.parent / 'results' / filename
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output_text)
    return output_path

def create_sensitivity_plots(df, eta_values, sigma_values, baseline_eta, baseline_sigma):
    """Create sensitivity analysis plots."""
    sigma_col, eta_col = get_column_names(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: η sensitivity
    ax1 = axes[0, 0]
    q1_pcts, q2_pcts, q3_pcts, q4_pcts = [], [], [], []
    
    for eta in eta_values:
        df['quadrant'] = classify_quadrants(df, eta, baseline_sigma)
        counts = df['quadrant'].value_counts()
        total = len(df)
        q1_pcts.append(100 * counts.get('Dissipative', 0) / total)
        q2_pcts.append(100 * counts.get('Stable', 0) / total)
        q3_pcts.append(100 * counts.get('Forced', 0) / total)
        q4_pcts.append(100 * counts.get('Stagnant', 0) / total)
    
    ax1.plot(eta_values, q1_pcts, 'o-', label='Dissipative (Q1)', linewidth=2, markersize=8)
    ax1.plot(eta_values, q2_pcts, 's-', label='Stable (Q2)', linewidth=2, markersize=8)
    ax1.plot(eta_values, q3_pcts, '^-', label='Forced (Q3)', linewidth=2, markersize=8)
    ax1.plot(eta_values, q4_pcts, 'v-', label='Stagnant (Q4)', linewidth=2, markersize=8)
    ax1.axvline(baseline_eta, color='red', linestyle='--', alpha=0.7, label=f'Baseline η={baseline_eta}')
    ax1.set_xlabel('η Threshold', fontsize=12)
    ax1.set_ylabel('% of MSAs', fontsize=12)
    ax1.set_title(f'Sensitivity to η Threshold (σ = {baseline_sigma})', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 50)
    
    # Plot 2: σ sensitivity
    ax2 = axes[0, 1]
    q1_pcts, q2_pcts, q3_pcts, q4_pcts = [], [], [], []
    
    for sigma in sigma_values:
        df['quadrant'] = classify_quadrants(df, baseline_eta, sigma)
        counts = df['quadrant'].value_counts()
        total = len(df)
        q1_pcts.append(100 * counts.get('Dissipative', 0) / total)
        q2_pcts.append(100 * counts.get('Stable', 0) / total)
        q3_pcts.append(100 * counts.get('Forced', 0) / total)
        q4_pcts.append(100 * counts.get('Stagnant', 0) / total)
    
    ax2.plot(sigma_values, q1_pcts, 'o-', label='Dissipative (Q1)', linewidth=2, markersize=8)
    ax2.plot(sigma_values, q2_pcts, 's-', label='Stable (Q2)', linewidth=2, markersize=8)
    ax2.plot(sigma_values, q3_pcts, '^-', label='Forced (Q3)', linewidth=2, markersize=8)
    ax2.plot(sigma_values, q4_pcts, 'v-', label='Stagnant (Q4)', linewidth=2, markersize=8)
    ax2.axvline(baseline_sigma, color='red', linestyle='--', alpha=0.7, label=f'Baseline σ={baseline_sigma}')
    ax2.set_xlabel('σ Threshold', fontsize=12)
    ax2.set_ylabel('% of MSAs', fontsize=12)
    ax2.set_title(f'Sensitivity to σ Threshold (η = {baseline_eta})', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 50)
    
    # Plot 3: Heatmap of Dissipative classification
    ax3 = axes[1, 0]
    heatmap_data = np.zeros((len(eta_values), len(sigma_values)))
    
    for i, eta in enumerate(eta_values):
        for j, sigma in enumerate(sigma_values):
            df['quadrant'] = classify_quadrants(df, eta, sigma)
            heatmap_data[i, j] = 100 * (df['quadrant'] == 'Dissipative').sum() / len(df)
    
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower')
    ax3.set_xticks(range(len(sigma_values)))
    ax3.set_yticks(range(len(eta_values)))
    ax3.set_xticklabels([f'{s:.0f}' for s in sigma_values])
    ax3.set_yticklabels([f'{e:.1f}' for e in eta_values])
    ax3.set_xlabel('σ Threshold', fontsize=12)
    ax3.set_ylabel('η Threshold', fontsize=12)
    ax3.set_title('Dissipative (Q1) Classification %', fontsize=13, fontweight='bold')
    
    # Add baseline marker
    baseline_eta_idx = eta_values.index(baseline_eta) if baseline_eta in eta_values else None
    baseline_sigma_idx = sigma_values.index(baseline_sigma) if baseline_sigma in sigma_values else None
    if baseline_eta_idx is not None and baseline_sigma_idx is not None:
        ax3.plot(baseline_sigma_idx, baseline_eta_idx, 'b*', markersize=20, label='Baseline')
    
    plt.colorbar(im, ax=ax3, label='% Dissipative')
    
    # Add text annotations
    for i in range(len(eta_values)):
        for j in range(len(sigma_values)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Plot 4: Heatmap of Forced classification
    ax4 = axes[1, 1]
    heatmap_data = np.zeros((len(eta_values), len(sigma_values)))
    
    for i, eta in enumerate(eta_values):
        for j, sigma in enumerate(sigma_values):
            df['quadrant'] = classify_quadrants(df, eta, sigma)
            heatmap_data[i, j] = 100 * (df['quadrant'] == 'Forced').sum() / len(df)
    
    im = ax4.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', origin='lower')
    ax4.set_xticks(range(len(sigma_values)))
    ax4.set_yticks(range(len(eta_values)))
    ax4.set_xticklabels([f'{s:.0f}' for s in sigma_values])
    ax4.set_yticklabels([f'{e:.1f}' for e in eta_values])
    ax4.set_xlabel('σ Threshold', fontsize=12)
    ax4.set_ylabel('η Threshold', fontsize=12)
    ax4.set_title('Forced (Q3) Classification %', fontsize=13, fontweight='bold')
    
    # Add baseline marker
    if baseline_eta_idx is not None and baseline_sigma_idx is not None:
        ax4.plot(baseline_sigma_idx, baseline_eta_idx, 'r*', markersize=20, label='Baseline')
    
    plt.colorbar(im, ax=ax4, label='% Forced')
    
    # Add text annotations
    for i in range(len(eta_values)):
        for j in range(len(sigma_values)):
            text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'sensitivity_analysis_quadrants.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sensitivity_analysis_quadrants.png', dpi=300, bbox_inches='tight')
    print(f"\n\nPlots saved to: {output_dir / 'sensitivity_analysis_quadrants.pdf'}")
    
    plt.close()

if __name__ == '__main__':
    import sys
    output_path = Path(__file__).parent.parent / 'results' / 'sensitivity_analysis_output.txt'
    output_path.parent.mkdir(exist_ok=True)
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
        def flush(self):
            pass
    
    f = open(output_path, 'w')
    original = sys.stdout
    sys.stdout = Tee(original, f)
    
    run_sensitivity_analysis()
    
    sys.stdout = original
    f.close()
    
    print(f"\nOutput saved to: {output_path}")
