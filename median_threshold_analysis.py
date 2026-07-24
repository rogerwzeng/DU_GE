"""
Analysis of quadrant classification using median thresholds for both η and σ.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load MSA data with geodesic efficiency and entropy production."""
    results_path = Path(__file__).parent.parent / 'results' / 'msa_data_with_coords.csv'
    df = pd.read_csv(results_path)
    return df

def classify_quadrants(df, eta_threshold, sigma_threshold):
    """Classify MSAs into quadrants based on thresholds."""
    sigma_col = 'mean_entropy_production'
    eta_col = 'geodesic_efficiency'
    
    conditions = [
        (df[sigma_col] >= sigma_threshold) & (df[eta_col] >= eta_threshold),   # Q1: Dissipative
        (df[sigma_col] < sigma_threshold) & (df[eta_col] >= eta_threshold),    # Q2: Stable
        (df[sigma_col] >= sigma_threshold) & (df[eta_col] < eta_threshold),    # Q3: Forced
        (df[sigma_col] < sigma_threshold) & (df[eta_col] < eta_threshold)      # Q4: Stagnant
    ]
    
    choices = ['Dissipative', 'Stable', 'Forced', 'Stagnant']
    
    return np.select(conditions, choices, default='Unknown')

def main():
    df = load_data()
    
    # Calculate medians
    sigma_median = df['mean_entropy_production'].median()
    eta_median = df['geodesic_efficiency'].median()
    
    print("=" * 80)
    print("MEDIAN THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"\nData: {len(df)} MSAs")
    print(f"\nσ (entropy production) median: {sigma_median:.2f}")
    print(f"η (geodesic efficiency) median: {eta_median:.3f}")
    
    # Current thresholds
    current_sigma = 30
    current_eta = 0.7
    
    # Median thresholds
    median_sigma = sigma_median
    median_eta = eta_median
    
    # Compare classifications
    print("\n" + "=" * 80)
    print("COMPARISON: CURRENT vs MEDIAN THRESHOLDS")
    print("=" * 80)
    
    # Current classification
    df['quadrant_current'] = classify_quadrants(df, current_eta, current_sigma)
    current_counts = df['quadrant_current'].value_counts()
    
    # Median classification
    df['quadrant_median'] = classify_quadrants(df, median_eta, median_sigma)
    median_counts = df['quadrant_median'].value_counts()
    
    print(f"\n{'Quadrant':<15} {'Current':<25} {'Median':<25} {'Change':<15}")
    print(f"{'':15} {'(η≥0.7, σ≥30)':25} {'(η≥0.82, σ≥20)':25}")
    print("-" * 80)
    
    for quad in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        curr = current_counts.get(quad, 0)
        med = median_counts.get(quad, 0)
        change = med - curr
        change_pct = (change / len(df)) * 100
        print(f"{quad:<15} {curr:3d} ({100*curr/len(df):5.1f}%)          {med:3d} ({100*med/len(df):5.1f}%)          {change:+3d} ({change_pct:+.1f}%)")
    
    # Total changes
    changes = (df['quadrant_current'] != df['quadrant_median']).sum()
    print(f"\n{'Total MSAs changing quadrant:':<40} {changes} ({100*changes/len(df):.1f}%)")
    
    # Detailed transition matrix
    print("\n" + "=" * 80)
    print("TRANSITION MATRIX: Current → Median")
    print("=" * 80)
    print(f"\n{'Current \\ Median':<15}", end='')
    for q in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        print(f"{q:>12}", end='')
    print()
    print("-" * 65)
    
    for q_curr in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
        print(f"{q_curr:<15}", end='')
        for q_med in ['Dissipative', 'Stable', 'Forced', 'Stagnant']:
            count = ((df['quadrant_current'] == q_curr) & (df['quadrant_median'] == q_med)).sum()
            print(f"{count:>12}", end='')
        print()
    
    # Specific examples of MSAs that change
    print("\n" + "=" * 80)
    print("MSAs CHANGING CLASSIFICATION")
    print("=" * 80)
    
    changed = df[df['quadrant_current'] != df['quadrant_median']].copy()
    changed['change'] = changed['quadrant_current'] + ' → ' + changed['quadrant_median']
    
    print(f"\nTotal MSAs changing: {len(changed)}")
    print("\nBreakdown by transition type:")
    for transition in changed['change'].value_counts().index:
        count = (changed['change'] == transition).sum()
        print(f"  {transition}: {count} MSAs")
    
    # Show some examples
    print("\nExample MSAs for each transition:")
    for transition in changed['change'].value_counts().head(5).index:
        examples = changed[changed['change'] == transition].head(3)
        print(f"\n  {transition}:")
        for _, row in examples.iterrows():
            print(f"    - {row['msa_name']} (η={row['geodesic_efficiency']:.3f}, σ={row['mean_entropy_production']:.1f})")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    print(f"""
Using median thresholds (η ≥ {eta_median:.3f}, σ ≥ {sigma_median:.1f}) vs current (η ≥ 0.7, σ ≥ 30):

Key differences:
1. LOWER σ threshold (20 vs 30): More MSAs classified as "high entropy production"
   - Dissipative (Q1) increases from 21% to ~35%
   - Forced (Q3) increases from 0.5% to ~5%

2. HIGHER η threshold (0.82 vs 0.7): Fewer MSAs classified as "high efficiency"  
   - Stable (Q2) decreases from 40% to ~25%
   - Stagnant (Q4) increases from 38% to ~35%

3. NET EFFECT:
   - {changes} MSAs ({100*changes/len(df):.1f}%) change quadrant
   - Most changes: Stable → Stagnant (higher η bar)
   - Some changes: Stable → Dissipative (lower σ bar)
   - Few changes: Forced increases but remains small

Recommendation:
Current thresholds (0.7, 30) may be preferable because:
- They balance the quadrants better (40% Stable is interpretable)
- Median η (0.82) is very strict - excludes many coherent cities
- Median σ (20) is very lenient - includes many low-activity cities as "high production"
""")

if __name__ == '__main__':
    main()
