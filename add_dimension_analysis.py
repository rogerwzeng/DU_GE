#!/usr/bin/env python3
"""
Add Per-Dimension Geodesic Efficiency to Existing Results

This script loads the existing full solver results and adds per-dimension
geodesic efficiency (age, income, race) without re-running the computationally
expensive geodesic solver.

Much faster than re-running the full analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict

# Add src to path
HOME = Path.home()
sys.path.insert(0, str(HOME / 'DissipativeUrbanism/src'))

from analysis.geodesic_framework import compute_geodesic_efficiency_all_dimensions


# Paths (relative to geodesic_efficiency working directory)
BASE_DIR = HOME / 'DissipativeUrbanism' / 'geodesic_efficiency'
DATA_DIR = HOME / 'DissipativeUrbanism' / 'results/data'
OUTPUT_DIR = BASE_DIR / 'results'


def load_raw_probability_data() -> Dict[int, Dict]:
    """
    Load raw probability distributions for age, income, and race.
    
    Returns:
        Dictionary mapping msa_code to probability data
    """
    print("Loading raw probability data...")
    raw_file = DATA_DIR / 'msa_demographics_raw_annual.csv'
    df = pd.read_csv(raw_file)
    
    print(f"  Loaded {len(df)} records for {df['msa_code'].nunique()} MSAs")
    
    # Age cohort columns (18 cohorts)
    age_cols = ['age_0_4', 'age_5_9', 'age_10_14', 'age_15_17', 'age_18_19',
                'age_20_24', 'age_25_29', 'age_30_34', 'age_35_44', 'age_45_54',
                'age_55_59', 'age_60_64', 'age_65_74', 'age_75_84', 'age_85_plus']
    
    # Race columns (7 categories)
    race_cols = ['race_white', 'race_black', 'race_asian', 'race_aian', 
                 'race_nhpi', 'race_other', 'race_hispanic']
    
    # Income decile columns (10 deciles)
    income_cols = [f'income_decile_{i}' for i in range(1, 11)]
    
    msa_data = {}
    
    for msa_code, group in df.groupby('msa_code'):
        msa_code = int(msa_code)
        msa_name = group['msa_name'].iloc[0]
        group = group.sort_values('year')
        
        years = group['year'].tolist()
        
        # Extract probability distributions for each year
        age_probs_list = []
        income_probs_list = []
        race_probs_list = []
        
        for _, row in group.iterrows():
            # Age probabilities
            age_counts = np.array([row.get(col, 0) for col in age_cols], dtype=float)
            age_total = age_counts.sum()
            if age_total > 0:
                age_probs = age_counts / age_total
            else:
                age_probs = np.ones(len(age_cols)) / len(age_cols)
            age_probs_list.append(age_probs)
            
            # Income probabilities
            income_counts = np.array([row.get(col, 0) for col in income_cols], dtype=float)
            income_total = income_counts.sum()
            if income_total > 0:
                income_probs = income_counts / income_total
            else:
                income_probs = np.ones(len(income_cols)) / len(income_cols)
            income_probs_list.append(income_probs)
            
            # Race probabilities
            race_counts = np.array([row.get(col, 0) for col in race_cols], dtype=float)
            race_total = race_counts.sum()
            if race_total > 0:
                race_probs = race_counts / race_total
            else:
                race_probs = np.ones(len(race_cols)) / len(race_cols)
            race_probs_list.append(race_probs)
        
        msa_data[msa_code] = {
            'msa_name': msa_name,
            'age_probs': np.array(age_probs_list),
            'income_probs': np.array(income_probs_list),
            'race_probs': np.array(race_probs_list),
            'years': years
        }
    
    print(f"  Processed {len(msa_data)} MSAs")
    return msa_data


def identify_dominant_dimension(row: pd.Series) -> str:
    """Identify which demographic dimension has the highest geodesic efficiency."""
    eta_age = row.get('geodesic_efficiency_age', np.nan)
    eta_income = row.get('geodesic_efficiency_income', np.nan)
    eta_race = row.get('geodesic_efficiency_race', np.nan)
    
    etas = {
        'age': eta_age,
        'income': eta_income,
        'race': eta_race
    }
    
    # Filter out NaN values
    valid_et = {k: v for k, v in etas.items() if not np.isnan(v)}
    
    if not valid_et:
        return 'unknown'
    
    return max(valid_et, key=valid_et.get)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("ADDING PER-DIMENSION GEODESIC EFFICIENCY TO EXISTING RESULTS")
    print("="*80)
    
    # Load existing results
    print("\nLoading existing full solver results...")
    existing_file = OUTPUT_DIR / 'msa_geodesic_efficiency_full_solver.csv'
    results_df = pd.read_csv(existing_file)
    print(f"  Loaded {len(results_df)} MSAs")
    
    # Load raw probability data
    prob_data = load_raw_probability_data()
    
    # Compute per-dimension geodesic efficiency for each MSA
    print("\nComputing per-dimension geodesic efficiency...")
    
    eta_age_list = []
    eta_income_list = []
    eta_race_list = []
    
    for _, row in results_df.iterrows():
        msa_code = int(row['msa_code'])
        
        if msa_code in prob_data:
            prob = prob_data[msa_code]
            try:
                mean_eta, eta_by_dim = compute_geodesic_efficiency_all_dimensions(
                    prob['age_probs'],
                    prob['income_probs'],
                    prob['race_probs']
                )
                eta_age_list.append(eta_by_dim['age'])
                eta_income_list.append(eta_by_dim['income'])
                eta_race_list.append(eta_by_dim['race'])
            except Exception as e:
                print(f"  Warning: Failed for {msa_code}: {e}")
                eta_age_list.append(np.nan)
                eta_income_list.append(np.nan)
                eta_race_list.append(np.nan)
        else:
            eta_age_list.append(np.nan)
            eta_income_list.append(np.nan)
            eta_race_list.append(np.nan)
    
    # Add to results
    results_df['geodesic_efficiency_age'] = eta_age_list
    results_df['geodesic_efficiency_income'] = eta_income_list
    results_df['geodesic_efficiency_race'] = eta_race_list
    
    # Add dominant dimension
    results_df['dominant_dimension'] = results_df.apply(identify_dominant_dimension, axis=1)
    
    # Save results
    output_file = OUTPUT_DIR / 'msa_geodesic_efficiency_with_dimensions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # Generate report
    print("\n" + "="*80)
    print("PER-DIMENSION ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    for dim in ['age', 'income', 'race']:
        col = f'geodesic_efficiency_{dim}'
        vals = results_df[col].dropna()
        print(f"\n{dim.upper()} DIMENSION:")
        print(f"  Mean η: {vals.mean():.4f}")
        print(f"  Std:    {vals.std():.4f}")
        print(f"  Min:    {vals.min():.4f}")
        print(f"  Max:    {vals.max():.4f}")
        print(f"  Median: {vals.median():.4f}")
    
    # Dominant dimension distribution
    print("\n" + "="*80)
    print("DOMINANT DIMENSION DISTRIBUTION (All MSAs)")
    print("="*80)
    dim_counts = results_df['dominant_dimension'].value_counts()
    for dim, count in dim_counts.items():
        pct = 100 * count / len(results_df)
        print(f"  {dim.capitalize():10s}: {count:3d} MSAs ({pct:5.1f}%)")
    
    # Top 20 MSAs by overall efficiency
    print("\n" + "="*80)
    print("TOP 20 MSAs BY OVERALL GEODESIC EFFICIENCY")
    print("="*80)
    print(f"{'Rank':<5} {'MSA Name':<50} {'η':<8} {'Dominant':<10} {'η_age':<8} {'η_inc':<8} {'η_race':<8}")
    print("-"*80)
    
    top20 = results_df.nlargest(20, 'geodesic_efficiency')
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        name = row['msa_name'][:48]
        eta = row['geodesic_efficiency']
        dom = row['dominant_dimension']
        eta_age = row.get('geodesic_efficiency_age', np.nan)
        eta_inc = row.get('geodesic_efficiency_income', np.nan)
        eta_race = row.get('geodesic_efficiency_race', np.nan)
        
        eta_age_str = f"{eta_age:.3f}" if not np.isnan(eta_age) else "N/A"
        eta_inc_str = f"{eta_inc:.3f}" if not np.isnan(eta_inc) else "N/A"
        eta_race_str = f"{eta_race:.3f}" if not np.isnan(eta_race) else "N/A"
        
        print(f"{rank:<5} {name:<50} {eta:.4f}   {dom:<10} {eta_age_str:<8} {eta_inc_str:<8} {eta_race_str:<8}")
    
    # Analysis for high-efficiency MSAs
    print("\n" + "="*80)
    print("DIMENSION ANALYSIS FOR HIGH-EFFICIENCY MSAs (η > 0.9)")
    print("="*80)
    
    high_eta = results_df[results_df['geodesic_efficiency'] > 0.9]
    print(f"\nTotal high-efficiency MSAs: {len(high_eta)}")
    
    # Dominant dimension for high-η MSAs
    dim_counts_high = high_eta['dominant_dimension'].value_counts()
    print("\nDominant dimension distribution (η > 0.9 MSAs):")
    for dim, count in dim_counts_high.items():
        pct = 100 * count / len(high_eta)
        print(f"  {dim.capitalize():10s}: {count:3d} MSAs ({pct:5.1f}%)")
    
    # Show breakdown for each dimension
    for dim in ['age', 'income', 'race']:
        col = f'geodesic_efficiency_{dim}'
        if col in high_eta.columns:
            dim_high = high_eta[high_eta[col] > 0.9]
            print(f"\n{dim.capitalize()} η > 0.9: {len(dim_high)} MSAs")
            if len(dim_high) > 0:
                print(f"  Mean {dim} η: {dim_high[col].mean():.4f}")
                
                # Show top 5 for this dimension
                top5 = dim_high.nlargest(5, col)[['msa_name', col, 'geodesic_efficiency', 'dominant_dimension']]
                print(f"  Top 5 by {dim} efficiency:")
                for _, row in top5.iterrows():
                    print(f"    {row['msa_name'][:40]:<40} η_{dim}={row[col]:.4f} (dominant: {row['dominant_dimension']})")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
