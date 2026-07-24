#!/usr/bin/env python3
"""
Panel Regression Analysis - Three Manifold Coupling

This script tests the coupling hypotheses:
1. Economic flux → Migration flux (σ_econ → σ_mig)
2. Migration flux → Demographic efficiency (σ_mig → η)
3. Direct economic → demographic effects (σ_econ → η)

Models include:
- Pooled OLS
- Entity (MSA) fixed effects
- Two-way fixed effects (entity + year)
- First differences

Usage:
    python panel_regression.py

Output:
    results/regression_results.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import linearmodels for panel regression
# If not available, use statsmodels
try:
    from linearmodels.panel import PanelOLS, PooledOLS, FirstDifferenceOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("Warning: linearmodels not installed. Using statsmodels instead.")

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Error: statsmodels not installed.")

# Configuration
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")


def load_panel():
    """Load the panel dataset."""
    print("Loading panel data...")
    panel_file = RESULTS_DIR / "panel_complete.csv"
    
    if not panel_file.exists():
        print(f"ERROR: {panel_file} not found. Run merge_panel.py first.")
        return None
    
    df = pd.read_csv(panel_file)
    
    # Ensure consistent types
    df['cbsa_code'] = df['cbsa_code'].astype(str)
    df['year'] = df['year'].astype(int)
    
    # Sort for lag creation
    df = df.sort_values(['cbsa_code', 'year'])
    
    print(f"  Loaded {len(df)} complete observations")
    print(f"  MSAs: {df['cbsa_code'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    
    return df


def create_lags(df):
    """Create lagged variables for temporal ordering tests."""
    print("\nCreating lagged variables...")
    
    # Sort by MSA and year
    df = df.sort_values(['cbsa_code', 'year'])
    
    # Create lags within each MSA
    df['L1_eta'] = df.groupby('cbsa_code')['eta'].shift(1)
    df['L1_sigma_econ'] = df.groupby('cbsa_code')['sigma_econ'].shift(1)
    df['L1_sigma_mig'] = df.groupby('cbsa_code')['sigma_mig'].shift(1)
    
    # Create 2-year lags for robustness
    df['L2_eta'] = df.groupby('cbsa_code')['eta'].shift(2)
    df['L2_sigma_econ'] = df.groupby('cbsa_code')['sigma_econ'].shift(2)
    df['L2_sigma_mig'] = df.groupby('cbsa_code')['sigma_mig'].shift(2)
    
    # Create differences (changes)
    df['D_eta'] = df.groupby('cbsa_code')['eta'].diff(1)
    df['D_sigma_econ'] = df.groupby('cbsa_code')['sigma_econ'].diff(1)
    df['D_sigma_mig'] = df.groupby('cbsa_code')['sigma_mig'].diff(1)
    
    # Drop rows with missing lags
    valid_obs = df[['L1_eta', 'L1_sigma_econ', 'L1_sigma_mig']].notna().all(axis=1).sum()
    print(f"  Valid observations with 1-year lag: {valid_obs}")
    
    return df


def run_pooled_ols(df, y_var, x_vars, model_name):
    """Run pooled OLS regression."""
    data = df.dropna(subset=[y_var] + x_vars)
    
    if len(data) == 0:
        return None, None
    
    X = sm.add_constant(data[x_vars])
    y = data[y_var]
    
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data['cbsa_code']})
    
    return model, len(data)


def run_fe_panel(df, y_var, x_vars, model_name, entity_effects=True, time_effects=False):
    """Run fixed effects panel regression using linearmodels."""
    if not HAS_LINEARMODELS:
        print("  Skipping FE model (linearmodels not available)")
        return None, None
    
    # Set index for panel
    panel_df = df.set_index(['cbsa_code', 'year'])
    
    # Select complete cases
    data = panel_df.dropna(subset=[y_var] + x_vars)
    
    if len(data) == 0:
        return None, None
    
    X = data[x_vars]
    y = data[y_var]
    
    try:
        model = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        return results, len(data)
    except Exception as e:
        print(f"  Error in FE model: {e}")
        return None, None


def test_coupling_econ_to_mig(df):
    """Test Hypothesis 1: Economic flux predicts migration flux."""
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Economic Flux → Migration Flux")
    print("="*70)
    print("  Model: σ_mig(t) = f(σ_econ(t-1), controls)")
    
    results_dict = {}
    
    # Model 1a: Pooled OLS with lag
    print("\nModel 1a: Pooled OLS (clustered SE)")
    model, n = run_pooled_ols(df, 'sigma_mig', ['L1_sigma_econ'], "Pooled OLS")
    if model:
        print(f"  N = {n}")
        print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['1a_pooled'] = model
    
    # Model 1b: Entity FE
    if HAS_LINEARMODELS:
        print("\nModel 1b: Entity Fixed Effects")
        model, n = run_fe_panel(df, 'sigma_mig', ['L1_sigma_econ'], "Entity FE", 
                                entity_effects=True, time_effects=False)
        if model:
            print(f"  N = {n}")
            print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
            print(f"  R² = {model.rsquared:.4f}")
            results_dict['1b_entity_fe'] = model
    
    # Model 1c: Two-way FE
    if HAS_LINEARMODELS:
        print("\nModel 1c: Two-Way Fixed Effects")
        model, n = run_fe_panel(df, 'sigma_mig', ['L1_sigma_econ'], "Two-way FE",
                                entity_effects=True, time_effects=True)
        if model:
            print(f"  N = {n}")
            print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
            print(f"  R² = {model.rsquared:.4f}")
            results_dict['1c_tw_fe'] = model
    
    # Model 1d: With lagged dependent variable
    print("\nModel 1d: With Lagged Dependent Variable")
    model, n = run_pooled_ols(df, 'sigma_mig', ['L1_sigma_mig', 'L1_sigma_econ'], "Dynamic")
    if model:
        print(f"  N = {n}")
        print(f"  L1_sigma_mig: coef = {model.params['L1_sigma_mig']:.4f}, p = {model.pvalues['L1_sigma_mig']:.4f}")
        print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['1d_dynamic'] = model
    
    return results_dict


def test_coupling_mig_to_eta(df):
    """Test Hypothesis 2: Migration flux predicts demographic efficiency."""
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Migration Flux → Demographic Efficiency")
    print("="*70)
    print("  Model: η(t) = f(σ_mig(t-1), controls)")
    
    results_dict = {}
    
    # Model 2a: Pooled OLS
    print("\nModel 2a: Pooled OLS (clustered SE)")
    model, n = run_pooled_ols(df, 'eta', ['L1_sigma_mig'], "Pooled OLS")
    if model:
        print(f"  N = {n}")
        print(f"  L1_sigma_mig: coef = {model.params['L1_sigma_mig']:.4f}, p = {model.pvalues['L1_sigma_mig']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['2a_pooled'] = model
    
    # Model 2b: Entity FE
    if HAS_LINEARMODELS:
        print("\nModel 2b: Entity Fixed Effects")
        model, n = run_fe_panel(df, 'eta', ['L1_sigma_mig'], "Entity FE",
                                entity_effects=True, time_effects=False)
        if model:
            print(f"  N = {n}")
            print(f"  L1_sigma_mig: coef = {model.params['L1_sigma_mig']:.4f}, p = {model.pvalues['L1_sigma_mig']:.4f}")
            print(f"  R² = {model.rsquared:.4f}")
            results_dict['2b_entity_fe'] = model
    
    # Model 2c: With lagged eta
    print("\nModel 2c: With Lagged Dependent Variable")
    model, n = run_pooled_ols(df, 'eta', ['L1_eta', 'L1_sigma_mig'], "Dynamic")
    if model:
        print(f"  N = {n}")
        print(f"  L1_eta: coef = {model.params['L1_eta']:.4f}, p = {model.pvalues['L1_eta']:.4f}")
        print(f"  L1_sigma_mig: coef = {model.params['L1_sigma_mig']:.4f}, p = {model.pvalues['L1_sigma_mig']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['2c_dynamic'] = model
    
    # Model 2d: Test interaction with education component
    df['L1_sigma_mig_x_L1_eta_education'] = df['L1_sigma_mig'] * df.groupby('cbsa_code')['eta_education'].shift(1)
    print("\nModel 2d: Interaction with Education Component")
    model, n = run_pooled_ols(df, 'eta', ['L1_eta', 'L1_sigma_mig', 'L1_sigma_mig_x_L1_eta_education'], "Interaction")
    if model:
        print(f"  N = {n}")
        print(f"  L1_sigma_mig: coef = {model.params['L1_sigma_mig']:.4f}, p = {model.pvalues['L1_sigma_mig']:.4f}")
        print(f"  Interaction: coef = {model.params['L1_sigma_mig_x_L1_eta_education']:.4f}, p = {model.pvalues['L1_sigma_mig_x_L1_eta_education']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['2d_interaction'] = model
    
    return results_dict


def test_coupling_econ_to_eta(df):
    """Test Hypothesis 3: Direct economic to demographic effects."""
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Direct Economic → Demographic Effects")
    print("="*70)
    print("  Model: η(t) = f(σ_econ(t-1), controls)")
    
    results_dict = {}
    
    # Model 3a: Pooled OLS
    print("\nModel 3a: Pooled OLS (clustered SE)")
    model, n = run_pooled_ols(df, 'eta', ['L1_sigma_econ'], "Pooled OLS")
    if model:
        print(f"  N = {n}")
        print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['3a_pooled'] = model
    
    # Model 3b: Entity FE
    if HAS_LINEARMODELS:
        print("\nModel 3b: Entity Fixed Effects")
        model, n = run_fe_panel(df, 'eta', ['L1_sigma_econ'], "Entity FE",
                                entity_effects=True, time_effects=False)
        if model:
            print(f"  N = {n}")
            print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
            print(f"  R² = {model.rsquared:.4f}")
            results_dict['3b_entity_fe'] = model
    
    # Model 3c: Both pathways
    print("\nModel 3c: Both Pathways (Econ→Mig→Eta)")
    model, n = run_pooled_ols(df, 'eta', ['L1_eta', 'L1_sigma_mig', 'L1_sigma_econ'], "Full Model")
    if model:
        print(f"  N = {n}")
        print(f"  L1_eta: coef = {model.params['L1_eta']:.4f}, p = {model.pvalues['L1_eta']:.4f}")
        print(f"  L1_sigma_mig: coef = {model.params['L1_sigma_mig']:.4f}, p = {model.pvalues['L1_sigma_mig']:.4f}")
        print(f"  L1_sigma_econ: coef = {model.params['L1_sigma_econ']:.4f}, p = {model.pvalues['L1_sigma_econ']:.4f}")
        print(f"  R² = {model.rsquared:.4f}")
        results_dict['3c_full'] = model
    
    return results_dict


def summarize_findings(all_results, df):
    """Summarize key findings."""
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    # Count significant results
    print("\nSignificance at p < 0.05:")
    
    h1_sig = sum(1 for r in all_results.get('h1', {}).values() 
                 if r is not None and r.pvalues.get('L1_sigma_econ', 1) < 0.05)
    h2_sig = sum(1 for r in all_results.get('h2', {}).values() 
                 if r is not None and r.pvalues.get('L1_sigma_mig', 1) < 0.05)
    h3_sig = sum(1 for r in all_results.get('h3', {}).values() 
                 if r is not None and r.pvalues.get('L1_sigma_econ', 1) < 0.05)
    
    print(f"  H1 (Econ→Mig): {h1_sig}/{len(all_results.get('h1', {}))} models significant")
    print(f"  H2 (Mig→Eta): {h2_sig}/{len(all_results.get('h2', {}))} models significant")
    print(f"  H3 (Econ→Eta): {h3_sig}/{len(all_results.get('h3', {}))} models significant")
    
    # Key coefficients from full model
    if '3c_full' in all_results.get('h3', {}):
        full = all_results['h3']['3c_full']
        print("\nFrom full pathway model (H3c):")
        print(f"  Lag η persistence: {full.params.get('L1_eta', 0):.4f}")
        print(f"  Mig effect on η: {full.params.get('L1_sigma_mig', 0):.4f}")
        print(f"  Direct Econ effect: {full.params.get('L1_sigma_econ', 0):.4f}")
    
    # Mean reversion test
    print("\n" + "="*70)
    print("IMPLICATIONS")
    print("="*70)
    
    if h2_sig > h1_sig:
        print("• Migration-demographic coupling stronger than econ-migration coupling")
    elif h1_sig > h2_sig:
        print("• Economic-migration coupling stronger than migration-demographic coupling")
    
    if h3_sig > 0 and 'L1_sigma_econ' in all_results.get('h3', {}).get('3c_full', {}).params:
        direct_effect = all_results['h3']['3c_full'].params['L1_sigma_econ']
        if abs(direct_effect) > 0.001:
            print("• Direct economic effects on demographics present")
        else:
            print("• Economic effects operate primarily through migration pathway")


def save_results(all_results, df):
    """Save detailed regression output to file."""
    output_file = RESULTS_DIR / "regression_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("THREE MANIFOLD PANEL REGRESSION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Sample: {df['cbsa_code'].nunique()} MSAs, {df['year'].min()}-{df['year'].max()}\n")
        f.write(f"Total observations: {len(df)}\n\n")
        
        for hypothesis, models in all_results.items():
            f.write("\n" + "="*70 + "\n")
            f.write(f"HYPOTHESIS {hypothesis[-1]}\n")
            f.write("="*70 + "\n\n")
            
            for model_name, result in models.items():
                if result is not None:
                    f.write(f"\n{model_name}:\n")
                    f.write("-" * 50 + "\n")
                    try:
                        f.write(result.summary().as_text())
                    except:
                        f.write(str(result.summary()))
                    f.write("\n\n")
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("PANEL REGRESSION ANALYSIS - Three Manifold Coupling")
    print("="*70)
    
    # Load data
    df = load_panel()
    if df is None:
        return
    
    # Create lags
    df = create_lags(df)
    
    # Run hypothesis tests
    all_results = {}
    all_results['h1'] = test_coupling_econ_to_mig(df)
    all_results['h2'] = test_coupling_mig_to_eta(df)
    all_results['h3'] = test_coupling_econ_to_eta(df)
    
    # Summarize findings
    summarize_findings(all_results, df)
    
    # Save detailed results
    save_results(all_results, df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
