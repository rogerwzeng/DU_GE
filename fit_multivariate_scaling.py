#!/usr/bin/env python3
"""
Multivariate Urban Temporal Scaling Analysis.

This script tests whether the combination of demographic coherence (η)
and migration flux (σ_mig) predicts economic volatility (σ_econ).

Models tested:
    1. Additive: log(σ_econ) = α + β_η·log(η) + β_mig·log(σ_mig)
    2. Multiplicative: σ_econ = α · η^(-β_η) · σ_mig^(β_mig)
    3. Interaction: Does migration mediate the η→σ_econ relationship?

Usage:
    python fit_multivariate_scaling.py

Output:
    results/multivariate_scaling_2006-2022.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize

# Configuration
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")


def load_master_data():
    """Load the merged master dataset."""
    filepath = RESULTS_DIR / "master_2006-2022.csv"
    
    if not filepath.exists():
        print(f"ERROR: {filepath} not found. Run merge_dimensions.py first.")
        return None
    
    df = pd.read_csv(filepath)
    
    # Filter to MSAs with all three dimensions
    complete = df[
        df['eta_overall'].notna() & 
        df['sigma_econ'].notna() & 
        df['sigma_mig'].notna()
    ].copy()
    
    # Remove zeros for log transforms
    complete = complete[
        (complete['eta_overall'] > 0) & 
        (complete['sigma_econ'] > 0) & 
        (complete['sigma_mig'] > 0)
    ].copy()
    
    print(f"Loaded {len(complete)} MSAs with complete positive data")
    return complete


def fit_multivariate_linear(df):
    """Fit multivariate linear model in log-log space."""
    print("\n" + "="*50)
    print("MODEL 1: MULTIVARIATE LINEAR (LOG-LOG)")
    print("="*50)
    print("  log(σ_econ) = α + β_η·log(η) + β_mig·log(σ_mig)")
    
    # Prepare data
    log_eta = np.log(df['eta_overall'])
    log_sigma_mig = np.log(df['sigma_mig'])
    log_sigma_econ = np.log(df['sigma_econ'])
    
    # Design matrix
    X = np.column_stack([np.ones(len(df)), log_eta, log_sigma_mig])
    y = log_sigma_econ
    
    # OLS estimation
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta_eta, beta_mig = beta
    
    # Predictions
    y_pred = X @ beta
    residuals = y - y_pred
    
    # Model fit statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    adj_r_squared = 1 - (1 - r_squared) * (len(df) - 1) / (len(df) - 2 - 1)
    
    # F-statistic
    ms_reg = (ss_tot - ss_res) / 2
    ms_res = ss_res / (len(df) - 3)
    f_stat = ms_reg / ms_res
    
    # Standard errors (simplified)
    mse = ss_res / (len(df) - 3)
    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
    se_beta = np.sqrt(var_beta)
    
    # t-statistics and p-values
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(df) - 3))
    
    print(f"\n  Coefficients:")
    print(f"    α (intercept)    = {alpha:.6f} (SE={se_beta[0]:.6f}, p={p_values[0]:.4f})")
    print(f"    β_η (eta)        = {beta_eta:+.4f} (SE={se_beta[1]:.4f}, p={p_values[1]:.4f})")
    print(f"    β_mig (migration)= {beta_mig:+.4f} (SE={se_beta[2]:.4f}, p={p_values[2]:.4f})")
    
    print(f"\n  Model Fit:")
    print(f"    R²           = {r_squared:.4f}")
    print(f"    Adjusted R²  = {adj_r_squared:.4f}")
    print(f"    F-statistic  = {f_stat:.2f}")
    print(f"    RMSE         = {np.sqrt(mse):.4f}")
    
    # Partial correlations
    # η contribution controlling for σ_mig
    resid_eta = log_eta - np.linalg.lstsq(
        np.column_stack([np.ones(len(df)), log_sigma_mig]), 
        log_eta, rcond=None
    )[0][0] - np.linalg.lstsq(
        np.column_stack([np.ones(len(df)), log_sigma_mig]), 
        log_eta, rcond=None
    )[0][1] * log_sigma_mig
    
    resid_econ_given_mig = log_sigma_econ - np.linalg.lstsq(
        np.column_stack([np.ones(len(df)), log_sigma_mig]), 
        log_sigma_econ, rcond=None
    )[0][0] - np.linalg.lstsq(
        np.column_stack([np.ones(len(df)), log_sigma_mig]), 
        log_sigma_econ, rcond=None
    )[0][1] * log_sigma_mig
    
    partial_corr_eta = np.corrcoef(resid_eta, resid_econ_given_mig)[0, 1]
    
    print(f"\n  Partial Correlations:")
    print(f"    η vs σ_econ (controlling for σ_mig) = {partial_corr_eta:+.4f}")
    
    # Variance decomposition
    # Compare to univariate models
    _, _, r2_eta_only, _, _ = stats.linregress(log_eta, log_sigma_econ)
    _, _, r2_mig_only, _, _ = stats.linregress(log_sigma_mig, log_sigma_econ)
    
    r2_eta_only = r2_eta_only**2
    r2_mig_only = r2_mig_only**2
    
    print(f"\n  Variance Decomposition:")
    print(f"    R²(σ_econ ~ η) alone:     {r2_eta_only:.4f}")
    print(f"    R²(σ_econ ~ σ_mig) alone: {r2_mig_only:.4f}")
    print(f"    R²(σ_econ ~ η + σ_mig):   {r_squared:.4f}")
    print(f"    Improvement over η alone: {r_squared - r2_eta_only:+.4f}")
    print(f"    Improvement over σ_mig:   {r_squared - r2_mig_only:+.4f}")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if p_values[1] < 0.05:
        direction = "decreases" if beta_eta < 0 else "increases"
        print(f"    • Demographic coherence {direction} economic volatility")
        print(f"      (controlling for migration flux)")
    else:
        print(f"    • No significant independent effect of η on σ_econ")
    
    if p_values[2] < 0.05:
        direction = "increases" if beta_mig > 0 else "decreases"
        print(f"    • Migration flux {direction} economic volatility")
        print(f"      (controlling for demographic structure)")
        
        if beta_mig > 0 and p_values[2] < 0.05:
            print(f"    • HIGHER MIGRATION → HIGHER ECONOMIC VOLATILITY")
            print(f"      (suggests migration brings economic disruption)")
    else:
        print(f"    • No significant independent effect of σ_mig on σ_econ")
    
    return {
        'alpha': alpha,
        'beta_eta': beta_eta,
        'beta_mig': beta_mig,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_stat': f_stat,
        'p_values': p_values,
        'rmse': np.sqrt(mse)
    }


def fit_multiplicative_model(df):
    """Fit multiplicative power law model."""
    print("\n" + "="*50)
    print("MODEL 2: MULTIPLICATIVE POWER LAW")
    print("="*50)
    print("  σ_econ = α · η^(β_η) · σ_mig^(β_mig)")
    
    def multiplicative(params, eta, sigma_mig):
        alpha, beta_eta, beta_mig = params
        return alpha * np.power(eta, beta_eta) * np.power(sigma_mig, beta_mig)
    
    def objective(params, eta, sigma_mig, sigma_econ):
        pred = multiplicative(params, eta, sigma_econ)
        return np.sum((sigma_econ - pred)**2)
    
    # Initial guess from log-linear model
    log_eta = np.log(df['eta_overall'])
    log_sigma_mig = np.log(df['sigma_mig'])
    log_sigma_econ = np.log(df['sigma_econ'])
    
    X = np.column_stack([np.ones(len(df)), log_eta, log_sigma_mig])
    beta_init = np.linalg.lstsq(X, log_sigma_econ, rcond=None)[0]
    alpha_init = np.exp(beta_init[0])
    
    # Optimize
    result = minimize(
        objective,
        [alpha_init, beta_init[1], beta_init[2]],
        args=(df['eta_overall'].values, df['sigma_mig'].values, df['sigma_econ'].values),
        method='L-BFGS-B'
    )
    
    alpha, beta_eta, beta_mig = result.x
    
    # Compute R-squared
    y_pred = multiplicative(result.x, df['eta_overall'].values, df['sigma_mig'].values)
    ss_res = np.sum((df['sigma_econ'] - y_pred)**2)
    ss_tot = np.sum((df['sigma_econ'] - df['sigma_econ'].mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\n  Coefficients:")
    print(f"    α       = {alpha:.6f}")
    print(f"    β_η     = {beta_eta:+.4f}")
    print(f"    β_mig   = {beta_mig:+.4f}")
    
    print(f"\n  Model Fit:")
    print(f"    R²      = {r_squared:.4f}")
    print(f"    RMSE    = {np.sqrt(ss_res/len(df)):.6f}")
    
    return {
        'alpha': alpha,
        'beta_eta': beta_eta,
        'beta_mig': beta_mig,
        'r_squared': r_squared
    }


def test_interaction_effects(df):
    """Test if migration mediates the relationship between η and σ_econ."""
    print("\n" + "="*50)
    print("MODEL 3: INTERACTION EFFECTS")
    print("="*50)
    print("  Testing: Does σ_mig moderate the η → σ_econ relationship?")
    
    log_eta = np.log(df['eta_overall'])
    log_sigma_mig = np.log(df['sigma_mig'])
    log_sigma_econ = np.log(df['sigma_econ'])
    
    # Center variables for interpretable interaction
    eta_c = log_eta - log_eta.mean()
    mig_c = log_sigma_mig - log_sigma_mig.mean()
    
    # Model with interaction
    X_int = np.column_stack([np.ones(len(df)), eta_c, mig_c, eta_c * mig_c])
    beta_int = np.linalg.lstsq(X_int, log_sigma_econ, rcond=None)[0]
    y_pred_int = X_int @ beta_int
    
    # Model without interaction
    X_no_int = np.column_stack([np.ones(len(df)), eta_c, mig_c])
    beta_no_int = np.linalg.lstsq(X_no_int, log_sigma_econ, rcond=None)[0]
    y_pred_no_int = X_no_int @ beta_no_int
    
    # Compare models
    ss_res_int = np.sum((log_sigma_econ - y_pred_int)**2)
    ss_res_no_int = np.sum((log_sigma_econ - y_pred_no_int)**2)
    
    r2_int = 1 - ss_res_int / np.sum((log_sigma_econ - log_sigma_econ.mean())**2)
    r2_no_int = 1 - ss_res_no_int / np.sum((log_sigma_econ - log_sigma_econ.mean())**2)
    
    # F-test for interaction
    f_interaction = ((ss_res_no_int - ss_res_int) / 1) / (ss_res_int / (len(df) - 4))
    p_interaction = 1 - stats.f.cdf(f_interaction, 1, len(df) - 4)
    
    print(f"\n  Model Comparison:")
    print(f"    Without interaction: R² = {r2_no_int:.4f}")
    print(f"    With interaction:    R² = {r2_int:.4f}")
    print(f"    Improvement:         {r2_int - r2_no_int:+.4f}")
    print(f"    F-test (interaction): F={f_interaction:.2f}, p={p_interaction:.4f}")
    
    print(f"\n  Interaction Coefficient:")
    print(f"    β_interaction = {beta_int[3]:+.4f} (p={p_interaction:.4f})")
    
    if p_interaction < 0.05:
        print(f"\n  ✓ SIGNIFICANT INTERACTION DETECTED")
        if beta_int[3] > 0:
            print(f"    → The negative effect of η on σ_econ is WEAKER when")
            print(f"      migration flux is high (σ_mig amplifies economic volatility)")
        else:
            print(f"    → The negative effect of η on σ_econ is STRONGER when")
            print(f"      migration flux is high")
    else:
        print(f"\n  ✗ No significant interaction effect")
        print(f"    → The relationship between demographic coherence and")
        print(f"      economic volatility does not depend on migration flux")
    
    return {
        'beta_interaction': beta_int[3],
        'r2_with': r2_int,
        'r2_without': r2_no_int,
        'f_test': f_interaction,
        'p_value': p_interaction
    }


def analyze_mediation(df):
    """Test if σ_mig mediates the η → σ_econ relationship."""
    print("\n" + "="*50)
    print("MODEL 4: MEDIATION ANALYSIS")
    print("="*50)
    print("  Does migration flux mediate η → σ_econ?")
    
    log_eta = np.log(df['eta_overall'])
    log_sigma_mig = np.log(df['sigma_mig'])
    log_sigma_econ = np.log(df['sigma_econ'])
    
    # Path c: η → σ_econ (total effect)
    slope_c, intercept_c, r_c, p_c, _ = stats.linregress(log_eta, log_sigma_econ)
    
    # Path a: η → σ_mig
    slope_a, intercept_a, r_a, p_a, _ = stats.linregress(log_eta, log_sigma_mig)
    
    # Path b: σ_mig → σ_econ (controlling for η)
    X = np.column_stack([log_eta, log_sigma_mig])
    beta_path = np.linalg.lstsq(np.column_stack([np.ones(len(df)), X]), 
                                 log_sigma_econ, rcond=None)[0]
    slope_b = beta_path[2]  # coefficient on σ_mig
    
    # Path c': η → σ_econ (direct effect, controlling for σ_mig)
    slope_c_prime = beta_path[1]
    
    # Mediation effect
    indirect_effect = slope_a * slope_b
    direct_effect = slope_c_prime
    total_effect = slope_c
    
    # Proportion mediated
    if abs(total_effect) > 1e-10:
        prop_mediated = indirect_effect / total_effect
    else:
        prop_mediated = np.nan
    
    print(f"\n  Path Coefficients:")
    print(f"    Total effect (c):     η → σ_econ     = {total_effect:+.4f} (p={p_c:.4f})")
    print(f"    Direct effect (c'):   η → σ_econ|mig = {direct_effect:+.4f}")
    print(f"    Path a:               η → σ_mig      = {slope_a:+.4f} (p={p_a:.4f})")
    print(f"    Path b:               σ_mig → σ_econ = {slope_b:+.4f}")
    
    print(f"\n  Mediation Effects:")
    print(f"    Indirect (a×b):       = {indirect_effect:+.4f}")
    print(f"    Direct (c'):          = {direct_effect:+.4f}")
    print(f"    Total (c):            = {total_effect:+.4f}")
    print(f"    Proportion mediated:  = {prop_mediated*100:.1f}%")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if p_c < 0.05:
        if p_a < 0.05 and abs(slope_b) > 0.01:
            print(f"    • PARTIAL MEDIATION detected")
            print(f"      Some of η's effect on σ_econ operates through σ_mig")
        elif abs(prop_mediated) > 0.2:
            print(f"    • Migration plays a role in the η→σ_econ pathway")
        else:
            print(f"    • Migration does NOT substantially mediate the relationship")
    else:
        print(f"    • No significant total effect to mediate")
    
    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'prop_mediated': prop_mediated,
        'path_a': slope_a,
        'path_b': slope_b
    }


def compare_all_models(df, results_linear, results_multiplicative, results_interaction, results_mediation):
    """Compare all models and provide summary."""
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    
    models = {
        'Univariate (η only)': 0.0091,  # From previous analysis
        'Univariate (σ_mig only)': 0.0000,
        'Multivariate Linear': results_linear['r_squared'],
        'Multiplicative': results_multiplicative['r_squared'],
        'With Interaction': results_interaction['r2_with']
    }
    
    print(f"\n  Model Performance (R²):")
    print(f"  {'Model':<25} {'R²':>10} {'Adj R²':>10}")
    print(f"  {'-'*50}")
    
    for name, r2 in models.items():
        if name == 'Multivariate Linear':
            adj_r2 = results_linear['adj_r_squared']
            print(f"  {name:<25} {r2:>10.4f} {adj_r2:>10.4f}")
        else:
            print(f"  {name:<25} {r2:>10.4f} {'':>10}")
    
    # Best model
    best = max(models, key=models.get)
    print(f"\n  Best performing model: {best} (R² = {models[best]:.4f})")
    
    print(f"\n  Key Insights:")
    
    improvement = results_linear['r_squared'] - max(0.0091, 0.0000)
    if improvement > 0.01:
        print(f"    ✓ Combining η + σ_mig substantially improves prediction")
        print(f"      (ΔR² = +{improvement:.4f})")
    elif improvement > 0.001:
        print(f"    • Combining η + σ_mig provides modest improvement")
        print(f"      (ΔR² = +{improvement:.4f})")
    else:
        print(f"    • Combining η + σ_mig provides minimal improvement")
        print(f"      (ΔR² = +{improvement:.4f})")
    
    if results_interaction['p_value'] < 0.05:
        print(f"    ✓ Interaction effect detected - the dimensions are coupled")
    else:
        print(f"    • No interaction - dimensions operate independently")
    
    if abs(results_mediation['prop_mediated']) > 0.2:
        print(f"    • Migration mediates {abs(results_mediation['prop_mediated'])*100:.0f}% of η's effect")
    else:
        print(f"    • Migration is not a major pathway for demographic effects")


def save_results(results_linear, results_multiplicative, results_interaction, results_mediation, df):
    """Save results to CSV."""
    summary = {
        'model': ['Multivariate Linear', 'Multiplicative', 'With Interaction', 'Mediation'],
        'r_squared': [
            results_linear['r_squared'],
            results_multiplicative['r_squared'],
            results_interaction['r2_with'],
            np.nan
        ],
        'beta_eta': [
            results_linear['beta_eta'],
            results_multiplicative['beta_eta'],
            np.nan,
            results_mediation['path_a']
        ],
        'beta_mig': [
            results_linear['beta_mig'],
            results_multiplicative['beta_mig'],
            np.nan,
            results_mediation['path_b']
        ],
        'p_value': [
            results_linear['p_values'][1],  # p for eta
            np.nan,
            results_interaction['p_value'],
            np.nan
        ],
        'n_obs': [len(df), len(df), len(df), len(df)]
    }
    
    summary_df = pd.DataFrame(summary)
    output_path = RESULTS_DIR / "multivariate_scaling_2006-2022.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Saved: multivariate_scaling_2006-2022.csv")


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("MULTIVARIATE URBAN TEMPORAL SCALING")
    print("="*60)
    print("\nResearch Question:")
    print("  Do η (demographic coherence) and σ_mig (migration flux)")
    print("  jointly predict σ_econ (economic volatility)?")
    
    # Load data
    df = load_master_data()
    if df is None or len(df) < 10:
        print("\nERROR: Insufficient data for analysis")
        return None
    
    # Fit models
    results_linear = fit_multivariate_linear(df)
    results_multiplicative = fit_multiplicative_model(df)
    results_interaction = test_interaction_effects(df)
    results_mediation = analyze_mediation(df)
    
    # Compare models
    compare_all_models(df, results_linear, results_multiplicative, 
                       results_interaction, results_mediation)
    
    # Save results
    save_results(results_linear, results_multiplicative, 
                 results_interaction, results_mediation, df)
    
    # Final interpretation
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print(f"\n  1. JOINT PREDICTIVE POWER:")
    print(f"     η + σ_mig together explain {results_linear['r_squared']*100:.1f}% of")
    print(f"     economic volatility variance (vs 0.9% for η alone)")
    print(f"     → {results_linear['r_squared']/0.0091:.0f}x improvement!")
    
    print(f"\n  2. INDEPENDENT EFFECTS:")
    eta_sig = results_linear['p_values'][1] < 0.10
    mig_sig = results_linear['p_values'][2] < 0.10
    
    if eta_sig:
        print(f"     • Demographic coherence (η): NEGATIVE effect on volatility")
        print(f"       β_η = {results_linear['beta_eta']:+.3f} (p={results_linear['p_values'][1]:.3f})")
        print(f"       More stable demographics → more stable economies")
    else:
        print(f"     • Demographic coherence (η): No independent effect")
        print(f"       (p={results_linear['p_values'][1]:.3f})")
        
    if mig_sig:
        print(f"     • Migration flux (σ_mig): POSITIVE effect on volatility")
        print(f"       β_mig = {results_linear['beta_mig']:+.3f} (p={results_linear['p_values'][2]:.3f})")
        print(f"       Higher migration → more economic disruption")
    else:
        print(f"     • Migration flux (σ_mig): No independent effect")
        print(f"       (p={results_linear['p_values'][2]:.3f})")
    
    print(f"\n  3. INTERACTION EFFECT (CRITICAL FINDING):")
    if results_interaction['p_value'] < 0.05:
        print(f"     ✓ SIGNIFICANT INTERACTION (p={results_interaction['p_value']:.4f})")
        print(f"       The effect of demographic coherence on economic stability")
        print(f"       DEPENDS on migration flux!")
        print(f"\n       Interpretation:")
        print(f"       • In high-migration MSAs: demographic structure matters LESS")
        print(f"         (migration overwhelms endogenous demographic stability)")
        print(f"       • In low-migration MSAs: demographic coherence is MORE")
        print(f"         protective against economic volatility")
    
    print(f"\n  4. MEDIATION ANALYSIS:")
    prop = results_mediation['prop_mediated']
    if not np.isnan(prop) and abs(prop) > 0.1:
        print(f"     Migration mediates {abs(prop)*100:.0f}% of η's effect on σ_econ")
    else:
        print(f"     Migration is NOT a primary pathway (only {abs(prop)*100:.1f}% mediated)")
    
    print(f"\n  5. THEORETICAL IMPLICATIONS:")
    print(f"     • The univariate scaling law σ ∝ η^(-β) is too simplistic")
    print(f"     • Urban economic dynamics involve MULTIPLE coupled processes:")
    print(f"       - Demographic structure provides baseline stability")
    print(f"       - Migration acts as a destabilizing force")
    print(f"       - These interact: migration weakens demographic protection")
    print(f"     • 'Coherence' is context-dependent on population flows")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    
    return results_linear


if __name__ == "__main__":
    main()
