#!/usr/bin/env python3
"""
Fit Urban Temporal Scaling Law.

This script tests the theoretical relationship between demographic coherence (η)
and economic/migration fluxes (σ_econ, σ_mig).

The temporal scaling law hypothesizes:
    σ ∝ η^(-β)  or equivalently  log(σ) = α - β·log(η)

Where:
    - η (eta): demographic structure coherence (higher = more stable structure)
    - σ (sigma): flux/volatility measure (economic or migration)
    - β (beta): scaling exponent (expected positive from theory)

Usage:
    python fit_scaling_law.py

Output:
    results/scaling_analysis_2006-2022.csv
    results/figures/scaling_*.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy for regression
from scipy import stats
from scipy.optimize import curve_fit

# Configuration
RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
    
    print(f"Loaded {len(complete)} MSAs with complete data")
    return complete


def power_law(x, alpha, beta):
    """Power law: y = alpha * x^(-beta)"""
    return alpha * np.power(x, -beta)


def log_log_linear(log_x, log_alpha, beta):
    """Linear in log-log: log(y) = log(alpha) - beta * log(x)"""
    return log_alpha - beta * log_x


def fit_scaling_relationship(df, y_var, y_label, output_prefix):
    """Fit scaling relationship between eta and a sigma variable."""
    
    # Remove zero or negative values for log transform
    valid = df[(df['eta_overall'] > 0) & (df[y_var] > 0)].copy()
    
    if len(valid) < 10:
        print(f"  Insufficient data for {y_var}")
        return None
    
    x = valid['eta_overall'].values
    y = valid[y_var].values
    
    log_x = np.log(x)
    log_y = np.log(y)
    
    results = {}
    
    # Method 1: Linear regression on log-log
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # In log-log: log(y) = intercept + slope * log(x)
    # We want: log(y) = log(alpha) - beta * log(x)
    # So slope = -beta, intercept = log(alpha)
    beta_lr = -slope
    alpha_lr = np.exp(intercept)
    r_squared_lr = r_value**2
    
    results['linear_regression'] = {
        'alpha': alpha_lr,
        'beta': beta_lr,
        'r_squared': r_squared_lr,
        'p_value': p_value,
        'std_err': std_err,
        'n_obs': len(valid)
    }
    
    # Method 2: Nonlinear least squares on power law
    try:
        popt, pcov = curve_fit(power_law, x, y, p0=[alpha_lr, beta_lr], maxfev=10000)
        alpha_nls, beta_nls = popt
        
        # Compute R-squared
        y_pred = power_law(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared_nls = 1 - (ss_res / ss_tot)
        
        results['nonlinear_ls'] = {
            'alpha': alpha_nls,
            'beta': beta_nls,
            'r_squared': r_squared_nls,
            'n_obs': len(valid)
        }
    except Exception as e:
        results['nonlinear_ls'] = {'error': str(e)}
    
    # Method 3: Reduced Major Axis (RMA) regression
    # Accounts for error in both variables
    try:
        # RMA slope = sign(r) * sd(y) / sd(x) for log-log
        rma_beta = np.std(log_y) / np.std(log_x)
        rma_intercept = np.mean(log_y) + rma_beta * np.mean(log_x)
        rma_alpha = np.exp(rma_intercept)
        
        # RMA doesn't directly give R^2, use correlation
        rma_r = np.corrcoef(log_x, log_y)[0, 1]
        
        results['rma'] = {
            'alpha': rma_alpha,
            'beta': rma_beta,
            'r_squared': rma_r**2,
            'n_obs': len(valid)
        }
    except Exception as e:
        results['rma'] = {'error': str(e)}
    
    # Print results
    print(f"\n  {y_label}:")
    print(f"    Observations: {len(valid)}")
    print(f"    Linear regression (log-log):")
    print(f"      α = {alpha_lr:.6f}")
    print(f"      β = {beta_lr:.4f}")
    print(f"      R² = {r_squared_lr:.4f}")
    print(f"      p-value = {p_value:.4f}")
    
    if 'nonlinear_ls' in results and 'error' not in results['nonlinear_ls']:
        print(f"    Nonlinear least squares:")
        print(f"      α = {alpha_nls:.6f}")
        print(f"      β = {beta_nls:.4f}")
        print(f"      R² = {r_squared_nls:.4f}")
    
    if 'rma' in results and 'error' not in results['rma']:
        print(f"    Reduced Major Axis:")
        print(f"      α = {rma_alpha:.6f}")
        print(f"      β = {rma_beta:.4f}")
        print(f"      R² = {rma_r**2:.4f}")
    
    # Compute predictions for plotting
    valid['y_pred'] = power_law(valid['eta_overall'], alpha_lr, beta_lr)
    valid['log_eta'] = np.log(valid['eta_overall'])
    valid['log_y'] = np.log(valid[y_var])
    valid['log_y_pred'] = np.log(valid['y_pred'])
    
    return results, valid


def create_visualization(df, y_var, y_label, beta, r_squared, output_file):
    """Create visualization of scaling relationship."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Log-log scatter with fit line
        ax1 = axes[0]
        ax1.scatter(df['log_eta'], df['log_y'], alpha=0.5, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
        
        # Fit line
        x_fit = np.linspace(df['log_eta'].min(), df['log_eta'].max(), 100)
        y_fit = np.log(df['y_pred'].iloc[0]) - beta * (x_fit - df['log_eta'].iloc[0])
        # Better fit line
        log_alpha = np.mean(df['log_y']) + beta * np.mean(df['log_eta'])
        y_fit = log_alpha - beta * x_fit
        
        ax1.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: β={beta:.3f}')
        ax1.set_xlabel('log(η)', fontsize=12)
        ax1.set_ylabel(f'log({y_var})', fontsize=12)
        ax1.set_title(f'{y_label} vs Demographic Coherence\n(R² = {r_squared:.3f})', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Linear scale
        ax2 = axes[1]
        ax2.scatter(df['eta_overall'], df[y_var], alpha=0.5, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
        
        # Power law curve
        x_curve = np.linspace(df['eta_overall'].min(), df['eta_overall'].max(), 100)
        alpha = np.exp(np.mean(df['log_y']) + beta * np.mean(df['log_eta']))
        y_curve = alpha * np.power(x_curve, -beta)
        ax2.plot(x_curve, y_curve, 'r--', linewidth=2, label=f'σ = {alpha:.4f}·η^(-{beta:.3f})')
        
        ax2.set_xlabel('η (demographic coherence)', fontsize=12)
        ax2.set_ylabel(y_label, fontsize=12)
        ax2.set_title(f'Power Law Relationship', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {output_file.name}")
        return True
    except ImportError:
        print("    (matplotlib not available, skipping visualization)")
        return False
    except Exception as e:
        print(f"    Error creating visualization: {e}")
        return False


def analyze_residuals(df, y_var, output_prefix):
    """Analyze residuals for systematic patterns."""
    residuals = df['log_y'] - df['log_y_pred']
    
    print(f"\n  Residual Analysis for {y_var}:")
    print(f"    Mean residual: {residuals.mean():.4f}")
    print(f"    Std residual: {residuals.std():.4f}")
    print(f"    Shapiro-Wilk p (normality): {stats.shapiro(residuals)[1]:.4f}")
    
    # Test for heteroscedasticity
    corr_eta_resid = np.corrcoef(df['log_eta'], np.abs(residuals))[0, 1]
    print(f"    Corr(|resid|, log(η)): {corr_eta_resid:.4f}")
    
    return residuals


def compare_scaling_exponents(results_econ, results_mig):
    """Compare scaling exponents between economic and migration flux."""
    print(f"\n{'='*50}")
    print("COMPARISON OF SCALING EXPONENTS")
    print(f"{'='*50}")
    
    if 'linear_regression' in results_econ and 'linear_regression' in results_mig:
        beta_econ = results_econ['linear_regression']['beta']
        beta_mig = results_mig['linear_regression']['beta']
        
        print(f"\n  β (economic flux):   {beta_econ:.4f}")
        print(f"  β (migration flux):  {beta_mig:.4f}")
        print(f"  Difference:          {beta_econ - beta_mig:.4f}")
        
        # Theoretical expectation: both should be positive
        # with economic flux potentially having stronger scaling
        if beta_econ > 0 and beta_mig > 0:
            print(f"\n  ✓ Both exponents positive (consistent with theory)")
        else:
            print(f"\n  ⚠ Some exponents non-positive (unexpected)")
        
        return {'beta_econ': beta_econ, 'beta_mig': beta_mig}
    
    return None


def create_combined_visualization(df_econ, df_mig, beta_econ, beta_mig, r2_econ, r2_mig):
    """Create combined visualization comparing both fluxes."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Row 1: Economic flux
        ax = axes[0, 0]
        ax.scatter(df_econ['log_eta'], df_econ['log_y'], alpha=0.5, s=50, c='darkgreen', edgecolors='white')
        log_alpha = np.mean(df_econ['log_y']) + beta_econ * np.mean(df_econ['log_eta'])
        x_fit = np.linspace(df_econ['log_eta'].min(), df_econ['log_eta'].max(), 100)
        y_fit = log_alpha - beta_econ * x_fit
        ax.plot(x_fit, y_fit, 'r--', linewidth=2)
        ax.set_xlabel('log(η)', fontsize=11)
        ax.set_ylabel('log(σ_econ)', fontsize=11)
        ax.set_title(f'Economic Flux (β={beta_econ:.3f}, R²={r2_econ:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.scatter(df_econ['eta_overall'], df_econ['sigma_econ'], alpha=0.5, s=50, c='darkgreen', edgecolors='white')
        x_curve = np.linspace(df_econ['eta_overall'].min(), df_econ['eta_overall'].max(), 100)
        alpha = np.exp(log_alpha)
        y_curve = alpha * np.power(x_curve, -beta_econ)
        ax.plot(x_curve, y_curve, 'r--', linewidth=2)
        ax.set_xlabel('η', fontsize=11)
        ax.set_ylabel('σ_econ', fontsize=11)
        ax.set_title('Economic Flux (linear)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Row 2: Migration flux
        ax = axes[1, 0]
        ax.scatter(df_mig['log_eta'], df_mig['log_y'], alpha=0.5, s=50, c='darkorange', edgecolors='white')
        log_alpha = np.mean(df_mig['log_y']) + beta_mig * np.mean(df_mig['log_eta'])
        x_fit = np.linspace(df_mig['log_eta'].min(), df_mig['log_eta'].max(), 100)
        y_fit = log_alpha - beta_mig * x_fit
        ax.plot(x_fit, y_fit, 'r--', linewidth=2)
        ax.set_xlabel('log(η)', fontsize=11)
        ax.set_ylabel('log(σ_mig)', fontsize=11)
        ax.set_title(f'Migration Flux (β={beta_mig:.3f}, R²={r2_mig:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.scatter(df_mig['eta_overall'], df_mig['sigma_mig'], alpha=0.5, s=50, c='darkorange', edgecolors='white')
        x_curve = np.linspace(df_mig['eta_overall'].min(), df_mig['eta_overall'].max(), 100)
        alpha = np.exp(log_alpha)
        y_curve = alpha * np.power(x_curve, -beta_mig)
        ax.plot(x_curve, y_curve, 'r--', linewidth=2)
        ax.set_xlabel('η', fontsize=11)
        ax.set_ylabel('σ_mig', fontsize=11)
        ax.set_title('Migration Flux (linear)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Urban Temporal Scaling Law: η vs σ', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = FIGURES_DIR / "scaling_combined.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Saved combined figure: scaling_combined.png")
        return True
    except Exception as e:
        print(f"  Error creating combined visualization: {e}")
        return False


def save_results_summary(results_econ, results_mig, comparison, df):
    """Save summary of scaling analysis."""
    summary = {
        'metric': [],
        'economic_flux': [],
        'migration_flux': []
    }
    
    # Add results from linear regression
    if 'linear_regression' in results_econ:
        lr = results_econ['linear_regression']
        summary['metric'].extend(['alpha', 'beta', 'r_squared', 'p_value', 'n_obs'])
        summary['economic_flux'].extend([lr['alpha'], lr['beta'], lr['r_squared'], lr['p_value'], int(lr['n_obs'])])
    
    if 'linear_regression' in results_mig:
        lr = results_mig['linear_regression']
        summary['migration_flux'].extend([lr['alpha'], lr['beta'], lr['r_squared'], lr['p_value'], int(lr['n_obs'])])
    
    summary_df = pd.DataFrame(summary)
    output_path = RESULTS_DIR / "scaling_analysis_2006-2022.csv"
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n  Saved: scaling_analysis_2006-2022.csv")


def main():
    """Main analysis pipeline."""
    print("="*50)
    print("URBAN TEMPORAL SCALING LAW")
    print("="*50)
    print("\nTheoretical Framework:")
    print("  Testing: σ = α · η^(-β)")
    print("  Where:")
    print("    η = demographic structure coherence")
    print("    σ = flux/volatility (economic or migration)")
    print("    α = proportionality constant")
    print("    β = scaling exponent (expected > 0)")
    
    # Load data
    df = load_master_data()
    if df is None:
        return None
    
    print(f"\n{'='*50}")
    print("SCALING ANALYSIS")
    print(f"{'='*50}")
    
    # Fit economic flux scaling
    results_econ, df_econ = fit_scaling_relationship(
        df, 'sigma_econ', 'Economic Flux (σ_econ)', 'econ'
    )
    
    if results_econ:
        lr = results_econ['linear_regression']
        create_visualization(
            df_econ, 'sigma_econ', 'Economic Flux',
            lr['beta'], lr['r_squared'],
            FIGURES_DIR / "scaling_economic.png"
        )
        analyze_residuals(df_econ, 'sigma_econ', 'econ')
    
    # Fit migration flux scaling
    results_mig, df_mig = fit_scaling_relationship(
        df, 'sigma_mig', 'Migration Flux (σ_mig)', 'mig'
    )
    
    if results_mig:
        lr = results_mig['linear_regression']
        create_visualization(
            df_mig, 'sigma_mig', 'Migration Flux',
            lr['beta'], lr['r_squared'],
            FIGURES_DIR / "scaling_migration.png"
        )
        analyze_residuals(df_mig, 'sigma_mig', 'mig')
    
    # Compare exponents
    comparison = None
    if results_econ and results_mig:
        comparison = compare_scaling_exponents(results_econ, results_mig)
        
        # Create combined visualization
        lr_econ = results_econ['linear_regression']
        lr_mig = results_mig['linear_regression']
        create_combined_visualization(
            df_econ, df_mig,
            lr_econ['beta'], lr_mig['beta'],
            lr_econ['r_squared'], lr_mig['r_squared']
        )
        
        # Save summary
        save_results_summary(results_econ, results_mig, comparison, df)
    
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    
    if results_econ and results_mig:
        beta_econ = results_econ['linear_regression']['beta']
        beta_mig = results_mig['linear_regression']['beta']
        r2_econ = results_econ['linear_regression']['r_squared']
        r2_mig = results_mig['linear_regression']['r_squared']
        p_econ = results_econ['linear_regression']['p_value']
        p_mig = results_mig['linear_regression']['p_value']
        
        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  SCALING LAW: σ = α · η^(-β)                   │")
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │  Parameter    │  Economic Flux  │ Migration    │")
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │  α (alpha)    │  {results_econ['linear_regression']['alpha']:.6f}  │ {results_mig['linear_regression']['alpha']:.6f} │")
        print(f"  │  β (beta)     │  {beta_econ:+.4f}       │ {beta_mig:+.4f}      │")
        print(f"  │  R²           │  {r2_econ:.4f}        │ {r2_mig:.4f}     │")
        print(f"  │  p-value      │  {p_econ:.4f}        │ {p_mig:.4f}     │")
        print(f"  │  n (MSAs)     │  {results_econ['linear_regression']['n_obs']:.0f}           │ {results_mig['linear_regression']['n_obs']:.0f}          │")
        print(f"  └─────────────────────────────────────────────────┘")
        
        print(f"\n  KEY FINDINGS:")
        
        # Economic flux interpretation
        if p_econ < 0.05:
            print(f"    • Economic flux: Significant negative scaling (β={beta_econ:.3f})")
            print(f"      → More demographically coherent MSAs have slightly lower")
            print(f"        economic volatility (σ_econ ∝ η^(-{beta_econ:.3f}))")
        elif p_econ < 0.10:
            print(f"    • Economic flux: Marginally significant (p={p_econ:.3f})")
            print(f"      → Weak trend: σ_econ decreases with η (β={beta_econ:.3f})")
        else:
            print(f"    • Economic flux: No significant scaling (p={p_econ:.3f})")
            print(f"      → Demographic structure explains only {r2_econ*100:.2f}% of")
            print(f"        economic volatility variance")
        
        # Migration flux interpretation
        if p_mig < 0.05:
            print(f"    • Migration flux: Significant scaling (β={beta_mig:.3f})")
        else:
            print(f"    • Migration flux: No significant relationship (p={p_mig:.3f})")
            print(f"      → Demographic structure explains virtually none of the")
            print(f"        migration flux variance (R²={r2_mig:.6f})")
        
        print(f"\n  THEORETICAL IMPLICATIONS:")
        
        if r2_econ < 0.05 and r2_mig < 0.05:
            print(f"    ⚠ The temporal scaling law σ ∝ η^(-β) finds LIMITED support:")
            print(f"")
            print(f"      1. The hypothesized inverse relationship is extremely weak")
            print(f"      2. Urban dynamics appear driven by factors beyond demographic")
            print(f"         structure alone (e.g., industry composition, policy,")
            print(f"         geographic constraints, network effects)")
            print(f"      3. The Fisher-Rao metric captures structural stability but")
            print(f"         not necessarily system-wide coherence that constrains flux")
            print(f"")
            print(f"      Possible explanations:")
            print(f"      • η measures demographic distribution stability, not the")
            print(f"        underlying social/economic integration that constrains flux")
            print(f"      • US MSAs during 2006-2022 experienced external shocks")
            print(f"        (financial crisis, COVID, tech disruption) dominating")
            print(f"        endogenous demographic coherence effects")
            print(f"      • The three dimensions (η, σ_econ, σ_mig) may operate on")
            print(f"        different time scales or through different mechanisms")
        else:
            print(f"    ✓ The temporal scaling law finds partial support")
            print(f"      • Economic flux shows weak but detectable scaling")
            print(f"      • Migration flux appears independent of demographic coherence")
    
    print(f"\n{'='*50}")
    print("COMPLETE")
    print(f"{'='*50}")
    
    return results_econ, results_mig


if __name__ == "__main__":
    main()
