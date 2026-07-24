#!/usr/bin/env python3
"""
Compare Eta Formulations: Efficiency vs. Flux

Tests whether reformulating η as flux (rather than efficiency) 
improves correlations with economic and migration dynamics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")
FIGURES_DIR = RESULTS_DIR / "figures"

def main():
    print("="*70)
    print("COMPARING ETA FORMULATIONS")
    print("="*70)
    
    # Load data
    eta_flux = pd.read_csv(RESULTS_DIR / "eta_flux_2006-2022.csv")
    panel = pd.read_csv(RESULTS_DIR / "panel_complete.csv")
    
    # Merge
    merged = eta_flux.merge(
        panel[['cbsa_code', 'year', 'eta', 'sigma_econ', 'sigma_mig', 'population']], 
        on=['cbsa_code', 'year'], 
        how='inner',
        suffixes=('', '_y')
    )
    
    # Rename if needed
    if 'eta_y' in merged.columns:
        merged['eta_efficiency'] = merged['eta_y']
    else:
        merged['eta_efficiency'] = merged['eta']
    
    print(f"\nMerged dataset: {len(merged)} observations")
    
    # Check columns
    print("\nAvailable columns:", merged.columns.tolist())
    
    # Correlation comparison
    print("\n" + "="*70)
    print("CORRELATION COMPARISON")
    print("="*70)
    
    # Old formulation (eta efficiency)
    print("\nOLD: η as Geodesic Efficiency")
    print("-" * 50)
    
    # Use 'eta' from panel_complete (this is eta_efficiency)
    eta_eff_col = 'eta' if 'eta' in merged.columns else 'eta_y'
    
    corr_ee = merged[eta_eff_col].corr(merged['sigma_econ'])
    corr_em = merged[eta_eff_col].corr(merged['sigma_mig'])
    corr_ec = merged[eta_eff_col].corr(merged['eta_flux_approx'])
    print(f"  η vs σ_econ:  {corr_ee:+.4f}")
    print(f"  η vs σ_mig:   {corr_em:+.4f}")
    print(f"  η vs η_flux:  {corr_ec:+.4f} (expected: negative)")
    
    # New formulation (eta flux)
    print("\nNEW: η as Demographic Flux")
    print("-" * 50)
    corr_fe = merged['eta_flux_approx'].corr(merged['sigma_econ'])
    corr_fm = merged['eta_flux_approx'].corr(merged['sigma_mig'])
    corr_fpop = merged['eta_flux_approx'].corr(np.log(merged['population']))
    print(f"  η_flux vs σ_econ:  {corr_fe:+.4f}")
    print(f"  η_flux vs σ_mig:   {corr_fm:+.4f}")
    print(f"  η_flux vs log(pop): {corr_fpop:+.4f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if abs(corr_fe) > abs(corr_ee):
        print(f"✓ η_flux correlates {abs(corr_fe)/abs(corr_ee):.1f}x STRONGER with σ_econ")
    else:
        print(f"✗ η_flux correlates {abs(corr_fe)/abs(corr_ee):.1f}x weaker with σ_econ")
    
    if abs(corr_fm) > abs(corr_em):
        print(f"✓ η_flux correlates {abs(corr_fm)/abs(corr_em):.1f}x STRONGER with σ_mig")
    else:
        print(f"✗ η_flux correlates {abs(corr_fm)/abs(corr_em):.1f}x weaker with σ_mig")
    
    if abs(corr_fpop) < 0.1:
        print(f"✓ η_flux is POPULATION-INVARIANT (r={corr_fpop:.3f})")
    else:
        print(f"⚠ η_flux still correlates with population (r={corr_fpop:.3f})")
    
    # Three-way flux correlation
    print("\n" + "="*70)
    print("THREE-FLUX CORRELATION MATRIX")
    print("="*70)
    
    flux_corr = merged[['eta_flux_approx', 'sigma_econ', 'sigma_mig']].corr()
    print("\n", flux_corr.round(4))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Aligned Formulation)")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'CV':<10}")
    print("-" * 60)
    for col, name in [('eta_flux_approx', 'η_flux'), 
                       ('sigma_econ', 'σ_econ'), 
                       ('sigma_mig', 'σ_mig')]:
        mean = merged[col].mean()
        std = merged[col].std()
        cv = std / mean if mean > 0 else np.nan
        print(f"{name:<20} {mean:<10.4f} {std:<10.4f} {cv:<10.4f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Eta Reformulation: Efficiency vs. Flux', fontsize=14, fontweight='bold')
    
    # Row 1: Old formulation (efficiency)
    # η vs σ_econ
    ax = axes[0, 0]
    ax.scatter(merged['sigma_econ'], merged['eta'], alpha=0.3, s=20)
    ax.set_xlabel('σ_econ')
    ax.set_ylabel('η (efficiency)')
    ax.set_title(f'η vs σ_econ (r={corr_ee:.3f})')
    ax.grid(True, alpha=0.3)
    
    # η vs σ_mig
    ax = axes[0, 1]
    ax.scatter(merged['sigma_mig'], merged['eta'], alpha=0.3, s=20)
    ax.set_xlabel('σ_mig')
    ax.set_ylabel('η (efficiency)')
    ax.set_title(f'η vs σ_mig (r={corr_em:.3f})')
    ax.grid(True, alpha=0.3)
    
    # η vs population
    ax = axes[0, 2]
    ax.scatter(np.log(merged['population']), merged[eta_eff_col], alpha=0.3, s=20)
    ax.set_xlabel('log(Population)')
    ax.set_ylabel('η (efficiency)')
    ax.set_title(f'η vs Size (r={merged[eta_eff_col].corr(np.log(merged["population"])):.3f})')
    ax.grid(True, alpha=0.3)
    
    # Row 2: New formulation (flux)
    # η_flux vs σ_econ
    ax = axes[1, 0]
    ax.scatter(merged['sigma_econ'], merged['eta_flux_approx'], alpha=0.3, s=20, color='orange')
    ax.set_xlabel('σ_econ')
    ax.set_ylabel('η_flux (change)')
    ax.set_title(f'η_flux vs σ_econ (r={corr_fe:.3f})')
    ax.grid(True, alpha=0.3)
    
    # η_flux vs σ_mig
    ax = axes[1, 1]
    ax.scatter(merged['sigma_mig'], merged['eta_flux_approx'], alpha=0.3, s=20, color='orange')
    ax.set_xlabel('σ_mig')
    ax.set_ylabel('η_flux (change)')
    ax.set_title(f'η_flux vs σ_mig (r={corr_fm:.3f})')
    ax.grid(True, alpha=0.3)
    
    # η_flux vs population
    ax = axes[1, 2]
    ax.scatter(np.log(merged['population']), merged['eta_flux_approx'], alpha=0.3, s=20, color='orange')
    ax.set_xlabel('log(Population)')
    ax.set_ylabel('η_flux (change)')
    ax.set_title(f'η_flux vs Size (r={corr_fpop:.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eta_reformulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: eta_reformulation_comparison.png")
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    improvements = 0
    if abs(corr_fe) > abs(corr_ee):
        improvements += 1
    if abs(corr_fm) > abs(corr_em):
        improvements += 1
    if abs(corr_fpop) < 0.1:
        improvements += 1
    
    if improvements >= 2:
        print("✓ RECOMMENDATION: Switch to η_flux formulation")
        print("  - Better alignment with economic/migration dynamics")
        print("  - Population-invariant (reduces statistical artifact)")
        print("  - All three metrics now measure RATES OF CHANGE")
    else:
        print("⚠ RECOMMENDATION: Keep η_efficiency but acknowledge limitations")
        print("  - Measures different concept (coherence vs. change)")
        print("  - Population artifact requires within-size-class analysis")

if __name__ == "__main__":
    main()
