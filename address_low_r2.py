#!/usr/bin/env python3
"""
Address the Low R² Problem Honestly.

R² = 0.139 (13.9%) is weak explanatory power. This script:
1. Acknowledges this limitation
2. Reinterprets what low R² actually means theoretically
3. Pivots contribution from "prediction" to "structural insight"
4. Identifies what the OTHER 86% of variance represents

Usage:
    python address_low_r2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("/home/roger/DissipativeUrbanism/geodesic_efficiency/results")


def load_data():
    """Load data."""
    df = pd.read_csv(RESULTS_DIR / "master_2006-2022.csv")
    complete = df[
        df['eta_overall'].notna() & 
        df['sigma_econ'].notna() & 
        df['sigma_mig'].notna()
    ].copy()
    return complete


def analyze_explanatory_power(df):
    """Analyze what explains the variance in economic volatility."""
    
    print("="*70)
    print("THE LOW R² PROBLEM: HONEST ASSESSMENT")
    print("="*70)
    
    print("""
THE BRUTAL TRUTH:
━━━━━━━━━━━━━━━
• η alone:      R² = 0.9%  (explains 0.9% of economic volatility)
• η + σ_mig:    R² = 13.9% (explains 13.9% of economic volatility)
• Interaction:  R² = 13.9% → 14.4% (trivial improvement)

REVIEWER OBJECTION: "This explains nothing. Why should we care?"

ANSWER: The LOW explanatory power IS the finding.
""")
    
    print("="*70)
    print("WHAT THE OTHER 86% REPRESENTS")
    print("="*70)
    
    print("""
If η and σ_mig explain only 14% of economic volatility,
the OTHER 86% comes from:

┌────────────────────────────────────────────────────────────────────┐
│ FACTOR                          │  LIKELY IMPORTANCE              │
├────────────────────────────────────────────────────────────────────┤
│ Industry composition            │  HIGH - Oil towns, manufacturing│
│                                 │        recessions, tech cycles  │
├────────────────────────────────────────────────────────────────────┤
│ National/regional shocks        │  HIGH - 2008 crisis, COVID      │
│                                 │        affect all cities        │
├────────────────────────────────────────────────────────────────────┤
│ Policy & governance             │  MEDIUM - Tax incentives,       │
│                                 │           business climate      │
├────────────────────────────────────────────────────────────────────┤
│ Geographic constraints          │  MEDIUM - Port access,          │
│                                 │           natural resources     │
├────────────────────────────────────────────────────────────────────┤
│ Network effects                 │  MEDIUM - Agglomeration,        │
│                                 │           connectivity          │
├────────────────────────────────────────────────────────────────────┤
│ Random/idiosyncratic            │  LOW-MEDIUM - City-specific     │
│                                 │               events            │
└────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Economic volatility is DOMINATED by exogenous factors
             (industry, national shocks) NOT endogenous demographic
             structure.
""")
    
    # Empirical evidence: Look at highest volatility MSAs
    print("\n" + "-"*70)
    print("EMPIRICAL EVIDENCE: What drives high σ_econ?")
    print("-"*70)
    
    high_vol = df.nlargest(10, 'sigma_econ')[['cbsa_title', 'sigma_econ', 'eta_overall', 'sigma_mig']]
    
    print("\nTop 10 most economically volatile MSAs:")
    for i, (_, row) in enumerate(high_vol.iterrows(), 1):
        city = row['cbsa_title'][:35]
        print(f"  {i}. {city:<35} σ_econ={row['sigma_econ']:.4f}")
    
    print("""
PATTERN: Top volatile MSAs are:
  • Oil towns (Midland TX, Odessa TX) - commodity price cycles
  • Manufacturing centers (Elkhart IN, Kokomo IN) - sectoral shocks
  • Tourism-dependent (Myrtle Beach, The Villages) - seasonal/boom-bust
  • Port cities (Lake Charles LA) - trade volatility

These are INDUSTRY-DRIVEN, not demographically-driven!
""")
    
    return high_vol


def reinterpret_contribution(df):
    """Pivot from 'prediction' to 'structural insight'."""
    
    print("\n" + "="*70)
    print("REFRAMING THE CONTRIBUTION")
    print("="*70)
    
    print("""
OLD FRAMING (WRONG):
  "We can predict economic volatility from demographics"
  → Fails (R² = 14% is not prediction)

NEW FRAMING (CORRECT):
  "Demographic structure has WEAK but STATISTICALLY SIGNIFICANT 
   effect on economic volatility, controlling for migration"
  
  The 14% that IS explained reveals STRUCTURAL coupling between
  population composition and economic dynamics.
  
  The 86% that ISN'T explained shows economic volatility is
  dominated by EXOGENOUS factors (industry, national shocks).
""")
    
    print("\n" + "-"*70)
    print("WHAT WE CAN CLAIM (Honestly)")
    print("-"*70)
    
    print("""
1. PARTIAL EFFECT (Not Prediction)
   "After controlling for migration, demographic coherence has a
    marginally significant negative effect on economic volatility
    (β = -0.10, p = 0.055), explaining approximately 1% of 
    incremental variance."
    
   This is a STRUCTURAL finding, not a predictive model.

2. THEORETICAL BOUNDARY
   The low R² establishes a THEORETICAL BOUNDARY:
   "Demographic structure constrains but does not determine 
    economic volatility."
   
   This is useful! It tells us what NOT to focus on.

3. INTERACTION EFFECT (The Real Finding)
   The interaction (η × σ_mig) is significant (p = 0.001) even
   though main effects are weak.
   
   This reveals COUPLING between dimensions, not prediction.
   The coupling exists even if the individual effects are small.
""")
    
    print("\n" + "-"*70)
    print("COMPARISON: What R² is 'normal' in urban studies?")
    print("-"*70)
    
    print("""
Urban scaling literature (Bettencourt et al.):
  • Population → GDP:         R² ≈ 0.80-0.95 (very high)
  • Population → patents:     R² ≈ 0.70-0.90 (high)
  • Population → infrastructure: R² ≈ 0.60-0.80 (moderate)

Our results:
  • η → σ_econ:               R² = 0.009 (terrible)
  • η + σ_mig → σ_econ:       R² = 0.139 (poor)

WHY THE DIFFERENCE?
  • Bettencourt predicts LEVELS (GDP, patents) from SIZE (population)
    → Size is the dominant determinant of aggregate outputs
    
  • We predict VOLATILITY (σ_econ) from STRUCTURE (η)
    → Volatility is driven by shocks, not structure
    → Structure constrains response to shocks, but shocks dominate
""")


def alternative_framing(df):
    """Suggest alternative ways to frame this work given low R²."""
    
    print("\n" + "="*70)
    print("ALTERNATIVE FRAMINGS (Given Low R²)")
    print("="*70)
    
    print("""
FRAMING 1: "Boundary Conditions, Not Determinants"
────────────────────────────────────────────────────
"Demographic structure sets BOUNDARY CONDITIONS for economic 
 volatility but does not determine it."

• High η: Narrows the range of possible σ_econ (constrains extremes)
• Low η: Allows wider volatility range
• Actual σ_econ determined by industry, shocks, policy

This is like: "Height constrains basketball performance but 
doesn't determine it (skill matters more)."


FRAMING 2: "Residual Analysis"  
────────────────────────────────────────
"After removing industry and shock effects, residual economic 
 volatility shows weak but significant correlation with 
 demographic coherence."

• First: Regress σ_econ on industry dummies, time fixed effects
• Then: Analyze residuals vs η
• This might INCREASE the η coefficient (removing noise)

This reframes as: "Among cities with similar industry structure..."


FRAMING 3: "Mechanism, Not Correlation"
────────────────────────────────────────
Don't claim to predict σ_econ. Instead:

"We identify a MECHANISM: Migration weakens the protective effect 
 of demographic coherence on economic stability."

• The mechanism exists even if overall prediction is poor
• It's about HOW dimensions interact, not HOW MUCH variance explained


FRAMING 4: "Typology from Anomalies"
────────────────────────────────────────
"Cities cluster into distinct regimes where different factors 
 dominate economic dynamics."

• Classify cities by industry, migration regime, size
• Show that η matters in SOME regimes but not others
• Make the heterogeneity THE finding
""")


def suggest_improvements(df):
    """Suggest ways to improve explanatory power."""
    
    print("\n" + "="*70)
    print("HOW TO IMPROVE EXPLANATORY POWER")
    print("="*70)
    
    print("""
OPTION 1: Control for Industry Composition
──────────────────────────────────────────
Add industry shares (from County Business Patterns):
  • % employment in manufacturing
  • % in finance/tech
  • % in oil/gas/extraction
  • % in tourism/hospitality

Expected improvement: R² might increase to 30-50%
Then test if η adds explanatory power BEYOND industry


OPTION 2: First-Differences (Remove Fixed Effects)
──────────────────────────────────────────────────
Instead of levels, analyze CHANGES:
  • Δσ_econ = f(Δη, Δσ_mig)
  • This removes time-invariant city characteristics

Expected: Might reveal stronger short-run coupling


OPTION 3: Non-Linear Models
────────────────────────────
Current: Linear in log-log space
Try: Threshold effects, spline models

Hypothesis: η matters only BELOW some threshold
  • High migration: η irrelevant
  • Low migration: η constrains volatility

Expected: Better fit if non-linearities exist


OPTION 4: Alternative σ_econ Measure
─────────────────────────────────────
Current: Mean absolute log-change (volatile)
Try: 
  • Detrended volatility (remove growth trends)
  • Cyclical component only
  • Recession sensitivity

Expected: Might better isolate endogenous vs exogenous volatility
""")


def bottom_line():
    """Honest bottom line."""
    
    print("\n" + "="*70)
    print("BOTTOM LINE: WHAT WE CAN HONESTLY CLAIM")
    print("="*70)
    
    print("""
WEAK CLAIMS (What we found):
  ✓ Demographic coherence (η) and economic volatility (σ_econ) are
    weakly negatively related (r = -0.10)
  ✓ This relationship is MODERATED by migration (interaction p<0.01)
  ✓ Together η + σ_mig explain 14% of σ_econ variance
  ✓ Structural stability (η) ≠ dynamic stability (r = +0.03)

HONEST INTERPRETATION:
  • Urban economic volatility is DOMINATED by exogenous factors
    (industry, national shocks, policy)
  • Demographic structure has weak but detectable constraint effect
  • The three dimensions (η, σ_econ, σ_mig) are partially coupled

WHAT WE CANNOT CLAIM:
  ✗ "Demographics predict economic volatility" (R² too low)
  ✗ "Cities follow universal scaling laws" (regime-dependent)
  ✗ "We have a predictive model" (14% is not predictive)

REAL CONTRIBUTION:
  We establish THEORETICAL BOUNDARIES:
    • Demographic structure matters, but weakly
    • Migration mediates the demographic-economic link
    • Structural and dynamic stability are orthogonal
    • "Coherence" requires multi-dimensional measurement
  
  This is STRUCTURAL insight, not predictive power.
""")
    
    print("\n" + "="*70)
    print("FINAL ASSESSMENT: Is this publishable?")
    print("="*70)
    
    print("""
VERDICT: Marginal without improvements

STRENGTHS:
  • Novel application of Fisher-Rao metric to urban demographics
  • Theoretical contribution (structural vs dynamic stability)
  • Challenge to Bettencourt's universal scaling
  • Identifies regime-dependence

WEAKNESSES:
  • Low explanatory power (R² = 14%)
  • Cannot claim prediction
  • Small effect sizes
  • Common-sense findings (migration disrupts)

PATH TO PUBLICATION:
  1. Control for industry composition (could raise R² to 30-50%)
  2. Then test if η adds explanatory power beyond industry
  3. Focus on the INTERACTION effect (more robust than main effects)
  4. Frame as "boundary conditions" not "determinants"
  5. Emphasize the regime-typology as the main contribution

WITHOUT IMPROVEMENTS:
  → Suitable for: Methods paper, exploratory analysis, conference
  → NOT suitable for: Top urban journal (Nature Cities, etc.)
""")


def main():
    df = load_data()
    
    analyze_explanatory_power(df)
    reinterpret_contribution(df)
    alternative_framing(df)
    suggest_improvements(df)
    bottom_line()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


def main():
    df = load_data()
    
    analyze_explanatory_power(df)
    reinterpret_contribution(df)
    alternative_framing(df)
    suggest_improvements(df)
    bottom_line()


if __name__ == "__main__":
    main()
