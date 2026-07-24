# Geodesic Efficiency Analysis - Excluding Puerto Rico MSAs

This script runs the complete geodesic solver analysis excluding the 4 Puerto Rico MSAs due to data quality concerns.

## Puerto Rico MSAs Excluded

| MSA Code | MSA Name | Observations | Reason |
|----------|----------|--------------|--------|
| 11640 | Arecibo, PR | 4 | Limited data |
| 32420 | Mayagüez, PR | 11 | Limited data |
| 38660 | Ponce, PR | 15 | Limited data |
| 41980 | San Juan-Bayamón-Caguas, PR | 16 | Limited data |

**Result**: 381 MSAs analyzed (down from 385)

## Features

1. **Bayesian Smoothing with Jeffreys Prior**
   - Applied to all demographic count data (age, income, race)
   - Non-informative prior: α = 0.5 per category
   - Prevents zero probabilities and stabilizes entropy calculations

2. **Full Numerical Geodesic Solver**
   - Uses shooting method for boundary value problems
   - Computes true geodesic paths on Fisher information manifold
   - More accurate than Fisher-Rao approximation

3. **Null Model Testing**
   - 1000 permutations per MSA for statistical significance
   - Randomized trajectory null model
   - P-values computed for geodesic hypothesis

4. **Dimensional Analysis**
   - Separate geodesic efficiency for each demographic dimension:
     - Age distribution entropy
     - Income inequality (Gini)
     - Racial/ethnic diversity (Shannon index)

## Usage

```bash
cd ~/DissipativeUrbanism
python3 geodesic_efficiency/src/run_geodesic_analysis_no_pr.py
```

## Output Files

All results are saved to `~/DissipativeUrbanism/results/thermodynamics/`:

1. **no_pr_geodesic_results.csv** - Main geodesic efficiency results
   - `msa_code`: MSA identifier
   - `msa_name`: MSA name
   - `geodesic_efficiency`: Overall efficiency η (0-1)
   - `p_value`: Statistical significance
   - `is_geodesic`: True if p < 0.05
   - `geodesic_deviation`: RMS deviation from geodesic
   - `actual_path_length`: Length of actual trajectory
   - `geodesic_distance`: Shortest geodesic distance

2. **no_pr_geodesic_dimensional.csv** - Dimensional breakdown
   - `geodesic_efficiency_age`: Age dimension efficiency
   - `geodesic_efficiency_income`: Income dimension efficiency
   - `geodesic_efficiency_race`: Race dimension efficiency
   - Corresponding p-values and significance flags

3. **no_pr_geodesic_report.txt** - Summary report with statistics

## Expected Runtime

- ~6-7 hours for 381 MSAs with 1000 permutations each
- Average: ~60 seconds per MSA
- Progress displayed every 10 MSAs

## Comparison with Original Analysis

| Metric | Original (385 MSAs) | No PR (381 MSAs) | Expected Impact |
|--------|---------------------|------------------|-----------------|
| Sample size | 385 | 381 | -1% |
| Mean η | TBD | TBD | Minimal change |
| Variance | TBD | TBD | Slightly reduced |
| Data quality | Mixed | Consistent | Improved |

## Methodology

### Geodesic Efficiency Formula

```
η = D_geodesic / D_actual
```

Where:
- `D_geodesic`: Shortest path distance on Fisher manifold
- `D_actual`: Integrated length of actual demographic trajectory

### Statistical Significance Testing

Null hypothesis (H₀): Trajectory is a random walk  
Alternative (H₁): Trajectory follows geodesic path

P-value computed as:
```
p = P(η_null ≥ η_observed)
```

Where `η_null` is the distribution of efficiencies from 1000 randomized trajectories.

### Bayesian Smoothing

For count data `n_i` in category `i`:
```
p_i = (n_i + α) / Σ(n_j + α)
```

With Jeffreys prior: `α = 0.5`

## References

- Amari, S. (2016). Information Geometry and Its Applications
- do Carmo, M. P. (1992). Riemannian Geometry
- Prigogine, I. (1967). Introduction to Thermodynamics of Irreversible Processes
