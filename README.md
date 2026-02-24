# DU_GE: Dissipative Urbanism - Geodesic Efficiency

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Core analysis scripts for the paper: **"Dissipative Urbanism: Non-Equilibrium Thermodynamics in American Metropolitan Areas"**

## Overview

This repository contains the essential Python scripts used to analyze 386 US Metropolitan Statistical Areas (MSAs) through the lens of non-equilibrium thermodynamics and information geometry.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `run_full_geodesic_analysis.py` | Main analysis runner - computes geodesic efficiency using full numerical solver |
| `generate_figures_full_solver.py` | Generates all final figures for the paper |
| `geodesic_solver.py` | Core numerical geodesic solver using shooting method |
| `entropy_production.py` | Computes entropy production rates from demographic data |
| `shock_classifier.py` | Classifies MSA shock responses (adiabatic vs non-adiabatic) |
| `null_models.py` | Generates IPF null models for statistical validation |

## Key Findings

- **386 MSAs** analyzed over **18 years** (2006-2024, excluding 2020)
- Mean geodesic efficiency: **η = 0.716** (71.6%)
- **88.6%** of MSAs exhibit geodesic trajectories (p < 0.05)
- **9 MSAs** (8.5%) confirmed as dissipative structures
- Geodesic efficiency strongly predicts recovery time from shocks (p < 0.001)

## Data Requirements

This code expects US Census Bureau data:
- American Community Survey (ACS) 2006-2019, 2021-2024
- Population Estimates Program
- MSA definitions from OMB Bulletin No. 23-01

## Citation

```bibtex
@article{zeng2026dissipative,
  title={Dissipative Urbanism: Non-Equilibrium Thermodynamics in American Metropolitan Areas},
  author={Zeng, Roger W.},
  journal={Computers, Environment and Urban Systems},
  year={2026}
}
```

## License

MIT License
