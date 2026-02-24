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

## Reproduction Protocol

### Step 1: Environment Setup

```bash
# Clone or navigate to the repository
cd /path/to/geodesic_efficiency

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt

# Install cartopy for map generation (requires system dependencies)
# Ubuntu/Debian:
# sudo apt-get install libgeos-dev libproj-dev
# pip install cartopy

# macOS:
# brew install geos proj
# pip install cartopy
```

### Step 2: Data Structure

The analysis expects data in your home directory (`~` or `$HOME`) under the following structure:

```
~/DissipativeUrbanism/
├── results/
│   ├── data/
│   │   ├── msa_demographics_raw_annual.csv    # Raw MSA demographic data (2006-2024)
│   │   └── msa_data_with_coords.csv           # MSA data with lat/lon coordinates
│   ├── geodesic_solver_full/
│   │   └── msa_geodesic_analysis_full.csv     # Geodesic analysis output
│   └── thermodynamics/
│       ├── official_msa_entropy_production.csv
│       ├── official_msa_classification.csv
│       ├── official_msa_covid_impact.csv
│       └── official_msa_lyapunov_analysis.csv
└── geodesic_efficiency/
    └── figures/                                 # Generated figures output
```

**Note:** Raw Census/ACS data is not included in this repository due to size constraints. Contact the authors for data access or use your own demographic data following the format specifications in `entropy_production.py`.

### Step 3: Core Analysis Workflow

#### Option A: Full Reproduction (Recommended)

Run the complete analysis pipeline:

```bash
# 1. Compute entropy production for all MSAs
python src/entropy_production.py

# 2. Run geodesic analysis (full numerical solver)
# This is computationally intensive (~2-4 hours for 386 MSAs)
python src/run_full_geodesic_analysis.py

# 3. Generate all publication figures
python src/generate_figures_full_solver.py
```

#### Option B: Quick Test (Subset of MSAs)

For testing with a smaller subset:

```bash
# Use the no-PR variant for faster testing
python src/run_geodesic_analysis_no_pr.py
```

### Step 4: Generate Graphic Abstract Map

```bash
# Generate the four-quadrant US map for graphic abstract
python src/generate_graphic_abstract_map.py

# Outputs:
#   - figures/graphic_abstract.pdf (main version)
#   - figures/graphic_abstract_clean.pdf (publication version)
#   - figures/graphic_abstract.png (PNG version)
```

### Expected Outputs

| Output File | Description | Size |
|-------------|-------------|------|
| `figures/Distribution_of_entropy_production.pdf` | Entropy production distribution | ~40 KB |
| `figures/figure_3a_conceptual_quadrant.pdf` | Conceptual quadrant framework (no data) | ~40 KB |
| `figures/figure_3b_quadrant_data.pdf` | Data quadrant scatter plot (I, II, III, IV) | ~50 KB |
| `figures/figure_4_recovery_boxplots.pdf` | Recovery time by η quartile | ~40 KB |
| `figures/figure_5_complexity_entropy.pdf` | Complexity vs efficiency | ~40 KB |
| `figures/geodesic_efficiency_distribution.pdf` | Geodesic efficiency distribution | ~40 KB |
| `figures/graphic_abstract.pdf` | US map with quadrant coloring (main) | ~800 KB |
| `figures/graphic_abstract_clean.pdf` | Clean version for publication | ~800 KB |
| `figures/graphic_abstract.png` | PNG version of graphic abstract | ~800 KB |

### Computational Requirements

- **RAM**: 8 GB minimum, 16 GB recommended
- **CPU**: Multi-core recommended (parallel processing used where possible)
- **Time**: 
  - Entropy production: ~5 minutes
  - Geodesic analysis: ~2-4 hours (full solver)
  - Figure generation: ~2 minutes

### Verification Checklist

After reproduction, verify these key statistics match the paper:

- [ ] 386 MSAs analyzed (377 in continental US)
- [ ] Mean geodesic efficiency η ≈ 0.716
- [ ] 88.6% of MSAs on geodesic trajectories (p < 0.05)
- [ ] Quadrant distribution: I:21%, II:40%, III:0.5%, IV:38%
- [ ] Geographic clustering: Southeast/West = Dissipative, Rust Belt = Stagnant

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cartopy'`
- **Solution**: Cartopy requires system GEOS/PROJ libraries. See Step 1 for OS-specific instructions.

**Issue**: `FileNotFoundError` for data files
- **Solution**: Ensure data files are placed in `results/data/` and `results/thermodynamics/` directories.

**Issue**: Geodesic solver running too slowly
- **Solution**: Reduce `n_msas` parameter in the script for testing, or use `run_geodesic_analysis_no_pr.py`.

**Issue**: Map figures showing incorrect projections
- **Solution**: Cartopy downloads Natural Earth data on first run. Ensure internet connection or download manually.

## License

MIT License
