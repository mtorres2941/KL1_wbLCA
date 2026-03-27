# KL1_wbLCA: Probabilistic Whole-Building LCA Using Kernel Density Estimation

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.buildenv.2025.113442-blue)](https://doi.org/10.1016/j.buildenv.2025.113442)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code and data supporting the publication:

> Torres, M. (2025). Characterizing Statistical Uncertainty and Variability of Building Material Emissions in Probabilistic Whole-Building Life Cycle Assessment Using Kernel Density Estimation. *Building and Environment*. https://doi.org/10.1016/j.buildenv.2025.113442

## Overview

This repository implements a probabilistic whole-building life cycle assessment (wbLCA) methodology that uses Kernel Density Estimation (KDE) to characterize the statistical uncertainty and variability of building material embodied carbon emissions. Environmental Product Declaration (EPD) data are used to construct non-parametric KDE distributions for material embodied carbon coefficients (ECCs), which are then propagated through a building LCA to produce probabilistic embodied carbon estimates.

## Repository Structure

```
KL1_wbLCA/
├── notebooks/
│   ├── KL1_MainAnalysis.ipynb      # Main analysis notebook (all paper figures)
│   └── SM3_PullEPDs.ipynb          # Supplementary: EPD data extraction from EC3
├── data/
│   ├── kde_dict.json               # Pre-computed KDE functions by material
│   └── SM4_ECCs.json               # Embodied carbon coefficient distributions
├── figures/                        # All output figures (paper + supplementary)
│   ├── Fig1_KDEFuncs.png
│   ├── Fig2_EPDCounts.png
│   ├── Fig3_LCI.png
│   ├── Fig5_wbECI.png
│   ├── Fig6_SensitivityAnalysis.png
│   ├── Fig7_VarianceReduction.png
│   ├── Fig8_ECReduction.png
│   ├── SM1_AllBandwidths.png
│   ├── SM5_wbECI.png
│   └── SM6_SensitivityAnalysis.png
├── src/                            # Helper modules
│   ├── funcs_kde.py                # KDE construction and evaluation utilities
│   ├── funcs_concrete_emissions.py # Concrete embodied carbon functions
│   ├── funcs_log_tools.py          # Lognormal distribution tools
│   └── funcs_unit_conversion.py    # Unit conversion utilities
├── environment.yml
├── CITATION.cff
└── LICENSE
```

## Getting Started

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate kl1-wblca
```

### 2. Add `src/` to your path

The notebooks import helper modules from `src/`. Add the project root to your Python path before launching Jupyter:

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
jupyter notebook
```

Or add this to the top of any notebook cell:

```python
import sys
sys.path.insert(0, '../src')
```

### 3. Run the notebooks

Open `notebooks/KL1_MainAnalysis.ipynb` for the full analysis. `SM3_PullEPDs.ipynb` documents how EPD data were retrieved from the EC3 database.

## Data

- **`kde_dict.json`** — KDE functions fitted to EPD GWP data, organized by material category.
- **`SM4_ECCs.json`** — Embodied carbon coefficient distributions used in the LCA.

EPD data were sourced from the [EC3 (Embodied Carbon in Construction Calculator)](https://buildingtransparency.org/ec3) database.

## Citation

If you use this code or data, please cite:

```bibtex
@article{torres2025kde,
  title   = {Characterizing Statistical Uncertainty and Variability of Building Material
             Emissions in Probabilistic Whole-Building Life Cycle Assessment Using
             Kernel Density Estimation},
  author  = {Torres, Martin},
  journal = {Building and Environment},
  year    = {2025},
  doi     = {10.1016/j.buildenv.2025.113442},
  url     = {https://doi.org/10.1016/j.buildenv.2025.113442}
}
```

## License

MIT — see [LICENSE](LICENSE).
