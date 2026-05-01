# RCI: Learning to Rank for Selected Configuration Interaction

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
![JAX](https://img.shields.io/badge/JAX-0.8+-A8192C?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

**RCI** combines Learning to Rank (LTR) techniques with Selected Configuration Interaction (SCI) to accelerate the selection of important determinants.


# Requirements

- NVIDIA GPU with CUDA 12+
- Miniconda or Anaconda
- GCC with OpenMP support


# Installation

## Step 1 — Create the Conda Environment

This sets up the environment and installs all Python dependencies in one step:

```bash
conda env create -f environment.yml
conda activate rci
```

## Step 2 — Install Local Packages

[SOLAX](https://github.com/pavlobilous/SOLAX) and [clic](https://github.com/bslhrzg/clic) are required for QM calculations on CUDA or CPU. 
These packages are not yet available on PyPI, so a dedicated script is provided:

```bash
bash install_local_pkgs.sh
```

The script clones each repository into `packages/` and builds them with OpenMP + O3 optimizations. It is re-run safe: any package whose source directory already exists will be skipped automatically. To force a reinstall, remove the corresponding directory and re-run.


# Usage

## Data

Precomputed one- and two-electron integrals are provided in `data/`, covering:

- Small molecules in Gaussian basis: N2, C2, H2O, and NH3
- Transition metal complex: Fe2S2 FCIDUMP files (sourced from [Active-space-model-for-Iron-Sulfur-Clusters](https://github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters))
  
The electron integrals in the PAW basis used in this work are publicly available for download via [Zenodo](https://zenodo.org/records/14740476).


## Running Experiments

Experiment scripts are located in `exp_scripts/`.
For example, 

```bash
cd exp_scripts
python gaussian.py
```

Results are saved to `result/`.


# Contact
If you have any questions, please feel free to contact us. We will respond as soon as possible.
- Email: wan.nie@outlook.com
