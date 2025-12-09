# time-grain
v1
# Temporal grain as a control parameter for viable recurrent computation

This repository provides minimal, reproducible simulation and analysis code for the paper  
**“Temporal grain as a control parameter for viable recurrent computation.”**

## Overview
The code implements a continuous-time recurrent neural-mass model with delayed coupling and an explicit energetic cost function.  
Temporal grain (G = ΔT / Tint) is varied to examine how time discretization affects dynamical dimensionality, temporal integration, and metabolic viability.

## Files
- **time_grain_model.py** — Main simulation and analysis script  
  - Runs the model for three temporal-grain conditions (G = 0.01, 0.30, 1.00)  
  - Computes:
    - Temporal integration *(I_T)*
    - Effective dimensionality *(D)*
    - Mean metabolic load *(⟨P_tot⟩)*

## Requirements
- Python ≥ 3.11  
- NumPy ≥ 1.26  
- SciPy ≥ 1.12

Install with:
```bash
pip install numpy scipy
Usage

Run simulations and compute all markers:
python time_grain_model.py
Example output:
G=0.01: I_T=0.020, D=0.790, P_tot=0.570
G=0.30: I_T=-0.994, D=0.016, P_tot=144.900
G=1.00: I_T=-0.993, D=0.016, P_tot=140.400
