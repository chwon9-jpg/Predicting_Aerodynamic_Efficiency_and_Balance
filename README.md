# Beyond the Wind Tunnel: Machine Learning for F1 Front Wing Aerodynamics

**Author:** Christopher Won  
**Institution:** Institut Polytechnique de Paris  
**Date:** April 2026

## Overview

This project applies supervised regression to predict two aerodynamically meaningful 
targets from CFD simulation data of a Formula 1 front wing:

- **Aerodynamic Efficiency** (CL/CD) — downforce-to-drag ratio
- **Aero Balance** (CLF/CL) — front-to-total downforce ratio

The dataset consists of 5,000 CFD timesteps with 12 input features (pressure and 
viscous force/moment components across X, Y, Z axes), sourced from Shah (2025).

## Models

| Model | CL/CD MSE | CLF/CL MSE |
|---|---|---|
| PCA Regression (baseline) | 0.020249 | 0.000009 |
| LASSO Regression | 0.005883 | 0.000001 |
| Ridge Regression | 0.006566 | 0.000001 |
| **Random Forest** | **0.000204** | **0.000001** |

Random Forest is the decisive winner for CL/CD (~99× lower MSE than PCA), while 
all regularised methods perform comparably for aero balance.

## Key Findings

- **Random Forest** best captures the non-linear transient dynamics (~first 500 
  timesteps), which linear methods cannot model.
- **LASSO** reduced CL/CD prediction to just 2 of 12 features 
  (`PRESSURE_MOMENT_Y`, `PRESSURE_FORCE_X`) — physically interpretable, 
  as drag directly enters the CL/CD denominator.
- **Ridge** was the most stable linear method across data fractions in the 
  degradation study.
- **Aero balance is a near-linear target** — model choice does not matter for CLF/CL.

## Repository Contents

| File | Description |
|---|---|
| `Predicting_Aerodynamic_Efficiency_and_Balance.ipynb` | Full analysis notebook |
| `f1project_dataset.csv` | CFD dataset (5,000 timesteps, 12 features + targets) |

## Reference

Shah, 2025 — *Physics-Informed Neural Networks for F1 Aerodynamics*  
[https://arxiv.org/abs/2509.01963](https://arxiv.org/abs/2509.01963)
