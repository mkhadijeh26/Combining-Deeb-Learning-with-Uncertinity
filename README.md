# Combining-Deeb-Learning-with-Uncertinity
Hybrid deep neural network–Gaussian process framework for predicting rheological properties (G* and δ) of asphalt binders and mastics, with uncertainty quantification.

# Hybrid DNN–GPR Framework for Asphalt Rheology

This repository contains the source code and workflows associated with the paper:

**Khadijeh, M., Kasbergen, C., Erkens, S., Varveri, A. (2025).  
Combining deep neural networks and Gaussian processes for asphalt rheological insights.  
Results in Engineering, 26, 105629.**  
https://doi.org/10.1016/j.rineng.2025.105629

## Overview

Understanding the rheological behavior of asphalt binders and mastics is essential for designing durable pavements. This project presents a **hybrid machine learning framework** that combines:

- **Deep Neural Networks (DNN)** for learning complex, nonlinear relationships between material properties and test conditions  
- **Gaussian Process Regression (GPR)** for refining predictions and providing **uncertainty quantification**

The framework is applied at two hierarchical scales:
- **Asphalt binder scale** (limited experimental data)
- **Asphalt mastic scale** (large dataset generated using FEM simulations)

The hybrid approach significantly improves prediction accuracy compared to standalone DNN models, particularly when data are scarce.

---

## Key Features

- Prediction of **complex shear modulus (G\*)** and **phase angle (δ)**
- Hybrid **DNN–GPR** architecture for accuracy and reliability
- **Uncertainty estimation** using Gaussian processes
- Multiscale modeling: binder → mastic
- Feature importance analysis using:
  - Random Forest
  - ANOVA
  - Chi-squared test
  - SHAP (SHapley Additive exPlanations)
- FEM-generated synthetic mastic dataset (ABAQUS-based)

---

## Model Architecture

1. Five independent DNN models are trained with identical architectures but different random initializations.
2. Each DNN predicts rheological properties independently.
3. The five DNN predictions are combined and used as inputs to a GPR model.
4. The GPR refines predictions and quantifies uncertainty.

For large mastic datasets, a **Sparse Gaussian Process (SGP)** is used to reduce computational cost.

---

## Input Parameters

### Asphalt Binder Model (11 inputs)
- Aging condition (Fresh, RTFOT, PAV)
- Penetration
- Softening point
- SARA fractions:
  - Saturates
  - Aromatics
  - Resins
  - Asphaltenes
- DSR test parameters:
  - Temperature
  - Frequency

### Asphalt Mastic Model (13 inputs)
- All binder inputs
- Filler content (%)
- Filler stiffness (GPa)

---


