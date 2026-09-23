# Phase 2: ML Surrogate Models

## Regression Models
XGBoost and Random Forest were trained to predict Liquidus, Solidus, and density.
Parity plots show strong agreement between predictions and CALPHAD ground truth.
Refer to `models/regression_metrics.csv` for exact R2 and RMSE values.

## Classification Model (Phase Stability)
Because of the heavy class imbalance (few Laves-forming examples), class weights were employed.
The primary metric optimized was the recall of the minority class (0 = Laves) to ensure the optimization phase avoids Laves compositions reliably.
Refer to `models/confusion_matrix.png` and `models/classification_report.txt` for details.

## Feature Importance (SHAP)
- **Liquidus:** Mo is consistently the strongest driver for increasing Liquidus temperatures.
- **Density:** Zr additions significantly decrease density, whereas Mo and Nb increase it.
- **Phase Stability:** Interpreting the SHAP plots for `is_single_bcc` reveals complex interactions where excessive amounts of certain transition metals trigger intermetallic formation.
