# Phase 1: Exploratory Data Analysis Summary

## Dataset Overview
- **Original Shape:** (2000, 18)
- **Missing Data:** Found 19 rows with missing CALPHAD outputs (Liquidus, Solidus, density, phases, number_phases). These were successfully dropped.
- **Cleaned Shape:** (1981, 19)
- **Composition Check:** All composition columns sum to 1.0 (±0.001) properly.

## Phase Stability (Class Balance)
The target variable `is_single_bcc` was created based on the `phases` column. 
- **Single BCC (1):** 1808 samples
- **Laves Present (0):** 173 samples
This confirms a severe class imbalance (~10:1) which will require special handling (e.g. SMOTE or balanced class weights) during machine learning model training in Phase 2.

## Key Observations
- **VEC Rule:** The histogram of VEC clearly shows that higher VEC values correlate with Laves phase formation, consistent with the general heuristic that low VEC stabilizes the BCC structure in refractory high entropy alloys.
- **Correlation with Melting Point:** The correlation heatmap shows that Mo, as a high-melting refractory element, positively correlates with Liquidus temperature.
- **Density:** Zr and Ti additions tend to lower the overall alloy density, whereas heavier elements like Mo and Nb increase it.
