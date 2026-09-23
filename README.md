# ML-Driven Design of a Single-Phase Nb-V-Ti-Mo-Zr RHEA

This repository contains the code for a machine learning-based workflow to design optimal refractory high entropy alloys for nuclear applications. The workflow trains surrogate models on CALPHAD data and uses NSGA-II to find the Pareto optimal compositions balancing melting point, density, and phase stability.

## Prerequisites

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

The project is broken into four independent phases. They must be run in order initially to generate the required artifacts for the subsequent steps.

### Phase 1: Data Cleaning & Exploratory Data Analysis
**Command:**
```bash
cd src
python phase1_eda.py
```
**What it does:** Reads `Nb_V_Ti_Mo_Zr_HEA_parameters_CALPHAD.xlsx`, handles missing rows (saving them to a report), engineers the binary target `is_single_bcc`, and generates EDA plots (correlation heatmap, pairplot, class balance, VEC distribution). Outputs are saved to the `data_cleaning/` and `report/` folders.

### Phase 2: Machine Learning Surrogate Models
**Command:**
```bash
cd src
python phase2_ml.py
```
**What it does:** Loads the cleaned dataset. Trains XGBoost and Random Forest models for regression (Liquidus, Solidus, Density) and classification (is_single_bcc, utilizing class weights to handle the severe imbalance). Evaluates performance, generates Parity plots, Confusion matrix, and SHAP interpretability plots. Saves the best models (using Joblib) to the `models/` folder.

### Phase 3: Multi-Objective Optimization (NSGA-II)
**Command:**
```bash
cd src
python phase3_nsga2.py
```
**What it does:** Loads the ML surrogate models and bounds from the training data. Formulates a multi-objective problem using `pymoo` to maximize Liquidus, minimize Density, and maximize the probability of forming a single BCC phase. Extracts the Pareto front and 3 recommended candidate compositions (Highest Melting Point, Lowest Density, Most Balanced) to the `optimization/` folder.

### Phase 4: Verification
**Command:**
```bash
cd src
python phase4_verification.py
```
**What it does:** Generates a clean CSV template at `verification/calphad_input_template.csv` for the recommended candidates. 

**Manual Action Required:**
1. You must run these recommended compositions through your actual CALPHAD software.
2. Fill in the missing values in `calphad_input_template.csv`.
3. Save the filled file as `calphad_results.csv` in the `verification/` folder.
4. Re-run `python phase4_verification.py` to automatically compute the prediction errors and generate the final phase comparison report.

## Directory Structure
- `src/`: Python source code for each phase.
- `data_cleaning/`: Cleaned dataset, missing rows report, and EDA plots.
- `models/`: Trained surrogate models (.joblib), metrics, parity plots, and SHAP plots.
- `optimization/`: NSGA-II convergence plots, Pareto front data, and recommended candidates.
- `verification/`: CALPHAD input templates and final validation comparisons.
- `report/`: Markdown snippets summarizing the findings of each phase.
