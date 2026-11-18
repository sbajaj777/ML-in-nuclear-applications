# ML-Driven Design of a Single-Phase Nb-V-Ti-Mo-Zr Refractory High Entropy Alloy (RHEA)

An end-to-end Machine Learning surrogate modeling and Multi-Objective Evolutionary Optimization (NSGA-II) pipeline to discover optimal, single-phase BCC **Nb-V-Ti-Mo-Zr Refractory High Entropy Alloys** for high-temperature nuclear applications.

---

## 🎯 Project Objective
Identify the optimal alloy composition balancing three competing criteria:
1. **Maximize Liquidus Melting Temperature** ($T_{\text{liq}} > 2350\text{ K}$)
2. **Minimize Alloy Density** ($\rho < 6700\text{ kg/m}^3$)
3. **Maximize Single-Phase BCC Stability** ($P(\text{BCC}) \rightarrow 1.0$, preventing brittle $C15\text{ Laves}$ phase)

---

## 🏆 The Optimized Alloy Composition

From the 200 non-dominated Pareto front candidates evaluated by NSGA-II:

$$\mathbf{Nb_{31.7}V_{14.1}Ti_{34.6}Mo_{9.4}Zr_{10.2}} \quad \text{(atomic \%)}$$

| Property | Value | Engineering Significance |
| :--- | :---: | :--- |
| **Liquidus Temperature ($T_{\text{liq}}$)** | **2,420.2 K** ($2,147^\circ\text{C}$) | Exceeds operating thresholds of Gen-IV nuclear cladding |
| **Solidus Temperature ($T_{\text{sol}}$)** | **2,004.9 K** ($1,732^\circ\text{C}$) | Wide processing window ($T_{\text{liq}} - T_{\text{sol}} = 415.3\text{ K}$) |
| **Alloy Density ($\rho$)** | **6,697.7 kg/m³** ($6.70\text{ g/cm}^3$) | $\sim 20\%$ lighter than traditional nickel superalloys (Inconel 718: $8.19\text{ g/cm}^3$) |
| **Single-Phase BCC Probability** | **99.9%** | Guarantees complete suppression of brittle $C15\text{ Laves}$ intermetallics |

---

## 📌 Technical Pipeline

### Phase 1: Data Engineering & Cleaning
- Unified two CALPHAD datasets (Dataset 1 original + Dataset 2 augmented) totaling 4,000 raw samples.
- Cleaned simulation convergence errors and missing rows $\rightarrow$ **3,927 verified samples** satisfying $\sum x_i = 1.0 \pm 0.001$.
- Identified severe 10:1 class imbalance between single BCC (95.6%) and brittle Laves (4.4%).

### Phase 2: Comprehensive Surrogate Model Benchmarking
- **9 Regression Models:** Simple Linear, Multiple Linear, Polynomial (degree 2), Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, and XGBoost evaluated on MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, and MAPE.
  - *Best Liquidus/Solidus Model:* **Random Forest** ($R^2 = 0.5642$ / $0.6907$, $\text{MAPE} < 1.45\%$)
  - *Best Density Model:* **XGBoost** ($R^2 = 0.2332$, $\text{MAPE} = 1.23\%$)
- **9 Classification Models:** Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, SVM, AdaBoost, Gradient Boosting, and XGBoost evaluated on Accuracy, Precision, Recall, F1, Specificity, Error Rate, and AUC-ROC.
  - *Best Phase Stability Model:* **Balanced Random Forest** ($\text{AUC} = 0.7874$, **$82.86\%$ Specificity** catching brittle Laves failures).
- **Hyperparameter Optimization:** 5-Fold Stratified Cross-Validation using `GridSearchCV` and `RandomizedSearchCV`.

### Phase 3: Physics Attribution via SHAP
- **Molybdenum (Mo):** Dominant driver elevating melting temperature ($+150\text{ K} / 0.05\text{ at.}$).
- **Zirconium (Zr) & Titanium (Ti):** Strongest levers reducing alloy density.
- **Valence Electron Concentration (VEC):** Compositions exceeding $\text{VEC} > 4.80$ trigger Laves precipitation.

### Phase 4: Multi-Objective Evolutionary Optimization (NSGA-II)
- Formulation: $\min [-T_{\text{liq}}(\mathbf{x}),\ \rho(\mathbf{x}),\ -P_{\text{BCC}}(\mathbf{x})]$ subject to $\sum x_i = 1.0$ within training boundaries.
- 300 generations with custom `SimplexBoundsRepair` operator to enforce stoichiometric closure.
- Extracted 200 Pareto-optimal solutions and identified the optimal candidate formulation.

---

## 📁 Repository Structure
```
├── data/
│   ├── raw/                       # Original & augmented CALPHAD datasets
│   └── processed/                 # Cleaned dataset (3,927 samples)
├── models/                        # Saved production models (.joblib)
├── models_comprehensive/          # Benchmark CSV tables, CV logs & comparison plots
│   └── plots/                     # Parity grids, ROC curves, confusion matrices
├── optimization/                  # Pareto front data & recommended compositions
├── report/
│   └── BTP_Final_Presentation.pptx # 19-slide presentation
├── src/
│   ├── phase1_eda.py              # Data cleaning & EDA
│   ├── phase2_ml_comprehensive.py # Full 18-model training & tuning
│   ├── phase3_nsga2.py            # NSGA-II optimization
│   └── generate_ppt.py            # Automated presentation builder
└── README.md
```

---

## 🏃 Reproduction
```bash
# Train all models and produce comparative metrics
python src/phase2_ml_comprehensive.py

# Run NSGA-II optimization
python src/phase3_nsga2.py

# Re-build presentation
python src/generate_ppt.py
```
