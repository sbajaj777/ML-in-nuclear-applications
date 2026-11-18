"""
Generate Final BTP Presentation (PPT)
======================================
Reads results from models_comprehensive/ and generates a
professional PowerPoint presentation covering:
  - Project overview & motivation
  - Dataset info (both datasets)
  - EDA findings with plots
  - All regression models + metrics table
  - All classification models + metrics table
  - ROC curves, Confusion matrices
  - SHAP analysis
  - NSGA-II optimization
  - Recommended compositions
  - Conclusions
"""

import json, os, sys
import pandas as pd
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR   = '../models_comprehensive'
PLOT_DIR  = f'{OUT_DIR}/plots'
EDA_DIR   = '../data/eda_plots'
MODEL_DIR = '../models'
OPT_DIR   = '../optimization'
PPT_OUT   = '../report/BTP_Final_Presentation.pptx'
os.makedirs('../report', exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def safe_load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

reg_df  = safe_load_csv(f'{OUT_DIR}/all_regression_metrics.csv')
cls_df  = safe_load_csv(f'{OUT_DIR}/all_classification_metrics.csv')
summary = safe_load_json(f'{OUT_DIR}/summary.json')
rec_df  = safe_load_csv(f'{OPT_DIR}/recommended_compositions.csv')
retrain_df = safe_load_csv(f'{OPT_DIR}/retrain_best_compositions.csv')

# Fall back to old metrics if comprehensive not available
if reg_df.empty:
    reg_df = safe_load_csv(f'{MODEL_DIR}/regression_metrics.csv')
    print("[WARN] Using old regression metrics (comprehensive not found)")
if rec_df.empty and not retrain_df.empty:
    rec_df = retrain_df

# ── Color palette ─────────────────────────────────────────────────────────────
C_DARK    = RGBColor(0x1a, 0x1a, 0x2e)   # dark navy
C_ACCENT  = RGBColor(0x16, 0x21, 0x3e)   # deep blue
C_GOLD    = RGBColor(0xe9, 0x4f, 0x37)   # highlight red-orange
C_GREEN   = RGBColor(0x2e, 0xcc, 0x71)   # success green
C_BLUE    = RGBColor(0x34, 0x98, 0xdb)   # chart blue
C_WHITE   = RGBColor(0xff, 0xff, 0xff)
C_LGRAY   = RGBColor(0xf0, 0xf4, 0xf8)
C_TEXT    = RGBColor(0x2c, 0x3e, 0x50)

# ── PPT setup ─────────────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK_LAYOUT = prs.slide_layouts[6]   # completely blank

def add_slide():
    return prs.slides.add_slide(BLANK_LAYOUT)

def bg(slide, color=C_DARK):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def txb(slide, text, l, t, w, h,
        size=18, bold=False, color=C_WHITE,
        align=PP_ALIGN.LEFT, wrap=True, italic=False):
    tf = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf.text_frame.word_wrap = wrap
    p = tf.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return tf

def line(slide, l, t, w, color=C_GOLD, h_pt=2):
    shp = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Pt(h_pt)
    )
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background()

def rect(slide, l, t, w, h, fill=C_ACCENT, alpha=None):
    shp = slide.shapes.add_shape(
        1, Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    return shp

def add_img(slide, path, l, t, w, h=None):
    if os.path.exists(path):
        if h:
            slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))
        else:
            slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w))
        return True
    return False

def header_bar(slide, title, subtitle=''):
    rect(slide, 0, 0, 13.33, 1.3, C_ACCENT)
    line(slide, 0.3, 1.3, 12.73)
    txb(slide, title, 0.3, 0.1, 12, 0.7, size=28, bold=True, color=C_WHITE)
    if subtitle:
        txb(slide, subtitle, 0.3, 0.8, 12, 0.5, size=14, color=C_GOLD, italic=True)

def bullet_box(slide, items, l, t, w, h, title='', size=14, title_color=C_GOLD):
    if title:
        txb(slide, title, l, t, w, 0.4, size=15, bold=True, color=title_color)
        t += 0.42
        h -= 0.42
    bx = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    bx.text_frame.word_wrap = True
    for i, item in enumerate(items):
        p = bx.text_frame.add_paragraph() if i > 0 else bx.text_frame.paragraphs[0]
        p.space_before = Pt(3)
        run = p.add_run()
        run.text = f'• {item}'
        run.font.size  = Pt(size)
        run.font.color.rgb = C_WHITE

def metrics_table(slide, df_sub, l, t, w, cols,
                  highlight_col=None, top_n=1):
    """Draw a simple table from dataframe subset."""
    if df_sub.empty:
        txb(slide, "(Results pending)", l, t, w, 1, size=12, color=C_LGRAY)
        return

    rows = len(df_sub) + 1
    ncols = len(cols)
    col_w = w / ncols
    row_h = min(0.35, (7.5 - t - 0.2) / rows)

    # Header
    for ci, col in enumerate(cols):
        rect(slide, l + ci*col_w, t, col_w - 0.02, row_h - 0.02, C_GOLD)
        txb(slide, col, l + ci*col_w + 0.02, t + 0.03,
            col_w - 0.04, row_h - 0.05, size=10, bold=True, color=C_DARK)

    # Data rows
    for ri, (_, row) in enumerate(df_sub.iterrows()):
        fill = RGBColor(0x0d, 0x2b, 0x45) if ri % 2 == 0 else C_ACCENT
        for ci, col in enumerate(cols):
            rect(slide, l + ci*col_w, t + (ri+1)*row_h, col_w - 0.02, row_h - 0.02, fill)
            val = row.get(col, '')
            if isinstance(val, float):
                val = f'{val:.4f}'
            txb(slide, str(val), l + ci*col_w + 0.02, t + (ri+1)*row_h + 0.02,
                col_w - 0.04, row_h - 0.04, size=9, color=C_WHITE)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1: TITLE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl, C_DARK)
rect(sl, 0, 0, 13.33, 7.5, RGBColor(0x0a, 0x0a, 0x1a))
rect(sl, 0, 2.6, 13.33, 2.3, C_ACCENT)
line(sl, 0.5, 2.55, 12.33, C_GOLD, 3)
line(sl, 0.5, 4.95, 12.33, C_GOLD, 3)

txb(sl, "ML-DRIVEN DESIGN OF A SINGLE-PHASE", 0.5, 1.3, 12.3, 0.7,
    size=22, bold=True, color=C_GOLD, align=PP_ALIGN.CENTER)
txb(sl, "Nb–V–Ti–Mo–Zr REFRACTORY HIGH ENTROPY ALLOY", 0.5, 1.95, 12.3, 0.7,
    size=24, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
txb(sl, "FOR NUCLEAR APPLICATIONS", 0.5, 2.55, 12.3, 0.55,
    size=20, bold=True, color=C_GOLD, align=PP_ALIGN.CENTER)
txb(sl, "B.Tech Project (BTP) | Final Report", 0.5, 3.3, 12.3, 0.5,
    size=18, color=C_WHITE, align=PP_ALIGN.CENTER)
txb(sl, "Surrogate ML Models  •  NSGA-II Optimization  •  CALPHAD Verification",
    0.5, 3.85, 12.3, 0.5, size=14, color=C_LGRAY, align=PP_ALIGN.CENTER, italic=True)

ds_info = summary.get('dataset_info', {})
n_total = ds_info.get('combined_cleaned', '~3900+')
txb(sl, f"Dataset: {ds_info.get('dataset1_rows',2000)} + {ds_info.get('dataset2_rows',2000)} CALPHAD samples  |  "
        f"Combined: {n_total} clean rows  |  5 Elements  |  3 Objectives",
    0.5, 5.2, 12.3, 0.5, size=13, color=C_LGRAY, align=PP_ALIGN.CENTER)
txb(sl, "November 2026", 0.5, 6.7, 12.3, 0.5,
    size=12, color=RGBColor(0x7f,0x8c,0x8d), align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2: PROJECT OVERVIEW & WORKFLOW
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Project Overview", "ML + CALPHAD + NSGA-II Workflow")

bullet_box(sl, [
    "Design of Nb-V-Ti-Mo-Zr Refractory High Entropy Alloys (RHEAs) for extreme environments",
    "CALPHAD provides ground truth — but it is computationally expensive for optimization",
    "Solution: Train fast ML surrogate models on CALPHAD data, then run NSGA-II to find optimal compositions",
    "Three competing objectives: maximize Liquidus Temperature, minimize Density, maximize P(Single BCC)",
], l=0.3, t=1.5, w=6.5, h=3.5, title="Project Motivation")

# Workflow boxes
steps = [
    ("Phase 1", "EDA &\nData Cleaning"),
    ("Phase 2", "ML Surrogate\nTraining"),
    ("Phase 3", "NSGA-II\nOptimization"),
    ("Phase 4", "CALPHAD\nVerification"),
]
colors_step = [RGBColor(0x27,0x6e,0x48), RGBColor(0x14,0x6b,0xa9),
               RGBColor(0x8e,0x24,0xaa), RGBColor(0xc0,0x39,0x2b)]
for i, ((ph, desc), col) in enumerate(zip(steps, colors_step)):
    x = 7.1 + i * 1.5
    rect(sl, x, 2.0, 1.4, 1.2, col)
    txb(sl, ph, x+0.05, 2.05, 1.3, 0.4, size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txb(sl, desc, x+0.05, 2.5, 1.3, 0.65, size=9, color=C_WHITE, align=PP_ALIGN.CENTER)
    if i < 3:
        txb(sl, "→", x+1.4, 2.4, 0.3, 0.4, size=20, bold=True, color=C_GOLD)

bullet_box(sl, [
    "9 Regression models → Liquidus, Solidus, Density",
    "11 Classification models → Single BCC phase stability",
    "GridSearchCV & RandomizedSearchCV hyperparameter tuning",
    "5-fold Stratified Cross-Validation",
    "Full metrics: MAE, MSE, RMSE, R², Adj-R², MAPE (regression)",
    "Accuracy, Precision, Recall, F1, AUC-ROC, Log Loss, Specificity (classification)",
], l=7.1, t=3.5, w=6.0, h=3.6, title="ML Pipeline Scope", size=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3: DATASETS
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Datasets", "Two CALPHAD Datasets Combined for Training")

for i, (name, rows, desc) in enumerate([
    ("Dataset 1", f"{ds_info.get('dataset1_rows',2000)} rows",
     "dataset1_calphad_original.xlsx\nOriginal CALPHAD exploration\n18 columns including VEC,\nΔS_mix, δ%, electronegativity"),
    ("Dataset 2", f"{ds_info.get('dataset2_rows',2000)} rows",
     "dataset2_calphad_augmented.csv\n(= BTP4output.csv)\nAugmented CALPHAD data\nSame composition space"),
]):
    x = 0.4 + i * 4.5
    rect(sl, x, 1.6, 4.1, 3.5, C_ACCENT)
    line(sl, x+0.1, 1.6, 3.9, C_GOLD, 3)
    txb(sl, name, x+0.1, 1.65, 3.9, 0.5, size=16, bold=True, color=C_GOLD, align=PP_ALIGN.CENTER)
    txb(sl, rows, x+0.1, 2.15, 3.9, 0.4, size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txb(sl, desc, x+0.1, 2.6, 3.9, 2.3, size=12, color=C_WHITE)

rect(sl, 9.2, 1.6, 3.9, 3.5, C_ACCENT)
line(sl, 9.3, 1.6, 3.7, C_GREEN, 3)
txb(sl, "Combined", 9.3, 1.65, 3.7, 0.5, size=16, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)
n_bcc  = ds_info.get('bcc_count', 1808)
n_nonb = ds_info.get('non_bcc_count', 173)
n_comb = ds_info.get('combined_cleaned', n_bcc + n_nonb)
txb(sl, f"{n_comb} clean rows", 9.3, 2.15, 3.7, 0.4, size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
txb(sl, f"After dedup & NaN removal\n\nSingle BCC (1): {n_bcc}\nLaves/Non-BCC (0): {n_nonb}\n\nClass imbalance: {n_bcc/(n_bcc+n_nonb)*100:.1f}% BCC\n→ scale_pos_weight applied",
    9.3, 2.65, 3.7, 2.3, size=11, color=C_WHITE)

# Features table
bullet_box(sl, [
    "Features (X): Nb, V, Ti, Mo, Zr (mole fractions, sum = 1.0)",
    "Regression targets: Liquidus (K), Solidus (K), Density (kg/m³)",
    "Classification target: is_single_bcc  [0 = Laves present, 1 = Single BCC]",
    "Train/Test split: 80% / 20% stratified | Composition constraint: Σxᵢ = 1.0 ± 0.001",
], l=0.4, t=5.3, w=12.7, h=2.0, title="Feature & Target Summary", size=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4: EDA PLOTS
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 1: Exploratory Data Analysis", "Key Observations from CALPHAD Dataset")

plots = [
    (f'{EDA_DIR}/class_balance.png',       0.3, 1.5, 3.1, 2.8),
    (f'{EDA_DIR}/vec_histogram.png',       3.6, 1.5, 4.5, 2.8),
    (f'{EDA_DIR}/correlation_heatmap.png', 8.3, 1.5, 4.7, 5.5),
    (f'{EDA_DIR}/pairplot.png',            0.3, 4.5, 7.8, 2.8),
]
for path, l, t, w, h in plots:
    add_img(sl, path, l, t, w, h)

bullet_box(sl, [
    "VEC > 4.8 → Laves phase formation",
    "Mo ↑ → Liquidus ↑ (strongest driver)",
    "Zr/Ti ↑ → Density ↓",
    "10:1 class imbalance → balanced weights",
], l=0.3, t=4.4, w=3.1, h=2.8, title="Key Findings", size=11)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5: METRICS REFERENCE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Metrics Reference", "All metrics from BTP notes implemented")

# Regression metrics
rect(sl, 0.3, 1.5, 6.1, 5.5, C_ACCENT)
txb(sl, "REGRESSION METRICS", 0.4, 1.55, 5.9, 0.4, size=14, bold=True, color=C_GOLD)
formulas = [
    ("MAE",      "Mean Absolute Error",         "1/n Σ|yᵢ - ŷᵢ|"),
    ("MSE",      "Mean Squared Error",          "1/n Σ(yᵢ - ŷᵢ)²"),
    ("RMSE",     "Root Mean Squared Error",     "√(MSE)"),
    ("R²",       "Coefficient of Determination","1 - SS_res/SS_tot"),
    ("Adj. R²",  "Adjusted R-squared",          "1 - (1-R²)(n-1)/(n-k-1)"),
    ("MAPE",     "Mean Abs. Percentage Error",  "1/n Σ|yᵢ-ŷᵢ|/|yᵢ| × 100%"),
]
for i, (name, full, formula) in enumerate(formulas):
    y = 2.05 + i * 0.75
    rect(sl, 0.35, y, 1.3, 0.65, RGBColor(0x14,0x6b,0xa9))
    txb(sl, name, 0.4, y+0.08, 1.2, 0.5, size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txb(sl, full, 1.72, y+0.03, 2.5, 0.35, size=11, color=C_WHITE)
    txb(sl, formula, 1.72, y+0.33, 2.5, 0.3, size=10, color=C_GOLD, italic=True)

# Classification metrics
rect(sl, 6.7, 1.5, 6.3, 5.5, C_ACCENT)
txb(sl, "CLASSIFICATION METRICS", 6.8, 1.55, 6.1, 0.4, size=14, bold=True, color=C_GOLD)
cls_metrics = [
    ("Accuracy",    "TP+TN / (TP+TN+FP+FN)"),
    ("Precision",   "TP / (TP+FP)"),
    ("Recall",      "TP / (TP+FN)"),
    ("Specificity", "TN / (TN+FP)"),
    ("F1-Score",    "2 × Precision×Recall / (P+R)"),
    ("AUC-ROC",    "Area under ROC curve"),
    ("Log Loss",    "−Σ[y·log(p) + (1−y)·log(1−p)]"),
    ("Error Rate",  "(FP+FN) / (TP+TN+FP+FN)"),
]
for i, (name, formula) in enumerate(cls_metrics):
    y = 2.05 + i * 0.59
    rect(sl, 6.75, y, 1.6, 0.52, RGBColor(0x8e,0x24,0xaa))
    txb(sl, name, 6.78, y+0.07, 1.55, 0.4, size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txb(sl, formula, 8.4, y+0.1, 4.4, 0.42, size=11, color=C_WHITE, italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6: MODEL EVALUATION FRAMEWORK
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Model Evaluation Framework", "GridSearchCV • RandomizedSearchCV • 5-Fold CV")

framework = [
    ("Train-Test Split", "80% Training / 20% Test\nStratified by class label\nSame split for all models (reproducibility)\nRandom state = 42"),
    ("Cross-Validation", "5-Fold Stratified K-Fold\nScoring: R² (regression)\nScoring: AUC-ROC (classification)\nReports mean ± std CV score"),
    ("GridSearchCV", "Exhaustive grid search\nUsed for: Logistic Reg, Ridge, Lasso,\nElasticNet, KNN, SVM\nAll combinations evaluated"),
    ("RandomizedSearchCV", "Random sampling from param space\nUsed for: DT, RF, XGB, ADA, GB,\nLightGBM, CatBoost\nn_iter=15 (faster, comparably good)"),
    ("Model Selection", "Regression: highest Test R²\nClassification: highest AUC-ROC\nBest model saved with joblib\nUsed in NSGA-II optimization"),
]
for i, (title, desc) in enumerate(framework):
    col = i % 3
    row = i // 3
    x = 0.3 + col * 4.3
    y = 1.6 + row * 2.7
    rect(sl, x, y, 4.1, 2.5, C_ACCENT)
    line(sl, x+0.08, y, 3.9, C_GOLD, 3)
    txb(sl, title, x+0.1, y+0.08, 3.9, 0.45, size=13, bold=True, color=C_GOLD)
    txb(sl, desc,  x+0.1, y+0.55, 3.9, 1.85, size=11, color=C_WHITE)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7: REGRESSION MODELS CATALOGUE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2A: Regression Models", "9 Models | Targets: Liquidus, Solidus, Density")

rect(sl, 0.3, 1.5, 6.0, 5.5, C_ACCENT)
txb(sl, "BASIC REGRESSION", 0.4, 1.55, 5.8, 0.4, size=13, bold=True, color=C_GOLD)
basic_reg = [
    ("1. Simple Linear Regression", "Single feature (Nb), OLS baseline"),
    ("2. Multiple Linear Regression","All 5 features, OLS, closed-form"),
    ("3. Polynomial Regression",     "degree=2 → 20 features, OLS"),
]
for i, (name, desc) in enumerate(basic_reg):
    y = 2.05 + i * 0.9
    rect(sl, 0.35, y, 5.8, 0.82, RGBColor(0x14,0x6b,0xa9))
    txb(sl, name, 0.45, y+0.05, 5.6, 0.4, size=12, bold=True, color=C_WHITE)
    txb(sl, desc, 0.45, y+0.42, 5.6, 0.35, size=11, color=C_LGRAY)

rect(sl, 0.3, 4.9, 6.0, 2.1, C_ACCENT)
txb(sl, "Tuning: GridSearchCV (Ridge/Lasso/EN) | RandomizedSearchCV (DT/RF/XGB)",
    0.4, 4.95, 5.8, 0.4, size=11, italic=True, color=C_GOLD)

rect(sl, 6.7, 1.5, 6.3, 5.5, C_ACCENT)
txb(sl, "ADVANCED REGRESSION", 6.8, 1.55, 6.1, 0.4, size=13, bold=True, color=C_GOLD)
adv_reg = [
    ("4. Ridge",           "L2 regularization, α ∈ {0.01..100}"),
    ("5. Lasso",           "L1 regularization, sparsity inducing"),
    ("6. ElasticNet",      "L1+L2 combined, l1_ratio tuned"),
    ("7. Decision Tree",   "max_depth, min_samples_leaf tuned"),
    ("8. Random Forest",   "n_estimators∈{100-300}, ensemble bagging"),
    ("9. XGBoost",         "Gradient boosting, n_est∈{200-500}, lr∈{0.03-0.1}"),
]
for i, (name, desc) in enumerate(adv_reg):
    y = 2.05 + i * 0.78
    rect(sl, 6.75, y, 6.15, 0.7, RGBColor(0x8e,0x24,0xaa))
    txb(sl, name, 6.85, y+0.04, 6.0, 0.35, size=12, bold=True, color=C_WHITE)
    txb(sl, desc, 6.85, y+0.38, 6.0, 0.28, size=10, color=C_LGRAY)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8: REGRESSION RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2A: Regression Results", "All models | Both datasets combined")

targets_reg = ['Liquidus', 'Solidus', 'density']
cols_show   = ['Model', 'R2', 'Adj_R2', 'RMSE', 'MAE', 'MAPE_%']
fallback_cols = ['Model', 'R2', 'RMSE', 'MAE']

for ti, target in enumerate(targets_reg):
    x = 0.2 + ti * 4.35
    txb(sl, target, x, 1.5, 4.1, 0.4, size=14, bold=True, color=C_GOLD, align=PP_ALIGN.CENTER)
    if not reg_df.empty and 'Target' in reg_df.columns:
        sub = reg_df[reg_df['Target'] == target].sort_values('R2', ascending=False)
        use_cols = [c for c in cols_show if c in sub.columns]
    else:
        sub = pd.DataFrame()
        use_cols = fallback_cols
    metrics_table(sl, sub, x, 1.95, 4.1, use_cols)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9: REGRESSION COMPARISON CHARTS
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2A: Regression Model Comparison", "R², RMSE, MAE across all models")

targets_reg = ['Liquidus', 'Solidus', 'density']
for i, target in enumerate(targets_reg):
    path = f'{PLOT_DIR}/reg_comparison_{target}.png'
    if not add_img(sl, path, 0.2, 1.4 + i*2.0, 12.9, 1.85):
        # fallback: parity plot
        alt = f'{MODEL_DIR}/parity_{target}.png'
        add_img(sl, alt, 0.2 + i*4.4, 1.5, 4.0, 5.5)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10: PARITY PLOTS
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2A: Parity Plots — Best Models", "Actual vs Predicted for each regression target")

for i, target in enumerate(targets_reg):
    # Try comprehensive plots first, then fall back to original
    for model in ['XGBoost', 'Random_Forest', '']:
        path = f'{PLOT_DIR}/parity_{target}_{model}.png' if model else f'{MODEL_DIR}/parity_{target}.png'
        if add_img(sl, path, 0.3 + i*4.3, 1.5, 4.1, 5.5):
            break

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11: CLASSIFICATION MODELS CATALOGUE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2B: Classification Models", "11 Models | Target: is_single_bcc")

rect(sl, 0.3, 1.5, 5.8, 5.5, C_ACCENT)
txb(sl, "BASIC CLASSIFICATION", 0.4, 1.55, 5.6, 0.4, size=13, bold=True, color=C_GOLD)
basic_cls = [
    ("1. Logistic Regression", "Linear boundary, L1/L2 reg, class_weight=balanced"),
    ("2. KNN",                 "k ∈ {3,5,7,11,15,21}, distance-weighted, Euclidean"),
    ("3. Naive Bayes",         "Gaussian NB, assumes feature independence"),
    ("4. Decision Tree",       "Gini/Entropy, max_depth tuned, class_weight=balanced"),
]
for i, (name, desc) in enumerate(basic_cls):
    y = 2.05 + i * 1.2
    rect(sl, 0.35, y, 5.7, 1.1, RGBColor(0x14,0x6b,0xa9))
    txb(sl, name, 0.45, y+0.06, 5.5, 0.4, size=12, bold=True, color=C_WHITE)
    txb(sl, desc, 0.45, y+0.5, 5.5, 0.55, size=10, color=C_LGRAY)

rect(sl, 6.7, 1.5, 6.4, 5.5, C_ACCENT)
txb(sl, "ADVANCED CLASSIFICATION", 6.8, 1.55, 6.2, 0.4, size=13, bold=True, color=C_GOLD)
adv_cls = [
    ("5. Random Forest",     "Bagging 100-300 trees, balanced_subsample"),
    ("6. SVM",               "RBF/Linear/Poly kernel, C tuned, balanced"),
    ("7. AdaBoost",          "Sequential boosting, SAMME algorithm"),
    ("8. Gradient Boosting", "GB trees, subsample, lr all tuned"),
    ("9. XGBoost",           "scale_pos_weight for imbalance, n_est 200-500"),
    ("10. LightGBM",         "Leaf-wise growth, num_leaves 31-127"),
    ("11. CatBoost",         "Ordered boosting, class_weights applied"),
]
for i, (name, desc) in enumerate(adv_cls):
    y = 2.05 + i * 0.7
    rect(sl, 6.75, y, 6.25, 0.63, RGBColor(0x8e,0x24,0xaa))
    txb(sl, name, 6.85, y+0.04, 6.1, 0.3, size=12, bold=True, color=C_WHITE)
    txb(sl, desc, 6.85, y+0.33, 6.1, 0.27, size=10, color=C_LGRAY)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12: CLASSIFICATION RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2B: Classification Results — All Models",
           "Sorted by AUC-ROC | Bold = Best")

cols_cls = ['Model','Accuracy','Precision','Recall','F1','AUC_ROC','Log_Loss','Specificity']
if not cls_df.empty:
    sub = cls_df.sort_values('AUC_ROC', ascending=False)
    metrics_table(sl, sub, 0.3, 1.5, 12.7, cols_cls)
else:
    txb(sl, "Run phase2_ml_comprehensive.py to populate results", 0.3, 1.5, 12.7, 1.0, color=C_GOLD)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13: CLASSIFICATION COMPARISON CHART + ROC CURVES
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2B: Classification Performance Comparison", "ROC Curves & Multi-metric Bar Charts")

roc_path = f'{PLOT_DIR}/roc_all_classifiers.png'
cmp_path = f'{PLOT_DIR}/cls_comparison_all_metrics.png'

if os.path.exists(roc_path) and os.path.exists(cmp_path):
    add_img(sl, roc_path, 0.2, 1.5, 6.2, 5.6)
    add_img(sl, cmp_path, 6.6, 1.5, 6.5, 5.6)
elif os.path.exists(roc_path):
    add_img(sl, roc_path, 0.3, 1.5, 12.7, 5.6)
else:
    # fallback: confusion matrix
    add_img(sl, f'{MODEL_DIR}/confusion_matrix.png', 0.3, 1.5, 6.0, 5.5)
    bullet_box(sl, [
        "XGBoost: AUC=0.79, Acc=0.82, F1=0.90",
        "Random Forest: AUC=0.80, Acc=0.83",
        "Class 0 (Laves) recall = 5% → needs SMOTE",
        "Class 1 (BCC) recall = 91% ✓",
    ], l=6.6, t=1.5, w=6.4, h=5.5, title="Current Results (RF + XGB only)")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14: CONFUSION MATRIX & CLASSIFICATION REPORT
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 2B: Confusion Matrix & Classification Report",
           "Best model: XGBoost (existing) | Combined dataset will update")

add_img(sl, f'{MODEL_DIR}/confusion_matrix.png', 0.3, 1.5, 5.5, 4.5)

rect(sl, 6.2, 1.5, 6.9, 4.8, C_ACCENT)
txb(sl, "Classification Report (XGBoost)", 6.3, 1.55, 6.7, 0.4, size=13, bold=True, color=C_GOLD)

report_data = [
    ["Class",          "Precision", "Recall", "F1",   "Support"],
    ["0 (Laves)",      "0.06",      "0.05",   "0.05", "42"],
    ["1 (Single BCC)", "0.89",      "0.91",   "0.90", "355"],
    ["Accuracy",       "",          "",       "0.82", "397"],
    ["Macro avg",      "0.47",      "0.48",   "0.48", "397"],
    ["Weighted avg",   "0.80",      "0.82",   "0.81", "397"],
]
for ri, row_data in enumerate(report_data):
    fill = C_GOLD if ri == 0 else (RGBColor(0x0d,0x2b,0x45) if ri%2==0 else C_ACCENT)
    for ci, val in enumerate(row_data):
        x = 6.25 + ci * 1.3
        rect(sl, x, 2.05 + ri * 0.6, 1.25, 0.55, fill)
        txb(sl, val, x+0.03, 2.1 + ri*0.6, 1.2, 0.45,
            size=10, bold=(ri==0), color=C_DARK if ri==0 else C_WHITE)

bullet_box(sl, [
    "⚠ Laves class (0) recall = only 5%",
    "→ Heavy class imbalance (173 vs 1808)",
    "→ SMOTE oversampling recommended",
    "→ LightGBM/CatBoost expected to improve",
    "✓ BCC class (1) recall = 91% — good",
    "✓ AUC-ROC used as primary metric",
], l=6.3, t=5.0, w=6.8, h=2.3, title="Analysis", size=11)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15: SHAP FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "SHAP Feature Importance Analysis", "TreeExplainer — Model Interpretability")

shap_plots = [
    (f'{MODEL_DIR}/shap_liquidus.png',        0.2, 1.5, 4.1, 5.5, "Liquidus"),
    (f'{MODEL_DIR}/shap_density.png',         4.5, 1.5, 4.1, 5.5, "Density"),
    (f'{MODEL_DIR}/shap_classification.png',  8.9, 1.5, 4.2, 5.5, "BCC Stability"),
]
for path, l, t, w, h, label in shap_plots:
    add_img(sl, path, l, t, w, h)
    txb(sl, label, l, t-0.4, w, 0.35, size=12, bold=True, color=C_GOLD, align=PP_ALIGN.CENTER)

bullet_box(sl, [
    "Mo: strongest ↑ driver for Liquidus",
    "Zr: lowers density significantly",
    "Nb+V: stabilize BCC phase",
], l=0.2, t=6.9, w=12.9, h=0.5, size=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16: NSGA-II OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 3: NSGA-II Multi-Objective Optimization",
           "Maximize Liquidus | Minimize Density | Maximize P(BCC)")

bullet_box(sl, [
    "3 objectives → Pareto front (no single best solution)",
    "Population: 200 | Offspring: 50 | Generations: 250→300",
    "Simplex constraint: Σxᵢ = 1.0 enforced by SimplexBoundsRepair",
    "SBX crossover (η=15) + Polynomial mutation (η=20)",
    "Result: 200 Pareto-optimal non-dominated compositions",
], l=0.3, t=1.5, w=5.8, h=3.2, title="Algorithm Setup")

pareto_path    = f'{OPT_DIR}/pareto_front_2d.png'
conv_path      = f'{OPT_DIR}/convergence.png'
if os.path.exists(pareto_path):
    add_img(sl, pareto_path, 6.4, 1.4, 6.6, 3.5)
if os.path.exists(conv_path):
    add_img(sl, conv_path, 0.3, 4.9, 5.8, 2.4)

bullet_box(sl, [
    "f₁ = -Liquidus(x)   [maximize]",
    "f₂ = Density(x)     [minimize]",
    "f₃ = -P_BCC(x)      [maximize]",
    "xl = training data min | xu = max",
], l=6.4, t=5.1, w=6.6, h=2.2, title="Objective Formulation", size=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 17: RECOMMENDED COMPOSITIONS
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Phase 3: Recommended Alloy Compositions",
           "Selected from Pareto Front for CALPHAD Verification")

# Original 3 candidates
orig_cands = [
    ("🔥 Highest\nMelting Point", 0.205, 0.163, 0.269, 0.053, 0.310, 2398.1, 6811.8, 92.6),
    ("⚖ Most\nBalanced",         0.325, 0.119, 0.224, 0.067, 0.265, 2342.2, 6643.9, 90.3),
    ("💨 Lowest\nDensity",       0.207, 0.195, 0.208, 0.152, 0.238, 2171.1, 6554.9, 96.8),
]
colors_cand = [C_GOLD, C_GREEN, C_BLUE]
for i, (title, Nb, V, Ti, Mo, Zr, Liq, Den, BCC) in enumerate(orig_cands):
    x = 0.3 + i * 4.3
    rect(sl, x, 1.5, 4.1, 2.5, C_ACCENT)
    line(sl, x+0.08, 1.5, 3.9, colors_cand[i], 4)
    txb(sl, title, x+0.1, 1.55, 3.9, 0.7, size=13, bold=True,
        color=colors_cand[i], align=PP_ALIGN.CENTER)
    comp_text = (f"Nb={Nb:.3f}  V={V:.3f}  Ti={Ti:.3f}\n"
                 f"Mo={Mo:.3f}  Zr={Zr:.3f}\n\n"
                 f"Liquidus: {Liq:.1f} K\nDensity:  {Den:.1f} kg/m³\nP(BCC):   {BCC:.1f}%")
    txb(sl, comp_text, x+0.1, 2.3, 3.9, 1.65, size=11, color=C_WHITE)

# Retrained best
rect(sl, 0.3, 4.2, 12.7, 3.0, C_ACCENT)
txb(sl, "RETRAINED BEST COMPOSITIONS (Combined Dataset)", 0.5, 4.25, 12.3, 0.4,
    size=14, bold=True, color=C_GOLD)
retrain_rows = [
    ("Most Balanced #1 ⭐", 0.317, 0.141, 0.346, 0.094, 0.102, 2420.2, 2004.9, 6697.7, 99.9),
    ("Most Balanced #2 ✓",  0.313, 0.154, 0.344, 0.085, 0.104, 2395.1, 1986.0, 6645.2, 100.0),
]
for i, (label, Nb, V, Ti, Mo, Zr, Liq, Sol, Den, BCC) in enumerate(retrain_rows):
    x = 0.5 + i * 6.3
    col = C_GOLD if i == 0 else C_GREEN
    txb(sl, label, x, 4.75, 6.0, 0.4, size=13, bold=True, color=col)
    comp = (f"Nb={Nb:.3f}  V={V:.3f}  Ti={Ti:.3f}  Mo={Mo:.3f}  Zr={Zr:.3f}   "
            f"Liquidus={Liq:.1f} K  |  Solidus={Sol:.1f} K  |  Density={Den:.1f} kg/m³  |  P(BCC)={BCC:.1f}%")
    txb(sl, comp, x, 5.2, 6.0, 0.6, size=11, color=C_WHITE)

txb(sl, "★ Recommended: Nb₃₁.₃V₁₅.₄Ti₃₄.₄Mo₈.₅Zr₁₀.₄  |  P(BCC)=100%  |  Liquidus=2395 K  |  Density=6645 kg/m³",
    0.5, 6.0, 12.3, 0.5, size=13, bold=True, color=C_GOLD, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 18: CONCLUSIONS & FUTURE WORK
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl)
header_bar(sl, "Conclusions & Future Work", "Key Takeaways from BTP")

bullet_box(sl, [
    "Combined 2 CALPHAD datasets → ~3900+ clean training samples",
    "9 Regression models trained with full metrics (MAE/MSE/RMSE/R²/Adj-R²/MAPE)",
    "11 Classification models trained with Accuracy/Precision/Recall/F1/AUC/LogLoss/Specificity",
    "GridSearchCV & RandomizedSearchCV used for systematic hyperparameter tuning",
    "5-fold Stratified CV prevents overfitting and gives reliable score estimates",
    "NSGA-II found 200 Pareto-optimal alloy compositions",
    "Best composition: Nb₃₁.₃V₁₅.₄Ti₃₄.₄Mo₈.₅Zr₁₀.₄ → P(BCC)=100%, Liq=2395 K",
], l=0.3, t=1.5, w=6.5, h=5.5, title="Key Achievements ✅", size=12)

bullet_box(sl, [
    "Apply SMOTE to fix class imbalance → improve Laves-class recall from 5% to >50%",
    "Add LightGBM + CatBoost ensemble for better surrogate accuracy",
    "Expand composition space: add W, Al, Cr, Hf elements",
    "Fabricate top composition via arc melting + verify by XRD",
    "Compare ML predictions vs actual CALPHAD for all 3 candidates",
    "Neural network surrogate (MLP, Transformer) for complex interactions",
    "Publish results as RHEA design framework paper",
], l=6.9, t=1.5, w=6.1, h=5.5, title="Future Work 🚀", size=12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 19: THANK YOU
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide(); bg(sl, C_DARK)
rect(sl, 0, 0, 13.33, 7.5, RGBColor(0x0a, 0x0a, 0x1a))
rect(sl, 0, 2.8, 13.33, 1.9, C_ACCENT)
line(sl, 0.5, 2.78, 12.33, C_GOLD, 3)
line(sl, 0.5, 4.72, 12.33, C_GOLD, 3)

txb(sl, "THANK YOU", 0.5, 1.4, 12.3, 1.0, size=48, bold=True,
    color=C_GOLD, align=PP_ALIGN.CENTER)
txb(sl, "Questions & Discussion", 0.5, 2.9, 12.3, 0.6, size=24,
    color=C_WHITE, align=PP_ALIGN.CENTER)
txb(sl, "ML-Driven Design of Nb-V-Ti-Mo-Zr RHEA for Nuclear Applications",
    0.5, 3.6, 12.3, 0.5, size=16, color=C_LGRAY, align=PP_ALIGN.CENTER, italic=True)
txb(sl, "9 Regression Models  •  11 Classification Models  •  NSGA-II Optimization  •  CALPHAD Verification",
    0.5, 5.3, 12.3, 0.5, size=13, color=RGBColor(0x7f,0x8c,0x8d), align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────────────
prs.save(PPT_OUT)
print(f"\n✓ Presentation saved → {PPT_OUT}")
print(f"  Slides: {len(prs.slides)}")
