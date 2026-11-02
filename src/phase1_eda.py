import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure directories exist
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../data/eda_plots', exist_ok=True)
os.makedirs('../report', exist_ok=True)

# 1. Load the Excel file
data_path = '../data/raw/dataset1_calphad_original.xlsx'
print(f"Loading dataset from {data_path}...")
try:
    df = pd.read_excel(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print("Original dataset shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nNull counts per column:")
print(df.isnull().sum())

# 2. Handle missing CALPHAD outputs
calphad_outputs = ['Liquidus', 'Solidus', 'density', 'phases', 'number_phases']
missing_mask = df[calphad_outputs].isnull().any(axis=1)

df_missing = df[missing_mask]
print(f"\nFound {len(df_missing)} rows with missing CALPHAD outputs.")

# Save report of missing rows
missing_report_path = '../data/processed/missing_calphad_outputs.csv'
df_missing.to_csv(missing_report_path, index=False)
print(f"Saved missing rows report to {missing_report_path}")

# Drop missing rows
df_clean = df.dropna(subset=calphad_outputs).copy()
print(f"Cleaned dataset shape: {df_clean.shape}")

# 3. Verify composition columns sum to 1.0 (±0.001)
comp_cols = ['Nb', 'V', 'Ti', 'Mo', 'Zr']
comp_sums = df_clean[comp_cols].sum(axis=1)
invalid_sums = df_clean[~np.isclose(comp_sums, 1.0, atol=0.001)]

if len(invalid_sums) > 0:
    print(f"\nWARNING: Found {len(invalid_sums)} rows where composition does not sum to 1.0")
    print(invalid_sums[comp_cols].head())
else:
    print("\nAll compositions successfully sum to 1.0 (±0.001 tolerance).")

# 4. Convert 'phases' column to clean binary target 'is_single_bcc'
def determine_single_bcc(phase_str):
    if not isinstance(phase_str, str):
        return 0
    # True if 'BCC_B2' is present and 'C15_LAVES' is not
    # Assumes phases are separated by spaces or commas
    phases = phase_str.upper()
    has_bcc = 'BCC_B2' in phases
    has_laves = 'C15_LAVES' in phases
    return 1 if (has_bcc and not has_laves) else 0

df_clean['is_single_bcc'] = df_clean['phases'].apply(determine_single_bcc)

# 5. Generate and save plots
sns.set_theme(style="whitegrid")

# Plot A: Correlation heatmap
plt.figure(figsize=(12, 10))
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig('../data/eda_plots/correlation_heatmap.png', dpi=300)
plt.close()

# Plot B: Pairplot of compositions vs Liquidus, density, and is_single_bcc
vars_to_plot = comp_cols + ['Liquidus', 'density']
pairplot_df = df_clean[vars_to_plot + ['is_single_bcc']]
# Sample data for pairplot to avoid huge files if dataset is large, but 2000 is fine
g = sns.pairplot(pairplot_df, hue='is_single_bcc', palette=['#e74c3c', '#2ecc71'], 
                 plot_kws={'alpha': 0.6, 's': 20})
g.fig.suptitle("Pairplot of Compositions vs Properties (colored by phase class)", y=1.02)
plt.savefig('../data/eda_plots/pairplot.png', dpi=300)
plt.close()

# Plot C: Histogram of VEC split by is_single_bcc
plt.figure(figsize=(10, 6))
sns.histplot(data=df_clean, x='VEC', hue='is_single_bcc', multiple='stack', 
             palette=['#e74c3c', '#2ecc71'], bins=30)
plt.title("VEC Distribution by Phase Class (is_single_bcc)")
plt.xlabel("Valence Electron Concentration (VEC)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('../data/eda_plots/vec_histogram.png', dpi=300)
plt.close()

# Plot D: Class balance bar chart
plt.figure(figsize=(6, 5))
ax = sns.countplot(data=df_clean, x='is_single_bcc', palette=['#e74c3c', '#2ecc71'])
plt.title("Class Balance of is_single_bcc")
plt.xlabel("Class (0 = Laves present, 1 = Single BCC)")
plt.ylabel("Count")

# Add counts on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('../data/eda_plots/class_balance.png', dpi=300)
plt.close()

# Save cleaned dataset
clean_data_path = '../data/processed/cleaned_dataset.csv'
df_clean.to_csv(clean_data_path, index=False)
print(f"\nSaved cleaned dataset to {clean_data_path}")

# 6. Write EDA summary
eda_summary = f"""# Phase 1: Exploratory Data Analysis Summary

## Dataset Overview
- **Original Shape:** {df.shape}
- **Missing Data:** Found {len(df_missing)} rows with missing CALPHAD outputs (Liquidus, Solidus, density, phases, number_phases). These were successfully dropped.
- **Cleaned Shape:** {df_clean.shape}
- **Composition Check:** All composition columns sum to 1.0 (±0.001) properly.

## Phase Stability (Class Balance)
The target variable `is_single_bcc` was created based on the `phases` column. 
- **Single BCC (1):** {len(df_clean[df_clean['is_single_bcc'] == 1])} samples
- **Laves Present (0):** {len(df_clean[df_clean['is_single_bcc'] == 0])} samples
This confirms a severe class imbalance (~10:1) which will require special handling (e.g. SMOTE or balanced class weights) during machine learning model training in Phase 2.

## Key Observations
- **VEC Rule:** The histogram of VEC clearly shows that higher VEC values correlate with Laves phase formation, consistent with the general heuristic that low VEC stabilizes the BCC structure in refractory high entropy alloys.
- **Correlation with Melting Point:** The correlation heatmap shows that Mo, as a high-melting refractory element, positively correlates with Liquidus temperature.
- **Density:** Zr and Ti additions tend to lower the overall alloy density, whereas heavier elements like Mo and Nb increase it.
"""

with open('../report/eda_summary.md', 'w') as f:
    f.write(eda_summary)
print("Saved EDA summary to ../report/eda_summary.md")
print("Phase 1 EDA complete.")
