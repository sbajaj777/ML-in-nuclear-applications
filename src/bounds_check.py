import pandas as pd
import numpy as np
import joblib

# 1. Load the training data to get exact bounds
train_df = pd.read_csv('../data_cleaning/cleaned_dataset.csv')
features = ['Nb', 'V', 'Ti', 'Mo', 'Zr']
train_min = train_df[features].min()
train_max = train_df[features].max()

# 2. Load the Pareto front
pareto_df = pd.read_csv('../optimization/pareto_front.csv')

# 3. Check bounds for every element across the entire Pareto front
bounds_violations = 0
for feat in features:
    min_val = train_min[feat]
    max_val = train_max[feat]
    
    out_of_bounds = pareto_df[(pareto_df[feat] < min_val - 1e-6) | (pareto_df[feat] > max_val + 1e-6)]
    if not out_of_bounds.empty:
        bounds_violations += len(out_of_bounds)
        print(f"VIOLATION: {len(out_of_bounds)} rows out of bounds for {feat}")

# 4. Check sum to 1.0
sums = pareto_df[features].sum(axis=1)
sum_violations = sums[~np.isclose(sums, 1.0, atol=0.001)]
if not sum_violations.empty:
    print(f"VIOLATION: {len(sum_violations)} rows do not sum to 1.0")
else:
    print("All rows successfully sum to 1.0 (±0.001).")

if bounds_violations == 0:
    print("All 200 Pareto front compositions strictly respect the training min/max bounds.")

# 5. Get full properties for 'Most Balanced' including Solidus
rec_df = pd.read_csv('../optimization/recommended_compositions.csv')
most_balanced = rec_df[rec_df['Rationale'] == 'Most Balanced'].iloc[0]

model_solidus = joblib.load('../models/best_reg_Solidus.joblib')
comp_df = pd.DataFrame([most_balanced[features].values], columns=features)
pred_solidus = model_solidus.predict(comp_df)[0]

print("\n--- MOST BALANCED PREDICTIONS ---")
print(f"Liquidus: {most_balanced['Liquidus']:.2f} K")
print(f"Solidus: {pred_solidus:.2f} K")
print(f"Density: {most_balanced['Density']:.2f} kg/m3")
print(f"Prob(BCC): {most_balanced['Prob_BCC']:.4f}")
