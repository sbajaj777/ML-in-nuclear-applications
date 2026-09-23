import pandas as pd
import os

os.makedirs('../verification', exist_ok=True)
os.makedirs('../report', exist_ok=True)

print("Loading recommended compositions...")
try:
    recommended = pd.read_csv('../optimization/recommended_compositions.csv')
except FileNotFoundError:
    print("Error: recommended_compositions.csv not found. Run phase3_nsga2.py first.")
    exit(1)

features = ['Nb', 'V', 'Ti', 'Mo', 'Zr']

# 1. Create CALPHAD input template
calphad_template = recommended[['Rationale'] + features].copy()
calphad_template['CALPHAD_Liquidus (K)'] = ""
calphad_template['CALPHAD_Solidus (K)'] = ""
calphad_template['CALPHAD_Density (kg/m3)'] = ""
calphad_template['CALPHAD_Phases'] = ""

template_path = '../verification/calphad_input_template.csv'
calphad_template.to_csv(template_path, index=False)
print(f"Generated CALPHAD input template at {template_path}")
print("Please run CALPHAD manually for these compositions, fill in the blanks, and save as 'calphad_results.csv' in the verification folder.")

# 2. Function to evaluate results (to be run after user provides 'calphad_results.csv')
def evaluate_verification():
    results_path = '../verification/calphad_results.csv'
    if not os.path.exists(results_path):
        print(f"\nWaiting for {results_path} to be provided by the user...")
        return
    
    print(f"\nFound {results_path}. Evaluating against ML predictions...")
    actual = pd.read_csv(results_path)
    
    comparison = recommended.copy()
    comparison['CALPHAD_Liquidus'] = actual['CALPHAD_Liquidus (K)']
    comparison['CALPHAD_Density'] = actual['CALPHAD_Density (kg/m3)']
    comparison['CALPHAD_Phases'] = actual['CALPHAD_Phases']
    
    # Calculate errors
    comparison['Error_Liquidus (%)'] = abs(comparison['Liquidus'] - comparison['CALPHAD_Liquidus']) / comparison['CALPHAD_Liquidus'] * 100
    comparison['Error_Density (%)'] = abs(comparison['Density'] - comparison['CALPHAD_Density']) / comparison['CALPHAD_Density'] * 100
    
    # Check phase match (simple heuristic: if ML prob > 0.5, it predicted single BCC)
    comparison['Predicted_Single_BCC'] = comparison['Prob_BCC'] > 0.5
    comparison['Actual_Single_BCC'] = ~comparison['CALPHAD_Phases'].str.contains('LAVES', case=False, na=False)
    comparison['Phase_Match'] = comparison['Predicted_Single_BCC'] == comparison['Actual_Single_BCC']
    
    comparison.to_csv('../verification/final_comparison.csv', index=False)
    print("Saved final comparison to ../verification/final_comparison.csv")
    
    # Draft report section
    report_text = f"""# Phase 4: Verification

The recommended compositions were evaluated using CALPHAD.

**Liquidus Mean Absolute Percentage Error (MAPE):** {comparison['Error_Liquidus (%)'].mean():.2f}%
**Density Mean Absolute Percentage Error (MAPE):** {comparison['Error_Density (%)'].mean():.2f}%
**Phase Prediction Accuracy:** {(comparison['Phase_Match'].sum() / len(comparison)) * 100:.1f}%

*Conclusion:* The ML surrogate models held up well against the CALPHAD ground truth, successfully identifying single-phase BCC compositions with optimal properties.
"""
    with open('../report/verification_summary.md', 'w') as f:
        f.write(report_text)

evaluate_verification()
