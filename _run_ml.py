"""
Runner for high_gainer_full_ml.py
"""
import subprocess, sys, os

# Fix the test_data bug in Phase 9b before running
# Patch the script to fix error analysis section
with open(r'C:\Users\pc\Downloads\stock hist data\high_gainer_full_ml.py', 'r') as f:
    code = f.read()

# Fix 1: Replace the test_data reference in Phase 9b with proper walkforward data join
old = """wf_df['feat_bin'] = pd.qcut(test_data[feat].reindex(index=wf_df.index).fillna(0), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')"""
new = """wf_df['feat_bin'] = pd.qcut(wf_df['ret'].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')"""
code = code.replace(old, new)

# Fix 2: Remove dead code on line 626
old = """shap_values = best_final_model.predict_proba(shap_sample)[:, 1]  # shap.TreeExplainer"""
new = """# shap_values not needed separately"""
code = code.replace(old, new)

with open(r'C:\Users\pc\Downloads\stock hist data\high_gainer_full_ml.py', 'w') as f:
    f.write(code)

print("Patches applied. Running high_gainer_full_ml.py...")
sys.stdout.flush()

result = subprocess.run(
    [sys.executable, 'high_gainer_full_ml.py'],
    cwd=r'C:\Users\pc\Downloads\stock hist data',
    capture_output=False
)
print(f"Exit code: {result.returncode}")
