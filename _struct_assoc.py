import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'deep_analysis_report'
ef = pd.read_parquet(BASE / 'cleaned_features.parquet')
struct_cols = [c for c in ef.columns if c in ('fvg_bullish','fvg_bearish','ob_bullish','ob_bearish','liq_sweep_high','liq_sweep_low','bos_up','bos_down','choch_sell','choch_buy','wyckoff_spring','wyckoff_upthrust','mkt_in_value_area','vol_profile_high_vol_node','vol_profile_low_vol_node')]
sa = []
for col in struct_cols:
    if col not in ef.columns: continue
    n = int(ef[col].sum())
    if n < 50: continue
    hr = float(ef.loc[ef[col]==1, 'target'].mean())
    br = float(ef['target'].mean())
    lift = hr/br if br>0 else 0
    try:
        tbl = pd.crosstab(ef[col], ef['target'])
        if tbl.shape == (2,2):
            _, p_val, _, _ = chi2_contingency(tbl)
        else: p_val = 1.0
    except: p_val = 1.0
    sa.append({'feature':col, 'n':n, 'hit_rate':hr, 'lift':lift, 'pval':float(p_val)})
sa_df = pd.DataFrame(sa).sort_values('lift', ascending=False)
sa_df.to_csv(OUT/'tables'/'structure_feature_association.csv', index=False)
print('Structure features:')
for _, r in sa_df.iterrows():
    print(f'  {r["feature"]:<30s} lift={r["lift"]:.2f} hit={r["hit_rate"]:.1%} p={r["pval"]:.4f}')
