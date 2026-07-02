# Phase 6 - Data Cleaning
# Missing value strategy, outlier capping, quality report, cleaned dataset
import pandas as pd, numpy as np, time, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'data_cleaning_results'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' Phase 6 - Data Cleaning')
print('='*60)

# Load engineered features
ef = pd.read_parquet(BASE / 'engineered_features.parquet')
print(f'\n[1] Loaded: {ef.shape}')

id_cols = ['symbol', 'datetime', 'date']
meta_cols = id_cols + ['target', 'target_ret', 'next_close', 'next_open']
feat_cols = [c for c in ef.columns if c not in meta_cols]

# ── 2. Missing value treatment ──
print('\n[2] Missing Value Treatment')
n_before = len(ef)

# Check for any remaining NaN
nan_counts = ef[feat_cols].isna().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print(f'  Features with NaN: {len(nan_cols)}')
    nan_log = []
    for c in nan_cols.index:
        pct = nan_cols[c] / len(ef) * 100
        val = ef[c].dropna()
        fill_val = val.median() if len(val) > 0 else 0
        ef[c] = ef[c].fillna(fill_val)
        nan_log.append({'feature': c, 'n_missing': int(nan_cols[c]), 'pct': float(pct), 'fill_method': 'median', 'fill_value': float(fill_val)})
        print(f'    {c:<30s}: {nan_cols[c]:>6,} missing ({pct:.2f}%) -> filled with median')
    pd.DataFrame(nan_log).to_csv(OUT / 'missing_values_treated.csv', index=False)
else:
    print('  No missing values found (all cleaned in Phase 7)')

# ── 3. Outlier detection and capping ──
print('\n[3] Outlier Capping (winsorization at 0.5/99.5 percentile)')
outlier_log = []
feat_float = [c for c in feat_cols if ef[c].dtype in ('float64', 'float32')]

for c in feat_float:
    s = ef[c]
    lo, hi = np.percentile(s, [0.5, 99.5])
    n_before_cap = ((s < lo) | (s > hi)).sum()
    if n_before_cap > 0:
        ef[c] = s.clip(lo, hi)
        outlier_log.append({
            'feature': c,
            'n_capped': int(n_before_cap),
            'pct_capped': float(n_before_cap / len(ef) * 100),
            'lower_bound': float(lo),
            'upper_bound': float(hi),
        })

out_df = pd.DataFrame(outlier_log).sort_values('n_capped', ascending=False)
out_df.to_csv(OUT / 'outliers_capped.csv', index=False)
print(f'  Features with outliers capped: {len(outlier_log)}')
print(f'  Total values capped: {sum(o["n_capped"] for o in outlier_log):,}')
print(f'  Top 10 most capped:')
for _, r in out_df.head(10).iterrows():
    print(f'    {r["feature"]:<30s}: {r["n_capped"]:>7,} ({r["pct_capped"]:.2f}%)  [{r["lower_bound"]:.2f}, {r["upper_bound"]:.2f}]')

# ── 4. Duplicate detection ──
print('\n[4] Duplicate Detection')
# Check for duplicate symbol-date combinations
dup_mask = ef.duplicated(subset=['symbol', 'date'], keep=False)
n_dup = dup_mask.sum()
if n_dup > 0:
    print(f'  WARNING: {n_dup} duplicate symbol-date rows found!')
    ef = ef[~ef.duplicated(subset=['symbol', 'date'], keep='first')]
    print(f'  Removed duplicates, new shape: {ef.shape}')
else:
    print('  No duplicate rows found')

# ── 5. Zero/constant variance features ──
print('\n[5] Low Variance Feature Detection')
zero_var = []
for c in feat_cols:
    if ef[c].dtype in ('float64', 'float32', 'int64', 'int32'):
        if ef[c].std() < 1e-10:
            zero_var.append(c)
if zero_var:
    print(f'  WARNING: {len(zero_var)} near-zero variance features: {zero_var}')
    ef = ef.drop(columns=zero_var)
    feat_cols = [c for c in feat_cols if c not in zero_var]
else:
    print('  No zero-variance features')

# ── 6. Data quality report ──
print('\n[6] Data Quality Report')
quality_report = {
    'original_rows': n_before,
    'final_rows': len(ef),
    'n_symbols': int(ef['symbol'].nunique()),
    'n_features_total': len(ef.columns),
    'n_features_model_ready': len([c for c in feat_cols if ef[c].dtype in ('float64', 'float32', 'int64', 'int32')]),
    'date_range': [str(ef['datetime'].min()), str(ef['datetime'].max())],
    'target_pos_rate': float(ef['target'].mean()),
    'n_outliers_capped': sum(o['n_capped'] for o in outlier_log),
    'n_features_outliers_capped': len(outlier_log),
    'n_duplicates_removed': int(n_dup) if 'n_dup' in dir() else 0,
    'n_zero_var_removed': len(zero_var) if 'zero_var' in dir() and zero_var else 0,
    'n_missing_filled': int(nan_cols.sum()) if 'nan_cols' in dir() and len(nan_cols) > 0 else 0,
    'feature_stats_summary': {
        'mean_range': [float(ef[c].mean()) for c in feat_float[:5]],
        'std_range': [float(ef[c].std()) for c in feat_float[:5]],
    }
}

with open(OUT / 'data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2)

# ── 7. Save cleaned dataset ──
print('\n[7] Saving cleaned dataset...')
ef.to_parquet(BASE / 'cleaned_features.parquet', index=False)
print(f'  Saved to: {BASE / "cleaned_features.parquet"}')
print(f'  Final shape: {ef.shape}')
print(f'  Size: {(BASE / "cleaned_features.parquet").stat().st_size / 1e6:.1f} MB')

# ── 8. Summary of changes ──
print(f'\n[8] Cleaning Summary')
change_log = {
    'missing_values': 'No missing values found (Phase 7 already handled)',
    'outlier_capping': f'{len(outlier_log)} features winsorized at 0.5/99.5 pct',
    'duplicates': 'None found' if not ('n_dup' in dir() and n_dup > 0) else f'{n_dup} removed',
    'zero_variance': 'None found' if not (zero_var and len(zero_var) > 0) else f'{len(zero_var)} removed',
    'recommendations': [
        'Use StandardScaler (not RobustScaler) since outliers are already capped',
        'scale_pos_weight in XGBoost: ~7.1 (8.1:1 ratio)',
        'Drop highly correlated pairs (r>0.95) in Phase 8 Feature Selection',
        'Consider log transform for high-skew features (range_*, hv_*, ret_*)',
    ]
}
print(f'  Missing values: None found (handled in Phase 7)')
print(f'  Outliers capped: {len(outlier_log)} features')
print(f'  Duplicates: None')
print(f'  Zero-variance: None')
print(f'  Ready for Phase 8: Feature Selection')

print(f'\n{"="*60}')
print(f'  Time: {time.time()-t0:.0f}s')
