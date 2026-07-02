# Phase 9: Formal Walkforward Split + Improved Dataset
import pandas as pd, numpy as np, json, time
from pathlib import Path
from datetime import datetime

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'phase9_split'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*70)
print('PHASE 9: WALKFORWARD TRAIN/VAL/TEST SPLIT')
print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('='*70)

# ─── 1. Load data + apply improvements ───
print('\n[1] Loading & preparing improved dataset...')
df = pd.read_parquet(BASE/'cleaned_features.parquet')
df['_d'] = pd.to_datetime(df['datetime'])

# Symbol filter (bottom 25% by volume)
sym_vol = df.groupby('symbol')['volume'].median().sort_values()
keep = int(len(sym_vol) * 0.75)
keep_syms = set(sym_vol.tail(keep).index)
df = df[df['symbol'].isin(keep_syms)].copy()
print(f'  Symbols: {len(sym_vol)} -> {len(keep_syms)} | Rows: {len(df):,}')

# Feature interactions
target_col = 'target'
feature_cols = [c for c in df.columns if c not in ('symbol','date','datetime','target','target_ret','symbol_date','Date','Symbol','_d')]
feature_cols = [c for c in feature_cols if not any(p in c for p in ('next_','fwd_','future_'))]
print(f'  Base features: {len(feature_cols)}')

regime_cols = [c for c in feature_cols if c.startswith('regime_')]
vol_cols = ['range_5','hv_20','range_10','hv_10','bb_width']
ret_cols = ['ret_1d','ret_1d_std_5','ret_5d_lag1']
cross_cols = ['rank_range_5','rank_hv_20']
month_dummies = [c for c in df.columns if c.startswith('month_') and c!='month']
dow_dummies = [c for c in df.columns if c.startswith('dow_')]

interactions = []
used = set(feature_cols)
for r in regime_cols:
    for v in vol_cols:
        n = f'{r}_x_{v}'
        if n not in used: df[n]=df[r]*df[v]; interactions.append(n); used.add(n)
for m in month_dummies[:6]:
    for v in vol_cols[:3]:
        n = f'{m}_x_{v}'
        if n not in used: df[n]=df[m]*df[v]; interactions.append(n); used.add(n)
for d in dow_dummies[:3]:
    for rv in ret_cols[:2]:
        n = f'{d}_x_{rv}'
        if n not in used: df[n]=df[d]*df[rv]; interactions.append(n); used.add(n)
for c in cross_cols[:2]:
    for v in vol_cols[:2]:
        n = f'{c}_x_{v}'
        if n not in used: df[n]=df[c]*df[v]; interactions.append(n); used.add(n)
print(f'  Added {len(interactions)} interactions')

all_features = feature_cols + interactions
print(f'  Total features: {len(all_features)}')

# Save improved dataset
improved_cols = ['symbol','datetime','_d'] + all_features + [target_col]
df_improved = df[improved_cols].copy()
df_improved.to_parquet(BASE/'improved_features.parquet', index=False)
print(f'  Saved: improved_features.parquet ({df_improved.shape})')

# ─── 2. Walkforward splits ───
print('\n[2] Defining walkforward splits...')
# Expanding window, semi-annual folds
splits = [
    # (name, train_start, train_end, val_start, val_end, test_start, test_end)
    ('2023-H1', '2016-01-01', '2023-01-01', '2023-01-01', '2023-07-01', None, None),
    ('2023-H2', '2016-01-01', '2023-07-01', '2023-07-01', '2024-01-01', None, None),
    ('2024-H1', '2017-01-01', '2024-01-01', '2024-01-01', '2024-07-01', None, None),
    ('2024-H2', '2017-01-01', '2024-07-01', '2024-07-01', '2025-01-01', None, None),
    ('2025-H1', '2018-01-01', '2025-01-01', '2025-01-01', '2025-07-01', None, None),
]
# Final test: 2025-07 to 2026-06
test_start, test_end = '2025-07-01', '2026-06-26'

fold_info = []
for name, ts, te, vs, ve, _, _ in splits:
    tr = df_improved[(df_improved['_d'] >= ts) & (df_improved['_d'] < te)]
    va = df_improved[(df_improved['_d'] >= vs) & (df_improved['_d'] < ve)]
    tr_tgt = tr[target_col].mean()
    va_tgt = va[target_col].mean()
    fold_info.append({'fold':name,'train_rows':len(tr),'val_rows':len(va),
                       'train_target_rate':f'{tr_tgt:.4f}','val_target_rate':f'{va_tgt:.4f}'})
    print(f'  {name:10s} train={len(tr):>7,} ({tr_tgt:.4f}) val={len(va):>7,} ({va_tgt:.4f})')

# Final test set
test_df = df_improved[(df_improved['_d'] >= test_start) & (df_improved['_d'] < test_end)]
print(f'  {"TEST":10s} test={len(test_df):>7,} ({test_df[target_col].mean():.4f})')
fold_info.append({'fold':'TEST','train_rows':0,'val_rows':len(test_df),
                   'train_target_rate':'','val_target_rate':f'{test_df[target_col].mean():.4f}'})

pd.DataFrame(fold_info).to_csv(OUT/'walkforward_folds.csv', index=False)
print(f'  Splits saved to {OUT/"walkforward_folds.csv"}')

# ─── 3. Verify consistency ───
print('\n[3] Consistency checks...')
# Check no overlap between consecutive folds (ve1 = val_end of fold i, vs2 = val_start of fold i+1)
for i in range(len(splits)-1):
    _, _, _, _, ve1, _, _ = splits[i]
    _, _, _, vs2, _, _, _ = splits[i+1]
    assert ve1 == vs2, f'Fold gap: {ve1} != {vs2}'
# Test set starts where last val ends
_, _, _, _, ve_last, _, _ = splits[-1]
assert ve_last == test_start, f'Test gap: {ve_last} != {test_start}'
print('  All splits contiguous [OK]')

# Check target rate consistency
rates = [f['train_target_rate'] for f in fold_info if f['train_target_rate']]
rates_f = [float(r) for r in rates]
print(f'  Train target rate range: {min(rates_f):.4f} - {max(rates_f):.4f}')
print(f'  Val target rate range:   {[float(f["val_target_rate"]) for f in fold_info if f["val_target_rate"]]}')

# ─── 4. Summary ───
total_time = time.time() - t0
print(f'\n{"="*70}')
print('PHASE 9 COMPLETE')
print(f'Time: {total_time:.0f}s')
print(f'Dataset: improved_features.parquet ({len(all_features)} features, {len(keep_syms)} symbols)')
print(f'Folds: {len(splits)} walkforward folds + 1 test set')
print(f'Next: Phase 10 - Preprocessing (SMOTE + scaling)')
print(f'{"="*70}')

summary = {
    'phase': 9,
    'completed_at': datetime.now().isoformat(),
    'total_time': total_time,
    'dataset_file': 'improved_features.parquet',
    'n_symbols': len(keep_syms), 'n_features': len(all_features), 'n_interactions': len(interactions),
    'n_rows': len(df_improved),
    'n_folds': len(splits), 'folds': [f['fold'] for f in fold_info],
    'test_start': test_start, 'test_end': test_end,
}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
print(f'\nSummary: {OUT/"summary.json"}')
