import pandas as pd, numpy as np
from pathlib import Path
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'eda_results'

ef = pd.read_parquet(BASE / 'engineered_features.parquet')
target_col = 'target'
feat_cols = [c for c in ef.columns if c not in ('symbol', 'datetime', 'date', target_col, 'target_ret', 'next_close', 'next_open', 'year')]

# Quintile analysis
print('Quintile Analysis...')
quint_results = []
for c in feat_cols:
    if ef[c].dtype not in ('float64', 'float32'):
        continue
    if ef[c].nunique() < 10:
        continue
    s = ef[c].dropna()
    if len(s) < 100:
        continue
    try:
        ef['_bin'] = pd.qcut(ef[c], 5, labels=False, duplicates='drop')
        if ef['_bin'].nunique() < 2:
            continue
        bin_rates = ef.groupby('_bin')[target_col].mean()
        spread = float(bin_rates.max() - bin_rates.min())
        quint_results.append({
            'feature': c,
            'q1_rate': float(bin_rates.iloc[0]) if len(bin_rates) >= 1 else 0,
            'q5_rate': float(bin_rates.iloc[-1]) if len(bin_rates) >= 5 else 0,
            'spread': spread,
        })
    except:
        pass

quint_df = pd.DataFrame(quint_results).sort_values('spread', ascending=False)
quint_df.to_csv(OUT / 'quintile_analysis.csv', index=False)
print(f'Top 15 by quintile spread:')
for _, r in quint_df.head(15).iterrows():
    print(f'  {r["feature"]:<30s} q1={r["q1_rate"]:.1%}  q5={r["q5_rate"]:.1%}  spread={r["spread"]:.1%}')

# Feature-target correlation
print('\nFeature-Target Correlation:')
corr_results = []
for c in feat_cols:
    if ef[c].dtype not in ('float64', 'float32', 'int64', 'int32'):
        continue
    if ef[c].nunique() < 2:
        continue
    corr = ef[c].corr(ef[target_col])
    if not np.isnan(corr):
        corr_results.append({'feature': c, 'corr_with_target': float(corr)})
corr_df = pd.DataFrame(corr_results).sort_values('corr_with_target', key=abs, ascending=False)
corr_df.to_csv(OUT / 'target_correlation.csv', index=False)
print(f'Top 20:')
for _, r in corr_df.head(20).iterrows():
    print(f'  {r["feature"]:<30s} corr={r["corr_with_target"]:+.4f}')

# Regime analysis
print('\nRegime-specific analysis:')
regime_cols = [c for c in feat_cols if c.startswith('regime_')]
for rc in regime_cols:
    mask = ef[rc] == 1
    n = mask.sum()
    rate = ef.loc[mask, target_col].mean() if n > 0 else 0
    print(f'  {rc}: n={n:>8,} ({n/len(ef)*100:.1f}%)  gainer_rate={rate:.1%}')

# Class imbalance
print(f'\nClass imbalance:')
print(f'  Non-gainers: {(1-ef[target_col]).sum():,} ({ef[target_col].mean()*100:.1f}%)')
print(f'  Gainers:     {ef[target_col].sum():,} ({ef[target_col].mean()*100:.1f}%)')
print(f'  Ratio:       {len(ef)/ef[target_col].sum():.1f}:1')

# Yearly stats
ef['year'] = pd.to_datetime(ef['datetime']).dt.year
yr = ef.groupby('year')[target_col].agg(['mean', 'sum', 'count'])
print(f'\nYearly:')
for y, r in yr.iterrows():
    print(f'  {int(y)}: {r["mean"]:.1%} (gainers={int(r["sum"])})')

# Cross-sectional rank features analysis
print('\nCross-sectional rank features (from Phase 7):')
rank_cols = [c for c in feat_cols if c.startswith('rank_')]
for rc in rank_cols:
    ef['_bin'] = pd.qcut(ef[rc], 5, labels=False, duplicates='drop')
    rates = ef.groupby('_bin')[target_col].mean()
    spread = rates.max() - rates.min()
    print(f'  {rc}: quintile rates: {[f"{rates.get(i, 0):.1%}" for i in range(5)]}  spread={spread:.1%}')
