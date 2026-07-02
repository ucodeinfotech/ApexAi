import pandas as pd, numpy as np
from pathlib import Path
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'eda_results'

# Re-run sections 6-9
ef = pd.read_parquet(BASE / 'engineered_features.parquet')
feat_cols = [c for c in ef.columns if c not in ('symbol', 'datetime', 'date', 'target', 'target_ret', 'next_close', 'next_open', 'year')]

# Continue from high_corr_pairs
high_corr_df = pd.read_csv(OUT / 'high_correlation_pairs.csv')
print(f'Highly correlated pairs (r>0.95): {len(high_corr_df)}')
for _, r in high_corr_df.head(10).iterrows():
    print(f'  {r["feat1"]:<25s} X {r["feat2"]:<25s} r={r["corr"]:.4f}')

# Quintile analysis
print('\nFeature-Target Relationship (quintile analysis)')
quint_df = pd.read_csv(OUT / 'quintile_analysis.csv')
print(f'Top 15 features by quintile spread:')
for _, r in quint_df.head(15).iterrows():
    print(f'  {r["feature"]:<30s} q1={r["q1_rate"]:.1%}  q5={r["q5_rate"]:.1%}  spread={r["spread"]:.1%}')

# Regime analysis
print('\nRegime-specific analysis:')
regime_cols = [c for c in feat_cols if c.startswith('regime_')]
for rc in regime_cols:
    mask = ef[rc] == 1
    n = mask.sum()
    rate = ef.loc[mask, 'target'].mean() if n > 0 else 0
    print(f'  {rc}: n={n:>6,}  gainer_rate={rate:.1%}')
