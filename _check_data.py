import pandas as pd, numpy as np
from pathlib import Path
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')

fd = pd.read_parquet(BASE / 'feature_data.parquet')
print('=== feature_data.parquet ===')
print(f'Shape: {fd.shape}')
print(f'Symbols: {fd["symbol"].nunique()}')
print(f'Date range: {fd["datetime"].min()} to {fd["datetime"].max()}')
target_cols = [c for c in fd.columns if 'label' in c or 'target' in c or 'fwd' in c]
for tc in target_cols:
    if tc in fd.columns:
        print(f'{tc} rate: {fd[tc].mean():.4f} (non-null: {fd[tc].notna().sum()})')
print()

reg = pd.read_csv(BASE / 'ts_analysis_output' / 'regime_data.csv')
print('=== regime_data.csv ===')
print(f'Shape: {reg.shape}')
print(f'Columns: {list(reg.columns)}')
print(f'Regime distribution:')
print(reg['regime'].value_counts().sort_index())
print(f'Date range: {reg["date"].min()} to {reg["date"].max()}')
