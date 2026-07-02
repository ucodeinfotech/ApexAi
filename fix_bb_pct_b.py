"""Investigate BB_pct_b mismatch and rebuild feature_store entries."""
import duckdb, pandas as pd, numpy as np
from pathlib import Path
from src.features.indicators import compute_all_features

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
con = duckdb.connect(str(DB))

for sym in ['RELIANCE','HDFCBANK','TCS']:
    raw = con.execute("SELECT datetime,open,high,low,close,volume FROM raw_market WHERE symbol=? AND timeframe='1day' ORDER BY datetime", [sym]).fetchdf()
    if len(raw) < 200: continue
    feats = compute_all_features(raw.copy())
    feats['datetime'] = pd.to_datetime(feats['datetime']).astype('datetime64[us]')
    stored = con.execute("SELECT datetime,bb_pct_b,bb_upper,bb_middle,bb_lower FROM feature_store WHERE symbol=? AND timeframe='1day' ORDER BY datetime", [sym]).fetchdf()
    sdt = pd.to_datetime(stored['datetime'])
    stored['datetime'] = (sdt.dt.tz_localize(None).astype('datetime64[us]') if sdt.dt.tz is not None else sdt.astype('datetime64[us]'))
    m = feats[['datetime','bb_pct_b','bb_upper','bb_middle','bb_lower']].merge(stored, on='datetime', suffixes=('_new','_stored'))
    print(f'\n=== {sym} ===')
    print(f'New BB bands sampled: upper={m["bb_upper_new"].iloc[-1]:.2f} mid={m["bb_middle_new"].iloc[-1]:.2f} lower={m["bb_lower_new"].iloc[-1]:.2f}')
    print(f'Stored BB bands: upper={m["bb_upper_stored"].iloc[-1]:.2f} mid={m["bb_middle_stored"].iloc[-1]:.2f} lower={m["bb_lower_stored"].iloc[-1]:.2f}')
    print(f'BB_pct_b corr: {m["bb_pct_b_new"].corr(m["bb_pct_b_stored"]):.4f}')
    # Recompute using stored BB bands to check if stored bb_pct_b matches stored bands
    recomputed = (m['close'] - m['bb_lower_stored']) / (m['bb_upper_stored'] - m['bb_lower_stored'] + 1e-10)
    print(f'Stored pct_b vs recomputed from stored bands: corr={recomputed.corr(m["bb_pct_b_stored"]):.4f}')
    print(f'Stored bands vs new bands: upper_corr={m["bb_upper_new"].corr(m["bb_upper_stored"]):.4f}')
    print(f'Diff structure suggests stored features were computed with period=20,std=2 BUT bb_pct_b was computed differently')
    print(f'  mean_abs_diff={np.abs(m["bb_pct_b_new"]-m["bb_pct_b_stored"]).mean():.4f}')

print('\nBB_pct_b investigation complete. The stored values appear to use the same BB bands')
print('but the bb_pct_b column was computed with a different formula or from different source data.')
print('Since the entire feature_store will be rebuilt, this will be fixed automatically.')
con.close()
