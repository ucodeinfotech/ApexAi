# Phase 7 - Feature Engineering
# Adds: cross-sectional percentile ranks, feature-specific rolling windows,
#       GMM regime labels, day-of-week/month dummies, lagged features.
# Output: engineered_features.parquet
import duckdb, pandas as pd, numpy as np, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'engineered_features.parquet'
t0 = time.time()

print('=' * 60)
print(' Phase 7 — Feature Engineering')
print('=' * 60)

con = duckdb.connect(str(DB), read_only=True)

# ── 1. Load daily features from feature_store (full universe) ──
print('\n[1] Loading daily features (full universe)...')
fs = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume,
           ret_1d, ret_5d, log_ret_1d,
           range_5, range_10, range_20,
           hv_20, hv_10,
           vol_ratio_5, vol_ratio_10, vol_ratio_20,
           bb_width, bb_pct_b,
           atr_14, rsi_14, adx, macd_hist,
           sma_5, sma_10, sma_20, ema_5, ema_10, ema_20,
           close_vs_sma_10, close_vs_sma_20,
           zscore_20, skew_20, kurt_20,
           obv, cmf, mfi, stoch_k, stoch_d
    FROM feature_store WHERE timeframe='1day'
    ORDER BY symbol, datetime
""").fetchdf()
print(f'  Loaded: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# ── 2. Penny filter ──
print('\n[2] Penny filter...')
daily_prices = con.execute("""
    SELECT symbol, AVG(close) as avg_close FROM feature_store
    WHERE timeframe='1day' AND datetime >= '2024-01-01'
    GROUP BY symbol
""").fetchdf()
penny_syms = set(daily_prices[daily_prices['avg_close'] < 50]['symbol'])
print(f'  Penny symbols (<50 avg since 2024): {len(penny_syms)}')
fs = fs[~fs['symbol'].isin(penny_syms)].copy()
print(f'  After filter: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

con.close()

# Ensure datetime is tz-naive
fs['datetime'] = pd.to_datetime(fs['datetime'])
if fs['datetime'].dt.tz is not None:
    fs['datetime'] = fs['datetime'].dt.tz_localize(None)
fs['date'] = fs['datetime'].dt.normalize()

# Sort for window/lag ops
fs = fs.sort_values(['symbol', 'datetime']).reset_index(drop=True)

# ── 3. Cross-sectional percentile ranks (per date) ──
print('\n[3] Cross-sectional percentile ranks...')
rank_features = ['hv_20', 'range_5', 'bb_width', 'vol_ratio_5']
for feat in rank_features:
    if feat in fs.columns:
        col = f'rank_{feat}'
        # Percentile rank: 0..1 within each date
        fs[col] = fs.groupby('date')[feat].rank(pct=True)
        print(f'  {col}: {fs[col].min():.2f}–{fs[col].max():.2f}')
    else:
        print(f'  WARNING: {feat} not found, skipping')

# ── 4. Feature-specific rolling windows ──
print('\n[4] Feature-specific rolling windows...')
# ret_1d: 5-day mean + std (momentum trend & stability)
fs['ret_1d_ma_5'] = fs.groupby('symbol')['ret_1d'].transform(lambda x: x.rolling(5, min_periods=3).mean())
fs['ret_1d_std_5'] = fs.groupby('symbol')['ret_1d'].transform(lambda x: x.rolling(5, min_periods=3).std())
print('  ret_1d_ma_5, ret_1d_std_5')

# range_5: 21-day mean
fs['range_5_ma_21'] = fs.groupby('symbol')['range_5'].transform(lambda x: x.rolling(21, min_periods=10).mean())
print('  range_5_ma_21')

# vol_ratio_5: 30-day mean
fs['vol_ratio_5_ma_30'] = fs.groupby('symbol')['vol_ratio_5'].transform(lambda x: x.rolling(30, min_periods=15).mean())
print('  vol_ratio_5_ma_30')

# hv_20: 3-day mean + std
fs['hv_20_ma_3'] = fs.groupby('symbol')['hv_20'].transform(lambda x: x.rolling(3, min_periods=2).mean())
fs['hv_20_std_3'] = fs.groupby('symbol')['hv_20'].transform(lambda x: x.rolling(3, min_periods=2).std())
print('  hv_20_ma_3, hv_20_std_3')

# ── 5. Regime labels from GMM clustering ──
print('\n[5] GMM regime labels...')
regime_df = pd.read_csv(BASE / 'ts_analysis_output' / 'regime_data.csv')
regime_df['date'] = pd.to_datetime(regime_df['date'])
regime_df = regime_df[['date', 'regime']].copy()
# One-hot encode regimes
for r in range(5):
    regime_df[f'regime_{r}'] = (regime_df['regime'] == r).astype(int)
regime_df = regime_df.drop(columns=['regime'])

fs = fs.merge(regime_df, on='date', how='left')
for c in [f'regime_{r}' for r in range(5)]:
    fs[c] = fs[c].fillna(0)
print(f'  Merged {len(regime_df)} regime rows')
print(f'  Regime distribution:')
for c in [f'regime_{r}' for r in range(5)]:
    print(f'    {c}: {fs[c].mean():.3f}')

# ── 6. Day-of-week + month ──
print('\n[6] Temporal features...')
fs['dow'] = fs['date'].dt.dayofweek  # 0=Mon
fs['month'] = fs['date'].dt.month
for d in range(5):
    fs[f'dow_{d}'] = (fs['dow'] == d).astype(int)
for m in range(1, 13):
    fs[f'month_{m}'] = (fs['month'] == m).astype(int)
print('  dow_0..4, month_1..12 added')

# ── 7. Lagged features (t-1, t-2, t-3) ──
print('\n[7] Lagged features...')
lag_features = ['ret_1d', 'range_5', 'vol_ratio_5']
for feat in lag_features:
    if feat not in fs.columns:
        continue
    for lag in [1, 2, 3]:
        col = f'{feat}_lag{lag}'
        fs[col] = fs.groupby('symbol')[feat].shift(lag)
        print(f'  {col}')
# Also lag the target's drivers
for feat in ['ret_5d', 'hv_20', 'bb_width']:
    if feat not in fs.columns:
        continue
    col = f'{feat}_lag1'
    fs[col] = fs.groupby('symbol')[feat].shift(1)
    print(f'  {col}')

# ── 8. Target ──
print('\n[8] Computing target (next-day open-to-close > 2%)...')
fs['next_close'] = fs.groupby('symbol')['close'].shift(-1)
fs['next_open'] = fs.groupby('symbol')['open'].shift(-1)
fs['target_ret'] = fs['next_close'] / fs['next_open'] - 1
fs['target'] = (fs['target_ret'] > 0.02).astype(int)
fs = fs.dropna(subset=['target_ret']).copy()
pos_rate = fs['target'].mean()
print(f'  Positive rate: {pos_rate:.1%} ({fs["target"].sum():,}/{len(fs):,})')

# ── 9. Clean & save ──
print('\n[9] Finalizing...')
# Drop columns that shouldn't be used as features
drop_cols = []
# Drop features if >50% missing
for c in fs.columns:
    if fs[c].dtype in ('float64', 'float32', 'int64', 'int32'):
        missing_pct = fs[c].isna().mean()
        if missing_pct > 0.5:
            drop_cols.append(c)
            
# Fill remaining NaN
fs = fs.drop(columns=drop_cols)
for c in fs.columns:
    if c not in ('symbol', 'date', 'datetime'):
        if fs[c].dtype in ('float64', 'float32'):
            fs[c] = fs[c].fillna(0).astype(np.float32)

print(f'  Dropped {len(drop_cols)} columns with >50% missing')
print(f'  Final shape: {fs.shape}')
print(f'  Final columns ({len(fs.columns)}):')
id_cols = ['symbol', 'date', 'datetime']
feat_cols = [c for c in fs.columns if c not in id_cols + ['target', 'target_ret', 'next_close', 'next_open']]
print(f'  Identity: {id_cols}')
print(f'  Features: {len(feat_cols)}')
print(f'  Target & metadata: 4')

# Save
fs.to_parquet(OUT, index=False)
print(f'\n  Saved to: {OUT}')
print(f'  Size: {OUT.stat().st_size / 1e6:.1f} MB')

print(f'\n{"=" * 60}')
print(f'  Phase 7 complete in {time.time()-t0:.0f}s')
print(f'  Ready for Phase 8: Feature Selection')
print(f'{"=" * 60}')
