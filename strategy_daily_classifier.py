"""
Binary classifier: predict which stocks will have next-day open-to-close > 2%.
Output: ranked list of stocks with probability scores.
"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'high_gainer_classifier'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' High Gainer Classifier (next-day open-close > 2%)')
print('='*60)

con = duckdb.connect(str(DB), read_only=True)

# ─── Step 1: Load daily data ───
print('\n[1] Loading daily features...')
fs = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume,
           sma_5, sma_10, sma_20, ema_5, ema_10, ema_20,
           rsi_14, macd_line, macd_signal, macd_hist, adx,
           plus_di, minus_di, atr_14, bb_pct_b, bb_width,
           obv, cmf, stoch_k, stoch_d, williams_r, mfi, uo, cci,
           trix, roc_5, roc_10, roc_20, zscore_20, hv_20,
           eom, fi, vpt, swing_high, swing_low,
           ret_1d, close_vs_sma_10, close_vs_sma_20,
           range_5, range_10, range_20, vol_ratio_10
    FROM feature_store WHERE timeframe='1day'
    ORDER BY symbol, datetime
""").fetchdf()
print(f'  Loaded: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# Penny filter
daily_prices = con.execute("""
    SELECT symbol, AVG(close) as avg_close FROM feature_store
    WHERE timeframe='1day' AND datetime >= '2024-01-01'
    GROUP BY symbol
""").fetchdf()
penny_syms = set(daily_prices[daily_prices['avg_close'] < 50]['symbol'])
fs = fs[~fs['symbol'].isin(penny_syms)].copy()
print(f'  After penny filter: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# ─── Step 2: Target (binary) ───
print('\n[2] Computing target...')
fs['date'] = pd.to_datetime(fs['datetime']).dt.normalize()
# Next day's open-to-close return
fs['next_close'] = fs.groupby('symbol')['close'].shift(-1)
fs['next_open'] = fs.groupby('symbol')['open'].shift(-1)
fs['target_ret'] = fs['next_close'] / fs['next_open'] - 1
fs = fs.dropna(subset=['target_ret']).copy()
fs['target'] = (fs['target_ret'] > 0.02).astype(int)
pos_rate = fs['target'].mean()
print(f'  Positive rate (>2% gainers): {pos_rate:.1%} ({fs["target"].sum():,}/{len(fs):,})')

# ─── Step 3: Features ───
print('\n[3] Preparing features...')
exclude = {'symbol', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume',
           'next_close', 'next_open', 'target_ret', 'target'}
feat_cols = [c for c in fs.columns if c not in exclude]
feat_cols = [c for c in feat_cols if fs[c].dtype in ('float64', 'int64', 'float32', 'int32')]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.5]
print(f'  Features: {len(feat_cols)}')

# ─── Step 4: Walkforward ───
print(f'\n[4] Walkforward training...')
fs['year'] = pd.to_datetime(fs['datetime']).dt.year
years = sorted(fs['year'].unique())
windows = [(fs['year'].isin(years[:i]), fs['year'] == years[i])
           for i in range(2, len(years))]
print(f'  Windows: {len(windows)}')

scale_pos = (1 - pos_rate) / pos_rate
print(f'  Scale pos weight: {scale_pos:.1f}')

results = []
last_valid = []; last_model = None

for wi, (train_mask, test_mask) in enumerate(windows):
    train = fs[train_mask].copy()
    test = fs[test_mask].copy()
    if len(test) < 500:
        continue

    test_start = test['datetime'].min()
    train = train[train['datetime'] < test_start - pd.Timedelta(days=7)].copy()

    valid = [c for c in feat_cols if train[c].notna().all() and train[c].std() > 1e-10]
    train = train.dropna(subset=valid)
    test = test.dropna(subset=valid)
    if len(train) < 500 or len(test) < 50:
        continue
    last_valid = valid

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[valid].values)
    X_te = scaler.transform(test[valid].values)
    y_tr = train['target'].values; y_te = test['target'].values

    model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=scale_pos,
                              random_state=42, n_jobs=-1, verbosity=0,
                              device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    last_model = model
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)

    yr_str = test['year'].iloc[0] if 'year' in test.columns else str(wi)
    precision = ((pred == 1) & (y_te == 1)).sum() / max((pred == 1).sum(), 1)
    recall = ((pred == 1) & (y_te == 1)).sum() / max((y_te == 1).sum(), 1)
    n_pos_day = y_te.sum()
    print(f'  [{wi+1:2d}/{len(windows)}] {yr_str}: '
          f'Prec={precision:.1%} Rec={recall:.1%} n_pos={n_pos_day}/{len(y_te)} ({y_te.mean():.1%})')

    for i in range(len(test)):
        results.append({
            'dt': test['datetime'].iloc[i], 'sym': test['symbol'].iloc[i],
            'prob': prob[i], 'pred': pred[i], 'act': y_te[i],
            'target_ret': test['target_ret'].iloc[i],
        })

# ─── Step 5: Overall metrics ───
print(f'\n[5] Overall Results')
rd = pd.DataFrame(results)
y_true, y_prob = rd['act'].values, rd['prob'].values
from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
print(f'  Predictions: {len(rd):,}  Pos rate: {y_true.mean():.1%}')
print(f'  ROC-AUC: {auc:.4f}  Avg Precision: {ap:.4f}')

# Top-K hit rate per day
daily_hits = rd.groupby('dt').apply(
    lambda d: d.nlargest(10, 'prob')['act'].sum() / max(d['act'].sum(), 1)).reset_index()
daily_hits.columns = ['dt', 'hit_rate']
print(f'  Avg top-10 hit rate (fraction of actual gainers captured): {daily_hits["hit_rate"].mean():.1%}')

# Feature importance
try:
    imp = pd.DataFrame({'feature': last_valid, 'importance': last_model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print('\n  Top 15 features:')
    for _, r in imp.head(15).iterrows():
        print(f'    {r["feature"]:<30s} {r["importance"]:.4f}')
except Exception:
    imp = pd.DataFrame()

# ─── Step 6: Example output ───
print(f'\n[6] Sample daily output (last day in dataset):')
last_date = rd['dt'].max()
today = last_date
last_day = rd[rd['dt'] == last_date].sort_values('prob', ascending=False).head(20)
last_day['expected_ret'] = last_day['target_ret']
print(f'  Date: {pd.Timestamp(last_date).date()}  Predictions for next day')
print(f'  {"Rank":>4s} {"Symbol":<12s} {"Prob>2%":>8s} {"Actual ret":>10s} {"Hit":>4s}')
print(f'  {"-"*40}')
for i, (_, r) in enumerate(last_day.iterrows()):
    hit = 'YES' if r['act'] == 1 else ''
    print(f'  {i+1:>4d} {r["sym"]:<12s} {r["prob"]:>7.1%} {r["target_ret"]:>+9.2%} {hit:>4s}')

# Save
pickle.dump({'rd': rd, 'imp': imp,
             'feat_cols': feat_cols, 'valid': last_valid,
             'auc': auc, 'ap': ap}, open(OUT / 'classifier_results.pkl', 'wb'))

print(f'\n{"="*60}')
print(f'  Files saved to {OUT}')
print(f'  Time: {time.time()-t0:.0f}s')
