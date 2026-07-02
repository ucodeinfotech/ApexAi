"""
High Gainer Classifier v2: predict next-day open-to-close > 2%.
Adds: intraday 60min features, VIX regime, delivery data, tuned params.
"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'high_gainer_classifier_v2'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' High Gainer Classifier v2')
print('='*60)

con = duckdb.connect(str(DB), read_only=True)

# ─── Step 1: Daily features ───
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
fs['date'] = pd.to_datetime(fs['datetime']).dt.normalize()
print(f'  After penny filter: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# ─── Step 2: Intraday (60min) features aggregated to daily ───
print('\n[2] Computing intraday features from 60min...')
m60 = con.execute("""
    SELECT symbol, datetime, datetime::DATE as date, close, high, low, volume
    FROM raw_market WHERE timeframe='60min'
    ORDER BY symbol, datetime
""").fetchdf()
m60['datetime'] = pd.to_datetime(m60['datetime'])
m60['hour'] = m60['datetime'].dt.hour
m60['ret_60min'] = m60.groupby('symbol')['close'].transform(lambda x: x.pct_change() * 100)
m60['range_60min'] = (m60['high'] - m60['low']) / m60['close'] * 100

# Per symbol+date: aggregate 60min bars
id_feats = m60.groupby(['symbol', 'date']).agg(
    intra_morn_ret=('ret_60min', lambda x: x.iloc[0] if len(x) > 0 else 0),
    intra_max_gain=('ret_60min', lambda x: x.max() if len(x) > 0 else 0),
    intra_max_loss=('ret_60min', lambda x: x.min() if len(x) > 0 else 0),
    intra_volatility=('range_60min', 'mean'),
    intra_range_total=('range_60min', 'sum'),
    intra_vol_sum=('volume', 'sum'),
    intra_n_bars=('close', 'count'),
).reset_index()
# Last hour return (15:00 bar = hour 14 in IST, or 14:00 bar = hour 14)
m60_last = m60[m60['hour'] == m60.groupby(['symbol', 'date'])['hour'].transform('max')]
last_ret = m60_last.groupby(['symbol', 'date'])['ret_60min'].last().reset_index()
last_ret.columns = ['symbol', 'date', 'intra_last_ret']
id_feats = id_feats.merge(last_ret, on=['symbol', 'date'], how='left')
id_feats = id_feats.fillna(0)
print(f'  Computed for {len(id_feats):,} symbol-days ({id_feats["symbol"].nunique()} symbols)')

# ─── Step 3: VIX regime ───
print('\n[3] Loading VIX & market regime...')
vix = con.execute("""
    SELECT datetime::DATE as date, vix_close, vix_change, vix_ma_20, vix_ratio_5, vix_zscore_20
    FROM vix_data ORDER BY datetime
""").fetchdf()
vix['date'] = pd.to_datetime(vix['date'])
vix['vix_high_regime'] = (vix['vix_zscore_20'] > 1.5).astype(float)

regimes = con.execute("""
    SELECT datetime::DATE as date, regime_label, volatility_regime
    FROM market_regimes WHERE timeframe='1day'
    ORDER BY datetime
""").fetchdf()
regimes['date'] = pd.to_datetime(regimes['date'])
regimes['is_bull'] = (regimes['regime_label'] == 'bull').astype(float)
regimes['is_bear'] = (regimes['regime_label'] == 'bear').astype(float)
regimes['is_high_vol'] = (regimes['volatility_regime'] == 'high_vol').astype(float)

# ─── Step 4: Delivery data ───
print('  Loading delivery data...')
delivery = con.execute("""
    SELECT symbol, date, delivery_pct FROM delivery_data ORDER BY symbol, date
""").fetchdf()
delivery['date'] = pd.to_datetime(delivery['date'])
delivery['delivery_ma5'] = delivery.groupby('symbol')['delivery_pct'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())
delivery['delivery_high'] = (delivery['delivery_pct'] > 75).astype(float)
print(f'  Delivery data: {len(delivery):,} rows ({delivery["symbol"].nunique()} symbols)')

# ─── Step 5: Merge everything ───
print('\n[5] Merging all features...')
con.register('fs_d', fs)
con.register('idf', id_feats)
con.register('vix_tbl', vix)
con.register('reg_tbl', regimes)
con.register('del_tbl', delivery)

fs = con.execute("""
    SELECT fs_d.*,
           idf.intra_morn_ret, idf.intra_max_gain, idf.intra_max_loss,
           idf.intra_volatility, idf.intra_last_ret, idf.intra_vol_sum,
           vix_tbl.vix_close, vix_tbl.vix_change, vix_tbl.vix_ratio_5, vix_tbl.vix_zscore_20, vix_tbl.vix_high_regime,
           reg_tbl.is_bull, reg_tbl.is_bear, reg_tbl.is_high_vol,
           del_tbl.delivery_pct, del_tbl.delivery_ma5, del_tbl.delivery_high
    FROM fs_d
    LEFT JOIN idf ON fs_d.symbol = idf.symbol AND fs_d.date = idf.date
    LEFT JOIN vix_tbl ON fs_d.date = vix_tbl.date
    LEFT JOIN reg_tbl ON fs_d.date = reg_tbl.date
    LEFT JOIN del_tbl ON fs_d.symbol = del_tbl.symbol AND fs_d.date = del_tbl.date
    ORDER BY fs_d.symbol, fs_d.datetime
""").fetchdf()

for c in fs.columns:
    if fs[c].dtype in ('float64', 'float32'):
        fs[c] = fs[c].fillna(0)

con.unregister('fs_d'); con.unregister('idf'); con.unregister('vix_tbl')
con.unregister('reg_tbl'); con.unregister('del_tbl')

# VIX/regime already aligned on same date (correct: today's VIX predicts tomorrow)

print(f'  Merged shape: {fs.shape}')

# ─── Step 6: Target ───
print('\n[6] Computing target...')
fs['next_close'] = fs.groupby('symbol')['close'].shift(-1)
fs['next_open'] = fs.groupby('symbol')['open'].shift(-1)
fs['target_ret'] = fs['next_close'] / fs['next_open'] - 1
fs = fs.dropna(subset=['target_ret']).copy()
fs['target'] = (fs['target_ret'] > 0.02).astype(int)
pos_rate = fs['target'].mean()
print(f'  Positive rate (>2% gainers): {pos_rate:.1%} ({fs["target"].sum():,}/{len(fs):,})')

# ─── Step 7: Feature selection ───
print('\n[7] Preparing features...')
exclude = {'symbol', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume',
           'next_close', 'next_open', 'target_ret', 'target'}
feat_cols = [c for c in fs.columns if c not in exclude]
feat_cols = [c for c in feat_cols if fs[c].dtype in ('float64', 'int64', 'float32', 'int32')]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.5]
print(f'  Features: {len(feat_cols)}')

n_new = sum(1 for c in feat_cols if c.startswith('intra_') or c.startswith('vix_') or c.startswith('is_') or c.startswith('delivery_'))
print(f'  New features from v2: {n_new}')

# ─── Step 8: Walkforward ───
print(f'\n[8] Walkforward training...')
fs['year'] = pd.to_datetime(fs['datetime']).dt.year
years = sorted(fs['year'].unique())
windows = [(fs['year'].isin(years[:i]), fs['year'] == years[i])
           for i in range(2, len(years))]

scale_pos = (1 - pos_rate) / max(pos_rate, 0.001)
print(f'  Windows: {len(windows)}  Scale pos weight: {scale_pos:.1f}')

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

    # Also train a regressor for comparison
    ret_tr = train['target_ret'].values; ret_te = test['target_ret'].values

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.01, reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1, verbosity=0,
        device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    last_model = model
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)

    # Also try regressor threshold approach
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.01, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
        device='cuda', tree_method='hist')
    reg.fit(X_tr, ret_tr, eval_set=[(X_te, ret_te)], verbose=False)
    reg_pred = reg.predict(X_te)
    reg_class = (reg_pred > 0.02).astype(int)

    yr_str = test['year'].iloc[0]
    prec = ((pred == 1) & (y_te == 1)).sum() / max((pred == 1).sum(), 1)
    rec = ((pred == 1) & (y_te == 1)).sum() / max((y_te == 1).sum(), 1)
    reg_prec = ((reg_class == 1) & (y_te == 1)).sum() / max((reg_class == 1).sum(), 1)
    print(f'  [{wi+1:2d}/{len(windows)}] {yr_str}: '
          f'Clf Prec={prec:.1%} Rec={rec:.1%} | Reg Prec={reg_prec:.1%} '
          f'n_pos={y_te.sum()}/{len(y_te)} ({y_te.mean():.1%})')

    for i in range(len(test)):
        results.append({
            'dt': test['datetime'].iloc[i], 'sym': test['symbol'].iloc[i],
            'prob': prob[i], 'pred': pred[i], 'act': y_te[i],
            'ret': test['target_ret'].iloc[i],
            'reg_pred': reg_pred[i],
        })

# ─── Step 9: Overall ───
print(f'\n[9] Overall Results')
rd = pd.DataFrame(results)
y_true, y_prob = rd['act'].values, rd['prob'].values
y_reg = rd['reg_pred'].values
auc = roc_auc_score(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
reg_auc = roc_auc_score(y_true, y_reg)
reg_ap = average_precision_score(y_true, y_reg)
print(f'  Predictions: {len(rd):,}  Pos rate: {y_true.mean():.1%}')
print(f'  Classifier:  ROC-AUC={auc:.4f}  Avg Precision={ap:.4f}')
print(f'  Regressor:   ROC-AUC={reg_auc:.4f}  Avg Precision={reg_ap:.4f}')

# Top-K analysis
print(f'\n  Top-K analysis:')
for k in [5, 10, 20, 50]:
    topk_hits = rd.groupby('dt').apply(
        lambda d: d.nlargest(k, 'prob')['act'].sum())
    actual = rd.groupby('dt')['act'].sum()
    hit_rate = (topk_hits / actual.replace(0, np.nan)).dropna()
    print(f'    Top {k:>3d}: captures {hit_rate.mean():.1%} of gainers ({topk_hits.mean():.1f}/day)')

# Feature importance
try:
    imp = pd.DataFrame({'feature': last_valid, 'importance': last_model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print('\n  Top 20 features:')
    for _, r in imp.head(20).iterrows():
        print(f'    {r["feature"]:<30s} {r["importance"]:.4f}')
except Exception:
    imp = pd.DataFrame()

# Sample
print(f'\n[10] Sample output (last date):')
last_date = rd['dt'].max()
last_day = rd[rd['dt'] == last_date].sort_values('prob', ascending=False).head(20)
print(f'  {pd.Timestamp(last_date).date()} -> next day predictions')
print(f'  {"Rank":>4s} {"Symbol":<12s} {"Prob>2%":>8s} {"Reg ret":>8s} {"Actual ret":>10s} {"Hit":>4s}')
print(f'  {"-"*48}')
for i, (_, r) in enumerate(last_day.iterrows()):
    hit = 'YES' if r['act'] == 1 else ''
    reg_str = f'{r["reg_pred"]:>+7.2%}' if 'reg_pred' in r and not np.isnan(r.get('reg_pred', 0)) else '       N/A'
    ret_str = f'{r["ret"]:>+9.2%}' if not np.isnan(r.get('ret', 0)) else '       N/A'
    print(f'  {i+1:>4d} {r["sym"]:<12s} {r["prob"]:>7.1%} {reg_str} {ret_str} {hit:>4s}')

pickle.dump({'rd': rd, 'imp': imp, 'feat_cols': feat_cols, 'valid': last_valid,
             'auc': auc, 'ap': ap, 'reg_auc': reg_auc, 'reg_ap': reg_ap},
            open(OUT / 'results_v2.pkl', 'wb'))

print(f'\n{"="*60}')
print(f'  v2 summary: AUC={auc:.4f} (v1: 0.625)  AP={ap:.4f} (v1: 0.187)')
print(f'  Files saved to {OUT}')
print(f'  Time: {time.time()-t0:.0f}s')
