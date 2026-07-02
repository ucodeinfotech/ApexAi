"""Optimized pipeline: DuckDB-heavy resample + features + selection + preprocess + train"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math, gc
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'report_resampled'; OUT.mkdir(exist_ok=True)
t0 = time.time()

print('=== STEP 1: DuckDB pre-processing (all heavy lifting in SQL) ===')
con = duckdb.connect(str(DB), read_only=True)

# 1a: Common non-penny symbols across 60min + 15min + 1day
con.execute("""
    CREATE TEMP VIEW common_syms AS
    SELECT DISTINCT s60.symbol
    FROM (SELECT DISTINCT symbol FROM raw_market WHERE timeframe='60min') s60
    JOIN (SELECT DISTINCT symbol FROM raw_market WHERE timeframe='15min') s15 ON s60.symbol = s15.symbol
    JOIN (SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day') s1d ON s60.symbol = s1d.symbol
""")

# Penny filter
penny_syms = con.execute("""
    SELECT symbol FROM raw_market
    WHERE timeframe='1day' AND datetime >= '2024-06-24'::TIMESTAMPTZ
    GROUP BY symbol
    HAVING AVG(close) < 50
""").fetchdf()['symbol'].tolist()
print(f'  Penny symbols: {len(penny_syms)}')

# 1b: 60min data (trading hours only, common symbols, no penny)
print('  Loading 60min data...')
q60 = """
    SELECT symbol, datetime, open, high, low, close, volume
    FROM raw_market
    WHERE timeframe='60min'
      AND symbol IN (SELECT symbol FROM common_syms)
      AND symbol NOT IN (%s)
      AND EXTRACT(HOUR FROM datetime) BETWEEN 4 AND 10
    ORDER BY datetime
""" % ','.join([f"'{s}'" for s in penny_syms])
raw60 = con.execute(q60).fetchdf()
print(f'    60min: {len(raw60):,} rows, {raw60["symbol"].nunique()} symbols')

# 1c: 15min rollup features via SQL
print('  Computing 15min rollup in SQL...')
q15rollup = """
    SELECT
        symbol,
        datetime::date AS date,
        EXTRACT(HOUR FROM datetime) AS hour,
        COUNT(*) AS n_bars_15m,
        (LAST(close ORDER BY datetime) / FIRST(open ORDER BY datetime) - 1) * 100 AS ret_sum_15m,
        MAX(high) AS high_max_15m,
        MIN(low) AS low_min_15m,
        SUM(volume) AS vol_sum_15m,
        (MAX(high) - MIN(low)) / NULLIF(LAST(close ORDER BY datetime), 0) * 100 AS range_pct_15m,
        SUM(CASE WHEN EXTRACT(MINUTE FROM datetime) < 30 THEN volume ELSE 0 END) AS vol_first_half_15m,
        SUM(CASE WHEN EXTRACT(MINUTE FROM datetime) >= 30 THEN volume ELSE 0 END) AS vol_second_half_15m
    FROM raw_market
    WHERE timeframe='15min'
      AND symbol IN (SELECT symbol FROM raw60 LIMIT 0)  -- will be replaced
    GROUP BY symbol, date, hour
"""
# Actually use raw60 symbol list
syms_60 = raw60['symbol'].unique().tolist()
sym_list = ','.join([f"'{s}'" for s in syms_60])

q15rollup = f"""
    SELECT symbol,
           CAST(datetime AT TIME ZONE 'Asia/Calcutta' AS DATE) AS bar_date,
           EXTRACT(HOUR FROM datetime AT TIME ZONE 'Asia/Calcutta') AS bar_hour,
           COUNT(*) AS n_bars_15m,
           (LAST(close ORDER BY datetime) / NULLIF(FIRST(open ORDER BY datetime), 0) - 1) * 100 AS ret_sum_15m,
           MAX(high) AS high_max_15m, MIN(low) AS low_min_15m,
           SUM(volume) AS vol_sum_15m,
           (MAX(high) - MIN(low)) / NULLIF(LAST(close ORDER BY datetime), 0) * 100 AS range_pct_15m,
           SUM(CASE WHEN EXTRACT(MINUTE FROM datetime AT TIME ZONE 'Asia/Calcutta') < 30 THEN volume ELSE 0 END) AS vol_first_15m,
           SUM(CASE WHEN EXTRACT(MINUTE FROM datetime AT TIME ZONE 'Asia/Calcutta') >= 30 THEN volume ELSE 0 END) AS vol_last_15m,
           FIRST(close ORDER BY datetime) AS first_close_15m,
           LAST(close ORDER BY datetime) AS last_close_15m
    FROM raw_market
    WHERE timeframe='15min'
      AND symbol IN ({sym_list})
      AND EXTRACT(HOUR FROM datetime AT TIME ZONE 'Asia/Calcutta') BETWEEN 9 AND 15
    GROUP BY symbol, bar_date, bar_hour
    ORDER BY symbol, bar_date, bar_hour
"""
rollup_15 = con.execute(q15rollup).fetchdf()
rollup_15['slope_15m'] = (rollup_15['last_close_15m'] / rollup_15['first_close_15m'].replace(0, np.nan) - 1) * 100
rollup_15['vol_profile_15m'] = rollup_15['vol_first_15m'] / rollup_15['vol_last_15m'].replace(0, 1)
rollup_15 = rollup_15.rename(columns={'bar_hour': 'hour_key', 'bar_date': 'date'})
print(f'    15min rollup: {len(rollup_15):,} rows')

# 1d: Daily context
print('  Loading daily context...')
# Daily returns + range
q1d = f"""
    SELECT symbol, CAST(datetime AT TIME ZONE 'Asia/Calcutta' AS DATE) AS bar_date,
           close AS close_d1,
           (close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY datetime), 0) - 1) * 100 AS ret_d1,
           (high - low) / NULLIF(close, 0) * 100 AS range_d1,
           volume AS volume_d1
    FROM raw_market
    WHERE timeframe='1day'
      AND symbol IN ({sym_list})
    ORDER BY symbol, datetime
"""
daily_ctx = con.execute(q1d).fetchdf()
# Shift context forward (use yesterday's data for today)
daily_ctx['date_next'] = pd.to_datetime(daily_ctx['bar_date']) + pd.Timedelta(days=1)
daily_ctx = daily_ctx.drop(columns=['bar_date']).rename(columns={'date_next': 'date'})
print(f'    Daily context: {len(daily_ctx):,} rows')

# 1e: Delivery, VIX, regimes
delivery = con.execute(f"""
    SELECT symbol, date::DATE AS date, delivery_pct
    FROM delivery_data
    WHERE symbol IN ({sym_list})
    ORDER BY symbol, date
""").fetchdf()
vix = con.execute("SELECT datetime, vix_close, vix_change, vix_ma_5, vix_ma_20, vix_zscore_20 FROM vix_data ORDER BY datetime").fetchdf()
regimes = con.execute("SELECT datetime, regime_label, regime_id FROM market_regimes ORDER BY datetime").fetchdf()
con.close()
gc.collect()

# --- Merge all in pandas ---
print('\n=== STEP 2: Merge in pandas ===')
dt60 = pd.to_datetime(raw60['datetime'])
raw60['datetime'] = dt60.dt.tz_localize(None) if dt60.dt.tz is not None else dt60
raw60['date'] = raw60['datetime'].dt.normalize()
raw60['hour'] = raw60['datetime'].dt.hour

# Merge 15min rollup
raw60['hour_key'] = raw60['hour']
raw60 = raw60.merge(rollup_15, left_on=['symbol','date','hour_key'], right_on=['symbol','date','hour_key'], how='left')
for c in ['n_bars_15m','ret_sum_15m','high_max_15m','low_min_15m','vol_sum_15m','range_pct_15m',
          'vol_first_15m','vol_last_15m','slope_15m','vol_profile_15m']:
    if c in raw60.columns:
        raw60[c] = raw60[c].fillna(0)
raw60 = raw60.drop(columns=['hour_key'])

# Merge daily context
raw60 = raw60.merge(daily_ctx, on=['symbol','date'], how='left')

# Delivery (shift forward 1 day)
delivery['date_dt'] = pd.to_datetime(delivery['date'])
delivery['date_next'] = delivery['date_dt'] + pd.Timedelta(days=1)
del_ctx = delivery[['symbol','date_next','delivery_pct']].rename(columns={'date_next': 'date'})
raw60 = raw60.merge(del_ctx, on=['symbol','date'], how='left')
raw60['delivery_pct'] = raw60.groupby('symbol')['delivery_pct'].ffill().fillna(0)

# VIX (fix dtype mismatch)
vix_dt = pd.to_datetime(vix['datetime'])
common_dtype = raw60['datetime'].dtype
vix['datetime'] = (vix_dt.dt.tz_localize(None) if vix_dt.dt.tz is not None else vix_dt).astype(common_dtype)
vix = vix.sort_values('datetime')
raw60 = pd.merge_asof(raw60.sort_values('datetime'), vix, on='datetime', direction='backward')
for c in ['vix_close','vix_change','vix_ma_5','vix_ma_20','vix_zscore_20']:
    raw60[c] = raw60[c].fillna(raw60[c].median() if raw60[c].notna().any() else 0)

# Regimes
reg_dt = pd.to_datetime(regimes['datetime'])
regimes['datetime'] = reg_dt.dt.tz_localize(None) if reg_dt.dt.tz is not None else reg_dt
raw60 = raw60.merge(regimes[['datetime','regime_label','regime_id']], on='datetime', how='left')
raw60['regime_label'] = raw60['regime_label'].fillna('sideways')
raw60['regime_id'] = raw60['regime_id'].fillna(0).astype(int)

# Fill daily context gaps
for c in ['close_d1','range_d1','ret_d1','volume_d1']:
    if c in raw60.columns:
        raw60[c] = raw60.groupby('symbol')[c].ffill().fillna(0)

del rollup_15, daily_ctx, delivery; gc.collect()
print(f'  Merged: {len(raw60):,} rows, {raw60["symbol"].nunique()} symbols')

print('\n=== STEP 3: Feature Engineering ===')
# Sort
raw60 = raw60.sort_values(['symbol','datetime']).reset_index(drop=True)

# Price returns
raw60['ret_1'] = raw60.groupby('symbol')['close'].transform(lambda x: x.pct_change()) * 100
raw60['ret_5'] = raw60.groupby('symbol')['close'].transform(lambda x: x.pct_change(5)) * 100
raw60['log_ret'] = np.log(raw60['close'] / raw60.groupby('symbol')['close'].shift(1)) * 100

# Candle features
raw60['range_pct'] = (raw60['high'] - raw60['low']) / raw60['close'] * 100
raw60['body_pct'] = abs(raw60['close'] - raw60['open']) / (raw60['high'] - raw60['low'] + 1e-10)
raw60['close_pos'] = (raw60['close'] - raw60['low']) / (raw60['high'] - raw60['low'] + 1e-10)

# Rolling windows — single groupby pass per window
for w in [3, 5, 10, 20]:
    g = raw60.groupby('symbol')
    raw60[f'sma_{w}'] = g['close'].transform(lambda x, ww=w: x.rolling(ww, min_periods=2).mean())
    raw60[f'std_{w}'] = g['close'].transform(lambda x, ww=w: x.rolling(ww, min_periods=2).std(ddof=0))
    raw60[f'zscore_{w}'] = (raw60['close'] - raw60[f'sma_{w}']) / raw60[f'std_{w}'].replace(0, np.nan)
    raw60[f'ret_std_{w}'] = g['ret_1'].transform(lambda x, ww=w: x.rolling(ww, min_periods=2).std(ddof=0))
    raw60[f'range_avg_{w}'] = g['range_pct'].transform(lambda x, ww=w: x.rolling(ww, min_periods=2).mean())
    raw60[f'vol_ma_{w}'] = g['volume'].transform(lambda x, ww=w: x.rolling(ww, min_periods=2).mean())
    raw60[f'vol_spike_{w}'] = raw60['volume'] / raw60[f'vol_ma_{w}'].replace(0, np.nan)

# Distance from SMA
for w in [3, 5, 10, 20]:
    raw60[f'close_vs_sma_{w}'] = (raw60['close'] / raw60[f'sma_{w}'] - 1) * 100

# RSI
def rsi(series, w=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(w, min_periods=2).mean()
    loss = (-delta.clip(upper=0)).rolling(w, min_periods=2).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))
raw60['rsi_14'] = raw60.groupby('symbol')['close'].transform(lambda x: rsi(x, 14))

# Bollinger
w = 20
ma = raw60.groupby('symbol')['close'].transform(lambda x: x.rolling(w, min_periods=2).mean())
sd = raw60.groupby('symbol')['close'].transform(lambda x: x.rolling(w, min_periods=2).std(ddof=0))
raw60['bb_pct_b'] = (raw60['close'] - ma + 2*sd) / (4*sd + 1e-10)
raw60['bb_width'] = 2 * sd / ma.replace(0, np.nan) * 100

# ATR (simplified — no groupby apply)
hl = raw60['high'] - raw60['low']
hc = abs(raw60['high'] - raw60.groupby('symbol')['close'].shift(1))
lc = abs(raw60['low'] - raw60.groupby('symbol')['close'].shift(1))
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
raw60['atr_14'] = raw60.groupby('symbol')['close'].transform(lambda x: pd.Series(tr.values).rolling(14, min_periods=2).mean())

# VWAP
raw60['vwap_val'] = (raw60['high'] + raw60['low'] + raw60['close']) / 3 * raw60['volume']
raw60['vwap_cum'] = raw60.groupby('symbol')['vwap_val'].cumsum()
raw60['vol_cum'] = raw60.groupby('symbol')['volume'].cumsum()
raw60['vwap_day'] = raw60['vwap_cum'] / raw60['vol_cum'].replace(0, np.nan)
raw60['vwap_dist'] = (raw60['close'] / raw60['vwap_day'] - 1) * 100

# Target
raw60['target'] = raw60.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * 100
raw60 = raw60.dropna(subset=['target'])

# Drop temp columns
drop_tmp = [c for c in raw60.columns if c in ['vwap_val','vwap_cum','vol_cum']]
raw60 = raw60.drop(columns=drop_tmp)
gc.collect()
print(f'  Features done: {len(raw60):,} rows')

print('\n=== STEP 4: Preprocessing ===')
EXCLUDE = {'symbol','datetime','date','hour','target','open','high','low','close','volume',
           'regime_label','regime_id','first_close_15m','last_close_15m','vol_first_15m','vol_last_15m'}
feat_cols = [c for c in raw60.columns if c not in EXCLUDE and raw60[c].dtype in ['float64','int64','float32','int32']]
feat_cols = [c for c in feat_cols if raw60[c].notna().sum() > len(raw60) * 0.5]
print(f'  Raw features: {len(feat_cols)}')

# Winsorize 99/1
for c in feat_cols:
    lo, hi = raw60[c].quantile(0.01), raw60[c].quantile(0.99)
    raw60[c] = raw60[c].clip(lo, hi)

raw60 = raw60.dropna(subset=feat_cols + ['target'])
raw60 = raw60.sort_values(['symbol','datetime']).reset_index(drop=True)
print(f'  Clean data: {len(raw60):,} rows')

print('\n=== STEP 5: Feature Selection ===')
cutoff = raw60['datetime'].min() + pd.Timedelta(days=730)
sel_data = raw60[raw60['datetime'] < cutoff].copy()
if len(sel_data) > 100000:
    sel_data = sel_data.sample(100000, random_state=42)
elif len(sel_data) < 10000:
    sel_data = raw60.sample(min(50000, len(raw60)), random_state=42)

X_sel = sel_data[feat_cols].fillna(0).values
y_sel = sel_data['target'].values
ss = StandardScaler()
X_ss = ss.fit_transform(X_sel)

# MI
print('  MI...')
mi = mutual_info_regression(X_ss, y_sel, random_state=42, n_neighbors=5)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
mi_keep = set(mi_s[mi_s > mi_s.max() * 0.02].index)
print(f'    MI: {len(mi_keep)}/{len(feat_cols)}')

# XGB importance
print('  XGBoost...')
mx = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1, verbosity=0,
                       device='cuda', tree_method='hist')
mx.fit(X_ss, y_sel)
fi_s = pd.Series(mx.feature_importances_, index=feat_cols).sort_values(ascending=False)
fi_keep = set(fi_s[fi_s > fi_s.max() * 0.02].index)
print(f'    XGB: {len(fi_keep)}/{len(feat_cols)}')

# Boruta
print('  Boruta...')
X_b = np.hstack([X_ss, np.random.permutation(X_ss.T).T])
n_r = X_ss.shape[1]
mb = xgb.XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.05,
                       subsample=0.8, random_state=42, n_jobs=-1, verbosity=0,
                       device='cuda', tree_method='hist')
mb.fit(X_b, y_sel)
ri = mb.feature_importances_[:n_r]
sm = mb.feature_importances_[n_r:].max()
boruta_keep = set([feat_cols[i] for i, v in enumerate(ri) if v > sm])
print(f'    Boruta: {len(boruta_keep)}/{len(feat_cols)}')

consensus = mi_keep & fi_keep & boruta_keep
union_2 = (mi_keep & fi_keep) | (mi_keep & boruta_keep) | (fi_keep & boruta_keep)
print(f'  Consensus: {len(consensus)}, 2-of-3: {len(union_2)}')
final_feats = list(union_2 if len(union_2) > 10 else consensus)
if len(final_feats) < 10:
    final_feats = sorted(mi_keep, key=lambda c: mi_s[c], reverse=True)[:30]
print(f'  Final features: {len(final_feats)}')

pickle.dump({'mi': mi_s, 'fi': fi_s, 'boruta': boruta_keep, 'consensus': consensus,
             'union_2': union_2, 'final_feats': final_feats},
            open(OUT/'feature_selection.pkl', 'wb'))

print('\n=== STEP 6: Walkforward ===')
raw60['year'] = raw60['datetime'].dt.year
years = sorted(raw60['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
ap, aa, asy, adt = [], [], [], []
yearly = {}

for wi, (ty, test_yr) in enumerate(windows):
    tr = raw60[raw60['year'].isin(ty)].copy()
    te = raw60[raw60['year'] == test_yr].copy()
    if len(te) < 500: continue
    embargo = te['datetime'].min() - pd.Timedelta(days=7)
    tr = tr[tr['datetime'] < embargo].copy()
    valid = [c for c in final_feats if c in tr.columns and c in te.columns]
    valid = [c for c in valid if tr[c].notna().all() and tr[c].std() > 1e-10]
    if len(valid) > 20:
        cs = tr[valid].sample(min(20000, len(tr)), random_state=42).corr().abs()
        up = cs.where(np.triu(np.ones(cs.shape), k=1).astype(bool))
        ds = set()
        for col in up.columns:
            if col in ds: continue
            ds.update(list(up.index[up[col] > 0.95]))
        valid = [c for c in valid if c not in ds]
    tr = tr.dropna(subset=valid + ['target'])
    te = te.dropna(subset=valid + ['target'])
    if len(tr) < 500 or len(te) < 50: continue
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[valid].values)
    X_te = scaler.transform(te[valid].values)
    y_tr, y_te = tr['target'].values, te['target'].values
    m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, n_jobs=-1, verbosity=0,
                          device='cuda', tree_method='hist')
    m.fit(X_tr, y_tr); pred = m.predict(X_te)
    r2 = r2_score(y_te, pred)
    corr = np.corrcoef(pred, y_te)[0,1] if np.std(pred) > 1e-12 else 0
    da = ((pred > 0) == (y_te > 0)).mean()
    yearly[int(test_yr)] = {'r2': r2, 'corr': corr, 'dir_acc': da, 'n_train': len(tr), 'n_test': len(te), 'n_feats': len(valid)}
    print(f'  [{wi+1:2d}/{len(windows)}] {test_yr}: train={len(tr):,} test={len(te):,} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%} feats={len(valid)}')
    ap.extend(pred.tolist()); aa.extend(y_te.tolist()); asy.extend(te['symbol'].tolist()); adt.extend(te['datetime'].tolist())

ap = np.array(ap); aa = np.array(aa)
print(f'\n{"="*55}\nOVERALL\n{"="*55}')
print(f'Predictions: {len(ap):,}')
print(f'R2:    {r2_score(aa, ap):+.4f}')
print(f'Corr:  {np.corrcoef(ap, aa)[0,1]:+.4f}')
print(f'DirAcc:{((ap>0)==(aa>0)).mean():.1%}')
print(f'MAE:   {np.mean(np.abs(ap-aa)):.3f}%')
print(f'RMSE:  {np.sqrt(np.mean((ap-aa)**2)):.3f}%')

# Backtest
rd = pd.DataFrame({'dt': pd.to_datetime(adt), 'sym': asy, 'pred': ap, 'act': aa}).sort_values('dt').reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000
def cost_rt(ps):
    if ps <= 0: return 1.0
    b = max(BRK*ps, MIN_BRK)/ps*2; gst = b + EXCH*2 + SEBI*2
    return b + STT + EXCH*2 + SEBI*2 + STAMP + gst*GST + SLIP*2

pp = None; bt = []
for d in sorted(rd['dt_norm'].unique()):
    day = rd[rd['dt_norm'] == d]
    if len(day) < 3: continue
    for hr in sorted(day['dt'].unique()):
        bar = day[day['dt'] == hr]
        if len(bar) < 3: continue
        pk = bar.sort_values('pred', ascending=False).iloc[0]
        r = pk['act']; to = 1.0 if pp is not None and pk['sym'] != pp else 0.0
        bt.append({'dt': hr, 'sym': pk['sym'], 'ret': r, 'net': r - cost_rt(TOTAL_POS)*to*100})
        pp = pk['sym']

btf = pd.DataFrame(bt)
g, n = btf['ret'], btf['net']
def cm(s, n_p=252*6.5):
    if len(s)<5 or s.std()==0: return (0,0,0,0)
    c = ((1+s/100).prod()**(n_p/len(s))-1)*100
    sh = s.mean()/s.std()*math.sqrt(n_p); wr = (s>0).mean()*100
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return (c, sh, wr, dd)
gc, gs, gw, gdd = cm(g); nc, ns, nw, ndd = cm(n)
print(f'\nBacktest: {len(btf)} trades')
print(f'Gross: CAGR={gc:+.1f}% Sharpe={gs:.2f} WinRate={gw:.1f}% Mean={g.mean():+.4f}%')
print(f'Net:   CAGR={nc:+.1f}% Sharpe={ns:.2f} WinRate={nw:.1f}% Mean={n.mean():+.4f}%')

pickle.dump({'yearly': yearly, 'rd': rd, 'bt': btf, 'final_feats': final_feats,
             'sel': {'mi': mi_s, 'fi': fi_s, 'boruta': boruta_keep},
             'n_symbols': raw60['symbol'].nunique(), 'time': time.time()-t0},
            open(OUT/'results.pkl', 'wb'))
btf.to_csv(OUT/'backtest.csv', index=False)
rd.to_csv(OUT/'predictions.csv', index=False)
print(f'\nSaved to {OUT} [{time.time()-t0:.0f}s]')
