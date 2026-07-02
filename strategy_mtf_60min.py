"""MTF Approach A: 60min target + 15min rollup features + 1day context"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'report_mtf_60min'; OUT.mkdir(exist_ok=True)
t0 = time.time()
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000
def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
def calc_metrics(s, n=252*6.5):
    if len(s)<5 or s.std()==0: return (0,0,0,0)
    cagr = ((1+s/100).prod()**(n/len(s))-1)*100
    sh = s.mean()/s.std()*math.sqrt(n)
    wr = (s>0).mean()*100
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return (cagr, sh, wr, dd)

print('Loading 60min features...')
con = duckdb.connect(str(DB), read_only=True)
fs60 = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume,
           sma_5, sma_10, sma_20, ema_5, ema_10, ema_20,
           rsi_14, macd_line, macd_signal, macd_hist, adx,
           plus_di, minus_di, atr_14, bb_pct_b, bb_width,
           kc_width, dc_width, obv, cmf, stoch_k, stoch_d,
           williams_r, mfi, uo, cci, trix, roc_5, roc_10, roc_20,
           zscore_20, hv_20, eom, fi, vpt, swing_high, swing_low,
           ret_1d, log_ret_1d, close_vs_sma_10, close_vs_sma_20,
           range_5, range_10, range_20, vol_ratio_10
    FROM feature_store WHERE timeframe='60min' ORDER BY datetime
""").fetchdf()
print(f'60min: {len(fs60):,} rows, {fs60["symbol"].nunique()} symbols')

print('Loading 15min OHLCV...')
f15 = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume
    FROM feature_store WHERE timeframe='15min' ORDER BY datetime
""").fetchdf()
print(f'15min: {len(f15):,} rows, {f15["symbol"].nunique()} symbols')

print('Loading daily context (RS, delivery, VIX, regime)...')
daily_fs = con.execute("""
    SELECT symbol, datetime, close, hv_20, atr_14, bb_pct_b,
           sma_20, ema_20, rsi_14, adx
    FROM feature_store WHERE timeframe='1day' ORDER BY datetime
""").fetchdf()
ms = con.execute("""
    SELECT symbol, datetime, rs_vs_market, rs_vs_sector, rs_ratio_market,
           rs_ratio_sector, rs_momentum_10, rs_momentum_20
    FROM market_structure WHERE timeframe='1day' ORDER BY datetime
""").fetchdf()
vix = con.execute("""
    SELECT datetime, vix_close, vix_change, vix_range, vix_ma_5, vix_ma_20, vix_zscore_20
    FROM vix_data ORDER BY datetime
""").fetchdf()
dv = con.execute("""
    SELECT symbol, date, delivery_pct FROM delivery_data ORDER BY symbol, date
""").fetchdf()
reg = con.execute("""
    SELECT datetime, regime_label, regime_id FROM market_regimes ORDER BY datetime
""").fetchdf()
con.close()

# --- Process timestamps ---
def fix_dt(df, col='datetime'):
    dt = pd.to_datetime(df[col])
    df[col] = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    return df

fs60 = fix_dt(fs60); fs60['date'] = fs60['datetime'].dt.normalize(); fs60['hour'] = fs60['datetime'].dt.hour
f15 = fix_dt(f15); f15['date'] = f15['datetime'].dt.normalize(); f15['hour'] = f15['datetime'].dt.hour
daily_fs = fix_dt(daily_fs); daily_fs['date'] = daily_fs['datetime'].dt.normalize()
ms = fix_dt(ms); ms['date'] = ms['datetime'].dt.normalize()
vix = fix_dt(vix)

# --- 15min rollup: for each 60min bar, aggregate 4x15min sub-bars ---
# Group 15min by (symbol, date, hour) — each hour has 4 15min bars
print('Computing 15min rollup features...')
f15 = f15.sort_values(['symbol','datetime']).reset_index(drop=True)
f15['15min_ret'] = f15.groupby('symbol')['close'].transform(lambda x: x.pct_change()) * 100
f15['15min_range'] = (f15['high'] - f15['low']) / f15['close'] * 100

rollup_15 = f15.groupby(['symbol','date','hour']).agg(
    ret_sum_15=('15min_ret', 'sum'),
    ret_std_15=('15min_ret', 'std'),
    ret_first_15=('15min_ret', lambda x: x.iloc[0] if len(x) > 0 else 0),
    ret_last_15=('15min_ret', lambda x: x.iloc[-1] if len(x) > 0 else 0),
    close_first_15=('close', 'first'),
    close_last_15=('close', 'last'),
    high_max_15=('high', 'max'),
    low_min_15=('low', 'min'),
    vol_sum_15=('volume', 'sum'),
    vol_first_half=('volume', lambda x: x.iloc[:2].sum() if len(x) >= 2 else x.sum()),
    vol_second_half=('volume', lambda x: x.iloc[2:].sum() if len(x) >= 4 else 0),
    n_bars_15=('close', 'count'),
    range_mean_15=('15min_range', 'mean'),
).reset_index()
rollup_15['vol_ratio_15'] = np.where(rollup_15['vol_second_half'] > 0,
    rollup_15['vol_first_half'] / rollup_15['vol_second_half'], 1.0)
rollup_15['15min_slope'] = (rollup_15['close_last_15'] / rollup_15['close_first_15'] - 1) * 100
rollup_15['15min_range_pct'] = (rollup_15['high_max_15'] - rollup_15['low_min_15']) / rollup_15['close_last_15'] * 100

# Merge 15min rollup into 60min (same symbol + date + hour window)
fs60['hour_key'] = fs60['datetime'].dt.hour
rollup_15['hour_key'] = rollup_15['hour']
merge_cols = ['symbol','date','hour_key']
drop_rollup = [c for c in rollup_15.columns if c not in ['symbol','date','hour']]
rollup_map = {f'{c}_15m': c for c in drop_rollup}
rollup_15_renamed = rollup_15.rename(columns=rollup_map)
fs60 = fs60.merge(rollup_15_renamed, left_on=merge_cols, right_on=['symbol','date','hour_key'], how='left')
for c in ['hour_key', 'hour', 'hour_x', 'hour_y', 'hour_key_x', 'hour_key_y']:
    if c in fs60.columns:
        fs60 = fs60.drop(columns=[c])

# Fill missing 15min features
for c in fs60.columns:
    if c.endswith('_15m'):
        fs60[c] = fs60[c].fillna(0)

# --- 1day context: previous day's RS, delivery, VIX, regime ---
daily_fs['date_next'] = daily_fs['date'] + pd.Timedelta(days=1)
daily_context = daily_fs[['symbol','date_next','close','hv_20','atr_14','bb_pct_b','sma_20','ema_20','rsi_14','adx']].rename(
    columns={'date_next': 'date', 'close': 'close_daily', 'hv_20': 'hv_20_daily', 'atr_14': 'atr_14_daily',
             'bb_pct_b': 'bb_pct_b_daily', 'sma_20': 'sma_20_daily', 'ema_20': 'ema_20_daily',
             'rsi_14': 'rsi_14_daily', 'adx': 'adx_daily'})

ms['date_next'] = ms['date'] + pd.Timedelta(days=1)
rs_context = ms[['symbol','date_next','rs_vs_market','rs_vs_sector']].rename(
    columns={'date_next': 'date', 'rs_vs_market': 'rs_vs_market_daily', 'rs_vs_sector': 'rs_vs_sector_daily'})

dv['date_next'] = pd.to_datetime(dv['date']) + pd.Timedelta(days=1)
dv_context = dv[['symbol','date_next','delivery_pct']].rename(columns={'date_next': 'date', 'delivery_pct': 'delivery_pct_daily'})

vix = vix.sort_values('datetime')
vix['date'] = vix['datetime'].dt.normalize()
vix['date_next'] = vix['date'] + pd.Timedelta(days=1)
vix_context = vix[['date_next','vix_close','vix_change','vix_ma_5','vix_ma_20','vix_zscore_20']].rename(
    columns={'date_next': 'date'})

reg_dt = pd.to_datetime(reg['datetime']); reg['date'] = (reg_dt.dt.tz_localize(None) if reg_dt.dt.tz is not None else reg_dt).dt.normalize()
reg['date_next'] = reg['date'] + pd.Timedelta(days=1)
reg_context = reg[['date_next','regime_label','regime_id']].rename(columns={'date_next': 'date'})

# Merge all daily context
fs60 = fs60.merge(daily_context, on=['symbol','date'], how='left')
fs60 = fs60.merge(rs_context, on=['symbol','date'], how='left')
fs60 = fs60.merge(dv_context, on=['symbol','date'], how='left')
# VIX and regime are market-wide (not per symbol)
vix_dedup = vix_context.drop_duplicates('date')
reg_dedup = reg_context.drop_duplicates('date')
fs60 = fs60.merge(vix_dedup, on='date', how='left')
fs60 = fs60.merge(reg_dedup, on='date', how='left')

for c in ['rs_vs_market_daily','rs_vs_sector_daily','delivery_pct_daily','vix_close',
          'vix_change','vix_ma_5','vix_ma_20','vix_zscore_20']:
    fs60[c] = fs60[c].fillna(0)
fs60['regime_label'] = fs60['regime_label'].fillna('sideways')
fs60['regime_id'] = fs60['regime_id'].fillna(0).astype(int)
# Fill daily features
for c in ['close_daily','hv_20_daily','atr_14_daily','bb_pct_b_daily',
          'sma_20_daily','ema_20_daily','rsi_14_daily','adx_daily']:
    fs60[c] = fs60.groupby('symbol')[c].ffill().fillna(0)

# --- Target: next-60min close return ---
fs60 = fs60.sort_values(['symbol','datetime']).reset_index(drop=True)
fs60['target'] = fs60.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * 100
fs60 = fs60.dropna(subset=['target'])

# --- Feature list ---
EXCLUDE = {'symbol','datetime','date','target','open','high','low','close','volume'}
feat_cols = [c for c in fs60.columns if c not in EXCLUDE and fs60[c].dtype in ['float64','int64','float32','int32']]
feat_cols = [c for c in feat_cols if fs60[c].notna().sum() > len(fs60) * 0.5]
# Drop regime_label (string) from numeric features
feat_cols = [c for c in feat_cols if c != 'regime_label']
print(f'Features: {len(feat_cols)}')

# Regime one-hot
fs60 = pd.get_dummies(fs60, columns=['regime_label'], prefix='regime')
reg_cols = [c for c in fs60.columns if c.startswith('regime_')]
feat_cols = feat_cols + reg_cols

fs60 = fs60.dropna(subset=feat_cols + ['target'])
print(f'Clean rows: {len(fs60):,}, Symbols: {fs60["symbol"].nunique()}')

# --- Walkforward ---
fs60['year'] = fs60['datetime'].dt.year
years = sorted(fs60['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
print(f'Walkforward: {len(windows)} windows\n')

all_results = []; yearly = {}

for wi, (ty, test_yr) in enumerate(windows):
    train_raw = fs60[fs60['year'].isin(ty)].copy()
    test = fs60[fs60['year'] == test_yr].copy()
    if len(test) < 500: continue
    embargo = test['datetime'].min() - pd.Timedelta(days=7)
    train = train_raw[train_raw['datetime'] < embargo].copy()
    valid = [c for c in feat_cols if c in train.columns and c in test.columns]
    valid = [c for c in valid if train[c].notna().all() and train[c].std() > 1e-10]
    train = train.dropna(subset=valid); test = test.dropna(subset=valid)
    if len(train) < 500 or len(test) < 50: continue
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[valid].values)
    X_te = scaler.transform(test[valid].values)
    y_tr, y_te = train['target'].values, test['target'].values
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbosity=0,
                             device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr); pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    corr = np.corrcoef(pred, y_te)[0,1] if np.std(pred) > 1e-12 else 0
    da = ((pred > 0) == (y_te > 0)).mean()
    yearly[int(test_yr)] = {'r2':r2, 'corr':corr, 'dir_acc':da, 'n_train':len(train), 'n_test':len(test)}
    print(f'[{wi+1:2d}/{len(windows)}] {test_yr}: train={len(train):,} test={len(test):,} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')
    for i in range(len(test)):
        all_results.append({'dt':test['datetime'].iloc[i], 'sym':test['symbol'].iloc[i], 'act':y_te[i], 'pred':pred[i]})

rd = pd.DataFrame(all_results)
print(f'\nTotal predictions: {len(rd):,}')
print(f'Overall: R2={r2_score(rd["act"], rd["pred"]):+.4f} Corr={np.corrcoef(rd["pred"], rd["act"])[0,1]:+.4f} DirAcc={((rd["pred"]>0)==(rd["act"]>0)).mean():.1%}')

# Quick backtest
rd = rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()
prev_pick = None; bt = []
for d in sorted(rd['dt_norm'].unique()):
    day = rd[rd['dt_norm'] == d]
    if len(day) < 5: continue
    for hour in sorted(day['dt'].unique()):
        bar = day[day['dt'] == hour]
        if len(bar) < 3: continue
        pick = bar.sort_values('pred', ascending=False).iloc[0]
        ret = pick['act']
        to = 1.0 if prev_pick is not None and pick['sym'] != prev_pick else 0.0
        cost = cost_rt(TOTAL_POS) * to * 100
        bt.append({'dt': hour, 'sym': pick['sym'], 'ret': ret, 'net': ret - cost, 'to': to})
        prev_pick = pick['sym']

bt_df = pd.DataFrame(bt)
g, n = bt_df['ret'], bt_df['net']
gc, gs, gw, gdd = calc_metrics(g, n=252*6.5)
nc, ns, nw, ndd = calc_metrics(n, n=252*6.5)
print(f'\nBacktest: {len(bt_df)} trades')
print(f'Gross: CAGR={gc:+.1f}% Sharpe={gs:.2f} WinRate={gw:.1f}% Mean={g.mean():+.4f}%')
print(f'Net:   CAGR={nc:+.1f}% Sharpe={ns:.2f} WinRate={nw:.1f}% Mean={n.mean():+.4f}%')

pickle.dump({'rd':rd, 'bt':bt_df, 'yearly':yearly, 'feat_cols':feat_cols}, open(OUT/'results.pkl','wb'))
print(f'\nSaved to {OUT} [{time.time()-t0:.0f}s]')
