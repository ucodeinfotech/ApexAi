"""MTF Approach B: 1day target + 60min intraday features (from same day)"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'report_mtf_daily'; OUT.mkdir(exist_ok=True)
t0 = time.time()
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000
def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
def calc_metrics(s, n=252):
    if len(s)<5 or s.std()==0: return (0,0,0,0)
    cagr = ((1+s/100).prod()**(n/len(s))-1)*100
    sh = s.mean()/s.std()*math.sqrt(n)
    wr = (s>0).mean()*100
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return (cagr, sh, wr, dd)

print('Loading data...')
con = duckdb.connect(str(DB), read_only=True)

# 1day features (all symbols)
fs1 = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume,
           sma_5, sma_10, sma_20, sma_50, sma_200,
           ema_5, ema_10, ema_20, ema_50, ema_200,
           rsi_7, rsi_14, rsi_21, macd_line, macd_signal, macd_hist,
           adx, plus_di, minus_di, atr_7, atr_14, atr_21,
           bb_pct_b, bb_width, bb_lower, bb_middle, bb_upper,
           kc_width, dc_width, obv, cmf, stoch_k, stoch_d,
           williams_r, mfi, uo, cci, trix,
           roc_5, roc_10, roc_20, zscore_20, skew_20, kurt_20,
           hv_10, hv_20, hv_30, eom, fi, vpt,
           swing_high, swing_low, psar,
           ret_1d, ret_5d, ret_10d, ret_20d,
           log_ret_1d, log_ret_5d, log_ret_10d, log_ret_20d,
           close_vs_sma_10, close_vs_sma_20, close_vs_sma_50,
           body_ratio_5, body_ratio_10, body_ratio_20,
           aroon_up, aroon_down, aroon_osc, serial_corr_20,
           vol_ratio_5, vol_ratio_10, vol_ratio_20,
           range_5, range_10, range_20,
           "pivot", r1, r2, s1, s2,
           ad_line, wma_10, wma_20
    FROM feature_store WHERE timeframe='1day' ORDER BY datetime
""").fetchdf()
print(f'1day: {len(fs1):,} rows, {fs1["symbol"].nunique()} symbols')

# 60min OHLCV for intraday features
f60raw = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume
    FROM feature_store WHERE timeframe='60min' ORDER BY datetime
""").fetchdf()
print(f'60min OHLCV: {len(f60raw):,} rows, {f60raw["symbol"].nunique()} symbols')

# RS, VIX, delivery, regimes
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

fs1 = fix_dt(fs1); fs1['date'] = fs1['datetime'].dt.normalize()
fs1['year'] = fs1['datetime'].dt.year; fs1['dow'] = fs1['datetime'].dt.dayofweek; fs1['month'] = fs1['datetime'].dt.month
fs1['is_month_end'] = fs1['datetime'].dt.is_month_end.astype(int)

f60raw = fix_dt(f60raw); f60raw['date'] = f60raw['datetime'].dt.normalize()
ms = fix_dt(ms); ms['date'] = ms['datetime'].dt.normalize()
vix = fix_dt(vix)
dv['date'] = pd.to_datetime(dv['date'])
reg_dt = pd.to_datetime(reg['datetime']); reg['date'] = (reg_dt.dt.tz_localize(None) if reg_dt.dt.tz is not None else reg_dt).dt.normalize()

# --- Penny filter (keep same as best run) ---
avg_close = fs1[fs1['date'] >= '2024-06-24'].groupby('symbol')['close'].mean()
penny_syms = set(avg_close[avg_close < 50].index)
fs1 = fs1[~fs1['symbol'].isin(penny_syms)].copy()
f60raw = f60raw[~f60raw['symbol'].isin(penny_syms)].copy()
print(f'After penny filter: {fs1["symbol"].nunique()} symbols (1day), {f60raw["symbol"].nunique()} symbols (60min)')

# --- Intraday features from 60min bars aggregated to daily ---
print('Computing intraday features...')
f60raw = f60raw.sort_values(['symbol','datetime']).reset_index(drop=True)
# VWAP
f60raw['vwap'] = (f60raw['high'] + f60raw['low'] + f60raw['close']) / 3 * f60raw['volume']
f60raw['vwap_cum'] = f60raw.groupby(['symbol','date'])['vwap'].cumsum()
f60raw['vol_cum'] = f60raw.groupby(['symbol','date'])['volume'].cumsum()
f60raw['vwap_day'] = f60raw['vwap_cum'] / f60raw['vol_cum'].replace(0, np.nan)
f60raw['vwap_dist'] = (f60raw['close'] / f60raw['vwap_day'] - 1) * 100

# Per-bar return
f60raw['bar_ret'] = f60raw.groupby('symbol')['close'].transform(lambda x: x.pct_change()) * 100

# Aggregate to daily intraday features
intra = f60raw.groupby(['symbol','date']).agg(
    n_bars=('close','count'),
    open_first=('open','first'),
    close_last=('close','last'),
    high_day=('high','max'),
    low_day=('low','min'),
    vol_total=('volume','sum'),
    vwap_dist_last=('vwap_dist','last'),
    bar_ret_mean=('bar_ret','mean'),
    bar_ret_std=('bar_ret','std'),
    bar_ret_first=('bar_ret','first'),
    bar_ret_last=('bar_ret','last'),
    up_bars=('bar_ret', lambda x: (x > 0).sum()),
    down_bars=('bar_ret', lambda x: (x < 0).sum()),
    vol_first2=('volume', lambda x: x.iloc[:2].sum() if len(x) >= 2 else x.sum()),
    vol_last2=('volume', lambda x: x.iloc[-2:].sum() if len(x) >= 2 else x.sum()),
).reset_index()

intra['intra_range'] = (intra['high_day'] - intra['low_day']) / intra['close_last'] * 100
intra['intra_trend'] = (intra['close_last'] / intra['open_first'] - 1) * 100
intra['intra_vol_ratio'] = np.where(intra['vol_last2'] > 0, intra['vol_first2'] / intra['vol_last2'], 1.0)
intra['up_ratio'] = intra['up_bars'] / intra['n_bars'].replace(0, 1)
intra['close_vs_open'] = (intra['close_last'] / intra['open_first'] - 1) * 100

# Store prev day close for opening gap
intra = intra.sort_values(['symbol','date']).reset_index(drop=True)
intra['prev_close'] = intra.groupby('symbol')['close_last'].shift(1)
intra['opening_gap'] = (intra['open_first'] / intra['prev_close'] - 1) * 100

# --- Merge all data ---
fs1['date'] = fs1['date'].dt.normalize()
# Intraday features on same date (all bars available at end-of-day, predicting next day)
df = fs1.merge(intra, on=['symbol','date'], how='left')

# RS
df = df.merge(ms[['symbol','date','rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20']], on=['symbol','date'], how='left')

# Delivery
df = df.merge(dv, on=['symbol','date'], how='left')

# VIX (market-wide, merge_asof)
common_dtype = df['datetime'].dtype
vix['datetime'] = vix['datetime'].astype(common_dtype)
df = pd.merge_asof(df.sort_values('datetime'), vix.sort_values('datetime'), on='datetime', direction='backward')

# Regime (strip timezone to match df)
reg_dt = pd.to_datetime(reg['datetime'])
reg['datetime'] = reg_dt.dt.tz_localize(None) if reg_dt.dt.tz is not None else reg_dt
df = df.merge(reg[['datetime','regime_label','regime_id']], on='datetime', how='left')

# Fill NAs
for c in ['rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20']:
    df[c] = df[c].fillna(0)
df['delivery_pct'] = df.groupby('symbol')['delivery_pct'].ffill().fillna(0)
for c in ['vix_close','vix_change','vix_range','vix_ma_5','vix_ma_20','vix_zscore_20']:
    df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
df['regime_label'] = df['regime_label'].fillna('sideways')
df['regime_id'] = df['regime_id'].fillna(0).astype(int)

# Fill missing intraday features
intra_cols = ['n_bars','vwap_dist_last','bar_ret_mean','bar_ret_std','bar_ret_first','bar_ret_last',
              'up_ratio','intra_range','intra_trend','intra_vol_ratio','close_vs_open','opening_gap']
for c in intra_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# --- Target: next-day return ---
df = df.sort_values(['symbol','datetime']).reset_index(drop=True)
df['fwd_return_1d'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * 100
df['fwd_open_ret_1d'] = (df.groupby('symbol')['close'].shift(-1) / df.groupby('symbol')['open'].shift(-1) - 1) * 100

# --- Feature list ---
EXCLUDE = {'symbol','datetime','date','year','dow','month','is_month_end',
           'open','high','low','close','volume','open_first','close_last','high_day','low_day',
           'fwd_return_1d','fwd_open_ret_1d','regime_label','regime_id'}
feat_cols = [c for c in df.columns if c not in EXCLUDE and df[c].dtype in ['float64','int64','float32','int32']]
feat_cols = [c for c in feat_cols if df[c].notna().sum() > len(df) * 0.9]
# Drop intraday raw aggregation outputs that aren't useful
feat_cols = [c for c in feat_cols if c not in ['up_bars','down_bars','vol_total','vol_first2','vol_last2','bar_ret_mean','bar_ret_std']]
# Add regime dummies
df = pd.get_dummies(df, columns=['regime_label'], prefix='regime')
reg_cols = [c for c in df.columns if c.startswith('regime_')]
feat_cols = feat_cols + reg_cols

print(f'Features: {len(feat_cols)}')

df = df.dropna(subset=feat_cols + ['fwd_return_1d'])
print(f'Clean rows: {len(df):,}, Symbols: {df["symbol"].nunique()}, Years: {sorted(df["year"].unique())}')

# --- Walkforward ---
years = sorted(df['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
print(f'Walkforward: {len(windows)} windows\n')

all_preds, all_actual, all_actual_open, all_syms, all_dates = [], [], [], [], []
yearly = {}

for wi, (ty, test_yr) in enumerate(windows):
    train_raw = df[df['year'].isin(ty)].copy()
    test = df[df['year'] == test_yr].copy()
    if len(test) < 100: continue
    embargo = test['datetime'].min() - pd.Timedelta(days=7)
    train = train_raw[train_raw['datetime'] < embargo].copy()
    valid = [c for c in feat_cols if c in train.columns and c in test.columns]
    valid = [c for c in valid if train[c].notna().all() and train[c].std() > 1e-10]
    # Drop high-correlation features
    corr_sample = train[valid].sample(min(20000, len(train)), random_state=42).corr().abs()
    upper = corr_sample.where(np.triu(np.ones(corr_sample.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop: continue
        hi = list(upper.index[upper[col] > 0.95])
        to_drop.update(hi)
    valid = [c for c in valid if c not in to_drop]
    train = train.dropna(subset=valid); test = test.dropna(subset=valid)
    if len(train) < 500 or len(test) < 50: continue
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[valid].values)
    X_te = scaler.transform(test[valid].values)
    y_tr, y_te = train['fwd_return_1d'].values, test['fwd_return_1d'].values
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.01, reg_lambda=0.01,
                             random_state=42, n_jobs=-1, verbosity=0,
                             device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr); pred = model.predict(X_te)
    if np.isnan(pred).any(): continue
    r2 = r2_score(y_te, pred); corr = np.corrcoef(pred, y_te)[0,1] if np.std(pred) > 1e-12 else 0
    mae = mean_absolute_error(y_te, pred); da = ((pred>0)==(y_te>0)).mean()
    yearly[int(test_yr)] = {'r2':r2, 'corr':corr, 'mae':mae, 'dir_acc':da, 'n_train':len(train), 'n_test':len(test), 'n_feats':len(valid)}
    print(f'[{wi+1:2d}/{len(windows)}] {test_yr}: train={len(train):,} test={len(test):,} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%} feats={len(valid)}')
    all_preds.extend(pred.tolist()); all_actual.extend(y_te.tolist())
    all_actual_open.extend(test['fwd_open_ret_1d'].tolist())
    all_syms.extend(test['symbol'].tolist()); all_dates.extend(test['datetime'].tolist())

all_preds = np.array(all_preds); all_actual = np.array(all_actual); all_actual_open = np.array(all_actual_open)
print(f'\n{"="*55}\nOVERALL\n{"="*55}')
print(f'Predictions: {len(all_preds):,}')
print(f'R2:    {r2_score(all_actual, all_preds):+.4f}')
print(f'Corr:  {np.corrcoef(all_preds, all_actual)[0,1]:+.4f}')
print(f'DirAcc:{((all_preds>0)==(all_actual>0)).mean():.1%}')
print(f'MAE:   {mean_absolute_error(all_actual, all_preds):.3f}%')
print(f'RMSE:  {np.sqrt(mean_squared_error(all_actual, all_preds)):.3f}%')

# --- Directional backtest ---
print(f'\n{"="*55}\nDIRECTIONAL BACKTEST\n{"="*55}')
td = pd.DataFrame({'date':pd.to_datetime(all_dates), 'pred':all_preds, 'actual':all_actual,
                   'actual_open':all_actual_open, 'symbol':all_syms})
td['date_norm'] = td['date'].dt.normalize()
dates_sorted = sorted(td['date_norm'].unique())

def run_strategy(name, pick_fn):
    trades = []
    for d in dates_sorted:
        day = td[td['date_norm'] == d].sort_values('pred', ascending=False)
        if len(day) < 3: continue
        picks = pick_fn(day)
        if picks is None or len(picks) == 0:
            trades.append({'date': d, 'ret': 0.0, 'n': 0, 'to': 0.0, 'cost': 0.0, 'net': 0.0})
            continue
        picks_df = day[day['symbol'].isin(picks)]
        if len(picks_df) == 0:
            trades.append({'date': d, 'ret': 0.0, 'n': 0, 'to': 0.0, 'cost': 0.0, 'net': 0.0})
            continue
        avg_ret = picks_df['actual_open'].mean()
        n_pos = len(picks_df)
        pos_cost = cost_rt(TOTAL_POS / max(n_pos, 1))
        trades.append({'date': d, 'ret': avg_ret, 'n': n_pos, 'to': 1.0, 'cost': pos_cost * 100, 'net': avg_ret - pos_cost * 100})
    return pd.DataFrame(trades)

def run_ls(name, n_top, n_bot):
    trades = []
    for d in dates_sorted:
        day = td[td['date_norm'] == d].sort_values('pred', ascending=False)
        if len(day) < n_top + n_bot: continue
        top = day.head(n_top); bot = day.tail(n_bot)
        long_ret = top['actual_open'].mean(); short_ret = bot['actual_open'].mean()
        ret = (long_ret - short_ret) / 2
        n = n_top + n_bot; pos_cost = cost_rt(TOTAL_POS / max(n, 1))
        trades.append({'date': d, 'ret': ret, 'n': n, 'to': 1.0, 'cost': pos_cost * 100 * 2, 'net': ret - pos_cost * 100 * 2})
    return pd.DataFrame(trades)

strategies = {
    'Long-Only D9': run_strategy('D9', lambda d: d.head(max(1, min(len(d)//10, 9)))['symbol'].tolist()),
    'LS-D10/D10': run_ls('LS-D10/D10', 10, 10),
    'LS-D9/B9': run_ls('LS-D9/B9', 9, 9),
    'Directional Long': run_strategy('DirLong', lambda d: d[d['pred']>0]['symbol'].tolist()),
    'Directional Short': run_strategy('DirShort', lambda d: d[d['pred']<0]['symbol'].tolist()),
    'Top-5': run_strategy('Top5', lambda d: d.head(5)['symbol'].tolist()),
}

print(f'\n{"Strategy":25s} {"Days":>5s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"MeanRet":>8s}')
print('-'*88)
for sname, sdf in strategies.items():
    if len(sdf) < 5: continue
    gc, gs, gw, gdd = calc_metrics(sdf['ret'])
    nc, ns, nw, ndd = calc_metrics(sdf['net'])
    print(f'{sname:25s} {len(sdf):5d} {gc:>+11.1f}% {nc:>+11.1f}% {gs:>7.2f} {gw:>7.1f}% {gdd:>7.1f}% {sdf["ret"].mean():>+7.3f}%')

output = {'yearly': yearly, 'td': td, 'strategies': strategies, 'n_symbols': df['symbol'].nunique(), 'time': time.time()-t0}
with open(OUT/'results.pkl', 'wb') as f: pickle.dump(output, f)
td.to_csv(OUT/'predictions.csv', index=False)
for sname, sdf in strategies.items():
    sdf.to_csv(OUT / f'strategy_{sname.replace(" ", "_").replace("/", "_")}.csv', index=False)
print(f'\nSaved to {OUT} [{time.time()-t0:.0f}s]')
