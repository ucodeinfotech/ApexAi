"""Approach 2: Intraday 60min model — XGBoost on 60min OHLCV features + VWAP reversion"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_60min'
OUT.mkdir(exist_ok=True)
t0 = time.time()

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

def calc_metrics(s, n=252):
    if len(s)<5 or s.std()==0: return 0,0,0,0
    cagr=(1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh=s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr=(s>0).mean()
    dd=((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd

print('Loading 60min features...')
con = duckdb.connect(str(DB), read_only=True)
fs = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume,
           sma_5, sma_10, sma_20, ema_5, ema_10, ema_20,
           rsi_14, macd_line, macd_signal, macd_hist, adx,
           plus_di, minus_di, atr_14, bb_pct_b, bb_width,
           kc_width, dc_width, obv, cmf, stoch_k, stoch_d,
           williams_r, mfi, uo, cci, trix, roc_5, roc_10, roc_20,
           zscore_20, hv_20, eom, fi, vpt, swing_high, swing_low,
           ret_1d, log_ret_1d, close_vs_sma_10, close_vs_sma_20,
           range_5, range_10, range_20, vol_ratio_10
    FROM feature_store WHERE timeframe='60min'
    ORDER BY datetime
""").fetchdf()
print(f'Loaded: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

con.close()
print(f'Total rows: {len(fs):,}, Total symbols: {fs["symbol"].nunique()} (no penny filter)')

# Process datetime
dt = pd.to_datetime(fs['datetime'])
fs['datetime'] = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
fs['date'] = fs['datetime'].dt.normalize()
fs['hour'] = fs['datetime'].dt.hour

# Target: next-60min close return
fs = fs.sort_values(['symbol','datetime']).reset_index(drop=True)
fs['target'] = fs.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * 100

# VWAP-based features
fs['vwap'] = (fs['high'] + fs['low'] + fs['close']) / 3 * fs['volume']
fs['vwap_cum'] = fs.groupby(['symbol','date'])['vwap'].cumsum()
fs['vol_cum'] = fs.groupby(['symbol','date'])['volume'].cumsum()
fs['vwap_cum'] = fs['vwap_cum'] / fs['vol_cum'].replace(0, np.nan)
fs['vwap_distance'] = (fs['close'] / fs['vwap_cum'] - 1) * 100

# Intraday time signals
fs['is_open'] = (fs['hour'] == 9).astype(float)  # 9:00 AM
fs['is_close'] = (fs['hour'] == 14).astype(float)  # 2:00 PM (close is 3:30 but last 60min is 14:00-15:00)
fs['mid_session'] = ((fs['hour'] >= 10) & (fs['hour'] <= 12)).astype(float)

# Drop NaN targets (last bar of each day)
fs = fs.dropna(subset=['target'])

# Feature engineering
feat_cols = [c for c in fs.columns if c not in ['symbol','datetime','date','hour','vwap','vwap_cum','vol_cum',
                                                  'vwap','target','open','high','low','close','volume']]
# Remove any non-numeric
feat_cols = [c for c in feat_cols if fs[c].dtype in ['float64','int64','float32','int32']]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.5]
print(f'Features: {len(feat_cols)}')

# Walkforward by year
fs['year'] = fs['datetime'].dt.year
years = sorted(fs['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
print(f'Walkforward: {len(windows)} windows')

all_results = []
for wi, (ty, test_yr) in enumerate(windows):
    train = fs[fs['year'].isin(ty)].copy()
    test = fs[fs['year'] == test_yr].copy()
    if len(test) < 500: continue

    # 7-day embargo
    embargo = test['datetime'].min() - pd.Timedelta(days=7)
    train = train[train['datetime'] < embargo].copy()

    # Feature cleaning
    valid = [c for c in feat_cols if train[c].notna().all() and train[c].std() > 1e-10]
    train = train.dropna(subset=valid)
    test = test.dropna(subset=valid)
    if len(train) < 500 or len(test) < 50: continue

    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[valid].values)
    X_te = scaler.transform(test[valid].values)
    y_tr = train['target'].values; y_te = test['target'].values

    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbosity=0,
                             device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    r2 = r2_score(y_te, pred)
    corr = np.corrcoef(pred, y_te)[0,1] if np.std(pred) > 1e-12 and np.std(y_te) > 1e-12 else 0
    da = ((pred > 0) == (y_te > 0)).mean()
    print(f'[{wi+1:2d}/{len(windows)}] Test {test_yr}: train={len(train):,} test={len(test):,} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

    for i in range(len(test)):
        all_results.append({'dt': test['datetime'].iloc[i], 'sym': test['symbol'].iloc[i],
                            'act': y_te[i], 'pred': pred[i]})

rd = pd.DataFrame(all_results)
print(f'\nTotal predictions: {len(rd):,}')
print(f'Overall: R2={r2_score(rd["act"], rd["pred"]):+.4f} Corr={np.corrcoef(rd["pred"], rd["act"])[0,1]:+.4f} DirAcc={((rd["pred"]>0)==(rd["act"]>0)).mean():.1%}')

# Quick backtest: Top-1 per bar (open-close PnL as proxy)
rd = rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()
prev_pick = None; bt = []
for d in sorted(rd['dt_norm'].unique()):
    day = rd[rd['dt_norm'] == d]
    if len(day) < 5: continue
    # Per-hour top pick (since we have 60min bars)
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
g = bt_df['ret']; n = bt_df['net']
gc, gs, gw, gdd = calc_metrics(g, n=252*6.5)  # ~6.5 bars per day
nc, ns, nw, ndd = calc_metrics(n, n=252*6.5)
print(f'\n60min Backtest: {len(bt_df)} trades')
print(f'Gross: CAGR={gc:+.1f}% Sharpe={gs:.2f} WinRate={gw:.1f}% Mean={g.mean():+.4f}%')
print(f'Net:   CAGR={nc:+.1f}% Sharpe={ns:.2f} WinRate={nw:.1f}% Mean={n.mean():+.4f}%')

pickle.dump({'rd':rd,'bt':bt_df}, open(OUT/'results_60min.pkl','wb'))
print(f'\nSaved to {OUT} [{time.time()-t0:.0f}s]')
