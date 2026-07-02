"""Directional model — fixed look-ahead, all stocks, cost-aware backtest"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, math, time
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'directional_model_v2'
OUT.mkdir(exist_ok=True)
t0 = time.time()

# Costs
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

def calc_metrics(s, n=252):
    if len(s) < 5 or s.std() == 0: return (0,0,0,0)
    cagr = ((1+s/100).prod()**(n/len(s))-1)*100
    sh = s.mean()/s.std()*math.sqrt(n)
    wr = (s>0).mean()*100
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return (cagr, sh, wr, dd)

print(f'Loading data from DB...')
con = duckdb.connect(str(DB), read_only=True)

# Load feature_store 1day for all symbols
fs = con.execute("""
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
    FROM feature_store WHERE timeframe='1day'
    ORDER BY datetime
""").fetchdf()
print(f'Features loaded: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# RS features
ms = con.execute("""
    SELECT symbol, datetime, rs_vs_market, rs_vs_sector, rs_ratio_market,
           rs_ratio_sector, rs_momentum_10, rs_momentum_20
    FROM market_structure WHERE timeframe='1day' ORDER BY datetime
""").fetchdf()
print(f'Market structure: {len(ms):,} rows')

# VIX
vix = con.execute("""
    SELECT datetime, vix_close, vix_change, vix_range, vix_ma_5, vix_ma_20,
           vix_zscore_20 FROM vix_data ORDER BY datetime
""").fetchdf()
print(f'VIX: {len(vix):,} rows')

# Delivery
dv = con.execute("""
    SELECT symbol, date, delivery_pct FROM delivery_data ORDER BY symbol, date
""").fetchdf()
print(f'Delivery: {len(dv):,} rows')

# Regimes
reg = con.execute("""
    SELECT datetime, regime_label, regime_id FROM market_regimes ORDER BY datetime
""").fetchdf()
print(f'Regimes: {len(reg):,} rows')

con.close()

# --- Data cleaning ---
print('\nCleaning data...')
dt = pd.to_datetime(fs['datetime'])
fs['datetime'] = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
fs['date'] = fs['datetime'].dt.normalize()
fs['year'] = fs['datetime'].dt.year
fs['dow'] = fs['datetime'].dt.dayofweek
fs['month'] = fs['datetime'].dt.month
fs['is_month_end'] = fs['datetime'].dt.is_month_end.astype(int)

print(f'Total symbols before filtering: {fs["symbol"].nunique()} (no penny filter)')

# Merge RS
ms_dt = pd.to_datetime(ms['datetime'])
ms['date'] = ms_dt.dt.tz_localize(None).dt.normalize() if ms_dt.dt.tz is not None else ms_dt.dt.normalize()
fs = fs.merge(ms[['symbol','date','rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20']], on=['symbol','date'], how='left')
for c in ['rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20']:
    fs[c] = fs[c].fillna(0)

# Merge VIX
vix_dt = pd.to_datetime(vix['datetime'])
common_dtype = fs['datetime'].dtype
vix['datetime'] = (vix_dt.dt.tz_localize(None) if vix_dt.dt.tz is not None else vix_dt).astype(common_dtype)
fs = pd.merge_asof(fs.sort_values('datetime'), vix.sort_values('datetime'), on='datetime', direction='backward')
for c in ['vix_close','vix_change','vix_range','vix_ma_5','vix_ma_20','vix_zscore_20']:
    fs[c] = fs[c].fillna(fs[c].median() if fs[c].notna().any() else 0)

# Merge delivery
dv['date'] = pd.to_datetime(dv['date'])
fs = fs.merge(dv, on=['symbol','date'], how='left')
fs['delivery_pct'] = fs.groupby('symbol')['delivery_pct'].ffill().fillna(0)

# Merge regimes
reg_dt = pd.to_datetime(reg['datetime'])
reg['datetime'] = reg_dt.dt.tz_localize(None) if reg_dt.dt.tz is not None else reg_dt
fs = fs.merge(reg[['datetime','regime_label','regime_id']], on='datetime', how='left')
fs['regime_label'] = fs['regime_label'].fillna('sideways')
fs['regime_id'] = fs['regime_id'].fillna(0).astype(int)

# Target: next-day close return
fs = fs.sort_values(['symbol','datetime']).reset_index(drop=True)
fs['fwd_return_1d'] = fs.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * 100
fs['fwd_open_ret_1d'] = (fs.groupby('symbol')['close'].shift(-1) / fs.groupby('symbol')['open'].shift(-1) - 1) * 100

# Feature list
EXCLUDE = {'symbol','datetime','date','year','dow','month','is_month_end',
           'open','high','low','close','volume',
           'fwd_return_1d','fwd_open_ret_1d','regime_label','regime_id'}
feat_cols = [c for c in fs.columns if c not in EXCLUDE and fs[c].dtype in ['float64','int64','float32','int32']]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.9]
print(f'Features: {len(feat_cols)}')

# Drop rows with NaN in features or target
fs = fs.dropna(subset=feat_cols + ['fwd_return_1d'])
print(f'Clean rows: {len(fs):,}, Symbols: {fs["symbol"].nunique()}, Years: {sorted(fs["year"].unique())}')

# --- Walkforward training ---
years = sorted(fs['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
print(f'\nWalkforward: {len(windows)} windows')

all_preds = []; all_actual = []; all_actual_open = []
all_symbols = []; all_dates = []; all_regimes = []
yearly_metrics = {}

for wi, (ty, test_yr) in enumerate(windows):
    train_raw = fs[fs['year'].isin(ty)].copy()
    test = fs[fs['year'] == test_yr].copy()
    if len(test) < 100: continue

    # 7-day embargo
    embargo = test['datetime'].min() - pd.Timedelta(days=7)
    train = train_raw[train_raw['datetime'] < embargo].copy()

    # Filter valid features
    valid = [c for c in feat_cols if c in train.columns and c in test.columns]
    valid = [c for c in valid if train[c].notna().all() and train[c].std() > 1e-10]

    train = train.dropna(subset=valid + ['fwd_return_1d'])
    test = test.dropna(subset=valid + ['fwd_return_1d'])

    if len(train) < 500 or len(test) < 50: continue

    # Drop high-correlation features
    corr_sample = train[valid].sample(min(20000, len(train)), random_state=42).corr().abs()
    upper = corr_sample.where(np.triu(np.ones(corr_sample.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop: continue
        hi = list(upper.index[upper[col] > 0.95])
        to_drop.update(hi)
    valid = [c for c in valid if c not in to_drop]

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[valid].values)
    X_te = scaler.transform(test[valid].values)
    y_tr = train['fwd_return_1d'].values
    y_te = test['fwd_return_1d'].values

    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.01, reg_lambda=0.01,
                             random_state=42, n_jobs=-1, verbosity=0,
                             device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    if np.isnan(pred).any(): continue

    r2 = r2_score(y_te, pred)
    corr = np.corrcoef(pred, y_te)[0,1] if np.std(pred) > 1e-12 else 0
    mae = mean_absolute_error(y_te, pred)
    da = ((pred > 0) == (y_te > 0)).mean()
    yearly_metrics[int(test_yr)] = {'r2':r2, 'corr':corr, 'mae':mae, 'dir_acc':da,
                                     'n_train':len(train), 'n_test':len(test),
                                     'n_feats':len(valid)}
    print(f'[{wi+1:2d}/{len(windows)}] Test {test_yr}: train={len(train):,} test={len(test):,} '
          f'R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%} feats={len(valid)}')

    all_preds.extend(pred.tolist())
    all_actual.extend(y_te.tolist())
    all_actual_open.extend(test['fwd_open_ret_1d'].tolist())
    all_symbols.extend(test['symbol'].tolist())
    all_dates.extend(test['datetime'].tolist())
    all_regimes.extend(test['regime_label'].tolist())

all_preds = np.array(all_preds)
all_actual = np.array(all_actual)
all_actual_open = np.array(all_actual_open)

print(f'\n{"="*55}')
print(f'OVERALL PERFORMANCE')
print(f'{"="*55}')
print(f'Predictions: {len(all_preds):,}')
print(f'R2:    {r2_score(all_actual, all_preds):+.4f}')
print(f'Corr:  {np.corrcoef(all_preds, all_actual)[0,1]:+.4f}')
print(f'DirAcc:{((all_preds>0)==(all_actual>0)).mean():.1%}')
print(f'MAE:   {mean_absolute_error(all_actual, all_preds):.3f}%')
print(f'RMSE:  {np.sqrt(mean_squared_error(all_actual, all_preds)):.3f}%')

# --- Directional Strategy Backtest (cost-aware) ---
print(f'\n{"="*55}')
print(f'DIRECTIONAL STRATEGY BACKTEST')
print(f'{"="*55}')

td = pd.DataFrame({'date': pd.to_datetime(all_dates), 'pred': all_preds,
                    'actual': all_actual, 'actual_open': all_actual_open,
                    'symbol': all_symbols, 'regime': all_regimes})
td['date_norm'] = td['date'].dt.normalize()
dates_sorted = sorted(td['date_norm'].unique())

def run_strategy(name, pick_fn):
    """Generic strategy runner. pick_fn(day_df) -> (symbols_list, weights_or_None)"""
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
        # Use open-close PnL + transaction costs
        avg_ret = picks_df['actual_open'].mean()
        n_pos = len(picks_df)
        # Turnover: always 1.0 since we rebalance daily
        pos_cost = cost_rt(TOTAL_POS / max(n_pos, 1))
        trades.append({'date': d, 'ret': avg_ret, 'n': n_pos,
                       'to': 1.0, 'cost': pos_cost * 100, 'net': avg_ret - pos_cost * 100})
    return pd.DataFrame(trades)

strategies = {}

# 1. Long-Only D9: top decile (capped at 9)
def d9_picks(day):
    n = max(1, min(len(day)//10, 9))
    return day.head(n)['symbol'].tolist()
strategies['Long-Only D9'] = run_strategy('D9', d9_picks)

# 2. LS-D10/D10: long top decile, short bottom decile
def lsd10_picks(day):
    n = max(1, len(day)//10)
    top = day.head(n)['symbol'].tolist()
    bot = day.tail(n)['symbol'].tolist()
    # Return as long+short combined
    return top + bot
# For LS: ret = (long_avg - short_avg) / 2
def run_ls(name, n_top, n_bot):
    trades = []
    for d in dates_sorted:
        day = td[td['date_norm'] == d].sort_values('pred', ascending=False)
        if len(day) < n_top + n_bot: continue
        top = day.head(n_top)
        bot = day.tail(n_bot)
        long_ret = top['actual_open'].mean()
        short_ret = bot['actual_open'].mean()
        ret = (long_ret - short_ret) / 2  # 50% long, 50% short = 100% notional
        n = n_top + n_bot
        pos_cost = cost_rt(TOTAL_POS / max(n, 1))
        trades.append({'date': d, 'ret': ret, 'n': n,
                       'to': 1.0, 'cost': pos_cost * 100 * 2,  # x2 for long+short legs
                       'net': ret - pos_cost * 100 * 2})
    return pd.DataFrame(trades)

strategies['LS-D10/D10'] = run_ls('LS-D10/D10', 10, 10)
strategies['LS-D9/B9'] = run_ls('LS-D9/B9', 9, 9)

# 3. Directional Long: all stocks with pred > 0
def dir_long_picks(day):
    picks = day[day['pred'] > 0]
    if len(picks) == 0: return None
    return picks['symbol'].tolist()
strategies['Directional Long'] = run_strategy('Directional Long', dir_long_picks)

# 4. Directional Short: all stocks with pred < 0
def dir_short_picks(day):
    picks = day[day['pred'] < 0]
    if len(picks) == 0: return None
    return picks['symbol'].tolist()
strategies['Directional Short'] = run_strategy('Directional Short', dir_short_picks)

# 5. Top-5 concentrated
def top5_picks(day):
    return day.head(5)['symbol'].tolist()
strategies['Top-5'] = run_strategy('Top-5', top5_picks)

# --- Report ---
print(f'\n{"Strategy":25s} {"Days":>5s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"MeanRet":>8s}')
print('-'*88)
for sname, sdf in strategies.items():
    g = sdf['ret']
    n = sdf['net']
    if len(g) < 5: continue
    gc, gs, gw, gdd = calc_metrics(g)
    nc, ns, nw, ndd = calc_metrics(n)
    print(f'{sname:25s} {len(sdf):5d} {gc:>+11.1f}% {nc:>+11.1f}% {gs:>7.2f} {gw:>7.1f}% {gdd:>7.1f}% {g.mean():>+7.3f}%')

# --- Save ---
output = {'yearly': yearly_metrics, 'td': td, 'strategies': strategies,
          'n_symbols': fs['symbol'].nunique(), 'n_rows': len(fs), 'time': time.time()-t0}
with open(OUT/'results.pkl', 'wb') as f: pickle.dump(output, f)

# Save CSVs
td.to_csv(OUT/'predictions.csv', index=False)
for sname, sdf in strategies.items():
    sdf.to_csv(OUT / f'strategy_{sname.replace(" ", "_").replace("/", "_")}.csv', index=False)

print(f'\nSaved to {OUT}')
print(f'Time: {time.time()-t0:.0f}s')
