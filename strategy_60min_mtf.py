"""
60min Intraday Strategy with Multi-Timeframe Features.
Now with ALL symbols from raw_market (not just feature_store).
"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_60min_mtf'
OUT.mkdir(exist_ok=True)
t0 = time.time()

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000; MULT=100

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

def calc_metrics(s, n=252*6.5):
    if len(s)<5 or s.std()==0: return 0,0,0,0
    cagr = (1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh = s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr = (s>0).mean()
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd

def compute_ta(df):
    """Compute technical indicators for 60min OHLCV data."""
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    def _g(c):
        return df.groupby('symbol')[c]

    # Returns
    df['ret_1d'] = _g('close').pct_change(1) * 100
    df['log_ret_1d'] = np.log(df['close'] / _g('close').shift(1)) * 100

    # SMAs
    for p in [5, 10, 20]:
        df[f'sma_{p}'] = _g('close').transform(lambda x: x.rolling(p, min_periods=1).mean())

    # EMAs
    for p in [5, 10, 20]:
        df[f'ema_{p}'] = _g('close').transform(lambda x: x.ewm(span=p, min_periods=1, adjust=False).mean())

    # RSI 14
    delta = _g('close').diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.transform(lambda x: x.rolling(14, min_periods=1).mean())
    avg_loss = loss.transform(lambda x: x.rolling(14, min_periods=1).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = _g('close').transform(lambda x: x.ewm(span=12, min_periods=1, adjust=False).mean())
    ema26 = _g('close').transform(lambda x: x.ewm(span=26, min_periods=1, adjust=False).mean())
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df.groupby('symbol')['macd_line'].transform(
        lambda x: x.ewm(span=9, min_periods=1, adjust=False).mean())
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # ATR 14
    prev_close = _g('close').shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.groupby(df['symbol']).transform(lambda x: x.rolling(14, min_periods=1).mean())

    # ADX
    up_move = df['high'] - _g('high').shift(1)
    down_move = _g('low').shift(1) - df['low']
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    atr14 = df['atr_14'].replace(0, np.nan)
    pdi = 100 * plus_dm.groupby(df['symbol']).transform(lambda x: x.rolling(14, min_periods=1).mean()) / atr14
    mdi = 100 * minus_dm.groupby(df['symbol']).transform(lambda x: x.rolling(14, min_periods=1).mean()) / atr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df['adx'] = dx.groupby(df['symbol']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    df['plus_di'] = pdi; df['minus_di'] = mdi

    # Bollinger Bands (20, 2)
    bb_sma = _g('close').transform(lambda x: x.rolling(20, min_periods=1).mean())
    bb_std = _g('close').transform(lambda x: x.rolling(20, min_periods=1).std())
    df['bb_width'] = 2 * bb_std / bb_sma * 100
    df['bb_pct_b'] = (df['close'] - (bb_sma - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    # OBV
    direction = np.sign(df['ret_1d'].fillna(0))
    df['obv'] = _g('volume').transform(lambda x: (x * direction).cumsum())
    # CMF (20-period)
    mfv = df['volume'] * ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1)
    cmf_num = mfv.groupby(df['symbol']).transform(lambda x: x.rolling(20, min_periods=1).sum())
    cmf_den = _g('volume').transform(lambda x: x.rolling(20, min_periods=1).sum())
    df['cmf'] = cmf_num / cmf_den.replace(0, np.nan)

    # VWAP (within day)
    df['date'] = df['datetime'].dt.normalize()
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap_num = (tp * df['volume']).groupby([df['symbol'], df['date']]).cumsum()
    vwap_vol = df['volume'].groupby([df['symbol'], df['date']]).cumsum()
    df['vwap'] = vwap_num / vwap_vol.replace(0, np.nan)
    df['vwap_distance'] = (df['close'] / df['vwap'] - 1) * MULT

    # Swing high/low (trailing 5-bar)
    df['swing_high'] = _g('high').transform(lambda x: x.rolling(5, min_periods=1).max())
    df['swing_low'] = _g('low').transform(lambda x: x.rolling(5, min_periods=1).min())

    # Range metrics
    df['range_5'] = (_g('high').transform(lambda x: x.rolling(5, min_periods=1).max()) -
                     _g('low').transform(lambda x: x.rolling(5, min_periods=1).min())) / df['close']
    df['range_10'] = (_g('high').transform(lambda x: x.rolling(10, min_periods=1).max()) -
                      _g('low').transform(lambda x: x.rolling(10, min_periods=1).min())) / df['close']
    df['range_20'] = (_g('high').transform(lambda x: x.rolling(20, min_periods=1).max()) -
                      _g('low').transform(lambda x: x.rolling(20, min_periods=1).min())) / df['close']

    # Vol ratio
    avg_vol_10 = _g('volume').transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['vol_ratio_10'] = df['volume'] / avg_vol_10.replace(0, np.nan)

    # Price vs SMA
    for p in [10, 20]:
        df[f'close_vs_sma_{p}'] = (df['close'] / df[f'sma_{p}'] - 1) * 100

    # ROC
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = _g('close').transform(lambda x: x.pct_change(p) * 100)

    # Z-score (20)
    roll_mean = _g('close').transform(lambda x: x.rolling(20, min_periods=1).mean())
    roll_std = _g('close').transform(lambda x: x.rolling(20, min_periods=1).std())
    df['zscore_20'] = (df['close'] - roll_mean) / roll_std.replace(0, np.nan)

    # HV (20-day historical volatility)
    log_ret = df.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['hv_20'] = log_ret.groupby(df['symbol']).transform(
        lambda x: x.rolling(20, min_periods=1).std() * np.sqrt(252 * 6.5))

    # Stochastic
    ll14 = _g('low').transform(lambda x: x.rolling(14, min_periods=1).min())
    hh14 = _g('high').transform(lambda x: x.rolling(14, min_periods=1).max())
    df['stoch_k'] = 100 * (df['close'] - ll14) / (hh14 - ll14).replace(0, 1)
    df['stoch_d'] = df.groupby('symbol')['stoch_k'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())

    # Williams %R
    df['williams_r'] = -100 * (hh14 - df['close']) / (hh14 - ll14).replace(0, 1)

    # Keltner / Donchian width
    kc_ma = _g('close').transform(lambda x: x.ewm(span=20, min_periods=1, adjust=False).mean())
    kc_atr = df['atr_14'].fillna(0)
    df['kc_width'] = 2 * kc_atr / kc_ma * 100
    dc_hi = _g('high').transform(lambda x: x.rolling(20, min_periods=1).max())
    dc_lo = _g('low').transform(lambda x: x.rolling(20, min_periods=1).min())
    df['dc_width'] = (dc_hi - dc_lo) / df['close'] * 100

    # VPT (Volume Price Trend)
    vpt = (df['volume'] * df['ret_1d'].fillna(0) / 100)
    df['vpt'] = vpt.groupby(df['symbol']).cumsum()

    # Ease of Movement
    distance = (df['high'] + df['low']) / 2 - (_g('high').shift(1) + _g('low').shift(1)) / 2
    df['eom'] = distance / (df['volume'] / 1000000).replace(0, np.nan)

    # Force Index
    df['fi'] = df['volume'] * df['ret_1d'].fillna(0)

    # Intraday time signals
    df['hour'] = df['datetime'].dt.floor('h')
    df['is_open'] = (df['hour'].dt.hour == 9).astype(float)
    df['is_close'] = (df['hour'].dt.hour == 14).astype(float)
    df['mid_session'] = ((df['hour'].dt.hour >= 10) & (df['hour'].dt.hour <= 12)).astype(float)

    return df

print('='*60)
print(' 60min + MTF Strategy (ALL symbols)')
print('='*60)

con = duckdb.connect(str(DB), read_only=True)

# ─── Step 1: Load raw 60min data for ALL symbols ───
print('\n[1] Loading raw 60min data...')
fs = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume
    FROM raw_market WHERE timeframe='60min'
    ORDER BY symbol, datetime
""").fetchdf()
print(f'  Loaded: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# Penny filter from daily
daily_prices = con.execute("""
    SELECT symbol, AVG(close) as avg_close FROM feature_store
    WHERE timeframe='1day' AND datetime >= '2024-01-01'
    GROUP BY symbol
""").fetchdf()
penny_syms = set(daily_prices[daily_prices['avg_close'] < 50]['symbol'])
fs = fs[~fs['symbol'].isin(penny_syms)].copy()
print(f'  After penny filter: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# ─── Step 2: Compute technical indicators ───
print('\n[2] Computing technical indicators...')
fs = compute_ta(fs)
print(f'  Done. Shape: {fs.shape}')

# ─── Step 3: MTF features ───
print('\n[3] Computing MTF features...')

# 15min rollup
print('  15min rollup features...')
m15 = con.execute("""
    SELECT symbol, datetime, close, volume, high, low
    FROM raw_market WHERE timeframe='15min'
    ORDER BY symbol, datetime
""").fetchdf()
m15['datetime'] = pd.to_datetime(m15['datetime'])
m15['hour'] = m15['datetime'].dt.floor('h')
m15['range_pct'] = (m15['high'] - m15['low']) / m15['close'] * 100
rollup = m15.groupby(['symbol', 'hour']).agg(
    m15_range_sum=('range_pct', 'sum'),
    m15_vol_sum=('volume', 'sum'),
    m15_n_bars=('close', 'count'),
).reset_index()
rollup['m15_range_avg'] = rollup['m15_range_sum'] / rollup['m15_n_bars'].replace(0, 1)
# Shift rollup by +1 hour: use PREVIOUS hour's 15min aggregates as features for current 60min bar
# (15min bars within hour H are only fully known at the END of the hour, so they predict hour H+1)
rollup['hour'] = rollup['hour'] + pd.Timedelta(hours=1)
print(f'    {len(rollup):,} rows ({rollup["symbol"].nunique()} symbols)')

# Daily context (from feature_store - already has all symbols)
print('  Daily context features...')
daily_ctx = con.execute("""
    SELECT symbol, datetime, close, volume,
           rsi_14, macd_hist, bb_width, adx, atr_14,
           ret_1d, ret_5d, ret_10d, close_vs_sma_20,
           close_vs_sma_50, vol_ratio_10
    FROM feature_store
    WHERE timeframe='1day' AND rsi_14 IS NOT NULL
    ORDER BY symbol, datetime
""").fetchdf()
daily_ctx['datetime'] = pd.to_datetime(daily_ctx['datetime'])
daily_ctx['date'] = daily_ctx['datetime'].dt.normalize()
daily_ctx['d_date'] = daily_ctx['date']
daily_ctx['date_shift'] = daily_ctx['date'] + pd.Timedelta(days=1)
d_feats = [c for c in daily_ctx.columns if c not in ('symbol', 'datetime', 'date', 'd_date', 'date_shift')]
d_rename = {c: f'd_{c}' for c in d_feats}
daily_ctx = daily_ctx.rename(columns=d_rename)
print(f'    {len(daily_ctx):,} rows ({daily_ctx["symbol"].nunique()} symbols)')

# ─── Step 4: Merge via DuckDB ───
print('\n[4] Merging MTF features...')
con.register('fs60', fs)
con.register('m15', rollup)
con.register('dly', daily_ctx)

m15_cols = [c for c in rollup.columns if c not in ('symbol', 'hour')]
m15_list = ','.join(f'm15."{c}"' for c in m15_cols)
d_cols = [c for c in daily_ctx.columns if c not in ('symbol', 'datetime', 'date', 'd_date', 'date_shift')]
d_list = ','.join(f'dly."{c}"' for c in d_cols)

fs = con.execute(f"""
    SELECT fs60.*, {m15_list}, {d_list}
    FROM fs60
    LEFT JOIN m15 ON fs60.symbol = m15.symbol AND fs60.hour = m15.hour
    LEFT JOIN dly ON fs60.symbol = dly.symbol AND fs60.date = dly.date_shift
    ORDER BY fs60.symbol, fs60.datetime
""").fetchdf()

for c in m15_cols + d_cols:
    if c in fs.columns:
        fs[c] = fs[c].fillna(0)

con.unregister('fs60'); con.unregister('m15'); con.unregister('dly')

print(f'  Merged {len(m15_cols)} 15min rollup + {len(d_cols)} daily features')

# ─── Step 5: Target ───
fs['target'] = fs.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * MULT
fs = fs.dropna(subset=['target'])

# ─── Step 6: Feature selection ───
print(f'\n[6] Preparing features...')
exclude = {'symbol', 'datetime', 'date', 'hour', 'target', 'vwap', 'vwap_distance',
           'open', 'high', 'low', 'close', 'volume'}
feat_cols = [c for c in fs.columns if c not in exclude]
feat_cols = [c for c in feat_cols if fs[c].dtype in ('float64', 'int64', 'float32', 'int32')]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.5]
n_m15 = sum(1 for c in feat_cols if c.startswith('m15_'))
n_d = sum(1 for c in feat_cols if c.startswith('d_'))
print(f'  Features: {len(feat_cols)} ({n_m15} 15min, {n_d} daily, {len(feat_cols)-n_m15-n_d} base)')

# ─── Step 7: Walkforward ───
print(f'\n[7] Walkforward training...')
fs['year'] = fs['datetime'].dt.year
years = sorted(fs['year'].unique())
windows = [(fs['year'].isin(years[:i]), fs['year'] == years[i])
           for i in range(2, len(years))]
print(f'  Windows: {len(windows)}')

all_results = []
last_valid = []; last_imp = None
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

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbosity=0,
                             device='cuda', tree_method='hist')
    model.fit(X_tr, y_tr)
    last_imp = model.feature_importances_
    pred = model.predict(X_te)

    r2 = r2_score(y_te, pred)
    corr = np.corrcoef(pred, y_te)[0, 1] if np.std(pred) > 1e-12 and np.std(y_te) > 1e-12 else 0
    da = ((pred > 0) == (y_te > 0)).mean()
    yr_str = test['year'].iloc[0] if 'year' in test.columns else str(wi)
    print(f'  [{wi+1:2d}/{len(windows)}] Test {yr_str}: '
          f'train={len(train):,} test={len(test):,} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

    for i in range(len(test)):
        all_results.append({
            'dt': test['datetime'].iloc[i], 'sym': test['symbol'].iloc[i],
            'act': y_te[i], 'pred': pred[i],
        })

# ─── Step 8: Overall metrics ───
print(f'\n[8] Results')
rd = pd.DataFrame(all_results)
print(f'  Total predictions: {len(rd):,}')
overall_r2 = r2_score(rd['act'], rd['pred'])
overall_corr = np.corrcoef(rd['pred'], rd['act'])[0, 1] if len(rd) > 2 else 0
overall_da = ((rd['pred'] > 0) == (rd['act'] > 0)).mean()
print(f'  Overall: R2={overall_r2:+.4f} Corr={overall_corr:+.4f} DirAcc={overall_da:.1%}')

try:
    imp = pd.DataFrame({'feature': last_valid, 'importance': last_imp})
    imp = imp.sort_values('importance', ascending=False)
    print('\n  Top 15 features:')
    for _, r in imp.head(15).iterrows():
        print(f'    {r["feature"]:<35s} {r["importance"]:.4f}')
except Exception:
    imp = pd.DataFrame()

# ─── Step 9: Backtest (daily P&L aggregation) ───
print(f'\n[9] Backtest (quintile long-short, daily P&L)')
rd = rd.sort_values('dt').reset_index(drop=True)
bt = []
for dt_uniq in sorted(rd['dt'].unique()):
    bar = rd[rd['dt'] == dt_uniq]
    if len(bar) < 20:
        continue
    bar = bar.sort_values('pred', ascending=False)
    n_ls = max(1, len(bar) // 5)
    long = bar.head(n_ls)
    short = bar.tail(n_ls)
    spread = long['act'].mean() - short['act'].mean()
    bt.append({'dt': dt_uniq, 'n': n_ls,
               'long_ret': long['act'].mean(), 'short_ret': short['act'].mean(),
               'spread': spread})

bt_df = pd.DataFrame(bt)
bt_df['date'] = pd.to_datetime(bt_df['dt']).dt.normalize()

# Daily P&L: average across bars within each day
daily = bt_df.groupby('date').agg(
    n_bars=('spread', 'count'), avg_n=('n', 'mean'),
    lo_ret=('long_ret', 'mean'), ls_ret=('spread', 'mean'),
).reset_index()
daily['lo_net'] = daily['lo_ret'] - cost_rt(TOTAL_POS) * 2 * MULT / daily['avg_n']
daily['ls_net'] = daily['ls_ret'] - cost_rt(TOTAL_POS) * 4 * MULT / daily['avg_n']

lc, lsr, lw, ldd = calc_metrics(daily['lo_ret'], n=252)
lnc, lns, lnw, lndd = calc_metrics(daily['lo_net'], n=252)
sc, ssr, sw, sdd = calc_metrics(daily['ls_ret'], n=252)
snc, sns, snw, sndd = calc_metrics(daily['ls_net'], n=252)

print(f'  Bars: {len(bt_df):,}  Days: {len(daily):,}  Avg positions/bar: {bt_df["n"].mean():.1f}')
print(f'  Long-only top-quintile (daily P&L):')
print(f'    Gross: CAGR={lc:+.1f}% Sharpe={lsr:.2f} WinRate={lw:.1f}% Mean={daily["lo_ret"].mean():+.4f}%')
print(f'    Net:   CAGR={lnc:+.1f}% Sharpe={lns:.2f} WinRate={lnw:.1f}% Mean={daily["lo_net"].mean():+.4f}%')
print(f'  Long-Short spread (daily P&L):')
print(f'    Gross: CAGR={sc:+.1f}% Sharpe={ssr:.2f} WinRate={sw:.1f}% Mean={daily["ls_ret"].mean():+.4f}%')
print(f'    Net:   CAGR={snc:+.1f}% Sharpe={sns:.2f} WinRate={snw:.1f}% Mean={daily["ls_net"].mean():+.4f}%')

pickle.dump({'rd': rd, 'bt': bt_df, 'imp': imp, 'feat_cols': feat_cols,
             'valid': last_valid}, open(OUT / 'results_60min_mtf.pkl', 'wb'))

n_syms = rd['sym'].nunique() if 'sym' in rd.columns else 0
print(f'\n{"="*60}')
print(f'  60min MTF Summary ({n_syms} symbols, {len(bt_df)} bars)')
print(f'{"="*60}')
print(f'  Features: {len(feat_cols)} ({n_m15} 15min, {n_d} daily, {len(feat_cols)-n_m15-n_d} base)')
print(f'  Walkforward DirAcc: {overall_da:.1%}  Corr: {overall_corr:+.4f}')
print(f'  Long-only CAGR={lnc:+.1f}% Sharpe={lns:.2f}')
print(f'  L/S spread CAGR={snc:+.1f}% Sharpe={sns:.2f}')
print(f'  Time: {time.time()-t0:.0f}s')
