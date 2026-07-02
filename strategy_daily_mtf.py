"""
Daily Intraday Strategy with Multi-Timeframe Features.
Covers ALL 475 symbols by using daily frequency + 15min rollup + daily context.
"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_daily_mtf'
OUT.mkdir(exist_ok=True)
t0 = time.time()

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000; MULT=100

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

def calc_metrics(s, n=252):
    if len(s)<5 or s.std()==0: return 0,0,0,0
    cagr = (1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh = s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr = (s>0).mean()
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd

def compute_15min_rollup(con):
    """Aggregate 15min bars into daily rollup features."""
    print("  15min rollup features...")
    m15 = con.execute("""
        SELECT symbol, datetime, close, volume, rsi_14, bb_width,
               high, low, adx
        FROM feature_store
        WHERE timeframe='15min' AND rsi_14 IS NOT NULL
        ORDER BY symbol, datetime
    """).fetchdf()
    if len(m15) < 1000:
        return pd.DataFrame()
    m15['datetime'] = pd.to_datetime(m15['datetime'])
    m15['date'] = m15['datetime'].dt.normalize()
    m15['range_pct'] = (m15['high'] - m15['low']) / m15['close'] * 100
    m15['prev_close'] = m15.groupby('symbol')['close'].shift(1)
    m15['ret'] = (m15['close'] / m15['prev_close'] - 1) * 100

    rollup = m15.groupby(['symbol', 'date']).agg(
        m15_rsi_mean=('rsi_14', 'mean'),
        m15_rsi_std=('rsi_14', 'std'),
        m15_range_sum=('range_pct', 'sum'),
        m15_vol_sum=('volume', 'sum'),
        m15_bbw_mean=('bb_width', 'mean'),
        m15_adx_mean=('adx', 'mean'),
        m15_ret_std=('ret', 'std'),
        m15_n_bars=('close', 'count'),
    ).reset_index()
    print(f"    {len(rollup):,} rows ({rollup['symbol'].nunique()} symbols)")
    return rollup

def compute_daily_context(con):
    """Daily feature set (already in feature_store, just rebrand as d_)."""
    print("  Daily context features...")
    daily = con.execute("""
        SELECT symbol, datetime, close, volume,
               rsi_14, macd_hist, bb_width, adx, atr_14,
               ret_1d, ret_5d, ret_10d, close_vs_sma_20,
               close_vs_sma_50, vol_ratio_10, zscore_20,
               hv_20, kc_width, dc_width
        FROM feature_store
        WHERE timeframe='1day' AND rsi_14 IS NOT NULL
        ORDER BY symbol, datetime
    """).fetchdf()
    daily['datetime'] = pd.to_datetime(daily['datetime'])
    daily['date'] = daily['datetime'].dt.normalize()
    daily['prev_close'] = daily.groupby('symbol')['close'].shift(1)
    daily['d_ret'] = (daily['close'] / daily['prev_close'] - 1) * 100
    d_cols = {c: f'd_{c}' for c in daily.columns
              if c not in ('symbol', 'datetime', 'date', 'prev_close')}
    daily = daily.rename(columns=d_cols)
    daily = daily.drop(columns=['d_prev_close'], errors='ignore')
    print(f"    {len(daily):,} rows ({daily['symbol'].nunique()} symbols)")
    return daily

print('='*60)
print(' Daily MTF Strategy (all 475 symbols)')
print('='*60)

con = duckdb.connect(str(DB), read_only=True)

# ─── Step 1: Load daily data (ALL symbols) ───
print('\n[1] Loading daily features...')
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
    FROM feature_store WHERE timeframe='1day'
    ORDER BY datetime
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

# ─── Step 2: MTF features ───
print('\n[2] Computing MTF features...')
rollup_15min = compute_15min_rollup(con)
daily_ctx = compute_daily_context(con)

# ─── Step 3: Merge via DuckDB ───
print('\n[3] Merging MTF features...')
fs['date'] = pd.to_datetime(fs['datetime']).dt.normalize()
con.register('fs_d', fs)

m15_cols = []; d_cols = []

if len(rollup_15min) > 0 and len(daily_ctx) > 0:
    con.register('m15', rollup_15min)
    con.register('dly', daily_ctx)
    m15_cols = [c for c in rollup_15min.columns if c not in ('symbol', 'date')]
    d_cols = [c for c in daily_ctx.columns if c not in ('symbol', 'datetime', 'date', 'prev_close')]
    m15_list = ','.join(f'm15."{c}"' for c in m15_cols)
    d_list = ','.join(f'dly."{c}"' for c in d_cols)
    fs = con.execute(f"""
        SELECT fs_d.*, {m15_list}, {d_list}
        FROM fs_d
        LEFT JOIN m15 ON fs_d.symbol = m15.symbol AND fs_d.date = m15.date
        LEFT JOIN dly ON fs_d.symbol = dly.symbol AND fs_d.date = dly.date
        ORDER BY fs_d.symbol, fs_d.datetime
    """).fetchdf()
elif len(rollup_15min) > 0:
    con.register('m15', rollup_15min)
    m15_cols = [c for c in rollup_15min.columns if c not in ('symbol', 'date')]
    m15_list = ','.join(f'm15."{c}"' for c in m15_cols)
    fs = con.execute(f"""
        SELECT fs_d.*, {m15_list}
        FROM fs_d
        LEFT JOIN m15 ON fs_d.symbol = m15.symbol AND fs_d.date = m15.date
        ORDER BY fs_d.symbol, fs_d.datetime
    """).fetchdf()
elif len(daily_ctx) > 0:
    con.register('dly', daily_ctx)
    d_cols = [c for c in daily_ctx.columns if c not in ('symbol', 'datetime', 'date', 'prev_close')]
    d_list = ','.join(f'dly."{c}"' for c in d_cols)
    fs = con.execute(f"""
        SELECT fs_d.*, {d_list}
        FROM fs_d
        LEFT JOIN dly ON fs_d.symbol = dly.symbol AND fs_d.date = dly.date
        ORDER BY fs_d.symbol, fs_d.datetime
    """).fetchdf()

for c in m15_cols + d_cols:
    if c in fs.columns:
        fs[c] = fs[c].fillna(0)

con.unregister('fs_d')
for tbl in ('m15', 'dly'):
    try: con.unregister(tbl)
    except: pass

fs = fs.sort_values(['symbol', 'datetime']).reset_index(drop=True)
fs['target'] = fs.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1) * MULT
fs = fs.dropna(subset=['target'])
if m15_cols:
    print(f'  Merged {len(m15_cols)} 15min rollup features')
if d_cols:
    print(f'  Merged {len(d_cols)} daily context features')

# ─── Step 4: Features ───
print('\n[4] Preparing features...')
exclude = {'symbol', 'datetime', 'date', 'target', 'open', 'high', 'low', 'close', 'volume',
           'prev_close'}
feat_cols = [c for c in fs.columns if c not in exclude]
feat_cols = [c for c in feat_cols if fs[c].dtype in ('float64', 'int64', 'float32', 'int32')]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.5]
n_m15 = sum(1 for c in feat_cols if c.startswith('m15_'))
n_d = sum(1 for c in feat_cols if c.startswith('d_'))
print(f'  Features: {len(feat_cols)} ({n_m15} 15min, {n_d} daily, {len(feat_cols)-n_m15-n_d} base)')

# ─── Step 5: Walkforward ───
print('\n[5] Walkforward training...')
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

# ─── Step 6: Overall metrics ───
print(f'\n[6] Results')
rd = pd.DataFrame(all_results)
print(f'  Total predictions: {len(rd):,}')
overall_r2 = r2_score(rd['act'], rd['pred'])
overall_corr = np.corrcoef(rd['pred'], rd['act'])[0, 1] if len(rd) > 2 else 0
overall_da = ((rd['pred'] > 0) == (rd['act'] > 0)).mean()
print(f'  Overall: R2={overall_r2:+.4f} Corr={overall_corr:+.4f} DirAcc={overall_da:.1%}')

# Feature importance
try:
    imp = pd.DataFrame({'feature': last_valid, 'importance': last_imp})
    imp = imp.sort_values('importance', ascending=False)
    print('\n  Top 15 features:')
    for _, r in imp.head(15).iterrows():
        print(f'    {r["feature"]:<35s} {r["importance"]:.4f}')
except Exception:
    imp = pd.DataFrame()

# ─── Step 7: Backtest (quintile long-short) ───
print(f'\n[7] Backtest (quintile long-short, daily)')
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
    long_ret = long['act'].mean()
    short_ret = short['act'].mean()
    spread = long_ret - short_ret
    cost_each = cost_rt(TOTAL_POS) * MULT * 2  # enter + exit per side
    net_spread = spread - (cost_each * 2) / n_ls  # 2 sides: long + short
    bt.append({'dt': dt_uniq, 'n': n_ls,
               'long_ret': long_ret, 'short_ret': short_ret,
               'spread': spread, 'net_spread': net_spread})

bt_df = pd.DataFrame(bt)
daily = bt_df.copy()
daily['lo_net'] = daily['long_ret'] - cost_rt(TOTAL_POS) * 2 * MULT / daily['n']
daily['ls_net'] = daily['spread'] - cost_rt(TOTAL_POS) * 4 * MULT / daily['n']

lg, ln = daily['long_ret'], daily['lo_net']
sg, sn = daily['spread'], daily['ls_net']

lc, lsr, lw, ldd = calc_metrics(lg)
lnc, lns, lnw, lndd = calc_metrics(ln)
sc, ssr, sw, sdd = calc_metrics(sg)
snc, sns, snw, sndd = calc_metrics(sn)

print(f'  Days: {len(daily)}  Avg positions/day: {daily["n"].mean():.1f}')
print(f'  Long-only top-quintile:')
print(f'    Gross: CAGR={lc:+.1f}% Sharpe={lsr:.2f} WinRate={lw:.1f}% Mean={lg.mean():+.4f}%')
print(f'    Net:   CAGR={lnc:+.1f}% Sharpe={lns:.2f} WinRate={lnw:.1f}% Mean={ln.mean():+.4f}%')
print(f'  Long-Short spread:')
print(f'    Gross: CAGR={sc:+.1f}% Sharpe={ssr:.2f} WinRate={sw:.1f}% Mean={sg.mean():+.4f}%')
print(f'    Net:   CAGR={snc:+.1f}% Sharpe={sns:.2f} WinRate={snw:.1f}% Mean={sn.mean():+.4f}%')

# Save
pickle.dump({'rd': rd, 'bt': bt_df, 'imp': imp, 'feat_cols': feat_cols,
             'valid': last_valid, 'daily_metrics': {
                 'lo_gross': (lc, lsr, lw, ldd), 'lo_net': (lnc, lns, lnw, lndd),
                 'ls_gross': (sc, ssr, sw, sdd), 'ls_net': (snc, sns, snw, sndd),
             }}, open(OUT / 'results_daily_mtf.pkl', 'wb'))

print(f'\n{"="*60}')
print(f'  Daily MTF Summary (all {rd["sym"].nunique() if "sym" in rd.columns else "?"} symbols)')
print(f'{"="*60}')
print(f'  MTF features: {n_m15} (15min) + {n_d} (daily)')
print(f'  Total features: {len(feat_cols)}')
print(f'  Walkforward DirAcc: {overall_da:.1%}  Corr: {overall_corr:+.4f}')
print(f'  Long-only top-quintile (daily): CAGR={lnc:+.1f}% Sharpe={lns:.2f}')
print(f'  Long-Short spread (daily):     CAGR={snc:+.1f}% Sharpe={sns:.2f}')
print(f'  Time: {time.time()-t0:.0f}s')
