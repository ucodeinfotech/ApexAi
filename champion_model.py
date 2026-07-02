"""CHAMPION: Weekly ML model — predict 5d returns, rebalance Friday, cost-aware"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math, gc
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'champion_model'; OUT.mkdir(exist_ok=True)
t0 = time.time()
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000
def cost_rt(ps):
    if ps <= 0: return 1.0
    b = max(BRK*ps, MIN_BRK)/ps*2; gst = b + EXCH*2 + SEBI*2
    return b + STT + EXCH*2 + SEBI*2 + STAMP + gst*GST + SLIP*2

print('=== Loading & merging data ===')
con = duckdb.connect(str(DB), read_only=True)
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
    FROM feature_store WHERE timeframe='1day' ORDER BY datetime
""").fetchdf()
print(f'Features: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

ms = con.execute("SELECT symbol, datetime, rs_vs_market, rs_vs_sector, rs_ratio_market, rs_ratio_sector, rs_momentum_10, rs_momentum_20 FROM market_structure WHERE timeframe='1day' ORDER BY datetime").fetchdf()
vix = con.execute("SELECT datetime, vix_close, vix_change, vix_ma_5, vix_ma_20, vix_zscore_20 FROM vix_data ORDER BY datetime").fetchdf()
dv = con.execute("SELECT symbol, date, delivery_pct FROM delivery_data ORDER BY symbol, date").fetchdf()
reg = con.execute("SELECT datetime, regime_label, regime_id FROM market_regimes ORDER BY datetime").fetchdf()
con.close()

def fix_dt(df, c='datetime'):
    d = pd.to_datetime(df[c]); df[c] = d.dt.tz_localize(None) if d.dt.tz is not None else d; return df

fs = fix_dt(fs); fs['date'] = fs['datetime'].dt.normalize(); fs['dow'] = fs['datetime'].dt.dayofweek; fs['month'] = fs['datetime'].dt.month
# Penny filter
avg_c = fs[fs['date'] >= '2024-06-24'].groupby('symbol')['close'].mean()
fs = fs[~fs['symbol'].isin(set(avg_c[avg_c < 50].index))].copy()
print(f'After penny filter: {fs["symbol"].nunique()} symbols')

# Daily target: next 5 trading days return
fs = fs.sort_values(['symbol','datetime']).reset_index(drop=True)
fs['fwd_5d'] = fs.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1) * 100
fs['fwd_1w'] = fs.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1) * 100

# Merge RS
ms = fix_dt(ms); ms['date'] = ms['datetime'].dt.normalize()
fs = fs.merge(ms.drop(columns=['datetime']), on=['symbol','date'], how='left')
for c in ['rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20']:
    fs[c] = fs[c].fillna(0)

# Merge VIX
vix = fix_dt(vix)
cd = fs['datetime'].dtype; vix['datetime'] = vix['datetime'].astype(cd)
fs = pd.merge_asof(fs.sort_values('datetime'), vix.sort_values('datetime'), on='datetime', direction='backward')
for c in ['vix_close','vix_change','vix_ma_5','vix_ma_20','vix_zscore_20']:
    fs[c] = fs[c].fillna(fs[c].median() if fs[c].notna().any() else 0)

# Delivery
dv['date'] = pd.to_datetime(dv['date'])
fs = fs.merge(dv, on=['symbol','date'], how='left')
fs['delivery_pct'] = fs.groupby('symbol')['delivery_pct'].ffill().fillna(0)

# Regimes
reg = fix_dt(reg)
fs = fs.merge(reg[['datetime','regime_label','regime_id']], on='datetime', how='left')
fs['regime_label'] = fs['regime_label'].fillna('sideways')
fs['regime_id'] = fs['regime_id'].fillna(0).astype(int)

print(f'Merged: {len(fs):,} rows')

print('\n=== Preprocessing ===')
EXCLUDE = {'symbol','datetime','date','dow','month','open','high','low','close','volume',
           'fwd_5d','fwd_1w','regime_label','regime_id'}
feat_cols = [c for c in fs.columns if c not in EXCLUDE and fs[c].dtype in ['float64','int64','float32','int32']]
feat_cols = [c for c in feat_cols if fs[c].notna().sum() > len(fs) * 0.9]

# Regime dummies
fs = pd.get_dummies(fs, columns=['regime_label'], prefix='regime')
reg_cols = [c for c in fs.columns if c.startswith('regime_')]
feat_cols = feat_cols + reg_cols

# Winsorize 99/1 (skip bool columns)
for c in feat_cols:
    if fs[c].dtype == bool: continue
    lo, hi = fs[c].quantile(0.01), fs[c].quantile(0.99)
    fs[c] = fs[c].clip(lo, hi)

fs = fs.dropna(subset=feat_cols + ['fwd_5d'])
print(f'Clean: {len(fs):,} rows, features: {len(feat_cols)}')

print('\n=== Feature Selection ===')
sel = fs.sample(min(100000, len(fs)), random_state=42) if len(fs) > 100000 else fs.copy()
X_s = sel[feat_cols].fillna(0).values; y_s = sel['fwd_5d'].values
ss = StandardScaler(); X_ss = ss.fit_transform(X_s)

# MI
mi = mutual_info_regression(X_ss, y_s, random_state=42, n_neighbors=5)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
mi_k = set(mi_s[mi_s > mi_s.max() * 0.05].index)
print(f'  MI: {len(mi_k)}/{len(feat_cols)}')

# XGB importance
mx = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0,
                       device='cuda', tree_method='hist')
mx.fit(X_ss, y_s)
fi_s = pd.Series(mx.feature_importances_, index=feat_cols).sort_values(ascending=False)
fi_k = set(fi_s[fi_s > fi_s.max() * 0.05].index)
print(f'  XGB: {len(fi_k)}/{len(feat_cols)}')

# Boruta
X_b = np.hstack([X_ss, np.random.permutation(X_ss.T).T]); n_r = X_ss.shape[1]
mb = xgb.XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, subsample=0.8,
                       random_state=42, n_jobs=-1, verbosity=0, device='cuda', tree_method='hist')
mb.fit(X_b, y_s)
ri = mb.feature_importances_[:n_r]; sm = mb.feature_importances_[n_r:].max()
bk = set([feat_cols[i] for i, v in enumerate(ri) if v > sm])
print(f'  Boruta: {len(bk)}/{len(feat_cols)}')

# Consensus: features passing 2 of 3
u2 = (mi_k & fi_k) | (mi_k & bk) | (fi_k & bk)
final = list(u2 if len(u2) > 10 else (mi_k & fi_k))
if len(final) < 10: final = sorted(mi_k, key=lambda c: mi_s[c], reverse=True)[:30]
print(f'  Final features: {len(final)}')

pickle.dump({'mi':mi_s,'fi':fi_s,'bk':bk,'final':final}, open(OUT/'selection.pkl','wb'))

print('\n=== Walkforward (weekly rebalance) ===')
fs['year'] = fs['datetime'].dt.year
years = sorted(fs['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
all_preds, all_actual = [], []
yearly = {}

for wi, (ty, test_yr) in enumerate(windows):
    tr_raw = fs[fs['year'].isin(ty)].copy(); te = fs[fs['year'] == test_yr].copy()
    if len(te) < 100: continue
    embargo = te['datetime'].min() - pd.Timedelta(days=7)
    tr = tr_raw[tr_raw['datetime'] < embargo].copy()
    v = [c for c in final if c in tr.columns and c in te.columns]
    v = [c for c in v if tr[c].notna().all() and tr[c].std() > 1e-10]
    if len(v) > 20:
        cs = tr[v].sample(min(20000, len(tr)), random_state=42).corr().abs()
        up = cs.where(np.triu(np.ones(cs.shape), k=1).astype(bool))
        ds = set()
        for col in up.columns:
            if col in ds: continue
            ds.update(list(up.index[up[col] > 0.95]))
        v = [c for c in v if c not in ds]
    tr = tr.dropna(subset=v+['fwd_5d']); te = te.dropna(subset=v+['fwd_5d'])
    if len(tr) < 500 or len(te) < 50: continue
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[v].values); X_te = scaler.transform(te[v].values)
    y_tr = tr['fwd_5d'].values; y_te = te['fwd_5d'].values
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.7,
                          reg_alpha=0.1, reg_lambda=0.1,
                          random_state=42, n_jobs=-1, verbosity=0,
                          device='cuda', tree_method='hist')
    m.fit(X_tr, y_tr); pred = m.predict(X_te)
    r2 = r2_score(y_te, pred); corr = np.corrcoef(pred, y_te)[0,1] if np.std(pred) > 1e-12 else 0
    da = ((pred>0)==(y_te>0)).mean()
    yearly[int(test_yr)] = {'r2':r2,'corr':corr,'dir_acc':da,'n_train':len(tr),'n_test':len(te),'n_feats':len(v)}
    print(f'  [{wi+1:2d}/{len(windows)}] {test_yr}: train={len(tr):,} test={len(te):,} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%} feats={len(v)}')
    all_preds.extend(pred.tolist()); all_actual.extend(y_te.tolist())

ap = np.array(all_preds); aa = np.array(all_actual)
print(f'\n{"="*55}\nOVERALL PREDICTION\n{"="*55}')
print(f'Predictions: {len(ap):,}')
print(f'R2:    {r2_score(aa, ap):+.4f}')
print(f'Corr:  {np.corrcoef(ap, aa)[0,1]:+.4f}')
print(f'DirAcc:{((ap>0)==(aa>0)).mean():.1%}')

print(f'\n{"="*55}\nWEEKLY REBALANCE BACKTEST\n{"="*55}')
# Build weekly predictions dataset
td = pd.DataFrame({'date':fs[fs['year'].isin([w[1] for w in windows])]['datetime'].values[:len(ap)] if len(ap) > 0 else [],
                   'pred':ap, 'actual':aa})
if len(td) == 0:
    td = pd.DataFrame({'date': pd.to_dataltime([]), 'pred': [], 'actual': []})
# Actually build td properly from the data
td_list = []
for wi, (ty, test_yr) in enumerate(windows):
    te = fs[fs['year'] == test_yr].copy()
    if len(te) < 100: continue
    embargo = te['datetime'].min() - pd.Timedelta(days=7)
    tr = fs[fs['year'].isin(ty) & (fs['datetime'] < embargo)].copy()
    v = [c for c in final if c in tr.columns and c in te.columns]
    v = [c for c in v if tr[c].notna().all() and tr[c].std() > 1e-10]
    tr = tr.dropna(subset=v+['fwd_5d']); te = te.dropna(subset=v+['fwd_5d'])
    if len(tr) < 500 or len(te) < 50: continue
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[v].values); X_te = scaler.transform(te[v].values)
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1,
                          random_state=42, n_jobs=-1, verbosity=0, device='cuda', tree_method='hist')
    m.fit(X_tr, tr['fwd_5d'].values); pred = m.predict(X_te)
    for i in range(len(te)):
        td_list.append({'date':te['datetime'].iloc[i], 'symbol':te['symbol'].iloc[i],
                        'pred':pred[i], 'actual':te['fwd_5d'].iloc[i],
                        'close':te['close'].iloc[i]})

td = pd.DataFrame(td_list)
print(f'Predictions with symbols: {len(td):,}')

# Weekly rebalance: Friday close
td['date_norm'] = td['date'].dt.normalize()
td['dow'] = td['date'].dt.dayofweek
# Find Fridays
td = td.sort_values(['date_norm','symbol']).reset_index(drop=True)
# For each Friday, rank by prediction → buy top 5, hold 1 week
td['week'] = td['date_norm'].dt.isocalendar().week.astype(int)
td['year_wk'] = td['date_norm'].dt.year.astype(str) + '_' + td['week'].astype(str).str.zfill(2)

weeks = sorted(td['year_wk'].unique())
bt = []; prev5 = None

for wk in weeks:
    wk_data = td[td['year_wk'] == wk].copy()
    if len(wk_data) < 20: continue
    wk_data = wk_data.sort_values('date_norm')
    friday = wk_data['date_norm'].iloc[-1]
    # Only trade on Friday
    if wk_data[wk_data['date_norm'] == friday].iloc[0]['dow'] != 4:
        continue  # not a Friday, skip
    reb_day = wk_data[wk_data['date_norm'] == friday].sort_values('pred', ascending=False)
    if len(reb_day) < 10: continue
    # Top 5
    buys5 = set(reb_day.head(5)['symbol'].tolist())
    # Forward return: we have fwd_5d which is already computed
    picks = reb_day[reb_day['symbol'].isin(buys5)]
    if len(picks) == 0: continue
    avg_ret = picks['actual'].mean()
    # Turnover
    to5 = 1.0 if prev5 is None else (len(buys5-prev5)+len(prev5-buys5))/max(len(buys5|prev5),1)
    cost5 = cost_rt(TOTAL_POS/len(picks)) * to5 * 100
    bt.append({'date': friday, 'ret': avg_ret, 'net': avg_ret-cost5, 'to': to5, 'n': len(picks), 'cost': cost5})
    prev5 = buys5

btf = pd.DataFrame(bt)
if len(btf) > 0:
    def cm(s, n=52):
        if len(s)<5 or s.std()==0: return (0,0,0,0)
        c = ((1+s/100).prod()**(n/len(s))-1)*100; sh = s.mean()/s.std()*math.sqrt(n)
        wr = (s>0).mean()*100; dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
        return (c, sh, wr, dd)
    g = btf['ret']; n = btf['net']
    gc, gs, gw, gdd = cm(g); nc, ns, nw, ndd = cm(n)
    print(f'Trades: {len(btf)}')
    print(f'{"":25s} {"Gross":>12s} {"Net":>12s}')
    print(f'{"CAGR":25s} {gc:>+11.1f}% {nc:>+11.1f}%')
    print(f'{"Sharpe":25s} {gs:>11.2f} {ns:>11.2f}')
    print(f'{"WinRate":25s} {gw:>10.1f}% {nw:>10.1f}%')
    print(f'{"MaxDD":25s} {gdd:>10.1f}% {ndd:>10.1f}%')
    print(f'{"Mean weekly ret":25s} {g.mean():>+10.3f}% {n.mean():>+10.3f}%')
    print(f'{"Avg turnover":25s} {btf["to"].mean():>10.1%}')
    print(f'{"Avg cost/trade":25s} {btf["cost"].mean():>9.3f}%')
    btf.to_csv(OUT/'backtest_weekly_top5.csv', index=False)

# Also try LS-D10/D10 and Long-Only D9
print(f'\n{"="*55}')
print('ADDITIONAL STRATEGIES')
print(f'{"="*55}')
for strat_name, top_n, short in [('Long-Only D9', 9, False), ('LS-D10/D10', 10, True)]:
    bt2 = []; prev = None
    for wk in weeks:
        wk_data = td[td['year_wk'] == wk].copy()
        if len(wk_data) < 20: continue
        wk_data = wk_data.sort_values('date_norm')
        friday = wk_data['date_norm'].iloc[-1]
        if wk_data[wk_data['date_norm'] == friday].iloc[0]['dow'] != 4: continue
        reb_day = wk_data[wk_data['date_norm'] == friday].sort_values('pred', ascending=False)
        if len(reb_day) < 20: continue
        n = min(top_n, len(reb_day)//10)
        if n < 1: continue
        if short:
            top = set(reb_day.head(n)['symbol'].tolist())
            bot = set(reb_day.tail(n)['symbol'].tolist())
            all_picks = top | bot
            long_ret = reb_day[reb_day['symbol'].isin(top)]['actual'].mean()
            short_ret = reb_day[reb_day['symbol'].isin(bot)]['actual'].mean()
            avg_ret = (long_ret - short_ret) / 2
            cost_mult = 2
        else:
            all_picks = set(reb_day.head(n)['symbol'].tolist())
            picks_df = reb_day[reb_day['symbol'].isin(all_picks)]
            avg_ret = picks_df['actual'].mean()
            cost_mult = 1
        to = 1.0 if prev is None else (len(all_picks-prev)+len(prev-all_picks))/max(len(all_picks|prev),1)
        cost = cost_rt(TOTAL_POS/n) * to * cost_mult * 100
        bt2.append({'date':friday,'ret':avg_ret,'net':avg_ret-cost,'to':to,'n':n,'cost':cost})
        prev = all_picks
    btf2 = pd.DataFrame(bt2)
    if len(btf2) > 0:
        g2 = btf2['ret']; n2 = btf2['net']
        gc2, gs2, gw2, gdd2 = cm(g2); nc2, ns2, nw2, ndd2 = cm(n2)
        print(f'\n{strat_name}:')
        print(f'  Trades: {len(btf2)}')
        print(f'  Gross: CAGR={gc2:+.1f}% Sharpe={gs2:.2f} WinRate={gw2:.1f}%')
        print(f'  Net:   CAGR={nc2:+.1f}% Sharpe={ns2:.2f} WinRate={nw2:.1f}%')
        print(f'  Avg cost: {btf2["cost"].mean():.3f}%')
        btf2.to_csv(OUT/f'backtest_{strat_name.replace(" ","_").replace("/","_")}.csv', index=False)

pickle.dump({'yearly': yearly, 'td': td, 'bt_top5': btf if len(btf) > 0 else None,
             'final_feats': final, 'time': time.time()-t0}, open(OUT/'results.pkl','wb'))
print(f'\nSaved to {OUT} [{time.time()-t0:.0f}s]')
