# Full Time Series Analysis
# ACF/PACF, Stationarity, Granger Causality, Rolling Window Optimization,
# GMM Regimes, Seasonality, Cross-sectional Spread
import duckdb, pandas as pd, numpy as np, time, warnings, json, itertools
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'ts_analysis_output'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' Time Series Analysis')
print('='*60)

con = duckdb.connect(str(DB), read_only=True)

# ── 1. Load daily gainer rate time series ──
print('\n[1] Loading daily data for time series analysis...')
fs = con.execute("""
    SELECT symbol, datetime::DATE as date, open, high, low, close, volume,
           ret_1d, range_5, hv_20, rsi_14, vol_ratio_5, bb_width, adx
    FROM feature_store WHERE timeframe='1day'
    ORDER BY symbol, date
""").fetchdf()
print(f'  Loaded: {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# Compute target
fs = fs.sort_values(['symbol', 'date'])
fs['next_close'] = fs.groupby('symbol')['close'].shift(-1)
fs['next_open'] = fs.groupby('symbol')['open'].shift(-1)
fs['target_ret'] = fs['next_close'] / fs['next_open'] - 1
fs['target'] = (fs['target_ret'] > 0.02).astype(int)
fs = fs.dropna(subset=['target'])

# Daily market-wide gainer rate
daily = fs.groupby('date').agg(
    gainer_rate=('target', 'mean'),
    n_stocks=('symbol', 'count'),
    avg_ret=('ret_1d', 'mean'),
    avg_range=('range_5', 'mean'),
    avg_hv=('hv_20', 'mean'),
).reset_index()
daily = daily.sort_values('date')
daily['gainer_rate'] = daily['gainer_rate'].fillna(0)
print(f'  Daily series: {len(daily)} days')

# ── 2. ACF / PACF ──
print('\n[2] ACF/PACF of daily gainer rate...')
ts = daily['gainer_rate'].values

def acf(series, nlags=40):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series, ddof=0)
    if var == 0:
        return np.zeros(nlags+1)
    c0 = np.sum((series - mean)**2) / n
    acf_vals = np.ones(nlags+1)
    for k in range(1, nlags+1):
        acf_vals[k] = np.sum((series[:-k] - mean) * (series[k:] - mean)) / n / c0
    return acf_vals

def pacf(series, nlags=40):
    """Compute PACF using Durbin-Levinson recursion."""
    acf_vals = acf(series, nlags)
    pacf_vals = np.zeros(nlags+1)
    pacf_vals[0] = 1.0
    for k in range(1, nlags+1):
        if k == 1:
            pacf_vals[k] = acf_vals[1]
        else:
            phi = np.zeros(k+1)
            phi[1] = acf_vals[1]
            for i in range(2, k+1):
                num = acf_vals[i] - np.sum(phi[1:i] * acf_vals[i-1:0:-1])
                denom = 1 - np.sum(phi[1:i] * acf_vals[1:i])
                phi[i] = num / denom if abs(denom) > 1e-10 else 0
                for j in range(1, i):
                    phi[j] = phi[j] - phi[i] * phi[i-j]
            pacf_vals[k] = phi[k]
    return pacf_vals

acf_vals = acf(ts, 40)
pacf_vals = pacf(ts, 40)

# Significance threshold (95% CI)
se = 1.96 / np.sqrt(len(ts))
sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > se]
sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > se]

print(f'  Significant ACF lags: {sig_acf[:15]}')
print(f'  Significant PACF lags: {sig_pacf[:15]}')
print(f'  Key lags:')
for lag in [1, 2, 3, 5, 10, 16, 17, 18, 20]:
    if lag < len(acf_vals):
        print(f'    Lag {lag:2d}: ACF={acf_vals[lag]:.4f}  PACF={pacf_vals[lag]:.4f}')

# Save ACF/PACF
acf_df = pd.DataFrame({'lag': range(41), 'acf': acf_vals, 'pacf': pacf_vals, 'significant_acf': [abs(acf_vals[i]) > se for i in range(41)], 'significant_pacf': [abs(pacf_vals[i]) > se for i in range(41)]})
acf_df.to_csv(OUT / 'acf_pacf.csv', index=False)

# ── 3. Stationarity tests ──
print('\n[3] Stationarity Tests (ADF + KPSS)...')
from statsmodels.tsa.stattools import adfuller, kpss

feature_list = ['gainer_rate', 'avg_ret', 'avg_hv', 'avg_range']

stationarity_results = []
for feat in feature_list:
    series = daily[feat].dropna().values
    if len(series) < 100:
        continue
    # ADF
    try:
        adf_stat, adf_pval, _, _, adf_cv, _ = adfuller(series, maxlag=20)
        adf_stationary = adf_pval < 0.05
    except:
        adf_stat, adf_pval, adf_stationary = np.nan, np.nan, False
    # KPSS
    try:
        kpss_stat, kpss_pval, _, kpss_cv = kpss(series, regression='c', nlags='auto')
        kpss_stationary = kpss_pval >= 0.05
    except:
        kpss_stat, kpss_pval, kpss_stationary = np.nan, np.nan, True

    stationarity_results.append({
        'series': feat,
        'adf_stat': float(adf_stat), 'adf_pval': float(adf_pval), 'adf_stationary': bool(adf_stationary),
        'kpss_stat': float(kpss_stat), 'kpss_pval': float(kpss_pval), 'kpss_stationary': bool(kpss_stationary),
        'interpretation': 'Stationary' if adf_stationary and kpss_stationary else 'Non-stationary' if not adf_stationary else 'Borderline',
    })

# Per-symbol ADF on key features
print('  Per-symbol stationarity (sampled)...')
sym_features = ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5']
sym_adf_results = []
syms_sample = fs['symbol'].unique()[:50]
for sym in syms_sample:
    sym_data = fs[fs['symbol'] == sym].sort_values('date')
    for feat in sym_features:
        series = sym_data[feat].dropna().values
        if len(series) < 100:
            continue
        try:
            adf_pval = adfuller(series, maxlag=10, autolag='AIC')[1]
            sym_adf_results.append({
                'symbol': sym, 'feature': feat, 'adf_pval': float(adf_pval),
                'stationary': bool(adf_pval < 0.05)
            })
        except:
            pass

sym_adf_df = pd.DataFrame(sym_adf_results)
if len(sym_adf_df) > 0:
    pct_stationary = sym_adf_df.groupby('feature')['stationary'].mean() * 100
    for feat in sym_features:
        pct = pct_stationary.get(feat, 0)
        print(f'    {feat}: {pct:.0f}% symbols stationary')

stat_df = pd.DataFrame(stationarity_results)
stat_df.to_csv(OUT / 'stationarity_tests.csv', index=False)
print('  Market-level stationarity:')
for _, r in stat_df.iterrows():
    print(f'    {r["series"]:<15s} ADF p={r["adf_pval"]:.6f} ({r["adf_stationary"]})  KPSS p={r["kpss_pval"]:.6f} ({r["kpss_stationary"]})  -> {r["interpretation"]}')

# ── 4. Granger Causality ──
print('\n[4] Granger Causality Tests...')
from statsmodels.tsa.stattools import grangercausalitytests

gc_features = ['ret_1d', 'range_5', 'hv_20', 'rsi_14', 'vol_ratio_5', 'bb_width', 'adx']
maxlag = 5

gc_all_results = []
for sym in fs['symbol'].unique():
    sym_data = fs[fs['symbol'] == sym].sort_values('date').dropna(subset=['target'] + gc_features)
    if len(sym_data) < 100:
        continue
    for feat in gc_features:
        data = sym_data[[feat, 'target']].values
        if np.isnan(data).any():
            continue
        try:
            gc_res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            best_pval = min(gc_res[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1))
            gc_all_results.append({
                'symbol': sym, 'feature': feat, 'best_pval': float(best_pval),
                'significant': bool(best_pval < 0.05)
            })
        except:
            pass

gc_df = pd.DataFrame(gc_all_results)
gc_pct = gc_df.groupby('feature')['significant'].mean() * 100
gc_df.to_csv(OUT / 'granger_causality.csv', index=False)
print('  % symbols where feature Granger-causes target (p<0.05):')
for feat in gc_features:
    pct = gc_pct.get(feat, 0)
    print(f'    {feat:<15s}: {pct:.0f}%')

# ── 5. Rolling Window Optimization ──
print('\n[5] Rolling Window Optimization...')
windows = [3, 5, 10, 15, 21, 30]
opt_features = [('ret_1d', 'target'), ('range_5', 'target'), ('hv_20', 'target'), ('vol_ratio_5', 'target')]

win_results = []
for sym in fs['symbol'].unique()[:50]:
    sym_data = fs[fs['symbol'] == sym].sort_values('date').dropna(subset=['target'])
    if len(sym_data) < 200:
        continue
    for feat, target_col in opt_features:
        if feat not in sym_data.columns:
            continue
        for w in windows:
            rolled = sym_data[feat].rolling(w, min_periods=w//2).mean()
            corr = rolled.corr(sym_data[target_col])
            win_results.append({
                'symbol': sym, 'feature': feat, 'window': w,
                'corr': float(corr) if not np.isnan(corr) else 0
            })

win_df = pd.DataFrame(win_results)
win_agg = win_df.groupby(['feature', 'window'])['corr'].agg(['mean', 'std']).reset_index()
win_agg.columns = ['feature', 'window', 'corr_mean', 'corr_std']
win_agg.to_csv(OUT / 'window_optimization.csv', index=False)

print('  Optimal windows (highest abs corr):')
for feat in ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5']:
    feat_wins = win_agg[win_agg['feature'] == feat]
    if len(feat_wins) > 0:
        best = feat_wins.loc[feat_wins['corr_mean'].abs().idxmax()]
        print(f'    {feat:<15s}: window={int(best["window"])}  corr={best["corr_mean"]:+.4f}')

# ── 6. GMM Regime Clustering ──
print('\n[6] GMM Regime Clustering...')
# Use daily market-level features
regime_feats = daily[['avg_ret', 'avg_hv', 'avg_range']].dropna().values
scaler_mean = regime_feats.mean(axis=0)
scaler_std = regime_feats.std(axis=0)
regime_feats_scaled = (regime_feats - scaler_mean) / (scaler_std + 1e-10)

# Find optimal n_components
bic_scores = []
for n in range(2, 7):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(regime_feats_scaled)
    bic_scores.append({'n_components': n, 'bic': gmm.bic(regime_feats_scaled)})

bic_df = pd.DataFrame(bic_scores)
best_n = bic_df.loc[bic_df['bic'].idxmin(), 'n_components']
print(f'  Optimal regimes: {int(best_n)} (BIC={bic_df["bic"].min():.0f})')

gmm = GaussianMixture(n_components=int(best_n), random_state=42, n_init=10)
regime_labels_raw = gmm.fit_predict(regime_feats_scaled)
daily['regime'] = np.nan
daily.loc[daily[['avg_ret', 'avg_hv', 'avg_range']].dropna().index, 'regime'] = regime_labels_raw
daily = daily.dropna(subset=['regime'])
daily['regime'] = daily['regime'].astype(int)

# Map regimes to interpretable labels
regime_profiles = daily.groupby('regime')[['gainer_rate', 'avg_ret', 'avg_hv', 'avg_range']].mean()
print('  Regime profiles:')
regime_labels = {}
for r in sorted(regime_profiles.index):
    prof = regime_profiles.loc[r]
    if prof['avg_ret'] > 1 and prof['gainer_rate'] > 0.15:
        label = 'Strong_Bull'
    elif prof['avg_ret'] < -1 and prof['gainer_rate'] > 0.20:
        label = 'Crash_Spike'
    elif prof['avg_hv'] > 12:
        label = 'High_Vol'
    elif prof['avg_ret'] < -0.3:
        label = 'Bearish'
    else:
        label = 'Normal'
    regime_labels[int(r)] = label
    n_days = (daily['regime'] == r).sum()
    print(f'    Regime {r} ({label}): n={n_days} days  gainer={prof["gainer_rate"]:.1%}  ret={prof["avg_ret"]:.2f}%  hv={prof["avg_hv"]:.1f}%')

# Transition matrix
n_regimes = int(best_n)
trans_mat = np.zeros((n_regimes, n_regimes))
regimes_arr = daily['regime'].values
for t in range(1, len(regimes_arr)):
    trans_mat[regimes_arr[t-1], regimes_arr[t]] += 1
trans_mat = trans_mat / (trans_mat.sum(axis=1, keepdims=True) + 1e-10)
print('  Transition matrix:')
for i in range(n_regimes):
    row_str = '  '.join([f'{trans_mat[i,j]:.3f}' for j in range(n_regimes)])
    print(f'    Regime {i}: {row_str}')

# Save regime data
daily[['date', 'avg_ret', 'avg_hv', 'avg_range', 'gainer_rate', 'n_stocks', 'regime']].to_csv(OUT / 'regime_data.csv', index=False)
print(f'  Regime data saved to {OUT / "regime_data.csv"}')

# ── 7. Seasonality ──
print('\n[7] Seasonality Analysis...')
daily['dow'] = pd.to_datetime(daily['date']).dt.dayofweek
daily['month'] = pd.to_datetime(daily['date']).dt.month
daily['year'] = pd.to_datetime(daily['date']).dt.year

# Day-of-week
dow_effect = daily.groupby('dow')['gainer_rate'].agg(['mean', 'std', 'count'])
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print('  Day-of-week effect:')
for d in range(5):
    r = dow_effect.loc[d]
    print(f'    {dow_names[d]}: gainer={r["mean"]:.1%}  n={int(r["count"])} days')

# Month effect
mon_effect = daily.groupby('month')['gainer_rate'].agg(['mean', 'std', 'count'])
print('  Month effect:')
mon_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in range(1, 13):
    if m in mon_effect.index:
        r = mon_effect.loc[m]
        print(f'    {mon_names[m-1]}: gainer={r["mean"]:.1%}  n={int(r["count"])} days')

# ANOVA for day-of-week
dow_groups = [daily[daily['dow'] == d]['gainer_rate'].values for d in range(5)]
dow_groups = [g for g in dow_groups if len(g) > 0]
if len(dow_groups) >= 2:
    f_stat, p_val = stats.f_oneway(*dow_groups)
    print(f'  DOW ANOVA: F={f_stat:.4f}  p={p_val:.4f}')

# ── 8. Cross-sectional feature spread analysis ──
print('\n[8] Cross-sectional Feature Spread Analysis...')
cs_features = ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5', 'bb_width']
cs_results = []
for feat in cs_features:
    if feat not in fs.columns:
        continue
    # Percentile ranks within each date
    fs[f'_{feat}_rank'] = fs.groupby('date')[feat].rank(pct=True)
    fs[f'_{feat}_quintile'] = pd.qcut(fs[f'_{feat}_rank'], 5, labels=False, duplicates='drop')
    if fs[f'_{feat}_quintile'].isna().all():
        continue
    quintile_rates = fs.groupby(f'_{feat}_quintile')['target'].mean()
    if len(quintile_rates) >= 2:
        spread = quintile_rates.max() - quintile_rates.min()
        cs_results.append({
            'feature': feat,
            'q1_gainer_rate': float(quintile_rates.iloc[0]) if len(quintile_rates) >= 1 else 0,
            'q5_gainer_rate': float(quintile_rates.iloc[-1]) if len(quintile_rates) >= 5 else 0,
            'spread': float(spread),
        })
        print(f'    {feat:<15s}: q1={quintile_rates.iloc[0]:.1%}  q5={quintile_rates.iloc[-1]:.1%}  spread={spread:.1%}')

# Clean up temp columns
fs = fs.drop(columns=[c for c in fs.columns if c.startswith('_')])

cs_df = pd.DataFrame(cs_results)
cs_df.to_csv(OUT / 'cross_sectional_spread.csv', index=False)

# ── 9. Summary ──
print(f'\n[9] Time Series Analysis Complete')
summary = {
    'sig_acf_lags': [int(x) for x in sig_acf],
    'sig_pacf_lags': [int(x) for x in sig_pacf],
    'n_regimes': int(best_n),
    'dow_effect': {str(dow_names[d]): float(dow_effect.loc[d, 'mean']) for d in range(5) if d in dow_effect.index},
    'features_granger': {feat: float(gc_pct.get(feat, 0)/100) for feat in gc_features},
    'optimal_windows': {},
}
# Add optimal windows
for feat in ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5']:
    feat_wins = win_agg[win_agg['feature'] == feat]
    if len(feat_wins) > 0:
        best = feat_wins.loc[feat_wins['corr_mean'].abs().idxmax()]
        summary['optimal_windows'][feat] = int(best['window'])

with open(OUT / 'ts_analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

con.close()
print(f'  Results saved to: {OUT}')
print(f'  Time: {time.time()-t0:.0f}s')
print('='*60)
