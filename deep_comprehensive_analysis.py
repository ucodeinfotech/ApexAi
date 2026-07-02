# Deep Comprehensive Analysis - Phases 4/5/6 + Time Series
# Generates detailed results + charts for PDF report
import duckdb, pandas as pd, numpy as np, time, warnings, json, gc
from pathlib import Path
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'deep_analysis_report'
OUT.mkdir(exist_ok=True)
(OUT / 'charts').mkdir(exist_ok=True)
(OUT / 'tables').mkdir(exist_ok=True)

t0 = time.time()
print('='*60)
print(' DEEP COMPREHENSIVE ANALYSIS')
print('='*60)

# ─── LOAD DATA ───
print('\n[0] Loading data...')
con = duckdb.connect(str(DB), read_only=True)
fs = con.execute("""
    SELECT symbol, datetime::DATE as date, open, high, low, close, volume,
           ret_1d, range_5, hv_20, rsi_14, vol_ratio_5, bb_width, adx
    FROM feature_store WHERE timeframe='1day'
    ORDER BY symbol, date
""").fetchdf()
fs = fs.sort_values(['symbol', 'date'])
fs['next_close'] = fs.groupby('symbol')['close'].shift(-1)
fs['next_open'] = fs.groupby('symbol')['open'].shift(-1)
fs['target_ret'] = fs['next_close'] / fs['next_open'] - 1
fs['target'] = (fs['target_ret'] > 0.02).astype(int)
fs = fs.dropna(subset=['target'])
con.close()
print(f'  {len(fs):,} rows, {fs["symbol"].nunique()} symbols')

# Daily aggregate
daily = fs.groupby('date').agg(
    gainer_rate=('target', 'mean'), n_stocks=('symbol', 'count'),
    avg_ret=('ret_1d', 'mean'), avg_range=('range_5', 'mean'), avg_hv=('hv_20', 'mean'),
).reset_index().sort_values('date')
daily['gainer_rate'] = daily['gainer_rate'].fillna(0)

# ─── HELPER FUNCTIONS ───
def safe_acf(series, nlags=40):
    n = len(series); mean = np.mean(series); var = np.var(series, ddof=0)
    if var == 0: return np.zeros(nlags+1)
    c0 = np.sum((series - mean)**2) / n; acf = np.ones(nlags+1)
    for k in range(1, nlags+1):
        acf[k] = np.sum((series[:-k] - mean) * (series[k:] - mean)) / n / c0
    return acf

def safe_pacf(series, nlags=40):
    acf_vals = safe_acf(series, nlags); pacf = np.zeros(nlags+1); pacf[0] = 1.0
    for k in range(1, nlags+1):
        if k == 1: pacf[k] = acf_vals[1]
        else:
            phi = np.zeros(k+1); phi[1] = acf_vals[1]
            for i in range(2, k+1):
                num = acf_vals[i] - np.sum(phi[1:i] * acf_vals[i-1:0:-1])
                denom = 1 - np.sum(phi[1:i] * acf_vals[1:i])
                phi[i] = num / denom if abs(denom) > 1e-10 else 0
                for j in range(1, i): phi[j] = phi[j] - phi[i] * phi[i-j]
            pacf[k] = phi[k]
    return pacf

def fmt_pct(v): return f'{v:.2%}'
def fmt_num(v): return f'{v:,.0f}'
def fmt4(v): return f'{v:.4f}'

# ════════════════════════════════════════════════════════════════
# SECTION 1: DEEP TIME SERIES ANALYSIS
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print(' SECTION 1: DEEP TIME SERIES ANALYSIS')
print('='*60)

ts_results = {}

# 1a. Full ACF/PACF for daily gainer rate (lags 1-40)
print('\n[1a] ACF/PACF Analysis (lags 1-40)...')
ts = daily['gainer_rate'].values
acf_vals = safe_acf(ts, 40)
pacf_vals = safe_pacf(ts, 40)
se_95 = 1.96 / np.sqrt(len(ts))
sig_acf = [i for i in range(1, 41) if abs(acf_vals[i]) > se_95]
sig_pacf = [i for i in range(1, 41) if abs(pacf_vals[i]) > se_95]

# ACF/PACF chart
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].bar(range(1, 41), acf_vals[1:], color='steelblue', width=0.6)
axes[0].axhline(se_95, color='red', linestyle='--', alpha=0.5, label=f'95% CI (={se_95:.3f})')
axes[0].axhline(-se_95, color='red', linestyle='--', alpha=0.5)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('ACF - Daily Gainer Rate', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Lag (days)'); axes[0].set_ylabel('Autocorrelation')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].bar(range(1, 41), pacf_vals[1:], color='darkorange', width=0.6)
axes[1].axhline(se_95, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(-se_95, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('PACF - Daily Gainer Rate', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Lag (days)'); axes[1].set_ylabel('Partial Autocorrelation')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()

# ACF/PACF table
acf_table = pd.DataFrame({
    'lag': range(1, 41), 'ACF': acf_vals[1:], 'PACF': pacf_vals[1:],
    'sig_ACF': [abs(acf_vals[i]) > se_95 for i in range(1, 41)],
    'sig_PACF': [abs(pacf_vals[i]) > se_95 for i in range(1, 41)],
})
acf_table.to_csv(OUT / 'tables' / 'acf_pacf.csv', index=False)
ts_results['acf'] = {
    'sig_acf_lags': [int(x) for x in sig_acf],
    'sig_pacf_lags': [int(x) for x in sig_pacf],
    'se_95': float(se_95),
    'key_lags': {str(lag): {'acf': float(acf_vals[lag]), 'pacf': float(pacf_vals[lag])}
                 for lag in [1,2,3,4,5,10,16,17,18,20] if lag < len(acf_vals)},
}
print(f'  Significant ACF lags: {sig_acf[:10]}...')
print(f'  Significant PACF lags: {sig_pacf[:10]}...')

# 1b. ACF/PACF for all key features (per symbol averaged)
print('\n[1b] Per-symbol ACF/PACF for key features...')
features_ts = ['ret_1d', 'range_5', 'hv_20']
feat_acf_summary = []
for feat in features_ts:
    feat_acfs = []
    for sym in fs['symbol'].unique()[:100]:
        s = fs[fs['symbol']==sym][feat].dropna().values
        if len(s) > 200:
            feat_acfs.append(safe_acf(s, 20)[1:])
    if feat_acfs:
        mean_acf = np.mean(feat_acfs, axis=0)
        feat_acf_summary.append({
            'feature': feat, 'lag1': float(mean_acf[0]),
            'lag5': float(mean_acf[4]) if len(mean_acf)>=5 else 0,
            'lag10': float(mean_acf[9]) if len(mean_acf)>=10 else 0,
            'lag_mean': float(np.mean(mean_acf)),
            'decay_rate': float(mean_acf[1]/mean_acf[0]) if mean_acf[0]!=0 else 0,
        })
pd.DataFrame(feat_acf_summary).to_csv(OUT / 'tables' / 'feature_acf_summary.csv', index=False)
print(f'  Computed for {len(feat_acf_summary)} features')

# 1c. Stationarity: ALL features (ADF + KPSS) on market-level and per-symbol
print('\n[1c] Stationarity Tests...')
from statsmodels.tsa.stattools import adfuller, kpss
stationarity_results = []
for feat in ['gainer_rate', 'avg_ret', 'avg_hv', 'avg_range']:
    s = daily[feat].dropna().values
    try:
        adf_p = adfuller(s, maxlag=20)[1]
    except: adf_p = np.nan
    try:
        kpss_s, kpss_p, _, _ = kpss(s, regression='c', nlags='auto')
    except: kpss_p = np.nan
    stationarity_results.append({
        'feature': feat, 'adf_pval': float(adf_p), 'kpss_pval': float(kpss_p),
        'adf_stationary': bool(adf_p < 0.05) if not np.isnan(adf_p) else False,
        'kpss_stationary': bool(kpss_p >= 0.05) if not np.isnan(kpss_p) else True,
    })

# Per-symbol stationarity on all features
sym_feats = ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5', 'bb_width', 'rsi_14']
sym_stat = []
for sym in fs['symbol'].unique()[:100]:
    sd = fs[fs['symbol']==sym].sort_values('date')
    for feat in sym_feats:
        s = sd[feat].dropna().values
        if len(s) < 100: continue
        try:
            p = adfuller(s, maxlag=10, autolag='AIC')[1]
            sym_stat.append({'symbol': sym, 'feature': feat, 'adf_pval': float(p), 'stationary': p < 0.05})
        except: pass

sym_stat_df = pd.DataFrame(sym_stat)
sym_pct = sym_stat_df.groupby('feature')['stationary'].mean() * 100
pd.DataFrame(stationarity_results).to_csv(OUT / 'tables' / 'stationarity.csv', index=False)
sym_stat_df.to_csv(OUT / 'tables' / 'stationarity_per_symbol.csv', index=False)
ts_results['stationarity'] = {
    'market_level': {r['feature']: {'adf_pval': r['adf_pval'], 'kpss_pval': r['kpss_pval'],
        'verdict': 'Stationary' if r['adf_stationary'] and r['kpss_stationary'] else 'Borderline'}
        for r in stationarity_results},
    'per_symbol_100': {feat: float(sym_pct[feat]/100) for feat in sym_feats if feat in sym_pct.index},
}

# 1d. Granger Causality - full detail
print('\n[1d] Granger Causality Tests...')
from statsmodels.tsa.stattools import grangercausalitytests
gc_features_detail = ['ret_1d', 'range_5', 'hv_20', 'rsi_14', 'vol_ratio_5', 'bb_width', 'adx']
gc_detail = []
for sym in fs['symbol'].unique():
    sd = fs[fs['symbol']==sym].sort_values('date').dropna(subset=['target'] + gc_features_detail)
    if len(sd) < 100: continue
    for feat in gc_features_detail:
        data = sd[[feat, 'target']].values
        if np.isnan(data).any(): continue
        try:
            gc_res = grangercausalitytests(data, maxlag=5, verbose=False)
            best_lag = min(range(1,6), key=lambda l: gc_res[l][0]['ssr_chi2test'][1])
            best_p = gc_res[best_lag][0]['ssr_chi2test'][1]
            gc_detail.append({'symbol': sym, 'feature': feat, 'best_lag': best_lag,
                'best_pval': float(best_p), 'significant': best_p < 0.05})
        except: pass

gc_df = pd.DataFrame(gc_detail)
gc_agg = gc_df.groupby('feature').agg(
    pct_sig=('significant', 'mean'), median_pval=('best_pval', 'median'),
    mean_lag=('best_lag', 'mean'), n_symbols=('symbol', 'nunique'),
).reset_index()
gc_agg.columns = ['feature', 'pct_significant', 'median_pval', 'mean_optimal_lag', 'n_symbols']
gc_agg.to_csv(OUT / 'tables' / 'granger_detailed.csv', index=False)
gc_df.to_csv(OUT / 'tables' / 'granger_per_symbol.csv', index=False)
ts_results['granger'] = {
    feat: {'pct_significant': float(r['pct_significant']), 'median_pval': float(r['median_pval']),
           'mean_optimal_lag': float(r['mean_optimal_lag']), 'n_symbols': int(r['n_symbols'])}
    for _, r in gc_agg.iterrows()
}
print('  Granger Causality Summary:')
for _, r in gc_agg.iterrows():
    print(f'    {r["feature"]:<15s} {r["pct_significant"]:.0%} sig  median_p={r["median_pval"]:.2e}  opt_lag={r["mean_optimal_lag"]:.1f}')

# 1e. Rolling window optimization - full analysis
print('\n[1e] Rolling Window Optimization...')
windows = [3, 5, 10, 15, 21, 30, 50]
win_detail = []
for sym in fs['symbol'].unique()[:100]:
    sd = fs[fs['symbol']==sym].sort_values('date').dropna(subset=['target'] + features_ts + ['vol_ratio_5'])
    if len(sd) < 200: continue
    for feat in features_ts + ['vol_ratio_5']:
        for w in windows:
            rolled = sd[feat].rolling(w, min_periods=max(3, w//3)).mean()
            corr = rolled.corr(sd['target'])
            mi = mutual_info_classif(rolled.fillna(0).values.reshape(-1,1), sd['target'].values,
                                      random_state=42, discrete_features=False)[0]
            win_detail.append({'symbol': sym, 'feature': feat, 'window': w,
                'corr': float(corr) if not np.isnan(corr) else 0, 'mutual_info': float(mi)})

win_df = pd.DataFrame(win_detail)
win_agg = win_df.groupby(['feature', 'window']).agg(
    corr_mean=('corr', 'mean'), corr_std=('corr', 'std'), mi_mean=('mutual_info', 'mean')
).reset_index()
win_agg.to_csv(OUT / 'tables' / 'window_optimization.csv', index=False)

# Window optimization chart
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
feat_list = features_ts + ['vol_ratio_5']
for feat in feat_list:
    fw = win_agg[win_agg['feature'] == feat]
    axes[0].plot(fw['window'], fw['corr_mean'], marker='o', label=feat, linewidth=2)
    axes[1].plot(fw['window'], fw['mi_mean'], marker='s', label=feat, linewidth=2)
axes[0].set_title('Feature-Window Correlation with Target', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Window (days)'); axes[0].set_ylabel('Mean Correlation')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_title('Feature-Window Mutual Information with Target', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Window (days)'); axes[1].set_ylabel('Mutual Information')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_window_optimization.png', dpi=150, bbox_inches='tight')
plt.close()

# Optimal windows
optimal_wins = {}
for feat in feat_list:
    fw = win_agg[win_agg['feature'] == feat]
    best = fw.loc[fw['corr_mean'].abs().idxmax()]
    optimal_wins[feat] = {'window': int(best['window']), 'corr': float(best['corr_mean']), 'mi': float(best['mi_mean'])}
ts_results['optimal_windows'] = optimal_wins
print('  Optimal windows:')
for feat, opt in optimal_wins.items():
    print(f'    {feat:<15s} window={opt["window"]}  corr={opt["corr"]:+.4f}  mi={opt["mi"]:.4f}')

# 1f. GMM Regime Clustering - full analysis with BIC/AIC
print('\n[1f] GMM Regime Clustering...')
regime_feats = daily[['avg_ret', 'avg_hv', 'avg_range']].dropna().values
scaler = StandardScaler()
regime_feats_scaled = scaler.fit_transform(regime_feats)

bic_aic = []
gmm_models = {}
for n in range(2, 8):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=15, max_iter=500)
    gmm.fit(regime_feats_scaled)
    bic_aic.append({'n_components': n, 'bic': gmm.bic(regime_feats_scaled),
                    'aic': gmm.aic(regime_feats_scaled), 'log_lik': gmm.score(regime_feats_scaled)})
    gmm_models[n] = gmm
bic_df = pd.DataFrame(bic_aic)
best_n = int(bic_df.loc[bic_df['bic'].idxmin(), 'n_components'])

# BIC/AIC chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(bic_df['n_components'], bic_df['bic'], marker='o', linewidth=2, label='BIC', color='steelblue')
ax.plot(bic_df['n_components'], bic_df['aic'], marker='s', linewidth=2, label='AIC', color='darkorange')
ax.axvline(best_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal (n={best_n})')
ax.set_xlabel('Number of Regimes'); ax.set_ylabel('Information Criterion')
ax.set_title('GMM Regime Selection - BIC/AIC', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_gmm_bic.png', dpi=150, bbox_inches='tight')
plt.close()

gmm = gmm_models[best_n]
labels = gmm.fit_predict(regime_feats_scaled)
valid_idx = daily[['avg_ret', 'avg_hv', 'avg_range']].dropna().index
daily['regime'] = np.nan
daily.loc[valid_idx, 'regime'] = labels
daily = daily.dropna(subset=['regime']); daily['regime'] = daily['regime'].astype(int)

regime_profiles = daily.groupby('regime').agg(
    gainer_rate=('gainer_rate', 'mean'), avg_ret=('avg_ret', 'mean'),
    avg_hv=('avg_hv', 'mean'), avg_range=('avg_range', 'mean'),
    n_days=('date', 'count'), gainer_spread=('gainer_rate', 'std'),
).reset_index()

# Transition matrix
n_regimes = best_n
trans_mat = np.zeros((n_regimes, n_regimes))
reg_arr = daily['regime'].values
for t in range(1, len(reg_arr)):
    trans_mat[reg_arr[t-1], reg_arr[t]] += 1
trans_prob = trans_mat / (trans_mat.sum(axis=1, keepdims=True) + 1e-10)

# Regime transition chart
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(trans_prob, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
for i in range(n_regimes):
    for j in range(n_regimes):
        ax.text(j, i, f'{trans_prob[i,j]:.2f}', ha='center', va='center',
                fontsize=9, color='black' if trans_prob[i,j] < 0.5 else 'white')
ax.set_xticks(range(n_regimes)); ax.set_yticks(range(n_regimes))
ax.set_xlabel('To Regime'); ax.set_ylabel('From Regime')
ax.set_title(f'Regime Transition Matrix ({best_n} regimes)', fontsize=13, fontweight='bold')
plt.colorbar(im, fraction=0.046)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_regime_transition.png', dpi=150, bbox_inches='tight')
plt.close()

# Regime time series chart
fig, ax = plt.subplots(figsize=(16, 6))
dates = pd.to_datetime(daily['date'])
ax.fill_between(range(len(daily)), 0, daily['regime'], alpha=0.3, color='steelblue')
ax.plot(daily['regime'], color='navy', linewidth=0.8, alpha=0.7)
ax.set_ylabel('Regime ID'); ax.set_xlabel('Date')
ax.set_title(f'Regime Time Series ({best_n} regimes)', fontsize=13, fontweight='bold')
ax.set_yticks(range(n_regimes))
# Label regime transitions on secondary axis
colors = plt.cm.tab10(np.linspace(0, 1, n_regimes))
for r in range(n_regimes):
    mask = daily['regime'] == r
    ax.scatter(np.where(mask)[0], [r]*mask.sum(), s=1, color=colors[r], alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_regime_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# Regime profiles
daily[['date','avg_ret','avg_hv','avg_range','gainer_rate','n_stocks','regime']].to_csv(
    OUT / 'tables' / 'regime_data_full.csv', index=False)
regime_profiles.to_csv(OUT / 'tables' / 'regime_profiles.csv', index=False)
np.savetxt(OUT / 'tables' / 'transition_matrix.csv', trans_prob, delimiter=',', fmt='%.4f')
ts_results['regimes'] = {
    'n_regimes': best_n, 'bic_aic': bic_aic,
    'profiles': {int(r['regime']): {
        'gainer_rate': float(r['gainer_rate']), 'avg_ret': float(r['avg_ret']),
        'avg_hv': float(r['avg_hv']), 'n_days': int(r['n_days'])}
        for _, r in regime_profiles.iterrows()},
    'transition_matrix': trans_prob.tolist(),
}
print(f'  Optimal regimes: {best_n}')
for _, r in regime_profiles.iterrows():
    print(f'    Regime {int(r["regime"])}: n={int(r["n_days"])}  gainer={r["gainer_rate"]:.1%}  ret={r["avg_ret"]:.2f}%')

# 1g. Seasonality - deep analysis
print('\n[1g] Seasonality Analysis...')
daily['dow'] = pd.to_datetime(daily['date']).dt.dayofweek
daily['month'] = pd.to_datetime(daily['date']).dt.month
daily['quarter'] = pd.to_datetime(daily['date']).dt.quarter
daily['year'] = pd.to_datetime(daily['date']).dt.year

# DOW analysis
dow_effect = daily.groupby('dow')['gainer_rate'].agg(['mean', 'std', 'count'])
dow_groups = [daily[daily['dow']==d]['gainer_rate'].values for d in range(5)]
dow_anova = stats.f_oneway(*dow_groups)
dow_kw = stats.kruskal(*dow_groups)

# Month analysis
mon_effect = daily.groupby('month')['gainer_rate'].agg(['mean', 'std', 'count'])
mon_groups = [daily[daily['month']==m]['gainer_rate'].values for m in range(1,13) if m in daily['month'].unique()]
mon_anova = stats.f_oneway(*mon_groups) if len(mon_groups) >= 2 else (0, 1)
mon_kw = stats.kruskal(*mon_groups) if len(mon_groups) >= 2 else (0, 1)

# Year analysis
yr_effect = daily.groupby('year')['gainer_rate'].agg(['mean', 'std', 'count'])

# Seasonality charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
mon_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

axes[0,0].bar(dow_names, [dow_effect.loc[d,'mean'] for d in range(5)], color='steelblue', alpha=0.8)
axes[0,0].errorbar(dow_names, [dow_effect.loc[d,'mean'] for d in range(5)],
                   yerr=[dow_effect.loc[d,'std']/np.sqrt(max(dow_effect.loc[d,'count'],1)) for d in range(5)],
                   fmt='none', color='black', capsize=5)
axes[0,0].axhline(daily['gainer_rate'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Overall mean: {daily["gainer_rate"].mean():.1%}')
axes[0,0].set_title(f'Day-of-Week Effect (ANOVA p={dow_anova[1]:.3f})', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('Gainer Rate'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].bar(mon_names, [mon_effect.loc[m,'mean'] for m in range(1,13) if m in mon_effect.index],
              color='darkorange', alpha=0.8)
axes[0,1].axhline(daily['gainer_rate'].mean(), color='red', linestyle='--', alpha=0.5)
axes[0,1].set_title(f'Month Effect (ANOVA p={mon_anova[1]:.3f})', fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('Gainer Rate'); axes[0,1].grid(True, alpha=0.3)
for label in axes[0,1].get_xticklabels(): label.set_rotation(45)

axes[1,0].bar([str(int(y)) for y in yr_effect.index], [yr_effect.loc[y,'mean'] for y in yr_effect.index],
              color='forestgreen', alpha=0.8)
axes[1,0].axhline(daily['gainer_rate'].mean(), color='red', linestyle='--', alpha=0.5)
axes[1,0].set_title('Year-over-Year Gainer Rate', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('Gainer Rate'); axes[1,0].grid(True, alpha=0.3)

# Quarter
qrt_effect = daily.groupby('quarter')['gainer_rate'].mean()
axes[1,1].bar([f'Q{q}' for q in range(1,5)], [qrt_effect.get(q,0) for q in range(1,5)],
              color='purple', alpha=0.8)
axes[1,1].axhline(daily['gainer_rate'].mean(), color='red', linestyle='--', alpha=0.5)
axes[1,1].set_title('Quarterly Effect', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('Gainer Rate'); axes[1,1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()

# Volume effect (trading day of month)
daily['dom'] = pd.to_datetime(daily['date']).dt.day
dom_effect = daily.groupby('dom')['gainer_rate'].mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(1, 32), [dom_effect.get(d,0) for d in range(1,32)], color='teal', alpha=0.7)
ax.axhline(daily['gainer_rate'].mean(), color='red', linestyle='--', alpha=0.5)
ax.set_title('Day-of-Month Effect', fontsize=13, fontweight='bold')
ax.set_xlabel('Day of Month'); ax.set_ylabel('Gainer Rate'); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_day_of_month.png', dpi=150, bbox_inches='tight')
plt.close()

ts_results['seasonality'] = {
    'dow': {str(dow_names[d]): float(dow_effect.loc[d,'mean']) for d in range(5)},
    'dow_anova_pval': float(dow_anova[1]), 'dow_kruskal_pval': float(dow_kw[1]),
    'month': {str(mon_names[m-1]): float(mon_effect.loc[m,'mean']) for m in range(1,13) if m in mon_effect.index},
    'month_anova_pval': float(mon_anova[1]), 'month_kruskal_pval': float(mon_kw[1]),
    'yearly': {str(int(y)): {'mean': float(yr_effect.loc[y,'mean']), 'n_days': int(yr_effect.loc[y,'count'])}
               for y in yr_effect.index},
}
print(f'  DOW ANOVA p={dow_anova[1]:.4f}  KW p={dow_kw[1]:.4f}')
print(f'  Month ANOVA p={mon_anova[1]:.4f}  KW p={mon_kw[1]:.4f}')

# 1h. Cross-sectional spread analysis
print('\n[1h] Cross-sectional Feature Spread...')
cs_feats = ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5', 'bb_width']
cs_results_list = []
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, feat in enumerate(cs_feats):
    if idx >= len(axes): break
    fs[f'_r_{feat}'] = fs.groupby('date')[feat].rank(pct=True)
    try:
        fs['_q'] = pd.qcut(fs[f'_r_{feat}'], 5, labels=False, duplicates='drop')
        q_rates = fs.groupby('_q')['target'].mean()
        spread = q_rates.max() - q_rates.min()
        cs_results_list.append({'feature': feat, 'q1': float(q_rates.iloc[0]), 'q3': float(q_rates.iloc[2]) if len(q_rates)>=3 else 0,
            'q5': float(q_rates.iloc[-1]), 'spread': float(spread)})
        axes[idx].bar(range(len(q_rates)), q_rates.values, color='steelblue', alpha=0.8)
        axes[idx].axhline(fs['target'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Base: {fs["target"].mean():.1%}')
        axes[idx].set_title(f'{feat} (spread={spread:.1%})', fontsize=11, fontweight='bold')
        axes[idx].set_xticks(range(len(q_rates)))
        axes[idx].set_xticklabels([f'Q{i+1}' for i in range(len(q_rates))])
        axes[idx].set_ylabel('Gainer Rate'); axes[idx].grid(True, alpha=0.3)
    except: pass
fs = fs.drop(columns=[c for c in fs.columns if c.startswith('_')])
for i in range(len(cs_feats), len(axes)): axes[i].set_visible(False)
axes[-1].set_visible(False)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'ts_cross_sectional.png', dpi=150, bbox_inches='tight')
plt.close()

pd.DataFrame(cs_results_list).to_csv(OUT / 'tables' / 'cross_sectional_spread.csv', index=False)
ts_results['cross_sectional'] = {r['feature']: {'spread': r['spread'], 'q1': r['q1'], 'q5': r['q5']}
                                  for r in cs_results_list}
print('  Cross-sectional spreads:')
for r in cs_results_list: print(f'    {r["feature"]:<15s} spread={r["spread"]:.1%}')

# Save TSA results
with open(OUT / 'tables' / 'ts_analysis_results.json', 'w') as f:
    json.dump(ts_results, f, indent=2)

# ════════════════════════════════════════════════════════════════
# SECTION 2: DEEP DATA MINING (Phase 4)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print(' SECTION 2: DEEP DATA MINING (Phase 4)')
print('='*60)
import sys; sys.path.insert(0, str(BASE))
from src.patterns.candlestick import detect_patterns as detect_candle
from src.patterns.chart_patterns import detect_chart_patterns

p4_results = {}

# 2a. Pattern detection from raw_market (sample 200 symbols)
print('\n[2a] Pattern Detection (full universe via chunks)...')
con = duckdb.connect(str(DB), read_only=True)
syms = [r[0] for r in con.execute(
    "SELECT symbol, COUNT(*) as cnt FROM raw_market WHERE timeframe='1day' GROUP BY symbol ORDER BY cnt DESC LIMIT 200"
).fetchall()]
con.close()

all_patterns = []
for sym in syms:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT datetime, open, high, low, close, volume FROM raw_market "
        "WHERE symbol=? AND timeframe='1day' ORDER BY datetime", [sym]
    ).fetchdf()
    con.close()
    if len(df) < 100: continue
    candle_masks = detect_candle(df)
    chart_masks = detect_chart_patterns(df)
    combined = pd.concat([candle_masks, chart_masks], axis=1)
    for col in combined.columns:
        occ = int(combined[col].sum())
        if occ > 0:
            all_patterns.append({'symbol': sym, 'pattern': col, 'occurrences': occ,
                'frequency': occ/len(df)})

pat_df = pd.DataFrame(all_patterns)
pat_freq = pat_df.groupby('pattern').agg(
    total_occ=('occurrences', 'sum'), n_symbols=('symbol', 'nunique'),
    avg_freq=('frequency', 'mean')
).sort_values('total_occ', ascending=False).reset_index()
pat_freq.to_csv(OUT / 'tables' / 'pattern_frequency.csv', index=False)
print(f'  Total patterns: {len(pat_freq)}')
for _, r in pat_freq.head(15).iterrows():
    print(f'    {r["pattern"]:<25s} occ={r["total_occ"]:>8,}  symbols={r["n_symbols"]}  freq={r["avg_freq"]:.2%}')

# 2b. Pattern performance (forward returns)
print('\n[2b] Pattern Forward Performance...')
con = duckdb.connect(str(DB), read_only=True)
perf_data = []
for sym in syms[:50]:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT datetime, open, high, low, close, volume FROM raw_market "
        "WHERE symbol=? AND timeframe='1day' ORDER BY datetime", [sym]
    ).fetchdf()
    con.close()
    if len(df) < 100: continue
    df = df.sort_values('datetime')
    df['fwd_1d'] = df['close'].pct_change(1).shift(-1)
    df['fwd_3d'] = df['close'].pct_change(3).shift(-3)
    df['fwd_5d'] = df['close'].pct_change(5).shift(-5)
    df['range_next'] = (df['high'].shift(-1) - df['low'].shift(-1)) / df['close']
    df['gap_next'] = df['open'].shift(-1) / df['close'] - 1

    candle_masks = detect_candle(df)
    chart_masks = detect_chart_patterns(df)
    combined = pd.concat([candle_masks, chart_masks], axis=1)

    for col in combined.columns:
        mask = combined[col].astype(bool)
        n = mask.sum()
        if n < 5: continue
        perf_data.append({
            'pattern': col, 'symbol': sym, 'n_occurrences': n,
            'fwd_1d_mean': float(df.loc[mask, 'fwd_1d'].mean()),
            'fwd_3d_mean': float(df.loc[mask, 'fwd_3d'].mean()),
            'fwd_5d_mean': float(df.loc[mask, 'fwd_5d'].mean()),
            'win_rate_1d': float((df.loc[mask, 'fwd_1d'] > 0).mean()),
            'win_rate_3d': float((df.loc[mask, 'fwd_3d'] > 0).mean()),
            'fwd_1d_std': float(df.loc[mask, 'fwd_1d'].std()),
            'range_next_mean': float(df.loc[mask, 'range_next'].mean()),
            'gap_next_mean': float(df.loc[mask, 'gap_next'].mean()),
        })

perf_df = pd.DataFrame(perf_data)
perf_agg = perf_df.groupby('pattern').agg(
    total_occ=('n_occurrences', 'sum'), n_symbols=('symbol', 'nunique'),
    fwd_1d_mean=('fwd_1d_mean', 'mean'), fwd_3d_mean=('fwd_3d_mean', 'mean'),
    win_rate_1d=('win_rate_1d', 'mean'), win_rate_3d=('win_rate_3d', 'mean'),
    avg_range_next=('range_next_mean', 'mean'),
).reset_index()
perf_agg.to_csv(OUT / 'tables' / 'pattern_performance.csv', index=False)

print('  Top patterns by forward 1d return:')
for _, r in perf_agg.sort_values('fwd_1d_mean', ascending=False).head(10).iterrows():
    print(f'    {r["pattern"]:<25s} fwd1d={r["fwd_1d_mean"]:+.2%}  wr1d={r["win_rate_1d"]:.1%}  fwd3d={r["fwd_3d_mean"]:+.2%}  n={int(r["total_occ"])}')

# 2c. Pattern co-occurrence matrix
print('\n[2c] Pattern Co-occurrence Analysis...')
cooc = pat_df.groupby(['pattern', 'symbol']).size().reset_index(name='cnt')
top_pats = pat_freq.head(15)['pattern'].tolist()
cooc_matrix = pd.DataFrame(0, index=top_pats, columns=top_pats)

for sym in syms[:100]:
    sym_pats = pat_df[pat_df['symbol'] == sym]
    pats_present = sym_pats[sym_pats['pattern'].isin(top_pats)]['pattern'].tolist()
    for i in range(len(pats_present)):
        for j in range(i+1, len(pats_present)):
            if pats_present[i] in top_pats and pats_present[j] in top_pats:
                cooc_matrix.loc[pats_present[i], pats_present[j]] += 1
                cooc_matrix.loc[pats_present[j], pats_present[i]] += 1

# Co-occurrence heatmap
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cooc_matrix.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(top_pats))); ax.set_yticks(range(len(top_pats)))
ax.set_xticklabels(top_pats, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(top_pats, fontsize=8)
ax.set_title('Pattern Co-occurrence Matrix', fontsize=13, fontweight='bold')
plt.colorbar(im, fraction=0.046)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'pattern_cooccurrence.png', dpi=150, bbox_inches='tight')
plt.close()

p4_results['pattern_frequency'] = pat_freq.to_dict('records')[:20]
p4_results['top_performers'] = perf_agg.sort_values('fwd_1d_mean', ascending=False).head(10).to_dict('records')

# ════════════════════════════════════════════════════════════════
# SECTION 3: DEEP EDA (Phase 5)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print(' SECTION 3: DEEP EDA (Phase 5)')
print('='*60)
from sklearn.preprocessing import StandardScaler

p5_results = {}
ef = pd.read_parquet(BASE / 'cleaned_features.parquet')
feat_cols = [c for c in ef.columns if c not in ('symbol', 'datetime', 'date', 'target', 'target_ret', 'next_close', 'next_open', 'year')]
num_cols = [c for c in feat_cols if ef[c].dtype in ('float64', 'float32', 'int64', 'int32')]
print(f'  Loaded: {ef.shape}, features: {len(feat_cols)}')

# 3a. Feature distributions - full statistics
print('\n[3a] Feature Distribution Analysis...')
dist_data = []
for c in num_cols:
    s = ef[c].dropna()
    q = np.percentile(s, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    dist_data.append({
        'feature': c, 'mean': float(s.mean()), 'std': float(s.std()),
        'min': float(s.min()), 'p0.1': float(q[0]), 'p1': float(q[1]),
        'p5': float(q[2]), 'p25': float(q[3]), 'p50': float(q[4]),
        'p75': float(q[5]), 'p95': float(q[6]), 'p99': float(q[7]), 'p99.9': float(q[8]),
        'max': float(s.max()), 'skew': float(s.skew()), 'kurtosis': float(s.kurtosis()),
        'missing': int(ef[c].isna().sum()), 'missing_pct': float(ef[c].isna().mean()*100),
    })

dist_df = pd.DataFrame(dist_data)
dist_df.to_csv(OUT / 'tables' / 'feature_distributions_deep.csv', index=False)

# Distribution chart - sample 20 features
sample_feats = num_cols[:20]
n_cols = 5; n_rows = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
axes = axes.flatten()
for i, feat in enumerate(sample_feats):
    if i >= len(axes): break
    s = ef[feat].dropna()
    axes[i].hist(s, bins=80, color='steelblue', alpha=0.7, density=True)
    axes[i].axvline(s.mean(), color='red', linestyle='--', label=f'μ={s.mean():.2f}')
    axes[i].set_title(f'{feat}\nskew={s.skew():.1f}  kurt={s.kurtosis():.1f}', fontsize=8)
    axes[i].tick_params(axis='x', labelsize=6)
for i in range(len(sample_feats), len(axes)): axes[i].set_visible(False)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'eda_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# 3b. Target distribution by various cuts
print('\n[3b] Target Deep Analysis...')
target_analysis = {
    'overall': float(ef['target'].mean()),
    'n_pos': int(ef['target'].sum()),
    'n_neg': int((1-ef['target']).sum()),
    'ratio': len(ef)/ef['target'].sum(),
}

# By deciles of best features
best_feats = ['range_5', 'hv_20', 'bb_width', 'rank_hv_20', 'regime_2']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, feat in enumerate(best_feats):
    if feat not in ef.columns: continue
    ef['_d'] = pd.qcut(ef[feat].rank(method='first'), 10, labels=False, duplicates='drop')
    d10 = ef.groupby('_d')['target'].mean()
    axes[i].bar(range(len(d10)), d10.values, color='steelblue', alpha=0.8)
    axes[i].axhline(ef['target'].mean(), color='red', linestyle='--', alpha=0.5)
    axes[i].set_title(f'{feat} (decile spread={d10.max()-d10.min():.1%})', fontsize=10)
    axes[i].set_xlabel('Decile'); axes[i].set_ylabel('Gainer Rate')
    axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'eda_decile_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
ef = ef.drop(columns=['_d'], errors='ignore')

# Target by symbol (top 20 / bottom 20)
sym_target = ef.groupby('symbol')['target'].agg(['mean', 'sum', 'count']).sort_values('mean', ascending=False)
sym_top = sym_target.head(20)
sym_bot = sym_target.tail(20).sort_values('mean', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].barh(range(len(sym_top)), sym_top['mean'].values, color='forestgreen', alpha=0.8)
axes[0].set_yticks(range(len(sym_top))); axes[0].set_yticklabels(sym_top.index, fontsize=7)
axes[0].axvline(ef['target'].mean(), color='red', linestyle='--', label=f'Overall: {ef["target"].mean():.1%}')
axes[0].set_title('Top 20 Symbols by Gainer Rate', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Gainer Rate'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].barh(range(len(sym_bot)), sym_bot['mean'].values, color='crimson', alpha=0.8)
axes[1].set_yticks(range(len(sym_bot))); axes[1].set_yticklabels(sym_bot.index, fontsize=7)
axes[1].axvline(ef['target'].mean(), color='red', linestyle='--', label=f'Overall: {ef["target"].mean():.1%}')
axes[1].set_title('Bottom 20 Symbols by Gainer Rate', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Gainer Rate'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'eda_symbol_ranking.png', dpi=150, bbox_inches='tight')
plt.close()

p5_results['target'] = target_analysis

# 3c. Feature-target correlation (multiple metrics)
print('\n[3c] Feature-Target Relationship...')
corr_data = []
for c in num_cols:
    if ef[c].nunique() < 2: continue
    s = ef[c].values; t = ef['target'].values
    # Pearson
    pcorr = np.corrcoef(s, t)[0,1] if s.std() > 0 else 0
    # Point biserial (binary target)
    pos_mean = ef.loc[ef['target']==1, c].mean()
    neg_mean = ef.loc[ef['target']==0, c].mean()
    # Mutual information
    mi = mutual_info_classif(s.reshape(-1,1), t, random_state=42, discrete_features=False)[0]
    # Spearman rank
    from scipy.stats import spearmanr
    sp_corr = spearmanr(s, t)[0]
    corr_data.append({'feature': c, 'pearson': float(pcorr), 'spearman': float(sp_corr),
        'mutual_info': float(mi), 'pos_mean': float(pos_mean), 'neg_mean': float(neg_mean)})

corr_df = pd.DataFrame(corr_data)
corr_df.to_csv(OUT / 'tables' / 'feature_target_correlation.csv', index=False)
p5_results['top_features'] = corr_df.sort_values('mutual_info', ascending=False).head(30).to_dict('records')

# Correlation chart
top_corr = corr_df.sort_values('mutual_info', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(12, 8))
x = range(len(top_corr))
ax.bar(x, top_corr['mutual_info'].values, color='steelblue', alpha=0.8, label='Mutual Info')
ax2 = ax.twinx()
ax2.plot(top_corr['pearson'].values, 'ro-', markersize=5, linewidth=1.5, label='Pearson r', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(top_corr['feature'].values, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mutual Information'); ax2.set_ylabel('Pearson Correlation')
ax.set_title('Top 20 Features by Mutual Information with Target', fontsize=13, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'eda_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# 3d. Feature-feature correlation network (top correlated pairs)
print('\n[3d] Feature-Feature Correlation...')
corr_matrix = ef[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_pairs = []
for i in range(len(upper.columns)):
    for j in range(i+1, len(upper.columns)):
        if upper.iloc[i, j] > 0.95:
            high_pairs.append({'feat1': upper.columns[i], 'feat2': upper.columns[j],
                'corr': float(corr_matrix.iloc[i, j])})
high_pairs_df = pd.DataFrame(high_pairs).sort_values('corr', ascending=False)
high_pairs_df.to_csv(OUT / 'tables' / 'highly_correlated_features.csv', index=False)

# Correlation heatmap (top features)
top_n = min(30, len(num_cols))
top_features = corr_df.sort_values('mutual_info', ascending=False).head(top_n)['feature'].tolist()
sub_corr = ef[top_features].corr()
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(sub_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(top_features))); ax.set_yticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=90, fontsize=6)
ax.set_yticklabels(top_features, fontsize=6)
ax.set_title(f'Feature Correlation Matrix (top {top_n} by MI)', fontsize=13, fontweight='bold')
plt.colorbar(im, fraction=0.046)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

p5_results['highly_correlated_pairs'] = len(high_pairs)
print(f'  Highly correlated pairs (r>0.95): {len(high_pairs)}')
print(f'  Top feature by MI: {corr_df.sort_values("mutual_info", ascending=False).iloc[0]["feature"]}')

# 3e. PCA analysis
print('\n[3e] PCA Analysis...')
pca_feats = num_cols[:50]  # Limit to 50 for speed
pca_data = ef[pca_feats].fillna(0).values
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)
pca = PCA(n_components=10)
pca_result = pca.fit_transform(pca_scaled)

var_exp = pca.explained_variance_ratio_
cum_var = np.cumsum(var_exp)
pca_loadings = pd.DataFrame(pca.components_[:5, :], columns=pca_feats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 11), var_exp[:10], alpha=0.7, color='steelblue', label='Individual')
axes[0].plot(range(1, 11), cum_var[:10], 'ro-', markersize=6, label='Cumulative')
axes[0].axhline(0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Variance Explained')
axes[0].set_title('PCA Variance Explained', fontsize=12, fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# PC1 vs PC2 colored by target
colors = ['crimson' if t == 1 else 'lightgray' for t in ef['target'].values[:len(pca_result)]]
axes[1].scatter(pca_result[:10000, 0], pca_result[:10000, 1], c=colors[:10000],
                alpha=0.5, s=3)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA Projection (10K samples, red=gainer)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'eda_pca.png', dpi=150, bbox_inches='tight')
plt.close()

p5_results['pca'] = {
    'n_components_80pct': int(np.searchsorted(cum_var, 0.8) + 1),
    'var_exp_pc1': float(var_exp[0]),
    'var_exp_pc2': float(var_exp[1]),
    'cum_var_10': float(cum_var[9]) if len(cum_var) > 9 else float(cum_var[-1]),
}
print(f'  PCA: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}  Cum10={cum_var[9]:.1%}')

# 3f. Temporal patterns
print('\n[3f] Temporal Pattern Analysis...')
ef['year'] = pd.to_datetime(ef['datetime']).dt.year
yr_feat = ef.groupby('year')[num_cols].mean()
yr_feat.to_csv(OUT / 'tables' / 'features_by_year.csv')

# Best & worst features by year
yr_target = ef.groupby('year')['target'].mean()
print('  Year-over-year feature drift:')
print(f'    Target: {dict(yr_target.round(4))}')
top_drift = yr_feat.std().sort_values(ascending=False).head(10)
print('  Top 10 features by year-over-year drift:')
for c in top_drift.index[:10]:
    print(f'    {c:<30s} year_std={top_drift[c]:.4f}  range=[{yr_feat[c].min():.2f}, {yr_feat[c].max():.2f}]')

# 3g. Outlier analysis (multiple methods)
print('\n[3g] Outlier Analysis (IQR + Z-score + MAD)...')
out_methods = []
for c in num_cols:
    s = ef[c].dropna()
    if len(s) < 100: continue
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    if iqr == 0: continue
    # IQR (3x)
    n_iqr = ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).sum()
    # Z-score (3)
    z = np.abs((s - s.mean()) / s.std()) if s.std() > 0 else np.zeros(len(s))
    n_z = (z > 3).sum()
    # MAD
    mad = np.median(np.abs(s - np.median(s)))
    if mad > 0:
        mod_z = 0.6745 * (s - np.median(s)) / mad
        n_mad = (np.abs(mod_z) > 3.5).sum()
    else: n_mad = 0
    out_methods.append({
        'feature': c, 'n_iqr': int(n_iqr), 'pct_iqr': float(n_iqr/len(s)*100),
        'n_zscore': int(n_z), 'pct_zscore': float(n_z/len(s)*100),
        'n_mad': int(n_mad), 'pct_mad': float(n_mad/len(s)*100),
    })

out_df = pd.DataFrame(out_methods).sort_values('pct_iqr', ascending=False)
out_df.to_csv(OUT / 'tables' / 'outlier_comparison.csv', index=False)
print(f'  Features with >5% IQR outliers: {(out_df["pct_iqr"] > 5).sum()}')

p5_results['outlier_summary'] = {
    'n_features_iqr_gt5pct': int((out_df['pct_iqr'] > 5).sum()),
    'top_outlier_feature': str(out_df.iloc[0]['feature']) if len(out_df) > 0 else 'N/A',
}

# Save EDA results
with open(OUT / 'tables' / 'eda_results.json', 'w') as f:
    json.dump(p5_results, f, indent=2)

# ════════════════════════════════════════════════════════════════
# SECTION 4: DEEP DATA CLEANING ANALYSIS (Phase 6)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print(' SECTION 4: DEEP DATA CLEANING ANALYSIS (Phase 6)')
print('='*60)

p6_results = {}
# Compare pre-cleaning vs post-cleaning
raw_ef = pd.read_parquet(BASE / 'engineered_features.parquet')
cleaned_ef = pd.read_parquet(BASE / 'cleaned_features.parquet')

print(f'  Raw shape: {raw_ef.shape}')
print(f'  Cleaned shape: {cleaned_ef.shape}')

# Features distribution before/after for key feats
key_feats = ['range_5', 'hv_20', 'ret_1d', 'bb_width', 'vol_ratio_5']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, feat in enumerate(key_feats):
    if feat not in raw_ef.columns or feat not in cleaned_ef.columns: continue
    raw_s = raw_ef[feat].dropna()
    clean_s = cleaned_ef[feat].dropna()
    axes[i].hist(raw_s, bins=80, alpha=0.5, color='red', density=True, label=f'Raw (n={len(raw_s)})')
    axes[i].hist(clean_s, bins=80, alpha=0.5, color='steelblue', density=True, label=f'Cleaned (n={len(clean_s)})')
    axes[i].set_title(f'{feat}\nRaw: μ={raw_s.mean():.2f} σ={raw_s.std():.2f} | Clean: μ={clean_s.mean():.2f} σ={clean_s.std():.2f}',
                      fontsize=9)
    axes[i].legend(fontsize=7); axes[i].tick_params(axis='x', labelsize=7)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
plt.tight_layout()
fig.savefig(OUT / 'charts' / 'cleaning_before_after.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature quality score
print('\n[4a] Feature Quality Scoring...')
quality_scores = []
for c in num_cols:
    if c not in cleaned_ef.columns: continue
    s = cleaned_ef[c].dropna()
    if len(s) == 0: continue
    missing_pct = (1 - len(s)/len(cleaned_ef)) * 100
    q1, q3 = np.percentile(s, [25, 75])
    iqr_score = min(100, max(0, 100 - (q3-q1)/max(s.std(), 1e-10)/10))
    skew_penalty = min(50, abs(s.skew()) * 5)
    quality = max(0, 100 - missing_pct * 2 - skew_penalty)
    quality_scores.append({
        'feature': c, 'missing_pct': float(missing_pct),
        'skew': float(s.skew()), 'outlier_pct_iqr': float(out_df[out_df['feature']==c]['pct_iqr'].iloc[0]) if c in out_df['feature'].values else 0,
        'quality_score': float(quality),
    })

qual_df = pd.DataFrame(quality_scores).sort_values('quality_score')
qual_df.to_csv(OUT / 'tables' / 'feature_quality_scores.csv', index=False)
print(f'  Features with quality < 50: {(qual_df["quality_score"] < 50).sum()}')
p6_results['quality_scores'] = {
    'mean_quality': float(qual_df['quality_score'].mean()),
    'n_low_quality': int((qual_df['quality_score'] < 50).sum()),
    'lowest_quality': str(qual_df.head(5)['feature'].tolist()),
}

# Correlation before/after cleaning
print('\n[4b] Correlation Stability (before vs after cleaning)...')
raw_corr = raw_ef[key_feats].corr() if all(f in raw_ef.columns for f in key_feats) else None
clean_corr = cleaned_ef[key_feats].corr() if all(f in cleaned_ef.columns for f in key_feats) else None

if raw_corr is not None and clean_corr is not None:
    corr_diff = (raw_corr - clean_corr).abs().values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title in zip(axes, [raw_corr, clean_corr, corr_diff],
                               ['Raw Data Correlations', 'Cleaned Data Correlations', 'Absolute Difference']):
        im = ax.imshow(mat, cmap='RdBu_r' if 'Diff' not in title else 'YlOrRd',
                       vmin=-1 if 'Diff' not in title else 0,
                       vmax=1 if 'Diff' not in title else max(0.05, corr_diff.max()))
        ax.set_xticks(range(len(key_feats))); ax.set_yticks(range(len(key_feats)))
        ax.set_xticklabels(key_feats, rotation=45, fontsize=8)
        ax.set_yticklabels(key_feats, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(OUT / 'charts' / 'cleaning_correlation_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Mean abs correlation change: {corr_diff.mean():.4f}')

p6_results['correlation_stability'] = {'mean_abs_change': float(corr_diff.mean()) if raw_corr is not None else 0}

with open(OUT / 'tables' / 'cleaning_results.json', 'w') as f:
    json.dump(p6_results, f, indent=2)

# ════════════════════════════════════════════════════════════════
# COMPLETE
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f' Deep Analysis Complete in {time.time()-t0:.0f}s')
print(f' Results in: {OUT}')
print('='*60)
