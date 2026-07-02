# Resume from GMM regime clustering
import duckdb, pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy import stats

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'ts_analysis_output'
OUT.mkdir(exist_ok=True)

DB = BASE / 'warehouse' / 'market_data.duckdb'
con = duckdb.connect(str(DB), read_only=True)

# Reload daily data
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

daily = fs.groupby('date').agg(
    gainer_rate=('target', 'mean'),
    n_stocks=('symbol', 'count'),
    avg_ret=('ret_1d', 'mean'),
    avg_range=('range_5', 'mean'),
    avg_hv=('hv_20', 'mean'),
).reset_index().sort_values('date')

daily['gainer_rate'] = daily['gainer_rate'].fillna(0)

# ── 6. GMM Regime Clustering ──
print('[6] GMM Regime Clustering...')
regime_feats = daily[['avg_ret', 'avg_hv', 'avg_range']].dropna().values
scaler_mean = regime_feats.mean(axis=0)
scaler_std = regime_feats.std(axis=0)
regime_feats_scaled = (regime_feats - scaler_mean) / (scaler_std + 1e-10)

bic_scores = []
for n in range(2, 7):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(regime_feats_scaled)
    bic_scores.append({'n_components': n, 'bic': gmm.bic(regime_feats_scaled)})

bic_df = pd.DataFrame(bic_scores)
best_n = int(bic_df.loc[bic_df['bic'].idxmin(), 'n_components'])
print(f'  Optimal regimes: {best_n} (BIC={bic_df["bic"].min():.0f})')

gmm = GaussianMixture(n_components=best_n, random_state=42, n_init=10)
regime_labels_raw = gmm.fit_predict(regime_feats_scaled)
daily['regime'] = np.nan
valid_idx = daily[['avg_ret', 'avg_hv', 'avg_range']].dropna().index
daily.loc[valid_idx, 'regime'] = regime_labels_raw
daily = daily.dropna(subset=['regime'])
daily['regime'] = daily['regime'].astype(int)

regime_profiles = daily.groupby('regime')[['gainer_rate', 'avg_ret', 'avg_hv', 'avg_range']].mean()
print('  Regime profiles:')
for r in sorted(regime_profiles.index):
    prof = regime_profiles.loc[r]
    n_days = (daily['regime'] == r).sum()
    print(f'    Regime {r}: n={n_days} days gainer={prof["gainer_rate"]:.1%} ret={prof["avg_ret"]:.2f}% hv={prof["avg_hv"]:.1f}%')

# Transition matrix
n_regimes = best_n
trans_mat = np.zeros((n_regimes, n_regimes))
regimes_arr = daily['regime'].values
for t in range(1, len(regimes_arr)):
    trans_mat[regimes_arr[t-1], regimes_arr[t]] += 1
trans_mat = trans_mat / (trans_mat.sum(axis=1, keepdims=True) + 1e-10)
print('  Transition matrix:')
for i in range(n_regimes):
    row_str = '  '.join([f'{trans_mat[i,j]:.3f}' for j in range(n_regimes)])
    print(f'    Regime {i}: {row_str}')

daily[['date', 'avg_ret', 'avg_hv', 'avg_range', 'gainer_rate', 'n_stocks', 'regime']].to_csv(OUT / 'regime_data.csv', index=False)

# ── 7. Seasonality ──
print('\n[7] Seasonality Analysis...')
daily['dow'] = pd.to_datetime(daily['date']).dt.dayofweek
daily['month'] = pd.to_datetime(daily['date']).dt.month

dow_effect = daily.groupby('dow')['gainer_rate'].agg(['mean', 'std', 'count'])
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
print('  Day-of-week effect:')
for d in range(5):
    r = dow_effect.loc[d]
    print(f'    {dow_names[d]}: gainer={r["mean"]:.1%}  n={int(r["count"])} days')

mon_effect = daily.groupby('month')['gainer_rate'].agg(['mean', 'std', 'count'])
mon_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print('  Month effect:')
for m in range(1, 13):
    if m in mon_effect.index:
        r = mon_effect.loc[m]
        print(f'    {mon_names[m-1]}: gainer={r["mean"]:.1%}  n={int(r["count"])} days')

dow_groups = [daily[daily['dow'] == d]['gainer_rate'].values for d in range(5)]
dow_groups = [g for g in dow_groups if len(g) > 0]
f_stat, p_val = stats.f_oneway(*dow_groups)
print(f'  DOW ANOVA: F={f_stat:.4f}  p={p_val:.4f}')

# ── 8. Cross-sectional spread ──
print('\n[8] Cross-sectional Feature Spread...')
cs_features = ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5', 'bb_width']
for feat in cs_features:
    if feat not in fs.columns:
        continue
    fs[f'_{feat}_rank'] = fs.groupby('date')[feat].rank(pct=True)
    fs[f'_{feat}_quintile'] = pd.qcut(fs[f'_{feat}_rank'], 5, labels=False, duplicates='drop')
    quintile_rates = fs.groupby(f'_{feat}_quintile')['target'].mean()
    if len(quintile_rates) >= 2:
        spread = quintile_rates.max() - quintile_rates.min()
        print(f'    {feat:<15s}: q1={quintile_rates.iloc[0]:.1%}  q5={quintile_rates.iloc[-1]:.1%}  spread={spread:.1%}')

# ── 9. Update summary ──
print('\n[9] Updating summary...')
with open(OUT / 'ts_analysis_results.json', 'r') as f:
    summary = json.load(f)
summary['n_regimes'] = best_n
dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
summary['dow_effect'] = {dow_map[d]: float(dow_effect.loc[d, 'mean']) for d in range(5) if d in dow_effect.index}
with open(OUT / 'ts_analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'  All results saved to: {OUT}')
print('='*60)
