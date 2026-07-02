# Phase 5 - Exploratory Data Analysis
# Full EDA: distributions, target analysis, correlation, outlier detection, feature-target relationships
import pandas as pd, numpy as np, time, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'eda_results'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' Phase 5 - Exploratory Data Analysis')
print('='*60)

# Load engineered features
ef = pd.read_parquet(BASE / 'engineered_features.parquet')
print(f'\n[1] Dataset shape: {ef.shape}')
print(f'    Symbols: {ef["symbol"].nunique()}')
print(f'    Date range: {ef["datetime"].min()} to {ef["datetime"].max()}')
print(f'    Columns: {len(ef.columns)}')
print(f'    Memory: {ef.memory_usage(deep=True).sum() / 1e6:.0f} MB')

# ── 2. Target analysis ──
print('\n[2] Target Analysis')
target_col = 'target'
pos_rate = ef[target_col].mean()
print(f'  Positive rate (>2% open-to-close): {pos_rate:.4f} ({ef[target_col].sum():,}/{len(ef):,})')

# Target by year
ef['year'] = pd.to_datetime(ef['datetime']).dt.year
yr_rate = ef.groupby('year')[target_col].agg(['mean', 'sum', 'count'])
yr_rate.columns = ['pos_rate', 'n_pos', 'n_total']
print(f'\n  Target rate by year:')
for yr, r in yr_rate.iterrows():
    print(f'    {int(yr)}: {r["pos_rate"]:.1%} ({int(r["n_pos"]):,}/{int(r["n_total"]):,})')

# Target by symbol (top/bottom)
sym_rate = ef.groupby('symbol')[target_col].mean().sort_values()
print(f'\n  Top 10 symbols by gainer rate:')
for sym in sym_rate.tail(10).index:
    print(f'    {sym:<15s} {sym_rate[sym]:.1%}')
print(f'  Bottom 10 symbols by gainer rate:')
for sym in sym_rate.head(10).index:
    print(f'    {sym:<15s} {sym_rate[sym]:.1%}')

# Target autocorrelation
ts = ef.groupby('date')[target_col].mean()
acf_vals = [ts.autocorr(lag=i) for i in range(1, 21)]
print(f'\n  Target autocorrelation (daily gainer rate):')
for i, v in enumerate(acf_vals[:10], 1):
    print(f'    Lag {i:2d}: {v:.4f}')
print(f'  Key: lag1={acf_vals[0]:.4f} lag5={acf_vals[4]:.4f} lag16-18={acf_vals[15]:.4f},{acf_vals[16]:.4f},{acf_vals[17]:.4f}')

# ── 3. Feature distributions ──
print('\n[3] Feature Distribution Summary')
feat_cols = [c for c in ef.columns if c not in ('symbol', 'datetime', 'date', 'target', 'target_ret', 'next_close', 'next_open', 'year')]
dist_stats = []
for c in feat_cols:
    if ef[c].dtype not in ('float64', 'float32', 'int64', 'int32'):
        continue
    s = ef[c].dropna()
    if len(s) == 0:
        continue
    qs = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
    dist_stats.append({
        'feature': c,
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        'p1': float(qs[0]),
        'p5': float(qs[1]),
        'p25': float(qs[2]),
        'p50': float(qs[3]),
        'p75': float(qs[4]),
        'p95': float(qs[5]),
        'p99': float(qs[6]),
        'max': float(s.max()),
        'skew': float(s.skew()),
        'kurtosis': float(s.kurtosis()),
        'missing_pct': float((1 - len(s) / len(ef)) * 100),
        'n_unique': int(s.nunique()),
    })

dist_df = pd.DataFrame(dist_stats)
dist_df.to_csv(OUT / 'feature_distributions.csv', index=False)

high_skew = dist_df[dist_df['skew'].abs() > 5].sort_values('skew', key=abs, ascending=False)
high_kurt = dist_df[dist_df['kurtosis'] > 20].sort_values('kurtosis', ascending=False)
print(f'  Features with high skew (>5): {len(high_skew)}')
for _, r in high_skew.head(10).iterrows():
    print(f'    {r["feature"]:<30s} skew={r["skew"]:.1f}  kurt={r["kurtosis"]:.1f}')
print(f'  Features with high kurtosis (>20): {len(high_kurt)}')

# ── 4. Missing value analysis ──
print('\n[4] Missing Value Analysis')
missing = []
for c in feat_cols:
    n_miss = ef[c].isna().sum()
    if n_miss > 0:
        missing.append({'feature': c, 'n_missing': n_miss, 'pct_missing': n_miss / len(ef) * 100})
missing_df = pd.DataFrame(missing).sort_values('n_missing', ascending=False) if missing else pd.DataFrame()
if len(missing_df) > 0:
    missing_df.to_csv(OUT / 'missing_values.csv', index=False)
print(f'  Features with missing values: {len(missing_df)}')
for _, r in missing_df.head(15).iterrows():
    print(f'    {r["feature"]:<30s} missing={r["n_missing"]:>8,} ({r["pct_missing"]:.1f}%)')

# ── 5. Outlier detection ──
print('\n[5] Outlier Detection (IQR method)')
outlier_stats = []
for c in feat_cols:
    if ef[c].dtype not in ('float64', 'float32'):
        continue
    s = ef[c].dropna()
    if len(s) == 0:
        continue
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        continue
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    n_out = ((s < lower) | (s > upper)).sum()
    outlier_stats.append({
        'feature': c,
        'n_outliers': int(n_out),
        'pct_outliers': float(n_out / len(s) * 100),
        'lower_bound': float(lower),
        'upper_bound': float(upper),
    })

out_df = pd.DataFrame(outlier_stats).sort_values('pct_outliers', ascending=False)
out_df.to_csv(OUT / 'outlier_analysis.csv', index=False)
high_out = out_df[out_df['pct_outliers'] > 5]
print(f'  Features with >5% outliers: {len(high_out)}')
for _, r in high_out.head(10).iterrows():
    print(f'    {r["feature"]:<30s} outliers={r["pct_outliers"]:.1f}%  bounds=[{r["lower_bound"]:.2f}, {r["upper_bound"]:.2f}]')

# ── 6. Correlation analysis ──
print('\n[6] Correlation Analysis')
# Target correlation
target_corr = []
for c in feat_cols:
    if ef[c].dtype not in ('float64', 'float32', 'int64', 'int32'):
        continue
    if ef[c].nunique() < 2:
        continue
    corr = ef[c].corr(ef[target_col])
    if not np.isnan(corr):
        target_corr.append({'feature': c, 'corr_with_target': float(corr)})

corr_df = pd.DataFrame(target_corr).sort_values('corr_with_target', key=abs, ascending=False)
corr_df.to_csv(OUT / 'target_correlation.csv', index=False)
print(f'  Top 15 features correlated with target:')
for _, r in corr_df.head(15).iterrows():
    print(f'    {r["feature"]:<30s} corr={r["corr_with_target"]:+.4f}')

# Feature-feature correlation (find highly collinear pairs)
print(f'\n  Feature-feature multicollinearity:')
feat_matrix = ef[feat_cols].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
corr_matrix = feat_matrix.corr().abs()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append({
                'feat1': corr_matrix.columns[i],
                'feat2': corr_matrix.columns[j],
                'corr': float(corr_matrix.iloc[i, j]),
            })
high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('corr', ascending=False)
high_corr_df.to_csv(OUT / 'high_correlation_pairs.csv', index=False)
print(f'  Highly correlated pairs (r>0.95): {len(high_corr_df)}')
for _, r in high_corr_df.head(10).iterrows():
    print(f'    {r["feat1"]:<25s} X {r["feat2"]:<25s} r={r["corr"]:.4f}')

# ── 7. Feature-target relationship (binned analysis) ──
print('\n[7] Feature-Target Relationship (quintile analysis)')
quintile_analysis = []
for c in feat_cols:
    if ef[c].dtype not in ('float64', 'float32'):
        continue
    if ef[c].nunique() < 10:
        continue
    s = ef[c].dropna()
    if len(s) < 100:
        continue
    # Quintile hit rates
    try:
        ef['_bin'] = pd.qcut(ef[c], 5, labels=False, duplicates='drop')
        if ef['_bin'].nunique() < 2:
            continue
        bin_rates = ef.groupby('_bin')[target_col].mean()
        spread = float(bin_rates.max() - bin_rates.min())
        quintile_analysis.append({
            'feature': c,
            'q1_rate': float(bin_rates.iloc[0]) if len(bin_rates) >= 1 else 0,
            'q5_rate': float(bin_rates.iloc[-1]) if len(bin_rates) >= 5 else 0,
            'spread': spread,
        })
    except:
        pass

ef = ef.drop(columns=['_bin'], errors='ignore')
quint_df = pd.DataFrame(quintile_analysis).sort_values('spread', ascending=False)
quint_df.to_csv(OUT / 'quintile_analysis.csv', index=False)
print(f'  Top 15 features by quintile spread:')
for _, r in quint_df.head(15).iterrows():
    print(f'    {r["feature"]:<30s} q1={r["q1_rate"]:.1%}  q5={r["q5_rate"]:.1%}  spread={r["spread"]:.1%}')

# ── 8. Class imbalance by regime ──
print('\n[8] Regime-specific analysis')
regime_cols = [c for c in feat_cols if c.startswith('regime_')]
if regime_cols:
    for rc in regime_cols:
        mask = ef[rc] == 1
        n = mask.sum()
        rate = ef.loc[mask, target_col].mean() if n > 0 else 0
        print(f'    {rc}: n={n:>6,}  gainer_rate={rate:.1%}')

# ── 9. Summary report ──
print(f'\n[9] EDA Summary')
report = {
    'dataset_shape': list(ef.shape),
    'n_symbols': int(ef['symbol'].nunique()),
    'date_range': [str(ef['datetime'].min()), str(ef['datetime'].max())],
    'pos_rate': float(pos_rate),
    'n_features': len(feat_cols),
    'n_highly_correlated_pairs': len(high_corr_pairs),
    'n_features_high_skew': len(high_skew),
    'n_features_with_missing': len(missing_df),
    'n_features_high_outliers': len(high_out),
    'top_features_by_corr': {r['feature']: r['corr_with_target'] for _, r in corr_df.head(10).iterrows()},
    'top_features_by_quintile_spread': {r['feature']: r['spread'] for _, r in quint_df.head(10).iterrows()},
    'yearly_pos_rates': {int(k): float(v['pos_rate']) for k, v in yr_rate.iterrows()},
}
with open(OUT / 'eda_summary.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f'  EDA results saved to: {OUT}')
print(f'  Time: {time.time()-t0:.0f}s')
print('='*60)
