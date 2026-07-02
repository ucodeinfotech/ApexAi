# Phase 6 - Deep Data Cleaning & Quality Analysis
import duckdb, pandas as pd, numpy as np, time, warnings, json, gc
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'phase6_deep_cleaning'
OUT.mkdir(exist_ok=True)
(OUT/'charts').mkdir(exist_ok=True)
(OUT/'tables').mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' PHASE 6 - DEEP DATA CLEANING & QUALITY ANALYSIS')
print('='*60)

# ─── LOAD RAW & CLEANED DATA ───
print('\n[0] Loading data...')
raw = pd.read_parquet(BASE / 'engineered_features.parquet')
clean = pd.read_parquet(BASE / 'cleaned_features.parquet')

# Load from DB for OHLC integrity checks
con = duckdb.connect(str(DB), read_only=True)
ohlc_raw = con.execute("""
    SELECT symbol, datetime::DATE as date, open, high, low, close, volume
    FROM raw_market WHERE timeframe='1day'
    ORDER BY symbol, date
""").fetchdf()
con.close()

print(f'  Raw: {raw.shape} ({raw["symbol"].nunique()} symbols)')
print(f'  Cleaned: {clean.shape} ({clean["symbol"].nunique()} symbols)')
print(f'  OHLC DB: {ohlc_raw.shape} ({ohlc_raw["symbol"].nunique()} symbols)')

id_cols = ['symbol', 'datetime', 'date']
meta = id_cols + ['target', 'target_ret', 'next_close', 'next_open']
feat_raw = [c for c in raw.columns if c not in meta]
feat_clean = [c for c in clean.columns if c not in meta]
num_raw = [c for c in feat_raw if raw[c].dtype in ('float64','float32','int64','int32')]
num_clean = [c for c in feat_clean if clean[c].dtype in ('float64','float32','int64','int32')]
print(f'  Raw features: {len(feat_raw)}, Clean features: {len(feat_clean)}')

# ════════════════════════════════════════════════════════════════
# 1. SYSTEMATIC MISSING VALUE ANALYSIS
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(' SECTION 1: MISSING VALUE ANALYSIS')
print(f'{"="*60}')

# 1a. Feature-level missing analysis (raw data)
print('\n[1a] Feature-level missing analysis...')
miss_data = []
for c in raw.columns:
    n_miss = raw[c].isna().sum()
    if n_miss > 0:
        miss_data.append({
            'feature': c, 'n_missing': n_miss, 'pct_missing': n_miss/len(raw)*100,
            'dtype': str(raw[c].dtype)
        })
miss_df = pd.DataFrame(miss_data) if len(miss_data) > 0 else pd.DataFrame(columns=['feature','n_missing','pct_missing','dtype'])
if len(miss_df) > 0:
    miss_df = miss_df.sort_values('n_missing', ascending=False)
    miss_df.to_csv(OUT/'tables'/'missing_values_raw.csv', index=False)

print(f'  Features with missing values: {len(miss_df)}')
if len(miss_df) > 0:
    for _, r in miss_df.head(20).iterrows():
        print(f'    {r["feature"]:<35s} missing={r["n_missing"]:>8,} ({r["pct_missing"]:.3f}%)')
else:
    print('  (none found)')

# 1b. Missing pattern by symbol
print('\n[1b] Missing pattern by symbol...')
sym_miss = raw.groupby('symbol').apply(lambda g: g.isna().sum(axis=1).sum())
sym_miss = sym_miss.sort_values(ascending=False)
heavy_miss = sym_miss[sym_miss > 0]
print(f'  Symbols with missing values: {len(heavy_miss)}')

if len(heavy_miss) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    top_miss = heavy_miss.head(30)
    ax.barh(range(len(top_miss)), top_miss.values, color='crimson', alpha=0.7)
    ax.set_yticks(range(len(top_miss)))
    ax.set_yticklabels(top_miss.index, fontsize=7)
    ax.set_xlabel('Total Missing Cells')
    ax.set_title('Top 30 Symbols by Missing Value Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT/'charts'/'missing_by_symbol.png', dpi=150, bbox_inches='tight')
    plt.close()

# 1c. Missing pattern by date
print('\n[1c] Missing pattern over time...')
raw['date'] = pd.to_datetime(raw['datetime']).dt.normalize()
date_miss = raw.groupby(raw['date'].dt.to_period('M')).apply(
    lambda g: g.isna().sum().sum() / (len(g) * len(g.columns)) * 100
)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(date_miss.index.astype(str), date_miss.values, color='steelblue', linewidth=1.5)
ax.fill_between(range(len(date_miss)), 0, date_miss.values, alpha=0.2, color='steelblue')
ax.set_xlabel('Date')
ax.set_ylabel('Missing % (all cells)')
ax.set_title('Missing Values Over Time', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT/'charts'/'missing_over_time.png', dpi=150, bbox_inches='tight')
plt.close()

# 1d. Cross-feature missing co-occurrence
if len(miss_df) > 1:
    print('\n[1d] Missing co-occurrence patterns...')
    miss_cols = miss_df.head(min(20, len(miss_df)))['feature'].tolist()
    miss_indicator = raw[miss_cols].isna().astype(int)
    co_occur = miss_indicator.T @ miss_indicator

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(co_occur, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(miss_cols)))
    ax.set_yticks(range(len(miss_cols)))
    ax.set_xticklabels([c[:15] for c in miss_cols], rotation=90, fontsize=6)
    ax.set_yticklabels([c[:15] for c in miss_cols], fontsize=6)
    ax.set_title('Missing Value Co-occurrence Matrix', fontsize=13, fontweight='bold')
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    fig.savefig(OUT/'charts'/'missing_cooccurrence.png', dpi=150, bbox_inches='tight')
    plt.close()

# 1e. Target missing analysis
print('\n[1e] Target missing analysis...')
target_miss = raw['target'].isna().sum() if 'target' in raw.columns else 0
print(f'  Target missing: {target_miss} ({target_miss/len(raw)*100:.1f}%)')
# When is target missing? (last row per symbol)
if target_miss > 0:
    raw_date = raw.copy()
    raw_date['_target_miss'] = raw_date['target'].isna()
    target_miss_by_date = raw_date.groupby(raw_date['date'].dt.to_period('M'))['_target_miss'].mean()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(target_miss_by_date.index.astype(str), target_miss_by_date.values, color='darkorange', linewidth=1.5)
    ax.set_xlabel('Date'); ax.set_ylabel('Target Missing Rate')
    ax.set_title('Target Missing Rate Over Time (last observation per symbol)', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT/'charts'/'target_missing_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

# ════════════════════════════════════════════════════════════════
# 2. OUTLIER DETECTION — MULTI-METHOD
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(' SECTION 2: OUTLIER DETECTION')
print(f'{"="*60}')

# 2a. IQR, Z-score, MAD comparison on all numeric features
print('\n[2a] Comparative outlier detection (IQR/Z-score/MAD)...')
outlier_methods = []
for c in num_clean:
    s = clean[c].dropna()
    if len(s) < 100: continue
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    if iqr == 0: continue

    n = len(s)

    # IQR 1.5x
    n_iqr15 = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    # IQR 3x
    n_iqr3 = int(((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).sum())

    # Z-score
    z = np.abs((s - s.mean()) / s.std()) if s.std() > 0 else np.zeros(n)
    n_z2 = int((z > 2).sum())
    n_z3 = int((z > 3).sum())

    # Modified Z-score (MAD)
    med = np.median(s); mad = np.median(np.abs(s - med))
    if mad > 0:
        mod_z = 0.6745 * (s - med) / mad
        n_mad2 = int((np.abs(mod_z) > 2).sum())
        n_mad35 = int((np.abs(mod_z) > 3.5).sum())
    else:
        n_mad2 = n_mad35 = 0

    outlier_methods.append({
        'feature': c, 'n_total': n,
        'iqr_1_5x': n_iqr15, 'pct_iqr_1_5x': n_iqr15/n*100,
        'iqr_3x': n_iqr3, 'pct_iqr_3x': n_iqr3/n*100,
        'z_2': n_z2, 'pct_z_2': n_z2/n*100,
        'z_3': n_z3, 'pct_z_3': n_z3/n*100,
        'mad_2': n_mad2, 'pct_mad_2': n_mad2/n*100,
        'mad_3_5': n_mad35, 'pct_mad_3_5': n_mad35/n*100,
    })

om_df = pd.DataFrame(outlier_methods)
om_df.to_csv(OUT/'tables'/'outlier_method_comparison.csv', index=False)

# Method agreement analysis
print('\n  Method agreement analysis...')
agreement_data = []
for _, r in om_df.iterrows():
    agree_count = 0
    if r['pct_iqr_3x'] > 5: agree_count += 1
    if r['pct_z_3'] > 5: agree_count += 1
    if r['pct_mad_3_5'] > 5: agree_count += 1
    agreement_data.append({
        'feature': r['feature'],
        'n_methods_agree': agree_count,
        'iqr_3x': r['pct_iqr_3x'],
        'z_3': r['pct_z_3'],
        'mad_3_5': r['pct_mad_3_5'],
    })
agree_df = pd.DataFrame(agreement_data)
agree_df.to_csv(OUT/'tables'/'outlier_method_agreement.csv', index=False)
print(f'  Features flagged by all 3 methods: {len(agree_df[agree_df["n_methods_agree"] >= 2])}')
print(f'  Features flagged by any method: {len(agree_df[agree_df["n_methods_agree"] > 0])}')

# Outlier method comparison chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (col, title) in enumerate([
    ('pct_iqr_3x', 'IQR (3x)'),
    ('pct_z_3', 'Z-score (3)'),
    ('pct_mad_3_5', 'MAD (3.5)')
]):
    vals = om_df[col].values
    axes[i].hist(vals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[i].axvline(5, color='red', ls='--', alpha=0.5, label='5% threshold')
    axes[i].set_xlabel('Outlier %'); axes[i].set_ylabel('Count')
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT/'charts'/'outlier_method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2b. Isolation Forest
print('\n[2b] Isolation Forest anomaly detection...')
n_feat_sample = min(30, len(num_clean))
sample_feats = om_df.sort_values('pct_iqr_3x', ascending=False).head(n_feat_sample)['feature'].tolist()
sample_data = clean[sample_feats].fillna(0).values[:50000]  # 50K sample for speed

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso_pred = iso.fit_predict(StandardScaler().fit_transform(sample_data))

iso_outliers = (iso_pred == -1).sum()
iso_inliers = (iso_pred == 1).sum()
print(f'  Isolation Forest on 50K x {n_feat_sample} sample:')
print(f'    Outliers: {iso_outliers} ({iso_outliers/len(iso_pred)*100:.1f}%)')
print(f'    Inliers: {iso_inliers} ({iso_inliers/len(iso_pred)*100:.1f}%)')

# Outlier characterization
print('\n  Outlier characterization (which symbols/dates produce outliers)...')
clean_sample = clean.iloc[:len(iso_pred)].copy()
clean_sample['_iso_outlier'] = iso_pred == -1

iso_by_symbol = clean_sample.groupby('symbol')['_iso_outlier'].mean().sort_values(ascending=False)
print('  Top 15 symbols by Isolation Forest outlier rate:')
for sym in iso_by_symbol.head(15).index:
    rate = iso_by_symbol[sym]
    count = clean_sample[clean_sample['symbol']==sym]['_iso_outlier'].sum()
    total = len(clean_sample[clean_sample['symbol']==sym])
    print(f'    {sym:<15s} outlier_rate={rate:.1%} ({int(count)}/{total})')

# Outlier temporal distribution
iso_by_month = clean_sample.groupby(pd.to_datetime(clean_sample['datetime']).dt.to_period('M'))['_iso_outlier'].mean()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(iso_by_month.index.astype(str), iso_by_month.values, color='crimson', linewidth=1.5)
ax.fill_between(range(len(iso_by_month)), 0, iso_by_month.values, alpha=0.2, color='crimson')
ax.axhline(0.05, color='gray', ls='--', alpha=0.5, label='Expected (5%)')
ax.set_xlabel('Date'); ax.set_ylabel('Outlier Rate')
ax.set_title('Isolation Forest Outlier Rate Over Time', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT/'charts'/'iso_forest_over_time.png', dpi=150, bbox_inches='tight')
plt.close()

# 2c. DBSCAN clustering for outlier detection
print('\n[2c] DBSCAN density-based outlier detection...')
db_data = StandardScaler().fit_transform(sample_data[:20000])
db = DBSCAN(eps=3, min_samples=10, n_jobs=-1)
db_labels = db.fit_predict(db_data)
db_outliers = (db_labels == -1).sum()
print(f'  DBSCAN on 20K sample:')
print(f'    Outliers: {db_outliers} ({db_outliers/20000*100:.1f}%)')
print(f'    Clusters found: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}')

# 2d. Outlier impact analysis
print('\n[2d] Outlier impact on feature-target correlation...')
impact_data = []
for c in num_clean:
    s = clean[c].dropna()
    if len(s) < 1000: continue
    # With outliers
    corr_all = np.corrcoef(s, clean.loc[s.index, 'target'])[0,1] if s.std() > 0 else 0
    # Without outliers (IQR 3x)
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    if iqr > 0:
        mask = (s >= q1 - 3*iqr) & (s <= q3 + 3*iqr)
        s_clean = s[mask]
        t_clean = clean.loc[s.index, 'target'].values[mask]
        corr_clean = np.corrcoef(s_clean, t_clean)[0,1] if s_clean.std() > 0 else 0
        delta = corr_clean - corr_all
        impact_data.append({'feature': c, 'corr_all': float(corr_all),
            'corr_no_outliers': float(corr_clean), 'corr_delta': float(delta),
            'abs_corr_delta': abs(delta)})
    else:
        impact_data.append({'feature': c, 'corr_all': float(corr_all),
            'corr_no_outliers': float(corr_all), 'corr_delta': 0, 'abs_corr_delta': 0})

imp_df = pd.DataFrame(impact_data).sort_values('abs_corr_delta', ascending=False)
imp_df.to_csv(OUT/'tables'/'outlier_correlation_impact.csv', index=False)
print(f'  Features with correlation change > 0.01: {(imp_df["abs_corr_delta"] > 0.01).sum()}')

# 2e. Outlier distribution charts
top_out = om_df.sort_values('pct_iqr_1_5x', ascending=False).head(12)['feature'].tolist()
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()
for i, feat in enumerate(top_out):
    if i >= len(axes): break
    s = clean[feat].dropna()
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    axes[i].hist(s, bins=100, color='steelblue', alpha=0.7, density=True)
    axes[i].axvline(s.mean(), color='green', ls='--', lw=1.5, label=f'mean={s.mean():.2f}')
    axes[i].axvline(q1-3*iqr, color='red', ls='--', lw=1, label=f'IQR3 bounds')
    axes[i].axvline(q3+3*iqr, color='red', ls='--', lw=1)
    axes[i].set_title(f'{feat[:25]}\noutliers: {om_df[om_df["feature"]==feat]["pct_iqr_1_5x"].iloc[0]:.1f}% (1.5x), {om_df[om_df["feature"]==feat]["pct_iqr_3x"].iloc[0]:.1f}% (3x)',
                      fontsize=8)
    axes[i].tick_params(axis='x', labelsize=6)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
plt.tight_layout()
fig.savefig(OUT/'charts'/'outlier_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ════════════════════════════════════════════════════════════════
# 3. FEATURE QUALITY ASSESSMENT
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(' SECTION 3: FEATURE QUALITY ASSESSMENT')
print(f'{"="*60}')

quality_scores = []

for c in num_clean:
    s = clean[c].dropna()
    if len(s) < 100:
        quality_scores.append({'feature': c, 'quality': 0, 'n_obs': len(s), 'reason': 'insufficient_obs'})
        continue

    n_obs = len(s)
    miss_pct = (1 - n_obs/len(clean)) * 100
    # Missing penalty
    miss_penalty = min(30, miss_pct * 3)

    # Skew penalty
    skew = abs(s.skew())
    skew_penalty = min(25, skew * 3)

    # Kurtosis penalty (extreme tails)
    kurt = s.kurtosis()
    kurt_penalty = min(20, abs(kurt) / 5) if kurt > 0 else 0

    # Outlier penalty
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    if iqr > 0:
        out_pct = ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).mean() * 100
    else:
        out_pct = 0
    out_penalty = min(15, out_pct * 2)

    # Temporal stability (year-over-year variance vs total variance)
    try:
        yr_means = clean.groupby(pd.to_datetime(clean['datetime']).dt.year)[c].mean()
        temporal_stability = 1 - min(1, yr_means.std() / s.std()) if s.std() > 0 else 1
    except:
        temporal_stability = 0.5

    # Signal-to-noise ratio (mean / std)
    snr = abs(s.mean()) / s.std() if s.std() > 0 else 0

    quality = max(0, 100 - miss_penalty - skew_penalty - kurt_penalty - out_penalty)
    quality_scores.append({
        'feature': c,
        'quality_score': float(quality),
        'n_obs': int(n_obs),
        'missing_pct': float(miss_pct),
        'skew': float(s.skew()),
        'kurtosis': float(s.kurtosis()),
        'outlier_pct_3iqr': float(out_pct),
        'temporal_stability': float(temporal_stability),
        'snr': float(snr),
        'miss_penalty': float(miss_penalty),
        'skew_penalty': float(skew_penalty),
        'kurt_penalty': float(kurt_penalty),
        'out_penalty': float(out_penalty),
    })

qs_df = pd.DataFrame(quality_scores).sort_values('quality_score')
qs_df.to_csv(OUT/'tables'/'feature_quality_deep.csv', index=False)

print(f'  Feature quality distribution:')
print(f'    Mean: {qs_df["quality_score"].mean():.1f}')
print(f'    Median: {qs_df["quality_score"].median():.1f}')
print(f'    Excellent (>90): {(qs_df["quality_score"]>90).sum()}')
print(f'    Good (70-90): {((qs_df["quality_score"]>=70)&(qs_df["quality_score"]<=90)).sum()}')
print(f'    Fair (50-70): {((qs_df["quality_score"]>=50)&(qs_df["quality_score"]<70)).sum()}')
print(f'    Poor (<50): {(qs_df["quality_score"]<50).sum()}')

# Quality distribution chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(qs_df['quality_score'], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
axes[0,0].axvline(70, color='orange', ls='--', alpha=0.5, label='Good threshold')
axes[0,0].axvline(50, color='red', ls='--', alpha=0.5, label='Poor threshold')
axes[0,0].set_xlabel('Quality Score'); axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Feature Quality Distribution', fontsize=12, fontweight='bold')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].scatter(qs_df['skew'], qs_df['kurtosis'], c=qs_df['quality_score'],
                   cmap='RdYlGn', alpha=0.6, s=20)
axes[0,1].set_xlabel('Skewness'); axes[0,1].set_ylabel('Kurtosis')
axes[0,1].set_title('Skew-Kurtosis vs Quality (color)', fontsize=12, fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].scatter(qs_df['outlier_pct_3iqr'], qs_df['quality_score'],
                   c='steelblue', alpha=0.5, s=15)
axes[1,0].set_xlabel('Outlier % (IQR 3x)'); axes[1,0].set_ylabel('Quality Score')
axes[1,0].set_title('Outlier Rate vs Quality', fontsize=12, fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(qs_df['snr'], qs_df['quality_score'],
                   c='darkorange', alpha=0.5, s=15)
axes[1,1].set_xlabel('Signal-to-Noise Ratio'); axes[1,1].set_ylabel('Quality Score')
axes[1,1].set_title('SNR vs Quality', fontsize=12, fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT/'charts'/'feature_quality_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ════════════════════════════════════════════════════════════════
# 4. DATA INTEGRITY CHECKS
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(' SECTION 4: DATA INTEGRITY CHECKS')
print(f'{"="*60}')

# 4a. OHLC consistency
print('\n[4a] OHLC consistency checks...')
ohlc_raw = ohlc_raw.sort_values(['symbol', 'date'])
violations = []

# High >= Low
n_hl = (ohlc_raw['high'] < ohlc_raw['low']).sum()
violations.append(['High < Low', f'{n_hl:,}', f'{n_hl/len(ohlc_raw)*100:.4f}%',
                  'CRITICAL' if n_hl > 0 else 'OK'])

# Close within [Low, High]
n_close_range = ((ohlc_raw['close'] < ohlc_raw['low']) | (ohlc_raw['close'] > ohlc_raw['high'])).sum()
violations.append(['Close outside range', f'{n_close_range:,}', f'{n_close_range/len(ohlc_raw)*100:.4f}%',
                  'CRITICAL' if n_close_range > 0 else 'OK'])

# Open within [Low, High]
n_open_range = ((ohlc_raw['open'] < ohlc_raw['low']) | (ohlc_raw['open'] > ohlc_raw['high'])).sum()
violations.append(['Open outside range', f'{n_open_range:,}', f'{n_open_range/len(ohlc_raw)*100:.4f}%',
                  'CRITICAL' if n_open_range > 0 else 'OK'])

# Volume > 0
n_vol_zero = (ohlc_raw['volume'] <= 0).sum()
violations.append(['Volume <= 0', f'{n_vol_zero:,}', f'{n_vol_zero/len(ohlc_raw)*100:.4f}%',
                  'WARNING' if n_vol_zero > 0 else 'OK'])

# Stale prices (close == close.shift() for consecutive days)
stale_count = 0
for sym in ohlc_raw['symbol'].unique():
    sd = ohlc_raw[ohlc_raw['symbol']==sym].sort_values('date')
    stale_count += (sd['close'] == sd['close'].shift()).sum()
violations.append(['Stale prices (close unchanged)', f'{stale_count:,}', f'{stale_count/len(ohlc_raw)*100:.2f}%',
                  'WARNING' if stale_count/len(ohlc_raw) > 0.01 else 'OK'])

# Gap detection (returns > 20% intraday)
ohlc_raw['daily_return'] = ohlc_raw.groupby('symbol')['close'].pct_change()
gap_up = (ohlc_raw['open'] > ohlc_raw['close'].shift() * 1.1).sum()
gap_down = (ohlc_raw['open'] < ohlc_raw['close'].shift() * 0.9).sum()
violations.append(['Gap up (>10%)', f'{gap_up:,}', f'{gap_up/len(ohlc_raw)*100:.2f}%', 'INFO'])
violations.append(['Gap down (<-10%)', f'{gap_down:,}', f'{gap_down/len(ohlc_raw)*100:.2f}%', 'INFO'])

# Extreme returns (outlier trading days)
extreme_up = (ohlc_raw['daily_return'] > 0.20).sum()
extreme_down = (ohlc_raw['daily_return'] < -0.20).sum()
violations.append(['Return > 20%', f'{extreme_up:,}', f'{extreme_up/len(ohlc_raw)*100:.4f}%', 'INFO'])
violations.append(['Return < -20%', f'{extreme_down:,}', f'{extreme_down/len(ohlc_raw)*100:.4f}%', 'INFO'])

print('  OHLC Integrity Summary:')
vio_df = pd.DataFrame(violations, columns=['Check', 'Count', 'Rate', 'Severity'])
vio_df.to_csv(OUT/'tables'/'ohlc_integrity.csv', index=False)
for _, r in vio_df.iterrows():
    sev = r['Severity']
    print(f'    [{sev:>8s}] {r["Check"]:<30s} {r["Count"]:>10s} ({r["Rate"]})')

# 4b. Per-symbol data completeness
print('\n[4b] Per-symbol data completeness...')
sym_complete = ohlc_raw.groupby('symbol').agg(
    n_bars=('date', 'count'),
    first_date=('date', 'min'),
    last_date=('date', 'max'),
    n_missing_dates=('daily_return', lambda x: x.isna().sum() if hasattr(x, 'isna') else 0),
).reset_index()

# Count business days between first and last
sym_complete['expected_days'] = (pd.to_datetime(sym_complete['last_date']) -
                                  pd.to_datetime(sym_complete['first_date'])).dt.days + 1
sym_complete['completeness'] = sym_complete['n_bars'] / sym_complete['expected_days']
sym_complete = sym_complete.sort_values('completeness')

print(f'  Mean completeness: {sym_complete["completeness"].mean():.1%}')
print(f'  Symbols with < 70% completeness: {(sym_complete["completeness"] < 0.7).sum()}')

low_complete = sym_complete[sym_complete['completeness'] < 0.7]
if len(low_complete) > 0:
    print('  Low-completeness symbols:')
    for _, r in low_complete.head(10).iterrows():
        print(f'    {r["symbol"]:<15s} completeness={r["completeness"]:.1%} bars={int(r["n_bars"])} '
              f'from={str(r["first_date"])[:10]} to={str(r["last_date"])[:10]}')

sym_complete.to_csv(OUT/'tables'/'symbol_completeness.csv', index=False)

# 4c. Duplicate detection
print('\n[4c] Duplicate detection...')
dup_sym_date = ohlc_raw.duplicated(subset=['symbol', 'date'], keep=False)
n_dup = dup_sym_date.sum()
print(f'  Duplicate symbol-date rows: {n_dup}')

if n_dup > 0:
    dups = ohlc_raw[dup_sym_date].sort_values(['symbol', 'date'])
    dups.to_csv(OUT/'tables'/'duplicate_rows.csv', index=False)
    print(f'  Duplicates saved to {OUT/"tables"/"duplicate_rows.csv"}')

# ════════════════════════════════════════════════════════════════
# 5. CLEANING IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(' SECTION 5: CLEANING IMPACT ANALYSIS')
print(f'{"="*60}')

# 5a. Before/after distribution comparison
print('\n[5a] Before/after distribution comparison...')
shared_feats = [c for c in num_clean if c in raw.columns]
comparison = []
for c in shared_feats[:50]:
    r = raw[c].dropna(); cl = clean[c].dropna()
    if len(r) < 100 or len(cl) < 100: continue
    ks_stat, ks_pval = stats.ks_2samp(r, cl)
    mean_change = (cl.mean() - r.mean()) / r.mean() * 100 if r.mean() != 0 else 0
    std_change = (cl.std() - r.std()) / r.std() * 100 if r.std() != 0 else 0
    comparison.append({
        'feature': c,
        'raw_mean': float(r.mean()), 'clean_mean': float(cl.mean()),
        'raw_std': float(r.std()), 'clean_std': float(cl.std()),
        'mean_change_pct': float(mean_change),
        'std_change_pct': float(std_change),
        'ks_stat': float(ks_stat), 'ks_pval': float(ks_pval),
        'distribution_changed': bool(ks_pval < 0.05),
    })

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(OUT/'tables'/'before_after_comparison.csv', index=False)
n_changed = comp_df['distribution_changed'].sum()
print(f'  Features with significant distribution change (KS p<0.05): {n_changed}/{len(comp_df)}')
print(f'  Mean absolute change: mean={comp_df["mean_change_pct"].abs().mean():.2f}%, '
      f'std={comp_df["std_change_pct"].abs().mean():.2f}%')

# 5b. Correlation stability before/after
print('\n[5b] Correlation stability...')
top_n = min(15, len(shared_feats))
top_feats = om_df.sort_values('pct_iqr_3x', ascending=False).head(top_n)['feature'].tolist()
top_feats = [f for f in top_feats if f in raw.columns and f in clean.columns]

if len(top_feats) >= 2:
    raw_corr = raw[top_feats].corr()
    clean_corr = clean[top_feats].corr()
    corr_diff = (raw_corr - clean_corr).abs().values

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, mat, title in zip(axes,
        [raw_corr.values, clean_corr.values, corr_diff],
        ['Raw Correlations', 'Cleaned Correlations', '|Difference|']):
        im = ax.imshow(mat, cmap='RdBu_r' if 'Diff' not in title else 'YlOrRd',
                       vmin=-1 if 'Diff' not in title else 0,
                       vmax=1 if 'Diff' not in title else corr_diff.max())
        ax.set_xticks(range(len(top_feats))); ax.set_yticks(range(len(top_feats)))
        ax.set_xticklabels([f[:10] for f in top_feats], rotation=90, fontsize=7)
        ax.set_yticklabels([f[:10] for f in top_feats], fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(OUT/'charts'/'correlation_stability.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  Mean abs correlation change: {corr_diff.mean():.6f}')
    print(f'  Max abs correlation change: {corr_diff.max():.6f}')

# 5c. Cleaning strategy comparison for missing values
print('\n[5c] Imputation strategy comparison...')
# Take a feature with missing values for comparison
if len(miss_df) > 0:
    worst_feat = miss_df.iloc[0]['feature']
    print(f'  Testing imputation strategies on "{worst_feat}"...')

    orig_series = raw[worst_feat].copy()
    missing_mask = orig_series.isna()
    n_missing = missing_mask.sum()

    if n_missing > 10 and n_missing/len(orig_series) < 0.3:
        observed = orig_series[~missing_mask]
        missing_idx = orig_series[missing_mask].index

        # Strategy 1: Mean imputation
        imp_mean = orig_series.fillna(observed.mean())

        # Strategy 2: Median imputation
        imp_median = orig_series.fillna(observed.median())

        # Strategy 3: Forward fill
        imp_ffill = orig_series.fillna(method='ffill')

        # Strategy 4: Linear interpolation
        imp_interp = orig_series.interpolate(method='linear')

        # Compare distributions at missing locations
        strategies = {
            'mean': imp_mean[missing_mask],
            'median': imp_median[missing_mask],
            'ffill': imp_ffill[missing_mask],
            'interp': imp_interp[missing_mask],
        }

        print(f'    Imputed values distribution at {n_missing} missing locations:')
        for name, vals in strategies.items():
            print(f'      {name:<8s} mean={vals.mean():.2f} std={vals.std():.2f} min={vals.min():.2f} max={vals.max():.2f}')

        # How do imputed values compare to the observed distribution?
        print(f'    Observed distribution: mean={observed.mean():.2f} std={observed.std():.2f}')

# 5d. Target rate stability before/after cleaning
print('\n[5d] Target rate stability...')
if 'target' in raw.columns and 'target' in clean.columns:
    raw_target_rate = raw['target'].mean()
    clean_target_rate = clean['target'].mean()
    print(f'  Raw target rate: {raw_target_rate:.4f}')
    print(f'  Clean target rate: {clean_target_rate:.4f}')
    print(f'  Change: {(clean_target_rate - raw_target_rate)*100:.2f}pp')

# ════════════════════════════════════════════════════════════════
# 6. CLEANING VALIDATION & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(' SECTION 6: CLEANING VALIDATION & RECOMMENDATIONS')
print(f'{"="*60}')

# 6a. Cleaning effect on model performance (quick test)
print('\n[6a] Cleaning effect on model performance...')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Quick test: train on raw vs clean, compare AUC
val_features = ['range_5', 'hv_20', 'bb_width', 'ret_1d', 'vol_ratio_5']
val_features = [f for f in val_features if f in raw.columns and f in clean.columns]

if len(val_features) >= 2:
    # Raw data
    raw_val = raw[val_features + ['target']].dropna()
    clean_val = clean[val_features + ['target']].dropna()

    # Align sample sizes
    n_test = min(100000, len(raw_val), len(clean_val))
    raw_sample = raw_val.sample(n=n_test, random_state=42)
    clean_sample = clean_val.sample(n=n_test, random_state=42)

    Xr, yr = raw_sample[val_features].values, raw_sample['target'].values
    Xc, yc = clean_sample[val_features].values, clean_sample['target'].values

    # Train RF on raw
    rf_raw = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf_raw.fit(Xr, yr)
    # Train RF on clean
    rf_clean = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf_clean.fit(Xc, yc)

    # Cross-validate: train on raw, test on clean and vice versa
    auc_raw_train_clean_test = roc_auc_score(yc, rf_raw.predict_proba(Xc)[:, 1])
    auc_clean_train_raw_test = roc_auc_score(yr, rf_clean.predict_proba(Xr)[:, 1])
    auc_raw_test = roc_auc_score(yr, rf_raw.predict_proba(Xr)[:, 1])
    auc_clean_test = roc_auc_score(yc, rf_clean.predict_proba(Xc)[:, 1])

    print(f'  RF baseline (50 trees, depth=6):')
    print(f'    Train-on-raw, test-on-raw:   AUC={auc_raw_test:.4f}')
    print(f'    Train-on-clean, test-on-clean: AUC={auc_clean_test:.4f}')
    print(f'    Train-on-raw, test-on-clean:  AUC={auc_raw_train_clean_test:.4f}')
    print(f'    Train-on-clean, test-on-raw:  AUC={auc_clean_train_raw_test:.4f}')

    clean_improvement = (auc_clean_test - auc_raw_test) * 100
    print(f'    Cleaning impact: {clean_improvement:+.2f}pp AUC')
    # This improvement is from training on clean data

# 6b. Outlier vs non-outlier target comparison
print('\n[6b] Outlier vs non-outlier target rate...')
for method, col in [('IQR 1.5x', 'pct_iqr_1_5x'), ('IQR 3x', 'pct_iqr_3x'),
                     ('Z-score 3', 'pct_z_3'), ('MAD 3.5', 'pct_mad_3_5')]:
    om_top = om_df.sort_values(col, ascending=False)
    top_feat = om_top.iloc[0]['feature']
    if top_feat not in clean.columns: continue
    s = clean[top_feat].dropna()
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    if iqr == 0: continue

    is_outlier = (s < q1 - 3*iqr) | (s > q3 + 3*iqr)
    if is_outlier.sum() > 10:
        outlier_target = clean.loc[s[is_outlier].index, 'target'].mean()
        inlier_target = clean.loc[s[~is_outlier].index, 'target'].mean()
        diff = (outlier_target - inlier_target) * 100
        print(f'    {top_feat:<30s} ({method}) outlier_target={outlier_target:.1%} '
              f'inlier_target={inlier_target:.1%} diff={diff:+.1f}pp')

# 6c. Feature ranking: which need most attention
print('\n[6c] Features needing most attention...')
need_attention = qs_df[qs_df['quality_score'] < 60].sort_values('quality_score')
if len(need_attention) > 0:
    print(f'  {len(need_attention)} features with quality < 60:')
    for _, r in need_attention.iterrows():
        reasons = []
        if r['skew_penalty'] > 10: reasons.append(f'skew={r["skew"]:.1f}')
        if r['kurt_penalty'] > 10: reasons.append(f'kurt={r["kurtosis"]:.1f}')
        if r['out_penalty'] > 5: reasons.append(f'outliers={r["outlier_pct_3iqr"]:.1f}%')
        if r['miss_penalty'] > 5: reasons.append(f'missing={r["missing_pct"]:.1f}%')
        print(f'    {r["feature"]:<30s} quality={r["quality_score"]:.0f}  {", ".join(reasons)}')

# 6d. Comprehensive cleaning recommendations
print('\n[6d] Cleaning recommendations...')
recommendations = []

# Feature removal candidates
remove_candidates = qs_df[qs_df['quality_score'] < 40]['feature'].tolist()
if remove_candidates:
    recommendations.append(f'Remove {len(remove_candidates)} features with quality < 40: {remove_candidates[:10]}...')

# Outlier capping
high_out = om_df[om_df['pct_iqr_1_5x'] > 10]['feature'].tolist()
if high_out:
    recommendations.append(f'Aggressive capping needed for {len(high_out)} features with >10% IQR outliers')

# Missing value treatment
if len(miss_df) > 0:
    high_miss = miss_df[miss_df['pct_missing'] > 5]['feature'].tolist()
    if high_miss:
        recommendations.append(f'Address high missing rate (>5%) in {len(high_miss)} features: {high_miss[:5]}...')

# Imputation strategy
recommendations.append('Use median imputation for skewed features, forward-fill for temporal features')

# Symbol-level
low_comp = sym_complete[sym_complete['completeness'] < 0.5]['symbol'].tolist()
if low_comp:
    recommendations.append(f'{len(low_comp)} symbols with <50% data completeness may need exclusion')

# Price integrity
if n_hl > 0 or n_close_range > 0:
    recommendations.append('OHLC violations found - investigate and fix source data')

for rec in recommendations:
    print(f'  - {rec}')

# Save all results summary
summary = {
    'n_features_raw': len(feat_raw),
    'n_features_clean': len(feat_clean),
    'n_features_removed': len(feat_raw) - len(feat_clean),
    'features_removed': [c for c in feat_raw if c not in feat_clean],
    'n_missing_features_raw': int(miss_df.shape[0]) if len(miss_df) > 0 else 0,
    'n_outlier_methods_3_agree': int(agree_df[agree_df['n_methods_agree'] >= 2].shape[0]),
    'mean_quality_score': float(qs_df['quality_score'].mean()),
    'n_poor_quality': int((qs_df['quality_score'] < 50).sum()),
    'n_ohlc_violations': {'high_lt_low': int(n_hl), 'close_outside': int(n_close_range)},
    'mean_symbol_completeness': float(sym_complete['completeness'].mean()),
    'n_low_completeness_symbols': int((sym_complete['completeness'] < 0.7).sum()),
    'n_duplicate_rows': int(n_dup),
    'cleaning_auc_impact_pp': float(clean_improvement) if 'clean_improvement' in dir() else 0,
}

with open(OUT/'tables'/'cleaning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n{"="*60}')
print(f' PHASE 6 DEEP ANALYSIS COMPLETE')
print(f' Time: {time.time()-t0:.0f}s')
print(f' Output: {OUT}')
print(f' Total: {len(list((OUT/"charts").glob("*.png")))} charts, '
      f'{len(list((OUT/"tables").glob("*.csv")))} tables')
print(f'{"="*60}')
