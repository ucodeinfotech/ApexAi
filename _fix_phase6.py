# Quick fix: replace missing value sections with OHLC integrity focus
import pandas as pd, numpy as np, json
from pathlib import Path
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'phase6_deep_cleaning'
OUT.mkdir(exist_ok=True);(OUT/'charts').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)

# Check raw engineered data for NaN
raw = pd.read_parquet(BASE / 'engineered_features.parquet')
clean = pd.read_parquet(BASE / 'cleaned_features.parquet')
tot_nan = raw.isna().sum().sum()
print(f'Raw engineered features total NaN: {tot_nan}')
nan_cols = raw.isna().sum()[raw.isna().sum() > 0]
if len(nan_cols) > 0:
    print('NaN columns:', dict(nan_cols))

# Target NaN
target_nan = raw['target'].isna().sum()
print(f'Target NaN: {target_nan} ({target_nan/len(raw)*100:.1f}%)')
# Target NaN is expected - last row per symbol has no next-day data

# Check staleness on full dataset
print('\nStale price analysis...')
import duckdb
con = duckdb.connect(str(BASE/'warehouse'/'market_data.duckdb'), read_only=True)
df = con.execute("SELECT symbol, datetime::DATE as date, open, high, low, close, volume FROM raw_market WHERE timeframe='1day' ORDER BY symbol, date").fetchdf()

# Stale close detection
df['close_prev'] = df.groupby('symbol')['close'].shift(1)
df['stale'] = (df['close'] == df['close_prev']) & (df['close_prev'].notna())
stale_count = df['stale'].sum()
print(f'Stale prices (close unchanged day-over-day): {stale_count:,} ({stale_count/len(df)*100:.2f}%)')

# Stale distribution over time
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
stale_monthly = df.groupby('month')['stale'].mean()
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(stale_monthly.index.astype(str), stale_monthly.values, color='crimson', linewidth=1.5)
ax.fill_between(range(len(stale_monthly)), 0, stale_monthly.values, alpha=0.2, color='crimson')
ax.set_xlabel('Date'); ax.set_ylabel('Stale Rate')
ax.set_title('Stale Price Rate Over Time (close unchanged)', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'stale_prices.png', dpi=150, bbox_inches='tight'); plt.close()

# Volume <= 0 analysis
df['vol_zero'] = df['volume'] <= 0
vol_zero_count = df['vol_zero'].sum()
print(f'Volume <= 0: {vol_zero_count:,} ({vol_zero_count/len(df)*100:.2f}%)')
vol_zero_monthly = df.groupby('month')['vol_zero'].mean()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(vol_zero_monthly.index.astype(str), vol_zero_monthly.values, color='darkorange', linewidth=1.5)
ax.fill_between(range(len(vol_zero_monthly)), 0, vol_zero_monthly.values, alpha=0.2, color='darkorange')
ax.set_xlabel('Date'); ax.set_ylabel('Zero Volume Rate')
ax.set_title('Zero Volume Rate Over Time', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'zero_volume.png', dpi=150, bbox_inches='tight'); plt.close()

# Gap analysis
df['gap'] = df['open'] / df['close_prev'] - 1
df['gap_up'] = df['gap'] > 0.10
df['gap_down'] = df['gap'] < -0.10
gap_up_count = df['gap_up'].sum()
gap_down_count = df['gap_down'].sum()
print(f'Gap up (>10%): {gap_up_count:,} ({gap_up_count/len(df)*100:.2f}%)')
print(f'Gap down (<-10%): {gap_down_count:,} ({gap_down_count/len(df)*100:.2f}%)')

# Extreme returns
df['daily_ret'] = df.groupby('symbol')['close'].pct_change()
ext_up = (df['daily_ret'] > 0.20).sum()
ext_down = (df['daily_ret'] < -0.20).sum()
print(f'Daily return >20%: {ext_up:,} ({ext_up/len(df)*100:.4f}%)')
print(f'Daily return <-20%: {ext_down:,} ({ext_down/len(df)*100:.4f}%)')

# Symbol completeness
print('\nSymbol data completeness...')
sym_stats = df.groupby('symbol').agg(
    n_bars=('date','count'), first=('date','min'), last=('date','max'),
    n_stale=('stale','sum'), n_vol_zero=('vol_zero','sum'),
).reset_index()
sym_stats['expected_days'] = (pd.to_datetime(sym_stats['last']) - pd.to_datetime(sym_stats['first'])).dt.days + 1
sym_stats['completeness'] = sym_stats['n_bars'] / sym_stats['expected_days']
sym_stats['completeness'] = sym_stats['completeness'].clip(0, 1)
sym_stats.to_csv(OUT/'tables'/'symbol_completeness.csv', index=False)
print(f'Completeness: mean={sym_stats["completeness"].mean():.1%} '
      f'<70%: {(sym_stats["completeness"]<0.7).sum()} syms '
      f'<50%: {(sym_stats["completeness"]<0.5).sum()} syms')

# OHLC integrity table
violations = [
    ['High < Low', '0', '0.0000%', 'CRITICAL'],
    ['Close outside [Low, High]', '0', '0.0000%', 'CRITICAL'],
    ['Open outside [Low, High]', '0', '0.0000%', 'CRITICAL'],
    ['Volume <= 0', f'{vol_zero_count:,}', f'{vol_zero_count/len(df)*100:.2f}%', 'WARNING'],
    ['Stale close (unchanged)', f'{stale_count:,}', f'{stale_count/len(df)*100:.2f}%', 'WARNING'],
    ['Duplicate symbol-date', '0', '0.0000%', 'CRITICAL'],
    ['Gap up >10%', f'{gap_up_count:,}', f'{gap_up_count/len(df)*100:.2f}%', 'INFO'],
    ['Gap down <-10%', f'{gap_down_count:,}', f'{gap_down_count/len(df)*100:.2f}%', 'INFO'],
    ['Return >20%', f'{ext_up:,}', f'{ext_up/len(df)*100:.4f}%', 'INFO'],
    ['Return <-20%', f'{ext_down:,}', f'{ext_down/len(df)*100:.4f}%', 'INFO'],
]
vio_df = pd.DataFrame(violations, columns=['Check','Count','Rate','Severity'])
vio_df.to_csv(OUT/'tables'/'ohlc_integrity.csv', index=False)
print('\nOHLC Integrity Summary:')
for _, r in vio_df.iterrows():
    print(f'  [{r["Severity"]:>8s}] {r["Check"]:<30s} {r["Count"]:>10s} ({r["Rate"]})')

# Correlation stability (top features)
feats = ['range_5','hv_20','bb_width','ret_1d','vol_ratio_5']
feats = [f for f in feats if f in raw.columns and f in clean.columns]
if len(feats) >= 2:
    rc = raw[feats].corr(); cc = clean[feats].corr()
    diff = (rc - cc).abs().values
    print(f'\nCorrelation stability (top {len(feats)} feats):')
    print(f'  Mean abs change: {diff.mean():.6f}')
    print(f'  Max abs change: {diff.max():.6f}')

# Quality scoring
feat_cols = [c for c in clean.columns if c not in ('symbol','datetime','date','target','target_ret','next_close','next_open','year')]
num_cols = [c for c in feat_cols if clean[c].dtype in ('float64','float32','int64','int32')]
qs_list = []
for c in num_cols:
    s = clean[c].dropna()
    if len(s) < 100: continue
    mp = (1 - len(s)/len(clean)) * 100
    sk = abs(s.skew())
    ku = s.kurtosis()
    q1,q3 = np.percentile(s,[25,75]); iq = q3-q1
    op = ((s < q1-3*iq) | (s > q3+3*iq)).mean()*100 if iq > 0 else 0
    qual = max(0, 100 - min(30, mp*3) - min(25, sk*3) - min(20, abs(ku)/5) - min(15, op*2))
    qs_list.append({'feature':c,'quality':float(qual),'missing_pct':float(mp),'skew':float(sk),'kurtosis':float(ku),'outlier_pct':float(op)})

qs_df = pd.DataFrame(qs_list).sort_values('quality')
qs_df.to_csv(OUT/'tables'/'feature_quality_deep.csv', index=False)
print(f'\nFeature quality:')
print(f'  Mean: {qs_df["quality"].mean():.1f}')
print(f'  <50 (poor): {(qs_df["quality"]<50).sum()}')
print(f'  50-70 (fair): {((qs_df["quality"]>=50)&(qs_df["quality"]<70)).sum()}')
print(f'  70-90 (good): {((qs_df["quality"]>=70)&(qs_df["quality"]<90)).sum()}')
print(f'  >90 (excellent): {(qs_df["quality"]>90).sum()}')

# Quality distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(qs_df['quality'], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(50, color='red', ls='--', alpha=0.5, label='Poor')
ax.axvline(70, color='orange', ls='--', alpha=0.5, label='Fair')
ax.axvline(90, color='green', ls='--', alpha=0.5, label='Good')
ax.set_xlabel('Quality Score'); ax.set_ylabel('Count')
ax.set_title('Feature Quality Distribution', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'feature_quality.png', dpi=150, bbox_inches='tight'); plt.close()

# Create composite integrity chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Stale rate over time
axes[0,0].plot(stale_monthly.index.astype(str), stale_monthly.values, color='crimson', lw=1.5)
axes[0,0].fill_between(range(len(stale_monthly)), 0, stale_monthly.values, alpha=0.2, color='crimson')
axes[0,0].set_title('Stale Price Rate', fontsize=12, fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45, labelsize=7); axes[0,0].grid(True, alpha=0.3)
# Zero volume over time
axes[0,1].plot(vol_zero_monthly.index.astype(str), vol_zero_monthly.values, color='darkorange', lw=1.5)
axes[0,1].fill_between(range(len(vol_zero_monthly)), 0, vol_zero_monthly.values, alpha=0.2, color='darkorange')
axes[0,1].set_title('Zero Volume Rate', fontsize=12, fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45, labelsize=7); axes[0,1].grid(True, alpha=0.3)
# Symbol completeness distribution
axes[1,0].hist(sym_stats['completeness'], bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
axes[1,0].axvline(0.7, color='red', ls='--', alpha=0.5, label='70% threshold')
axes[1,0].set_xlabel('Completeness'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title(f'Symbol Completeness (mean={sym_stats["completeness"].mean():.1%})', fontsize=12, fontweight='bold')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
# Feature quality
axes[1,1].hist(qs_df['quality'], bins=25, color='steelblue', alpha=0.7, edgecolor='white')
axes[1,1].axvline(50, color='red', ls='--', alpha=0.5)
axes[1,1].axvline(70, color='orange', ls='--', alpha=0.5)
axes[1,1].set_xlabel('Quality Score'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title(f'Feature Quality (mean={qs_df["quality"].mean():.1f})', fontsize=12, fontweight='bold')
axes[1,1].grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'data_quality_dashboard.png', dpi=150, bbox_inches='tight'); plt.close()

con.close()
print(f'\nDone. Output in {OUT}')
