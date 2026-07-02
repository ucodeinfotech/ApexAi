# Phase 4 - Data Mining
# Candlestick patterns, chart patterns, association rules with target
import duckdb, pandas as pd, numpy as np, time, warnings, json
from pathlib import Path
from scipy.stats import chi2_contingency
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'data_mining_results'
OUT.mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' Phase 4 - Data Mining')
print('='*60)

# Import pattern engines
import sys
sys.path.insert(0, str(BASE))
from src.patterns.candlestick import detect_patterns as detect_candle, compute_pattern_stats
from src.patterns.chart_patterns import detect_chart_patterns

con = duckdb.connect(str(DB), read_only=True)

# ── 1. Load a representative sample of symbols for pattern mining ──
print('\n[1] Loading symbol sample (all 1day data)...')
# Get top 100 by data volume
syms = [r[0] for r in con.execute("""
    SELECT symbol, COUNT(*) as cnt FROM raw_market
    WHERE timeframe='1day'
    GROUP BY symbol ORDER BY cnt DESC LIMIT 200
""").fetchall()]
print(f'  Selected {len(syms)} symbols')

# Load OHLCV for all selected symbols
all_data = []
for sym in syms:
    df = con.execute("""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM raw_market WHERE timeframe='1day' AND symbol=?
        ORDER BY datetime
    """, [sym]).fetchdf()
    if len(df) > 200:
        all_data.append(df)
all_data = pd.concat(all_data, ignore_index=True)
print(f'  Loaded: {len(all_data):,} rows, {all_data["symbol"].nunique()} symbols')

# ── 2. Detect candlestick patterns ──
print('\n[2] Detecting candlestick patterns...')
pattern_cols = {}
for sym in all_data['symbol'].unique():
    mask = all_data['symbol'] == sym
    sym_df = all_data[mask].copy()
    candle_masks = detect_candle(sym_df)
    chart_masks = detect_chart_patterns(sym_df)
    masks_combined = pd.concat([candle_masks, chart_masks], axis=1)
    for col in masks_combined.columns:
        if col not in pattern_cols:
            pattern_cols[col] = []
        pattern_cols[col].extend(masks_combined[col].values)

for col in list(pattern_cols.keys()):
    all_data[col] = 0
    vals = pattern_cols[col]
    if len(vals) != len(all_data):
        # Pad or trim
        pass

# Actually let's do this properly per symbol
print('  Computing patterns per symbol...')
for sym in all_data['symbol'].unique():
    mask = all_data['symbol'] == sym
    idx = all_data[mask].index
    sym_df = all_data.loc[idx].copy()
    candle_masks = detect_candle(sym_df)
    chart_masks = detect_chart_patterns(sym_df)
    for col in candle_masks.columns:
        all_data.loc[idx, col] = candle_masks[col].values
    for col in chart_masks.columns:
        all_data.loc[idx, col] = chart_masks[col].values

pat_cols = [c for c in all_data.columns if c not in ('symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume')]
n_pat = len(pat_cols)
print(f'  Total patterns detected: {n_pat}')

# ── 3. Compute target for association ──
print('\n[3] Computing target for pattern-target association...')
all_data = all_data.sort_values(['symbol', 'datetime'])
all_data['next_close'] = all_data.groupby('symbol')['close'].shift(-1)
all_data['next_open'] = all_data.groupby('symbol')['open'].shift(-1)
all_data['target_ret'] = all_data['next_close'] / all_data['next_open'] - 1
all_data['target'] = (all_data['target_ret'] > 0.02).astype(int)
all_data = all_data.dropna(subset=['target'])

# ── 4. Pattern frequency and association with target ──
print('\n[4] Pattern-target association analysis...')
pattern_stats = []
for pat in pat_cols:
    n_occur = all_data[pat].sum()
    if n_occur < 10:
        continue
    hit_rate = all_data.loc[all_data[pat] == 1, 'target'].mean()
    base_rate = all_data['target'].mean()
    lift = hit_rate / base_rate if base_rate > 0 else 0
    n_hits = all_data.loc[all_data[pat] == 1, 'target'].sum()
    n_total = len(all_data[all_data[pat] == 1])

    # Chi-square test
    try:
        tbl = pd.crosstab(all_data[pat], all_data['target'])
        if tbl.shape == (2, 2):
            chi2, p_val, _, _ = chi2_contingency(tbl)
        else:
            chi2, p_val = 0, 1.0
    except:
        chi2, p_val = 0, 1.0

    pattern_stats.append({
        'pattern': pat,
        'occurrences': int(n_occur),
        'frequency_pct': float(n_occur / len(all_data) * 100),
        'hit_rate': float(hit_rate),
        'base_rate': float(base_rate),
        'lift': float(lift),
        'n_gainer_hits': int(n_hits),
        'chi2_pval': float(p_val),
    })

ps_df = pd.DataFrame(pattern_stats)
ps_df = ps_df.sort_values('lift', ascending=False)
ps_df.to_csv(OUT / 'pattern_association.csv', index=False)

print(f'  Patterns with significant association (p<0.05): {(ps_df["chi2_pval"] < 0.05).sum()}/{len(ps_df)}')
print(f'  Top 10 patterns by lift:')
for _, r in ps_df.head(10).iterrows():
    print(f'    {r["pattern"]:<25s} lift={r["lift"]:.2f}  hit={r["hit_rate"]:.1%}  freq={r["frequency_pct"]:.2f}%  p={r["chi2_pval"]:.4f}')

# ── 5. Sequential pattern mining (pattern → next-day gainer) ──
print('\n[5] Sequential pattern mining...')
seq_stats = []
for pat in pat_cols:
    n_occur = all_data[pat].sum()
    if n_occur < 30:
        continue
    # Look at pattern occurrence + next day target
    pat_occur = all_data[pat] == 1
    next_day = pat_occur.shift(1)  # pattern today → gainer tomorrow
    next_day_hit = next_day & (all_data['target'] == 1)
    if next_day.sum() > 0:
        seq_hit_rate = next_day_hit.sum() / next_day.sum()
        seq_stats.append({
            'pattern': pat,
            'pattern_today': int(pat_occur.sum()),
            'gainer_tomorrow_hits': int(next_day_hit.sum()),
            'gainer_tomorrow_rate': float(seq_hit_rate),
            'base_gainer_rate': float(all_data['target'].mean()),
            'lift_vs_base': float(seq_hit_rate / max(all_data['target'].mean(), 0.001)),
        })

sq_df = pd.DataFrame(seq_stats).sort_values('lift_vs_base', ascending=False)
sq_df.to_csv(OUT / 'sequential_patterns.csv', index=False)
print(f'  Sequential patterns found: {len(sq_df)}')
print(f'  Top 10 sequential patterns:')
for _, r in sq_df.head(10).iterrows():
    print(f'    {r["pattern"]:<25s} tomorrow_gainer={r["gainer_tomorrow_rate"]:.1%}  lift={r["lift_vs_base"]:.2f}  n={int(r["pattern_today"])}')

# ── 6. Market structure features for the engineered dataset ──
print('\n[6] Computing market structure features on full universe...')
from src.patterns.market_structure import compute_all_structure

# Load engineered features
ef = pd.read_parquet(BASE / 'engineered_features.parquet')

# Compute structure features per symbol
all_parts = []
for sym in ef['symbol'].unique():
    mask = ef['symbol'] == sym
    idx = ef[mask].index
    sym_ef = ef.loc[idx].copy()
    # Create OHLCV columns from what we have
    sym_ef = sym_ef.sort_values('datetime')
    try:
        struct = compute_all_structure(sym_ef)
        all_parts.append(struct)
    except Exception as e:
        print(f'    Error on {sym}: {e}')

if all_parts:
    struct_df = pd.concat(all_parts, ignore_index=False if all_parts else True)
    struct_cols = [c for c in struct_df.columns if c not in ef.columns]
    ef = ef.join(struct_df[struct_cols], how='left')
    print(f'  Added {len(struct_cols)} structure features')

# Fill NaN in structure features
for c in ef.columns:
    if c not in ('symbol', 'datetime', 'date') and ef[c].dtype in ('float64', 'float32'):
        ef[c] = ef[c].fillna(0).astype(np.float32)

ef.to_parquet(BASE / 'engineered_features.parquet', index=False)
print(f'  Updated engineered_features.parquet with structure features')

# ── 7. Summary ──
print(f'\n[7] Phase 4 Summary')
print(f'  Patterns detected: {n_pat}')
print(f'  Significant associations: {(ps_df["chi2_pval"] < 0.05).sum()}/{len(ps_df)}')
print(f'  Sequential patterns: {len(sq_df)}')
print(f'  Structure features added: {len(struct_cols) if "struct_cols" in dir() else 0}')
print(f'  Time: {time.time()-t0:.0f}s')
print(f'  Results saved to: {OUT}')
