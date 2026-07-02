"""
Phase 6: Stress Testing — Slippage Sensitivity, Market Impact,
Combined Stress Scenarios, Swing Detection Look-ahead
"""
import pickle, math, warnings, numpy as np, pandas as pd, duckdb
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT_FILE = BASE / 'forensic_audit_phase6.txt'
DB = str(BASE / 'warehouse' / 'market_data.duckdb')
PKL = BASE / 'return_prediction_report_v5' / 'results_v5.pkl'

STT = 0.001; BRK = 0.0003; MIN_BRK = 20; EXCH = 0.0000345; SEBI = 1e-6; GST = 0.18; STAMP = 3e-5; SLIP = 0.0005
TOTAL_POS = 110000
# cost_rt needed for pickle deserialization
def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    total = brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
    return total

report_lines = []
def log(msg):
    report_lines.append(str(msg))
    print(msg)

def standard_cost_no_slip(pos_size):
    if pos_size <= 0:
        return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH * 2 + SEBI * 2
    total = brk_total + STT + EXCH * 2 + SEBI * 2 + STAMP + gst_base * GST
    return total

BASE_COST = standard_cost_no_slip(TOTAL_POS)

def cost_with_slippage(slippage):
    return BASE_COST + slippage * 2

def calc_cagr(rets_pct):
    if len(rets_pct) < 5 or np.std(rets_pct) < 1e-12:
        return 0.0
    return (1 + rets_pct / 100).prod() ** (252 / len(rets_pct)) - 1

def calc_sharpe(rets_pct):
    if len(rets_pct) < 5 or np.std(rets_pct) < 1e-12:
        return 0.0
    return rets_pct.mean() / rets_pct.std() * np.sqrt(252)

log("=" * 80)
log("FORENSIC AUDIT - Phase 6: Stress Testing")
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 80)
log("")

# ============================================================
# LOAD DATA
# ============================================================
log("LOADING DATA...")
log("-" * 40)

with open(PKL, 'rb') as f:
    res = pickle.load(f)
rd = res['rd']
bt = res['bt']

rd['dt'] = pd.to_datetime(rd['dt'])
rd = rd.sort_values(['dt', 'sym']).reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()
rd['yr'] = rd['dt'].dt.year

log(f"Backtest rows (bt): {len(bt)}")
log(f"Prediction rows (rd): {len(rd):,}")
log(f"Unique dates: {rd['dt_norm'].nunique()}")
log(f"Unique symbols: {rd['sym'].nunique()}")
log(f"Date range: {rd['dt'].min()} to {rd['dt'].max()}")
log(f"Base cost (no slippage): {BASE_COST*100:.4f}%")
log("")

conn = duckdb.connect(DB)

# ============================================================
# 1. SLIPPAGE SENSITIVITY
# ============================================================
log("=" * 60)
log("SECTION 1: SLIPPAGE SENSITIVITY")
log("=" * 60)
log("Recomputing strategy returns from scratch on rd data.")
log(f"Base cost (no slippage component): {BASE_COST*100:.4f}%")
log("Cost = base_cost + slippage*2  [slippage in decimal]")
log("")

SLIP_LEVELS = [0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]

dates = sorted(rd['dt_norm'].unique())
date_to_id = {d: i for i, d in enumerate(dates)}
day_ids = np.array([date_to_id[d] for d in rd['dt_norm']])
N_DAYS = len(dates)

stack_vals = rd['stack'].values.astype(np.float64)
avg_vals = rd['avg'].values.astype(np.float64)
act_open_vals = rd['act_open'].values.astype(np.float64)

day_groups = []
for did in range(N_DAYS):
    day_groups.append(np.where(day_ids == did)[0])

STRATEGIES = ['Top-1', 'Top-3', 'Top-5', 'Top-10', 'Top-3+Meta']
N_SEL = {'Top-1': 1, 'Top-3': 3, 'Top-5': 5, 'Top-10': 10}

def compute_strategy_daily_returns(sort_key):
    daily_gross = {s: np.zeros(N_DAYS) for s in STRATEGIES}
    rank_vals = avg_vals if sort_key == 'avg' else stack_vals
    for did, idx in enumerate(day_groups):
        day_ao = act_open_vals[idx]
        day_rank = rank_vals[idx]
        if len(idx) < 3:
            continue
        order = np.argsort(-day_rank)
        for s in STRATEGIES:
            if s == 'Top-3+Meta':
                meta_order = np.argsort(-avg_vals[idx])
                picks = idx[meta_order[:3]]
            else:
                picks = idx[order[:N_SEL[s]]]
            daily_gross[s][did] = day_ao[np.isin(idx, picks)].mean() if s != 'Top-1' else act_open_vals[picks[0]]
    return daily_gross

log("Pre-computing daily gross returns (ranked by stack)...")
daily_gross_stack = compute_strategy_daily_returns('stack')
log("Pre-computing daily gross returns (ranked by avg for Top-3+Meta)...")
daily_gross_avg = compute_strategy_daily_returns('avg')
# Use avg-ranked results for Top-3+Meta, stack-ranked for others
daily_gross_all = {s: daily_gross_stack[s] for s in ['Top-1', 'Top-3', 'Top-5', 'Top-10']}
daily_gross_all['Top-3+Meta'] = daily_gross_avg['Top-3+Meta']

log("")
log("Slippage Sensitivity Results:")
log("-" * 100)
header = f"{'Slippage(bp)':>14s}"
for s in STRATEGIES:
    header += f"  {s+' CAGR':>16s}  {s+' SR':>10s}"
log(header)
log("-" * 100)

slippage_results = {}
for sl in SLIP_LEVELS:
    cost = cost_with_slippage(sl)
    sl_bp = int(sl * 10000)
    row = f"{sl_bp:>8d} bp"
    slip_result = {}
    for s in STRATEGIES:
        gross = daily_gross_all[s]
        daily_net = gross - cost * 100
        cagr = calc_cagr(daily_net) * 100
        sharpe = calc_sharpe(daily_net)
        row += f"  {cagr:>+14.2f}%  {sharpe:>10.4f}"
        slip_result[s] = {'cagr_pct': cagr, 'sharpe': sharpe}
    slippage_results[sl] = slip_result
    log(row)

log("")
log("Slippage Sensitivity Summary Table (CAGR % only):")
summary_header = f"{'Slippage(bp)':>14s}"
for s in STRATEGIES:
    summary_header += f"  {s+' CAGR':>18s}"
log(summary_header)
log("-" * 104)
for sl in SLIP_LEVELS:
    sl_bp = int(sl * 10000)
    row = f"{sl_bp:>8d} bp"
    for s in STRATEGIES:
        row += f"  {slippage_results[sl][s]['cagr_pct']:>+14.2f}%"
    log(row)

log("")

# ============================================================
# 2. MARKET IMPACT ESTIMATION
# ============================================================
log("=" * 60)
log("SECTION 2: MARKET IMPACT ESTIMATION")
log("=" * 60)
log(f"Position size: Rs {TOTAL_POS:,}")
log("Market impact = position_value / (avg_daily_volume_rupees * 0.1)")
log("avg_daily_volume_rupees = avg_volume * avg_close")
log("Result in basis points (bp)")
log("")

vol_df = conn.execute("""
    SELECT symbol,
           AVG(volume) as avg_volume,
           AVG(close) as avg_close
    FROM raw_market
    WHERE timeframe = '1day'
    GROUP BY symbol
    ORDER BY symbol
""").fetchdf()

impact_data = []
for _, r in vol_df.iterrows():
    avg_vol = r['avg_volume']
    avg_close = r['avg_close']
    avg_rupee_vol = avg_vol * avg_close
    if avg_rupee_vol > 0:
        market_impact = TOTAL_POS / (avg_rupee_vol * 0.1) * 10000
    else:
        market_impact = 0
    impact_data.append({
        'symbol': r['symbol'],
        'avg_volume': avg_vol,
        'avg_close': avg_close,
        'avg_rupee_vol': avg_rupee_vol,
        'market_impact_bp': market_impact
    })

impact_df = pd.DataFrame(impact_data)
impact_df = impact_df.sort_values('market_impact_bp', ascending=False)

log(f"{'Symbol':>12s} {'Avg Volume':>14s} {'Avg Close':>12s} {'Avg Rps Vol':>16s} {'Impact(bp)':>12s}")
log("-" * 66)
for _, r in impact_df.iterrows():
    log(f"{r['symbol']:>12s} {r['avg_volume']:>14,.0f} {r['avg_close']:>11,.2f} {r['avg_rupee_vol']:>16,.0f} {r['market_impact_bp']:>10.4f}")

log("")
log("Market Impact Summary:")
log(f"  Mean impact: {impact_df['market_impact_bp'].mean():.4f} bp")
log(f"  Median impact: {impact_df['market_impact_bp'].median():.4f} bp")
log(f"  Max impact: {impact_df['market_impact_bp'].max():.4f} bp")
log(f"  Min impact: {impact_df['market_impact_bp'].min():.4f} bp")
log(f"  P95 impact: {impact_df['market_impact_bp'].quantile(0.95):.4f} bp")
log("")

# Merge with model symbols to get model-specific impact
model_syms = set(rd['sym'].unique())
model_impact = impact_df[impact_df['symbol'].isin(model_syms)].copy()
log(f"Model symbols with impact data: {len(model_impact)}")
log(f"  Mean impact (model syms): {model_impact['market_impact_bp'].mean():.4f} bp")
log(f"  Median impact (model syms): {model_impact['market_impact_bp'].median():.4f} bp")
log(f"  Max impact (model syms): {model_impact['market_impact_bp'].max():.4f} bp")
log("")

# ============================================================
# 3. COMBINED STRESS SCENARIO
# ============================================================
log("=" * 60)
log("SECTION 3: COMBINED STRESS SCENARIO")
log("=" * 60)

scenarios = {
    'Mild':     {'slippage': 0.001,  'survivorship_adj': 0.00, 'cagr_haircut': 0.0},
    'Moderate': {'slippage': 0.002,  'survivorship_adj': 0.02, 'cagr_haircut': 0.0},
    'Severe':   {'slippage': 0.003,  'survivorship_adj': 0.04, 'cagr_haircut': 0.5},
}

log(f"{'Scenario':>12s} {'Top-1 CAGR':>16s} {'Top-3 CAGR':>16s} {'Top-5 CAGR':>16s} {'Top-10 CAGR':>16s} {'T3M CAGR':>16s}")
log("-" * 92)

stress_results = {}
for scen_name, params in scenarios.items():
    cost = cost_with_slippage(params['slippage'])
    row = f"{scen_name:>12s}"
    scen_result = {}
    for s in STRATEGIES:
        gross = daily_gross_all[s]
        daily_net = gross - cost * 100
        raw_cagr = calc_cagr(daily_net) * 100
        if params['cagr_haircut'] > 0:
            net_cagr = raw_cagr * (1 - params['cagr_haircut'])
        else:
            net_cagr = raw_cagr
        net_cagr -= params['survivorship_adj'] * 100
        row += f"  {net_cagr:>+14.2f}%"
        scen_result[s] = net_cagr
    stress_results[scen_name] = scen_result
    log(row)

log("")
log("Scenario Notes:")
log("  Mild:     10bp slippage, normal costs")
log("  Moderate: 20bp slippage, 2% survivorship adjustment")
log("  Severe:   30bp slippage, 4% survivorship, 50% CAGR haircut for look-ahead")
log("")

# ============================================================
# 4. SWING DETECTION LOOK-AHEAD QUANTIFICATION
# ============================================================
log("=" * 60)
log("SECTION 4: SWING DETECTION LOOK-AHEAD QUANTIFICATION")
log("=" * 60)
log("Comparing stored swing_high/swing_low (center=True look-ahead)")
log("with correctly computed (center=False, no look-ahead)")
log("")

SWING_WINDOW = 5

symbols = sorted(rd['sym'].unique())[:20]
log(f"Sampling {len(symbols)} symbols for swing comparison...")

swing_high_total = 0
swing_high_leak = 0
swing_low_total = 0
swing_low_leak = 0

for sym in symbols:
    raw = conn.execute(f"""
        SELECT datetime, open, high, low, close, volume
        FROM raw_market
        WHERE symbol = '{sym}' AND timeframe = '1day'
        ORDER BY datetime
    """).fetchdf()

    if len(raw) < 20:
        continue

    raw['datetime'] = pd.to_datetime(raw['datetime']).dt.tz_localize(None)

    stored = conn.execute(f"""
        SELECT datetime, swing_high, swing_low
        FROM feature_store
        WHERE symbol = '{sym}' AND timeframe = '1day'
        ORDER BY datetime
    """).fetchdf()
    stored['datetime'] = pd.to_datetime(stored['datetime']).dt.tz_localize(None)

    merged = raw.merge(stored, on='datetime', how='inner')
    if len(merged) < 20:
        continue

    high = merged['high'].values.astype(np.float64)
    low = merged['low'].values.astype(np.float64)
    n = len(high)

    # Stored swing flags (non-zero means swing point)
    stored_sh = merged['swing_high'].values.astype(np.float64)
    stored_sl = merged['swing_low'].values.astype(np.float64)
    is_swing_high_stored = stored_sh > 0
    is_swing_low_stored = stored_sl > 0
    swing_high_total += is_swing_high_stored.sum()
    swing_low_total += is_swing_low_stored.sum()

    # Recompute correctly with center=False (no look-ahead)
    correct_sh = np.zeros(n, dtype=bool)
    correct_sl = np.zeros(n, dtype=bool)
    for i in range(n):
        left = max(0, i - SWING_WINDOW + 1)
        if high[i] == high[left:i+1].max() and i >= SWING_WINDOW - 1:
            correct_sh[i] = True
        if low[i] == low[left:i+1].min() and i >= SWING_WINDOW - 1:
            correct_sl[i] = True

    # Count leak: bars marked in stored but NOT in correct
    leak_sh = is_swing_high_stored & (~correct_sh)
    leak_sl = is_swing_low_stored & (~correct_sl)
    swing_high_leak += leak_sh.sum()
    swing_low_leak += leak_sl.sum()

log(f"\nSwing High Analysis ({len(symbols)} symbols, window={SWING_WINDOW}):")
log(f"  Total swing_high bars (stored):     {swing_high_total}")
log(f"  Leaked (not labeled with correct):  {swing_high_leak}")
if swing_high_total > 0:
    leak_pct_sh = swing_high_leak / swing_high_total * 100
    log(f"  Leak percentage:                    {leak_pct_sh:.2f}%")
else:
    log(f"  Leak percentage:                    N/A")

log(f"\nSwing Low Analysis ({len(symbols)} symbols, window={SWING_WINDOW}):")
log(f"  Total swing_low bars (stored):      {swing_low_total}")
log(f"  Leaked (not labeled with correct):  {swing_low_leak}")
if swing_low_total > 0:
    leak_pct_sl = swing_low_leak / swing_low_total * 100
    log(f"  Leak percentage:                    {leak_pct_sl:.2f}%")
else:
    log(f"  Leak percentage:                    N/A")

log("")
log("Note: Swing_high/swing_low in feature_store store the actual price")
log("value at swing points (0 otherwise). Comparison checks if bars")
log("marked as swings with center=True would still be marked with center=False.")
log("")

# ============================================================
# FINAL SUMMARY
# ============================================================
log("=" * 60)
log("PHASE 6 SUMMARY")
log("=" * 60)

log("\n1. Slippage Sensitivity:")
for sl in [SLIP_LEVELS[0], SLIP_LEVELS[1], SLIP_LEVELS[3], SLIP_LEVELS[-1]]:
    sl_bp = int(sl * 10000)
    t1 = slippage_results[sl]['Top-1']['cagr_pct']
    t3 = slippage_results[sl]['Top-3']['cagr_pct']
    log(f"   Slippage={sl_bp:>4d}bp: Top-1={t1:>+8.2f}% Top-3={t3:>+8.2f}%")

log("\n2. Market Impact:")
log(f"   Mean={impact_df['market_impact_bp'].mean():.4f}bp, "
     f"Median={impact_df['market_impact_bp'].median():.4f}bp, "
     f"P95={impact_df['market_impact_bp'].quantile(0.95):.4f}bp")

log("\n3. Combined Stress:")
for s in ['Mild', 'Moderate', 'Severe']:
    t1 = stress_results[s]['Top-1']
    t3 = stress_results[s]['Top-3']
    log(f"   {s:>10s}: Top-1={t1:>+8.2f}% Top-3={t3:>+8.2f}%")

if swing_high_total > 0:
    log(f"\n4. Swing Detection Leakage:")
    log(f"   Swing High leak: {leak_pct_sh:.2f}%")
    log(f"   Swing Low leak:  {leak_pct_sl:.2f}%")

log(f"\n{'='*80}")
log("PHASE 6 COMPLETE")
log(f"{'='*80}")

conn.close()

report = '\n'.join(report_lines)
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)
log(f"\nReport saved to: {OUT_FILE}")
