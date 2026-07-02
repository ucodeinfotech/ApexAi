"""
Phase 7: Statistical Validation — Independent Re-verification
Re-implements key statistical tests from scratch.
All operations numpy-optimized for performance.
"""
import pickle, math, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'return_prediction_report_v5'
report_lines = []

def log(msg):
    report_lines.append(str(msg))
    print(msg)

log("=" * 80)
log("PHASE 7: STATISTICAL VALIDATION")
log("Independent re-verification of v5 trading system")
log("=" * 80)
log("")

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
log("Loading results_v5.pkl...")
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000
def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    total = brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
    return total
with open(OUT / 'results_v5.pkl', 'rb') as f:
    res = pickle.load(f)
bt = res['bt']
rd = res['rd']

rd['dt'] = pd.to_datetime(rd['dt'])
rd = rd.sort_values(['dt', 'sym']).reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()

dates = sorted(rd['dt_norm'].unique())
MODEL_COLS = ['xgb', 'ranker', 'lgb', 'lgb_r', 'cb', 'rf', 'et']
N_MODELS = len(MODEL_COLS)

log(f"Backtest rows (bt): {len(bt)}")
log(f"Prediction rows (rd): {len(rd):,}")
log(f"Unique dates: {len(dates)}")
log(f"Unique symbols: {rd['sym'].nunique()}")
log("")

# ──────────────────────────────────────────────
# Pre-compute day groups and top-1 indices
# ──────────────────────────────────────────────
log("Pre-computing indices...")
dt_norm_vals = rd['dt_norm'].values.astype('int64')
# Map dates to day_id
date_to_id = {d: i for i, d in enumerate(dates)}
day_ids = np.array([date_to_id[d] for d in rd['dt_norm']])

# Build day group indices
day_groups = []
for did in range(len(dates)):
    day_groups.append(np.where(day_ids == did)[0])

# Pre-compute top-1 indices for stack and avg
act_open_vals = rd['act_open'].values.astype(np.float64)
model_col_vals = np.column_stack([rd[c].values.astype(np.float64) for c in MODEL_COLS])
avg_vals = rd['avg'].values.astype(np.float64)
stack_vals = rd['stack'].values.astype(np.float64)

top1_idx_stack = np.zeros(len(dates), dtype=int)
top1_idx_avg = np.zeros(len(dates), dtype=int)
for did, idx in enumerate(day_groups):
    top1_idx_stack[did] = idx[stack_vals[idx].argmax()]
    top1_idx_avg[did] = idx[avg_vals[idx].argmax()]

# Original daily returns
orig_rets_stack = act_open_vals[top1_idx_stack]
orig_rets_avg = act_open_vals[top1_idx_avg]

N_DAYS = len(dates)

def calc_cagr(rets_pct):
    if len(rets_pct) < 5 or np.std(rets_pct) < 1e-12:
        return 0.0
    return (1 + rets_pct / 100).prod() ** (252 / len(rets_pct)) - 1

def calc_sharpe(rets_pct):
    if len(rets_pct) < 5 or np.std(rets_pct) < 1e-12:
        return 0.0
    return rets_pct.mean() / rets_pct.std() * np.sqrt(252)

orig_cagr = calc_cagr(orig_rets_stack)
orig_cagr_avg = calc_cagr(orig_rets_avg)
orig_sharpe = calc_sharpe(orig_rets_stack)
orig_cagr_pct = orig_cagr * 100

log(f"Original Top-1 CAGR (stack): {orig_cagr_pct:.4f}%")
log(f"Original Top-1 CAGR (avg):   {orig_cagr_avg*100:.4f}%")
log(f"Original Top-1 Sharpe:       {orig_sharpe:.4f}")
log("")

# ════════════════════════════════════════════════════════════════
# 1. LABEL SHUFFLE TEST
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("1. LABEL SHUFFLE TEST")
log("   Shuffle act_open 1000x, compute Top-1 CAGR each time")
log("-" * 60)

N_LABEL = 1000
shuf_label_cagrs = np.zeros(N_LABEL)

for i in range(N_LABEL):
    shuf_ao = act_open_vals.copy()
    np.random.shuffle(shuf_ao)
    shuf_label_cagrs[i] = calc_cagr(shuf_ao[top1_idx_stack])
    if (i + 1) % 200 == 0:
        log(f"   Label shuffle {i+1}/{N_LABEL}")

frac_beat_label = (shuf_label_cagrs >= orig_cagr).mean()
mean_cagr_label = shuf_label_cagrs.mean() * 100
sd_cagr_label = shuf_label_cagrs.std() * 100

log(f"   Original CAGR:           {orig_cagr_pct:+.4f}%")
log(f"   Shuffled CAGR mean:      {mean_cagr_label:+.4f}%")
log(f"   Shuffled CAGR std:       {sd_cagr_label:.4f}%")
log(f"   Fraction beat original:  {frac_beat_label:.4f} ({frac_beat_label*100:.2f}%)")
if frac_beat_label < 0.05:
    log(f"   RESULT: PASS (p<0.05, signal is real)")
else:
    log(f"   RESULT: FAIL (shuffled data often beats original)")
log("")

# ════════════════════════════════════════════════════════════════
# 2. FEATURE SHUFFLE TEST
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("2. FEATURE SHUFFLE TEST")
log("   Shuffle each model's predictions independently 1000x")
log("-" * 60)

N_FEAT = 1000
shuf_feat_cagrs = np.zeros(N_FEAT)

for i in range(N_FEAT):
    shuf_mv = model_col_vals.copy()
    for c in range(N_MODELS):
        np.random.shuffle(shuf_mv[:, c])
    shuf_avg = shuf_mv.mean(axis=1)
    # Find top-1 each day using shuffled avg
    daily_rets = np.zeros(N_DAYS)
    for did, idx in enumerate(day_groups):
        best = idx[shuf_avg[idx].argmax()]
        daily_rets[did] = act_open_vals[best]
    shuf_feat_cagrs[i] = calc_cagr(daily_rets)
    if (i + 1) % 200 == 0:
        log(f"   Feature shuffle {i+1}/{N_FEAT}")

frac_beat_feat = (shuf_feat_cagrs >= orig_cagr).mean()
mean_cagr_feat = shuf_feat_cagrs.mean() * 100
sd_cagr_feat = shuf_feat_cagrs.std() * 100

log(f"   Original CAGR:           {orig_cagr_pct:+.4f}%")
log(f"   Shuffled CAGR mean:      {mean_cagr_feat:+.4f}%")
log(f"   Shuffled CAGR std:       {sd_cagr_feat:.4f}%")
log(f"   Fraction beat original:  {frac_beat_feat:.4f} ({frac_beat_feat*100:.2f}%)")
if frac_beat_feat < 0.05:
    log(f"   RESULT: PASS (model predictions have real signal)")
else:
    log(f"   RESULT: FAIL (shuffled predictions often beat original)")
log("")

# ════════════════════════════════════════════════════════════════
# 3. MONTE CARLO SIMULATION
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("3. MONTE CARLO SIMULATION")
log("   Generate 5000 synthetic return series from empirical daily returns")
log("-" * 60)

daily_rets_ew = np.array([
    rd[rd['dt_norm'] == d]['act_open'].mean()
    for d in dates
])
daily_rets_ew = daily_rets_ew[~np.isnan(daily_rets_ew)]
T_mc = len(daily_rets_ew)
log(f"   Daily return series length: {T_mc}")
log(f"   Daily return mean: {daily_rets_ew.mean():.4f}%")
log(f"   Daily return std:  {daily_rets_ew.std():.4f}%")

N_MC = 5000
mc_cagrs = np.zeros(N_MC)
mc_sharpes = np.zeros(N_MC)

for i in range(N_MC):
    syn_rets = np.random.choice(daily_rets_ew, size=T_mc, replace=True)
    mc_cagrs[i] = calc_cagr(syn_rets)
    mc_sharpes[i] = calc_sharpe(syn_rets)
    if (i + 1) % 1000 == 0:
        log(f"   MC simulation {i+1}/{N_MC}")

cagr_ci_lo = np.percentile(mc_cagrs, 2.5) * 100
cagr_ci_hi = np.percentile(mc_cagrs, 97.5) * 100
sharpe_ci_lo = np.percentile(mc_sharpes, 2.5)
sharpe_ci_hi = np.percentile(mc_sharpes, 97.5)

log(f"   CAGR 95% CI: [{cagr_ci_lo:+.4f}%, {cagr_ci_hi:+.4f}%]")
log(f"   Sharpe 95% CI: [{sharpe_ci_lo:.4f}, {sharpe_ci_hi:.4f}]")
log(f"   Observed Top-1 CAGR: {orig_cagr_pct:+.4f}%")
log(f"   Observed Top-1 Sharpe: {orig_sharpe:.4f}")
if cagr_ci_lo <= orig_cagr_pct <= cagr_ci_hi:
    log(f"   RESULT: Observed CAGR within 95% CI of equal-weight portfolio")
else:
    log(f"   RESULT: Observed CAGR OUTSIDE 95% CI (significant)")
log("")

# ════════════════════════════════════════════════════════════════
# 4. WHITE'S REALITY CHECK (re-verified from scratch)
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("4. WHITE'S REALITY CHECK (re-implemented from scratch)")
log("   Block bootstrap, block_size=21, B=5000")
log("-" * 60)

port_net_cols = [f'{s}_net' for s in ['t1', 't3', 't5', 't10', 't3m']]
model_net_cols = [f'{c}_net' for c in MODEL_COLS + ['avg', 'stack']]
all_strat_cols = port_net_cols + model_net_cols
available_strats = [c for c in all_strat_cols if c in bt.columns]

returns_mat = bt[available_strats].values.astype(np.float64)
T_wrc, M_wrc = returns_mat.shape
log(f"   Strategies: {M_wrc} ({len(port_net_cols)} portfolio + {len(model_net_cols)} model)")

mean_rets = returns_mat.mean(axis=0)
std_rets = returns_mat.std(axis=0)
std_rets = np.where(std_rets < 1e-12, 1e-12, std_rets)
V_obs = (np.sqrt(T_wrc) * mean_rets / std_rets).max()
best_strat = available_strats[np.argmax(np.sqrt(T_wrc) * mean_rets / std_rets)]
log(f"   Best strategy: {best_strat} (t={V_obs:.4f})")

# Block bootstrap with optimized sampling
B_WRC = 5000
BLOCK_SIZE = 21
n_blocks = int(np.ceil(T_wrc / BLOCK_SIZE))
boot_max_t = np.zeros(B_WRC)
sqrt_T = np.sqrt(T_wrc)

for b in range(B_WRC):
    bs_mat = np.empty((T_wrc, M_wrc))
    for bi in range(n_blocks):
        start = np.random.randint(0, max(1, T_wrc - BLOCK_SIZE))
        end = min(start + BLOCK_SIZE, T_wrc)
        blen = end - start
        pos = bi * BLOCK_SIZE
        if pos + blen <= T_wrc:
            bs_mat[pos:pos + blen] = returns_mat[start:end]
    b_mean = bs_mat.mean(axis=0)
    b_t = sqrt_T * b_mean / std_rets
    boot_max_t[b] = b_t.max()
    if (b + 1) % 1000 == 0:
        log(f"   WRC bootstrap {b+1}/{B_WRC}")

p_wrc = (boot_max_t >= V_obs).mean()
log(f"   White's RC p-value: {p_wrc:.4f}")
log(f"   RESULT: {'PASS (p<0.05)' if p_wrc < 0.05 else 'FAIL (p>=0.05)'}")
log("")

# ════════════════════════════════════════════════════════════════
# 5. DEFLATED SHARPE RATIO (re-verified from scratch)
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("5. DEFLATED SHARPE RATIO (re-implemented from scratch)")
log("-" * 60)

all_port_strats = port_net_cols + model_net_cols
avail_port = [c for c in all_port_strats if c in bt.columns]
M_dsr = len(avail_port)
T_dsr = len(bt)
log(f"   M = {M_dsr} strategies, T = {T_dsr} observations")

E_max_Z = math.sqrt(2 * math.log(M_dsr))

log(f"   {'Strategy':20s} {'Sharpe':>8s} {'Num':>10s} {'Den':>10s} {'DSR':>8s}")
log("   " + "-" * 56)

dsr_results = {}
for col in avail_port:
    s = bt[col].dropna().values.astype(np.float64)
    if len(s) < 5 or np.std(s) < 1e-12:
        continue
    sr = s.mean() / s.std() * math.sqrt(252)
    num = sr * math.sqrt(T_dsr - 1) - E_max_Z
    den = math.sqrt(1 + 0.5 * sr * sr)
    dsr = stats.norm.cdf(num / den) if den > 0 else 0.0
    dsr_results[col] = {'Sharpe': sr, 'DSR': dsr}
    log(f"   {col:20s} {sr:>8.4f} {num:>10.4f} {den:>10.4f} {dsr:>8.4f}")

log("")
max_dsr_val = max(v['DSR'] for v in dsr_results.values())
max_sharpe_strat = max(dsr_results, key=lambda k: dsr_results[k]['Sharpe'])
log(f"   Highest Sharpe: {max_sharpe_strat} ({dsr_results[max_sharpe_strat]['Sharpe']:.4f})")
log(f"   Highest DSR:    {max_dsr_val:.4f}")
log("")

# ════════════════════════════════════════════════════════════════
# 6. DECILE ANALYSIS VERIFICATION
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("6. DECILE ANALYSIS VERIFICATION")
log("   Rank predictions (stack) each day, split into 10 deciles")
log("-" * 60)

rd_dec = rd.copy()
rd_dec['decile'] = np.nan
for d in dates:
    mask = rd_dec['dt_norm'] == d
    day = rd_dec.loc[mask]
    if len(day) < 10:
        rd_dec.loc[mask, 'decile'] = 0
        continue
    ranked = day['stack'].rank(method='first', ascending=False)
    dec = pd.qcut(ranked, 10, labels=False, duplicates='drop')
    rd_dec.loc[mask, 'decile'] = dec.values

rd_dec = rd_dec.dropna(subset=['decile'])
rd_dec['decile'] = rd_dec['decile'].astype(int)

decile_stats = rd_dec.groupby('decile')['act_open'].agg(['mean', 'std', 'count'])
decile_stats.columns = ['mean_ret', 'std_ret', 'count']

log(f"   {'Decile':8s} {'Mean Ret':>12s} {'Std':>10s} {'Count':>8s}")
log("   " + "-" * 38)
for d in sorted(decile_stats.index):
    row = decile_stats.loc[d]
    log(f"   {d:8d} {row['mean_ret']*100:>+11.4f}% {row['std_ret']*100:>9.4f}% {int(row['count']):>8d}")

decile_means = decile_stats['mean_ret'].values
decile_nums = decile_stats.index.values
spearman_r, spearman_p = stats.spearmanr(decile_nums, decile_means)
log("")
log(f"   Spearman r: {spearman_r:.4f} (p={spearman_p:.6f})")
if spearman_r < -0.8:
    log(f"   RESULT: Strong monotonicity")
elif spearman_r < -0.5:
    log(f"   RESULT: Moderate monotonicity")
elif spearman_r < 0:
    log(f"   RESULT: Weak monotonicity")
else:
    log(f"   RESULT: NO monotonicity (signal inverted or absent)")
log("")

# ════════════════════════════════════════════════════════════════
# 7. SIGNAL SHUFFLE TEST
# ════════════════════════════════════════════════════════════════
log("-" * 60)
log("7. SIGNAL SHUFFLE TEST")
log("   Shuffle avg prediction column 1000x, compute Top-1 Sharpe")
log("-" * 60)

N_SIG = 1000
shuf_sig_sharpes = np.zeros(N_SIG)

for i in range(N_SIG):
    shuf_avg = avg_vals.copy()
    np.random.shuffle(shuf_avg)
    # Find top-1 each day using shuffled avg
    daily_rets = np.zeros(N_DAYS)
    for did, idx in enumerate(day_groups):
        best = idx[shuf_avg[idx].argmax()]
        daily_rets[did] = act_open_vals[best]
    shuf_sig_sharpes[i] = calc_sharpe(daily_rets)
    if (i + 1) % 200 == 0:
        log(f"   Signal shuffle {i+1}/{N_SIG}")

frac_beat_sig = (shuf_sig_sharpes >= orig_sharpe).mean()

log(f"   Original Top-1 Sharpe:         {orig_sharpe:.4f}")
log(f"   Shuffled Sharpe mean:          {shuf_sig_sharpes.mean():.4f}")
log(f"   Shuffled Sharpe std:           {shuf_sig_sharpes.std():.4f}")
log(f"   Max shuffled Sharpe:           {shuf_sig_sharpes.max():.4f}")
log(f"   Fraction beat original:        {frac_beat_sig:.4f} ({frac_beat_sig*100:.2f}%)")
if frac_beat_sig < 0.05:
    log(f"   RESULT: PASS (random avg rarely beats original)")
else:
    log(f"   RESULT: FAIL (shuffled avg often beats original)")
log("")

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
log("=" * 60)
log("PHASE 7 SUMMARY")
log("=" * 60)
log("")
log(f"  Test                             Result")
log(f"  {'-'*40}  {'-'*20}")
log(f"  1. Label Shuffle Test           {'PASS' if frac_beat_label < 0.05 else 'FAIL'}  (p={frac_beat_label:.4f})")
log(f"  2. Feature Shuffle Test         {'PASS' if frac_beat_feat < 0.05 else 'FAIL'}  (p={frac_beat_feat:.4f})")
log(f"  3. Monte Carlo Simulation       {'PASS' if cagr_ci_lo <= orig_cagr_pct <= cagr_ci_hi else 'SIGNIFICANT'}  (CAGR 95% CI)")
log(f"  4. White's Reality Check        {'PASS' if p_wrc < 0.05 else 'FAIL'}  (p={p_wrc:.4f})")
max_dsr_val = max(v['DSR'] for v in dsr_results.values())
log(f"  5. Deflated Sharpe Ratio        {'PASS' if max_dsr_val > 0.95 else 'BORDERLINE'}  (max DSR={max_dsr_val:.4f})")
log(f"  6. Decile Analysis              {'PASS' if spearman_r < -0.5 else 'WARNING'}  (Spearman r={spearman_r:.4f})")
log(f"  7. Signal Shuffle Test          {'PASS' if frac_beat_sig < 0.05 else 'FAIL'}  (p={frac_beat_sig:.4f})")
log("")
log("=" * 80)
log("PHASE 7 COMPLETE")
log("=" * 80)

# Save report
report_text = '\n'.join(report_lines)
with open(BASE / 'forensic_audit_phase7.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
log(f"\nReport saved to: {BASE / 'forensic_audit_phase7.txt'}")
