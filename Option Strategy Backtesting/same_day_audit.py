"""
COMPREHENSIVE AUDIT + SAME-DAY EXIT STRATEGY (3:25 PM exit)
Step-by-step verification of ALL calculations
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings
from datetime import timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT = 50
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
plt.rcParams.update({"font.size":8, "axes.titlesize":11, "axes.labelsize":9, "figure.dpi":120, "savefig.dpi":150})

DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# ============================================================
# STEP 1: AT + spot trade engine (same as before)
# ============================================================
print("=" * 70)
print("AUDIT 1: Spot trade engine")
print("=" * 70)

def atr(m5):
    tr = pd.concat([m5["high"]-m5["low"], abs(m5["high"]-m5["close"].shift(1)), abs(m5["low"]-m5["close"].shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=14, min_periods=14, adjust=False).mean()

h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in [h1, m5]:
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True)
    d.reset_index(drop=True, inplace=True)

print(f"  1H: {len(h1):,} bars ({h1['datetime'].min().date()} to {h1['datetime'].max().date()})")
print(f"  5M: {len(m5):,} bars ({m5['datetime'].min().date()} to {m5['datetime'].max().date()})")

a5 = atr(m5)
me = m5["datetime"].astype("int64").values
CUT = pd.Timestamp("14:15").time()
trades = []
b = (h1["close"] - h1["open"]).abs()
g = h1["close"] > h1["open"]
rr = h1["close"] < h1["open"]
for i in range(1, len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
    if b.iloc[i] < b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour == 9: continue
    lv = h1["high"].iloc[i]
    ts = h1["datetime"].iloc[i]
    idx = np.searchsorted(me, ts.asm8.view("int64"), side="right")
    if idx >= len(m5["close"]): continue
    bi = idx
    while bi < len(m5["close"]) and m5["close"].iloc[bi] <= lv: bi += 1
    if bi >= len(m5["close"]) - 1: continue
    ri = bi + 1
    while ri < len(m5["close"]):
        if m5["low"].iloc[ri] < lv and m5["close"].iloc[ri] > lv and pd.Series(m5["datetime"]).dt.time.iloc[ri] < CUT: break
        ri += 1
    if ri >= len(m5["close"]): continue
    ed = m5["datetime"].iloc[ri]
    ep_ = m5["close"].iloc[ri]
    if ep_ - m5["low"].iloc[ri] <= 0: continue
    he = ep_
    for j in range(ri, len(m5["close"])):
        ca = a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j] > he: he = m5["high"].iloc[j]
        if m5["close"].iloc[j] < he - 55*ca:
            trades.append({"entry_dt": ed, "exit_dt": m5["datetime"].iloc[j], "yr": ts.year, "mo": ts.month, "weekday": ed.weekday()})
            break

trades = pd.DataFrame(trades)
trades["ed_naive"] = trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"] = trades["exit_dt"].dt.tz_localize(None)
exp_w_l = []
for _, r in trades.iterrows():
    d_ = r["ed_naive"].date()
    da = (3-d_.weekday())%7
    da = da if da > 0 else 7
    exp_w_l.append((r["ed_naive"] + pd.Timedelta(days=da)).replace(hour=15, minute=25))
trades["exp_w"] = exp_w_l
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)

print(f"  Total spot trades: {len(trades)}")
print(f"  Option-era trades: {len(trades_pre)}")

# ============================================================
# STEP 2: Audit option data for ALL trade entry timestamps
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 2: Verifying option data for each trade entry")
print("=" * 70)

con = duckdb.connect(DB_PATH)
TABLE = "options_data_clean"
try:
    con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
except:
    TABLE = "options_data_dedup"
    print(f"  WARNING: options_data_clean not found, using {TABLE}")

# Load ATM data
df_atm = con.execute(f"""
    SELECT timestamp, close, strike FROM {TABLE}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
    ORDER BY timestamp
""").fetchdf()
con.close()

df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
atm_ts = df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl = df_atm["close"].values.astype(float)
atm_st = df_atm["strike"].values.astype(float)

# For each trade, find the ATM bar + verify data exists
print(f"\n  Spot-checking first 10 trades for data quality:")
found_all = True
for i in range(min(10, len(trades_pre))):
    ed = trades_pre.iloc[i]["ed_naive"]
    idx = np.searchsorted(atm_ts, np.datetime64(ed, "us"))
    if idx <= 0 or idx >= len(atm_ts):
        print(f"    Trade {i}: NO ATM DATA for {ed}")
        found_all = False
        continue
    si = idx - 1
    st = atm_st[si]
    ep = atm_cl[si]
    ts_match = atm_ts[si]
    print(f"    Trade {i}: entry={ed} match_ts={pd.Timestamp(ts_match)} strike={st:.0f} prem={ep:.1f}")

if found_all:
    print("  => First 10 trades all have ATM data")

# Now check: do the strike data files actually have bars for each matching strike/time?
print(f"\n  Verifying strike data overlap for first 10 trades:")
strike_set = set()
for ed in trades_pre["ed_naive"]:
    idx = np.searchsorted(atm_ts, np.datetime64(ed, "us"))
    if 0 < idx < len(atm_ts):
        strike_set.add(int(atm_st[idx-1]))

# Load strike data
stk_list = sorted(strike_set)
stk_where = ",".join(str(s) for s in stk_list)
con2 = duckdb.connect(DB_PATH)
try:
    df_all = con2.execute(f"""SELECT timestamp, close, strike FROM {TABLE}
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
        AND strike IN ({stk_where})
        ORDER BY strike, timestamp""").fetchdf()
except:
    df_all = con2.execute(f"""SELECT timestamp, close, strike FROM options_data_dedup
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
        AND strike IN ({stk_where})
        ORDER BY strike, timestamp""").fetchdf()
con2.close()

df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)

verify_ok = 0
for i in range(min(10, len(trades_pre))):
    ed = trades_pre.iloc[i]["ed_naive"]
    idx = np.searchsorted(atm_ts, np.datetime64(ed, "us"))
    if idx <= 0 or idx >= len(atm_ts): continue
    si = idx - 1
    st = int(atm_st[si])
    ts_atm = atm_ts[si]
    # Check this strike has data at this timestamp
    sd = df_all[(df_all["strike"] == st)]
    ts_match = sd["timestamp"].values.astype("datetime64[us]")
    oidx = np.searchsorted(ts_match, ts_atm)
    if 0 < oidx < len(ts_match):
        verify_ok += 1
    else:
        print(f"    Trade {i}: strike {st} has NO bar at {pd.Timestamp(ts_atm)}")

print(f"  Strike data verified for {verify_ok}/10 trades")

# ============================================================
# STEP 3: Add same-day exit function and verify manually
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 3: Same-day exit function (3:25 PM)")
print("=" * 70)

def exit_tp_sameday(stk_data, s_idx, tp, exit_hour=15, exit_min=25):
    """Same-strike day trade: TP exit. Exit at market close if not hit."""
    ep = stk_data["cl"][s_idx]
    entry_ns = stk_data["ts"][s_idx]
    # Build EOD at exit_hour:exit_min same day
    entry_dt = entry_ns.astype("datetime64[D]")
    eod_ns = entry_dt + np.timedelta64(exit_hour*60 + exit_min, "m")
    # Find EOD bar
    eod_idx = np.searchsorted(stk_data["ts"], eod_ns)
    if eod_idx >= len(stk_data["ts"]):
        eod_idx = len(stk_data["ts"]) - 1
    if eod_idx <= s_idx:
        return None, None  # No bars between entry and EOD
    # Scan for TP
    for i in range(s_idx + 1, eod_idx + 1):
        if stk_data["cl"][i] - ep >= tp:
            return stk_data["cl"][i], stk_data["ts"][i]
    # TP not hit, exit at EOD
    return stk_data["cl"][eod_idx], stk_data["ts"][eod_idx]

# Load full strike cache
strike_cache_from_all = {}
for stk, grp in df_all.groupby("strike"):
    ts_arr = grp["timestamp"].values.astype("datetime64[us]")
    cl_arr = grp["close"].values.astype(float)
    hi_arr = grp["close"].values.astype(float)  # Don't have hi/lo, use close
    lo_arr = grp["close"].values.astype(float)
    strike_cache_from_all[int(stk)] = {"ts": ts_arr, "cl": cl_arr, "hi": hi_arr, "lo": lo_arr,
                                        "atr": np.full_like(cl_arr, np.nan)}

# Manual spot-check: verify entry and exit for a few trades
print(f"\n  Manual spot-check of 5 trades with same-day exit (TP=30, exit=15:25):")
def get_trade_info(ed):
    idx = np.searchsorted(atm_ts, np.datetime64(ed, "us"))
    if idx <= 0 or idx >= len(atm_ts): return None
    si = idx - 1
    st = int(atm_st[si])
    ts_atm = atm_ts[si]
    sd = strike_cache_from_all.get(st)
    if sd is None: return None
    s_idx = np.searchsorted(sd["ts"], ts_atm)
    if s_idx >= len(sd["cl"]): return None
    return {"si": si, "strike": st, "ep": float(sd["cl"][s_idx]), "s_idx": s_idx, "stk_data": sd}

for i in range(5):
    ed = trades_pre.iloc[i]["ed_naive"]
    info = get_trade_info(ed)
    if info is None:
        print(f"    Trade {i}: No info for {ed}")
        continue
    sd = info["stk_data"]
    si = info["s_idx"]
    ep = info["ep"]
    st = info["strike"]
    
    # Check TP30 same-day
    r = exit_tp_sameday(sd, si, 30)
    if r[0] is None:
        tp_hit = "NO_DATA"
    else:
        xp, xts = r
        hit_tp = (xp - ep >= 30)
        tp_hit = f"TP_HIT={xp:.1f}" if hit_tp else f"EOD={xp:.1f}"
    
    # Also compute: how many bars available from entry to EOD
    entry_dt = sd["ts"][si].astype("datetime64[D]")
    eod_ns = entry_dt + np.timedelta64(15*60+25, "m")
    eod_idx = np.searchsorted(sd["ts"], eod_ns)
    if eod_idx >= len(sd["ts"]): eod_idx = len(sd["ts"]) - 1
    n_bars = eod_idx - si
    entry_time = pd.Timestamp(sd["ts"][si]).time()
    last_time = pd.Timestamp(sd["ts"][eod_idx]).time() if eod_idx > si and eod_idx < len(sd["ts"]) else "N/A"
    
    print(f"    Trade {i}: entry={ed.date()} T{entry_time} stk={st} ep={ep:.1f}"
          f" -> {tp_hit} | bars={n_bars} | last_time={last_time}")

# ============================================================
# STEP 4: Run same-day strategy for ALL trades + verify
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 4: Running full same-day strategy sweep")
print("=" * 70)

# Build trade info for all option-era trades
trade_infos = []
for i in range(len(trades_pre)):
    ed = trades_pre.iloc[i]["ed_naive"]
    info = get_trade_info(ed)
    trade_infos.append(info)

valid = sum(1 for t in trade_infos if t)
print(f"  Valid trade infos: {valid}/{len(trade_infos)}")

# Check how many entries are before EOD
entry_before_eod = 0
eod_times = []
for info in trade_infos:
    if info is None: continue
    sd = info["stk_data"]
    si = info["s_idx"]
    entry_dt = sd["ts"][si].astype("datetime64[D]")
    eod_ns = entry_dt + np.timedelta64(15*60+25, "m")
    eod_idx = np.searchsorted(sd["ts"], eod_ns)
    if eod_idx >= len(sd["ts"]): eod_idx = len(sd["ts"]) - 1
    if eod_idx > si:
        entry_before_eod += 1
    eod_times.append(pd.Timestamp(sd["ts"][eod_idx]).time() if eod_idx < len(sd["ts"]) else "N/A")

print(f"  Entries before EOD: {entry_before_eod}/{valid}")
print(f"  EOD time range: {min(eod_times)} to {max(eod_times)}")

# Run same-day with different TP targets
def run_strategy(trade_infos, tp, exit_h=15, exit_m=25, pmax=None):
    pnls = []
    for info in trade_infos:
        if info is None: continue
        if pmax is not None and info["ep"] > pmax: continue
        r = exit_tp_sameday(info["stk_data"], info["s_idx"], tp, exit_h, exit_m)
        if r[0] is None: continue
        xp, _ = r
        pnl = round(xp - info["ep"], 1)
        pnls.append(pnl)
    return np.array(pnls)

strategies = [
    ("TP5_EOD1525", 5, 15, 25, None),
    ("TP10_EOD1525", 10, 15, 25, None),
    ("TP15_EOD1525", 15, 15, 25, None),
    ("TP20_EOD1525", 20, 15, 25, None),
    ("TP30_EOD1525", 30, 15, 25, None),
    ("TP50_EOD1525", 50, 15, 25, None),
    ("TP75_EOD1525", 75, 15, 25, None),
    ("TP30_Prem120_EOD1525", 30, 15, 25, 120),
    ("TP20_Prem120_EOD1525", 20, 15, 25, 120),
    ("TP15_Prem120_EOD1525", 15, 15, 25, 120),
    ("TP10_Prem120_EOD1525", 10, 15, 25, 120),
]

results = {}
for name, tp, eh, em, pmax in strategies:
    pnls = run_strategy(trade_infos, tp, eh, em, pmax)
    n = len(pnls)
    net = pnls.sum()
    wr = (pnls > 0).mean() * 100
    avg = pnls.mean()
    std = pnls.std() if n > 1 else 0
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    cum = np.cumsum(pnls)
    mx = np.maximum.accumulate(cum)
    dd = mx - cum
    mdd = dd.max() if len(dd) > 0 else 0
    calmar = net / mdd if mdd > 0 else 0
    results[name] = {"n": n, "net": net, "wr": wr, "avg": avg, "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "pnls": pnls}
    print(f"  {name:<30} n={n:3d} net={net:>+8,.0f}pts (Rs{net*LOT:>+10,.0f}) WR={wr:5.1f}% Avg={avg:+.1f} Sharpe={sharpe:.2f} Calmar={calmar:.1f}x")

# Print comparison table
print(f"\n{'='*70}")
print("SAME-DAY EXIT STRATEGY COMPARISON")
print(f"{'='*70}")
print(f"{'Strategy':<30} {'Trades':>6} {'Net(pt)':>10} {'Net(Rs)':>12} {'WR':>6} {'Avg':>8} {'Sharpe':>7} {'Calmar':>7}")
print("-" * 85)
for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
    r = results[name]
    net_rs = "Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<30} {r['n']:>6} {r['net']:>+10,.0f} {net_rs:>12} {r['wr']:>5.1f}% {r['avg']:>+8.1f} {r['sharpe']:>6.2f} {r['calmar']:>5.1f}x")

# ============================================================
# STEP 5: Generate PDF for best same-day strategy
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Generating PDF report for best same-day strategy")
print("=" * 70)

# Pick top performer
best_name = max(results, key=lambda n: results[n]["net"])
print(f"  Best same-day: {best_name}")

b = results[best_name]
# Parse TP from name
import re
m_tp = re.search(r"TP(\d+)", best_name)
BEST_TP = int(m_tp.group(1)) if m_tp else 30

# Re-run to get trade book
best_pnls = []
best_trades = []
for i, info in enumerate(trade_infos):
    if info is None: continue
    # Check premium filter
    pmax = None
    if "Prem120" in best_name: pmax = 120
    if pmax is not None and info["ep"] > pmax: continue
    r = exit_tp_sameday(info["stk_data"], info["s_idx"], BEST_TP)
    if r[0] is None: continue
    xp, xts = r
    pnl = round(xp - info["ep"], 1)
    best_pnls.append(pnl)
    best_trades.append({
        "entry": trades_pre.iloc[i]["ed_naive"],
        "exit": pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts, np.datetime64) else xts,
        "strike": info["strike"],
        "entry_prem": round(info["ep"], 1),
        "exit_prem": round(xp, 1),
        "pnl": pnl,
        "yr": trades_pre.iloc[i]["yr"],
        "mo": trades_pre.iloc[i]["mo"],
        "weekday": trades_pre.iloc[i]["weekday"],
    })
best_pnls = np.array(best_pnls)

PDF_NAME = f"SameDay_{best_name}_Report.pdf"
with PdfPages(PDF_NAME) as pdf:
    # PAGE 1: Summary
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    p = pd.Series(best_pnls)
    n = len(p)
    net = p.sum()
    wr = (p > 0).mean()
    aw = p[p > 0].mean() if (p > 0).sum() > 0 else 0
    al = p[p < 0].mean() if (p < 0).sum() > 0 else 0
    cum = np.cumsum(p)
    mx_ = np.maximum.accumulate(cum)
    mdd = (mx_ - cum).max()
    sharpe = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
    calmar = net / mdd if mdd > 0 else 999
    trade_pnl_rs = [x * LOT for x in best_pnls]
    net_rs = sum(trade_pnl_rs)
    capital = 100000
    final_rs = capital + net_rs

    yl = lambda y: 0.94 - y * 0.030
    def t2(txt, y, fs=11, wt="bold"):
        ax.text(0.06, yl(y), txt, fontsize=fs, fontweight=wt, transform=ax.transAxes, verticalalignment="top")
    t2(f"NIFTY50 SAME-DAY OPTION STRATEGY", 0, fs=14)
    t2(f"Strategy: {best_name} | Exit 3:25 PM same day", 2, fs=12)
    t2(f"Entry: Bullish Engulfing (1H) + Breakout+Retest (5M)", 3)
    t2(f"Exit: Take profit {BEST_TP} pts on option, or market close (15:25)", 4)
    t2(f"Data: {TABLE} | Same-strike tracking | LOT={LOT}", 5)
    t2("", 6)
    t2(f"Trades: {n}", 7)
    t2(f"Net PnL: {net:+,.0f} pts = Rs {net_rs:+,.0f}", 8)
    t2(f"Win Rate: {wr:.1%}", 9)
    t2(f"Avg Trade: {p.mean():+.1f} pts = Rs {p.mean()*LOT:+,.0f}", 10)
    t2(f"Avg Win: {aw:+.1f} pts | Avg Loss: {al:+.1f} pts", 11)
    t2(f"Max Win: {p.max():+.1f} pts | Max Loss: {p.min():+.1f} pts", 12)
    t2(f"Sharpe: {sharpe:.2f} | Calmar: {calmar:.1f}x", 13)
    t2(f"Max DD: {mdd:,.0f} pts = Rs {mdd*LOT:,.0f}", 14)
    pf_ = p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    t2(f"Profit Factor: {pf_:.2f}x", 15)
    t2(f"Capital: Rs {capital:,} -> Rs {final_rs:,} ({(final_rs/capital-1)*100:+.1f}%)", 16)
    pdf.savefig(fig)
    plt.close()

    # PAGE 2: Equity curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={"height_ratios": [2, 1]})
    cum_rs = np.cumsum(trade_pnl_rs) + capital
    ax1.plot(cum_rs, color="green", lw=1)
    ax1.axhline(y=capital, color="gray", ls="--", alpha=0.5)
    ax1.fill_between(range(len(cum_rs)), capital, cum_rs, where=(cum_rs>=capital), color="green", alpha=0.1)
    ax1.fill_between(range(len(cum_rs)), capital, cum_rs, where=(cum_rs<capital), color="red", alpha=0.1)
    ax1.set_title(f"Equity Curve - {best_name} (Rs)")
    ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))
    ax1.grid(alpha=0.3)
    dd_curve = (mx_ - cum) * LOT
    ax2.fill_between(range(len(dd_curve)), 0, dd_curve, color="red", alpha=0.3)
    ax2.set_title("Drawdown (Rs)")
    ax2.set_ylabel("Drawdown (Rs)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))
    ax2.grid(alpha=0.3)
    pdf.savefig(fig)
    plt.close()

    # PAGE 3: Yearly + Monthly + Strategy Matrix
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    tb = pd.DataFrame(best_trades)
    yr = tb.groupby("yr").agg(trades=("pnl", "count"), net=("pnl", "sum"), wr=("pnl", lambda x: (x>0).mean()),
                              avg=("pnl", "mean"), mxx=("pnl", "max"), mnn=("pnl", "min"))
    t2("Yearly Breakdown", 0, fs=14, wt="bold")
    hd = ["Year", "Trades", "Net(pts)", "Net(Rs)", "WR", "Avg", "Max", "Min"]
    xs = [0.04, 0.08, 0.16, 0.28, 0.38, 0.46, 0.54, 0.62]
    for j, h in enumerate(hd):
        ax.text(xs[j], 0.88, h, fontsize=8, fontweight="bold", transform=ax.transAxes)
    for k, (y_, r) in enumerate(yr.iterrows()):
        yy = 0.86 - k * 0.028
        vals = [str(int(y_)), str(int(r["trades"])), f"{r['net']:+,.0f}", f"Rs{r['net']*LOT:+,.0f}",
                f"{r['wr']:.0%}", f"{r['avg']:+.1f}", f"{r['mxx']:+.1f}", f"{r['mnn']:+.1f}"]
        for j, v in enumerate(vals):
            ax.text(xs[j], yy, v, fontsize=7, transform=ax.transAxes)
    t2(f"Total: {n} trades | {net:+,.0f} pts | Rs {net_rs:+,.0f} | WR {wr:.0%}",
       0.86 - (len(yr)+1)*0.028, fs=9, wt="bold")

    mo = tb.groupby("mo").agg(trades=("pnl", "count"), net=("pnl", "sum"), wr=("pnl", lambda x: (x>0).mean()))
    t2("Monthly Breakdown", 0.86 - (len(yr)+4)*0.028, fs=14, wt="bold")
    bm = 0.86 - (len(yr)+6)*0.028
    hd2 = ["Month", "Trades", "Net(pts)", "Net(Rs)", "WR"]
    xs2 = [0.04, 0.12, 0.22, 0.34, 0.44]
    for j, h in enumerate(hd2):
        ax.text(xs2[j], bm, h, fontsize=8, fontweight="bold", transform=ax.transAxes)
    for k, (m_, r) in enumerate(mo.iterrows()):
        yy = bm - (k+1)*0.030
        ax.text(xs2[0], yy, MONTHS[int(m_)-1], fontsize=7, transform=ax.transAxes)
        ax.text(xs2[1], yy, str(int(r["trades"])), fontsize=7, transform=ax.transAxes)
        ax.text(xs2[2], yy, f"{r['net']:+,.0f}", fontsize=7, transform=ax.transAxes)
        ax.text(xs2[3], yy, f"Rs{r['net']*LOT:+,.0f}", fontsize=7, transform=ax.transAxes)
        ax.text(xs2[4], yy, f"{r['wr']:.0%}", fontsize=7, transform=ax.transAxes)

    t2("Strategy Matrix", bm - 14*0.030, fs=14, wt="bold")
    bm2 = bm - 16*0.030
    mx_hd = ["Strategy", "Trd", "Net(pt)", "Net(Rs)", "WR", "Avg", "Sharp", "Calmar"]
    mx_xs = [0.02, 0.06, 0.14, 0.26, 0.36, 0.44, 0.52, 0.60]
    for j, h in enumerate(mx_hd):
        ax.text(mx_xs[j], bm2, h, fontsize=7, fontweight="bold", transform=ax.transAxes)
    rank = 0
    for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
        r = results[name]
        rank += 1
        if rank > 14: continue
        yy = bm2 - rank*0.024
        ax.text(mx_xs[0], yy, name[:18], fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[1], yy, str(r["n"]), fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[2], yy, f"{r['net']:+,.0f}", fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[3], yy, f"Rs{r['net']*LOT:+,.0f}", fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[4], yy, f"{r['wr']:.1f}", fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[5], yy, f"{r['avg']:+.1f}", fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[6], yy, f"{r['sharpe']:.2f}", fontsize=5, transform=ax.transAxes)
        ax.text(mx_xs[7], yy, f"{r['calmar']:.1f}", fontsize=5, transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close()

    # PAGE 4+: Trade book
    tpb = tb.copy()
    tpb["entry_str"] = tpb["entry"].dt.strftime("%m/%d %H:%M")
    tpb["exit_str"] = pd.to_datetime(tpb["exit"]).dt.strftime("%m/%d %H:%M")
    tpb["rs"] = tpb["pnl"] * LOT
    tpb_show = tpb[["entry_str", "exit_str", "strike", "entry_prem", "exit_prem", "pnl", "rs"]]
    tpp = 45
    for ps in range(0, len(tpb_show), tpp):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        chunk = tpb_show.iloc[ps:ps+tpp]
        t2(f"Trade Book ({ps+1}-{ps+len(chunk)} of {len(tpb_show)})", 0, fs=11, wt="bold")
        t2(f"{best_name} | LOT={LOT} | TP={BEST_TP} Exit=15:25", 1, fs=8)
        hd_ = ["#", "Entry", "Exit", "Strike", "E.Prem", "X.Prem", "P(pt)", "P(Rs)"]
        xs_ = [0.02, 0.08, 0.22, 0.34, 0.42, 0.50, 0.58, 0.66]
        for j, h in enumerate(hd_):
            ax.text(xs_[j], 0.91, h, fontsize=6, fontweight="bold", transform=ax.transAxes)
        for k, (_, r) in enumerate(chunk.iterrows()):
            yy = 0.89 - k*0.019
            ax.text(xs_[0], yy, str(ps+k+1), fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[1], yy, str(r["entry_str"])[:14], fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[2], yy, str(r["exit_str"])[:14], fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[3], yy, f'{r["strike"]:.0f}', fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[4], yy, f'{r["entry_prem"]:.1f}', fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[5], yy, f'{r["exit_prem"]:.1f}', fontsize=4.5, transform=ax.transAxes)
            c = "green" if r["pnl"] > 0 else "red"
            ax.text(xs_[6], yy, f'{r["pnl"]:+.1f}', fontsize=4.5, color=c, transform=ax.transAxes)
            ax.text(xs_[7], yy, f'Rs{r["rs"]:+,.0f}', fontsize=4.5, color=c, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close()

print(f"\nPDF saved: {PDF_NAME}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("FINAL RESULTS - SAME-DAY EXIT (3:25 PM)")
print(f"{'='*70}")
print(f"{'Strategy':<30} {'N':>5} {'Net(pt)':>10} {'Net(Rs)':>12} {'WR':>6} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-" * 85)
for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
    r = results[name]
    net_rs = "Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<30} {r['n']:>5} {r['net']:>+10,.0f} {net_rs:>12} {r['wr']:>5.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>5.1f}x")

print(f"\nVerbose PDF: {PDF_NAME}")

# Also write a text summary
with open("SameDay_Summary.txt", "w") as f:
    f.write(f"Same-Day Exit Strategy Results (exit 15:25)\n")
    f.write(f"{'='*80}\n\n")
    f.write(f"{'Strategy':<30} {'N':>5} {'Net(pt)':>10} {'Net(Rs)':>12} {'WR':>6} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}\n")
    f.write("-"*80 + "\n")
    for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
        r = results[name]
        f.write(f"{name:<30} {r['n']:>5} {r['net']:>+10,.0f} Rs{r['net']*LOT:>+10,.0f} {r['wr']:>5.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>5.1f}x\n")
print("Summary saved to SameDay_Summary.txt")
