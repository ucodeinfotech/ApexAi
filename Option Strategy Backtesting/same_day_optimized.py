"""
Comprehensive same-day optimization sweep with PDF report
Tests: TP targets, premium filters, day filters, entry time filters
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, re, itertools
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
# STEP 1: Spot trades
# ============================================================
print("="*70)
print("STEP 1: Spot trade engine")
print("="*70)

def atr(m5):
    tr = pd.concat([m5["high"]-m5["low"], abs(m5["high"]-m5["close"].shift(1)), abs(m5["low"]-m5["close"].shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=14, min_periods=14, adjust=False).mean()

h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in [h1, m5]:
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True); d.reset_index(drop=True, inplace=True)

a5 = atr(m5); me = m5["datetime"].astype("int64").values
CUT = pd.Timestamp("14:15").time()
trades = []
b = (h1["close"] - h1["open"]).abs(); g = h1["close"] > h1["open"]; rr = h1["close"] < h1["open"]
for i in range(1, len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
    if b.iloc[i] < b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour == 9: continue
    lv = h1["high"].iloc[i]; ts = h1["datetime"].iloc[i]
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
    ed = m5["datetime"].iloc[ri]; ep_ = m5["close"].iloc[ri]
    if ep_ - m5["low"].iloc[ri] <= 0: continue
    he = ep_
    for j in range(ri, len(m5["close"])):
        ca = a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j] > he: he = m5["high"].iloc[j]
        if m5["close"].iloc[j] < he - 55 * ca:
            trades.append({"entry_dt": ed, "exit_dt": m5["datetime"].iloc[j], "yr": ts.year, "mo": ts.month, "weekday": ed.weekday(),
                           "entry_time": ed.time()})
            break

trades = pd.DataFrame(trades)
trades["ed_naive"] = trades["entry_dt"].dt.tz_localize(None)
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"  Option-era trades: {len(trades_pre)}")

# ============================================================
# STEP 2: Load option data
# ============================================================
print("\n" + "="*70)
print("STEP 2: Load option data")
print("="*70)

con = duckdb.connect(DB_PATH)
df_atm = con.execute("""SELECT timestamp, close, strike FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
atm_ts = df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl = df_atm["close"].values.astype(float)
atm_st = df_atm["strike"].values.astype(float)

def lookup_atm(ed):
    ts64 = np.datetime64(ed, "us")
    i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts): return len(atm_ts)-1, atm_cl[-1], atm_st[-1]
    if i == 0: return 0, atm_cl[0], atm_st[0]
    return (i, atm_cl[i], atm_st[i]) if atm_ts[i] == ts64 else (i-1, atm_cl[i-1], atm_st[i-1])

strike_set = set()
entry_times_info = []
for ed in trades_pre["ed_naive"]:
    _, _, st = lookup_atm(ed)
    strike_set.add(int(st))
stk_list = sorted(strike_set)
print(f"  Unique strikes: {len(stk_list)}")

con2 = duckdb.connect(DB_PATH)
stk_where = ",".join(str(s) for s in stk_list)
df_all = con2.execute(f"""SELECT timestamp, close, strike FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({stk_where})
ORDER BY strike, timestamp""").fetchdf()
con2.close()
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)

strike_cache = {}
for stk, grp in df_all.groupby("strike"):
    ts = grp["timestamp"].values.astype("datetime64[us]")
    cl = grp["close"].values.astype(float)
    strike_cache[int(stk)] = {"ts": ts, "cl": cl}

# Build trade infos
trade_infos = []
for ed in trades_pre["ed_naive"]:
    ts64 = np.datetime64(ed, "us")
    i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts): si = len(atm_ts)-1
    elif i == 0: si = 0
    else: si = i if atm_ts[i] == ts64 else i-1
    st = int(atm_st[si])
    sd = strike_cache.get(st)
    if sd is None: trade_infos.append(None); continue
    s_idx = np.searchsorted(sd["ts"], atm_ts[si])
    if s_idx >= len(sd["cl"]): trade_infos.append(None); continue
    trade_infos.append({"strike": st, "ep": float(sd["cl"][s_idx]), "s_idx": s_idx, "stk_data": sd})

# ============================================================
# STEP 3: Same-day exit function
# ============================================================
def exit_tp_eod(stk_data, s_idx, tp):
    ep = stk_data["cl"][s_idx]
    entry_ns = stk_data["ts"][s_idx]
    entry_date = entry_ns.astype("datetime64[D]")
    next_day = entry_date + np.timedelta64(24*60, "m")
    last_idx = np.searchsorted(stk_data["ts"], next_day) - 1
    if last_idx < 0 or last_idx <= s_idx: return None, None
    for i in range(s_idx + 1, last_idx + 1):
        if stk_data["cl"][i] - ep >= tp: return stk_data["cl"][i], stk_data["ts"][i]
    return stk_data["cl"][last_idx], stk_data["ts"][last_idx]

def run_strategy(trade_infos, tp, prem_max=None, day_filter=None, entry_hour_max=None):
    """Run same-day strategy with optional filters.
    day_filter: list of weekday ints (0=Mon, 6=Sun)
    entry_hour_max: max entry hour (e.g., 11 = only entries before 11:00)
    """
    pnls = []
    for idx, info in enumerate(trade_infos):
        if info is None: continue
        if prem_max is not None and info["ep"] > prem_max: continue
        if day_filter is not None and trades_pre.iloc[idx]["weekday"] not in day_filter: continue
        if entry_hour_max is not None:
            et = trades_pre.iloc[idx]["entry_time"]
            if et.hour >= entry_hour_max: continue
        r = exit_tp_eod(info["stk_data"], info["s_idx"], tp)
        if r[0] is None: continue
        xp, _ = r; pnls.append(round(xp - info["ep"], 1))
    return np.array(pnls)

def compute_stats(pnls):
    n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
    avg = pnls.mean(); std = pnls.std() if n>1 else 0
    sharpe = avg/std*np.sqrt(252) if std>0 else 0
    cum = np.cumsum(pnls); mx = np.maximum.accumulate(cum); mdd = (mx-cum).max() if len(cum)>0 else 0
    calmar = net/mdd if mdd>0 else 0
    pf_ = pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999
    return {"n":n, "net":net, "wr":wr, "avg":avg, "sharpe":sharpe, "mdd":mdd, "calmar":calmar, "pf":pf_, "pnls":pnls}

# ============================================================
# STEP 4: Optimization sweep
# ============================================================
print("\n" + "="*70)
print("STEP 3: Optimization sweep")
print("="*70)

all_results = {}
sweep_count = 0

# 1) TP targets (fixed)
tp_targets = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100]
print(f"\n--- TP sweep ({len(tp_targets)} targets) ---")
for tp in tp_targets:
    name = f"TP{tp}"
    pnls = run_strategy(trade_infos, tp)
    all_results[name] = compute_stats(pnls)
    r = all_results[name]
    print(f"  {name:<20} n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f} Cal={r['calmar']:.1f}x PF={r['pf']:.2f}x")

# 2) Premium filters
prem_filters = [60, 80, 100, 120, 150, 200]
best_tp = max(all_results, key=lambda n: all_results[n]["net"])
best_tp_num = int(re.search(r"TP(\d+)", best_tp).group(1))
print(f"\n--- Premium filters (TP={best_tp_num}) ---")
for pm in prem_filters:
    name = f"TP{best_tp_num}_P{pm}"
    pnls = run_strategy(trade_infos, best_tp_num, prem_max=pm)
    all_results[name] = compute_stats(pnls)
    r = all_results[name]
    print(f"  {name:<20} n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f} Cal={r['calmar']:.1f}x PF={r['pf']:.2f}x")

# 3) Day filters (best TP, best prem from above)
best_prem_name = max([n for n in all_results if "TP" in n and "P" in n and best_tp in n], 
                     key=lambda n: all_results[n]["net"]) if any("P" in n for n in all_results) else None
day_combo_name = best_tp
day_combo_prem = int(re.search(r"P(\d+)", best_prem_name).group(1)) if best_prem_name else None
print(f"\n--- Day filters (TP={best_tp_num}, prem<={day_combo_prem}) ---")
days = [0, 1, 2, 3, 4, 5, 6]  # Mon-Sun
for d in days:
    dname = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]
    name = f"TP{best_tp_num}_{dname}"
    pnls = run_strategy(trade_infos, best_tp_num, day_filter=[d])
    all_results[name] = compute_stats(pnls)
    r = all_results[name]
    print(f"  {name:<20} n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f} Cal={r['calmar']:.1f}x PF={r['pf']:.2f}x")

# 4) Entry time filters
print(f"\n--- Entry time filters (TP={best_tp_num}) ---")
for h in [10, 11, 12, 13, 14]:
    name = f"TP{best_tp_num}_Entry<{h}"
    pnls = run_strategy(trade_infos, best_tp_num, entry_hour_max=h)
    all_results[name] = compute_stats(pnls)
    r = all_results[name]
    print(f"  {name:<20} n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f} Cal={r['calmar']:.1f}x PF={r['pf']:.2f}x")

# 5) Combined best: TP + prem + day
print(f"\n--- Combined filters (best TP + best prem) ---")
for d in [0, 1, 2, 3, 4]:
    dname = ["Mon","Tue","Wed","Thu","Fri"][d]
    name = f"TP{best_tp_num}_P{day_combo_prem}_{dname}"
    pnls = run_strategy(trade_infos, best_tp_num, prem_max=day_combo_prem, day_filter=[d])
    all_results[name] = compute_stats(pnls)
    r = all_results[name]
    print(f"  {name:<25} n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f} Cal={r['calmar']:.1f}x PF={r['pf']:.2f}x")

# 6) TP + prem + entry time + day
print(f"\n--- Multi-filter combos ---")
for tp_test in [10, 15, 20, 30, 50]:
    for pm in [80, 100, 120]:
        for h in [11, 12, 13]:
            for d in [0, 1, 2, 3, 4]:
                dname = ["Mon","Tue","Wed","Thu","Fri"][d]
                name = f"TP{tp_test}_P{pm}_E<{h}_{dname}"
                pnls = run_strategy(trade_infos, tp_test, prem_max=pm, entry_hour_max=h, day_filter=[d])
                if len(pnls) >= 5:  # Only track if 5+ trades
                    all_results[name] = compute_stats(pnls)

# Sort and show top 30
print(f"\n{'='*70}")
print(f"TOP 30 SAME-DAY STRATEGIES (out of {len(all_results)})")
print(f"{'='*70}")
print(f"{'Strategy':<30} {'N':>4} {'Net(pt)':>9} {'Net(Rs)':>11} {'WR':>5} {'Avg':>6} {'Sharp':>6} {'Cal':>5} {'PF':>5}")
print("-"*85)
top30 = sorted(all_results.items(), key=lambda x: x[1]["net"], reverse=True)[:30]
for name, r in top30:
    rs_str = "Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<30} {r['n']:>4} {r['net']:>+8,.0f} {rs_str:>11} {r['wr']:>4.1f}% {r['avg']:>+6.1f} {r['sharpe']:>5.2f} {r['calmar']:>4.1f}x {r['pf']:>4.2f}x")

# ============================================================
# STEP 5: Generate PDF for top strategy
# ============================================================
print("\n" + "="*70)
print("STEP 4: Generating PDF report")
print("="*70)

best_name = max(all_results, key=lambda n: all_results[n]["net"])
b = all_results[best_name]
m_tp = re.search(r"TP(\d+)", best_name)
BEST_TP = int(m_tp.group(1)) if m_tp else 50

# Parse filters from name for re-run
pmax = None; dh = None; eh = None
if "P" in best_name:
    mp = re.search(r"P(\d+)", best_name)
    if mp: pmax = int(mp.group(1))
for dname in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
    if dname in best_name:
        dh = [["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dname)]
if "Entry" in best_name:
    me = re.search(r"Entry<(\d+)", best_name)
    if me: eh = int(me.group(1))

# Re-run for trade book
best_pnls = []; best_trades = []
for i, info in enumerate(trade_infos):
    if info is None: continue
    if pmax is not None and info["ep"] > pmax: continue
    if dh is not None and trades_pre.iloc[i]["weekday"] not in dh: continue
    if eh is not None and trades_pre.iloc[i]["entry_time"].hour >= eh: continue
    r = exit_tp_eod(info["stk_data"], info["s_idx"], BEST_TP)
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
        "wd": trades_pre.iloc[i]["weekday"],
        "et": trades_pre.iloc[i]["entry_time"],
    })
best_pnls = np.array(best_pnls)
n = len(best_pnls); net = best_pnls.sum()
trade_pnl_rs = [x*LOT for x in best_pnls]; net_rs = sum(trade_pnl_rs)
capital = 100000; final_rs = capital + net_rs
wr = (best_pnls>0).mean(); aw = best_pnls[best_pnls>0].mean() if (best_pnls>0).sum()>0 else 0
al = best_pnls[best_pnls<0].mean() if (best_pnls<0).sum()>0 else 0
cum = np.cumsum(best_pnls); mx_ = np.maximum.accumulate(cum); mdd_ = (mx_-cum).max()
sharpe = best_pnls.mean()/best_pnls.std()*np.sqrt(252) if best_pnls.std()>0 else 0
calmar_ = net/mdd_ if mdd_>0 else 999
pf_ = best_pnls[best_pnls>0].sum()/abs(best_pnls[best_pnls<0].sum()) if (best_pnls<0).sum()>0 else 999

PDF_NAME = f"SameDayOpt_{best_name}_Report.pdf"
with PdfPages(PDF_NAME) as pdf:
    # PAGE 1: Summary
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
    yl = lambda y: 0.94 - y*0.027
    def t2(txt, y, fs=11, wt="bold"):
        ax.text(0.06, yl(y), txt, fontsize=fs, fontweight=wt, transform=ax.transAxes, verticalalignment="top")
    t2("NIFTY50 SAME-DAY OPTION - OPTIMIZED", 0, fs=14)
    t2(f"Best: {best_name}", 2, fs=12)
    t2(f"Entry: BE(1H) + Breako(5M)", 3)
    t2(f"Exit: TP{BEST_TP} or EOD last bar (~15:20-15:30)", 4)
    t2(f"Same-strike tracking | options_data_clean | LOT={LOT}", 5)
    t2("", 6)
    t2(f"Trades: {n}", 7)
    t2(f"Net: {net:+,.0f}pts = Rs{net_rs:+,.0f}", 8)
    t2(f"WR: {wr:.1%}", 9)
    t2(f"Avg: {best_pnls.mean():+.1f}pts = Rs{best_pnls.mean()*LOT:+,.0f}", 10)
    t2(f"AvgW:{aw:+.1f} AvgL:{al:+.1f}", 11)
    t2(f"MaxW:{best_pnls.max():+.1f} MaxL:{best_pnls.min():+.1f}", 12)
    t2(f"Sharpe:{sharpe:.2f} Calmar:{calmar_:.1f}x", 13)
    t2(f"MDD:{mdd_:,.0f}pts RS{mdd_*LOT:,.0f}", 14)
    t2(f"PF:{pf_:.2f}x", 15)
    t2(f"Capital:Rs{capital:,} -> Rs{final_rs:,} ({(final_rs/capital-1)*100:+.1f}%)", 16)
    pdf.savefig(fig); plt.close()

    # PAGE 2: Equity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={"height_ratios": [2, 1]})
    cum_rs = np.cumsum(trade_pnl_rs) + capital
    ax1.plot(cum_rs, color="green", lw=1)
    ax1.axhline(y=capital, color="gray", ls="--", alpha=0.5)
    ax1.fill_between(range(len(cum_rs)), capital, cum_rs, where=(cum_rs>=capital), color="green", alpha=0.1)
    ax1.fill_between(range(len(cum_rs)), capital, cum_rs, where=(cum_rs<capital), color="red", alpha=0.1)
    ax1.set_title(f"Equity - {best_name}"); ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax1.grid(alpha=0.3)
    dd_curve = (mx_-cum)*LOT
    ax2.fill_between(range(len(dd_curve)), 0, dd_curve, color="red", alpha=0.3)
    ax2.set_title("Drawdown (Rs)"); ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax2.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # PAGE 3: Yearly + Monthly + Top 30
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
    tb = pd.DataFrame(best_trades)
    yr = tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),
                              avg=("pnl","mean"),mxx=("pnl","max"),mnn=("pnl","min"))
    t2("Yearly", 0, fs=12, wt="bold")
    hd = ["Year","N","Net(pt)","Net(Rs)","WR","Avg","Max","Min"]
    xs = [0.04,0.08,0.16,0.28,0.38,0.46,0.54,0.62]
    for j,h in enumerate(hd):
        ax.text(xs[j], 0.88, h, fontsize=7, fontweight="bold", transform=ax.transAxes)
    for k,(y_,r) in enumerate(yr.iterrows()):
        yy = 0.86 - k*0.026
        vals = [str(int(y_)),str(int(r["trades"])),f"{r['net']:+,.0f}",f"Rs{r['net']*LOT:+,.0f}",f"{r['wr']:.0%}",f"{r['avg']:+.1f}",f"{r['mxx']:+.1f}",f"{r['mnn']:+.1f}"]
        for j,v in enumerate(vals):
            ax.text(xs[j], yy, v, fontsize=6.5, transform=ax.transAxes)
    t2(f"Total:{n}trd {net:+,.0f}pts Rs{net_rs:+,.0f} WR{wr:.0%}", 0.86-(len(yr)+1)*0.026, fs=8, wt="bold")

    mo = tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    t2("Monthly", 0.86-(len(yr)+4)*0.026, fs=12, wt="bold")
    bm = 0.86-(len(yr)+6)*0.026
    hd2 = ["Mon","N","Net(pt)","Net(Rs)","WR"]
    xs2 = [0.04,0.10,0.20,0.32,0.42]
    for j,h in enumerate(hd2):
        ax.text(xs2[j], bm, h, fontsize=7, fontweight="bold", transform=ax.transAxes)
    for k,(m_,r) in enumerate(mo.iterrows()):
        yy = bm-(k+1)*0.028
        ax.text(xs2[0], yy, MONTHS[int(m_)-1], fontsize=6.5, transform=ax.transAxes)
        ax.text(xs2[1], yy, str(int(r["trades"])), fontsize=6.5, transform=ax.transAxes)
        ax.text(xs2[2], yy, f"{r['net']:+,.0f}", fontsize=6.5, transform=ax.transAxes)
        ax.text(xs2[3], yy, f"Rs{r['net']*LOT:+,.0f}", fontsize=6.5, transform=ax.transAxes)
        ax.text(xs2[4], yy, f"{r['wr']:.0%}", fontsize=6.5, transform=ax.transAxes)

    # Top 30 strategies
    t2("Top 30 Strategies", bm-15*0.026, fs=12, wt="bold")
    bm2 = bm-17*0.026
    mx_hd = ["Strategy","N","Net(pt)","Net(Rs)","WR","Avg","Sharp","Cal","PF"]
    mx_xs = [0.02,0.05,0.12,0.24,0.34,0.42,0.50,0.58,0.66]
    for j,h in enumerate(mx_hd):
        ax.text(mx_xs[j], bm2, h, fontsize=5.5, fontweight="bold", transform=ax.transAxes)
    rank = 0
    for name, r in top30:
        rank += 1
        yy = bm2 - rank*0.021
        ax.text(mx_xs[0], yy, name[:20], fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[1], yy, str(r["n"]), fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[2], yy, f"{r['net']:+,.0f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[3], yy, f"Rs{r['net']*LOT:+,.0f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[4], yy, f"{r['wr']:.1f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[5], yy, f"{r['avg']:+.1f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[6], yy, f"{r['sharpe']:.2f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[7], yy, f"{r['calmar']:.1f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[8], yy, f"{r['pf']:.2f}", fontsize=4.5, transform=ax.transAxes)
    pdf.savefig(fig); plt.close()

    # Trade book
    tpb = tb.copy()
    tpb["entry_str"] = tpb["entry"].dt.strftime("%m/%d %H:%M")
    tpb["exit_str"] = pd.to_datetime(tpb["exit"]).dt.strftime("%m/%d %H:%M")
    tpb["dow"] = tpb["wd"].apply(lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    tpb["rs"] = tpb["pnl"]*LOT
    tpb_show = tpb[["entry_str","exit_str","strike","entry_prem","exit_prem","pnl","rs","dow"]]
    tpp = 45
    for ps in range(0, len(tpb_show), tpp):
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
        chunk = tpb_show.iloc[ps:ps+tpp]
        t2(f"Trade Book ({ps+1}-{ps+len(chunk)}/{len(tpb_show)})", 0, fs=10, wt="bold")
        t2(f"{best_name} LOT={LOT} TP={BEST_TP} EOD", 1, fs=7)
        hd_ = ["#","Entry","Exit","Strike","E.P","X.P","P(pt)","P(Rs)","Day"]
        xs_ = [0.02,0.07,0.20,0.32,0.39,0.46,0.53,0.60,0.68]
        for j,h in enumerate(hd_):
            ax.text(xs_[j], 0.91, h, fontsize=5.5, fontweight="bold", transform=ax.transAxes)
        for k,(_,r) in enumerate(chunk.iterrows()):
            yy = 0.89 - k*0.018
            ax.text(xs_[0], yy, str(ps+k+1), fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[1], yy, str(r["entry_str"])[:12], fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[2], yy, str(r["exit_str"])[:12], fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[3], yy, f'{r["strike"]:.0f}', fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[4], yy, f'{r["entry_prem"]:.1f}', fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[5], yy, f'{r["exit_prem"]:.1f}', fontsize=4.5, transform=ax.transAxes)
            c = "green" if r["pnl"]>0 else "red"
            ax.text(xs_[6], yy, f'{r["pnl"]:+.1f}', fontsize=4.5, color=c, transform=ax.transAxes)
            ax.text(xs_[7], yy, f'Rs{r["rs"]:+,.0f}', fontsize=4.5, color=c, transform=ax.transAxes)
            ax.text(xs_[8], yy, r["dow"], fontsize=4.5, transform=ax.transAxes)
        pdf.savefig(fig); plt.close()

print(f"\nPDF: {PDF_NAME}")

# Final summary
print(f"\n{'='*70}")
print("FINAL OPTIMIZED RESULTS")
print(f"{'='*70}")
print(f"{'Strategy':<30} {'N':>4} {'Net(pt)':>9} {'Net(Rs)':>11} {'WR':>5} {'Avg':>6} {'Sharp':>6} {'Cal':>5} {'PF':>5}")
print("-"*85)
for name, r in top30:
    rs_str = "Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<30} {r['n']:>4} {r['net']:>+8,.0f} {rs_str:>11} {r['wr']:>4.1f}% {r['avg']:>+6.1f} {r['sharpe']:>5.2f} {r['calmar']:>4.1f}x {r['pf']:>4.2f}x")
