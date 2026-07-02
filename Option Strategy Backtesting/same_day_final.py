"""
SAME-DAY EXIT: fixed edge cases + last-bar-of-day exit
Strategy: TP target hit during day, or exit at last bar of same trading day
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, re
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
# STEP 1: Spot trades (same engine)
# ============================================================
print("=" * 70)
print("STEP 1: Spot trade engine")
print("=" * 70)

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
            trades.append({"entry_dt": ed, "exit_dt": m5["datetime"].iloc[j], "yr": ts.year, "mo": ts.month, "weekday": ed.weekday()})
            break

trades = pd.DataFrame(trades)
trades["ed_naive"] = trades["entry_dt"].dt.tz_localize(None)
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"  Spot trades: {len(trades)}")
print(f"  Option-era: {len(trades_pre)}")

# ============================================================
# STEP 2: Load option data with same-strike cache
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Load option data")
print("=" * 70)

con = duckdb.connect(DB_PATH)
TABLE = "options_data_clean"
try:
    con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()
except:
    TABLE = "options_data_dedup"
    print(f"  Using {TABLE}")

# Load ATM
df_atm = con.execute(f"""SELECT timestamp, close, strike FROM {TABLE}
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
atm_ts = df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl = df_atm["close"].values.astype(float)
atm_st = df_atm["strike"].values.astype(float)

# Correct lookup: handle idx=0 (first bar match)
def lookup_atm(ed):
    ts64 = np.datetime64(ed, "us")
    i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts):
        return len(atm_ts)-1, atm_cl[-1], atm_st[-1]
    if i == 0 or (i > 0 and atm_ts[i] == ts64):
        # Exact match or before first: use the matched bar
        return i, atm_cl[i], atm_st[i]
    return i-1, atm_cl[i-1], atm_st[i-1]

# Collect unique strikes
strike_set = set()
for ed in trades_pre["ed_naive"]:
    _, _, st = lookup_atm(ed)
    strike_set.add(int(st))
stk_list = sorted(strike_set)
print(f"  Unique strikes: {len(stk_list)}")

# Load per-strike data
con2 = duckdb.connect(DB_PATH)
stk_where = ",".join(str(s) for s in stk_list)
try:
    df_all = con2.execute(f"""SELECT timestamp, close, strike FROM {TABLE}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({stk_where})
    ORDER BY strike, timestamp""").fetchdf()
except:
    df_all = con2.execute(f"""SELECT timestamp, close, strike FROM options_data_dedup
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({stk_where})
    ORDER BY strike, timestamp""").fetchdf()
con2.close()
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)

strike_cache = {}
for stk, grp in df_all.groupby("strike"):
    ts = grp["timestamp"].values.astype("datetime64[us]")
    cl = grp["close"].values.astype(float)
    strike_cache[int(stk)] = {"ts": ts, "cl": cl}

# ============================================================
# STEP 3: Fixed entry + same-day exit functions
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Exit functions")
print("=" * 70)

def get_entry_info(ed):
    """Get (strike, entry_price, s_idx, stk_data) for an entry datetime"""
    si, ep, st = lookup_atm(ed)
    st = int(st)
    sd = strike_cache.get(st)
    if sd is None: return None
    entry_ns = atm_ts[si]
    s_idx = np.searchsorted(sd["ts"], entry_ns)
    if s_idx >= len(sd["cl"]): return None
    return {"strike": st, "ep": float(sd["cl"][s_idx]), "s_idx": s_idx, "stk_data": sd}

def exit_tp_eod(stk_data, s_idx, tp):
    """Same-strike: TP exit if hit, else exit at last bar of same trading day."""
    ep = stk_data["cl"][s_idx]
    entry_ns = stk_data["ts"][s_idx]
    entry_date = entry_ns.astype("datetime64[D]")
    # Find last bar on entry day: look for first bar of next day
    next_day = entry_date + np.timedelta64(24*60, "m")
    last_idx = np.searchsorted(stk_data["ts"], next_day) - 1
    if last_idx < 0 or last_idx <= s_idx:
        return None, None
    # Scan for TP
    for i in range(s_idx + 1, last_idx + 1):
        if stk_data["cl"][i] - ep >= tp:
            return stk_data["cl"][i], stk_data["ts"][i]
    return stk_data["cl"][last_idx], stk_data["ts"][last_idx]

# Build trade infos
trade_infos = []
for ed in trades_pre["ed_naive"]:
    info = get_entry_info(ed)
    trade_infos.append(info)
valid = sum(1 for t in trade_infos if t)
print(f"  Trade infos: {valid}/{len(trade_infos)}")

# Verify: how many have same-day bars after entry?
valid_eod = 0
for info in trade_infos:
    if info is None: continue
    r = exit_tp_eod(info["stk_data"], info["s_idx"], 99999)  # TP unreachable, ensures EOD exit
    if r[0] is not None: valid_eod += 1
print(f"  Trades with same-day exit: {valid_eod}/{valid}")

# ============================================================
# STEP 4: Run strategies
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Running strategies")
print("=" * 70)

def run_strategy(trade_infos, tp, prem_max=None):
    pnls = []
    for info in trade_infos:
        if info is None: continue
        if prem_max is not None and info["ep"] > prem_max: continue
        r = exit_tp_eod(info["stk_data"], info["s_idx"], tp)
        if r[0] is None: continue
        xp, _ = r
        pnl = round(xp - info["ep"], 1)
        pnls.append(pnl)
    return np.array(pnls)

strategies = [
    ("TP5_EOD", 5, None),
    ("TP10_EOD", 10, None),
    ("TP15_EOD", 15, None),
    ("TP20_EOD", 20, None),
    ("TP30_EOD", 30, None),
    ("TP50_EOD", 50, None),
    ("TP75_EOD", 75, None),
    ("TP100_EOD", 100, None),
    ("TP150_EOD", 150, None),
    ("TP200_EOD", 200, None),
    ("TP10_EOD_Prem120", 10, 120),
    ("TP15_EOD_Prem120", 15, 120),
    ("TP20_EOD_Prem120", 20, 120),
    ("TP30_EOD_Prem120", 30, 120),
    ("TP5_EOD_Prem120", 5, 120),
    ("TP50_EOD_Prem80", 50, 80),
    ("TP30_EOD_Prem100", 30, 100),
    ("TP20_EOD_Prem80", 20, 80),
]

results = {}
for name, tp, pmax in strategies:
    pnls = run_strategy(trade_infos, tp, pmax)
    n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
    avg = pnls.mean(); std = pnls.std() if n>1 else 0
    sharpe = avg/std*np.sqrt(252) if std>0 else 0
    cum = np.cumsum(pnls); mx = np.maximum.accumulate(cum); mdd = (mx-cum).max() if len(cum)>0 else 0
    calmar = net/mdd if mdd>0 else 0
    pf_ = pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999
    results[name] = {"n":n, "net":net, "wr":wr, "avg":avg, "sharpe":sharpe, "mdd":mdd, "calmar":calmar, "pf":pf_, "pnls":pnls}
    print(f"  {name:<25} n={n:3d} net={net:>+8,.0f}pts (Rs{net*LOT:>+10,.0f}) WR={wr:5.1f}% Avg={avg:+.1f} Sharpe={sharpe:.2f} Calmar={calmar:.1f}x PF={pf_:.2f}x")

# Also run multi-day hold for comparison
print("\n--- Multi-day hold comparison ---")
def exit_tp_maxd(stk_data, s_idx, tp, max_days):
    end_ns = stk_data["ts"][s_idx] + np.timedelta64(int(max_days*86400*1e6), "us")
    ep = stk_data["cl"][s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(stk_data["cl"]))):
        if stk_data["ts"][i] > end_ns:
            return stk_data["cl"][i], stk_data["ts"][i]
        if stk_data["cl"][i] - ep >= tp: return stk_data["cl"][i], stk_data["ts"][i]
    return None, None

md_strategies = [
    ("TP30_Max7d", 30, 7, None),
    ("TP75_Max10d", 75, 10, None),
    ("TP30_Max7d_Prem130", 30, 7, 130),
    ("TP200_Max10d", 200, 10, None),
    ("TP75_Max10d_Prem120", 75, 10, 120),
]
for name, tp, maxd, pmax in md_strategies:
    pnls = np.array([])
    for info in trade_infos:
        if info is None: continue
        if pmax is not None and info["ep"] > pmax: continue
        r = exit_tp_maxd(info["stk_data"], info["s_idx"], tp, maxd)
        if r[0] is None: continue
        pnls = np.append(pnls, round(r[0]-info["ep"], 1))
    n=len(pnls); net=pnls.sum(); wr=(pnls>0).mean()*100
    avg=pnls.mean(); std=pnls.std() if n>1 else 0
    sharpe=avg/std*np.sqrt(252) if std>0 else 0
    cum=np.cumsum(pnls); mx=np.maximum.accumulate(cum); mdd=(mx-cum).max() if len(cum)>0 else 0
    calmar=net/mdd if mdd>0 else 0
    results["MD_"+name] = {"n":n, "net":net, "wr":wr, "avg":avg, "sharpe":sharpe, "mdd":mdd, "calmar":calmar, "pf":0, "pnls":pnls}
    print(f"  MD_{name:<25} n={n:3d} net={net:>+8,.0f}pts (Rs{net*LOT:>+10,.0f}) WR={wr:5.1f}% Avg={avg:+.1f} Sharpe={sharpe:.2f}")

# ============================================================
# STEP 5: Generate PDF
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Generating PDF report")
print("=" * 70)

# Best same-day
best_name_sd = max(results, key=lambda n: results[n]["net"])
best_name_all = max(results, key=lambda n: results[n]["net"])
print(f"  Best same-day: {best_name_sd}")
print(f"  Best overall: {best_name_all}")

b = results[best_name_sd]
m_tp = re.search(r"TP(\d+)", best_name_sd)
BEST_TP = int(m_tp.group(1)) if m_tp else 30

# Re-run for trade book
best_pnls = []; best_trades = []
pmax = 120 if "Prem120" in best_name_sd else (80 if "Prem80" in best_name_sd else (100 if "Prem100" in best_name_sd else None))
for i, info in enumerate(trade_infos):
    if info is None: continue
    if pmax is not None and info["ep"] > pmax: continue
    r = exit_tp_eod(info["stk_data"], info["s_idx"], BEST_TP)
    if r[0] is None: continue
    xp, xts = r
    best_pnls.append(round(xp-info["ep"], 1))
    best_trades.append({
        "entry": trades_pre.iloc[i]["ed_naive"],
        "exit": pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts, np.datetime64) else xts,
        "strike": info["strike"],
        "entry_prem": round(info["ep"], 1),
        "exit_prem": round(xp, 1),
        "pnl": round(xp-info["ep"], 1),
        "yr": trades_pre.iloc[i]["yr"],
        "mo": trades_pre.iloc[i]["mo"],
    })
best_pnls = np.array(best_pnls)
n = len(best_pnls); net = best_pnls.sum(); trade_pnl_rs = [x*LOT for x in best_pnls]; net_rs = sum(trade_pnl_rs)
capital = 100000; final_rs = capital + net_rs
wr = (best_pnls>0).mean(); aw = best_pnls[best_pnls>0].mean() if (best_pnls>0).sum()>0 else 0
al = best_pnls[best_pnls<0].mean() if (best_pnls<0).sum()>0 else 0
cum = np.cumsum(best_pnls); mx_ = np.maximum.accumulate(cum); mdd_ = (mx_-cum).max()
sharpe = best_pnls.mean()/best_pnls.std()*np.sqrt(252) if best_pnls.std()>0 else 0
calmar_ = net/mdd_ if mdd_>0 else 999
pf_ = best_pnls[best_pnls>0].sum()/abs(best_pnls[best_pnls<0].sum()) if (best_pnls<0).sum()>0 else 999

PDF_NAME = f"SameDay_{best_name_sd}_Report.pdf"
with PdfPages(PDF_NAME) as pdf:
    # PAGE 1: Summary
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
    yl = lambda y: 0.94 - y*0.028
    def t2(txt, y, fs=11, wt="bold"):
        ax.text(0.06, yl(y), txt, fontsize=fs, fontweight=wt, transform=ax.transAxes, verticalalignment="top")
    t2("NIFTY50 SAME-DAY OPTION STRATEGY", 0, fs=14)
    t2(f"Best: {best_name_sd} | Same-day EOD exit", 2, fs=12)
    t2("Entry: Bullish Engulfing (1H) + Breakout+Retest (5M)", 3)
    t2(f"Exit: TP+{BEST_TP}pts or last bar of same day (EOD)", 4)
    t2(f"Data: {TABLE} | Same-strike tracking | LOT={LOT}", 5)
    t2("", 6)
    t2(f"Trades: {n}", 7)
    t2(f"Net PnL: {net:+,.0f} pts = Rs {net_rs:+,.0f}", 8)
    t2(f"Win Rate: {wr:.1%}", 9)
    t2(f"Avg Trade: {best_pnls.mean():+.1f} pts = Rs {best_pnls.mean()*LOT:+,.0f}", 10)
    t2(f"Avg Win: {aw:+.1f} | Avg Loss: {al:+.1f}", 11)
    t2(f"Max Win: {best_pnls.max():+.1f} | Max Loss: {best_pnls.min():+.1f}", 12)
    t2(f"Sharpe: {sharpe:.2f} | Calmar: {calmar_:.1f}x", 13)
    t2(f"Max DD: {mdd_:,.0f} pts = Rs {mdd_*LOT:,.0f}", 14)
    t2(f"Profit Factor: {pf_:.2f}x", 15)
    t2(f"Capital: Rs {capital:,} -> Rs {final_rs:,} ({(final_rs/capital-1)*100:+.1f}%)", 16)
    pdf.savefig(fig); plt.close()

    # PAGE 2: Equity curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={"height_ratios": [2, 1]})
    cum_rs = np.cumsum(trade_pnl_rs) + capital
    ax1.plot(cum_rs, color="green", lw=1)
    ax1.axhline(y=capital, color="gray", ls="--", alpha=0.5)
    ax1.fill_between(range(len(cum_rs)), capital, cum_rs, where=(cum_rs>=capital), color="green", alpha=0.1)
    ax1.fill_between(range(len(cum_rs)), capital, cum_rs, where=(cum_rs<capital), color="red", alpha=0.1)
    ax1.set_title(f"Equity Curve - {best_name_sd} (Rs)"); ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax1.grid(alpha=0.3)
    dd_curve = (mx_-cum)*LOT
    ax2.fill_between(range(len(dd_curve)), 0, dd_curve, color="red", alpha=0.3)
    ax2.set_title("Drawdown (Rs)"); ax2.set_ylabel("Drawdown (Rs)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax2.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # PAGE 3: Yearly + Monthly + Matrix
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
    tb = pd.DataFrame(best_trades)
    yr = tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),
                              avg=("pnl","mean"),mxx=("pnl","max"),mnn=("pnl","min"))
    t2("Yearly Breakdown", 0, fs=14, wt="bold")
    hd = ["Year","Trades","Net(pts)","Net(Rs)","WR","Avg","Max","Min"]
    xs = [0.04,0.08,0.16,0.28,0.38,0.46,0.54,0.62]
    for j,h in enumerate(hd):
        ax.text(xs[j], 0.88, h, fontsize=8, fontweight="bold", transform=ax.transAxes)
    for k,(y_,r) in enumerate(yr.iterrows()):
        yy = 0.86 - k*0.028
        vals = [str(int(y_)),str(int(r["trades"])),f"{r['net']:+,.0f}",f"Rs{r['net']*LOT:+,.0f}",f"{r['wr']:.0%}",f"{r['avg']:+.1f}",f"{r['mxx']:+.1f}",f"{r['mnn']:+.1f}"]
        for j,v in enumerate(vals):
            ax.text(xs[j], yy, v, fontsize=7, transform=ax.transAxes)
    t2(f"Total: {n} trades | {net:+,.0f} pts | Rs {net_rs:+,.0f} | WR {wr:.0%}", 0.86-(len(yr)+1)*0.028, fs=9, wt="bold")

    mo = tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    t2("Monthly Breakdown", 0.86-(len(yr)+4)*0.028, fs=14, wt="bold")
    bm = 0.86-(len(yr)+6)*0.028
    hd2 = ["Month","Trades","Net(pts)","Net(Rs)","WR"]
    xs2 = [0.04,0.12,0.22,0.34,0.44]
    for j,h in enumerate(hd2):
        ax.text(xs2[j], bm, h, fontsize=8, fontweight="bold", transform=ax.transAxes)
    for k,(m_,r) in enumerate(mo.iterrows()):
        yy = bm-(k+1)*0.030
        ax.text(xs2[0], yy, MONTHS[int(m_)-1], fontsize=7, transform=ax.transAxes)
        ax.text(xs2[1], yy, str(int(r["trades"])), fontsize=7, transform=ax.transAxes)
        ax.text(xs2[2], yy, f"{r['net']:+,.0f}", fontsize=7, transform=ax.transAxes)
        ax.text(xs2[3], yy, f"Rs{r['net']*LOT:+,.0f}", fontsize=7, transform=ax.transAxes)
        ax.text(xs2[4], yy, f"{r['wr']:.0%}", fontsize=7, transform=ax.transAxes)

    # Strategy matrix (all strategies)
    t2("All Strategies Comparison", bm-14*0.030, fs=14, wt="bold")
    bm2 = bm-16*0.030
    mx_hd = ["Strategy","N","Net(pt)","Net(Rs)","WR","Avg","Sharp","Calmar","PF"]
    mx_xs = [0.02,0.05,0.12,0.24,0.34,0.41,0.48,0.56,0.64]
    for j,h in enumerate(mx_hd):
        ax.text(mx_xs[j], bm2, h, fontsize=6, fontweight="bold", transform=ax.transAxes)
    rank = 0
    for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
        r = results[name]
        rank += 1
        if rank > 22: continue
        yy = bm2 - rank*0.022
        ax.text(mx_xs[0], yy, name[:20], fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[1], yy, str(r["n"]), fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[2], yy, f"{r['net']:+,.0f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[3], yy, f"Rs{r['net']*LOT:+,.0f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[4], yy, f"{r['wr']:.1f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[5], yy, f"{r['avg']:+.1f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[6], yy, f"{r['sharpe']:.2f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[7], yy, f"{r['calmar']:.1f}", fontsize=4.5, transform=ax.transAxes)
        ax.text(mx_xs[8], yy, f"{r['pf']:.2f}" if r['pf']<999 else "inf", fontsize=4.5, transform=ax.transAxes)
    pdf.savefig(fig); plt.close()

    # Trade book pages
    tpb = tb.copy()
    tpb["entry_str"] = tpb["entry"].dt.strftime("%m/%d %H:%M")
    tpb["exit_str"] = pd.to_datetime(tpb["exit"]).dt.strftime("%m/%d %H:%M")
    tpb["rs"] = tpb["pnl"]*LOT
    tpb_show = tpb[["entry_str","exit_str","strike","entry_prem","exit_prem","pnl","rs"]]
    tpp = 48
    for ps in range(0, len(tpb_show), tpp):
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
        chunk = tpb_show.iloc[ps:ps+tpp]
        t2(f"Trade Book ({ps+1}-{ps+len(chunk)} of {len(tpb_show)})", 0, fs=11, wt="bold")
        t2(f"{best_name_sd} | LOT={LOT} | TP={BEST_TP} Exit=EOD", 1, fs=8)
        hd_ = ["#","Entry","Exit","Strike","E.Prem","X.Prem","P(pt)","P(Rs)"]
        xs_ = [0.02,0.08,0.22,0.34,0.42,0.50,0.58,0.66]
        for j,h in enumerate(hd_):
            ax.text(xs_[j], 0.91, h, fontsize=6, fontweight="bold", transform=ax.transAxes)
        for k,(_,r) in enumerate(chunk.iterrows()):
            yy = 0.89 - k*0.018
            ax.text(xs_[0], yy, str(ps+k+1), fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[1], yy, str(r["entry_str"])[:14], fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[2], yy, str(r["exit_str"])[:14], fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[3], yy, f'{r["strike"]:.0f}', fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[4], yy, f'{r["entry_prem"]:.1f}', fontsize=4.5, transform=ax.transAxes)
            ax.text(xs_[5], yy, f'{r["exit_prem"]:.1f}', fontsize=4.5, transform=ax.transAxes)
            c = "green" if r["pnl"]>0 else "red"
            ax.text(xs_[6], yy, f'{r["pnl"]:+.1f}', fontsize=4.5, color=c, transform=ax.transAxes)
            ax.text(xs_[7], yy, f'Rs{r["rs"]:+,.0f}', fontsize=4.5, color=c, transform=ax.transAxes)
        pdf.savefig(fig); plt.close()

print(f"\nPDF: {PDF_NAME}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("COMPLETE RESULTS MATRIX")
print(f"{'='*70}")
print(f"{'Strategy':<28} {'N':>5} {'Net(pt)':>10} {'Net(Rs)':>13} {'WR':>6} {'Avg':>7} {'Sharp':>7} {'Calmar':>7} {'PF':>7}")
print("-" * 90)
for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
    r = results[name]
    rs_str = "Rs{:+,.0f}".format(r["net"]*LOT)
    pf_str = f"{r['pf']:.2f}x" if r['pf'] < 999 else "inf"
    print(f"{name:<28} {r['n']:>5} {r['net']:>+10,.0f} {rs_str:>13} {r['wr']:>5.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>5.1f}x {pf_str:>7}")

# Data integrity summary
print(f"\n{'='*70}")
print("DATA INTEGRITY VERIFICATION")
print(f"{'='*70}")
print(f"  Spot trades (total): {len(trades)}")
print(f"  Option-era trades: {len(trades_pre)}")
print(f"  Valid ATM lookups: {valid}/{len(trades_pre)}")
print(f"  Same-day exits: {valid_eod}/{valid}")
print(f"  Failed lookups: {len(trades_pre)-valid}")
print(f"  Unique strikes: {len(strike_set)}")
print(f"  Table used: {TABLE}")
print(f"  Multi-day hold results included for comparison (MD_ prefix)")
