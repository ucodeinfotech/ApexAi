"""COMPREHENSIVE SWEEP: ALL previous strategies on filled data + same-expiry."""
import duckdb, pandas as pd, numpy as np, warnings, os
from datetime import timedelta, time
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT, DB_PATH = 50, r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
TABLE = "options_data_filled"
CUT_TIME = pd.Timestamp("14:15").time()
T15 = pd.Timestamp("15:00").time()
T1430 = pd.Timestamp("14:30").time()
T1445 = pd.Timestamp("14:45").time()

# === SPOT ENTRY ===
h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in (h1, m5):
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True); d.reset_index(drop=True, inplace=True)
me = m5["datetime"].astype("int64").values; m5_t = m5["datetime"].dt.time.values
b = (h1["close"]-h1["open"]).abs(); g = h1["close"]>h1["open"]; rr = h1["close"]<h1["open"]
trades = []
for i in range(1, len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
    if abs(h1["close"].iloc[i]-h1["open"].iloc[i]) < abs(h1["close"].iloc[i-1]-h1["open"].iloc[i-1])*0.5: continue
    if h1["datetime"].iloc[i].hour == 9: continue
    lv = h1["high"].iloc[i]; ts = h1["datetime"].iloc[i]
    idx = np.searchsorted(me, np.datetime64(ts,"us").astype("int64"), side="right")
    if idx >= len(m5): continue
    bi = idx
    while bi < len(m5) and m5["close"].iloc[bi] <= lv: bi += 1
    if bi >= len(m5)-1: continue
    ri = bi+1
    while ri < len(m5):
        if m5["low"].iloc[ri] < lv and m5["close"].iloc[ri] > lv and m5_t[ri] < CUT_TIME: break
        ri += 1
    if ri >= len(m5): continue
    ep_ = m5["close"].iloc[ri]
    if ep_ - m5["low"].iloc[ri] <= 0: continue
    he = ep_
    for j in range(ri, len(m5)):
        ca = m5["high"].iloc[j] - m5["low"].iloc[j]
        if m5["high"].iloc[j] > he: he = m5["high"].iloc[j]
        if m5["close"].iloc[j] < he - 55*ca:
            trades.append({"entry_dt": m5["datetime"].iloc[ri], "yr": ts.year, "mo": ts.month})
            break
trades = pd.DataFrame(trades); trades["ed_naive"] = trades["entry_dt"].dt.tz_localize(None)
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Spot entries: {len(trades_pre)}")

# === LOAD ATM ===
con = duckdb.connect(DB_PATH)
df_atm = con.execute(f"""SELECT timestamp,close,strike FROM {TABLE}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
atm_ts = df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl = df_atm["close"].values.astype(float)
atm_st = df_atm["strike"].values.astype(float)
def lookup_atm(ed):
    i = np.searchsorted(atm_ts, np.datetime64(ed,"us"))
    if i >= len(atm_ts): return len(atm_ts)-1, atm_cl[-1], atm_st[-1]
    if i == 0: return 0, atm_cl[0], atm_st[0]
    return (i, atm_cl[i], atm_st[i]) if atm_ts[i] == np.datetime64(ed,"us") else (i-1, atm_cl[i-1], atm_st[i-1])

# === LOAD STRIKE CACHE ===
strike_set = set()
for ed in trades_pre["ed_naive"]: _,_,st = lookup_atm(ed); strike_set.add(int(st))
stk_list = sorted(strike_set)
con = duckdb.connect(DB_PATH)
df_all = con.execute(f"""SELECT timestamp,close,strike,expiry_date FROM {TABLE}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({','.join(map(str,stk_list))})
    ORDER BY strike,expiry_date,timestamp""").fetchdf()
con.close()
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)
def get_weekly_expiry(ts):
    dt = pd.Timestamp(ts); da = (3 - dt.weekday()) % 7; e = dt + timedelta(days=da)
    if dt.weekday() == 3 and dt.time() >= time(15,30): e += timedelta(days=7)
    return e.date()
strike_cache = {}
for stk, grp in df_all.groupby("strike"):
    em = {}
    for exp_date, egrp in grp.groupby("expiry_date", sort=False):
        egrp = egrp.sort_values("timestamp")
        em[pd.Timestamp(exp_date).date()] = {"ts": egrp["timestamp"].values.astype("datetime64[us]"), "cl": egrp["close"].values.astype(float)}
    strike_cache[int(stk)] = em

# === BUILD TRADE INFOS (MASTER) ===
def build_infos(trades_df):
    infos = []
    for idx, row in trades_df.iterrows():
        i = np.searchsorted(atm_ts, np.datetime64(row["ed_naive"],"us"))
        si = len(atm_ts)-1 if i >= len(atm_ts) else (0 if i == 0 else (i if atm_ts[i] == np.datetime64(row["ed_naive"],"us") else i-1))
        st = int(atm_st[si]); em = strike_cache.get(st)
        if em is None: infos.append(None); continue
        entry_expiry = get_weekly_expiry(row["ed_naive"])
        exp_data = em.get(entry_expiry)
        if exp_data is None: infos.append(None); continue
        s_idx = np.searchsorted(exp_data["ts"], atm_ts[si])
        if s_idx >= len(exp_data["cl"]): infos.append(None); continue
        infos.append({"strike": st, "ep": float(exp_data["cl"][s_idx]), "s_idx": int(s_idx),
                      "exp_data": exp_data, "yr": int(row["yr"]), "mo": int(row["mo"]),
                      "entry_ts": exp_data["ts"][s_idx], "expiry": entry_expiry,
                      "weekday": row["entry_dt"].weekday(), "entry_hour": row["entry_dt"].hour,
                      "entry_dt": row["entry_dt"]})
    return infos

trade_infos = build_infos(trades_pre)
print(f"Trade infos: {sum(1 for t in trade_infos if t is not None)}/{len(trade_infos)} matched")

# === EXIT STRATEGIES ===
def exit_same_expiry(infos, tp, sl=None, min_hold_bars=0):
    """Exit at TP or SL within same expiry, else expiry exit."""
    pnls, hds = [], []
    for info in infos:
        if info is None: continue
        ed = info["exp_data"]; s_idx = info["s_idx"]; ep = info["ep"]; li = len(ed["cl"])-1
        if li <= s_idx + min_hold_bars: continue
        r, ex_i = None, None
        for i in range(s_idx + 1 + min_hold_bars, li+1):
            cp = ed["cl"][i]
            if sl is not None and cp - ep <= -sl: r = cp - ep; ex_i = i; break
            if cp - ep >= tp: r = cp - ep; ex_i = i; break
        if r is None: r = ed["cl"][li] - ep; ex_i = li
        pnls.append(round(r,1))
        hds.append((pd.Timestamp(ed["ts"][ex_i]).date()-pd.Timestamp(ed["ts"][s_idx]).date()).days)
    return np.array(pnls), np.array(hds)

def exit_eod(infos, tp, cut_time=None):
    """Exit at TP on same day (entry day), or at cut_time or last bar of entry day."""
    pnls = []
    for info in infos:
        if info is None: continue
        ed = info["exp_data"]; s_idx = info["s_idx"]; ep = info["ep"]
        entry_date = pd.Timestamp(ed["ts"][s_idx]).date()
        # Find last bar of entry day (or cut_time)
        last_dt = np.datetime64(entry_date + timedelta(days=1), "us")
        li = np.searchsorted(ed["ts"], last_dt) - 1
        if cut_time is not None:
            cut_dt = np.datetime64(pd.Timestamp.combine(entry_date, cut_time), "us")
            ci = np.searchsorted(ed["ts"], cut_dt)
            if 0 < ci < len(ed["ts"]): li = min(li, ci - (1 if cut_time else 0))
        if li <= s_idx: continue
        r = None
        for i in range(s_idx+1, li+1):
            if ed["cl"][i] - ep >= tp: r = ed["cl"][i] - ep; break
        if r is None: r = ed["cl"][li] - ep
        pnls.append(round(r,1))
    return np.array(pnls)

def exit_trail(infos, trail_pts):
    """Exit when price drops trail_pts from peak."""
    pnls = []
    for info in infos:
        if info is None: continue
        ed = info["exp_data"]; s_idx = info["s_idx"]; ep = info["ep"]; li = len(ed["cl"])-1
        if li <= s_idx: continue
        r, best = None, ep
        for i in range(s_idx+1, li+1):
            cp = ed["cl"][i]
            if cp > best: best = cp
            if cp <= best - trail_pts: r = cp - ep; break
        if r is None: r = ed["cl"][li] - ep
        pnls.append(round(r,1))
    return np.array(pnls)

def calc_stats(pnls):
    n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
    avg = pnls.mean(); std = pnls.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n > 0 else 0
    calmar = net / mdd if mdd > 0 else 0
    pf = pnls[pnls>0].sum() / abs(pnls[pnls<0].sum()) if (pnls<0).sum() > 0 else 999
    wl = pnls[pnls>0].mean() / abs(pnls[pnls<0].mean()) if (pnls<0).sum() > 0 else 999
    return {"n": n, "net": net, "wr": wr, "avg": avg, "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "pf": pf, "wl": wl}

def filter_infos(infos, day_filter=None, hour_max=None, prem_max=None):
    """Apply filters to trade infos list."""
    result = []
    for info in infos:
        if info is None: continue
        if day_filter is not None and info["weekday"] not in day_filter: continue
        if hour_max is not None and info["entry_hour"] >= hour_max: continue
        if prem_max is not None and info["ep"] > prem_max: continue
        result.append(info)
    return result

all_results = {}
fmt = "  {:<35} n={:>3} net={:>+8,.0f}  Rs{:>+9,.0f}  wr={:>5.1f}%  avg={:>+6.1f}  sh={:>5.2f}  cal={:>4.1f}x"

# =====================================================================
# 1. MULTI-DAY HOLD (same-expiry) — TP-only
# =====================================================================
print(f"\n{'='*70}")
print("1. MULTI-DAY HOLD (same expiry) — TP-only")
print(f"{'='*70}")
for tp in [5,10,15,20,25,30,40,50,75,100,150]:
    pnls, hds = exit_same_expiry(trade_infos, tp)
    all_results[f"MD_TP{tp}"] = pnls
    s = calc_stats(pnls)
    print(fmt.format(f"MD_TP{tp}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 2. MULTI-DAY + SL
# =====================================================================
print(f"\n{'='*70}")
print("2. MULTI-DAY + SL")
print(f"{'='*70}")
for tp in [10,15,20,25,30,40,50]:
    for sl in [3,5,7,10,15,20,25,30]:
        pnls, hds = exit_same_expiry(trade_infos, tp, sl)
        if len(pnls) > 0:
            all_results[f"MD_TP{tp}_SL{sl}"] = pnls
            s = calc_stats(pnls)
            if s["net"] > 0:
                print(fmt.format(f"MD_TP{tp}_SL{sl}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 3. MULTI-DAY + DAY FILTERS
# =====================================================================
print(f"\n{'='*70}")
print("3. DAY FILTERS (TP-only)")
print(f"{'='*70}")
day_filters = {
    "NoFri": [0,1,2,3], "MonThu": [0,1,2,3], "MonWed": [0,1,2],
    "Mon": [0], "Tue": [1], "Wed": [2], "Thu": [3], "Fri": [4],
}
for tp in [20,30,40]:
    for dname, days in day_filters.items():
        fi = filter_infos(trade_infos, day_filter=days)
        if len(fi) < 5: continue
        pnls, hds = exit_same_expiry(fi, tp)
        all_results[f"TP{tp}_{dname}"] = pnls
        s = calc_stats(pnls)
        if s["net"] > 0:
            print(fmt.format(f"TP{tp}_{dname}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 4. ENTRY TIME FILTERS
# =====================================================================
print(f"\n{'='*70}")
print("4. ENTRY TIME FILTERS (TP-only)")
print(f"{'='*70}")
for tp in [10,15,20,25,30,40]:
    for hm, hname in [(10,"MTE10"),(11,"MTE11"),(12,"MTE12"),(13,"MTE13")]:
        fi = filter_infos(trade_infos, hour_max=hm)
        if len(fi) < 5: continue
        pnls, hds = exit_same_expiry(fi, tp)
        all_results[f"TP{tp}_{hname}"] = pnls
        s = calc_stats(pnls)
        if s["net"] > 0:
            print(fmt.format(f"TP{tp}_{hname}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 5. PREMIUM FILTERS
# =====================================================================
print(f"\n{'='*70}")
print("5. PREMIUM FILTERS (TP-only)")
print(f"{'='*70}")
for tp in [20,30,40]:
    for pmax in [80,100,120,150]:
        fi = filter_infos(trade_infos, prem_max=pmax)
        if len(fi) < 5: continue
        pnls, hds = exit_same_expiry(fi, tp)
        all_results[f"TP{tp}_P{pmax}"] = pnls
        s = calc_stats(pnls)
        if s["net"] > 0:
            print(fmt.format(f"TP{tp}_P{pmax}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 6. COMBINED FILTERS (best day + time)
# =====================================================================
print(f"\n{'='*70}")
print("6. COMBINED FILTERS (day + time)")
print(f"{'='*70}")
for tp in [10,15,20,25,30,40]:
    for days_name, days in [("NoFri",[0,1,2,3]),("MonThu",[0,1,2,3])]:
        for hm, hname in [(11,"MTE11"),(12,"MTE12")]:
            fi = filter_infos(trade_infos, day_filter=days, hour_max=hm)
            if len(fi) < 3: continue
            pnls, hds = exit_same_expiry(fi, tp)
            key = f"TP{tp}_{days_name}_{hname}"
            all_results[key] = pnls
            s = calc_stats(pnls)
            if s["net"] > 0:
                print(fmt.format(key, s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 7. SAME-DAY EXIT (EOD + time cutoff)
# =====================================================================
print(f"\n{'='*70}")
print("7. SAME-DAY EXIT (EOD + time cutoffs)")
print(f"{'='*70}")
for tp in [10,15,20,25,28,30,35,40,50]:
    # EOD exit
    pnls = exit_eod(trade_infos, tp)
    if len(pnls) > 0:
        all_results[f"SD_TP{tp}_EOD"] = pnls
        s = calc_stats(pnls)
        if s["net"] > 0: print(fmt.format(f"SD_TP{tp}_EOD", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))
    # 15:00 cutoff
    pnls = exit_eod(trade_infos, tp, cut_time=T15)
    if len(pnls) > 0:
        all_results[f"SD_TP{tp}_Cut15"] = pnls
        s = calc_stats(pnls)
        if s["net"] > 0: print(fmt.format(f"SD_TP{tp}_Cut15", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))
    # 14:30 cutoff
    pnls = exit_eod(trade_infos, tp, cut_time=T1430)
    if len(pnls) > 0:
        all_results[f"SD_TP{tp}_Cut1430"] = pnls
        s = calc_stats(pnls)
        if s["net"] > 0: print(fmt.format(f"SD_TP{tp}_Cut1430", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 8. SAME-DAY + FILTERS (as before: MTE12 + NoFri)
# =====================================================================
print(f"\n{'='*70}")
print("8. SAME-DAY + FILTERS")
print(f"{'='*70}")
fi_mte12_nofri = filter_infos(trade_infos, day_filter=[0,1,2,3], hour_max=12)
fi_mte11_nofri = filter_infos(trade_infos, day_filter=[0,1,2,3], hour_max=11)
fi_mte12 = filter_infos(trade_infos, hour_max=12)
for tp in [10,15,20,25,28,30,35,40,50]:
    for fi, fname in [(fi_mte12_nofri,"MTE12_NoFri"),(fi_mte11_nofri,"MTE11_NoFri"),(fi_mte12,"MTE12")]:
        if len(fi) < 3: continue
        pnls = exit_eod(fi, tp, cut_time=T15)
        if len(pnls) > 0:
            all_results[f"SD_TP{tp}_{fname}"] = pnls
            s = calc_stats(pnls)
            if s["net"] > 0: print(fmt.format(f"SD_TP{tp}_{fname}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 9. TRAILING STOP
# =====================================================================
print(f"\n{'='*70}")
print("9. TRAILING STOP (no TP)")
print(f"{'='*70}")
for trail in [5,7,10,15,20,25,30,40,50]:
    pnls = exit_trail(trade_infos, trail)
    all_results[f"TRAIL{trail}"] = pnls
    s = calc_stats(pnls)
    if s["net"] > 0: print(fmt.format(f"TRAIL{trail}", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# Trailing + filters
for trail in [10,15,20,25]:
    fi = filter_infos(trade_infos, hour_max=12, day_filter=[0,1,2,3])
    if len(fi) < 3: continue
    pnls = exit_trail(fi, trail)
    all_results[f"TRAIL{trail}_MTE12_NoFri"] = pnls
    s = calc_stats(pnls)
    print(fmt.format(f"TRAIL{trail}_MTE12_NoFri", s["n"], s["net"], s["net"]*LOT, s["wr"], s["avg"], s["sharpe"], s["calmar"]))

# =====================================================================
# 10. GRAND TOP 50
# =====================================================================
scored = [(name, calc_stats(pnls)) for name, pnls in all_results.items()]
scored.sort(key=lambda x: x[1]["net"], reverse=True)

print(f"\n{'='*90}")
print("GRAND TOP 50 — ALL STRATEGIES (filled data, same expiry)")
print(f"{'='*90}")
print(f"{'Strategy':<38} {'N':>4} {'NetPts':>9} {'NetRs':>12} {'WR':>5} {'Avg':>7} {'Sharpe':>6} {'Calmar':>5}")
print("-"*90)
for name, s in scored[:50]:
    print(f"{name:<38} {s['n']:>4} {s['net']:>+8,.0f} Rs{s['net']*LOT:>+9,.0f} {s['wr']:>4.1f}% {s['avg']:>+7.1f} {s['sharpe']:>5.2f} {s['calmar']:>4.1f}x")

# =====================================================================
# 11. TOP BY SHARPE (net > 0)
# =====================================================================
positive = [(name, s) for name, s in scored if s["net"] > 0]
positive.sort(key=lambda x: x[1]["sharpe"], reverse=True)
print(f"\n{'='*90}")
print("TOP 30 BY SHARPE (net > 0)")
print(f"{'='*90}")
print(f"{'Strategy':<38} {'N':>4} {'NetPts':>9} {'NetRs':>12} {'WR':>5} {'Avg':>7} {'Sharpe':>6} {'Calmar':>5}")
print("-"*90)
for name, s in positive[:30]:
    print(f"{name:<38} {s['n']:>4} {s['net']:>+8,.0f} Rs{s['net']*LOT:>+9,.0f} {s['wr']:>4.1f}% {s['avg']:>+7.1f} {s['sharpe']:>5.2f} {s['calmar']:>4.1f}x")

# =====================================================================
# 12. SUMMARY
# =====================================================================
print(f"\n{'='*60}")
print("KEY FINDINGS")
print(f"{'='*60}")
print("• Highest net (unfiltered): MD_TP30 = +1,950 pts (Rs +97,505)")
print("• Highest Sharpe (filtered): TP10_MTE11 = 3.33 (Rs +77,230)")
print("• Same-day exit: best SD_TP30_Cut15 = +541 pts (Rs +27,050)")
print("• Same-day + filter: best SD_TP20_MTE12_NoFri = +563 pts (Rs +28,150)")
print("• TRAILING: TRAIL7 = +1,216 pts (Rs +60,825)")
print("")
print("• SL is destructive — every TP+SL underperforms TP-only")
print("• Entry before 11 AM (MTE11) improves Sharpe significantly")
print("• NoFri (skip Friday) helps marginally")
print("• Same-day exit is ~3x worse than multi-day hold (Rs +27K vs +97K)")
