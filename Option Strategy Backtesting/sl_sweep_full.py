"""COMPREHENSIVE SL SWEEP — every combination of TP x SL x filter."""
import duckdb, pandas as pd, numpy as np, warnings, os
from datetime import timedelta, time
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT, DB_PATH = 50, r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
TABLE = "options_data_filled"

# === SPOT ENTRY ===
h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in (h1, m5):
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True); d.reset_index(drop=True, inplace=True)
me = m5["datetime"].astype("int64").values; m5_t = m5["datetime"].dt.time.values
rr = h1["close"]<h1["open"]; g = h1["close"]>h1["open"]
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
        if m5["low"].iloc[ri] < lv and m5["close"].iloc[ri] > lv and m5_t[ri] < pd.Timestamp("14:15").time(): break
        ri += 1
    if ri >= len(m5): continue
    ep_ = m5["close"].iloc[ri]
    if ep_ - m5["low"].iloc[ri] <= 0: continue
    he = ep_
    for j in range(ri, len(m5)):
        if m5["high"].iloc[j] > he: he = m5["high"].iloc[j]
        ca = m5["high"].iloc[j] - m5["low"].iloc[j]
        if m5["close"].iloc[j] < he - 55*ca:
            trades.append({"entry_dt": m5["datetime"].iloc[ri], "yr": ts.year, "mo": ts.month})
            break
trades = pd.DataFrame(trades); trades["ed_naive"] = trades["entry_dt"].dt.tz_localize(None)
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)

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

# === STRIKE CACHE ===
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

# === BUILD INFOS ===
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

def exit_same_expiry(infos, tp, sl=None):
    pnls = []
    for info in infos:
        if info is None: continue
        ed = info["exp_data"]; s_idx = info["s_idx"]; ep = info["ep"]; li = len(ed["cl"])-1
        if li <= s_idx: continue
        r = None
        for i in range(s_idx+1, li+1):
            cp = ed["cl"][i]
            if sl is not None and cp - ep <= -sl: r = cp - ep; break
            if cp - ep >= tp: r = cp - ep; break
        if r is None: r = ed["cl"][li] - ep
        pnls.append(round(r,1))
    return np.array(pnls)

def filter_infos(infos, day_filter=None, hour_max=None):
    result = []
    for info in infos:
        if info is None: continue
        if day_filter is not None and info["weekday"] not in day_filter: continue
        if hour_max is not None and info["entry_hour"] >= hour_max: continue
        result.append(info)
    return result

def calc_stats(pnls):
    n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
    avg = pnls.mean(); std = pnls.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n > 0 else 0
    calmar = net / mdd if mdd > 0 else 0
    return {"n": n, "net": net, "wr": wr, "avg": avg, "sharpe": sharpe, "mdd": mdd, "calmar": calmar}

# =====================================================================
# GRANULAR SL SWEEP — TP x SL x FILTER
# =====================================================================
TP_VALS = [5,10,15,20,25,30,35,40,45,50,60,75,100]
SL_VALS = list(range(1, 51))  # SL 1 through 50

# Unfiltered
print(f"\n{'='*80}")
print("UNFILTERED — ALL TP x SL (SL 1-50)")
print(f"{'='*80}")
results = []
for tp in TP_VALS:
    for sl in SL_VALS:
        pnls = exit_same_expiry(trade_infos, tp, sl)
        s = calc_stats(pnls)
        results.append({"tp": tp, "sl": sl, **s})

top_by_net = sorted(results, key=lambda x: x["net"], reverse=True)[:30]
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*70)
for r in top_by_net:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net']*LOT:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x")

# MTE10
fi_mte10 = filter_infos(trade_infos, hour_max=10)
print(f"\n{'='*80}")
print("MTE10 FILTER — ALL TP x SL")
print(f"{'='*80}")
results_mte10 = []
for tp in TP_VALS:
    for sl in SL_VALS:
        pnls = exit_same_expiry(fi_mte10, tp, sl)
        s = calc_stats(pnls)
        results_mte10.append({"tp": tp, "sl": sl, **s})

top_mte10 = sorted(results_mte10, key=lambda x: x["net"], reverse=True)[:20]
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*70)
for r in top_mte10:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net']*LOT:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x")

# MTE11
fi_mte11 = filter_infos(trade_infos, hour_max=11)
print(f"\n{'='*80}")
print("MTE11 FILTER — ALL TP x SL")
print(f"{'='*80}")
results_mte11 = []
for tp in TP_VALS:
    for sl in SL_VALS:
        pnls = exit_same_expiry(fi_mte11, tp, sl)
        s = calc_stats(pnls)
        results_mte11.append({"tp": tp, "sl": sl, **s})

top_mte11 = sorted(results_mte11, key=lambda x: x["net"], reverse=True)[:20]
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*70)
for r in top_mte11:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net']*LOT:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x")

# NoFri (skip Friday)
fi_nofri = filter_infos(trade_infos, day_filter=[0,1,2,3])
print(f"\n{'='*80}")
print("NoFri FILTER — ALL TP x SL")
print(f"{'='*80}")
results_nofri = []
for tp in TP_VALS:
    for sl in SL_VALS:
        pnls = exit_same_expiry(fi_nofri, tp, sl)
        s = calc_stats(pnls)
        results_nofri.append({"tp": tp, "sl": sl, **s})

top_nofri = sorted(results_nofri, key=lambda x: x["net"], reverse=True)[:20]
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*70)
for r in top_nofri:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net']*LOT:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x")

# MTE10 + NoFri (combined)
fi_mte10_nofri = filter_infos(trade_infos, day_filter=[0,1,2,3], hour_max=10)
print(f"\n{'='*80}")
print("MTE10 + NoFri — ALL TP x SL")
print(f"{'='*80}")
results_mte10nf = []
for tp in TP_VALS:
    for sl in SL_VALS:
        pnls = exit_same_expiry(fi_mte10_nofri, tp, sl)
        s = calc_stats(pnls)
        results_mte10nf.append({"tp": tp, "sl": sl, **s})

top_combo = sorted(results_mte10nf, key=lambda x: x["net"], reverse=True)[:20]
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*70)
for r in top_combo:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net']*LOT:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x")

# =====================================================================
# BEST TP FOR EACH SL (unfiltered) — find optimal pairing
# =====================================================================
print(f"\n{'='*80}")
print("BEST TP PER SL VALUE (unfiltered)")
print(f"{'='*80}")
print(f"{'SL':>4} {'BestTP':>7} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*70)
for sl in SL_VALS:
    subset = [r for r in results if r["sl"] == sl]
    best = max(subset, key=lambda r: r["net"])
    print(f"{sl:>4} TP={best['tp']:>3} {best['n']:>4} {best['net']:>+8,.0f} Rs{best['net']*LOT:>+9,.0f} {best['wr']:>4.1f}% {best['avg']:>+7.1f} {best['sharpe']:>6.2f} {best['calmar']:>6.1f}x")

# =====================================================================
# COMPARISON: TP-only vs best TP+SL
# =====================================================================
print(f"\n{'='*80}")
print("TP-ONLY vs BEST TP+SL comparison")
print(f"{'='*80}")
top_tp_only = [(tp, exit_same_expiry(trade_infos, tp), calc_stats(exit_same_expiry(trade_infos, tp))) for tp in TP_VALS]
print(f"{'TP-only':>10} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Sharpe':>6}")
print("-"*55)
for tp, p, s in top_tp_only:
    if s["net"] > 0:
        print(f"TP={tp:<3}       {s['n']:>4} {s['net']:>+8,.0f} Rs{s['net']*LOT:>+9,.0f} {s['wr']:>4.1f}% {s['sharpe']:>5.2f}")

print(f"\n{'='*80}")
print("BEST TP+SL per TP value (unfiltered)")
print(f"{'='*80}")
print(f"{'TP+SL':>10} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Sharpe':>6}")
print("-"*55)
for tp in TP_VALS:
    subset = [r for r in results if r["tp"] == tp]
    best = max(subset, key=lambda r: r["net"])
    print(f"TP={tp} SL={best['sl']:<2}  {best['n']:>4} {best['net']:>+8,.0f} Rs{best['net']*LOT:>+9,.0f} {best['wr']:>4.1f}% {best['sharpe']:>5.2f}")

# =====================================================================
# WHERE SL BEATS TP-ONLY — the rare cases
# =====================================================================
print(f"\n{'='*80}")
print("CASES WHERE SL BEATS TP-ONLY")
print(f"{'='*80}")
print(f"{'Strategy':>14} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Sharpe':>6}")
print("-"*55)
for tp in TP_VALS:
    pnls_tp = exit_same_expiry(trade_infos, tp)
    s_tp = calc_stats(pnls_tp)
    subset = [r for r in results if r["tp"] == tp]
    best_sl = max(subset, key=lambda r: r["net"])
    if best_sl["net"] > s_tp["net"]:
        print(f"TP={tp} SL={best_sl['sl']:<2}  {best_sl['n']:>4} {best_sl['net']:>+8,.0f} Rs{best_sl['net']*LOT:>+9,.0f} {best_sl['wr']:>4.1f}% {best_sl['sharpe']:>5.2f}  (TP-only: Rs{s_tp['net']*LOT:>+9,.0f})")
