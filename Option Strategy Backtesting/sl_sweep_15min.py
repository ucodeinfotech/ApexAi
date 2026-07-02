"""15-min: ALL SL x TP x filter combinations. Every single one."""
import duckdb, pandas as pd, numpy as np, warnings, os
from datetime import timedelta, time
from collections import defaultdict
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT, DB_PATH = 50, r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
TABLE = "options_data_filled"

m5_raw = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
m5_raw["datetime"] = pd.to_datetime(m5_raw["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
m5_raw.sort_values("datetime", inplace=True); m5_raw.reset_index(drop=True, inplace=True)

m15 = m5_raw.set_index("datetime")["close"].resample("15min").agg(
    {"open":"first","high":"max","low":"min","close":"last"}
).dropna().reset_index()
m15.columns = ["datetime","open","high","low","close"]

def find_entries(df, entry_cutoff_time=None):
    m5_t = m5_raw["datetime"].dt.time.values
    me = m5_raw["datetime"].astype("int64").values
    trades = []
    for i in range(1, len(df)):
        if not (df["close"].iloc[i-1] < df["open"].iloc[i-1] and df["close"].iloc[i] > df["open"].iloc[i]): continue
        if df["open"].iloc[i] > df["close"].iloc[i-1] or df["close"].iloc[i] < df["open"].iloc[i-1]: continue
        prev_r, cur_r = abs(df["close"].iloc[i-1]-df["open"].iloc[i-1]), abs(df["close"].iloc[i]-df["open"].iloc[i])
        if cur_r < prev_r * 0.5: continue
        dt = df["datetime"].iloc[i]
        if dt.hour == 9: continue
        lv = df["high"].iloc[i]
        idx = np.searchsorted(me, np.datetime64(dt,"us").astype("int64"), side="right")
        if idx >= len(m5_raw): continue
        bi = idx
        while bi < len(m5_raw) and m5_raw["close"].iloc[bi] <= lv: bi += 1
        if bi >= len(m5_raw)-1: continue
        ri = bi + 1
        ctime = entry_cutoff_time if entry_cutoff_time else time(15, 30)
        while ri < len(m5_raw):
            if (m5_raw["low"].iloc[ri] < lv and m5_raw["close"].iloc[ri] > lv and
                m5_raw["datetime"].iloc[ri].time() < ctime): break
            ri += 1
        if ri >= len(m5_raw): continue
        ep = m5_raw["close"].iloc[ri]
        if ep - m5_raw["low"].iloc[ri] <= 0: continue
        he = ep
        for j in range(ri, len(m5_raw)):
            ca = m5_raw["high"].iloc[j] - m5_raw["low"].iloc[j]
            if m5_raw["high"].iloc[j] > he: he = m5_raw["high"].iloc[j]
            if m5_raw["close"].iloc[j] < he - 55*ca:
                trades.append({"entry_dt": m5_raw["datetime"].iloc[ri], "yr": dt.year, "mo": dt.month})
                break
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0: return trades_df
    trades_df["ed_naive"] = trades_df["entry_dt"].dt.tz_localize(None)
    return trades_df[trades_df["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)

entries_15 = find_entries(m15, time(13, 45))
print(f"15-min entries: {len(entries_15)}")

# === OPTION DATA ===
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

all_strikes = set()
for ed in entries_15["ed_naive"]: _,_,st = lookup_atm(ed); all_strikes.add(int(st))

con = duckdb.connect(DB_PATH)
df_all = con.execute(f"""SELECT timestamp,close,strike,expiry_date FROM {TABLE}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({','.join(map(str,sorted(all_strikes)))})
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
                      "weekday": row["entry_dt"].weekday(), "entry_hour": row["entry_dt"].hour})
    return [t for t in infos if t is not None]

infos = build_infos(entries_15)
print(f"Option-matched: {len(infos)}")

# === STRATEGY ENGINE ===
def exit_md(infos, tp, sl=None):
    pnls = []
    for info in infos:
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

def calc_stats(pnls):
    n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100; avg = pnls.mean()
    std = pnls.std() if n > 1 else 1; sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n > 0 else 0
    calmar = net / mdd if mdd > 0 else 0
    pf = pnls[pnls>0].sum() / abs(pnls[pnls<0].sum()) if (pnls<0).sum() > 0 else 999
    return {"n":n,"net":net,"net_rs":net*LOT,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pf":pf}

def filter_fn(infos, day_filter=None, hour_max=None):
    return [i for i in infos if (day_filter is None or i["weekday"] in day_filter) and
            (hour_max is None or i["entry_hour"] < hour_max)]

# =====================================================================
# ALL TP x SL COMBINATIONS — NO FILTER
# =====================================================================
TP_VALS = [5,10,15,20,25,30,35,40,45,50,60,75,100,150,200]
SL_VALS = list(range(1, 101))  # SL 1 through 100

print(f"\n{'='*90}")
print("15-min UNFILTERED: ALL TP x SL (SL 1-100)")
print(f"{'='*90}")
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7} {'PF':>7}")
print("-"*75)

results_all = []
for tp in TP_VALS:
    pnls_tp = exit_md(infos, tp)
    s_tp = calc_stats(pnls_tp)
    results_all.append({"tp": tp, "sl": 0, "name": f"TP{tp}", **s_tp})
    for sl in SL_VALS:
        pnls = exit_md(infos, tp, sl)
        s = calc_stats(pnls)
        results_all.append({"tp": tp, "sl": sl, "name": f"TP{tp} SL{sl}", **s})

# Print top 50
results_all.sort(key=lambda x: x["net"], reverse=True)
for r in results_all[:50]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net_rs']:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x {r['pf']:>6.2f}")

# =====================================================================
# MTE10 x ALL SL
# =====================================================================
fi_mte10 = filter_fn(infos, hour_max=10)
print(f"\n{'='*90}")
print("15-min MTE10: ALL TP x SL (SL 1-100)")
print(f"{'='*90}")
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7} {'PF':>7}")
print("-"*75)

results_mte10 = []
for tp in TP_VALS:
    for sl in [0] + SL_VALS:
        pnls = exit_md(fi_mte10, tp, sl if sl > 0 else None)
        s = calc_stats(pnls)
        results_mte10.append({"tp": tp, "sl": sl, "name": f"TP{tp} SL{sl}" if sl > 0 else f"TP{tp}", **s})

results_mte10.sort(key=lambda x: x["net"], reverse=True)
for r in results_mte10[:30]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net_rs']:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x {r['pf']:>6.2f}")

# =====================================================================
# MTE11 x ALL SL
# =====================================================================
fi_mte11 = filter_fn(infos, hour_max=11)
print(f"\n{'='*90}")
print("15-min MTE11: ALL TP x SL (SL 1-100)")
print(f"{'='*90}")
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7} {'PF':>7}")
print("-"*75)

results_mte11 = []
for tp in TP_VALS:
    for sl in [0] + SL_VALS:
        pnls = exit_md(fi_mte11, tp, sl if sl > 0 else None)
        s = calc_stats(pnls)
        results_mte11.append({"tp": tp, "sl": sl, "name": f"TP{tp} SL{sl}" if sl > 0 else f"TP{tp}", **s})

results_mte11.sort(key=lambda x: x["net"], reverse=True)
for r in results_mte11[:30]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net_rs']:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x {r['pf']:>6.2f}")

# =====================================================================
# MTE12 x ALL SL
# =====================================================================
fi_mte12 = filter_fn(infos, hour_max=12)
print(f"\n{'='*90}")
print("15-min MTE12: ALL TP x SL (SL 1-100)")
print(f"{'='*90}")
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7} {'PF':>7}")
print("-"*75)

results_mte12 = []
for tp in TP_VALS:
    for sl in [0] + SL_VALS:
        pnls = exit_md(fi_mte12, tp, sl if sl > 0 else None)
        s = calc_stats(pnls)
        results_mte12.append({"tp": tp, "sl": sl, "name": f"TP{tp} SL{sl}" if sl > 0 else f"TP{tp}", **s})

results_mte12.sort(key=lambda x: x["net"], reverse=True)
for r in results_mte12[:30]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net_rs']:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x {r['pf']:>6.2f}")

# =====================================================================
# MTE10 + NoFri x ALL SL
# =====================================================================
fi_mte10_nf = filter_fn(infos, hour_max=10, day_filter=[0,1,2,3])
print(f"\n{'='*90}")
print("15-min MTE10+NoFri: ALL TP x SL (SL 1-100)")
print(f"{'='*90}")
print(f"{'TP':>4} {'SL':>4} {'N':>4} {'NetPts':>9} {'NetRs':>11} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7} {'PF':>7}")
print("-"*75)

results_mte10nf = []
for tp in TP_VALS:
    for sl in [0] + SL_VALS:
        pnls = exit_md(fi_mte10_nf, tp, sl if sl > 0 else None)
        s = calc_stats(pnls)
        results_mte10nf.append({"tp": tp, "sl": sl, "name": f"TP{tp} SL{sl}" if sl > 0 else f"TP{tp}", **s})

results_mte10nf.sort(key=lambda x: x["net"], reverse=True)
for r in results_mte10nf[:20]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['n']:>4} {r['net']:>+8,.0f} Rs{r['net_rs']:>+9,.0f} {r['wr']:>4.1f}% {r['avg']:>+7.1f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x {r['pf']:>6.2f}")

# =====================================================================
# SL COMPARISON: TP-only vs best TP+SL per TP value
# =====================================================================
print(f"\n{'='*90}")
print("15-min: TP-ONLY vs BEST TP+SL (per TP value, unfiltered)")
print(f"{'='*90}")
print(f"{'TP':>4} {'TP-only Rs':>15} {'TP+SL Rs':>15} {'Best SL':>8} {'WR':>5} {'Sharpe':>7} {'Diff':>8}")
print("-"*65)
for tp in TP_VALS:
    tp_only = [r for r in results_all if r["tp"] == tp and r["sl"] == 0]
    tp_sl = [r for r in results_all if r["tp"] == tp and r["sl"] > 0]
    if not tp_only: continue
    to = tp_only[0]
    if tp_sl:
        best = max(tp_sl, key=lambda x: x["net"])
        diff = (best["net_rs"] - to["net_rs"]) / to["net_rs"] * 100 if to["net_rs"] != 0 else 0
        print(f"{tp:>4} Rs{to['net_rs']:>+9,.0f} Rs{best['net_rs']:>+9,.0f} SL={best['sl']:>3}  {best['wr']:>4.1f}% {best['sharpe']:>6.2f} {diff:>+7.0f}%")
    else:
        print(f"{tp:>4} Rs{to['net_rs']:>+9,.0f} {'N/A':>15} {'':>8}")

# =====================================================================
# WHERE SL BEATS TP-ONLY (15-min)
# =====================================================================
print(f"\n{'='*90}")
print("WHERE SL BEATS TP-ONLY (15-min, unfiltered)")
print(f"{'='*90}")
count = 0
for r in results_all:
    if r["sl"] == 0: continue
    tp_only = next((x for x in results_all if x["tp"] == r["tp"] and x["sl"] == 0), None)
    if tp_only and r["net"] > tp_only["net"]:
        diff = (r["net_rs"] - tp_only["net_rs"]) / tp_only["net_rs"] * 100 if tp_only["net_rs"] > 0 else 0
        print(f"  TP{r['tp']:>3} SL{r['sl']:>3}: Rs{r['net_rs']:>+9,.0f} vs TP-only Rs{tp_only['net_rs']:>+9,.0f} ({diff:+.0f}%)")
        count += 1
if count == 0:
    print("  NONE — every single SL combination underperforms TP-only")
else:
    print(f"  Total: {count} cases where SL helps")

# =====================================================================
# OPTIMAL SL PER TP (unfiltered)
# =====================================================================
print(f"\n{'='*90}")
print("15-min: OPTIMAL SL PER TP (unfiltered)")
print(f"{'='*90}")
print(f"{'TP':>4} {'OptSL':>7} {'N':>4} {'NetRs':>12} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>7}")
print("-"*60)
for tp in TP_VALS:
    candidates = [r for r in results_all if r["tp"] == tp]
    best = max(candidates, key=lambda x: x["net"])
    print(f"{tp:>4} SL={best['sl']:>3} {best['n']:>4} Rs{best['net_rs']:>+9,.0f} {best['wr']:>4.1f}% {best['avg']:>+7.1f} {best['sharpe']:>6.2f} {best['calmar']:>6.1f}x")

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*90}")
print("15-min SL SWEEP SUMMARY")
print(f"{'='*90}")
unf_net = [r for r in results_all if r["sl"] == 0]
unf_sl = [r for r in results_all if r["sl"] > 0]
sl_beats = sum(1 for r in results_all if r["sl"] > 0 and
               r["net"] > next((x["net"] for x in results_all if x["tp"] == r["tp"] and x["sl"] == 0), -999999))
print(f"  Total combos tested: {len(results_all)} ({len(TP_VALS)} TP x 100 SL + {len(TP_VALS)} TP-only)")
print(f"  Profitable TP-only: {sum(1 for r in unf_net if r['net'] > 0)}/{len(unf_net)}")
print(f"  Profitable TP+SL: {sum(1 for r in unf_sl if r['net'] > 0)}/{len(unf_sl)}")
print(f"  Cases where SL helps: {sl_beats} / {len(unf_sl)}")
if sl_beats > 0:
    winners = [r for r in results_all if r["sl"] > 0 and
               r["net"] > next((x["net"] for x in results_all if x["tp"] == r["tp"] and x["sl"] == 0), -999999)]
    for w in winners[:10]:
        tp_only_n = next((x["net_rs"] for x in results_all if x["tp"] == w["tp"] and x["sl"] == 0), 0)
        print(f"    TP{w['tp']:>3} SL{w['sl']:>3}: Rs{w['net_rs']:>+9,.0f} (TP-only: Rs{tp_only_n:>+9,.0f})")
print(f"\n  Best TP-only: {max(unf_net, key=lambda x: x['net'])['name']} = Rs{max(unf_net, key=lambda x: x['net'])['net_rs']:+,.0f}")
print(f"  Best TP+SL: {max(unf_sl, key=lambda x: x['net'])['name']} = Rs{max(unf_sl, key=lambda x: x['net'])['net_rs']:+,.0f}")
print(f"  Best MTE10 TP-only: {max(results_mte10, key=lambda x: x['net'])['name']} = Rs{max(results_mte10, key=lambda x: x['net'])['net_rs']:+,.0f}")
print(f"  Best MTE10+SL: {max((r for r in results_mte10 if r['sl'] > 0), key=lambda x: x['net'])['name']} = Rs{max((r for r in results_mte10 if r['sl'] > 0), key=lambda x: x['net'])['net_rs']:+,.0f}")

print("\nDone.")
