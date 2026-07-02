"""Full strategy sweep on filled data: TP + SL, same expiry."""
import duckdb, pandas as pd, numpy as np, warnings, os
from datetime import timedelta, time
from pathlib import Path
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT, DB_PATH = 50, r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
TABLE = "options_data_filled"
CUT_TIME = pd.Timestamp("14:15").time()

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

# === LOAD ATM DATA (entry strike lookup) ===
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

# === LOAD PER-STRIKE CACHE ===
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
    dt = pd.Timestamp(ts)
    days_ahead = (3 - dt.weekday()) % 7
    expiry = dt + timedelta(days=days_ahead)
    if dt.weekday() == 3 and dt.time() >= time(15, 30): expiry += timedelta(days=7)
    return expiry.date()

strike_cache = {}
for stk, grp in df_all.groupby("strike"):
    em = {}
    for exp_date, egrp in grp.groupby("expiry_date", sort=False):
        egrp = egrp.sort_values("timestamp")
        em[pd.Timestamp(exp_date).date()] = {"ts": egrp["timestamp"].values.astype("datetime64[us]"), "cl": egrp["close"].values.astype(float)}
    strike_cache[int(stk)] = em

# === BUILD TRADE INFO CACHE ===
trade_infos = []
for idx, row in trades_pre.iterrows():
    i = np.searchsorted(atm_ts, np.datetime64(row["ed_naive"],"us"))
    si = len(atm_ts)-1 if i >= len(atm_ts) else (0 if i == 0 else (i if atm_ts[i] == np.datetime64(row["ed_naive"],"us") else i-1))
    st = int(atm_st[si]); em = strike_cache.get(st)
    if em is None: trade_infos.append(None); continue
    entry_expiry = get_weekly_expiry(row["ed_naive"])
    exp_data = em.get(entry_expiry)
    if exp_data is None: trade_infos.append(None); continue
    s_idx = np.searchsorted(exp_data["ts"], atm_ts[si])
    if s_idx >= len(exp_data["cl"]): trade_infos.append(None); continue
    trade_infos.append({"strike": st, "ep": float(exp_data["cl"][s_idx]), "s_idx": int(s_idx),
                        "exp_data": exp_data, "yr": int(row["yr"]), "mo": int(row["mo"]),
                        "entry_ts": exp_data["ts"][s_idx], "expiry": entry_expiry,
                        "weekday": row["entry_dt"].weekday(), "entry_hour": row["entry_dt"].hour})

matched = sum(1 for t in trade_infos if t is not None)
print(f"Trade infos: {matched}/{len(trade_infos)} matched")

# === RUN STRATEGY ===
def run_strategy(trade_infos, tp, sl=None, trail=None, min_hold_bars=0):
    """Run strategy. sl in pts (optional). trail in pts (hit TP at high - trail from entry max)."""
    pnls, hds, tps, sls = [], [], [], []
    for info in trade_infos:
        if info is None: continue
        ed = info["exp_data"]; s_idx = info["s_idx"]; ep = info["ep"]
        last_idx = len(ed["cl"]) - 1
        if last_idx <= s_idx + min_hold_bars: continue
        result, ex_i = None, None
        best = ep  # for trailing
        
        for i in range(s_idx + 1 + min_hold_bars, last_idx+1):
            cp = ed["cl"][i]
            if trail is not None:
                if cp > best: best = cp
                if cp <= best - trail:
                    result = cp - ep; ex_i = i; sls.append(1); tps.append(0); break
            if sl is not None and cp - ep <= -sl:
                result = cp - ep; ex_i = i; sls.append(1); tps.append(0); break
            if cp - ep >= tp:
                result = cp - ep; ex_i = i; tps.append(1); sls.append(0); break
        
        if result is None:
            # Expiry exit or continue
            if tp > 0:  # TP strategy, exit at expiry
                result = ed["cl"][last_idx] - ep; ex_i = last_idx
                tps.append(0); sls.append(0)
            else:
                continue  # SL-only strategy; skip if not hit
        
        pnls.append(round(result,1))
        hds.append((pd.Timestamp(ed["ts"][ex_i]).date() - pd.Timestamp(ed["ts"][s_idx]).date()).days)
    return np.array(pnls) if pnls else np.array([0])

# === SWEEP ===
results = []
tp_vals = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150]
sl_vals = [None, 3, 5, 7, 10, 15, 20, 25, 30]
trail_vals = [None, 10, 15, 20, 25, 30, 50]

total_combos = len(tp_vals) * len(sl_vals) + len(tp_vals) + len(trail_vals)
cnt = 0
print(f"\nSweeping {total_combos} combinations...\n")

# TP only
for tp in tp_vals:
    pnls = run_strategy(trade_infos, tp)
    if len(pnls) > 0:
        n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
        avg = pnls.mean(); std = pnls.std() if n>1 else 1
        sharpe = avg/std*np.sqrt(252) if std>0 else 0
        cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n>0 else 0
        results.append(("TP", tp, None, None, n, net, wr, avg, sharpe, mdd, net/mdd if mdd>0 else 0))
    cnt += 1

# TP + SL
for tp in tp_vals:
    for sl in sl_vals:
        if sl is None: continue
        pnls = run_strategy(trade_infos, tp, sl=sl)
        if len(pnls) > 0:
            n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
            avg = pnls.mean(); std = pnls.std() if n>1 else 1
            sharpe = avg/std*np.sqrt(252) if std>0 else 0
            cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n>0 else 0
            results.append(("TP+SL", tp, sl, None, n, net, wr, avg, sharpe, mdd, net/mdd if mdd>0 else 0))
        cnt += 1

# Trailing
for trail in trail_vals:
    if trail is None: continue
    pnls = run_strategy(trade_infos, tp=999, trail=trail)  # trail only, no TP
    if len(pnls) > 0:
        n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100
        avg = pnls.mean(); std = pnls.std() if n>1 else 1
        sharpe = avg/std*np.sqrt(252) if std>0 else 0
        cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n>0 else 0
        results.append(("TRAIL", None, None, trail, n, net, wr, avg, sharpe, mdd, net/mdd if mdd>0 else 0))
    cnt += 1

# === RESULTS TABLE ===
cols = ["Type", "TP", "SL", "Trail", "N", "NetPts", "NetRs", "WR%", "Avg", "Sharpe", "MDD", "Calmar"]
rows = []
for r in results:
    tp_l = f"TP{r[1]}" if r[1] else ""
    sl_l = f"_SL{r[2]}" if r[2] else ""
    tr_l = f"_Tr{r[3]}" if r[3] else ""
    name = f"{tp_l}{sl_l}{tr_l}" if r[0]=="TP+SL" else (f"Tr{r[3]}" if r[0]=="TRAIL" else f"TP{r[1]}")
    rows.append([name, r[1] if r[1] else "-", r[2] if r[2] else "-", r[3] if r[3] else "-",
                 r[4], f"{r[5]:+,.0f}", f"Rs{r[5]*LOT:+,.0f}", f"{r[6]:.1f}%",
                 f"{r[7]:+.1f}", f"{r[8]:.2f}", f"{r[9]:,.0f}", f"{r[10]:.1f}x"])

dfr = pd.DataFrame(rows, columns=cols)

# Convert NetPts to numeric for sorting
dfr["NetNum"] = dfr["NetPts"].apply(lambda x: float(x.replace(",","")))

# TOP BY NET
print(f"\n{'='*70}")
print(f"TOP 30 BY NET PnL")
print(f"{'='*70}")
top_net = dfr.sort_values("NetNum", ascending=False).head(30)
for _, r in top_net.iterrows():
    print(f"  {r['Type']:<25} n={r['N']:>3} net={r['NetPts']:>8} {r['NetRs']:>11} wr={r['WR%']:>6} avg={r['Avg']:>6} sh={r['Sharpe']:5} mdd={r['MDD']:>6} cal={r['Calmar']:5}")

# TOP BY SHARPE (positive only)
print(f"\n{'='*70}")
print(f"TOP 20 BY SHARPE  (net > 0)")
print(f"{'='*70}")
top_sh = dfr.sort_values("Sharpe", ascending=False)
top_sh = top_sh[top_sh["NetNum"] > 0].head(20)
for _, r in top_sh.iterrows():
    print(f"  {r['Type']:<25} n={r['N']:>3} net={r['NetPts']:>8} {r['NetRs']:>11} wr={r['WR%']:>6} avg={r['Avg']:>6} sh={r['Sharpe']:5}")

# TOP BY WR (net > 0)
print(f"\n{'='*70}")
print(f"TOP 15 BY WIN RATE  (net > 0)")
print(f"{'='*70}")
top_wr = dfr.sort_values("WR%", ascending=False)
top_wr = top_wr[top_wr["NetNum"] > 0].head(15)
for _, r in top_wr.iterrows():
    print(f"  {r['Type']:<25} n={r['N']:>3} net={r['NetPts']:>8} {r['NetRs']:>11} wr={r['WR%']:>6} avg={r['Avg']:>6}")

print(f"\n{'='*70}")
print(f"BEST NET PnL STRATEGY:")
print(f"{'='*70}")
best = top_net.iloc[0]
print(f"  {best['Type']}: {best['NetPts']} {best['NetRs']} | WR {best['WR%']} | Sharpe {best['Sharpe']} | {best['N']} trades")
