"""All timeframes sweep — 1hr, 30min, 15min entry + all strategies + comparison."""
import duckdb, pandas as pd, numpy as np, warnings, os, glob
from datetime import timedelta, time
from collections import defaultdict
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT, DB_PATH = 50, r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
TABLE = "options_data_filled"

m5_raw = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
m5_raw["datetime"] = pd.to_datetime(m5_raw["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
m5_raw.sort_values("datetime", inplace=True); m5_raw.reset_index(drop=True, inplace=True)

def resample_ohlc(df, freq):
    """Resample 5-min to arbitrary frequency. Returns OHLC DataFrame."""
    df = df.set_index("datetime").copy()
    ohlc = df["close"].resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    ohlc.columns.values[0] = "datetime"
    return ohlc

# Create 15-min and 30-min data
m15 = resample_ohlc(m5_raw, "15min")
m30 = resample_ohlc(m5_raw, "30min")
h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
h1["datetime"] = pd.to_datetime(h1["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
h1.sort_values("datetime", inplace=True); h1.reset_index(drop=True, inplace=True)

# === GENERALIZED ENTRY SIGNAL ===
def find_entries(df, freq_name, entry_cutoff_time=None, skip_first_n=1):
    """
    Entry pattern:
    - Previous bar: red (close < open)
    - Current bar: green (close > open)
    - Open within previous close range (current open > prev close OR current close < prev open -> skip)
    - Current range >= 50% of previous range
    - Skip first N bars of day (avoid 9:XX manipulation)
    - Entry: price breaks above prev bar high at 5-min level
    - Re-entry: price dips below prev high then closes above it
    - Cutoff: entry_cutoff_time (or None = end of day)
    """
    m5_t = m5_raw["datetime"].dt.time.values
    me = m5_raw["datetime"].astype("int64").values
    trades = []
    red = df["close"] < df["open"]
    green = df["close"] > df["open"]
    for i in range(1, len(df)):
        if not (red.iloc[i-1] and green.iloc[i]): continue
        # Gap check
        if df["open"].iloc[i] > df["close"].iloc[i-1] or df["close"].iloc[i] < df["open"].iloc[i-1]: continue
        # Min size check
        prev_range = abs(df["close"].iloc[i-1] - df["open"].iloc[i-1])
        cur_range = abs(df["close"].iloc[i] - df["open"].iloc[i])
        if cur_range < prev_range * 0.5: continue
        # Skip first N bars of day
        dt = df["datetime"].iloc[i]
        if dt.hour == 9 and dt.minute < (skip_first_n * 5 * (60 / (df.index[-1] / len(df)))): continue
        # Simplified: if entry hour is 9 and we're in first hours
        if dt.hour == 9: continue

        lv = df["high"].iloc[i]  # level to break: previous bar high
        # Find breakout in 5-min data
        idx = np.searchsorted(me, np.datetime64(dt,"us").astype("int64"), side="right")
        if idx >= len(m5_raw): continue
        # Scan forward for breakout above lv
        bi = idx
        while bi < len(m5_raw) and m5_raw["close"].iloc[bi] <= lv: bi += 1
        if bi >= len(m5_raw)-1: continue
        # Re-entry: dip below lv then close above
        ri = bi + 1
        ctime = entry_cutoff_time if entry_cutoff_time else time(15, 30)
        while ri < len(m5_raw):
            if (m5_raw["low"].iloc[ri] < lv and m5_raw["close"].iloc[ri] > lv and
                m5_raw["datetime"].iloc[ri].time() < ctime): break
            ri += 1
        if ri >= len(m5_raw): continue
        entry_price = m5_raw["close"].iloc[ri]
        if entry_price - m5_raw["low"].iloc[ri] <= 0: continue

        # Trailing exit: 55x candle range
        high_since_entry = entry_price
        for j in range(ri, len(m5_raw)):
            ca = m5_raw["high"].iloc[j] - m5_raw["low"].iloc[j]
            if m5_raw["high"].iloc[j] > high_since_entry:
                high_since_entry = m5_raw["high"].iloc[j]
            if m5_raw["close"].iloc[j] < high_since_entry - 55 * ca:
                trades.append({"entry_dt": m5_raw["datetime"].iloc[ri], "yr": dt.year, "mo": dt.month})
                break
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0: return trades_df
    trades_df["ed_naive"] = trades_df["entry_dt"].dt.tz_localize(None)
    return trades_df[trades_df["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)

# Entry cutoff times per timeframe (for re-entry)
TIMEFRAME_CONFIG = {
    "1hr":   {"df": h1,  "cutoff": time(14, 15), "skip": 1},
    "30min": {"df": m30, "cutoff": time(14, 00), "skip": 2},
    "15min": {"df": m15, "cutoff": time(13, 45), "skip": 4},
}

# Generate entries for all timeframes
entries = {}
for tf, cfg in TIMEFRAME_CONFIG.items():
    print(f"Finding entries for {tf}...")
    edf = find_entries(cfg["df"], tf, cfg["cutoff"], cfg["skip"])
    entries[tf] = edf
    print(f"  {len(edf)} entries found")

# === LOAD OPTION DATA ===
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

# Load all needed strikes
all_strikes = set()
for tf_name, edf in entries.items():
    for ed in edf["ed_naive"]: _,_,st = lookup_atm(ed); all_strikes.add(int(st))

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
    return infos

trade_infos_all = {}
for tf_name, edf in entries.items():
    infos = build_infos(edf)
    matched = sum(1 for t in infos if t is not None)
    print(f"{tf_name}: {matched}/{len(infos)} option-matched")
    trade_infos_all[tf_name] = [t for t in infos if t is not None]

# === STRATEGY FUNCTIONS ===
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

def exit_eod(infos, tp, cut_time=None):
    pnls = []
    for info in infos:
        ed = info["exp_data"]; s_idx = info["s_idx"]; ep = info["ep"]
        entry_date = pd.Timestamp(ed["ts"][s_idx]).date()
        last_dt = np.datetime64(entry_date + timedelta(days=1), "us")
        li = np.searchsorted(ed["ts"], last_dt) - 1
        if cut_time is not None:
            cut_dt = np.datetime64(pd.Timestamp.combine(entry_date, cut_time), "us")
            ci = np.searchsorted(ed["ts"], cut_dt)
            if 0 < ci < len(ed["ts"]): li = min(li, ci - 1)
        if li <= s_idx: continue
        r = None
        for i in range(s_idx+1, li+1):
            if ed["cl"][i] - ep >= tp: r = ed["cl"][i] - ep; break
        if r is None: r = ed["cl"][li] - ep
        pnls.append(round(r,1))
    return np.array(pnls)

def exit_trail(infos, trail_pts):
    pnls = []
    for info in infos:
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
    n = len(pnls); net = pnls.sum(); wr = (pnls>0).mean()*100; avg = pnls.mean()
    std = pnls.std() if n > 1 else 1; sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    cum = np.cumsum(pnls); mdd = (np.maximum.accumulate(cum)-cum).max() if n > 0 else 0
    calmar = net / mdd if mdd > 0 else 0
    pf = pnls[pnls>0].sum() / abs(pnls[pnls<0].sum()) if (pnls<0).sum() > 0 else 999
    return {"n":n,"net":net,"net_rs":net*LOT,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pf":pf}

def filter_fn(infos, day_filter=None, hour_max=None):
    result = []
    for info in infos:
        if day_filter is not None and info["weekday"] not in day_filter: continue
        if hour_max is not None and info["entry_hour"] >= hour_max: continue
        result.append(info)
    return result

# =====================================================================
# SWEEP ALL TIMEFRAMES
# =====================================================================
TP_VALS = [5,10,15,20,25,30,35,40,45,50,60,75,100]
SL_VALS = list(range(1, 51))
T15 = time(15, 0)
HOUR_MAX_OPTS = [(10,"MTE10"),(11,"MTE11"),(12,"MTE12")]

all_results = defaultdict(list)  # tf_name -> [(strategy_name, pnls)]

for tf_name, infos in trade_infos_all.items():
    print(f"\n{'='*60}")
    print(f"TIMEFRAME: {tf_name} ({len(infos)} trades)")
    print(f"{'='*60}")
    if len(infos) < 10:
        print("  Too few trades, skipping")
        continue

    # 1. TP-only MD
    for tp in TP_VALS:
        pnls = exit_md(infos, tp)
        all_results[tf_name].append((f"MD TP{tp}", pnls))

    # 2. Best TP+SL
    for tp in TP_VALS:
        best_s, best_n = None, -999999
        for sl in SL_VALS:
            pnls = exit_md(infos, tp, sl)
            n = pnls.sum()
            if n > best_n: best_n = n; best_s = (sl, pnls)
        if best_s is not None and best_n > -999999:
            all_results[tf_name].append((f"MD TP{tp} SL{best_s[0]}", best_s[1]))

    # 3. MTE10, MTE11, MTE12
    for hm, hname in HOUR_MAX_OPTS:
        fi = filter_fn(infos, hour_max=hm)
        if len(fi) < 5: continue
        for tp in [10,20,25,30,40]:
            pnls = exit_md(fi, tp)
            if len(pnls) > 0:
                all_results[tf_name].append((f"{hname} TP{tp}", pnls))
            # MTE + best SL
            best_s, best_n = None, -999999
            for sl in SL_VALS:
                pnls2 = exit_md(fi, tp, sl)
                n = pnls2.sum()
                if n > best_n: best_n = n; best_s = (sl, pnls2)
            if best_s is not None and best_n > -999999:
                all_results[tf_name].append((f"{hname} TP{tp} SL{best_s[0]}", best_s[1]))

    # 4. Same-day exit
    for tp in [10,15,20,25,28,30,35,40,50]:
        for cut_name, cut in [("EOD", None), ("Cut15", T15)]:
            pnls = exit_eod(infos, tp, cut)
            if len(pnls) > 0:
                all_results[tf_name].append((f"SD TP{tp} {cut_name}", pnls))

    # 5. Same-day + MTE12 + NoFri
    fi_sd = filter_fn(infos, hour_max=12, day_filter=[0,1,2,3])
    for tp in [20,25,28,30,35,40]:
        pnls = exit_eod(fi_sd, tp, T15)
        if len(pnls) > 0:
            all_results[tf_name].append((f"SD TP{tp} MTE12+NoFri", pnls))

    # 6. Trailing
    for trail in [10,15,20,25,30]:
        pnls = exit_trail(infos, trail)
        all_results[tf_name].append((f"Trail {trail}", pnls))

    # 7. Day filters
    for days_name, days in [("NoFri",[0,1,2,3]),("MonWed",[0,1,2])]:
        fi = filter_fn(infos, day_filter=days)
        for tp in [20,30,40]:
            pnls = exit_md(fi, tp)
            if len(pnls) > 0:
                all_results[tf_name].append((f"TP{tp} {days_name}", pnls))

# =====================================================================
# BUILD COMPARISON TABLE
# =====================================================================
print(f"\n\n{'='*100}")
print("TIMEFRAME COMPARISON — BEST STRATEGIES PER CATEGORY")
print(f"{'='*100}")

categories = {
    "TP-only (MD)": lambda n, s: "MD TP" in n and "SL" not in n and "SD" not in n and "MTE" not in n,
    "TP+SL (MD)": lambda n, s: "SL" in n and "MD TP" in n and "MTE" not in n and "SD" not in n,
    "MTE10": lambda n, s: "MTE10" in n and "SL" not in n,
    "MTE10+SL": lambda n, s: "MTE10" in n and "SL" in n,
    "Same-Day EOD": lambda n, s: n.startswith("SD") and "EOD" in n and "MTE" not in n,
    "SD+Filter": lambda n, s: n.startswith("SD") and "MTE12" in n,
    "Trailing": lambda n, s: "Trail" in n,
    "DayFilter (NoFri)": lambda n, s: "NoFri" in n and "SD" not in n,
}

# Build comparison data
comp_data = {}  # (tf, cat) -> (name, stats)
for tf_name in trade_infos_all:
    results = all_results[tf_name]
    stats_map = {}
    for name, pnls in results:
        stats_map[name] = calc_stats(pnls)

    for cat_name, filter_fn in categories.items():
        candidates = [(n,s) for n,s in stats_map.items() if filter_fn(n,s)]
        if candidates:
            best = max(candidates, key=lambda x: x[1]["net"])
            comp_data[(tf_name, cat_name)] = best

# Print comparison table
print(f"{'Category':<20} {'1hr':>25} {'30min':>25} {'15min':>25}")
print("-"*95)
for cat_name in categories:
    row = [cat_name[:18]]
    for tf_name in ["1hr", "30min", "15min"]:
        d = comp_data.get((tf_name, cat_name))
        if d:
            name, s = d
            row.append(f"{name[:12]} Rs{s['net_rs']:+,.0f} Sh{s['sharpe']:.1f}")
        else:
            row.append(f"{'N/A':>25}")
    print(f"{row[0]:<20} {row[1]:>25} {row[2]:>25} {row[3]:>25}")

# =====================================================================
# OVERALL TOP 20 STRATEGIES ACROSS ALL TIMEFRAMES
# =====================================================================
print(f"\n\n{'='*100}")
print("TOP 30 STRATEGIES ACROSS ALL TIMEFRAMES")
print(f"{'='*100}")
all_ranked = []
for tf_name in trade_infos_all:
    for name, pnls in all_results[tf_name]:
        s = calc_stats(pnls)
        all_ranked.append((tf_name, name, s))
all_ranked.sort(key=lambda x: x[2]["net"], reverse=True)

print(f"{'TF':<6} {'Strategy':<25} {'N':>4} {'NetPts':>9} {'NetRs':>12} {'WR':>5} {'Sharpe':>7} {'Calmar':>6}")
print("-"*80)
for tf_name, name, s in all_ranked[:30]:
    print(f"{tf_name:<6} {name:<25} {s['n']:>4} {s['net']:>+8,.0f} Rs{s['net_rs']:>+9,.0f} {s['wr']:>4.1f}% {s['sharpe']:>6.2f} {s['calmar']:>5.1f}x")

# =====================================================================
# BEST PER TIMEFRAME
# =====================================================================
print(f"\n\n{'='*100}")
print("BEST STRATEGY PER TIMEFRAME")
print(f"{'='*100}")
for tf_name in ["1hr", "30min", "15min"]:
    if tf_name not in trade_infos_all: continue
    tf_best = [(n, calc_stats(p)) for n,p in all_results[tf_name]]
    tf_best.sort(key=lambda x: x[1]["net"], reverse=True)
    print(f"\n{tf_name.upper()} — Top 5:")
    print(f"  {'Strategy':<30} {'N':>4} {'NetRs':>12} {'WR':>5} {'Avg':>7} {'Sharpe':>7} {'Calmar':>6}")
    for name, s in tf_best[:5]:
        print(f"  {name:<30} {s['n']:>4} Rs{s['net_rs']:>+9,.0f} {s['wr']:>4.1f}% {s['avg']:>+7.1f} {s['sharpe']:>6.2f} {s['calmar']:>5.1f}x")

# =====================================================================
# COMPARISON: N
# =====================================================================
print(f"\n\n{'='*100}")
print("TRADE COUNT COMPARISON ACROSS TIMEFRAMES (TP30)")
print(f"{'='*100}")
print(f"{'Metric':<30} {'1hr':>15} {'30min':>15} {'15min':>15}")
print("-"*75)
for tf_name, name, pnls in all_ranked:
    pass
for metric in ["n", "net_rs", "wr", "sharpe", "calmar", "pf"]:
    row = [metric]
    for tf_name in ["1hr", "30min", "15min"]:
        tp30_key = f"MD TP30"
        found = [(n, calc_stats(p)) for n,p in all_results.get(tf_name, []) if n == tp30_key]
        if found:
            s = found[0][1]
            if metric == "wr": row.append(f"{s['wr']:.1f}%".rjust(15))
            elif metric == "net_rs": row.append(f"Rs{s['net_rs']:+,.0f}".rjust(15))
            elif metric == "sharpe": row.append(f"{s['sharpe']:.2f}".rjust(15))
            elif metric == "calmar": row.append(f"{s['calmar']:.1f}x".rjust(15))
            elif metric == "pf": row.append(f"{s['pf']:.2f}".rjust(15))
            else: row.append(str(s[metric]).rjust(15))
        else:
            row.append("N/A".rjust(15))
    print(f"{'TP30 '+metric:<30} {' '.join(row[1:])}")

# N count comparison
print(f"\n\n{'='*100}")
print("SUMMARY: TRADE COUNTS BY TIMEFRAME")
print(f"{'='*100}")
for tf_name in ["1hr", "30min", "15min"]:
    if tf_name in trade_infos_all:
        n = len(trade_infos_all[tf_name])
        print(f"  {tf_name:<10}: {n} trades")

# =====================================================================
# VERIFY RESULTS ARE NOT CROSS-EXPIRY CONTAMINATED
# =====================================================================
print(f"\n\n{'='*100}")
print("EXPIRY INTEGRITY CHECK")
print(f"{'='*100}")
for tf_name, infos in trade_infos_all.items():
    expiry_counts = defaultdict(int)
    for info in infos:
        expiry_counts[info["expiry"]] += 1
    max_per_expiry = max(expiry_counts.values())
    print(f"  {tf_name:<10}: {len(expiry_counts)} unique expiry dates, max {max_per_expiry} trades/expiry")

print(f"\n{'='*100}")
print("COMPARISON TABLE (CSV-ready)")
print(f"{'='*100}")
print("TF,Strategy,N,NetPts,NetRs,WR%,Avg,Sharpe,Calmar,Pf")
for tf_name, name, s in all_ranked[:50]:
    print(f"{tf_name},{name},{s['n']},{s['net']:+.0f},{s['net_rs']:+.0f},{s['wr']:.1f},{s['avg']:+.1f},{s['sharpe']:.2f},{s['calmar']:.1f},{s['pf']:.2f}")

print("\nDone.")
