"""Check how many trades span across weekly expiry dates."""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT, DB_PATH = 50, r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# SPOT
h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in [h1, m5]:
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True); d.reset_index(drop=True, inplace=True)
me = m5["datetime"].astype("int64").values
CUT = pd.Timestamp("14:15").time()

trades = []
b = (h1["close"] - h1["open"]).abs(); g = h1["close"] > h1["open"]; rr = h1["close"] < h1["open"]
for i in range(1, len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
    if b.iloc[i] < b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour == 9: continue
    lv = h1["high"].iloc[i]; ts = h1["datetime"].iloc[i]
    idx = np.searchsorted(me, np.datetime64(ts, "us").astype("int64"), side="right")
    if idx >= len(m5["close"]): continue
    bi = idx
    while bi < len(m5["close"]) and m5["close"].iloc[bi] <= lv: bi += 1
    if bi >= len(m5["close"])-1: continue
    ri = bi+1
    while ri < len(m5["close"]):
        if m5["low"].iloc[ri] < lv and m5["close"].iloc[ri] > lv and pd.Series(m5["datetime"]).dt.time.iloc[ri] < CUT: break
        ri += 1
    if ri >= len(m5["close"]): continue
    ed = m5["datetime"].iloc[ri]; ep_ = m5["close"].iloc[ri]
    if ep_ - m5["low"].iloc[ri] <= 0: continue
    he = ep_
    for j in range(ri, len(m5["close"])):
        ca = m5["high"].iloc[j] - m5["low"].iloc[j]
        if m5["high"].iloc[j] > he: he = m5["high"].iloc[j]
        if m5["close"].iloc[j] < he - 55*ca:
            trades.append({"entry_dt": m5["datetime"].iloc[ri], "yr": ts.year, "mo": ts.month})
            break
trades = pd.DataFrame(trades)
trades["ed_naive"] = trades["entry_dt"].dt.tz_localize(None)
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)

# OPTION
con = duckdb.connect(str(DB_PATH))
df_atm = con.execute("""SELECT timestamp,close,strike FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
atm_ts = df_atm["timestamp"].values.astype("datetime64[us]")
atm_st = df_atm["strike"].values.astype(float)
atm_cl = df_atm["close"].values.astype(float)

def lookup_atm(ed):
    ts64 = np.datetime64(ed, "us"); i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts): return len(atm_ts)-1, atm_cl[-1], atm_st[-1]
    if i == 0: return 0, atm_cl[0], atm_st[0]
    return (i, atm_cl[i], atm_st[i]) if atm_ts[i] == ts64 else (i-1, atm_cl[i-1], atm_st[i-1])

# Get expiry calendar from DuckDB
expiry_map = {}
con2 = duckdb.connect(str(DB_PATH))
expiry_rows = con2.execute("""
    SELECT DISTINCT expiry_code, date(timestamp) as expiry_date
    FROM expiry_availability
    WHERE expiry_flag='WEEK' AND expiry_code=1
    ORDER BY expiry_date
""").fetchdf()
con2.close()
# Get unique dates when expiry_code=1 data shows up = first day of each weekly cycle
cycle_dates = sorted(expiry_rows["expiry_date"].unique())
print(f"Total weekly cycles in data: {len(cycle_dates)}")
print(f"Sample cycle dates: {cycle_dates[:5]} ... {cycle_dates[-5:]}")

# Build a function to find which cycle a date belongs to
def get_cycle_entry_date(dt):
    """Find the date of the weekly cycle that contains this trade date."""
    dt_date = pd.Timestamp(dt).date()
    for i, cd in enumerate(cycle_dates):
        if cd > dt_date:
            # Previous cycle started at cycle_dates[i-1]
            # Typically cycle runs ~5 trading days
            # If dt is within 7 days of cycle start, it's in that cycle
            prev_cd = cycle_dates[i-1]
            if (dt_date - prev_cd).days <= 7:
                return prev_cd
            return None
    return None

# Simulate to get actual hold durations
TP, MAXD = 150, 14
strike_set = set()
for ed in trades_pre["ed_naive"]: _,_,st = lookup_atm(ed); strike_set.add(int(st))
stk_list = sorted(strike_set)
con2 = duckdb.connect(str(DB_PATH))
df_all = con2.execute(f"""SELECT timestamp,close,strike FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({','.join(map(str,stk_list))})
    ORDER BY strike,timestamp""").fetchdf()
con2.close()
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)
strike_cache = {}
for stk, grp in df_all.groupby("strike"):
    strike_cache[int(stk)] = {"ts": grp["timestamp"].values.astype("datetime64[us]"), "cl": grp["close"].values.astype(float)}

trade_infos = []
for idx, row in trades_pre.iterrows():
    ts64 = np.datetime64(row["ed_naive"], "us"); i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts): si = len(atm_ts)-1
    elif i == 0: si = 0
    else: si = i if atm_ts[i] == ts64 else i-1
    st = int(atm_st[si]); sd = strike_cache.get(st)
    if sd is None: trade_infos.append(None); continue
    s_idx = np.searchsorted(sd["ts"], atm_ts[si])
    if s_idx >= len(sd["cl"]): trade_infos.append(None); continue
    trade_infos.append({"strike": st, "ep": float(sd["cl"][s_idx]), "s_idx": int(s_idx), "stk_data": sd,
                        "yr": int(row["yr"]), "mo": int(row["mo"]),
                        "entry_date": sd["ts"][s_idx]})

hold_days_list = []
all_pnls = []
for info in trade_infos:
    if info is None: continue
    sd = info["stk_data"]; s_idx = info["s_idx"]; ep = info["ep"]
    end_ns = sd["ts"][s_idx] + np.timedelta64(int(MAXD * 86400 * 1e6), "us")
    result = None; exit_idx = None
    for i in range(s_idx+1, min(s_idx+3000, len(sd["cl"]))):
        if sd["ts"][i] > end_ns: result = sd["cl"][i] - ep; exit_idx = i; break
        if sd["cl"][i] - ep >= TP: result = sd["cl"][i] - ep; exit_idx = i; break
    if result is None: continue
    pnl = round(result, 1)
    all_pnls.append(pnl)
    hold_days = (pd.Timestamp(sd["ts"][exit_idx]) - pd.Timestamp(sd["ts"][s_idx])).days
    hold_days_list.append(hold_days)

pnls = np.array(all_pnls)
hold_days = np.array(hold_days_list)

print(f"\n{'='*60}")
print(f"HOLD DAYS DISTRIBUTION (MD_TP150_D14)")
print(f"{'='*60}")
print(f"Total trades: {len(hold_days)}")
print(f"Min hold: {hold_days.min()} day(s)")
print(f"Max hold: {hold_days.max()} day(s)")
print(f"Median hold: {np.median(hold_days):.0f} day(s)")
print(f"Mean hold: {hold_days.mean():.1f} day(s)")
print(f"\nHold days distribution (bucketed):")
buckets = [(0,1),(2,3),(4,5),(6,7),(8,10),(11,14),(15,21),(22,30),(31,60),(61,100),(101,999)]
for lo, hi in buckets:
    cnt = ((hold_days >= lo) & (hold_days <= hi)).sum()
    if cnt > 0:
        rng = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"  {rng:>5} days: {cnt:>4} trades ({cnt/len(hold_days)*100:>5.1f}%)")
print(f"\nDetailed:")
for d in sorted(set(hold_days)):
    cnt = (hold_days == d).sum()
    if cnt > 0:
        print(f"  {d:>3}d: {cnt} trades")

print(f"\n{'='*60}")
print(f"EXPIRY GAP: trades with hold >= 6 days")
print(f"{'='*60}")
cross_expiry = hold_days >= 6
print(f"Hold >= 6 days (cross expiry gap?): {cross_expiry.sum()} trades ({cross_expiry.mean()*100:.1f}%)")
print(f"Hold >= 7 days (definitely cross): {(hold_days>=7).sum()} trades ({(hold_days>=7).mean()*100:.1f}%)")

# Split: trades that hit TP before expiry vs those that hold across
tp_hit_before = hold_days <= 5
if tp_hit_before.sum() > 0:
    print(f"\n{'='*60}")
    print("TRADES WITH HOLD <= 5 days (safe, no expiry cross)")
    print(f"{'='*60}")
    safe_pnls = pnls[tp_hit_before]
    print(f"Trades: {len(safe_pnls)}")
    print(f"Net PnL: {safe_pnls.sum():>+,.0f} pts (Rs {safe_pnls.sum()*LOT:>+,.0f})")
    print(f"WR: {(safe_pnls>0).mean()*100:.1f}%")
    print(f"Avg: {safe_pnls.mean():+.1f} pts")

print(f"\n{'='*60}")
print("TRADES WITH HOLD > 5 days (may cross expiry)")
print(f"{'='*60}")
cross_pnls = pnls[~tp_hit_before]
if len(cross_pnls) > 0:
    print(f"Trades: {len(cross_pnls)}")
    print(f"Net PnL: {cross_pnls.sum():>+,.0f} pts (Rs {cross_pnls.sum()*LOT:>+,.0f})")
    print(f"WR: {(cross_pnls>0).mean()*100:.1f}%")
    print(f"Avg: {cross_pnls.mean():+.1f} pts")
    print(f"\nSuspicious: trades held >5 days that are profitable")
    print("(likely includes price jumps from NEW weekly cycle)")
    print(f"Profitable cross-expiry trades: {(cross_pnls>0).sum()}")
    print(f"Avg profit of those: {cross_pnls[cross_pnls>0].mean():+.1f} pts")
