"""Verify edge cases: Trade 0 + 58 skipped trades + data quality"""
import duckdb, pandas as pd, numpy as np, os, warnings
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
warnings.filterwarnings("ignore")
con = duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")

# Check: does options_data_clean exist and have correct columns?
try:
    cols = con.execute("DESCRIBE options_data_clean").fetchdf()
    print("options_data_clean columns:")
    for _, r in cols.iterrows():
        print(f"  {r['column_name']}: {r['column_type']}")
except Exception as e:
    print(f"options_data_clean: {e}")

# Check Trade 0: 2021-06-14 09:15
print("\n--- Trade 0: 2021-06-14 09:15:00 ---")
df0 = con.execute("""
    SELECT timestamp, close, strike FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
    AND timestamp='2021-06-14 09:15:00' AND atm_distance=0
    ORDER BY timestamp LIMIT 5
""").fetchdf()
print(f"ATM data at 09:15: {len(df0)} rows")
if len(df0)>0:
    for _, r in df0.iterrows():
        print(f"  ts={r['timestamp']} close={r['close']} strike={r['strike']}")

# Check the strike data for Trade 0's strike at 09:15
st = con.execute("""
    SELECT strike FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
    AND timestamp='2021-06-14 09:15:00' AND atm_distance=0
    LIMIT 1
""").fetchone()
if st:
    stk = st[0]
    print(f"\nChecking strike {stk} at 2021-06-14:")
    df_stk = con.execute(f"""
        SELECT timestamp, close FROM options_data_clean
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
        AND strike={stk} AND timestamp::date='2021-06-14'
        ORDER BY timestamp
    """).fetchdf()
    print(f"  Bars: {len(df_stk)}")
    for _, r in df_stk.iterrows():
        print(f"  {r['timestamp']} close={r['close']}")

# Check how many trades have EOD before entry (the 58 skipped)
print("\n--- Analyzing skipped trades ---")
# Load ATM data
df_atm = con.execute("""
    SELECT timestamp, close, strike FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
    ORDER BY timestamp
""").fetchdf()
con.close()
df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"])
atm_ts = df_atm["timestamp"].values.astype("datetime64[us]")
atm_st = df_atm["strike"].values.astype(float)

# Load trades
import pandas as pd
h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in [h1, m5]:
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True)
    d.reset_index(drop=True, inplace=True)

def atr(m5):
    tr = pd.concat([m5["high"]-m5["low"], abs(m5["high"]-m5["close"].shift(1)), abs(m5["low"]-m5["close"].shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=14, min_periods=14, adjust=False).mean()

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
trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)

# Check which trades have no EOD bar
print("\nTrades with no EOD bar (eod_idx <= s_idx):")
no_eod = []
for i in range(len(trades_pre)):
    ed = trades_pre.iloc[i]["ed_naive"]
    idx = np.searchsorted(atm_ts, np.datetime64(ed, "us"))
    if idx < 0 or idx >= len(atm_ts): 
        if idx == 0 or (idx == 1 and atm_ts[0] == np.datetime64(ed, "us")):
            no_eod.append((i, ed, "NO_ATM_IDX"))
        continue
    if idx == 0:
        si = 0
    else:
        si = idx - 1
    st = int(atm_st[si])
    ts_atm = atm_ts[si]
    
    # Load strike data for this trade
    df_sd = con.execute(f"""
        SELECT timestamp, close FROM options_data_clean
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
        AND strike={st} AND timestamp='{pd.Timestamp(ts_atm)}'
        ORDER BY timestamp LIMIT 1
    """).fetchdf()
    
    if len(df_sd) == 0:
        no_eod.append((i, ed, f"NO_STRIKE_BAR_{st}"))
        continue
    
    # Find EOD
    entry_dt = ts_atm.astype("datetime64[D]")
    eod_ns = entry_dt + np.timedelta64(15*60+25, "m")
    
    # Check if EOD bar exists for this strike
    cnt_eod = con.execute(f"""
        SELECT COUNT(*) FROM options_data_clean
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
        AND strike={st} AND timestamp >= '{pd.Timestamp(ts_atm)}' AND timestamp <= '{pd.Timestamp(eod_ns)}'
    """).fetchone()[0]
    
    if cnt_eod <= 1:  # Only entry bar itself
        no_eod.append((i, ed, f"NO_EOD_BARS_{st}_cnt={cnt_eod}"))

print(f"  Trades with no EOD bar: {len(no_eod)}")
for i, ed, reason in no_eod[:15]:
    print(f"  Trade {i}: ed={ed} reason={reason}")

# Check entry time distribution
print("\n--- Entry time distribution ---")
entry_times = [t["ed_naive"].time() for _, t in trades_pre.iterrows()]
time_counts = pd.Series(entry_times).value_counts().sort_index()
print(f"  Entry times: {len(entry_times)} total")
for t, c in time_counts.items():
    print(f"  {t}: {c} trades")
