"""Check if multi-day hold spans actual option expiry dates."""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# Load expiry calendar: when does each expiry_code date actually expire?
# Use expiry_availability: for a given expiry_code, the LAST timestamp = expiry?
con = duckdb.connect(DB_PATH)
expiry_cal = con.execute("""
    SELECT expiry_code, expiry_flag, 
           min(timestamp) as first_bar, max(timestamp) as last_bar,
           CAST(datediff('day', min(timestamp), max(timestamp)) AS INT) as span_days
    FROM expiry_availability
    WHERE expiry_flag='WEEK'
    GROUP BY expiry_code, expiry_flag
    ORDER BY expiry_code
    LIMIT 20
""").fetchdf()
print("=== EXPIRY CALENDAR ===")
print(expiry_cal.to_string(index=False))

# Check: for strike data in clean table, how many distinct weekly cycles does one data point span?
print("\n=== CHECK PRICE DISCONTINUITY ACROSS EXPIRY ===")
# Pick strike 22200 (773 days span), check price gaps
df = con.execute("""
    SELECT timestamp, close, strike, expiry_code, expiry_flag, atm_distance
    FROM options_data_clean
    WHERE option_type='CALL' AND strike=22200 AND expiry_code=1 AND expiry_flag='WEEK'
    ORDER BY timestamp
    LIMIT 50
""").fetchdf()
print(df.to_string(index=False))

# Check what dates the strike data has — is it continuous or gappy?
print("\n=== STRIKE 22200 DATE COVERAGE ===")
df2 = con.execute("""
    SELECT date(timestamp) as trade_date, count(*) as bars, min(close) as min_c, max(close) as max_c
    FROM options_data_clean
    WHERE option_type='CALL' AND strike=22200 AND expiry_code=1 AND expiry_flag='WEEK'
    GROUP BY date(timestamp)
    ORDER BY trade_date
    LIMIT 30
""").fetchdf()
print(df2.to_string(index=False))

# Key: what's the last date of weekly option data for expiry_code=1?
print("\n=== LAST DATA DATE PER STRIKE (expiry_code=1, WEEK) ===")
df3 = con.execute("""
    SELECT strike, max(date(timestamp)) as last_date, count(*) as total_bars
    FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
    GROUP BY strike
    ORDER BY last_date DESC
    LIMIT 15
""").fetchdf()
print(df3.to_string(index=False))

# Spot check: trade #1 from the backtest, trace actual hold period and option expiry
print("\n=== TRADE HOLD PERIOD SPOT CHECK ===")
# Load spot entries
h1 = pd.read_csv("NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
m5 = pd.read_csv("NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
for d in [h1, m5]:
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime", inplace=True); d.reset_index(drop=True, inplace=True)
me = m5["datetime"].astype("int64").values

# Replicate spot entry
b = (h1["close"] - h1["open"]).abs(); g = h1["close"] > h1["open"]; rr = h1["close"] < h1["open"]
entries = []
for i in range(1, len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
    if b.iloc[i] < b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour == 9: continue
    lv = h1["high"].iloc[i]; ts = h1["datetime"].iloc[i]
    idx = np.searchsorted(me, np.datetime64(ts, "us").astype("int64"), side="right")
    if idx >= len(m5): continue
    bi = idx
    while bi < len(m5) and m5["close"].iloc[bi] <= lv: bi += 1
    if bi >= len(m5)-1: continue
    entries.append({"entry_dt": m5["datetime"].iloc[bi+1], "yr": ts.year, "mo": ts.month})

# First 5 entries: show their dates and check if any are near weekly expiry (Thursday)
trades = pd.DataFrame(entries)
trades = trades[trades["entry_dt"] >= pd.Timestamp("2021-06-14", tz="Asia/Kolkata")].head(10)
print("\nFirst 10 trade entries:")
for i, row in trades.iterrows():
    print(f"  #{i+1}: {row['entry_dt'].strftime('%Y-%m-%d %H:%M')}  weekday={row['entry_dt'].weekday()} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][row['entry_dt'].weekday()]})")

con.close()
