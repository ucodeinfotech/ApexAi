import duckdb, pandas as pd
con = duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")

# Check EOD timestamps
df = con.execute("""
    SELECT DISTINCT timestamp FROM options_data_clean 
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
    AND timestamp::time >= '15:15:00'
    ORDER BY timestamp LIMIT 20
""").fetchdf()
print("Bars >= 15:15:")
for t in df["timestamp"]:
    print(f"  {t}")

# Most common timestamps
df2 = con.execute("""
    SELECT timestamp::time as t, COUNT(*) as n 
    FROM options_data_clean 
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
    GROUP BY t ORDER BY n DESC LIMIT 10
""").fetchdf()
print("\nMost common times:")
for _, r in df2.iterrows():
    print(f"  {r['t']}: {r['n']:,}")

# Last bar of typical day - check a specific strike on one day
df3 = con.execute("""
    SELECT timestamp FROM options_data_clean 
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
    AND timestamp::date='2025-03-11' AND strike=23800
    ORDER BY timestamp DESC LIMIT 5
""").fetchdf()
print("\n2025-03-11 strike 23800 last bars:")
for t in df3["timestamp"]:
    print(f"  {t}")

# Check if 15:20, 15:25, 15:30 exist
for h in ['15:20', '15:25', '15:30']:
    cnt = con.execute(f"SELECT COUNT(*) FROM options_data_clean WHERE timestamp::time='{h}'").fetchone()[0]
    print(f"  Bars at {h}: {cnt:,}")

con.close()
