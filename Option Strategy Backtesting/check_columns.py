import duckdb, pandas as pd
con = duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")
try:
    cols = con.execute("DESCRIBE options_data_clean").fetchdf()
    print("options_data_clean columns:")
    for _, r in cols.iterrows():
        print(f"  {r['column_name']}: {r['column_type']}")
except Exception as e:
    print(f"options_data_clean: {e}")

df = con.execute("""SELECT timestamp, close, strike FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
ORDER BY timestamp LIMIT 5""").fetchdf()
print(f"\nFirst 5 ATM bars:\n{df}")

# Check how many bars per date for ATM strike
df2 = con.execute("""SELECT timestamp::date as d, COUNT(*) as n
FROM options_data_clean WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
GROUP BY d ORDER BY d""").fetchdf()
print(f"\nATM bars per day: min={df2['n'].min()} max={df2['n'].max()} avg={df2['n'].mean():.0f}")
print(f"Days with < 10 bars: {(df2['n']<10).sum()}")

# Check a random strike at a random date to see how many EOD bars exist
df3 = con.execute("""SELECT timestamp FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
AND strike=15900 AND timestamp::date='2021-07-06'
ORDER BY timestamp""").fetchdf()
print(f"\nStrike 15900 on 2021-07-06: {len(df3)} bars")
if len(df3) > 0:
    print(f"  First: {df3['timestamp'].iloc[0]}, Last: {df3['timestamp'].iloc[-1]}")

# Count all distinct (date, strike) combos that lack 15:25 bar
c = con.execute("""SELECT COUNT(*) as n FROM (
  SELECT strike, timestamp::date as d, MAX(timestamp::time) as last_time
  FROM options_data_clean WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
  GROUP BY strike, d
  HAVING MAX(timestamp::time) < '15:25:00'
)""").fetchone()[0]
print(f"\n(strike, date) combos with last bar before 15:25: {c}")

c2 = con.execute("SELECT COUNT(DISTINCT strike || '_' || timestamp::date) FROM options_data_clean WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'").fetchone()[0]
print(f"Total (strike, date) combos: {c2}")
print(f"Percentage lacking 15:25: {c/c2*100:.1f}%")

con.close()
