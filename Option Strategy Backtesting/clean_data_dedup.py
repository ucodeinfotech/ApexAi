"""
Fix remaining data duplicates + rupee verification
"""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

DB=r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# Check if there's already a deeper dedup
con=duckdb.connect(DB)
print("=== Current state ===")
print(f"options_data_dedup: {con.execute('SELECT COUNT(*) FROM options_data_dedup').fetchone()[0]:,} rows")

# Count distinct (timestamp, strike, option_type, expiry_code, expiry_flag)
distinct_keys=con.execute("""SELECT COUNT(*) FROM (
  SELECT DISTINCT timestamp, strike, option_type, expiry_code, expiry_flag FROM options_data_dedup
)""").fetchone()[0]
print(f"Distinct (ts,strike,type,expiry) keys: {distinct_keys:,}")

# Rows where same key has different prices
dup_price_rows=con.execute("""SELECT SUM(cnt) FROM (
  SELECT COUNT(*) as cnt FROM options_data_dedup
  GROUP BY timestamp, strike, option_type, expiry_code, expiry_flag
  HAVING COUNT(*)>1
)""").fetchone()[0]
print(f"Rows with duplicate keys (different prices): {dup_price_rows:,}")

# Check CALL WEEK data specifically
ck=con.execute("""SELECT COUNT(*) FROM (
  SELECT COUNT(*) FROM options_data_dedup
  WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
  GROUP BY timestamp, strike
  HAVING COUNT(*)>1
)""").fetchone()[0]
print(f"CALL WEEK rows with duplicate (ts,strike): {ck}")

# Sample duplicates in CALL WEEK
print("\nSample CALL WEEK duplicates:")
sam=con.execute("""SELECT timestamp, strike, close, open, high, low
FROM options_data_dedup
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
AND (timestamp, strike) IN (
  SELECT timestamp, strike FROM options_data_dedup
  WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
  GROUP BY timestamp, strike HAVING COUNT(*)>1
  LIMIT 3
)
ORDER BY timestamp, strike""").df()
for _,r in sam.iterrows():
    print(f"  {r['timestamp']} strike={r['strike']:.0f} C={r['close']:.1f} O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f}")

# Check if CALL WEEK dupes affect our trades
print("\n=== Check: do CALL WEEK dupes affect ATM entry timestamps? ===")
dup_atm=con.execute("""SELECT COUNT(*) FROM options_data_dedup
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
AND (timestamp, strike) IN (
  SELECT timestamp, strike FROM options_data_dedup
  WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
  GROUP BY timestamp, strike HAVING COUNT(*)>1
)""").fetchone()[0]
print(f"ATM CALL WEEK rows affected: {dup_atm}")

# Clean: create a properly deduped table taking MIN close
print("\n=== Creating clean dedup (MIN close per key) ===")
con.execute("""CREATE OR REPLACE TABLE options_data_clean AS
SELECT timestamp, option_type, expiry_code, expiry_flag, strike, atm_distance,
  MIN(close) as close,
  MIN(open) as open,
  MAX(high) as high,
  MIN(low) as low,
  MAX(volume) as volume,
  MAX(oi) as oi,
  AVG(iv) as iv,
  AVG(spot) as spot,
  MIN(strike_label) as strike_label,
  MIN(security_id) as security_id,
  MIN(interval) as interval,
  MIN(rn) as rn
FROM options_data_dedup
GROUP BY timestamp, option_type, expiry_code, expiry_flag, strike, atm_distance""")
clean_cnt=con.execute("SELECT COUNT(*) FROM options_data_clean").fetchone()[0]
print(f"Cleaned table: {clean_cnt:,} rows")
print(f"Reduction: {distinct_keys:,} (was {distinct_keys:,} distinct keys expected)")

# Verify ATM distance=0 data looks correct
print("\nVerifying ATM distance=0 in clean table:")
atm_check=con.execute("""SELECT COUNT(*) as rows, MIN(close) as min_c, MAX(close) as max_c, AVG(close) as avg_c
FROM options_data_clean WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0""").df()
print(atm_check.to_string())

con.close()

print("\nDone! options_data_clean created with deduped data.")
