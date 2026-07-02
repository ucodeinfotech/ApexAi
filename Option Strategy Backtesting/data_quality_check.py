"""
Comprehensive option data quality check + rupee verification
"""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
print("="*70)
print("OPTION DATA QUALITY CHECK")
print("="*70)

con=duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")

# 1. Basic stats
print("\n1. TABLE STATS")
for tbl in ["options_data", "options_data_dedup"]:
    cnt=con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {cnt:,} rows")

# 2. CALL WEEK expiry_code=1 stats
print("\n2. CALL WEEK exp_code=1 (dedup)")
cnt=con.execute("SELECT COUNT(*) FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'").fetchone()[0]
print(f"  Total rows: {cnt:,}")

# 3. Check for nulls
print("\n3. NULL CHECK")
for col in ["timestamp","close","strike","spot","open","high","low"]:
    nulls=con.execute(f"SELECT COUNT(*) FROM options_data_dedup WHERE {col} IS NULL").fetchone()[0]
    if nulls>0: print(f"  WARNING: {col} has {nulls} nulls")

# 4. Duplicate check (exact rows)
print("\n4. DUPLICATE CHECK (exact rows)")
cnt_all=con.execute("SELECT COUNT(*) FROM options_data_dedup").fetchone()[0]
cnt_distinct=con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT * FROM options_data_dedup)").fetchone()[0]
print(f"  Total: {cnt_all:,} | Distinct: {cnt_distinct:,} | Dupes: {cnt_all-cnt_distinct:,}")

# 5. Check for duplicate (timestamp, strike, option_type, expiry)
print("\n5. CHECK: duplicate (timestamp, strike, option_type, expiry)")
cnt_distinct2=con.execute("""SELECT COUNT(*) FROM (
  SELECT DISTINCT timestamp, strike, option_type, expiry_code, expiry_flag FROM options_data_dedup
)""").fetchone()[0]
print(f"  Distinct (ts, strike, opt, exp): {cnt_distinct2:,}")
if cnt_all>cnt_distinct2:
    print(f"  --> {cnt_all-cnt_distinct2:,} rows have same (ts,strike,opt,exp) with different prices!")
    # Show sample
    print("  Sample duplicates:")
    dup_samples=con.execute("""SELECT timestamp, strike, option_type, expiry_flag, 
    COUNT(*) as cnt, MIN(close) as min_c, MAX(close) as max_c, MIN(open) as min_o, MAX(open) as max_o
    FROM options_data_dedup 
    GROUP BY timestamp, strike, option_type, expiry_code, expiry_flag
    HAVING COUNT(*)>1 
    LIMIT 5""").df()
    print(dup_samples.to_string())

# 6. Check atm_distance=0 data quality
print("\n6. ATM_DISTANCE=0 DATA QUALITY")
atm_stats=con.execute("""SELECT 
  COUNT(*) as rows,
  COUNT(DISTINCT strike) as strikes,
  MIN(timestamp) as first_ts, MAX(timestamp) as last_ts,
  MIN(close) as min_close, MAX(close) as max_close,
  AVG(close) as avg_close
FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0""").df()
print(atm_stats.to_string())

# 7. Spot price vs strike check for atm_distance=0
print("\n7. SPOT vs STRIKE check for atm_distance=0 (sample)")
spot_strike=con.execute("""SELECT timestamp, close, strike, spot 
FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
  AND timestamp IN (
    SELECT timestamp FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 
    AND expiry_flag='WEEK' AND atm_distance=0 
    USING SAMPLE 5
  )
ORDER BY timestamp""").df()
for _,r in spot_strike.iterrows():
    diff=abs(r["strike"]-r["spot"])
    print(f"  {r['timestamp']} strike={r['strike']:.0f} spot={r['spot']:.1f} diff={diff:.1f} (should be ~0 for ATM)")

# 8. Check for extreme close values
print("\n8. EXTREME CLOSE VALUES")
extreme=con.execute("""SELECT COUNT(*) as cnt, MIN(close) as min_c, 
  PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY close) as p01,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY close) as p99,
  MAX(close) as max_c
FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0""").df()
print(extreme.to_string())

# Check how many have close < 0.5
low_close=con.execute("""SELECT COUNT(*) FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
AND close < 0.5""").fetchone()[0]
print(f"  Close < 0.5: {low_close:,} rows")
high_close=con.execute("""SELECT COUNT(*) FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
AND close > 500""").fetchone()[0]
print(f"  Close > 500: {high_close:,} rows")

# 9. Verify timestamp alignment: spot 5-min data vs option 5-min data
print("\n9. TIMESTAMP ALIGNMENT CHECK")
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv")
spot_ts=pd.to_datetime(m5["datetime"],utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
opt_ts=con.execute("""SELECT DISTINCT timestamp FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
ORDER BY timestamp""").fetchdf()
opt_ts["timestamp"]=pd.to_datetime(opt_ts["timestamp"],utc=False)
# Compare
spot_ts_set=set(spot_ts.values)
opt_ts_set=set(opt_ts["timestamp"].values)
common=spot_ts_set & opt_ts_set
print(f"  Spot 5-min timestamps: {len(spot_ts_set):,}")
print(f"  Option 5-min timestamps: {len(opt_ts_set):,}")
print(f"  Common timestamps: {len(common):,}")

# Check if option timestamps are subset of spot timestamps
only_spot=spot_ts_set-opt_ts_set
only_opt=opt_ts_set-spot_ts_set
print(f"  Spot-only timestamps: {len(only_spot):,}")
print(f"  Option-only timestamps: {len(only_opt):,}")
if len(only_opt)>0:
    print(f"  Sample option-only: {sorted(list(only_opt))[:5]}")

# 10. Check per-strike data density
print("\n10. PER-STRIKE DATA DENSITY (sample)")
strikes_sample=con.execute("""SELECT DISTINCT strike FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
ORDER BY strike LIMIT 5""").fetchdf()
for s in strikes_sample["strike"]:
    cnt=con.execute(f"SELECT COUNT(*) FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike={s}").fetchone()[0]
    min_ts=con.execute(f"SELECT MIN(timestamp) FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike={s}").fetchone()[0]
    max_ts=con.execute(f"SELECT MAX(timestamp) FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike={s}").fetchone()[0]
    print(f"  Strike {s:.0f}: {cnt:,} rows, {str(min_ts)[:16]} to {str(max_ts)[:16]}")

# 11. Overall data range
print("\n11. DATA RANGE")
dr=con.execute("""SELECT MIN(timestamp) as first, MAX(timestamp) as last,
  COUNT(DISTINCT DATE(timestamp)) as trading_days
FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'""").df()
print(dr.to_string())

con.close()
print("\nDone!")
