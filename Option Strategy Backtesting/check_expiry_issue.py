"""Check if weekly expiry data spans correctly for multi-day holds."""
import duckdb, pandas as pd, numpy as np
DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
con = duckdb.connect(DB_PATH)

# 1. How expiry_code relates to actual expiry dates
print("=== EXPIRY AVAILABILITY (sample) ===")
df = con.execute("""
    SELECT timestamp, expiry_code, expiry_flag 
    FROM expiry_availability 
    WHERE expiry_flag='WEEK' 
    ORDER BY timestamp LIMIT 20
""").fetchdf()
df["ts_date"] = df["timestamp"].dt.date
print(df.to_string(index=False))

# 2. For a given strike in expiry_code=1, how long does data span?
print("\n=== DATA SPAN PER STRIKE (CALL WEEK, expiry_code=1) ===")
df2 = con.execute("""
    SELECT strike, count(*) as bars, min(timestamp) as first, max(timestamp) as last,
           CAST(datediff('day', min(timestamp), max(timestamp)) AS INT) as span_days
    FROM options_data_clean
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
    GROUP BY strike
    ORDER BY span_days DESC
    LIMIT 20
""").fetchdf()
print(df2.to_string(index=False))

# 3. How many distinct expiry_dates per expiry_code?
print("\n=== DISTINCT EXPIRY DATES per expiry_code (WEEK) ===")
df3 = con.execute("""
    SELECT expiry_code, count(DISTINCT timestamp) as distinct_timestamps,
           min(timestamp) as first, max(timestamp) as last
    FROM expiry_availability
    WHERE expiry_flag='WEEK'
    GROUP BY expiry_code
    ORDER BY expiry_code
    LIMIT 15
""").fetchdf()
print(df3.to_string(index=False))

# 4. Check: does expiry_code=1 keep changing to track nearest week?
print("\n=== DOES expiry_code=1 TRACK NEAREST WEEK? ===")
df4 = con.execute("""
    SELECT date_trunc('month', timestamp) as month,
           min(timestamp) as first_seen, max(timestamp) as last_seen,
           datediff('day', min(timestamp), max(timestamp)) as span
    FROM expiry_availability
    WHERE expiry_flag='WEEK' AND expiry_code=1
    GROUP BY month
    ORDER BY month
    LIMIT 12
""").fetchdf()
print(df4.to_string(index=False))

# 5. Sample: pick one strike, show price continuity across weeks
print("\n=== SAME STRIKE PRICE CONTINUITY (sample strike at different dates) ===")
df5 = con.execute("""
    WITH sample_strike AS (
        SELECT DISTINCT strike FROM options_data_clean 
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
        ORDER BY random() LIMIT 1
    )
    SELECT timestamp, close, strike, expiry_code, expiry_flag
    FROM options_data_clean
    WHERE option_type='CALL' AND strike = (SELECT strike FROM sample_strike)
      AND expiry_flag='WEEK' AND expiry_code=1
    ORDER BY timestamp
    LIMIT 30
""").fetchdf()
print(df5.to_string(index=False))

con.close()
