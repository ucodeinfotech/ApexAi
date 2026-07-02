"""Verify options_data_filled table."""
import duckdb
con = duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")
r = con.execute("SELECT count(*), min(timestamp), max(timestamp), count(DISTINCT strike), count(DISTINCT expiry_date) FROM options_data_filled").fetchone()
print(f"Rows: {r[0]:,} | {r[1]} to {r[2]} | Strikes: {r[3]} | Expiry dates: {r[4]}")
# Check specific strike/expiry for gaps
r2 = con.execute("SELECT count(*) FROM options_data_filled WHERE strike=22200 AND expiry_date='2024-02-22'").fetchone()
print(f"Strike 22200, expiry 2024-02-22: {r2[0]} bars")
# Check gaps
r3 = con.execute("""
WITH x AS (
  SELECT timestamp, lag(timestamp) OVER(ORDER BY timestamp) as p
  FROM options_data_filled
  WHERE strike=22200 AND expiry_date='2024-02-22'
)
SELECT count(*) FILTER(WHERE p IS NOT NULL AND datediff('minute',p,timestamp) != 5) as gaps
FROM x
""").fetchone()
print(f"Gaps (>5min): {r3[0]}")
# Show first/last bars for that strike/expiry
con.execute("SELECT timestamp, close FROM options_data_filled WHERE strike=22200 AND expiry_date='2024-02-22' ORDER BY timestamp").fetchdf().to_csv("_tmp_check.csv")
print(con.execute("SELECT min(timestamp) first, max(timestamp) last, min(close), max(close) FROM options_data_filled WHERE strike=22200 AND expiry_date='2024-02-22'").fetchdf().to_string())
# Check next expiry for same strike
print("\nStrike 22200, expiry 2024-02-29:")
print(con.execute("SELECT min(timestamp) first, max(timestamp) last, min(close), max(close) FROM options_data_filled WHERE strike=22200 AND expiry_date='2024-02-29'").fetchdf().to_string())
con.close()
