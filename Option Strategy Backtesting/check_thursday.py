"""Verify Thursday expiry pattern."""
import duckdb, pandas as pd
DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
con = duckdb.connect(DB_PATH)

# Simple query: strike 22200 daily bars
df = con.execute("""
SELECT date(timestamp) as d, count(*) as bars, min(close) min_c, max(close) max_c
FROM options_data_clean
WHERE option_type='CALL' AND strike=22200 AND expiry_code=1 AND expiry_flag='WEEK'
GROUP BY date(timestamp) ORDER BY d LIMIT 100
""").fetchdf()
print("Strike 22200 daily bars:")
df["dow"] = df["d"].dt.dayofweek
df["dow_name"] = df["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
print(df.to_string(index=False))

# Show Thur/Fri transitions
print("\n=== THU/THU transitions (weekly expiry boundaries) ===")
thu_dates = df[df["dow"]==3]["d"].tolist()
print(f"Thursdays with data: {len(thu_dates)}")
# Check: on each Thu, what's the next Thu?
for i in range(min(5, len(thu_dates)-1)):
    gap = (thu_dates[i+1] - thu_dates[i]).days
    print(f"  Thu {thu_dates[i]} -> Thu {thu_dates[i+1]}: {gap}d gap")
    # Are there bars between them?
    between = df[(df["d"] > thu_dates[i]) & (df["d"] < thu_dates[i+1])]
    if len(between):
        print(f"    Bars between: {between['d'].tolist()}")

con.close()
