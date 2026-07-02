import duckdb, pandas as pd, numpy as np
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')

print("=== raw_market 1day date range ===")
r = con.execute("SELECT MIN(datetime), MAX(datetime), COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='1day'").fetchone()
rc = con.execute("SELECT COUNT(*) FROM raw_market WHERE timeframe='1day'").fetchone()[0]
print(f"Date range: {r[0]} to {r[1]}, {r[2]} symbols, {rc:,} rows")

# Check raw_market 1day per year
print("\nraw_market 1day per year:")
df = con.execute("SELECT EXTRACT(YEAR FROM datetime) as y, COUNT(*) as c FROM raw_market WHERE timeframe='1day' GROUP BY y ORDER BY y").fetchdf()
for _, r in df.iterrows():
    print(f"  {int(r['y'])}: {r['c']:,}")

# Check unique symbols per year
print("\nraw_market 1day symbols per year:")
df2 = con.execute("SELECT EXTRACT(YEAR FROM datetime) as y, COUNT(DISTINCT symbol) as c FROM raw_market WHERE timeframe='1day' GROUP BY y ORDER BY y").fetchdf()
for _, r in df2.iterrows():
    print(f"  {int(r['y'])}: {r['c']} symbols")

# Check 5min data date range
print(f"\n=== raw_market 5min date range ===")
r = con.execute("SELECT MIN(datetime), MAX(datetime) FROM raw_market WHERE timeframe='5min'").fetchone()
print(f"5min: {r[0]} to {r[1]}")

# Check total 1day rows in feature_store BEFORE rebuild would have had
# Check if feature_store has ANY rows at other timeframes
print(f"\n=== feature_store timeframes ===")
df = con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM feature_store GROUP BY timeframe ORDER BY timeframe").fetchdf()
print(df.to_string())

# Check 60min data per year
print("\nfeature_store 1day year distribution:")
df = con.execute("SELECT EXTRACT(YEAR FROM datetime) as y, COUNT(*) as c FROM feature_store WHERE timeframe='1day' GROUP BY y ORDER BY y").fetchdf()
for _, r in df.iterrows():
    print(f"  {int(r['y'])}: {r['c']:,}")

con.close()
