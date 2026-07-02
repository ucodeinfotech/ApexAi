import duckdb, pandas as pd, numpy as np
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')

# 5min data available for resampling to 1day
print("=== 1day data count BEFORE rebuild ===")
rc_raw = con.execute("SELECT COUNT(*) FROM raw_market WHERE timeframe='1day'").fetchone()[0]
print(f"raw_market 1day rows: {rc_raw:,}")

# Check if we can resample 5min -> 1day for the gap
print("\n=== 5min to 1day resample potential ===")
r = con.execute("SELECT MIN(datetime), MAX(datetime), COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='5min'").fetchone()
print(f"5min: {r[0]} to {r[1]}, {r[2]} symbols")

# 1day data starts 2016-10-03, 5min from 2015-01-01
# How much 5min data exists before 2016-10-01?
pre = con.execute("SELECT COUNT(*) FROM raw_market WHERE timeframe='5min' AND datetime < '2016-10-01'").fetchone()[0]
print(f"5min rows before 2016-10-01: {pre:,}")

# But many symbols listed later - check when each symbol first appears in 5min
pre_syms = con.execute("""SELECT symbol, MIN(datetime) as first_dt, COUNT(*) as rows
    FROM raw_market WHERE timeframe='5min' AND datetime < '2016-10-01'
    GROUP BY symbol ORDER BY first_dt""").fetchdf()
print(f"\n5min pre-Oct2016: {len(pre_syms)} symbols")
print(f"Earliest: {pre_syms['first_dt'].min()}")
print(f"Symbols with >=100 rows before Oct2016: {(pre_syms['rows']>=100).sum()}")

# How about total symbol coverage in 5min?
print("\n=== Compare 1day vs 5min symbol coverage ===")
syms_1day = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day'").fetchall())
syms_5min = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='5min'").fetchall())
print(f"1day symbols: {len(syms_1day)}")
print(f"5min symbols: {len(syms_5min)}")
print(f"In 5min but not 1day: {syms_5min - syms_1day}")
print(f"In 1day but not 5min: {syms_1day - syms_5min}")

# When does each symbol first appear in 1day?
print("\n=== Symbol first dates in 1day ===")
first_dates = con.execute("""SELECT symbol, MIN(datetime) as first_dt, COUNT(*) as rows
    FROM raw_market WHERE timeframe='1day' GROUP BY symbol ORDER BY first_dt""").fetchdf()
print(f"First date range: {first_dates['first_dt'].min()} to {first_dates['first_dt'].max()}")
# Count how many symbols start in each year
first_dates['year'] = pd.to_datetime(first_dates['first_dt']).dt.year
print(f"Symbols starting per year:")
for y, c in first_dates['year'].value_counts().sort_index().items():
    print(f"  {int(y)}: {c} symbols")

con.close()
