"""Thorough data quality audit"""
import duckdb, json
con = duckdb.connect("warehouse/market_data.duckdb")

print("=== DATA QUALITY AUDIT ===")
print()

# 1. Price integrity
hilo = con.execute("SELECT COUNT(*) FROM raw_market WHERE high < low").fetchone()[0]
ooc = con.execute("SELECT COUNT(*) FROM raw_market WHERE open < low OR open > high OR close < low OR close > high").fetchone()[0]
neg = con.execute("SELECT COUNT(*) FROM raw_market WHERE open < 0 OR high < 0 OR low < 0 OR close < 0 OR volume < 0").fetchone()[0]
zero = con.execute("SELECT COUNT(*) FROM raw_market WHERE open = 0 OR high = 0 OR low = 0 OR close = 0").fetchone()[0]
print(f"High < Low:     {hilo}")
print(f"OOC range:      {ooc}")
print(f"Negatives:      {neg}")
print(f"Zero prices:    {zero}")

# 2. NULLs
cols = ["datetime","open","high","low","close","volume","symbol","timeframe","source"]
nulls = {c: con.execute(f"SELECT COUNT(*) FROM raw_market WHERE {c} IS NULL").fetchone()[0] for c in cols}
null_total = sum(nulls.values())
print(f"NULLs:           {null_total} ({nulls})")

# 3. Duplicates
total = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()[0]
unique = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT symbol, timeframe, datetime FROM raw_market)").fetchone()[0]
dups = total - unique
print(f"Duplicates:      {dups} ({total:,} rows, {unique:,} unique)")

# 4. Zero volume with nonzero close
badvol = con.execute("SELECT COUNT(*) FROM raw_market WHERE volume = 0 AND close > 0").fetchone()[0]
print(f"Zero vol+price:  {badvol}")

# 5. Staleness — latest date per symbol
stale = con.execute("""
    SELECT symbol, MAX(datetime::DATE) as last_dt
    FROM raw_market WHERE timeframe='1day'
    GROUP BY symbol
    ORDER BY last_dt
""").fetchall()
oldest = stale[0]
newest = stale[-1]
stale_count = sum(1 for _, dt in stale if dt < __import__('datetime').date(2026,6,24))
print(f"\nStaleness (1day):")
print(f"  Range: {oldest[1]} to {newest[1]}")
print(f"  Symbols ending before 2026-06-24: {stale_count}")
print(f"  Oldest 5: {', '.join(f'{s}({d})' for s,d in stale[:5])}")
too_stale = [(s,d) for s,d in stale if d < __import__('datetime').date(2026,6,1)]
if too_stale:
    print(f"  BEFORE JUNE 2026: {len(too_stale)} symbols")
    for s,d in too_stale[:10]:
        print(f"    {s}: last={d}")

# 6. Price outliers by symbol
print(f"\nPrice outliers (>20x median close):")
outliers = con.execute("""
    WITH stats AS (
        SELECT symbol, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY close) as med
        FROM raw_market WHERE timeframe='1day' AND close > 0
        GROUP BY symbol
    )
    SELECT r.symbol, r.datetime::DATE, r.close, s.med,
           r.close / NULLIF(s.med,0) as ratio
    FROM raw_market r JOIN stats s ON r.symbol=s.symbol
    WHERE r.timeframe='1day' AND r.close > s.med*20
    ORDER BY ratio DESC
""").fetchall()
if outliers:
    print(f"  Found {len(outliers)} rows")
    for s,dt,c,m,r in outliers[:10]:
        print(f"  {s} {dt}: close={c} med={m} ratio={r:.1f}x")
else:
    print("  None")

# 7. Row counts by timeframe
print(f"\nRows by timeframe:")
for tf,r in con.execute("SELECT timeframe, COUNT(*) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe=?", [tf]).fetchone()[0]
    print(f"  {tf}: {r:>12,} rows, {syms} symbols")

print(f"\nTotal rows: {total:,}")
print(f"Total symbols: {con.execute('SELECT COUNT(DISTINCT symbol) FROM raw_market').fetchone()[0]}")
con.close()
