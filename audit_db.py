import duckdb

con = duckdb.connect("warehouse/market_data.duckdb")

print("=== Database Summary ===")
for r in con.execute("""
    SELECT timeframe, COUNT(*) as rows, COUNT(DISTINCT symbol) as syms,
           CAST(MIN(datetime) AS DATE) as first_date,
           CAST(MAX(datetime) AS DATE) as last_date
    FROM raw_market GROUP BY timeframe ORDER BY timeframe
""").fetchall():
    print(f"  {r[0]}: {r[1]:,} rows, {r[2]} syms, {r[3]} to {r[4]}")

total_rows = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()[0]
total_syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market").fetchone()[0]
print(f"\nTotal rows: {total_rows:,}")
print(f"Total symbols: {total_syms}")

# Check for stale 1min data (ending before June 20 2026)
stale = con.execute("""
    SELECT symbol, CAST(MAX(datetime) AS DATE) as last_dt 
    FROM raw_market WHERE timeframe='1min' 
    GROUP BY symbol 
    HAVING MAX(datetime) < CAST('2026-06-20' AS TIMESTAMP)
    ORDER BY last_dt
""").fetchall()
print(f"\nStale 1min symbols (last < 2026-06-20): {len(stale)}")
for s in stale[:10]:
    print(f"  {s[0]}: ends {s[1]}")

# Check symbols missing higher timeframes
with_1min = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1min'").fetchall())
with_1day = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day'").fetchall())
missing_1day = with_1min - with_1day
print(f"\nSymbols with 1min but missing 1day: {len(missing_1day)}")
for s in sorted(missing_1day):
    print(f"  {s}")

# Check symbols in feature_store vs raw_market
fs_syms = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM feature_store").fetchall())
rm_syms = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day'").fetchall())
print(f"\nFeature store symbols: {len(fs_syms)}")
print(f"Raw market 1day symbols: {len(rm_syms)}")
print(f"Raw > Feature: {len(rm_syms - fs_syms)}")
for s in sorted(rm_syms - fs_syms):
    print(f"  {s} (has 1day in raw but no feature_store)")

con.close()
print("\nDone!")
