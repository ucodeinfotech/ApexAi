"""Fix all data quality issues in raw_market"""
import duckdb

con = duckdb.connect("warehouse/market_data.duckdb")
print("Connected")

# 1. Fix high<low rows — swap them
print("\n1. Fixing high<low rows...")
bad_hl = con.execute("SELECT COUNT(*) FROM raw_market WHERE high < low").fetchone()[0]
if bad_hl > 0:
    con.execute("UPDATE raw_market SET high = low, low = high WHERE high < low")
    print(f"   Fixed {bad_hl:,} rows (swapped high<low)")
else:
    print("   None found")

# 2. Fix open/close outside [low, high] — clamp to range
print("\n2. Fixing open/close outside range...")
bad_oc = con.execute("SELECT COUNT(*) FROM raw_market WHERE open < low OR open > high OR close < low OR close > high").fetchone()[0]
if bad_oc > 0:
    con.execute("UPDATE raw_market SET open = GREATEST(LEAST(open, high), low) WHERE open < low OR open > high")
    con.execute("UPDATE raw_market SET close = GREATEST(LEAST(close, high), low) WHERE close < low OR close > high")
    print(f"   Fixed {bad_oc:,} rows (clamped open/close to [low, high])")
else:
    print("   None found")

# 3. Check zero-volume rows — flag any where high=low=open=close=0 (dead rows)
print("\n3. Checking zero-volume rows...")
dead = con.execute("""
    SELECT COUNT(*) FROM raw_market 
    WHERE volume = 0 AND open = 0 AND high = 0 AND low = 0 AND close = 0
""").fetchone()[0]
if dead > 0:
    con.execute("DELETE FROM raw_market WHERE volume = 0 AND open = 0 AND high = 0 AND low = 0 AND close = 0")
    print(f"   Removed {dead:,} dead rows (all zero OHLC)")
else:
    print("   No dead rows found (zero-volume with valid prices — normal)")

# 4. Check for any remaining NULLs in critical columns
print("\n4. Checking NULL values...")
for col in ["open","high","low","close","volume","datetime","symbol","timeframe"]:
    n = con.execute(f"SELECT COUNT(*) FROM raw_market WHERE {col} IS NULL").fetchone()[0]
    if n > 0:
        con.execute(f"DELETE FROM raw_market WHERE {col} IS NULL")
        print(f"   Removed {n} rows with NULL {col}")

# 5. Verify duplicates (same symbol+timeframe+datetime)
print("\n5. Checking duplicates...")
dups = con.execute("""
    SELECT COUNT(*) - COUNT(DISTINCT (symbol || timeframe || datetime::VARCHAR)) as dup_count
    FROM raw_market
""").fetchone()[0]
if dups > 0:
    con.execute("""
        DELETE FROM raw_market WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM raw_market 
            GROUP BY symbol, timeframe, datetime
        )
    """)
    print(f"   Removed {dups:,} duplicate rows")
else:
    print("   No duplicates found")

# Final verification
print("\n=== FINAL VERIFICATION ===")
print(f"High<low: {con.execute('SELECT COUNT(*) FROM raw_market WHERE high < low').fetchone()[0]:,}")
print(f"OOC range: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open < low OR open > high OR close < low OR close > high').fetchone()[0]:,}")
print(f"Negative: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open < 0 OR high < 0 OR low < 0 OR close < 0').fetchone()[0]:,}")
print(f"Total rows: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")
print(f"Symbols: {con.execute('SELECT COUNT(DISTINCT symbol) FROM raw_market').fetchone()[0]}")
print("\nData quality fixes complete!")
con.close()
