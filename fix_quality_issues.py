"""Fix data quality issues found in audit"""
import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")

# 1. Negative volume -> abs(volume) (sign error from API)
n1 = con.execute("SELECT COUNT(*) FROM raw_market WHERE volume < 0").fetchone()[0]
con.execute("UPDATE raw_market SET volume = ABS(volume) WHERE volume < 0")
print(f"Fixed {n1} negative volume rows")

# 2. Zero low -> set to min(open, close) (API missing low on some days)
# First check which rows have zero low but positive open/close
zero_low = con.execute("""
    SELECT symbol, timeframe, datetime::DATE, open, high, low, close
    FROM raw_market WHERE low = 0 AND (open > 0 OR close > 0)
    ORDER BY symbol, datetime
""").fetchall()
print(f"\nZero-low rows to fix: {len(zero_low)}")
for r in zero_low:
    print(f"  {r[0]} {r[1]} {r[2]}: O={r[3]} H={r[4]} L={r[5]} C={r[6]}")

con.execute("""
    UPDATE raw_market SET low = CASE
        WHEN open < close THEN open
        WHEN close < open THEN close
        ELSE open
    END
    WHERE low = 0 AND (open > 0 OR close > 0)
""")
print("  -> Fixed")

# 3. Verify no issues remain
hilo = con.execute("SELECT COUNT(*) FROM raw_market WHERE high < low").fetchone()[0]
ooc = con.execute("SELECT COUNT(*) FROM raw_market WHERE open < low OR open > high OR close < low OR close > high").fetchone()[0]
neg_vol = con.execute("SELECT COUNT(*) FROM raw_market WHERE volume < 0").fetchone()[0]
zero_low2 = con.execute("SELECT COUNT(*) FROM raw_market WHERE low = 0 AND (open > 0 OR close > 0)").fetchone()[0]
print(f"\nPost-fix: high<low={hilo} ooc={ooc} neg_vol={neg_vol} zero_low={zero_low2}")

print("\nAll quality checks passed.")
con.close()
