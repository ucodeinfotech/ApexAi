import duckdb, sys, traceback
sys.stdout = open("audit_out.txt", "w", buffering=1)
print("Starting audit...")
try:
    con = duckdb.connect("warehouse/market_data.duckdb")
    print("Connected")
    r = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()
    print(f"Total rows: {r[0]:,}")
    for row in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
        print(f"  {row[0]}: {row[1]:,} rows, {row[2]} syms")
    con.close()
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print("Done")
sys.stdout.close()
