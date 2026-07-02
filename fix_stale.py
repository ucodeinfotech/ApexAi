"""Fix stale symbols: re-import from 2026-06-01 onward for affected symbols"""
import duckdb, pandas as pd, os
from pathlib import Path

BASE = Path(r"C:\Users\pc\Downloads\stock hist data")
DB = BASE / "warehouse" / "market_data.duckdb"
CSV_DIR = BASE / "comprehensive_data"

con = duckdb.connect(str(DB))

# Get stale 1min symbols
stale = [r[0] for r in con.execute("""
    SELECT symbol FROM raw_market WHERE timeframe='1min' 
    GROUP BY symbol HAVING MAX(datetime) < '2026-06-20'
""").fetchall()]
print(f"Stale symbols: {len(stale)}")

fixed = 0
for sym in stale:
    fp = CSV_DIR / f"{sym}_ONE_MINUTE.csv"
    if not fp.exists():
        print(f"  {sym}: CSV not found, skip")
        continue
    df = pd.read_csv(fp)
    if len(df) == 0: continue
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Filter to only June 2026 onward
    df_jun = df[df["datetime"] >= "2026-06-01"]
    if len(df_jun) == 0: continue
    
    df_jun["symbol"] = sym; df_jun["timeframe"] = "1min"; df_jun["source"] = "csv_import"
    df_jun = df_jun[["datetime","open","high","low","close","volume","symbol","timeframe","source"]]
    
    # Delete June data for this symbol
    con.execute("DELETE FROM raw_market WHERE symbol=? AND timeframe='1min' AND datetime>='2026-06-01'", [sym])
    
    # Insert fresh June data
    con.register("df_jun", df_jun)
    con.execute("INSERT INTO raw_market SELECT * FROM df_jun")
    con.unregister("df_jun")
    fixed += 1
    
    if fixed % 20 == 0:
        print(f"  {fixed}/{len(stale)} fixed")

print(f"\nFixed {fixed} symbols")

# Verify
remaining = con.execute("""
    SELECT COUNT(*) FROM raw_market WHERE timeframe='1min' 
    GROUP BY symbol HAVING MAX(datetime) < '2026-06-20'
""").fetchall()
print(f"Still stale: {len(remaining)}")
print(f"Total rows: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")
con.close()
