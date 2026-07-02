"""Fast re-import: delete stale rows, insert fresh from CSVs for affected symbols"""
import duckdb, pandas as pd, os, re, time
from pathlib import Path

BASE = Path(r"C:\Users\pc\Downloads\stock hist data")
DB = BASE / "warehouse" / "market_data.duckdb"
CSV_DIR = BASE / "comprehensive_data"
TF_MAP = {"ONE_DAY":"1day","ONE_HOUR":"60min","FIFTEEN_MINUTE":"15min","FIVE_MINUTE":"5min","ONE_MINUTE":"1min"}

con = duckdb.connect(str(DB))
print("Connected")

# Find symbols that need updating: stale 1min or missing 1day
needs_update = set()
for r in con.execute("""
    SELECT symbol FROM raw_market WHERE timeframe='1min' 
    GROUP BY symbol HAVING MAX(datetime) < '2026-06-20'
""").fetchall(): needs_update.add(r[0])

for r in con.execute("""
    SELECT DISTINCT rm.symbol FROM raw_market rm 
    WHERE rm.timeframe='1min' AND NOT EXISTS (
        SELECT 1 FROM raw_market WHERE symbol=rm.symbol AND timeframe='1day'
    )
""").fetchall(): needs_update.add(r[0])

print(f"Symbols needing update: {len(needs_update)}")

# Process each timeframe for affected symbols
total_ins = 0; total_del = 0
for sym in sorted(needs_update):
    for tf_raw, tf in TF_MAP.items():
        fp = CSV_DIR / f"{sym}_{tf_raw}.csv"
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        if len(df) == 0: continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["symbol"] = sym; df["timeframe"] = tf; df["source"] = "csv_import"
        df = df[["datetime","open","high","low","close","volume","symbol","timeframe","source"]]
        
        # Delete existing rows for this symbol+tf in the CSV's date range
        min_dt, max_dt = df["datetime"].min(), df["datetime"].max()
        del_count = con.execute("DELETE FROM raw_market WHERE symbol=? AND timeframe=? AND datetime>=? AND datetime<=?",
                                [sym, tf, min_dt, max_dt]).fetchone()
        # fetchone returns None for DELETE, but we can get rowcount differently
        # Actually just track via the return
        total_del += 1  # approximate
        
        con.register("df_new", df)
        con.execute("INSERT INTO raw_market SELECT * FROM df_new")
        con.unregister("df_new")
        total_ins += len(df)
    
    print(f"  {sym}: re-imported")

print(f"\nDone: {total_ins:,} rows inserted ({len(needs_update)} symbols)")
print(f"Total DB rows: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")
for r in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f"  {r[0]}: {r[1]:,} rows, {r[2]} syms")
con.close()
