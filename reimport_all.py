"""Re-import all CSVs into DuckDB, skipping existing rows"""
import duckdb, pandas as pd, os, re, time
from pathlib import Path

BASE = Path(r"C:\Users\pc\Downloads\stock hist data")
DB = BASE / "warehouse" / "market_data.duckdb"
CSV_DIR = BASE / "comprehensive_data"

TF_MAP = {"ONE_DAY":"1day","ONE_HOUR":"60min","FIFTEEN_MINUTE":"15min","FIVE_MINUTE":"5min","ONE_MINUTE":"1min"}

con = duckdb.connect(str(DB))
print(f"Connected. Existing rows: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")

files = [f for f in os.listdir(str(CSV_DIR)) if f.endswith(".csv") and not f.startswith("_")]
print(f"CSV files to process: {len(files)}")

total_inserted = 0; processed = 0; t0 = time.time()
for fname in sorted(files):
    m = re.match(r"(.+)_(ONE_DAY|ONE_HOUR|FIFTEEN_MINUTE|FIVE_MINUTE|ONE_MINUTE)\.csv$", fname)
    if not m: continue
    sym, tf_raw = m.group(1), m.group(2)
    tf = TF_MAP[tf_raw]
    fp = CSV_DIR / fname
    
    try:
        df = pd.read_csv(fp)
        if len(df) == 0: continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["symbol"] = sym; df["timeframe"] = tf; df["source"] = "csv_import"
        df = df[["datetime","open","high","low","close","volume","symbol","timeframe","source"]]
        
        con.register("df_new", df)
        count = con.execute(f"""
            SELECT COUNT(*) FROM df_new
            ANTI JOIN raw_market ON raw_market.symbol=df_new.symbol 
                AND raw_market.timeframe=df_new.timeframe 
                AND raw_market.datetime=df_new.datetime
        """).fetchone()[0]
        if count > 0:
            con.execute(f"""
                INSERT INTO raw_market 
                SELECT df_new.* FROM df_new
                ANTI JOIN raw_market ON raw_market.symbol=df_new.symbol 
                    AND raw_market.timeframe=df_new.timeframe 
                    AND raw_market.datetime=df_new.datetime
            """)
            total_inserted += count
        con.unregister("df_new")
        
        processed += 1
        if processed % 200 == 0:
            print(f"  {processed}/{len(files)} files ({time.time()-t0:.0f}s, {total_inserted:,} new rows)")
    except Exception as e:
        print(f"  ERROR {fname}: {e}")

t1 = time.time()
print(f"\nDone: {processed} files, {total_inserted:,} new rows in {t1-t0:.0f}s")

# Summary
print(f"\nTotal rows: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")
for r in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f"  {r[0]}: {r[1]:,} rows, {r[2]} symbols")

con.close()
