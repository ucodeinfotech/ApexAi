"""
Unified DuckDB Data Warehouse — native CSV loading for maximum speed.
"""
import duckdb
from pathlib import Path
import time

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"
FULL_HISTORY = BASE_DIR / "nifty50_full_history"
COMPREHENSIVE = BASE_DIR / "comprehensive_data"


def build_warehouse():
    (DB_PATH.parent).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_market (
            datetime TIMESTAMP WITH TIME ZONE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
            symbol VARCHAR, timeframe VARCHAR, source VARCHAR
        )
    """)

    t0 = time.time()
    total_files = 0

    # Helper: load all CSVs matching a glob pattern in a directory
    # Files are named: SYMBOL_TIMEFRAME.csv (e.g. RELIANCE_FIFTEEN_MINUTE.csv)
    def load_batch(source_name, directory, suffix, timeframe_label):
        nonlocal total_files
        pattern = str(directory / f"*_{suffix}.csv")
        # Get file list
        files = sorted(directory.glob(f"*_{suffix}.csv"))
        if not files:
            return

        for fpath in files:
            symbol = fpath.stem[:-len(f"_{suffix}")]
            exists = con.execute(
                "SELECT 1 FROM raw_market WHERE symbol=? AND timeframe=? AND source=? LIMIT 1",
                [symbol, timeframe_label, source_name]
            ).fetchone()
            if exists:
                continue

            con.execute(f"""
                INSERT INTO raw_market
                SELECT
                    datetime::TIMESTAMPTZ, open::DOUBLE, high::DOUBLE,
                    low::DOUBLE, close::DOUBLE, volume::DOUBLE,
                    '{symbol}', '{timeframe_label}', '{source_name}'
                FROM read_csv_auto('{fpath}')
            """)
            total_files += 1

    # ── full_history (Nifty 50, 1min + 15min) ──
    if FULL_HISTORY.exists():
        print("nifty50_full_history...")
        load_batch("full_history", FULL_HISTORY, "FIFTEEN_MINUTE", "15min")
        print(f"  15min: {total_files} files so far")
        load_batch("full_history", FULL_HISTORY, "ONE_MINUTE", "1min")
        print(f"  1min: {total_files} files so far")

    # ── comprehensive_data (90 stocks, 5 timeframes) ──
    if COMPREHENSIVE.exists():
        print("comprehensive_data...")
        for suffix, tf in [
            ("ONE_DAY", "1day"), ("ONE_HOUR", "60min"),
            ("FIFTEEN_MINUTE", "15min"), ("FIVE_MINUTE", "5min"), ("ONE_MINUTE", "1min"),
        ]:
            load_batch("comprehensive", COMPREHENSIVE, suffix, tf)
            print(f"  {tf}: {total_files} files so far")

    elapsed = time.time() - t0
    row_count = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()[0]
    print(f"\nFiles: {total_files} | Rows: {row_count:,} | Time: {elapsed:.0f}s")
    print(f"Symbols: {con.execute('SELECT COUNT(DISTINCT symbol) FROM raw_market').fetchone()[0]}")
    timeframes = [r[0] for r in con.execute("SELECT DISTINCT timeframe FROM raw_market ORDER BY timeframe").fetchall()]
    print(f"Timeframes: {timeframes}")
    dr = con.execute("SELECT MIN(datetime)::DATE, MAX(datetime)::DATE FROM raw_market").fetchone()
    print(f"Date range: {dr[0]} to {dr[1]}")

    print("Building indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sym_tf_dt ON raw_market (symbol, timeframe, datetime)")
    con.close()
    print(f"Done -> {DB_PATH}")


if __name__ == "__main__":
    build_warehouse()
