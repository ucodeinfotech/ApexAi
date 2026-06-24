"""
Pipeline for Smart Money Concepts, Wyckoff, Market/Volume Profile, Relative Strength.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from src.patterns.market_structure import compute_all_structure, STRUCTURE_COLUMNS

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def _ensure_structure_table(con):
    """Create market_structure table."""
    col_defs = ",\n".join(f'    "{c}" DOUBLE' for c in STRUCTURE_COLUMNS)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS market_structure (
            symbol VARCHAR, timeframe VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            {col_defs}
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ms_sym_tf_dt ON market_structure (symbol, timeframe, datetime)")


def _load_nifty_close(con):
    """Load NIFTY50 close prices for relative strength."""
    df = con.execute(
        "SELECT datetime, close FROM raw_market WHERE symbol='NIFTY50' AND timeframe='5min' ORDER BY datetime"
    ).fetchdf()
    if len(df) == 0:
        return None
    df = df.set_index("datetime")["close"]
    df = df[~df.index.duplicated(keep="first")]
    return df


def process_one(con, symbol, timeframe, nifty_close):
    """Compute market structure features for one pair."""
    done = con.execute(
        "SELECT 1 FROM market_structure WHERE symbol=? AND timeframe=? LIMIT 1",
        [symbol, timeframe]
    ).fetchone()
    if done:
        return 0

    df = con.execute(
        "SELECT datetime, open, high, low, close, volume FROM raw_market "
        "WHERE symbol=? AND timeframe=? ORDER BY datetime",
        [symbol, timeframe]
    ).fetchdf()

    if len(df) < 100:
        return 0

    # Use resampled Nifty data if timeframe differs
    market_close = nifty_close
    df = compute_all_structure(df, market_close=market_close)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Match columns to table schema by name
    table_cols = [r[1] for r in con.execute("PRAGMA table_info('market_structure')").fetchall()]
    avail = [c for c in table_cols if c in df.columns]
    con.register("df", df[avail])
    con.execute("INSERT INTO market_structure SELECT * FROM df")
    con.unregister("df")
    return len(df)


def run(timeframes=None):
    con = duckdb.connect(str(DB_PATH))
    _ensure_structure_table(con)
    nifty_close = _load_nifty_close(con)

    if timeframes is None:
        timeframes = [r[0] for r in con.execute(
            "SELECT DISTINCT timeframe FROM raw_market ORDER BY timeframe"
        ).fetchall()]

    t0 = time.time()
    total = 0

    for tf in timeframes:
        syms = [r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM raw_market WHERE timeframe=? AND symbol NOT IN ('NIFTY50','BANKNIFTY') ORDER BY symbol",
            [tf]
        ).fetchall()]

        for sym in syms:
            n = process_one(con, sym, tf, nifty_close)
            if n:
                total += 1
                if total % 50 == 0:
                    print(f"  {total} pairs ({time.time()-t0:.0f}s)")

        print(f"  {tf}: {total} pairs")

    row_count = con.execute("SELECT COUNT(1) FROM market_structure").fetchone()[0]
    con.close()
    print(f"\nDone: {row_count:,} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run(timeframes=["1day", "60min", "15min"])
