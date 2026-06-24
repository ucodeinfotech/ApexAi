"""
Feature engineering pipeline — sequential compute (avoids Windows spawn issues).
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from src.features.indicators import compute_all_features

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def _get_indicator_cols():
    """Discover all indicator column names from a sample run."""
    np.random.seed(0)
    sample = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=500, freq="h"),
        "open": np.random.randn(500).cumsum() + 100,
        "volume": np.random.randint(1e5, 1e7, 500),
    })
    sample["high"] = sample["open"] + np.random.rand(500) * 2
    sample["low"] = sample["open"] - np.random.rand(500) * 2
    sample["close"] = sample["open"] + np.random.randn(500) * 0.5
    result = compute_all_features(sample)
    return sorted([c for c in result.columns
                   if c not in ("datetime", "open", "high", "low", "close", "volume")])


def _ensure_feature_store(con):
    """Create feature_store table with all indicator columns."""
    cols = _get_indicator_cols()
    col_defs = ",\n".join(f'    "{c}" DOUBLE' for c in cols)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS feature_store (
            symbol VARCHAR, timeframe VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
            {col_defs}
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_feat_sym_tf_dt ON feature_store (symbol, timeframe, datetime)")


def process_one(con, symbol, timeframe):
    """Compute and store features for one (symbol, timeframe) pair."""
    done = con.execute(
        "SELECT 1 FROM feature_store WHERE symbol=? AND timeframe=? LIMIT 1",
        [symbol, timeframe]
    ).fetchone()
    if done:
        return 0

    df = con.execute(
        "SELECT datetime, open, high, low, close, volume FROM raw_market "
        "WHERE symbol=? AND timeframe=? ORDER BY datetime",
        [symbol, timeframe]
    ).fetchdf()

    if len(df) < 200:
        return 0

    df = compute_all_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Reorder DataFrame columns to match feature_store schema (by position)
    table_cols = [r[1] for r in con.execute("PRAGMA table_info('feature_store')").fetchall()]
    df_cols = [c for c in table_cols if c in df.columns]
    con.register("df", df[df_cols])
    con.execute("INSERT INTO feature_store SELECT * FROM df")
    con.unregister("df")
    return len(df)


def run_pipeline(timeframes=None, symbols=None):
    """Sequential feature pipeline — simple, reliable, no multiprocessing issues."""
    con = duckdb.connect(str(DB_PATH))
    _ensure_feature_store(con)

    if timeframes is None:
        timeframes = [r[0] for r in con.execute(
            "SELECT DISTINCT timeframe FROM raw_market ORDER BY timeframe"
        ).fetchall()]

    t0 = time.time()
    total = 0
    pairs = 0

    for tf in timeframes:
        syms = con.execute(
            "SELECT DISTINCT symbol FROM raw_market WHERE timeframe=? ORDER BY symbol",
            [tf]
        ).fetchall()
        syms = [r[0] for r in syms]
        if symbols is not None:
            syms = [s for s in syms if s in symbols]

        for sym in syms:
            n = process_one(con, sym, tf)
            if n:
                total += n
                pairs += 1
                if pairs % 20 == 0:
                    print(f"  {pairs} pairs, {total:,} rows ({time.time()-t0:.0f}s)")

        print(f"  {tf}: {pairs} current pairs, {total:,} rows")

    row_count = con.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
    con.close()
    print(f"\nDone: {pairs} pairs, {row_count:,} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run_pipeline(timeframes=["1day", "60min", "15min"])
