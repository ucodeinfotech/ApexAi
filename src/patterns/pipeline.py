"""
Combined Pattern Pipeline — candlestick + chart patterns across all symbols.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from src.patterns.candlestick import detect_patterns as detect_candle, compute_pattern_stats
from src.patterns.chart_patterns import detect_chart_patterns

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"

CANDLE_DIRECTION = {
    "hammer": "bullish", "bullish_engulfing": "bullish", "bullish_harami": "bullish",
    "piercing_line": "bullish", "morning_star": "bullish", "three_white_soldiers": "bullish",
    "marubozu": "bullish",
}


def create_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS pattern_occurrences (
            symbol VARCHAR, timeframe VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            pattern VARCHAR, direction VARCHAR, category VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pattern_stats (
            symbol VARCHAR, timeframe VARCHAR, pattern VARCHAR, category VARCHAR,
            total_occurrences INT, frequency_pct DOUBLE,
            fwd_period INT, avg_return DOUBLE, median_return DOUBLE,
            win_rate DOUBLE, std_return DOUBLE, best_return DOUBLE, worst_return DOUBLE
        )
    """)
    for t in ["pattern_occurrences", "pattern_stats"]:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{t[:3]}_sym_tf ON {t} (symbol, timeframe)")


def run(timeframes=None, forward_periods=[3, 5, 10]):
    con = duckdb.connect(str(DB_PATH))
    create_tables(con)

    if timeframes is None:
        timeframes = [r[0] for r in con.execute(
            "SELECT DISTINCT timeframe FROM raw_market ORDER BY timeframe"
        ).fetchall()]

    t0 = time.time()
    total_pairs = 0

    for tf in timeframes:
        syms = [r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM raw_market WHERE timeframe=? ORDER BY symbol", [tf]
        ).fetchall()]

        for sym in syms:
            df = con.execute(
                "SELECT datetime, open, high, low, close, volume FROM raw_market "
                "WHERE symbol=? AND timeframe=? ORDER BY datetime",
                [sym, tf]
            ).fetchdf()

            if len(df) < 100:
                continue

            # Detect all patterns
            candle_masks = detect_candle(df)
            chart_masks = detect_chart_patterns(df)

            # Store occurrences
            occ_rows = []
            for pname in candle_masks.columns:
                mask = candle_masks[pname].astype(bool)
                for dt in df["datetime"][mask]:
                    occ_rows.append({
                        "symbol": sym, "timeframe": tf, "datetime": dt,
                        "pattern": pname, "direction": CANDLE_DIRECTION.get(pname, "bearish"),
                        "category": "candle"
                    })

            for pname in chart_masks.columns:
                mask = chart_masks[pname].astype(bool)
                for dt in df["datetime"][mask]:
                    occ_rows.append({
                        "symbol": sym, "timeframe": tf, "datetime": dt,
                        "pattern": pname, "direction": "",
                        "category": "chart"
                    })

            if occ_rows:
                con.register("occ", pd.DataFrame(occ_rows))
                con.execute("INSERT INTO pattern_occurrences SELECT * FROM occ")
                con.unregister("occ")

            # Stats for candle + chart patterns
            for pname, masks, cat in [
                *((pn, candle_masks[[pn]], "candle") for pn in candle_masks.columns),
                *((pn, chart_masks[[pn]], "chart") for pn in chart_masks.columns),
            ]:
                stats = compute_pattern_stats(df, masks, forward_periods)
                if pname not in stats:
                    continue
                pstats = stats[pname]
                stat_rows = []
                for fp, fwd in pstats.get("forward_returns", {}).items():
                    stat_rows.append({
                        "symbol": sym, "timeframe": tf, "pattern": pname, "category": cat,
                        "total_occurrences": pstats["total_occurrences"],
                        "frequency_pct": pstats["frequency_pct"],
                        "fwd_period": fp,
                        "avg_return": fwd["avg_return"],
                        "median_return": fwd["median_return"],
                        "win_rate": fwd["win_rate"],
                        "std_return": fwd["std_return"],
                        "best_return": fwd["best"],
                        "worst_return": fwd["worst"],
                    })
                if stat_rows:
                    con.register("st", pd.DataFrame(stat_rows))
                    con.execute("INSERT INTO pattern_stats SELECT * FROM st")
                    con.unregister("st")

            total_pairs += 1
            if total_pairs % 50 == 0:
                print(f"  {total_pairs} pairs ({time.time()-t0:.0f}s)")

        print(f"  {tf}: {total_pairs} pairs")

    con.close()
    print(f"\nDone: {total_pairs} pairs in {time.time()-t0:.0f}s")
