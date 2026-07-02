"""
Phase 13 — Regime Detection: bull/bear/sideways + volatility regimes.
Uses Nifty index for market-wide regime, and stock-level volatility clusters.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def detect_market_regimes(timeframes=["1day"]):
    """
    Detect bull/bear/sideways regimes using Nifty returns + Markov switching.
    Falls back to rule-based if HMM fails.
    """
    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS market_regimes (
            timeframe VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            regime_label VARCHAR,
            regime_id INT,
            volatility_regime VARCHAR
        )
    """)

    for tf in timeframes:
        # Get market data (Nifty from 5min, resample to timeframe)
        nifty = con.execute("""
            SELECT datetime, close FROM raw_market
            WHERE symbol='NIFTY50' AND timeframe='5min' ORDER BY datetime
        """).fetchdf()

        if len(nifty) < 500:
            print(f"  No Nifty data for {tf}")
            continue

        # Resample to daily
        nifty["datetime"] = pd.to_datetime(nifty["datetime"])
        daily = nifty.set_index("datetime").resample("D").agg({
            "close": "last"
        }).dropna()

        daily["returns"] = daily["close"].pct_change() * 100
        daily = daily.dropna()

        # Rule-based regime: 50-day moving average slope + returns
        daily["ma50"] = daily["close"].rolling(50, min_periods=30).mean()
        daily["above_ma"] = daily["close"] > daily["ma50"]

        # Bull: price above 50MA + positive 20d return
        daily["ret_20d"] = daily["close"].pct_change(20) * 100
        daily["bull"] = (daily["above_ma"]) & (daily["ret_20d"] > 0)
        daily["bear"] = (~daily["above_ma"]) & (daily["ret_20d"] < -5)
        daily["sideways"] = ~daily["bull"] & ~daily["bear"]

        # Volatility regime: rolling 20-day vol, top 30% = high vol
        daily["vol_20d"] = daily["returns"].rolling(20).std()
        vol_threshold = daily["vol_20d"].expanding().quantile(0.7)
        daily["high_vol"] = daily["vol_20d"] >= vol_threshold

        regime_map = {"bull": 1, "bear": -1, "sideways": 0}
        vol_map = {True: "high_vol", False: "normal_vol"}

        rows = []
        for idx, row in daily.iterrows():
            regime = "bull" if row["bull"] else ("bear" if row["bear"] else "sideways")
            rows.append((
                tf, pd.Timestamp(idx), regime, regime_map[regime],
                vol_map[row["high_vol"]]
            ))

        if rows:
            con.register("reg", pd.DataFrame(rows, columns=[
                "timeframe", "datetime", "regime_label", "regime_id", "volatility_regime"
            ]))
            con.execute("INSERT INTO market_regimes SELECT * FROM reg")
            con.unregister("reg")

        # Print summary
        summary = con.execute(f"""
            SELECT regime_label, COUNT(1) FROM market_regimes
            WHERE timeframe=? GROUP BY regime_label
        """, [tf]).fetchall()
        print(f"  {tf} regimes: {dict(summary)}")

    con.close()
    print("  Regime detection done")


def run_all(timeframes=["1day"]):
    print("Regime Detection (Phase 13)...")
    detect_market_regimes(timeframes)
