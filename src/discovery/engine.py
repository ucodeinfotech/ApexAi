"""
Phase 18 — High-Gainer Discovery Engine.
Combines range predictions, structure, and regime into a daily 0-100 score.
Predicts probability of NEXT candle having high-low range > 2%, 5%, 6%.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"

RANGE_TARGETS = ["hr_2pct", "hr_5pct", "hr_6pct"]
MODEL_NAMES = {t: f"xgb_all_{t}" for t in RANGE_TARGETS}
DIRECTIONAL_MODELS = {
    "wide_bullish": "xgb_dir_wide_bullish",
    "wide_bearish": "xgb_dir_wide_bearish",
}


def build_scores(timeframe="1day"):
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS discovery_scores (
            timeframe VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            symbol VARCHAR,
            range_2pct_prob DOUBLE,
            range_5pct_prob DOUBLE,
            range_6pct_prob DOUBLE,
            expected_range_pct DOUBLE,
            structure_score DOUBLE,
            regime_alignment DOUBLE,
            wide_bullish_prob DOUBLE,
            wide_bearish_prob DOUBLE,
            net_directional DOUBLE,
            composite_score DOUBLE,
            confidence VARCHAR
        )
    """)

    print(f"  Building {timeframe} scores...")

    # Load range predictions
    preds = {}
    for name in RANGE_TARGETS:
        df = con.execute("""
            SELECT symbol, datetime, score, expected_return
            FROM ml_predictions_oos
            WHERE timeframe=? AND model_name=?
            ORDER BY datetime
        """, [timeframe, MODEL_NAMES[name]]).fetchdf()
        if len(df) > 0:
            df["datetime"] = pd.to_datetime(df["datetime"])
            preds[name] = df

    if not preds:
        print("  No range predictions found")
        con.close(); return

    base = preds["hr_2pct"][["symbol", "datetime", "score", "expected_return"]].copy()
    base.columns = ["symbol", "datetime", "hr_2pct_prob", "hr_2pct_range"]
    for name in ["hr_5pct", "hr_6pct"]:
        other = preds[name][["symbol", "datetime", "score"]].copy()
        other.columns = ["symbol", "datetime", f"{name}_prob"]
        base = base.merge(other, on=["symbol", "datetime"], how="outer")

    # Load directional predictions
    for dir_name, dir_model in DIRECTIONAL_MODELS.items():
        dir_df = con.execute("""
            SELECT symbol, datetime, score
            FROM ml_predictions_oos
            WHERE timeframe=? AND model_name=?
            ORDER BY datetime
        """, [timeframe, dir_model]).fetchdf()
        if len(dir_df) > 0:
            dir_df["datetime"] = pd.to_datetime(dir_df["datetime"])
            dir_df = dir_df.rename(columns={"score": f"{dir_name}_prob"})
            base = base.merge(dir_df, on=["symbol", "datetime"], how="left")

    for c in ["wide_bullish_prob", "wide_bearish_prob"]:
        if c not in base.columns:
            base[c] = 0.0

    base = base.dropna(subset=["hr_2pct_prob", "hr_5pct_prob", "hr_6pct_prob"])

    # Latest date per symbol
    base["dt_rank"] = base.groupby("symbol")["datetime"].rank(ascending=False)
    latest = base[base["dt_rank"] == 1].copy()
    latest_date = latest["datetime"].max()

    # Market structure
    structure = con.execute("""
        SELECT ms.symbol, ms.datetime,
               COALESCE(wyckoff_phase, 0) as wyckoff_phase,
               COALESCE(wyckoff_spring, 0) as wyckoff_spring,
               COALESCE(wyckoff_upthrust, 0) as wyckoff_upthrust,
               COALESCE(fvg_bullish, 0) as fvg_bull,
               COALESCE(fvg_bearish, 0) as fvg_bear,
               COALESCE(ob_bullish, 0) as ob_bull,
               COALESCE(ob_bearish, 0) as ob_bear,
               COALESCE(bos_up, 0) as bos_up,
               COALESCE(bos_down, 0) as bos_down,
               COALESCE(rs_vs_market, 0) as rs
        FROM market_structure ms WHERE ms.timeframe=?
        ORDER BY ms.symbol, ms.datetime
    """, [timeframe]).fetchdf()
    if len(structure) > 0:
        structure["datetime"] = pd.to_datetime(structure["datetime"])
        structure["dt_diff"] = structure.groupby("symbol")["datetime"].transform(
            lambda x: (latest_date - x).abs()
        )
        latest_struct = structure.loc[structure.groupby("symbol")["dt_diff"].idxmin()]
    else:
        latest_struct = pd.DataFrame()

    regime = con.execute("""
        SELECT datetime, regime_label, volatility_regime
        FROM market_regimes WHERE timeframe=?
        ORDER BY datetime DESC LIMIT 1
    """, [timeframe]).fetchdf()
    current_regime = regime.iloc[0]["regime_label"] if len(regime) > 0 else "sideways"

    merged = latest.merge(
        latest_struct[["symbol", "rs", "wyckoff_phase", "bos_up", "fvg_bull", "ob_bull"]],
        on="symbol", how="left"
    ).fillna(0)

    merged["structure_score"] = (
        (merged["rs"] > 0).astype(int) * 20 +
        (merged["wyckoff_phase"] == 3).astype(int) * 20 +
        (merged["bos_up"] > 0).astype(int) * 20 +
        (merged["fvg_bull"] > 0).astype(int) * 20 +
        (merged["ob_bull"] > 0).astype(int) * 20
    ).clip(0, 100)

    if current_regime == "bull":
        merged["regime_alignment"] = merged["structure_score"]
    elif current_regime == "bear":
        merged["regime_alignment"] = 100 - merged["structure_score"]
    else:
        merged["regime_alignment"] = 50

    merged["wide_bullish_prob"] = merged["wide_bullish_prob"].fillna(0)
    merged["wide_bearish_prob"] = merged["wide_bearish_prob"].fillna(0)
    merged["net_directional"] = merged["wide_bullish_prob"] - merged["wide_bearish_prob"]

    # Composite: range probs + structure + directional bias
    merged["composite_score"] = (
        merged["hr_2pct_prob"] * 100 * 0.20 +
        merged["hr_5pct_prob"] * 100 * 0.20 +
        merged["hr_6pct_prob"] * 100 * 0.10 +
        merged["structure_score"] * 0.15 +
        merged["regime_alignment"] * 0.10 +
        (merged["net_directional"].clip(-1, 1) * 50 + 50) * 0.25
    )
    merged["composite_score"] = merged["composite_score"].rank(pct=True) * 100

    merged["confidence"] = np.where(
        merged["composite_score"] >= 80, "HIGH",
        np.where(merged["composite_score"] >= 50, "MEDIUM", "LOW")
    )

    merged["expected_range_pct"] = (
        merged["hr_2pct_range"] * 0.5 +
        (merged["hr_5pct_prob"] * 5) * 0.3 +
        (merged["hr_6pct_prob"] * 6) * 0.2
    )

    # Store
    date_str = str(latest_date.date())
    con.execute(f"DELETE FROM discovery_scores WHERE timeframe='{timeframe}' AND datetime::DATE='{date_str}'")
    rows = []
    for _, r in merged.iterrows():
        rows.append((
            timeframe, latest_date, r["symbol"],
            float(r["hr_2pct_prob"]), float(r["hr_5pct_prob"]), float(r["hr_6pct_prob"]),
            float(r["expected_range_pct"]),
            float(r["structure_score"]), float(r["regime_alignment"]),
            float(r["wide_bullish_prob"]), float(r["wide_bearish_prob"]),
            float(r["net_directional"]),
            float(r["composite_score"]), r["confidence"]
        ))

    con.executemany(
        "INSERT INTO discovery_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )

    high = sum(1 for r in rows if r[13] == "HIGH")
    med = sum(1 for r in rows if r[13] == "MEDIUM")
    low_sum = sum(1 for r in rows if r[13] == "LOW")
    top5 = sorted(rows, key=lambda r: r[12], reverse=True)[:5]
    print(f"  Date: {latest_date.date()}")
    print(f"  Stocks scored: {len(rows)}")
    print(f"  HIGH: {high} | MEDIUM: {med} | LOW: {low_sum}")
    print(f"  Regime: {current_regime}")
    print(f"  Top 5:")
    for r in top5:
        bullish = r[9]; bearish = r[10]; net = r[11]
        dir_label = "BULL" if net > 0.1 else "BEAR" if net < -0.1 else "NEUT"
        print(f"    {r[2]:15s} score={r[12]:.0f} conf={r[13]}  >2%:{r[3]:.0%} >5%:{r[4]:.0%} "
              f"bull:{bullish:.0%} bear:{bearish:.0%} dir:{dir_label}")

    con.close()


def run(timeframes=["1day"]):
    for tf in timeframes:
        build_scores(tf)
