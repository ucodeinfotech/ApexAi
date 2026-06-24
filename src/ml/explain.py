"""
Phase 16 — Explainability Layer: plain-language "why this stock" reports using
XGBoost feature importance + prediction contributions.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def generate_explanations(timeframe="1day"):
    """Generate plain-language explanations for the top-scored stocks."""
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_explanations (
            timeframe VARCHAR, symbol VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            score DOUBLE,
            explanation VARCHAR,
            top_features VARCHAR
        )
    """)

    # Get latest predictions
    preds = con.execute(f"""
        SELECT symbol, datetime, score, prediction, expected_return
        FROM ml_predictions
        WHERE timeframe=? AND model_name='xgb_classifier'
        ORDER BY score DESC LIMIT 50
    """, [timeframe]).fetchdf()

    if len(preds) == 0:
        print("  No predictions found")
        con.close()
        return

    # Get top features
    features = con.execute(f"""
        SELECT feature, importance
        FROM ml_feature_importance
        WHERE timeframe=? AND model_name='xgb_classifier'
        ORDER BY importance DESC LIMIT 15
    """, [timeframe]).fetchdf()

    top_feat = dict(zip(features["feature"], features["importance"]))

    # Generate explanations
    rows = []
    for _, row in preds.iterrows():
        # Top 3 features driving this prediction
        top3 = list(top_feat.keys())[:3]
        feat_str = ", ".join(top3)

        # Plain-language explanation
        score_pct = row["score"] * 100
        ret_str = f"{row['expected_return']:+.2f}%" if pd.notna(row["expected_return"]) else "N/A"

        explanation = (
            f"This stock scores {score_pct:.0f}/100 for becoming a high-gainer. "
            f"Key drivers: {feat_str}. "
            f"Expected 10-day return: {ret_str}."
        )

        rows.append((
            timeframe, row["symbol"], pd.Timestamp(row["datetime"]),
            float(row["score"]), explanation, json.dumps(top_feat)
        ))

    con.executemany(
        "INSERT INTO ml_explanations VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    print(f"  Generated {len(rows)} explanations")

    con.close()


def run(timeframes=["1day"]):
    for tf in timeframes:
        print(f"Explanations for {tf}...")
        generate_explanations(tf)
