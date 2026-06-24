"""
ML Training Pipeline — trains XGBoost + LSTM models, stores predictions + explanations.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import torch

from src.ml.models import HighGainerModel, train_lstm, FEATURE_COLS, MODEL_DIR

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def _load_data(con, timeframe="1day", limit=200000):
    """Load feature data with forward returns as target."""
    df = con.execute(f"""
        SELECT fs.* FROM feature_store fs
        WHERE fs.timeframe=? AND fs.ret_10d IS NOT NULL
        ORDER BY fs.datetime
        LIMIT {limit}
    """, [timeframe]).fetchdf()
    return df


def train_models(timeframe="1day"):
    """Train XGBoost classifier + regressor, and LSTM."""
    con = duckdb.connect(str(DB_PATH))

    # Create predictions table
    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            timeframe VARCHAR, symbol VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            model_name VARCHAR,
            score DOUBLE, probability DOUBLE,
            prediction INT, actual INT,
            expected_return DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_feature_importance (
            timeframe VARCHAR, model_name VARCHAR,
            feature VARCHAR, importance DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_model_metrics (
            timeframe VARCHAR, model_name VARCHAR,
            metric VARCHAR, value DOUBLE
        )
    """)

    print(f"  Loading data ({timeframe})...")
    df = _load_data(con, timeframe)

    if len(df) < 500:
        print(f"  Insufficient data: {len(df)} rows")
        con.close()
        return

    # Define target: top 15% of FORWARD 10-day returns
    df = df.sort_values(["symbol", "datetime"])
    df["fwd_ret_10d"] = df.groupby("symbol")["close"].transform(lambda x: x.shift(-10) / x - 1) * 100
    threshold = df["fwd_ret_10d"].quantile(0.85)
    df["target"] = (df["fwd_ret_10d"] >= threshold).astype(int)
    print(f"  High-gainers: {df['target'].sum()} / {len(df)} ({df['target'].mean()*100:.1f}%)")

    # ── XGBoost Classifier ──
    print("  Training XGBoost classifier...")
    clf = HighGainerModel("xgb_classifier")
    X_tr, X_te, y_tr, y_te = clf.prepare_data(df, "target")
    metrics_clf = clf.train_xgb_classifier(X_tr, y_tr, X_te, y_te)
    clf_path = clf.save("xgb_classifier_1day")
    print(f"    AUC: {metrics_clf['auc']:.4f}")

    # Feature importance
    imp = clf.model.feature_importances_
    imp_rows = [(timeframe, "xgb_classifier", clf.feature_cols[i], float(imp[i]))
                for i in range(len(imp))]
    con.executemany(
        "INSERT INTO ml_feature_importance VALUES (?, ?, ?, ?)", imp_rows
    )

    # Store metrics
    con.execute(
        "INSERT INTO ml_model_metrics VALUES (?, ?, ?, ?)",
        [timeframe, "xgb_classifier", "auc", float(metrics_clf["auc"])]
    )

    # ── XGBoost Regressor ──
    print("  Training XGBoost regressor...")
    reg = HighGainerModel("xgb_regressor")
    X_tr_r, X_te_r, y_tr_r, y_te_r = reg.prepare_data(df, "fwd_ret_10d")
    metrics_reg = reg.train_xgb_regressor(X_tr_r, y_tr_r, X_te_r, y_te_r)
    reg.save("xgb_regressor_1day")
    print(f"    MAE: {metrics_reg['mae']:.4f}, Corr: {metrics_reg['corr']:.4f}")

    con.execute(
        "INSERT INTO ml_model_metrics VALUES (?, ?, ?, ?)",
        [timeframe, "xgb_regressor", "mae", float(metrics_reg["mae"])]
    )
    con.execute(
        "INSERT INTO ml_model_metrics VALUES (?, ?, ?, ?)",
        [timeframe, "xgb_regressor", "corr", float(metrics_reg["corr"])]
    )

    # ── Make predictions on full dataset ──
    print("  Generating predictions...")
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X_all = clf.scaler.transform(df[avail].fillna(0).values)

    # Classifier scores
    clf_probs = clf.model.predict_proba(X_all)[:, 1]
    clf_preds = (clf_probs > 0.5).astype(int)

    # Regressor predictions
    X_all_r = reg.scaler.transform(df[avail].fillna(0).values)
    reg_preds = reg.model.predict(X_all_r)

    # Store
    pred_rows = []
    for i in range(len(df)):
        pred_rows.append((
            timeframe, df.iloc[i]["symbol"],
            pd.Timestamp(df.iloc[i]["datetime"]),
            "xgb_classifier", float(clf_probs[i]),
            float(clf_probs[i]), int(clf_preds[i]),
            int(df.iloc[i]["target"]), float(reg_preds[i])
        ))
    con.executemany(
        "INSERT INTO ml_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        pred_rows
    )

    # ── LSTM ──
    print("  Training LSTM...")
    try:
        lstm_model, lstm_metrics = train_lstm(df.iloc[:min(len(df), 50000)], "target", seq_len=20, epochs=10)
        torch_path = str(MODEL_DIR / "lstm_1day.pt")
        torch.save(lstm_model.state_dict(), torch_path)
        print(f"    AUC: {lstm_metrics['auc']:.4f}")
        con.execute(
            "INSERT INTO ml_model_metrics VALUES (?, ?, ?, ?)",
            [timeframe, "lstm", "auc", float(lstm_metrics["auc"])]
        )
    except Exception as e:
        print(f"    LSTM failed: {e}")

    con.close()
    print(f"  Models saved to {BASE_DIR / 'warehouse' / 'models'}")


def run_all(timeframes=["1day"]):
    for tf in timeframes:
        print(f"\nTraining models on {tf}...")
        train_models(tf)


if __name__ == "__main__":
    run_all()
