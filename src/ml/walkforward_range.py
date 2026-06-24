"""
Phase 14c — Walk-Forward: predict daily high-low range > thresholds.
Targets: next day range/close > 2%, 5%, 6%
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml.models import FEATURE_COLS

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"

TARGETS = {
    "hr_2pct": 0.02,
    "hr_5pct": 0.05,
    "hr_6pct": 0.06,
}


def run_walkforward(timeframe="1day"):
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions_oos (
            timeframe VARCHAR, model_name VARCHAR, symbol VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE, score DOUBLE, expected_return DOUBLE
        )
    """)

    print("  Loading features...")
    cols_sql = ','.join(['f.'+c for c in FEATURE_COLS])
    df = con.execute(f"""
        SELECT symbol, datetime, {cols_sql}, high, low, close
        FROM feature_store f
        WHERE timeframe=? AND close IS NOT NULL
        ORDER BY datetime
    """, [timeframe]).fetchdf()
    print(f"  Rows: {len(df)}, Symbols: {df['symbol'].nunique()}")

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Forward range targets (predict NEXT candle's range)
    next_high = df.groupby("symbol")["high"].shift(-1)
    next_low = df.groupby("symbol")["low"].shift(-1)
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["fwd_range_pct"] = (next_high - next_low) / next_close * 100
    for name, threshold in TARGETS.items():
        df[f"target_{name}"] = (df["fwd_range_pct"] > threshold * 100).astype(int)

    df["year"] = df["datetime"].dt.year
    df_model = df.dropna(subset=FEATURE_COLS + [f"target_{t}" for t in TARGETS]).copy()
    years = sorted(df_model["year"].unique())
    print(f"  Years: {years[0]}-{years[-1]}")

    windows = [(years[:i], years[i]) for i in range(4, len(years))]
    print(f"  Windows: {len(windows)}")

    all_preds = {t: [] for t in TARGETS}

    for train_years, test_year in windows:
        train = df_model[df_model["year"].isin(train_years)]
        test = df_model[df_model["year"] == test_year]
        if len(test) == 0:
            continue

        print(f"\n  Train {train_years[0]}-{train_years[-1]} -> Test {test_year} ({len(test)} rows)")

        X_train = train[FEATURE_COLS].values
        X_test = test[FEATURE_COLS].values
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, threshold in TARGETS.items():
            y_train = train[f"target_{name}"].values
            y_test = test[f"target_{name}"].values
            hg_rate = y_train.mean()

            clf = xgb.XGBClassifier(
                n_estimators=80, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, n_jobs=1
            )
            clf.fit(X_train_s, y_train)
            pred_prob = clf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, pred_prob) if len(np.unique(y_test)) > 1 else 0

            # Regressor: predict exact range %
            reg = xgb.XGBRegressor(
                n_estimators=80, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1
            )
            train_range = train["fwd_range_pct"].values
            test_range = test["fwd_range_pct"].values
            reg.fit(X_train_s, train_range)
            pred_ret = reg.predict(X_test_s)
            mae = float(np.mean(np.abs(pred_ret - test_range)))
            corr = float(np.corrcoef(pred_ret, test_range)[0, 1]) if len(pred_ret) > 2 else 0.0

            print(f"    {name}: HG={hg_rate:.1%} AUC={auc:.4f} MAE={mae:.2f}% Corr={corr:.4f}")

            for idx, (_, r) in enumerate(test.iterrows()):
                all_preds[name].append(
                    (timeframe, f"xgb_range_{name}", r["symbol"], r["datetime"],
                     float(pred_prob[idx]), float(pred_ret[idx]))
                )

    # Store all predictions
    con.execute("DELETE FROM ml_predictions_oos WHERE timeframe=?", [timeframe])
    for name in TARGETS:
        if all_preds[name]:
            con.executemany("INSERT INTO ml_predictions_oos VALUES (?, ?, ?, ?, ?, ?)", all_preds[name])
            print(f"\n  Stored {len(all_preds[name])} predictions for {name}")

    con.close()


if __name__ == "__main__":
    run_walkforward("1day")
