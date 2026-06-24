"""
Phase 14b — True Walk-Forward ML Training.
Trains on expanding annual windows, predicts out-of-sample on next year.
No lookahead bias — test predictions are genuinely unseen.
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
from src.ml.models import FEATURE_COLS, MODEL_DIR

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def run_walkforward(timeframe="1day"):
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions_oos (
            timeframe VARCHAR,
            model_name VARCHAR,
            symbol VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            score DOUBLE,
            expected_return DOUBLE
        )
    """)

    print("  Loading features...")
    cols_sql = ','.join(['f.'+c for c in FEATURE_COLS])
    df = con.execute(f"""
        SELECT symbol, datetime, {cols_sql}, close
        FROM feature_store f
        WHERE timeframe=? AND close IS NOT NULL
        ORDER BY datetime
    """, [timeframe]).fetchdf()
    print(f"  Rows: {len(df)}, Symbols: {df['symbol'].nunique()}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["fwd_ret_10d"] = df.groupby("symbol")["close"].transform(lambda x: x.shift(-10) / x - 1) * 100
    df["year"] = df["datetime"].dt.year

    df_model = df.dropna(subset=FEATURE_COLS + ["fwd_ret_10d"]).copy()
    years = sorted(df_model["year"].unique())
    print(f"  Years: {years[0]}-{years[-1]}")

    windows = []
    for i in range(4, len(years)):
        windows.append((years[:i], years[i]))

    print(f"  Walk-forward windows: {len(windows)}")
    all_preds = []

    for train_years, test_year in windows:
        print(f"\n  Window: train {train_years[0]}-{train_years[-1]} -> test {test_year}")

        train = df_model[df_model["year"].isin(train_years)]
        test = df_model[df_model["year"] == test_year]

        if len(test) == 0:
            print(f"    No test data for {test_year}, skipping")
            continue

        thresh = train["fwd_ret_10d"].quantile(0.85)
        train_t = train.copy()
        train_t["target"] = (train_t["fwd_ret_10d"] >= thresh).astype(int)
        test_t = test.copy()
        test_t["target"] = (test_t["fwd_ret_10d"] >= thresh).astype(int)
        print(f"    Train: {len(train_t)} rows, HG: {train_t['target'].mean():.1%} thr={thresh:.2f}%")
        print(f"    Test:  {len(test_t)} rows,  HG: {test_t['target'].mean():.1%}")

        X_train = train_t[FEATURE_COLS].values
        y_train = train_t["target"].values
        X_test = test_t[FEATURE_COLS].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=1
        )
        clf.fit(X_train_s, y_train)
        pred_prob = clf.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(test_t["target"].values, pred_prob) if len(np.unique(test_t["target"])) > 1 else 0
        print(f"    AUC: {auc:.4f}")

        reg = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1
        )
        reg.fit(X_train_s, train_t["fwd_ret_10d"].values)
        pred_ret = reg.predict(X_test_s)
        mae = float(np.mean(np.abs(pred_ret - test_t["fwd_ret_10d"].values)))
        corr = float(np.corrcoef(pred_ret, test_t["fwd_ret_10d"].values)[0, 1]) if len(pred_ret) > 2 else 0.0
        print(f"    MAE: {mae:.2f}%, Corr: {corr:.4f}")

        for idx, (_, r) in enumerate(test_t.iterrows()):
            all_preds.append((timeframe, "xgb_classifier_wf", r["symbol"], r["datetime"],
                              float(pred_prob[idx]), float(pred_ret[idx])))

    if len(all_preds) == 0:
        print("  No predictions generated")
        con.close()
        return

    con.execute("DELETE FROM ml_predictions_oos WHERE timeframe=?", [timeframe])
    con.executemany(
        "INSERT INTO ml_predictions_oos VALUES (?, ?, ?, ?, ?, ?)",
        all_preds
    )
    print(f"\n  Stored {len(all_preds)} OOS predictions")

    summary = pd.DataFrame(all_preds, columns=["tf", "model", "symbol", "datetime", "score", "exp_ret"])
    summary["year"] = pd.to_datetime(summary["datetime"]).dt.year
    for yr in sorted(summary["year"].unique()):
        yr_data = summary[summary["year"] == yr]
        print(f"    {yr}: {len(yr_data)} predictions")

    con.close()


if __name__ == "__main__":
    run_walkforward("1day")
