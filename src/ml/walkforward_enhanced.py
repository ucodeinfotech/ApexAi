"""
Phase 14d — Enhanced Walk-Forward with range features, calendar, and hyperparameter tuning.
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
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml.models import FEATURE_COLS

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"
TARGETS = {"hr_2pct": 0.02, "hr_5pct": 0.05, "hr_6pct": 0.06}


def compute_features(df):
    """Add range features, calendar features, and combine with existing FEATURE_COLS."""
    result = df.copy()

    result["today_range_pct"] = (result["high"] - result["low"]) / result["close"] * 100

    # Compute per-symbol rolling features
    result = result.sort_values(["symbol", "datetime"])
    result["range_ma_5"] = result.groupby("symbol")["today_range_pct"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()
    )
    result["range_ma_10"] = result.groupby("symbol")["today_range_pct"].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    result["range_ma_20"] = result.groupby("symbol")["today_range_pct"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    result["range_ratio_1d"] = result.groupby("symbol")["today_range_pct"].transform(
        lambda x: x / x.shift(1).clip(lower=0.01)
    )
    result["range_ratio_5d"] = result.groupby("symbol")["today_range_pct"].transform(
        lambda x: x / x.rolling(5, min_periods=2).mean().clip(lower=0.01)
    )
    result["atr_ratio"] = result["atr_14"] / result["close"] * 100

    # Calendar features
    dt = pd.to_datetime(result["datetime"])
    result["day_of_week"] = dt.dt.dayofweek
    result["is_thursday"] = (dt.dt.dayofweek == 3).astype(int)
    result["is_monthend"] = dt.dt.is_month_end.astype(int)
    result["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    result["month"] = dt.dt.month
    result["day_of_month"] = dt.dt.day

    return result


ENHANCED_FEATURES = FEATURE_COLS + [
    "today_range_pct", "range_ma_5", "range_ma_10", "range_ma_20",
    "range_ratio_1d", "range_ratio_5d", "atr_ratio",
    "day_of_week", "is_thursday", "is_monthend", "is_quarter_end",
    "month", "day_of_month",
    "vix_close", "vix_change", "vix_ratio_5", "vix_zscore_20", "vix_range",
    "delivery_pct", "del_ma_5", "del_ma_20", "del_change"
]


def tune_hyperparams(X_train, y_train, n_splits=3):
    """Grid search with time-series CV to find best params."""
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "n_estimators": [80, 120],
    }
    best_auc = 0
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        params["eval_metric"] = "logloss"
        params["random_state"] = 42
        params["n_jobs"] = 1
        cv_aucs = []
        for train_idx, val_idx in tscv.split(X_train):
            try:
                clf = xgb.XGBClassifier(**params)
                clf.fit(X_train[train_idx], y_train[train_idx])
                pred = clf.predict_proba(X_train[val_idx])[:, 1]
                auc = roc_auc_score(y_train[val_idx], pred)
                cv_aucs.append(auc)
            except:
                continue
        if cv_aucs:
            mean_auc = np.mean(cv_aucs)
            results.append((mean_auc, params))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params

    results.sort(key=lambda x: -x[0])
    print(f"\n    Grid search: {len(results)} combos tested (CV {n_splits}-fold)")
    print(f"    Best CV AUC: {best_auc:.4f}")
    if results:
        print(f"    Best params: {best_params}")
    if best_params is None:
        best_params = {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 100}
    return best_params


def run_walkforward(timeframe="1day"):
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions_oos (
            timeframe VARCHAR, model_name VARCHAR, symbol VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE, score DOUBLE, expected_return DOUBLE
        )
    """)

    print("Loading features + raw data...")
    cols_sql = ','.join(['f.'+c for c in FEATURE_COLS])
    df = con.execute(f"""
        SELECT symbol, datetime, {cols_sql}, f.high, f.low, f.close
        FROM feature_store f
        WHERE f.timeframe=? AND f.close IS NOT NULL
        ORDER BY f.datetime
    """, [timeframe]).fetchdf()
    print(f"  Rows: {len(df)}, Symbols: {df['symbol'].nunique()}")

    # Load VIX data
    vix_df = con.execute("""
        SELECT datetime, vix_close, vix_change, vix_ratio_5, vix_zscore_20, vix_range
        FROM vix_data ORDER BY datetime
    """).fetchdf()
    vix_df["datetime"] = pd.to_datetime(vix_df["datetime"]).dt.tz_localize("Asia/Calcutta")
    vix_df["datetime"] = vix_df["datetime"].astype("datetime64[us, Asia/Calcutta]")

    df["datetime"] = pd.to_datetime(df["datetime"])
    print("  Computing enhanced features...")
    df = compute_features(df)

    # Merge VIX (forward-fill for non-trading days)
    print("  Merging VIX data...")
    df = df.sort_values("datetime")
    vix_df = vix_df.sort_values("datetime")
    df = pd.merge_asof(df, vix_df, on="datetime", direction="backward")

    # Merge delivery data
    print("  Merging delivery data...")
    del_df = con.execute("""
        SELECT symbol, date, delivery_pct
        FROM delivery_data ORDER BY symbol, date
    """).fetchdf()
    del_df["date"] = pd.to_datetime(del_df["date"])
    del_df["date_ts"] = del_df["date"].dt.tz_localize("Asia/Calcutta").astype("datetime64[us, Asia/Calcutta]")
    # Add delivery rolling features
    del_df["del_ma_5"] = del_df.groupby("symbol")["delivery_pct"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()
    )
    del_df["del_ma_20"] = del_df.groupby("symbol")["delivery_pct"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    del_df["del_change"] = del_df.groupby("symbol")["delivery_pct"].transform(
        lambda x: x - x.rolling(5, min_periods=2).mean()
    )
    # Merge per symbol using merge_asof on datetime
    df = df.sort_values(["symbol", "datetime"])
    del_df = del_df.sort_values(["symbol", "date_ts"])
    dfs = []
    for sym in df["symbol"].unique():
        stock_df = df[df["symbol"] == sym].sort_values("datetime")
        stock_del = del_df[del_df["symbol"] == sym].sort_values("date_ts")
        if len(stock_del) > 0:
            stock_df = pd.merge_asof(stock_df, stock_del[["date_ts", "delivery_pct", "del_ma_5", "del_ma_20", "del_change"]],
                                      left_on="datetime", right_on="date_ts", direction="backward")
        else:
            stock_df["delivery_pct"] = 50.0
            stock_df["del_ma_5"] = 50.0
            stock_df["del_ma_20"] = 50.0
            stock_df["del_change"] = 0.0
        dfs.append(stock_df)
    df = pd.concat(dfs)
    print(f"  After delivery merge: {len(df)} rows")

    # Forward range targets
    next_high = df.groupby("symbol")["high"].shift(-1)
    next_low = df.groupby("symbol")["low"].shift(-1)
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["fwd_range_pct"] = (next_high - next_low) / next_close * 100
    for name, threshold in TARGETS.items():
        df[f"target_{name}"] = (df["fwd_range_pct"] > threshold * 100).astype(int)

    df["year"] = df["datetime"].dt.year
    df_model = df.dropna(subset=ENHANCED_FEATURES + [f"target_{t}" for t in TARGETS]).copy()
    years = sorted(df_model["year"].unique())
    windows = [(years[:i], years[i]) for i in range(4, len(years))]
    print(f"  Windows: {len(windows)}, Years: {years[0]}-{years[-1]}, Rows: {len(df_model)}")

    all_preds = {t: [] for t in TARGETS}
    tuned_params = None

    for wi, (train_years, test_year) in enumerate(windows):
        train = df_model[df_model["year"].isin(train_years)]
        test = df_model[df_model["year"] == test_year]
        if len(test) == 0:
            continue

        print(f"\n  [{wi+1}/{len(windows)}] Train {train_years[0]}-{train_years[-1]} -> Test {test_year} ({len(test)} rows)")

        X_train = train[ENHANCED_FEATURES].values
        X_test = test[ENHANCED_FEATURES].values
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Hyperparameter tuning on first window only
        if tuned_params is None:
            print("  Tuning hyperparameters...")
            tuned_params = tune_hyperparams(X_train_s, train[f"target_hr_2pct"].values)
            print(f"\n  Using params: {tuned_params}")

        for name, threshold in TARGETS.items():
            y_train = train[f"target_{name}"].values
            y_test = test[f"target_{name}"].values
            hg_rate = y_train.mean()

            clf = xgb.XGBClassifier(**tuned_params)
            clf.fit(X_train_s, y_train)
            pred_prob = clf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, pred_prob) if len(np.unique(y_test)) > 1 else 0

            reg = xgb.XGBRegressor(
                n_estimators=80, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1
            )
            reg.fit(X_train_s, train["fwd_range_pct"].values)
            pred_ret = reg.predict(X_test_s)
            mae = float(np.mean(np.abs(pred_ret - test["fwd_range_pct"].values)))
            corr = float(np.corrcoef(pred_ret, test["fwd_range_pct"].values)[0, 1]) if len(pred_ret) > 2 else 0

            print(f"    {name}: HG={hg_rate:.1%} AUC={auc:.4f} MAE={mae:.2f}% Corr={corr:.4f}")

            for idx, (_, r) in enumerate(test.iterrows()):
                all_preds[name].append(
                    (timeframe, f"xgb_range_enh_{name}", r["symbol"], r["datetime"],
                     float(pred_prob[idx]), float(pred_ret[idx]))
                )

    # Store predictions
    con.execute("DELETE FROM ml_predictions_oos WHERE timeframe=?", [timeframe])
    for name in TARGETS:
        if all_preds[name]:
            con.executemany("INSERT INTO ml_predictions_oos VALUES (?, ?, ?, ?, ?, ?)", all_preds[name])
            print(f"\n  Stored {len(all_preds[name])} enh predictions for {name}")

    if tuned_params:
        print(f"\n  Final tuned params: {tuned_params}")

    # Summary
    summary = pd.DataFrame(all_preds["hr_2pct"], columns=["tf","model","symbol","datetime","score","exp_ret"])
    summary["year"] = pd.to_datetime(summary["datetime"]).dt.year
    print(f"\n  Predictions per year:")
    for yr in sorted(summary["year"].unique()):
        print(f"    {yr}: {len(summary[summary['year']==yr])}")

    con.close()


if __name__ == "__main__":
    run_walkforward("1day")
