"""
Phase 10 — Statistical Analysis: distributions, correlations, significance tests.
Results are stored in DuckDB analysis_results table.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_classif

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"
TABLE = "analysis_results"


def _ensure_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            analysis_type VARCHAR, timeframe VARCHAR, symbol VARCHAR,
            feature_name VARCHAR, metric VARCHAR, value DOUBLE
        )
    """)


def run_feature_distributions(timeframes=["1day"]):
    """Compute distribution stats for all features."""
    con = duckdb.connect(str(DB_PATH))
    _ensure_table(con)
    con.execute(f"DELETE FROM {TABLE} WHERE analysis_type='distribution'")

    # Get numeric feature columns from feature_store
    cols = [r[1] for r in con.execute("PRAGMA table_info('feature_store')").fetchall()]
    feature_cols = [c for c in cols if c not in ("symbol", "timeframe", "datetime",
                    "open", "high", "low", "close", "volume")]

    for tf in timeframes:
        for col in feature_cols:
            data = con.execute(
                f"SELECT \"{col}\" FROM feature_store WHERE timeframe=? AND \"{col}\" IS NOT NULL",
                [tf]
            ).fetchdf()[col].dropna().values

            if len(data) < 50:
                continue

            rows = [
                (tf, col, "mean", float(np.mean(data))),
                (tf, col, "std", float(np.std(data))),
                (tf, col, "skew", float(sp_stats.skew(data))),
                (tf, col, "kurtosis", float(sp_stats.kurtosis(data))),
                (tf, col, "min", float(np.min(data))),
                (tf, col, "max", float(np.max(data))),
                (tf, col, "p25", float(np.percentile(data, 25))),
                (tf, col, "p50", float(np.percentile(data, 50))),
                (tf, col, "p75", float(np.percentile(data, 75))),
                (tf, col, "count", float(len(data))),
            ]
            con.executemany(
                f"INSERT INTO {TABLE} (analysis_type, timeframe, feature_name, metric, value) "
                "VALUES ('distribution', ?, ?, ?, ?)", rows
            )

        print(f"  Distributions done for {tf}")

    con.close()


def run_correlations(timeframes=["1day"], methods=["pearson", "spearman"]):
    """Compute inter-feature correlations."""
    con = duckdb.connect(str(DB_PATH))
    _ensure_table(con)
    con.execute(f"DELETE FROM {TABLE} WHERE analysis_type='correlation'")

    for tf in timeframes:
        df = con.execute(
            "SELECT * FROM feature_store WHERE timeframe=? ORDER BY datetime LIMIT 50000",
            [tf]
        ).fetchdf()

        if len(df) < 100:
            continue

        # Select numeric feature columns
        feat_cols = [c for c in df.columns if c not in ("symbol", "timeframe", "datetime",
                      "open", "high", "low", "close", "volume")]
        df_feat = df[feat_cols].select_dtypes(include=[np.number]).dropna(axis=1, how="all")

        for method in methods:
            corr_func = df_feat.corr(method=method)
            rows = []
            for i, c1 in enumerate(corr_func.columns):
                for c2 in corr_func.columns[i + 1:]:
                    v = corr_func.loc[c1, c2]
                    if not np.isnan(v) and abs(v) > 0.3:
                        rows.append((tf, f"{c1}__{c2}", method, float(v)))
            if rows:
                # Sample to avoid too many rows
                if len(rows) > 5000:
                    rows = sorted(rows, key=lambda r: abs(r[3]), reverse=True)[:5000]
                con.executemany(
                    f"INSERT INTO {TABLE} (analysis_type, timeframe, feature_name, metric, value) "
                    "VALUES ('correlation', ?, ?, ?, ?)", rows
                )

        print(f"  Correlations done for {tf} ({len(df_feat.columns)} features)")

    con.close()


def run_significance_tests(timeframes=["1day"]):
    """
    T-tests and mutual information between features and forward returns.
    High-gainer defined as top 10% returns over next 5 days.
    """
    con = duckdb.connect(str(DB_PATH))
    _ensure_table(con)
    con.execute(f"DELETE FROM {TABLE} WHERE analysis_type='significance'")

    for tf in timeframes:
        df = con.execute(
            "SELECT * FROM feature_store WHERE timeframe=? ORDER BY datetime LIMIT 50000",
            [tf]
        ).fetchdf()

        if len(df) < 200:
            continue

        feat_cols = [c for c in df.columns if c not in ("symbol", "timeframe", "datetime",
                      "open", "high", "low", "close", "volume")]
        df_num = df[feat_cols].select_dtypes(include=[np.number]).dropna(how="all")

        if "ret_5d" not in df_num.columns:
            print(f"  No ret_5d in {tf}, skipping significance")
            continue

        # Define high-gainer: top 10% of forward returns
        threshold = df_num["ret_5d"].quantile(0.90)
        y = (df_num["ret_5d"] >= threshold).astype(int)

        rows = []
        for col in feat_cols:
            if col == "ret_5d" or col not in df_num.columns:
                continue
            X = df_num[col].dropna()
            if len(X) < 100:
                continue

            # T-test between high-gainer and non-high-gainer groups
            mask = y.loc[X.index]
            group_high = X[mask == 1]
            group_low = X[mask == 0]

            if len(group_high) < 10 or len(group_low) < 10:
                continue

            t_stat, p_val = sp_stats.ttest_ind(group_high, group_low, equal_var=False)
            if not np.isnan(p_val):
                rows.append((tf, col, "t_stat", float(t_stat)))
                rows.append((tf, col, "p_value", float(p_val)))

                # Effect size (Cohen's d)
                n1, n2 = len(group_high), len(group_low)
                s1, s2 = group_high.std(), group_low.std()
                pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
                d = (group_high.mean() - group_low.mean()) / pooled if pooled > 0 else 0
                rows.append((tf, col, "cohens_d", float(d)))

        if rows:
            # Keep most significant
            if len(rows) > 10000:
                rows = sorted(rows, key=lambda r: abs(r[3]) if r[2] in ("t_stat", "cohens_d") else 0,
                              reverse=True)[:10000]
            con.executemany(
                f"INSERT INTO {TABLE} (analysis_type, timeframe, feature_name, metric, value) "
                "VALUES ('significance', ?, ?, ?, ?)", rows
            )

        print(f"  Significance tests done for {tf}")

    con.close()


def run_all(timeframes=["1day"]):
    print("Statistical Analysis (Phase 10)...")
    print("  Distributions...")
    run_feature_distributions(timeframes)
    print("  Correlations...")
    run_correlations(timeframes)
    print("  Significance tests...")
    run_significance_tests(timeframes)
    print("  Done")
