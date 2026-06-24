"""
Phase 12 — Clustering: KMeans, DBSCAN, GMM to discover stock behavior types.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def run_clustering(timeframes=["1day"]):
    """Cluster stocks by their feature profiles."""
    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS cluster_labels (
            timeframe VARCHAR, symbol VARCHAR,
            datetime TIMESTAMP WITH TIME ZONE,
            cluster_kmeans INT, cluster_dbscan INT, cluster_gmm INT,
            cluster_label VARCHAR
        )
    """)

    cluster_features = [
        "rsi_14", "atr_14", "adx", "bb_pct_b", "bb_width",
        "vol_ratio_10", "ret_5d", "ret_10d", "close_vs_sma_20",
        "close_vs_sma_50", "mfi", "macd_hist",
    ]

    for tf in timeframes:
        print(f"  Clustering {tf}...")

        df = con.execute(f"""
            SELECT symbol, datetime, {','.join(f'"{c}"' for c in cluster_features)}
            FROM feature_store WHERE timeframe=? ORDER BY datetime
        """, [tf]).fetchdf()

        if len(df) < 500:
            continue

        # Aggregate per symbol to get average profile
        feat_cols = [c for c in cluster_features if c in df.columns]
        profile = df.groupby("symbol")[feat_cols].mean().dropna()
        symbols = profile.index.tolist()

        if len(profile) < 10:
            continue

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(profile.values)

        # KMeans (assume 4 behavior types)
        km = KMeans(n_clusters=min(4, len(X) - 1), random_state=42, n_init=10)
        km_labels = km.fit_predict(X)

        # DBSCAN
        db = DBSCAN(eps=1.5, min_samples=3)
        db_labels = db.fit_predict(X)

        # GMM
        gmm = GaussianMixture(n_components=min(4, len(X) - 1), random_state=42)
        gmm_labels = gmm.fit_predict(X)

        # Label each symbol
        label_map = {0: "momentum", 1: "breakout", 2: "mean_reversion", 3: "low_vol"}
        rows = []
        for i, sym in enumerate(symbols):
            km_l = int(km_labels[i])
            db_l = int(db_labels[i])
            gmm_l = int(gmm_labels[i])
            label = label_map.get(km_l, "other")
            rows.append((tf, sym, km_l, db_l, gmm_l, label))

        con.executemany(
            "INSERT INTO cluster_labels (timeframe, symbol, cluster_kmeans, cluster_dbscan, cluster_gmm, cluster_label) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows
        )

        # Print cluster stats
        for method, labels in [("KMeans", km_labels)]:
            counts = pd.Series(labels).value_counts().sort_index()
            print(f"    {method}: {dict(counts)}")

    con.close()
    print("  Clustering done")
