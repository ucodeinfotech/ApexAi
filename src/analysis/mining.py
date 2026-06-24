"""
Phase 11 — Pattern Mining: frequent itemsets of feature conditions preceding high-gainers.
Uses Apriori-like approach on discretized features.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"

# Key features to discretize for pattern mining
KEY_FEATURES = [
    "rsi_14", "atr_14", "adx", "bb_pct_b", "bb_width",
    "vol_ratio_10", "ret_5d", "ret_10d", "close_vs_sma_20",
    "close_vs_sma_50", "rs_vs_market", "macd_hist",
    "stoch_k", "mfi", "obv", "vwap",
]


def _discretize(series, name, n_bins=3):
    """Convert numeric series to categorical bins (low/med/high)."""
    labels = [f"{name}_low", f"{name}_med", f"{name}_high"]
    return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")


def run_mining(timeframes=["1day"], min_support=0.02, min_confidence=0.5):
    """
    Mine frequent pattern combinations that precede high-gainer moves.
    High-gainer = top 10% of 5-day forward returns.
    """
    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS pattern_rules (
            timeframe VARCHAR,
            antecedent VARCHAR,
            consequent VARCHAR,
            support DOUBLE,
            confidence DOUBLE,
            lift DOUBLE,
            occurrence_count INT
        )
    """)

    for tf in timeframes:
        print(f"  Mining {tf}...")

        # Load features + forward returns + pattern occurrences
        df = con.execute(f"""
            SELECT fs.datetime, fs.rsi_14, fs.atr_14, fs.adx, fs.bb_pct_b, fs.bb_width,
                   fs.vol_ratio_10, fs.ret_5d, fs.ret_10d, fs.close_vs_sma_20,
                   fs.close_vs_sma_50, fs.macd_hist, fs.stoch_k, fs.mfi,
                   fs.ret_1d as ret_1d
            FROM feature_store fs
            WHERE fs.timeframe=? AND fs.ret_5d IS NOT NULL
            ORDER BY fs.datetime
        """, [tf]).fetchdf()

        if len(df) < 500:
            continue

        # Define high-gainer
        threshold = df["ret_10d"].quantile(0.85)
        df["is_high_gainer"] = (df["ret_10d"] >= threshold).astype(int)

        # Add candlestick patterns
        po = con.execute(f"""
            SELECT datetime, pattern
            FROM pattern_occurrences WHERE timeframe=? AND category='candle'
        """, [tf]).fetchdf()
        if len(po) > 0:
            po["datetime"] = pd.to_datetime(po["datetime"])
            # One row per datetime with pattern columns
            po_pivot = po.pivot_table(index="datetime", columns="pattern",
                                       aggfunc="size", fill_value=0)
            for col in po_pivot.columns:
                po_pivot[col] = (po_pivot[col] > 0).astype(int)
            df = df.merge(po_pivot, left_on="datetime", right_index=True, how="left").fillna(0)

        # Discretize continuous features
        feat_cols = [c for c in KEY_FEATURES if c in df.columns]
        for col in feat_cols:
            try:
                df[f"bin_{col}"] = _discretize(df[col], col)
            except Exception:
                pass

        # Build transaction database: only rows where is_high_gainer=1
        hg_df = df[df["is_high_gainer"] == 1]
        if len(hg_df) < 50:
            continue

        bin_cols = [f"bin_{c}" for c in feat_cols if f"bin_{c}" in hg_df.columns]
        pattern_cols = [c for c in po_pivot.columns if c in hg_df.columns]
        all_item_cols = bin_cols + pattern_cols

        # Extract itemsets
        itemset_counter = Counter()
        hg_indices = hg_df.index

        # Single items
        for col in all_item_cols:
            for val in hg_df[col].dropna().unique():
                item = f"{col}={val}" if not isinstance(val, str) else f"{col}=1"
                itemset_counter[(item,)] += 1

        # Pairs (for speed, sample if too many rows)
        hg_sample = hg_df.sample(min(len(hg_df), 2000)) if len(hg_df) > 2000 else hg_df
        for idx in hg_sample.index:
            row = hg_sample.loc[idx]
            items = []
            for col in all_item_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                item = f"{col}={val}" if not isinstance(val, str) else f"{col}=1"
                items.append(item)
            for a, b in combinations(sorted(items), 2):
                itemset_counter[(a, b)] += 1

        total_hg = len(hg_df)
        min_cnt = max(5, int(total_hg * min_support))

        # Generate rules
        rules = []
        for itemset, count in itemset_counter.most_common(2000):
            if count < min_cnt:
                continue

            if len(itemset) == 2:
                a, b = itemset
                support = count / total_hg
                conf_a_b = count / itemset_counter.get((a,), 1)
                conf_b_a = count / itemset_counter.get((b,), 1)

                # Lift
                p_a = itemset_counter.get((a,), 0) / total_hg
                p_b = itemset_counter.get((b,), 0) / total_hg
                lift = support / (p_a * p_b + 1e-10)

                if conf_a_b >= min_confidence and lift > 1.1:
                    rules.append((tf, a, b, support, conf_a_b, lift, count))
                if conf_b_a >= min_confidence and lift > 1.1:
                    rules.append((tf, b, a, support, conf_b_a, lift, count))

        if rules:
            con.executemany(
                "INSERT INTO pattern_rules VALUES (?, ?, ?, ?, ?, ?, ?)",
                rules[:5000]
            )
            print(f"    {len(rules)} rules found")

    con.close()
    print("  Mining done")
