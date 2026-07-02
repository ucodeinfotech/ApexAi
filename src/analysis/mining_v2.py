"""
Phase 11v2 — Enhanced Pattern Mining.
Three modes: discriminative (fast) | XGBoost rule extraction (accurate) | Apriori (legacy).
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"

TECH_FEATURES = [
    "rsi_14", "rsi_7", "atr_14", "adx", "bb_pct_b", "bb_width",
    "vol_ratio_10", "vol_ratio_5", "ret_5d", "ret_10d", "ret_1d",
    "close_vs_sma_20", "close_vs_sma_50", "close_vs_sma_10",
    "macd_hist", "macd_line", "stoch_k", "stoch_d", "mfi",
    "obv", "williams_r", "cci", "cmf", "eom", "uo", "trix",
    "zscore_20", "swing_high", "swing_low",
    "dc_width", "kc_width", "hv_20", "skew_20", "kurt_20",
    "plus_di", "minus_di", "aroon_osc",
]

TARGET_DEFS = [
    ("ret_1d_oc", 0.85, "hg_1d_oc"),
]


def _compute_ret_1d_oc(con, tf):
    """Compute next-day open-to-close return per symbol via LEAD."""
    quoted = [f'"{c}"' for c in TECH_FEATURES]
    df = con.execute(f"""
        SELECT "datetime", "symbol",
               LEAD("close") OVER w / LEAD("open") OVER w - 1 AS ret_1d_oc,
               {', '.join(quoted)}
        FROM feature_store
        WHERE timeframe=? AND ret_5d IS NOT NULL
        WINDOW w AS (PARTITION BY symbol ORDER BY datetime)
        ORDER BY "datetime"
    """, [tf]).fetchdf()
    return df


def _quantile_bin(series, name, n_bins=3):
    try:
        labels = [f"{name}_LOW", f"{name}_MID", f"{name}_HIGH"]
        binned = pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")
        return binned.astype(str)
    except Exception:
        return pd.Series([f"{name}_MID"] * len(series), index=series.index)


def _discretize_features(df, features):
    bin_cols = []
    for col in features:
        if col not in df.columns:
            continue
        if df[col].nunique() < 3:
            continue
        try:
            bname = f"BIN_{col}"
            df[bname] = _quantile_bin(df[col], col)
            bin_cols.append(bname)
        except Exception:
            pass
    return bin_cols


def _encode_transactions(df, item_cols):
    encoded = pd.get_dummies(df[item_cols].astype(str), sparse=True)
    return encoded.astype(bool)


# ─────────────────────────────────────────────
#  Mode 1: Discriminative pattern mining (fast)
# ─────────────────────────────────────────────

def _discriminative_patterns(hg_items, non_hg_items):
    """Mine patterns that discriminate high-gainers from non-high-gainers."""
    hg_X = hg_items.astype(int).to_numpy()
    non_X = non_hg_items.astype(int).to_numpy()
    cols = list(hg_items.columns)
    ng = hg_X.shape[0]
    nn = non_X.shape[0]
    if ng < 10 or len(cols) < 2:
        return pd.DataFrame()

    hg_pct = hg_X.sum(axis=0) / ng
    non_pct = non_X.sum(axis=0) / nn
    lift_1 = np.divide(hg_pct, non_pct, out=np.ones_like(hg_pct, dtype=float), where=non_pct > 0.001)
    hg_cnt = hg_X.sum(axis=0)
    min_sup = max(3, int(ng * 0.02))
    active = np.where((hg_cnt >= min_sup) & (hg_pct > 0.01))[0]

    rows = []
    # Single-item patterns
    for i in active:
        if lift_1[i] > 1.5 and hg_pct[i] >= 0.02:
            rows.append({
                "antecedents": cols[i],
                "num_antecedents": 1,
                "support": hg_pct[i],
                "confidence": hg_pct[i],
                "lift": float(lift_1[i]),
                "net_conf": float(hg_pct[i] - non_pct[i]),
                "zhang": float((hg_pct[i] - non_pct[i]) / (max(hg_pct[i], non_pct[i]) + 1e-10)),
                "composite": float(lift_1[i] * 0.5 + hg_pct[i] * 0.3 + (1 + lift_1[i]) * 0.2),
            })

    # Pair patterns via co-occurrence
    hg_cooc = hg_X.T @ hg_X
    non_cooc = non_X.T @ non_X
    for i in active:
        for j in active:
            if i >= j:
                continue
            hg_ab = max(1, hg_cooc[i, j])
            non_ab = max(1, non_cooc[i, j])
            hg_p = hg_ab / ng
            non_p = non_ab / nn
            lift_pair = hg_p / non_p
            if lift_pair < 1.3 or hg_p < 0.01:
                continue
            conf = hg_ab / max(1, hg_X[:, i].sum())
            rows.append({
                "antecedents": f"{cols[i]}, {cols[j]}",
                "num_antecedents": 2,
                "support": hg_p,
                "confidence": conf,
                "lift": float(lift_pair),
                "net_conf": conf - non_p,
                "zhang": float((conf - non_p) / (max(conf, non_p) + 1e-10)),
                "composite": float(lift_pair * 0.35 + conf * 0.25 + hg_p * 0.2 + (1 + lift_pair) * 0.2),
            })

    if not rows:
        return pd.DataFrame()
    rules_df = pd.DataFrame(rows)
    print(f"      Singles: {len([r for r in rows if r['num_antecedents'] == 1])}, Pairs: {len(rows) - len([r for r in rows if r['num_antecedents'] == 1])}")
    return rules_df.sort_values("composite", ascending=False)


# ──────────────────────────────────────────────────────
#  Mode 2: XGBoost rule extraction (accurate for 1-day)
# ──────────────────────────────────────────────────────

def _xgb_rules_extract(model, feature_names, X_train, y_train,
                        min_precision=0.17, max_rules=5000):
    """Extract decision paths from XGBoost trees as rules."""
    import re
    from collections import defaultdict

    booster = model.get_booster()
    trees_str = booster.get_dump(with_stats=True)
    n_trees = len(trees_str)

    rule_map = defaultdict(lambda: {"count": 0, "cover_sum": 0})
    total_hg = y_train.sum()

    for tid, dump in enumerate(trees_str):
        if tid > 0 and tid % 50 == 0:
            print(f"      Processed {tid}/{n_trees} trees...")

        lines = dump.strip().split("\n")
        # Build node map from lines
        nodes = {}  # id -> (type, data)
        for line in lines:
            if not line.strip():
                continue
            # Depth = leading tabs
            stripped = line.lstrip("\t")
            depth = len(line) - len(stripped)

            # Extract node id
            m = re.match(r"(\d+):", stripped)
            if not m:
                continue
            node_id = int(m.group(1))
            rest = stripped[m.end():]

            if rest.startswith("leaf="):
                leaf_m = re.search(r"leaf=(-?[\d.]+)", rest)
                cover_m = re.search(r"cover=([\d.]+)", rest)
                leaf_val = float(leaf_m.group(1)) if leaf_m else 0
                cover = float(cover_m.group(1)) if cover_m else 0
                nodes[node_id] = ("leaf", leaf_val, cover, depth)
            else:
                # Decision node: [feature<value] yes=id,no=id,missing=id,...
                feat_m = re.search(r"\[(.+?)<([^\]]+)\]", rest)
                yes_m = re.search(r"yes=(\d+)", rest)
                no_m = re.search(r"no=(\d+)", rest)
                if feat_m and yes_m and no_m:
                    feat = feat_m.group(1).strip()
                    val = float(feat_m.group(2))
                    nodes[node_id] = ("split", feat, val,
                                      int(yes_m.group(1)),
                                      int(no_m.group(1)),
                                      depth)

        # DFS from root (id 0) to extract paths
        def dfs(nid, conditions):
            node = nodes.get(nid)
            if node is None:
                return
            if node[0] == "leaf":
                leaf_val = node[1]
                if leaf_val > 0 and conditions:
                    key = " AND ".join(sorted(conditions))
                    rule_map[key]["count"] += 1
                    rule_map[key]["cover_sum"] += node[2]
            else:
                _, feat, val, yes_id, no_id, _ = node
                # Yes branch: feature < value
                dfs(yes_id, conditions + [f"{feat} < {val}"])
                # No branch: feature >= value
                dfs(no_id, conditions + [f"{feat} >= {val}"])

        dfs(0, [])

    if not rule_map:
        return pd.DataFrame()

    print(f"      Evaluating {len(rule_map)} unique rule paths on training data...")
    results = []
    for rule_key, meta in rule_map.items():
        conds = rule_key.split(" AND ")
        mask = pd.Series(True, index=X_train.index)
        valid = True
        for c in conds:
            cm = re.match(r"(\w+(?:_\w+)*)\s*(<|>=)\s*(-?[\d.e+]+)", c)
            if not cm:
                valid = False
                break
            col, op, val_str = cm.group(1), cm.group(2), cm.group(3)
            if col not in X_train.columns:
                valid = False
                break
            val = float(val_str)
            if op == "<":
                mask &= X_train[col] < val
            else:
                mask &= X_train[col] >= val

        if not valid or mask.sum() < 5:
            continue

        tp = y_train[mask].sum()
        fp = mask.sum() - tp
        precision = tp / mask.sum()
        lift_val = precision / (total_hg / len(y_train))
        if precision >= min_precision:
            results.append({
                "antecedents": rule_key,
                "num_antecedents": len(conds),
                "support": mask.sum() / len(y_train),
                "confidence": precision,
                "lift": lift_val,
                "net_conf": precision - (total_hg / len(y_train)),
                "zhang": (precision - (total_hg / len(y_train))) / (max(precision, total_hg / len(y_train)) + 1e-10),
            })

    if not results:
        return pd.DataFrame()

    rules_df = pd.DataFrame(results)
    rules_df["composite"] = (
        rules_df["lift"] * 0.25 +
        rules_df["confidence"] * 0.25 +
        rules_df["support"] * 0.25 +
        rules_df["net_conf"].clip(0) * 0.25
    )
    print(f"      Extracted {len(rules_df)} valid rules (from {len(rule_map)} unique paths)")
    return rules_df.sort_values("composite", ascending=False).head(max_rules)


def _parse_condition(cond_str, feature_names):
    """Parse 'feat <= 0.5' into (feat, op, val)."""
    import re
    m = re.match(r"(\w+)\s*(<=|>=|<|>|==)\s*([-\d.e+]+)", cond_str)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return cond_str, "==", "0"


# ─────────────────────────────────────────────────────────────
#  Main entry point — choose mode: 'xgb' (default) or 'discriminative'
# ─────────────────────────────────────────────────────────────

def run_mining_v2(timeframes=None, max_rules=10000, mode="xgb"):
    """
    Enhanced pattern mining.

    Parameters
    ----------
    mode : str
        'xgb' — XGBoost rule extraction (best for 1-day, handles noise)
        'discriminative' — fast co-occurrence based (good for longer horizons)
    """
    if timeframes is None:
        timeframes = ["1day"]

    con = duckdb.connect(str(DB_PATH))

    # Refresh table schema
    con.execute("DROP TABLE IF EXISTS pattern_rules_v2")
    con.execute("""
        CREATE TABLE pattern_rules_v2 (
            timeframe VARCHAR,
            target_label VARCHAR,
            antecedents VARCHAR,
            consequents VARCHAR DEFAULT '(high_gainer)',
            support DOUBLE,
            confidence DOUBLE,
            lift DOUBLE,
            leverage DOUBLE,
            conviction DOUBLE DEFAULT 0,
            zhang DOUBLE,
            jaccard DOUBLE DEFAULT 0,
            net_conf DOUBLE,
            composite DOUBLE,
            antecedent_support DOUBLE DEFAULT 0,
            consequent_support DOUBLE DEFAULT 1,
            num_antecedents INT,
            is_anti_rule BOOLEAN DEFAULT FALSE,
            train_start DATE,
            train_end DATE,
            mined_at TIMESTAMP DEFAULT NOW()
        )
    """)

    for tf in timeframes:
        print(f"\n{'='*60}")
        print(f"  Mining {tf} — {mode} mode")
        print(f"{'='*60}")

        df = _compute_ret_1d_oc(con, tf)
        if len(df) < 500:
            print(f"  Skipping {tf}: only {len(df)} rows")
            continue

        print(f"  Loaded {len(df):,} rows ({df['symbol'].nunique()} symbols)")

        # Merge pattern occurrences (chart + candle)
        for cat in ("chart", "candle"):
            po = con.execute(f"""
                SELECT datetime, pattern FROM pattern_occurrences
                WHERE timeframe=? AND category=?
            """, [tf, cat]).fetchdf()
            if len(po) > 0:
                po["datetime"] = pd.to_datetime(po["datetime"])
                po_pivot = po.pivot_table(index="datetime", columns="pattern",
                                           aggfunc="size", fill_value=0)
                po_pivot = (po_pivot > 0).astype(int)
                df = df.merge(po_pivot, left_on="datetime", right_index=True,
                              how="left").fillna(0)

        for target_col, threshold, label in TARGET_DEFS:
            if target_col not in df.columns:
                continue

            print(f"\n  --- Target: {label} ---")

            t_val = df[target_col].quantile(threshold)
            df["_target"] = (df[target_col] >= t_val).astype(int)
            hg_count = df["_target"].sum()
            print(f"    High-gainers: {hg_count:,} / {len(df):,} ({hg_count/len(df)*100:.1f}%)")

            if hg_count < 50:
                print("    Skipping - too few high-gainers")
                continue

            # Split train/val temporally
            dates_sorted = df["datetime"].sort_values()
            split_idx = int(len(dates_sorted) * 0.8)
            train_end = dates_sorted.iloc[split_idx]
            train_mask = df["datetime"] <= train_end
            val_mask = df["datetime"] > train_end

            X_train = df[train_mask].copy()
            X_val = df[val_mask].copy()

            if mode == "xgb":
                rules_df = _run_xgb_mode(X_train, X_val, df, target_col, threshold,
                                          label, tf, tech_features=TECH_FEATURES)
            else:
                rules_df = _run_discriminative_mode(X_train, label, tf, df)

            # Store rules
            if len(rules_df) > 0:
                rules_df = rules_df.head(max_rules)
                db_rows = []
                for _, r in rules_df.iterrows():
                    db_rows.append((
                        tf, label,
                        r.get("antecedents", ""),
                        float(r.get("support", 0)),
                        float(r.get("confidence", 0)),
                        float(r.get("lift", 0)),
                        float(r.get("net_conf", 0)),
                        float(r.get("zhang", 0)),
                        float(r.get("composite", 0)),
                        int(r.get("num_antecedents", 1)),
                        df["datetime"].min().date(),
                        train_end.date(),
                    ))
                con.executemany("""
                    INSERT INTO pattern_rules_v2 (
                        timeframe, target_label, antecedents,
                        support, confidence, lift,
                        net_conf, zhang, composite, num_antecedents,
                        train_start, train_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, db_rows)
                print(f"    Stored {len(db_rows)} rules")
            else:
                print("    No rules found")

    con.close()
    print(f"\n{'='*60}")
    print(f"  Mining v2 ({mode} mode) complete!")
    print(f"{'='*60}")


def _run_discriminative_mode(X_train, label, tf, full_df):
    """Discriminative mode (original fast co-occurrence mining)."""
    from datetime import date

    print(f"    Mode: discriminative")
    target_col = "ret_1d_oc"  # only target for now

    t_val = X_train[target_col].quantile(0.85)
    X_train["_t"] = (X_train[target_col] >= t_val).astype(int)

    hg_df = X_train[X_train["_t"] == 1]
    non_hg_df = X_train[X_train["_t"] == 0]

    # Pattern columns (exclude computed cols)
    exclude = set(TECH_FEATURES + ["datetime", "symbol", "ret_1d_oc", "_target", "_t",
                                    "open", "high", "low", "close", "volume"])
    pattern_cols = [c for c in X_train.columns if c not in exclude]

    # Discretize
    bin_cols = _discretize_features(X_train, TECH_FEATURES)
    all_item_cols = bin_cols + pattern_cols

    hg_sample = hg_df.sample(n=min(len(hg_df), 20000), random_state=42)
    non_hg_sample = non_hg_df.sample(n=min(len(non_hg_df), 50000), random_state=42)

    hg_items = _encode_transactions(hg_sample, all_item_cols)
    non_hg_items = _encode_transactions(non_hg_sample, all_item_cols)

    all_cols = hg_items.columns.union(non_hg_items.columns)
    hg_items = hg_items.reindex(columns=all_cols, fill_value=False)
    non_hg_items = non_hg_items.reindex(columns=all_cols, fill_value=False)

    print(f"    Matrices: HG={hg_items.shape}, non-HG={non_hg_items.shape}")
    rules_df = _discriminative_patterns(hg_items, non_hg_items)
    return rules_df


def _run_xgb_mode(X_train, X_val, full_df, target_col, threshold, label, tf,
                   tech_features=None):
    """Train XGBoost and extract decision paths as rules."""
    if tech_features is None:
        tech_features = TECH_FEATURES

    import xgboost as xgb
    from sklearn.metrics import precision_score, recall_score

    print(f"    Mode: XGBoost rule extraction")

    pattern_cols = [c for c in X_train.columns
                    if c not in set(tech_features + ["datetime", "symbol", "ret_1d_oc",
                                                      "_target", "open", "high", "low",
                                                      "close", "volume"])]

    feature_cols = tech_features + pattern_cols
    feature_cols = [c for c in feature_cols if c in X_train.columns]

    # Compute target
    t = X_train[target_col].quantile(threshold)
    y_train = (X_train[target_col] >= t).astype(int)
    y_val = (X_val[target_col] >= t).astype(int) if len(X_val) > 0 else None

    X_train_feats = X_train[feature_cols].fillna(0)
    X_val_feats = X_val[feature_cols].fillna(0) if len(X_val) > 0 else None

    # Handle class imbalance
    scale_pos = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    print(f"    Training XGBoost on {X_train_feats.shape} with {y_train.sum()} positives...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train_feats, y_train,
            eval_set=[(X_train_feats, y_train)],
            verbose=False,
        )

    # Validation metrics
    if X_val_feats is not None and len(X_val_feats) > 0:
        y_pred = model.predict(X_val_feats)
        val_prec = precision_score(y_val, y_pred, zero_division=0)
        val_rec = recall_score(y_val, y_pred, zero_division=0)
        print(f"    Val precision={val_prec:.3f}, recall={val_rec:.3f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    top_feats = imp[imp["importance"] > 0.01]["feature"].tolist()
    print(f"    Top features ({len(top_feats)}/{len(feature_cols)} with >0.01 importance):")
    for _, r in imp.head(8).iterrows():
        print(f"      {r['feature']:<30s} {r['importance']:.4f}")

    # Extract rules only from top features (filtered model)
    # Re-train on just top features for cleaner rules
    # Actually, extract from the full model for best accuracy
    print(f"    Extracting decision rules from {model.n_estimators} trees...")

    rules_df = _xgb_rules_extract(
        model, feature_cols,
        X_train_feats, y_train,
        min_precision=0.17,  # slightly above baseline 15%
        max_rules=10000,
    )

    # Also add single-feature rules from top features
    if len(rules_df) > 0:
        single_rows = []
        for feat in top_feats[:15]:
            for direction in ("high", "low"):
                vals = X_train_feats[feat]
                if vals.nunique() < 5:
                    continue
                threshold_val = vals.quantile(0.33 if direction == "low" else 0.67)
                if direction == "low":
                    mask = X_train_feats[feat] <= threshold_val
                else:
                    mask = X_train_feats[feat] >= threshold_val

                if mask.sum() < 10:
                    continue
                p_hg = y_train[mask].mean()
                p_base = y_train.mean()
                if p_hg <= p_base:
                    continue
                lift_single = p_hg / p_base if p_base > 0 else 0
                single_rows.append({
                    "antecedents": f"{feat}_{direction}",
                    "num_antecedents": 1,
                    "support": mask.sum() / len(y_train),
                    "confidence": p_hg,
                    "lift": lift_single,
                    "net_conf": p_hg - p_base,
                    "zhang": (p_hg - p_base) / (max(p_hg, p_base) + 1e-10),
                    "composite": lift_single * 0.4 + p_hg * 0.3 + (mask.sum() / len(y_train)) * 0.3,
                })

        if single_rows:
            single_df = pd.DataFrame(single_rows)
            rules_df = pd.concat([rules_df, single_df], ignore_index=True)

    return rules_df


if __name__ == "__main__":
    run_mining_v2(timeframes=["1day"], mode="xgb")
