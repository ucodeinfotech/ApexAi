"""
Backtest pattern rules from pattern_rules_v2 on 1-day open-to-close returns.
Tests: per-rule precision, combined signal, time-segmented performance.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_rules(con, min_lift=1.3, min_conf=0.17, max_rules=200):
    """Load top rules from pattern_rules_v2."""
    rules = con.execute(f"""
        SELECT antecedents, lift, confidence, support, composite,
               num_antecedents
        FROM pattern_rules_v2
        WHERE lift >= ? AND confidence >= ?
        ORDER BY composite DESC
        LIMIT ?
    """, [min_lift, min_conf, max_rules]).fetchdf()
    return rules


def evaluate_rules(con, rules_df, feature_cols, pattern_cols):
    """Evaluate each rule on full dataset: precision, avg return, count."""

    # Load all features + compute target (next-day open-to-close per symbol)
    quoted = [f'fs."{c}"' for c in feature_cols]
    df = con.execute(f"""
        SELECT fs.datetime, fs.symbol,
               LEAD(fs."close") OVER w / LEAD(fs."open") OVER w - 1 AS ret_1d_oc,
               {', '.join(quoted)}
        FROM feature_store fs
        WHERE fs.timeframe='1day' AND fs.ret_5d IS NOT NULL
        WINDOW w AS (PARTITION BY fs.symbol ORDER BY fs.datetime)
        ORDER BY fs.datetime
    """).fetchdf()

    # Merge pattern occurrences
    po = con.execute("""
        SELECT datetime, pattern FROM pattern_occurrences
        WHERE timeframe='1day' AND category='chart'
    """).fetchdf()
    if len(po) > 0:
        po["datetime"] = pd.to_datetime(po["datetime"])
        po_pivot = po.pivot_table(index="datetime", columns="pattern",
                                   aggfunc="size", fill_value=0)
        po_pivot = (po_pivot > 0).astype(int)
        df = df.merge(po_pivot, left_on="datetime", right_index=True,
                      how="left").fillna(0)

    pc = con.execute("""
        SELECT datetime, pattern FROM pattern_occurrences
        WHERE timeframe='1day' AND category='candle'
    """).fetchdf()
    if len(pc) > 0:
        pc["datetime"] = pd.to_datetime(pc["datetime"])
        pc_pivot = pc.pivot_table(index="datetime", columns="pattern",
                                   aggfunc="size", fill_value=0)
        pc_pivot = (pc_pivot > 0).astype(int)
        df = df.merge(pc_pivot, left_on="datetime", right_index=True,
                      how="left").fillna(0)

    # Compute target: high-gainer = top 15% of ret_1d_oc
    ALL_COLS = set(feature_cols + pattern_cols)
    pct_85 = df["ret_1d_oc"].quantile(0.85)
    df["_target"] = (df["ret_1d_oc"] >= pct_85).astype(int)

    # Split into train/val temporally
    dates = df["datetime"].sort_values()
    split_idx = int(len(dates) * 0.8)
    split_date = dates.iloc[split_idx]
    train_mask = df["datetime"] <= split_date
    val_mask = df["datetime"] > split_date

    val_df = df[val_mask].copy()
    print(f"  Validation set: {len(val_df):,} rows, "
          f"{val_df['_target'].sum():,} high-gainers "
          f"({val_df['_target'].mean()*100:.1f}%)")

    results = []
    for _, rule in rules_df.iterrows():
        ant = rule["antecedents"]
        conditions = ant.split(" AND ")
        mask = _apply_conditions(val_df, conditions)
        if mask is None or mask.sum() < 2:
            continue

        tp = val_df.loc[mask, "_target"].sum()
        fp = mask.sum() - tp
        prec = tp / mask.sum() if mask.sum() > 0 else 0
        avg_ret = val_df.loc[mask, "ret_1d_oc"].mean()
        med_ret = val_df.loc[mask, "ret_1d_oc"].median()
        base_avg = val_df["ret_1d_oc"].mean()

        results.append({
            "antecedents": ant,
            "num_conds": rule["num_antecedents"],
            "n_signals": int(mask.sum()),
            "tp": int(tp), "fp": int(fp),
            "precision": prec,
            "avg_ret": avg_ret,
            "med_ret": med_ret,
            "base_avg_ret": base_avg,
            "lift_vs_base": avg_ret / base_avg if base_avg != 0 else 0,
            "in_lift": rule["lift"],
        })

    return pd.DataFrame(results)


def _apply_conditions(df, conditions):
    """Apply list of 'feat < val' or 'feat >= val' conditions to df, return bool mask."""
    mask = pd.Series(True, index=df.index)
    for c in conditions:
        m = re.match(r"(\w+(?:_\w+)*)\s*(<|>=)\s*(-?[\d.e+]+)", c)
        if not m:
            return None
        col, op, val_str = m.group(1), m.group(2), m.group(3)
        if col not in df.columns:
            return None
        val = float(val_str)
        if op == "<":
            mask &= df[col] < val
        else:
            mask &= df[col] >= val
    return mask


def combined_signal_backtest(val_df, rules_df, top_n=50):
    """Backtest a combined signal: score each stock-day by how many top rules fire."""
    from collections import defaultdict

    # Use top N rules
    top_rules = rules_df.head(top_n)
    conditions_list = []
    for _, r in top_rules.iterrows():
        conds = r["antecedents"].split(" AND ")
        parsed = []
        valid = True
        for c in conds:
            m = re.match(r"(\w+(?:_\w+)*)\s*(<|>=)\s*(-?[\d.e+]+)", c)
            if not m:
                valid = False
                break
            parsed.append((m.group(1), m.group(2), float(m.group(3))))
        if valid:
            conditions_list.append((r["lift"], parsed))

    print(f"  Using {len(conditions_list)} valid rules for combined signal")

    # Score each row by rule votes weighted by lift
    scores = np.zeros(len(val_df))
    for lift, conds in conditions_list:
        mask = pd.Series(True, index=val_df.index)
        ok = True
        for col, op, val in conds:
            if col not in val_df.columns:
                ok = False
                break
            if op == "<":
                mask &= val_df[col] < val
            else:
                mask &= val_df[col] >= val
        if ok:
            scores[mask.values] += lift

    val_df = val_df.copy()
    val_df["_score"] = scores

    # Test multiple score thresholds
    results = []
    for pct in [50, 60, 70, 80, 90, 95, 99]:
        thresh = np.percentile(scores[scores > 0], pct) if (scores > 0).sum() > 10 else 0
        if thresh <= 0:
            continue
        signal = scores >= thresh
        if signal.sum() < 5:
            continue
        tp = val_df.loc[signal, "_target"].sum()
        prec = tp / signal.sum()
        avg_ret = val_df.loc[signal, "ret_1d_oc"].mean()
        base_avg = val_df["ret_1d_oc"].mean()
        results.append({
            "score_pctile": pct,
            "threshold": round(thresh, 2),
            "n_signals": int(signal.sum()),
            "precision": prec,
            "avg_ret": avg_ret,
            "base_avg_ret": base_avg,
            "lift": avg_ret / base_avg if base_avg != 0 else 0,
        })

    return pd.DataFrame(results)


def daily_long_short(val_df, top_n_rules=50):
    """Simple long-short: each day, rank stocks by rule score, long top 5, short bottom 5."""
    from collections import defaultdict

    results = []
    rules = load_rules(con, min_lift=1.3, min_conf=0.17, max_rules=200)
    top_rules = rules.head(top_n_rules)

    conditions_list = []
    for _, r in top_rules.iterrows():
        conds = r["antecedents"].split(" AND ")
        parsed = []
        valid = True
        for c in conds:
            m = re.match(r"(\w+(?:_\w+)*)\s*(<|>=)\s*(-?[\d.e+]+)", c)
            if not m:
                valid = False
                break
            parsed.append((m.group(1), m.group(2), float(m.group(3))))
        if valid:
            conditions_list.append(parsed)

    # Score each row
    scores = np.zeros(len(val_df))
    for conds in conditions_list:
        mask = pd.Series(True, index=val_df.index)
        ok = True
        for col, op, val in conds:
            if col not in val_df.columns:
                ok = False
                break
            if op == "<":
                mask &= val_df[col] < val
            else:
                mask &= val_df[col] >= val
        if ok:
            scores[mask.values] += 1

    val_df = val_df.copy()
    val_df["_score"] = scores

    # Daily ranking
    val_df["_date"] = val_df["datetime"].dt.date
    daily_stats = []
    for date, group in val_df.groupby("_date"):
        if len(group) < 10:
            continue
        ranked = group.sort_values("_score", ascending=False)
        long = ranked.head(5)
        short = ranked.tail(5)
        long_ret = long["ret_1d_oc"].mean()
        short_ret = short["ret_1d_oc"].mean()
        daily_stats.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "spread": long_ret - short_ret,
            "n_stocks": len(group),
        })

    return pd.DataFrame(daily_stats)


def print_results(per_rule, combined, daily):
    """Print a summary report."""
    print("\n" + "="*70)
    print("  PATTERN RULE BACKTEST REPORT")
    print("="*70)

    # Per-rule summary
    if len(per_rule) > 0:
        print(f"\n--- PER-RULE PERFORMANCE (OOS, {len(per_rule)} rules tested) ---")
        print(f"{'Signals':>8} {'Prec':>6} {'AvgRet':>8} {'Lift':>6} {'Rule snippet':<50}")
        print("-"*80)
        for _, r in per_rule.sort_values("avg_ret", ascending=False).head(15).iterrows():
            snippet = r["antecedents"][:48]
            print(f"{r['n_signals']:>8} {r['precision']:>6.3f} {r['avg_ret']:>8.4f} "
                  f"{r['lift_vs_base']:>6.2f} {snippet:<50}")

    # Combined signal summary
    if len(combined) > 0:
        print(f"\n--- COMBINED SIGNAL (top rules weighted by lift) ---")
        print(f"{'Pctile':>8} {'Signals':>8} {'Prec':>6} {'AvgRet':>8} {'Lift':>6}")
        print("-"*40)
        for _, r in combined.iterrows():
            print(f"{r['score_pctile']:>8} {r['n_signals']:>8} {r['precision']:>6.3f} "
                  f"{r['avg_ret']:>8.4f} {r['lift']:>6.2f}")

    # Daily long-short
    if len(daily) > 0:
        print(f"\n--- DAILY LONG-SHORT (top 5 long / bottom 5 short) ---")
        print(f"  Days: {len(daily)}")
        print(f"  Avg long return:  {daily['long_ret'].mean():.4f}")
        print(f"  Avg short return: {daily['short_ret'].mean():.4f}")
        print(f"  Avg spread:       {daily['spread'].mean():.4f}")
        print(f"  Win rate (spread>0): {(daily['spread']>0).mean():.1%}")
        # Cumulative
        daily = daily.sort_values("date")
        daily["cum_spread"] = (1 + daily["spread"]).cumprod()
        print(f"  Cumulative spread: {daily['cum_spread'].iloc[-1]:.4f}")

    # Save report
    report_path = REPORT_DIR / f"pattern_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write("PATTERN RULE BACKTEST REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*70 + "\n")
        if len(per_rule) > 0:
            per_rule.to_csv(report_path.with_suffix(".csv"))
            f.write(f"\nPer-rule results saved to {report_path.with_suffix('.csv')}\n")
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Feature columns (technical features)
    feature_cols = [
        "rsi_14", "rsi_7", "atr_14", "adx", "bb_pct_b", "bb_width",
        "vol_ratio_10", "vol_ratio_5", "ret_5d", "ret_10d", "ret_1d",
        "close_vs_sma_20", "close_vs_sma_50", "close_vs_sma_10",
        "macd_hist", "macd_line", "stoch_k", "stoch_d", "mfi",
        "obv", "williams_r", "cci", "cmf", "eom", "uo", "trix",
        "zscore_20", "swing_high", "swing_low",
        "dc_width", "kc_width", "hv_20", "skew_20", "kurt_20",
        "plus_di", "minus_di", "aroon_osc",
    ]
    pattern_cols = [
        "flag", "pennant", "channel", "double_top", "double_bottom",
        "head_and_shoulders", "inside_bar", "nr4", "nr7",
        "volatility_contraction", "darvas_box_up", "darvas_box_down",
        "doji", "hammer", "hanging_man", "shooting_star",
        "marubozu", "spinning_top", "bullish_engulfing", "bearish_engulfing",
        "bullish_harami", "bearish_harami", "morning_star", "evening_star",
        "piercing_line", "dark_cloud_cover", "three_white_soldiers",
        "three_black_crows",
    ]

    print("Loading rules...")
    rules = load_rules(con, min_lift=1.3, min_conf=0.17, max_rules=200)
    print(f"  {len(rules)} rules loaded from DB")

    print("\nEvaluating rules on OOS data...")
    per_rule = evaluate_rules(con, rules, feature_cols, pattern_cols)
    print(f"  {len(per_rule)} rules had >=2 OOS signals")

    print("\nRunning combined signal backtest...")
    df_full = _apply_conditions.__globals__  # hack to get around scope
    combined = combined_signal_backtest.__globals__
    
    # Actually run the combined backtest directly
    # Load data needed for combined test
    quoted = [f'fs."{c}"' for c in feature_cols]
    df = con.execute(f"""
        SELECT fs.datetime, fs.symbol,
               LEAD(fs."close") OVER w / LEAD(fs."open") OVER w - 1 AS ret_1d_oc,
               {', '.join(quoted)}
        FROM feature_store fs
        WHERE fs.timeframe='1day' AND fs.ret_5d IS NOT NULL
        WINDOW w AS (PARTITION BY fs.symbol ORDER BY fs.datetime)
        ORDER BY fs.datetime
    """).fetchdf()

    # Merge pattern occurrences
    for cat in ("chart", "candle"):
        po = con.execute(f"""
            SELECT datetime, pattern FROM pattern_occurrences
            WHERE timeframe='1day' AND category='{cat}'
        """).fetchdf()
        if len(po) > 0:
            po["datetime"] = pd.to_datetime(po["datetime"])
            po_pivot = po.pivot_table(index="datetime", columns="pattern",
                                       aggfunc="size", fill_value=0)
            po_pivot = (po_pivot > 0).astype(int)
            df = df.merge(po_pivot, left_on="datetime", right_index=True,
                          how="left").fillna(0)

    # Filter to val period
    dates = df["datetime"].sort_values()
    split_idx = int(len(dates) * 0.8)
    split_date = dates.iloc[split_idx]
    val_df = df[df["datetime"] > split_date].copy()
    pct_85 = val_df["ret_1d_oc"].quantile(0.85)
    val_df["_target"] = (val_df["ret_1d_oc"] >= pct_85).astype(int)
    print(f"  Val period: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
    print(f"  Val rows: {len(val_df):,}, high-gainers: {val_df['_target'].sum():,}")

    combined = combined_signal_backtest(val_df, rules, top_n=50)
    daily = daily_long_short(val_df, top_n_rules=50)

    print_results(per_rule, combined, daily)

    con.close()
