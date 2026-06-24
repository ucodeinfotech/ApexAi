"""Thorough data quality check on index data"""
import os, csv, pandas as pd
from datetime import datetime, timedelta

d = "nifty50_full_history"
index_files = [f for f in os.listdir(d) if any(x in f for x in ["NIFTY50_","BANKNIFTY_","SENSEX_"])]

for f in sorted(index_files):
    path = f"{d}/{f}"
    sym = f.split("_")[0]
    interval = "5min" if "FIVE" in f else "1hr"
    exp_gap = 5 if "FIVE" in f else 60
    
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    total = len(df)
    
    print(f"\n{'='*60}")
    print(f"{sym} {interval} ({total:,} rows)")
    print(f"{'='*60}")
    
    # 1) Sort & duplicates
    dup = df["datetime"].duplicated().sum()
    if dup:
        print(f"  DUPLICATES: {dup} rows")
    
    # 2) Coverage by year
    years = df["date"].apply(lambda x: x.year).value_counts().sort_index()
    print(f"  Yearly data points:")
    for y, c in years.items():
        print(f"    {y}: {c:>7,} rows")
    
    # 3) Missing days - check consecutive trading days
    all_dates = sorted(df["date"].unique())
    missing_dates = []
    for i in range(1, len(all_dates)):
        diff = (all_dates[i] - all_dates[i-1]).days
        if diff > 4:  # More than a long weekend
            missing_dates.append((all_dates[i-1], all_dates[i], diff))
    if missing_dates:
        print(f"  Missing day gaps (>4 day breaks):")
        for prev, nxt, diff_d in missing_dates[:10]:
            print(f"    {prev} to {nxt} ({diff_d-1} days missing)")
        if len(missing_dates) > 10:
            print(f"    ... and {len(missing_dates)-10} more")
    
    # 4) Daily completeness - how many bars per day
    bars_per_day = df.groupby("date").size()
    expected = 75 if interval == "5min" else 6  # ~75 five-min bars/day, ~6 one-hour bars/day
    low_days = bars_per_day[bars_per_day < expected * 0.5]
    if len(low_days) > 0:
        print(f"  Days with <50% expected bars ({expected}): {len(low_days)} days")
        for dt_, c in low_days.head(5).items():
            print(f"    {dt_}: {c} bars (expected ~{expected})")
        if len(low_days) > 5:
            print(f"    ... and {len(low_days)-5} more")
    
    # 5) OHLC checks
    bad_lo = (df["low"] > df[["open","close"]].max(axis=1)).sum()
    bad_hi = (df["high"] < df[["open","close"]].min(axis=1)).sum()
    bad_all = ((df["low"] > df[["open","close"]].max(axis=1)) | (df["high"] < df[["open","close"]].min(axis=1))).sum()
    if bad_all:
        print(f"  OHLC errors: {bad_all} rows")
        print(f"    High < min(O,C): {bad_hi}, Low > max(O,C): {bad_lo}")
        samples = df[(df["low"] > df[["open","close"]].max(axis=1)) | (df["high"] < df[["open","close"]].min(axis=1))]
        print(f"  Samples:")
        for _, r in samples.head(5).iterrows():
            print(f"    {r['datetime']} O={r['open']} H={r['high']} L={r['low']} C={r['close']}")
    
    # 6) Price outliers (Z-score > 5)
    for col in ["open","high","low","close"]:
        mean = df[col].mean()
        std = df[col].std()
        outliers = (abs(df[col] - mean) > 5 * std).sum()
        if outliers:
            print(f"  {col}: {outliers} extreme outliers (>5σ)")
    
    # 7) Zero/negative checks
    zero_vol = (df["volume"] != 0).sum()
    if zero_vol:
        print(f"  Non-zero volume: {zero_vol} rows (should be 0 for indices)")
    
    neg_close = (df["close"] <= 0).sum()
    if neg_close:
        print(f"  Non-positive close: {neg_close} rows")
    
    # 8) Gap analysis within trading days
    intraday_gaps = 0
    for date, grp in df.groupby("date"):
        grp = grp.sort_values("datetime")
        for i in range(1, len(grp)):
            gap = (grp["datetime"].iloc[i] - grp["datetime"].iloc[i-1]).total_seconds() / 60
            if gap > exp_gap * 2:
                intraday_gaps += 1
    if intraday_gaps:
        print(f"  Intra-day gaps >{exp_gap*2}min: {intraday_gaps}")
    
    # 9) First & last 3 rows
    print(f"  First 3:")
    for _, r in df.head(3).iterrows():
        print(f"    {r['datetime']} O={r['open']} H={r['high']} L={r['low']} C={r['close']} V={r['volume']}")
    print(f"  Last 3:")
    for _, r in df.tail(3).iterrows():
        print(f"    {r['datetime']} O={r['open']} H={r['high']} L={r['low']} C={r['close']} V={r['volume']}")
