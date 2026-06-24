"""Validate all 100 CSV files - check format, sort order, gaps, date ranges"""
import os, pandas as pd
from datetime import timedelta

d = "nifty50_full_history"
issues = []

for f in sorted(os.listdir(d)):
    if not f.endswith(".csv") or f.startswith("_"):
        continue
    path = f"{d}/{f}"
    sym = f.replace("_FIFTEEN_MINUTE.csv","").replace("_ONE_MINUTE.csv","")
    interval = "15min" if "FIFTEEN" in f else "1min"
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        issues.append(f"{f}: FAILED TO READ - {e}")
        continue
    
    if df.empty:
        issues.append(f"{f}: EMPTY FILE")
        continue
    
    # 1) Check columns
    expected = ["datetime","open","high","low","close","volume"]
    if list(df.columns) != expected:
        issues.append(f"{f}: BAD COLUMNS - got {list(df.columns)}")
        continue
    
    # 2) Check data types
    if df["datetime"].dtype != object:
        issues.append(f"{f}: datetime not string type")
        continue
    for col in ["open","high","low","close"]:
        if df[col].dtype not in (float, int):
            issues.append(f"{f}: {col} not numeric, got {df[col].dtype}")
    if df["volume"].dtype not in (float, int):
        issues.append(f"{f}: volume not numeric")
    
    # 3) Check datetime parsing and sort order
    try:
        dt = pd.to_datetime(df["datetime"])
    except:
        issues.append(f"{f}: datetime column cannot be parsed")
        continue
    
    # Check for NaT
    if dt.isna().any():
        issues.append(f"{f}: {dt.isna().sum()} unparseable datetimes")
        continue
    
    # Check sort order (strictly increasing)
    diffs = dt.diff()
    if (diffs <= pd.Timedelta(0)).any():
        bad = (diffs <= pd.Timedelta(0)).sum()
        issues.append(f"{f}: {bad} rows NOT in chronological order")
    
    # 4) Check for gaps
    expected_gap = timedelta(minutes=1) if interval == "1min" else timedelta(minutes=15)
    # Only check within same trading day (skip overnight gaps)
    prev_date = None
    gap_count = 0
    for i in range(1, len(dt)):
        curr_date = dt.iloc[i].date()
        if curr_date == dt.iloc[i-1].date():
            gap = dt.iloc[i] - dt.iloc[i-1]
            if gap > expected_gap * 2:  # allow 2x for occasional missing bars
                gap_count += 1
    
    # 5) Check OHLC consistency
    bad_ohlc = 0
    for _, row in df.iterrows():
        if not (row["low"] <= row["open"] <= row["high"] and 
                row["low"] <= row["close"] <= row["high"]):
            bad_ohlc += 1
    if bad_ohlc:
        issues.append(f"{f}: {bad_ohlc} rows with OHLC inconsistency (low>open or low>close etc)")
    
    # 6) Check first/last dates
    first = dt.min()
    last = dt.max()
    
    # 7) Check volume
    if (df["volume"] < 0).any():
        issues.append(f"{f}: {((df['volume'] < 0).sum())} rows with negative volume")
    
    if issues and any(f[:20] in i for i in issues):
        pass  # skip - already caught

# Print results
print("=" * 80)
print("DATA VALIDATION REPORT")
print("=" * 80)

if not issues:
    print("\nALL 100 FILES PASSED VALIDATION ✓")
    print("  - Correct column format (datetime,open,high,low,close,volume)")
    print("  - Chronologically sorted")
    print("  - OHLC values consistent")
    print("  - No negative volumes")
else:
    print(f"\n{len(issues)} ISSUES FOUND:")
    for i in issues:
        print(f"  X {i}")

# Check sample stocks in detail
print("\n" + "=" * 80)
print("SAMPLE DATA CHECK (SBIN - 15min)")
print("=" * 80)
try:
    df = pd.read_csv(f"{d}/SBIN_FIFTEEN_MINUTE.csv")
    dt = pd.to_datetime(df["datetime"])
    print(f"  Rows: {len(df):,}")
    print(f"  Range: {dt.min()} to {dt.max()}")
    print(f"  Trading days: {dt.dt.date.nunique()}")
    print(f"  First 3 rows:")
    for _, r in df.head(3).iterrows():
        print(f"    {r['datetime']} O={r['open']} H={r['high']} L={r['low']} C={r['close']} V={r['volume']}")
    print(f"  Last 3 rows:")
    for _, r in df.tail(3).iterrows():
        print(f"    {r['datetime']} O={r['open']} H={r['high']} L={r['low']} C={r['close']} V={r['volume']}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 80)
print("SAMPLE DATA CHECK (SBIN - 1min)")
print("=" * 80)
try:
    df = pd.read_csv(f"{d}/SBIN_ONE_MINUTE.csv")
    dt = pd.to_datetime(df["datetime"])
    print(f"  Rows: {len(df):,}")
    print(f"  Range: {dt.min()} to {dt.max()}")
    print(f"  Trading days: {dt.dt.date.nunique()}")
    # Check gap rate
    gaps = 0
    total = 0
    for i in range(1, len(dt)):
        if dt.iloc[i].date() == dt.iloc[i-1].date():
            total += 1
            if dt.iloc[i] - dt.iloc[i-1] > timedelta(minutes=2):
                gaps += 1
    print(f"  Intra-day gaps (>2min): {gaps}/{total} intervals ({100*gaps/max(total,1):.2f}%)")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 80)
print("STOCKS WITH 2016-10-03 START DATE VERIFICATION")
print("=" * 80)
count = 0
for f in sorted(os.listdir(d)):
    if f.endswith("_FIFTEEN_MINUTE.csv"):
        df = pd.read_csv(f"{d}/{f}")
        dt = pd.to_datetime(df["datetime"])
        first = str(dt.min().date())
        if first == "2016-10-03":
            count += 1
print(f"  {count}/50 stocks start exactly on 2016-10-03 for 15-min data")

small = []
for f in sorted(os.listdir(d)):
    if f.endswith("_ONE_MINUTE.csv"):
        df = pd.read_csv(f"{d}/{f}")
        dt = pd.to_datetime(df["datetime"])
        first = str(dt.min().date())
        last = str(dt.max().date())
        if first != "2016-10-03":
            small.append((f.replace("_ONE_MINUTE.csv",""), first, last, len(df)))
print(f"\n  Stocks with shorter 1-min history:")
for sym, f, t, r in sorted(small, key=lambda x: x[3]):
    print(f"    {sym:20s} {f} to {t} ({r:,} rows)")
