"""Verify all stocks have complete data across all timeframes"""
import os, pandas as pd
from datetime import datetime

DATA_DIR = "comprehensive_data"
ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]

# Get all unique stocks across both directories
all_stocks = set()
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_MINUTE.csv"):
                all_stocks.add(f.replace("_ONE_MINUTE.csv", ""))

all_stocks = sorted(all_stocks)
print(f"Total unique stocks with 1-min data: {len(all_stocks)}")
print(f"{'='*120}")

timeframes = ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "ONE_DAY"]
issues = []
ok_count = 0

for sym in all_stocks:
    row_counts = {}
    date_ranges = {}
    missing = []
    empty = []
    
    for tf in timeframes:
        found = False
        for d in ALL_DIRS:
            fpath = f"{d}/{sym}_{tf}.csv"
            if os.path.exists(fpath):
                found = True
                try:
                    df = pd.read_csv(fpath)
                    row_counts[tf] = len(df)
                    if "datetime" in df.columns:
                        dt = pd.to_datetime(df["datetime"])
                        date_ranges[tf] = (str(dt.min().date()), str(dt.max().date()))
                    else:
                        date_ranges[tf] = ("?", "?")
                except:
                    empty.append(tf)
                break
        if not found:
            missing.append(tf)
    
    if missing or empty:
        issues.append((sym, missing, empty, row_counts))
        status = "MISSING FILES!" if missing else "EMPTY FILES!"
        print(f"  [{status:15s}] {sym:15s} missing={missing} empty={empty}")
    else:
        ok_count += 1
        # Print summary for first 5 and last 5
        if ok_count <= 5 or ok_count > len(all_stocks) - 5:
            r1 = row_counts.get("ONE_MINUTE", 0)
            rf = row_counts.get("FIFTEEN_MINUTE", 0)
            rd = row_counts.get("ONE_DAY", 0)
            dr = date_ranges.get("ONE_MINUTE", ("?","?"))
            print(f"  [OK] {sym:15s} 1m={r1:>7,} 15m={rf:>6,} 1d={rd:>5,} {dr[0]} to {dr[1]}")

print(f"{'='*120}")
print(f"OK: {ok_count}/{len(all_stocks)}")
print(f"Issues: {len(issues)}")

if issues:
    print(f"\n{'='*60}")
    print("DETAILED ISSUES:")
    for sym, missing, empty, counts in issues:
        print(f"  {sym}: missing={missing}, empty={empty}, partial_counts={counts}")

# Additional check: files that exist but are not tracked
print(f"\n{'='*60}")
print("Checking for stale/extra files...")
all_files = set()
for d in ALL_DIRS:
    if os.path.exists(d):
        all_files.update(os.listdir(d))

expected_suffixes = ["_ONE_MINUTE.csv", "_FIVE_MINUTE.csv", "_FIFTEEN_MINUTE.csv", "_ONE_HOUR.csv", "_ONE_DAY.csv"]
stale = []
for f in sorted(all_files):
    if any(f.endswith(s) for s in expected_suffixes):
        sym = f.rsplit("_", 2)[0]
        suffix = f[len(sym)+1:]
        if sym not in all_stocks:
            stale.append(f)
            
if stale:
    print(f"Orphan files (stock not in 1-min list): {stale[:10]}")
else:
    print("No orphan files found.")

print(f"\nDone! All data verified.")
