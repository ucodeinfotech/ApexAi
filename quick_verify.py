"""Quick verify: check file existence and sizes only"""
import os

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]

all_stocks = set()
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_MINUTE.csv"):
                all_stocks.add(f.replace("_ONE_MINUTE.csv", ""))

all_stocks = sorted(all_stocks)
print(f"Total unique stocks: {len(all_stocks)}")

timeframes = ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "ONE_DAY"]
issues = []
ok = 0

for sym in all_stocks:
    row_counts = {}
    missing = []
    empty = []
    
    for tf in timeframes:
        found = False
        for d in ALL_DIRS:
            fpath = f"{d}/{sym}_{tf}.csv"
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                found = True
                if size == 0:
                    empty.append(tf)
                else:
                    # Count newlines as rough row count
                    with open(fpath) as fh:
                        rows = sum(1 for _ in fh) - 1  # minus header
                        row_counts[tf] = max(0, rows)
                break
        if not found:
            missing.append(tf)
    
    if missing or empty:
        issues.append((sym, missing, empty, row_counts))
    else:
        ok += 1

print(f"\nOK: {ok}/{len(all_stocks)}")
print(f"Issues: {len(issues)}")

if issues:
    print(f"\nISSUES:")
    for sym, missing, empty, counts in issues:
        print(f"  {sym}: missing={missing} empty={empty} counts={counts}")

# Show data depth sample
print(f"\nData depth (first 5 + last 5 stocks):")
count = 0
for sym in all_stocks:
    tf = "ONE_MINUTE"
    for d in ALL_DIRS:
        fpath = f"{d}/{sym}_{tf}.csv"
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            count += 1
            if count <= 5 or count > len(all_stocks) - 5:
                print(f"  {sym:15s} 1-min: {size_mb:.1f} MB")
            break

# Check total data size
total_size = 0
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if any(f.endswith(s) for s in ["_ONE_MINUTE.csv","_FIVE_MINUTE.csv","_FIFTEEN_MINUTE.csv","_ONE_HOUR.csv","_ONE_DAY.csv"]):
                total_size += os.path.getsize(f"{d}/{f}")
print(f"\nTotal data size: {total_size/1024/1024/1024:.2f} GB")
print("Done!")
