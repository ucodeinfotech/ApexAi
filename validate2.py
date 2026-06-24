"""Fast validation - read files line by line"""
import os
from datetime import datetime, timedelta
import csv

d = "nifty50_full_history"
files = sorted([f for f in os.listdir(d) if f.endswith(".csv") and not f.startswith("_")])
print(f"Validating {len(files)} files...")
issues = []

for f in files:
    path = f"{d}/{f}"
    sym = f.replace("_FIFTEEN_MINUTE.csv","").replace("_ONE_MINUTE.csv","")
    interval = "15min" if "FIFTEEN" in f else "1min"
    expected_gap = 1 if interval == "1min" else 15
    
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["datetime","open","high","low","close","volume"]:
            issues.append(f"{f}: bad header: {header}")
            continue
        
        prev_dt = None
        prev_date = None
        row_num = 1
        bad_order = 0
        gaps = 0
        bad_ohlc = 0
        neg_vol = 0
        first_dt = last_dt = None
        
        for row in reader:
            row_num += 1
            if len(row) != 6:
                issues.append(f"{f}: row {row_num} has {len(row)} cols")
                continue
            
            dt_str, o, h, l, c, v = row
            
            # Parse datetime
            try:
                dt = datetime.fromisoformat(dt_str)
            except:
                issues.append(f"{f}: row {row_num} bad datetime: {dt_str}")
                continue
            
            if first_dt is None:
                first_dt = dt
            last_dt = dt
            
            # Check sort order
            if prev_dt and dt < prev_dt:
                bad_order += 1
            
            # Check intra-day gaps
            if prev_date and dt.date() == prev_date and prev_dt:
                gap_min = (dt - prev_dt).total_seconds() / 60
                if gap_min > expected_gap * 2:
                    gaps += 1
            
            # Parse OHLCV
            try:
                o, h, l, c, v = float(o), float(h), float(l), float(c), int(v)
            except:
                issues.append(f"{f}: row {row_num} bad numeric values")
                continue
            
            # OHLC consistency
            if not (l <= o <= h and l <= c <= h):
                bad_ohlc += 1
            
            if v < 0:
                neg_vol += 1
            
            prev_dt = dt
            prev_date = dt.date()
    
    if bad_order:
        issues.append(f"{f}: {bad_order} rows out of order")
    if gaps:
        issues.append(f"{f}: {gaps} gaps >{expected_gap*2}min")
    if bad_ohlc:
        issues.append(f"{f}: {bad_ohlc} OHLC inconsistencies")
    if neg_vol:
        issues.append(f"{f}: {neg_vol} negative volumes")

print(f"\nResults: {len(issues)} issues across {len(files)} files")
if issues:
    for i in issues[:20]:
        print(f"  {i}")
    if len(issues) > 20:
        print(f"  ... and {len(issues)-20} more")
else:
    print("ALL FILES PASSED!")

# Quick summary
print("\n--- Range Summary ---")
f15 = sorted([f for f in os.listdir(d) if f.endswith("_FIFTEEN_MINUTE.csv")])
f1 = sorted([f for f in os.listdir(d) if f.endswith("_ONE_MINUTE.csv")])

oct2016_15 = 0
oct2016_1 = 0
total15 = total1 = 0
for f in f15:
    with open(f"{d}/{f}") as fh:
        lines = fh.readlines()
    first = lines[1].split(",")[0][:10]
    r = len(lines)-1
    total15 += r
    if first == "2016-10-03": oct2016_15 += 1

for f in f1:
    with open(f"{d}/{f}") as fh:
        lines = fh.readlines()
    first = lines[1].split(",")[0][:10]
    r = len(lines)-1
    total1 += r
    if first == "2016-10-03": oct2016_1 += 1

print(f"  15-min: {oct2016_15}/50 start 2016-10-03, {total15:,} total rows")
print(f"  1-min:  {oct2016_1}/50 start 2016-10-03, {total1:,} total rows")

# Stocks that don't start at Oct 2016
print("\n  Stocks with shorter history:")
for f in f15:
    sym = f.replace("_FIFTEEN_MINUTE.csv","")
    with open(f"{d}/{f}") as fh:
        lines = fh.readlines()
    first = lines[1].split(",")[0][:10]
    if first != "2016-10-03":
        print(f"    {sym:20s} 15-min from {first} ({len(lines)-1:,} rows)")
for f in f1:
    sym = f.replace("_ONE_MINUTE.csv","")
    with open(f"{d}/{f}") as fh:
        lines = fh.readlines()
    first = lines[1].split(",")[0][:10]
    if first != "2016-10-03":
        with open(f"{d}/{f.replace('_ONE_MINUTE.csv','_FIFTEEN_MINUTE.csv')}") as fh2:
            pass  # already printed above
        print(f"    {sym:20s} 1-min  from {first} ({len(lines)-1:,} rows)")
