"""Validate index data files"""
import os, csv
from datetime import datetime, timedelta

d = "nifty50_full_history"
index_files = [f for f in os.listdir(d) if any(x in f for x in ["NIFTY50_","BANKNIFTY_","SENSEX_"])]
print(f"Checking {len(index_files)} index files...\n")

for f in index_files:
    path = f"{d}/{f}"
    interval = "5min" if "FIVE" in f else "1hr"
    sym = f.split("_")[0]
    
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["datetime","open","high","low","close","volume"]:
            print(f"  {f}: BAD HEADER: {header}")
            continue
        
        rows = []
        for row in reader:
            rows.append(row)
    
    total = len(rows)
    first = rows[0][0] if rows else "?"
    last = rows[-1][0] if rows else "?"
    
    # Parse datetimes
    dts = []
    for r in rows:
        try:
            dts.append(datetime.fromisoformat(r[0]))
        except:
            pass
    
    out_of_order = sum(1 for i in range(1, len(dts)) if dts[i] < dts[i-1])
    
    # Check gaps (within same day)
    exp_gap = 5 if "FIVE" in f else 60
    gaps = 0
    for i in range(1, len(dts)):
        if dts[i].date() == dts[i-1].date():
            gap_min = (dts[i] - dts[i-1]).total_seconds() / 60
            if gap_min > exp_gap * 2:
                gaps += 1
    
    # Check negative prices
    neg_price = 0
    neg_vol = 0
    bad_ohlc = 0
    for r in rows:
        try:
            o, h, l, c, v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[5])
        except:
            continue
        if o < 0 or h < 0 or l < 0 or c < 0:
            neg_price += 1
        if v < 0:
            neg_vol += 1
        if not (l <= o <= h and l <= c <= h):
            bad_ohlc += 1
    
    # Check volume - should be 0 for indices
    non_zero_vol = sum(1 for r in rows if int(r[5]) != 0)
    
    num_dates = len(set(d.date() for d in dts))
    
    print(f"{sym:12s} {interval:4s} {total:>8,} rows  {str(dts[0].date()) if dts else '?':12s} to {str(dts[-1].date()) if dts else '?':12s}")
    print(f"  Dates: {num_dates} trading days")
    if out_of_order:
        print(f"  ! {out_of_order} out-of-order rows")
    if gaps:
        print(f"  ! {gaps} intra-day gaps >{exp_gap*2}min")
    if bad_ohlc:
        print(f"  ! {bad_ohlc} OHLC inconsistencies")
    if neg_price:
        print(f"  ! {neg_price} negative prices")
    if neg_vol:
        print(f"  ! {neg_vol} negative volumes")
    if non_zero_vol:
        print(f"  Note: {non_zero_vol} rows with non-zero volume (index should be 0)")
    print()

# Quick sample spot check
print("="*60)
print("SAMPLE DATA CHECK")
print("="*60)
for f in index_files:
    if "FIVE" in f:
        with open(f"{d}/{f}") as fh:
            lines = fh.readlines()
        print(f"\n{f}:")
        print(f"  First: {lines[1].strip()}")
        print(f"  Last:  {lines[-1].strip()}")
        # Check a few middle rows
        mid = len(lines) // 2
        print(f"  Mid:   {lines[mid].strip()}")
