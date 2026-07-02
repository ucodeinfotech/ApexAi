"""Ultra-fast gap check: check only previously-gappy stocks, stream CSV"""
import os
from datetime import datetime, date

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]
istocks = ["ABB","ADANIENSOL","ADANIPOWER","AMBUJACEM","BAJAJHLDNG","BANKBARODA",
    "BOSCHLTD","BPCL","BRITANNIA","CANBK","CHOLAFIN","CUMMINSIND","DIXON","DMART",
    "FEDERALBNK","FORTIS","FSL","GAIL","GMRINFRA","GODREJPROP","HEROMOTOCO",
    "HINDPETRO","HUDCO","ICICIPRULI","IDBI","IDFCFIRSTB","INDUSINDBK","JBCHEPHARM",
    "JSWENERGY","JUBLFOOD","KALYANKJIL","LICI","LUPIN","MANKIND","MARICO",
    "NATIONALUM","NHPC","NMDC","OIL","PAGEIND","PERSISTENT","PIIND","POLYCAB",
    "SYNGENE"]

gaps_found = 0
stocks_with_gaps = 0

for sym in istocks:
    fpath = None
    for d in ALL_DIRS:
        p = f"{d}/{sym}_ONE_MINUTE.csv"
        if os.path.exists(p):
            fpath = p; break
    if not fpath:
        print(f"{sym}: file not found")
        continue
    
    # Stream the CSV, track trading days
    prev_date = None
    first_date = None
    last_date = None
    gaps = []
    row_count = 0
    
    with open(fpath) as f:
        header = f.readline()  # skip header
        for line in f:
            row_count += 1
            parts = line.split(',')
            dt_str = parts[0].strip('"')
            d = dt_str.split()[0]  # get date part
            curr = date.fromisoformat(d)
            
            if first_date is None:
                first_date = curr
            
            if prev_date is not None:
                gap = (curr - prev_date).days
                if gap > 10:
                    gaps.append((prev_date, curr, gap))
            
            prev_date = curr
            last_date = curr
    
    if gaps:
        stocks_with_gaps += 1
        gaps_found += len(gaps)
        print(f"GAP: {sym:15s} {first_date} to {last_date} {row_count:>7,}r {len(gaps)} gap(s)")
        for g in gaps[:3]:
            print(f"      {g[0]} -> {g[1]} ({g[2]}d)")

print(f"\nStocks with remaining gaps: {stocks_with_gaps}/{len(istocks)}")
print(f"Total gaps: {gaps_found}")
if stocks_with_gaps == 0:
    print("ALL ORIGINALLY-GAPPY STOCKS NOW GAP-FREE!")
print("Done!")
