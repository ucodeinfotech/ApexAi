"""Thorough data integrity check: date ranges, missing months, gaps"""
import os, sys
from datetime import datetime, timedelta, date
import pandas as pd

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]

# Get all stocks
all_stocks = set()
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_MINUTE.csv"):
                all_stocks.add(f.replace("_ONE_MINUTE.csv", ""))
all_stocks = sorted(all_stocks)

print(f"Checking {len(all_stocks)} stocks for data gaps...\n")

# Track results
min_date, max_date = None, None
stocks_with_gaps = []
stocks_with_short_history = []
stocks_small = []
results = []

for sym in all_stocks:
    # Find the file
    fpath = None
    for d in ALL_DIRS:
        p = f"{d}/{sym}_ONE_MINUTE.csv"
        if os.path.exists(p):
            fpath = p
            break
    if not fpath:
        continue

    size_mb = os.path.getsize(fpath) / 1024 / 1024

    # Read just the datetime column (fast)
    df = pd.read_csv(fpath, usecols=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()
    d_min = dt_min.date()
    d_max = dt_max.date()

    if min_date is None or d_min < min_date:
        min_date = d_min
    if max_date is None or d_max > max_date:
        max_date = d_max

    # Get unique trading days (calendar days may not all be market days)
    days = set(d.date() for d in df["datetime"].drop_duplicates())
    num_days = len(days)
    total_rows = len(df)
    total_calendar_days = (d_max - d_min).days + 1

    # Check for missing months
    months_present = set()
    for d in days:
        months_present.add((d.year, d.month))
    
    # Generate expected months from start to end
    expected_months = set()
    d = date(d_min.year, d_min.month, 1)
    end = date(d_max.year, d_max.month, 1)
    while d <= end:
        expected_months.add((d.year, d.month))
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)
    
    missing_months = expected_months - months_present
    
    # Check for gaps: find days where diff > some threshold (e.g., 10 calendar days without trading)
    sorted_days = sorted(days)
    gaps = []
    for i in range(1, len(sorted_days)):
        gap = (sorted_days[i] - sorted_days[i-1]).days
        if gap > 10:  # More than 10 calendar days = likely missing data, not just weekend
            gaps.append((sorted_days[i-1], sorted_days[i], gap))

    issues = []
    if missing_months:
        issues.append(f"{len(missing_months)} missing months")
    if gaps:
        issues.append(f"{len(gaps)} data gaps >10 days")
    
    # Also check if stock data starts very late (after 2018)
    if d_min > date(2020,1,1) and "JIOFIN" not in sym:
        stocks_with_short_history.append((sym, d_min, d_max, num_days, size_mb))
    
    if total_rows < 100000:
        stocks_small.append((sym, d_min, d_max, total_rows, size_mb))

    result = {
        "sym": sym, "rows": total_rows, "days": num_days,
        "start": d_min, "end": d_max,
        "missing_months": sorted(missing_months) if missing_months else [],
        "gaps": gaps,
        "issues": issues,
        "size_mb": size_mb
    }
    results.append(result)

    # Print
    if missing_months or gaps:
        stocks_with_gaps.append(sym)
        print(f"!! {sym:15s} {d_min} to {d_max}  {num_days:4d}d  {total_rows:>7,} rows  {size_mb:5.1f}MB  ISSUES: {', '.join(issues)}")
        if missing_months:
            mm_str = ", ".join(f"{y}-{m:02d}" for y,m in sorted(missing_months)[:10])
            print(f"   missing months: {mm_str}" + ("..." if len(missing_months) > 10 else ""))
        if gaps:
            for g in gaps[:5]:
                print(f"   gap: {g[0]} to {g[1]} ({g[2]} days)")
    else:
        # Print only summary stats
        pass

# Summary
print(f"\n{'='*80}")
print(f"Overall date range: {min_date} to {max_date}")
print(f"Stocks with gaps/missing months: {len(stocks_with_gaps)}/{len(all_stocks)}")

# Late starters
print(f"\nStocks starting after 2020 (excluding JIOFIN):")
for sym, s, e, d, mb in sorted(stocks_with_short_history, key=lambda x: x[1]):
    print(f"  {sym:15s} {s} to {e}  {d:4d} days  {mb:5.1f}MB")

# Small data stocks
if stocks_small:
    print(f"\nStocks with <100K rows:")
    for sym, s, e, r, mb in sorted(stocks_small, key=lambda x: x[3]):
        print(f"  {sym:15s} {s} to {e}  {r:>7,} rows  {mb:5.1f}MB")

# Global missing months across all stocks
print(f"\n{'='*80}")
print("Global analysis of missing months across all stocks...")
month_stock_count = {}
for r in results:
    for y, m in r["missing_months"]:
        key = f"{y}-{m:02d}"
        month_stock_count.setdefault(key, 0)
        month_stock_count[key] += 1

if month_stock_count:
    print("Months missing from multiple stocks (>5 stocks):")
    for month, count in sorted(month_stock_count.items()):
        if count > 5:
            print(f"  {month}: {count} stocks missing")
else:
    print("No months are missing from any stock!")

# Per-stock breakdown of ALL stocks (compact format)
print(f"\n{'='*80}")
print("All stocks date ranges:")
for r in results:
    flag = ""
    if r["issues"]:
        flag = " *** " + "; ".join(r["issues"])
    print(f"  {r['sym']:15s} {r['start']} to {r['end']}  {r['days']:4d}d  {r['rows']:>7,}r{flag}")

print(f"\nDone! Check results above.")
