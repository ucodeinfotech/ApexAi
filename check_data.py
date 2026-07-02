import os, csv, glob
from datetime import datetime, timedelta
import pandas as pd

DATA = r"C:\Users\pc\Downloads\stock hist data\comprehensive_data"

min_files = sorted(glob.glob(os.path.join(DATA, "*_ONE_MINUTE.csv")))
stocks = []
for f in min_files:
    base = os.path.basename(f)
    sym = base.replace("_ONE_MINUTE.csv", "")
    stocks.append((sym, f))

print(f"Found {len(stocks)} stocks with ONE_MINUTE data\n")

results = []
for sym, fpath in stocks:
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        results.append((sym, "PARSE_ERROR", "", "", 0, 0))
        continue

    expected_cols = {"datetime", "open", "high", "low", "close", "volume"}
    actual_cols = set(str(c).lower().strip() for c in df.columns)
    if not expected_cols.issubset(actual_cols):
        results.append((sym, "BAD_COLS", "", "", 0, 0))
        continue

    ts_col = [c for c in df.columns if str(c).lower().strip() == "datetime"][0]
    try:
        df["_ts"] = pd.to_datetime(df[ts_col])
    except Exception as e:
        results.append((sym, "BAD_TS", "", "", 0, 0))
        continue

    start = df["_ts"].min()
    end = df["_ts"].max()
    rows = len(df)

    df_sorted = df.sort_values("_ts")
    diffs = df_sorted["_ts"].diff()
    big_gaps = (diffs > timedelta(minutes=60)).sum()

    results.append((sym, "OK", start, end, rows, big_gaps))

print(f"{'Symbol':<20} {'Status':<12} {'Start':<22} {'End':<22} {'Rows':<10} {'Gaps>60m'}")
print("="*100)
for r in results:
    sym = r[0]
    status = r[1]
    if status == "OK":
        print(f"{sym:<20} {status:<12} {str(r[2]):<22} {str(r[3]):<22} {r[4]:<10} {r[5]}")
    else:
        print(f"{sym:<20} {status:<12}")

print("\n" + "="*100)
ok = [r for r in results if r[1] == "OK"]
errors = [r for r in results if r[1] != "OK"]
print(f"\nOK: {len(ok)}/{len(results)}")
if errors:
    print(f"Errors: {len(errors)}")
    for r in errors:
        print(f"  {r[0]}: {r[1]}")

if ok:
    earliest = min(r[2] for r in ok)
    latest = max(r[3] for r in ok)
    total_rows = sum(r[4] for r in ok)
    total_gaps = sum(r[5] for r in ok)
    print(f"\nEarliest start: {earliest}")
    print(f"Latest end:     {latest}")
    print(f"Total rows:     {total_rows:,}")
    print(f"Total gaps:     {total_gaps} (across all stocks)")

if ok:
    with_gaps = [r for r in ok if r[5] > 0]
    if with_gaps:
        print(f"\nStocks with gaps (biggest first):")
        for r in sorted(with_gaps, key=lambda x: -x[5])[:30]:
            print(f"  {r[0]:<20} {r[5]} gaps")
