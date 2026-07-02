import os, csv, glob
from datetime import datetime, timedelta

DATA = r"C:\Users\pc\Downloads\stock hist data\comprehensive_data"

min_files = sorted(glob.glob(os.path.join(DATA, "*_ONE_MINUTE.csv")))
print(f"Found {len(min_files)} ONE_MINUTE files\n")

header_ok = 0
results = []

for fpath in min_files:
    base = os.path.basename(fpath)
    sym = base.replace("_ONE_MINUTE.csv", "")

    with open(fpath, "r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            results.append((sym, "EMPTY"))
            continue

        h = [c.strip().lower() for c in header]
        if h != ["datetime", "open", "high", "low", "close", "volume"]:
            results.append((sym, f"BAD_HEADER:{h}"))
            continue

        # Count rows and get first/last datetime
        row_count = 0
        first_dt = None
        last_dt = None
        for row in reader:
            if row_count == 0:
                first_dt = row[0]
            last_dt = row[0]
            row_count += 1

    header_ok += 1
    results.append((sym, "OK", first_dt, last_dt, row_count))

# Print results
print(f"{'Symbol':<20} {'Status':<20} {'First':<30} {'Last':<30} {'Rows':<10}")
print("="*110)
for r in sorted(results, key=lambda x: x[0]):
    if r[1] == "OK":
        print(f"{r[0]:<20} {r[1]:<20} {str(r[2]):<30} {str(r[3]):<30} {r[4]:<10}")
    else:
        print(f"{r[0]:<20} {r[1]:<20}")

print("\n" + "="*110)
ok = [r for r in results if r[1] == "OK"]
errors = [r for r in results if r[1] != "OK"]
print(f"\nOK:     {len(ok)}/{len(results)}")
print(f"Errors: {len(errors)}/{len(results)}")
for r in errors[:20]:
    print(f"  {r[0]}: {r[1]}")

if ok:
    # Parse all dates to find overall range
    all_first = []
    all_last = []
    total_rows = 0
    for r in ok:
        try:
            all_first.append(datetime.fromisoformat(r[2]))
            all_last.append(datetime.fromisoformat(r[3]))
            total_rows += r[4]
        except:
            pass

    if all_first:
        print(f"\nEarliest data: {min(all_first)}")
        print(f"Latest data:   {max(all_last)}")
        print(f"Total rows:    {total_rows:,}")
