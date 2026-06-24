import os

d = "nifty50_full_history"
results = []
for f in sorted(os.listdir(d)):
    if not f.endswith("_FIFTEEN_MINUTE.csv"):
        continue
    sym = f.replace("_FIFTEEN_MINUTE.csv", "")
    mtime = os.path.getmtime(f"{d}/{f}")
    with open(f"{d}/{f}") as fh:
        lines = fh.readlines()
    r15 = len(lines) - 1
    f15 = lines[1].split(",")[0][:10] if len(lines) > 1 else "?"
    t15 = lines[-1].split(",")[0][:10] if len(lines) > 1 else "?"
    
    f1 = f"{d}/{sym}_ONE_MINUTE.csv"
    if os.path.exists(f1):
        with open(f1) as fh:
            lines = fh.readlines()
        r1 = len(lines) - 1
        f1_d = lines[1].split(",")[0][:10] if len(lines) > 1 else "?"
        t1_d = lines[-1].split(",")[0][:10] if len(lines) > 1 else "?"
    else:
        r1, f1_d, t1_d = "N/A", "N/A", "N/A"
    
    results.append((sym, r15, f15, t15, r1, f1_d, t1_d, mtime))

results.sort(key=lambda x: x[7])  # sort by modified time (most recent last)

print(f"{'Symbol':20s} {'15-min Rows':>10s} {'15-min From':>12s} {'15-min To':>12s} {'1-min Rows':>10s} {'1-min From':>12s} {'1-min To':>12s}")
print("=" * 90)
total15 = total1 = 0
for sym, r15, f15, t15, r1, f1, t1, _ in results:
    print(f"{sym:20s} {r15:>10,} {f15:>12s} {t15:>12s} {r1:>10,} {f1:>12s} {t1:>12s}")
    total15 += r15
    total1 += r1 if isinstance(r1, int) else 0

print(f"\n{'TOTAL':20s} {total15:>10,} {'':>12s} {'':>12s} {total1:>10,}")
# Count stocks at Oct 2016 level
oct2016 = sum(1 for _,_,f,_,_,_,_,_ in results if f == "2016-10-03")
print(f"\n{oct2016}/50 stocks have full 15-min history from Oct 3, 2016")
print(f"File count: {len([f for f in os.listdir(d) if f.endswith('.csv') and not f.startswith('_')])}")
