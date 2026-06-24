import os, pandas as pd

d = "nifty50_full_history"
results = []
for f in sorted(os.listdir(d)):
    if not f.endswith("_FIFTEEN_MINUTE.csv"):
        continue
    sym = f.replace("_FIFTEEN_MINUTE.csv", "")
    df = pd.read_csv(f"{d}/{f}")
    r15, f15, t15 = len(df), df["datetime"].min(), df["datetime"].max()
    
    f1 = f"{d}/{sym}_ONE_MINUTE.csv"
    if os.path.exists(f1):
        df1 = pd.read_csv(f1)
        r1, f1_d, t1_d = len(df1), df1["datetime"].min(), df1["datetime"].max()
    else:
        r1, f1_d, t1_d = "N/A", "N/A", "N/A"
    
    results.append((sym, r15, str(f15)[:10], str(t15)[:10], r1, str(f1_d)[:10], str(t1_d)[:10]))

results.sort(key=lambda x: x[4] if isinstance(x[4], int) else 0)

print(f"{'Symbol':20s} {'15-min Rows':>10s} {'15-min From':>12s} {'15-min To':>12s} {'1-min Rows':>10s} {'1-min From':>12s} {'1-min To':>12s}")
print("=" * 90)
for sym, r15, f15, t15, r1, f1, t1 in results:
    print(f"{sym:20s} {r15:>10,} {f15:>12s} {t15:>12s} {r1:>10,} {f1:>12s} {t1:>12s}")

total15 = sum(r[1] for r in results)
total1 = sum(r[4] for r in results if isinstance(r[4], int))
print(f"\n{'TOTAL':20s} {total15:>10,} {'':>12s} {'':>12s} {total1:>10,}")
print(f"Combined: {total15 + total1:,} rows, {(total15 + total1) * 50 / 1024 / 1024:.1f} MB")
