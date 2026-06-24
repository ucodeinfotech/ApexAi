import os, pandas as pd

d = "nifty50_full_history"
results = []
for f in sorted(os.listdir(d)):
    if not f.endswith("_FIFTEEN_MINUTE.csv"):
        continue
    sym = f.replace("_FIFTEEN_MINUTE.csv", "")
    df = pd.read_csv(f"{d}/{f}")
    first15 = df["datetime"].min()
    last15 = df["datetime"].max()
    rows15 = len(df)
    
    f1 = f"{d}/{sym}_ONE_MINUTE.csv"
    if os.path.exists(f1):
        df1 = pd.read_csv(f1)
        first1 = df1["datetime"].min()
        last1 = df1["datetime"].max()
        rows1 = len(df1)
    else:
        first1 = last1 = rows1 = "N/A"
    
    results.append((sym, rows15, first15, last15, rows1, first1, last1))

print(f"{'Symbol':20s} {'15-min Rows':>10s} {'15-min From':>20s} {'15-min To':>20s} {'1-min Rows':>10s} {'1-min From':>20s} {'1-min To':>20s}")
print("=" * 120)
for sym, r15, f15, t15, r1, f1, t1 in results:
    print(f"{sym:20s} {r15:>10,} {f15:>20s} {t15:>20s} {str(r1):>10s} {str(f1):>20s} {str(t1):>20s}")
