import os, pandas as pd

d = "nifty50_full_history"
rows = []
for f in sorted(os.listdir(d)):
    if not f.endswith("_FIFTEEN_MINUTE.csv"):
        continue
    sym = f.replace("_FIFTEEN_MINUTE.csv","")
    # Skip indices
    if sym in ("NIFTY50","BANKNIFTY","SENSEX"):
        continue
    df = pd.read_csv(f"{d}/{f}")
    r15 = len(df)
    f15 = df["datetime"].min()[:10]
    t15 = df["datetime"].max()[:10]
    
    f1 = f"{d}/{sym}_ONE_MINUTE.csv"
    if os.path.exists(f1):
        df1 = pd.read_csv(f1)
        r1 = len(df1)
        f1_d = df1["datetime"].min()[:10]
        t1_d = df1["datetime"].max()[:10]
    else:
        r1, f1_d, t1_d = 0, "N/A", "N/A"
    
    extra = ""
    for ext in ["FIVE_MINUTE","ONE_HOUR","ONE_DAY"]:
        if os.path.exists(f"{d}/{sym}_{ext}.csv"):
            extra += f" {ext}"
    
    rows.append((sym, r15, f15, t15, r1, f1_d, t1_d, extra.strip()))

print(f"{'Symbol':18s} {'15-min Rows':>10s} {'15-min From':>14s} {'15-min To':>14s} {'1-min Rows':>10s} {'1-min From':>14s} {'1-min To':>14s} {'Extra':>20s}")
print("=" * 114)
total15 = total1 = 0
for sym, r15, f15, t15, r1, f1, t1, ext in rows:
    print(f"{sym:18s} {r15:>10,} {f15:>14s} {t15:>14s} {r1:>10,} {f1:>14s} {t1:>14s} {ext:>20s}")
    total15 += r15
    total1 += r1
print("=" * 114)
print(f"{'TOTAL':18s} {total15:>10,} {'' :>14s} {'' :>14s} {total1:>10,}")

print(f"\nBacktest will use: 15-min for signals + 1-min for entries")
print(f"Total data: {total15 + total1:,} rows across {len(rows)} stocks")
