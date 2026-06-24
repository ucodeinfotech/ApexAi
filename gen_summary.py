import os, pandas as pd

d = "nifty50_full_history"
rows = []
for f in sorted(os.listdir(d)):
    if not (f.endswith("_MINUTE.csv") or f.endswith("_HOUR.csv") or f.endswith("_DAY.csv")):
        continue
    sym = f.replace("_FIFTEEN_MINUTE.csv","").replace("_ONE_MINUTE.csv","").replace("_FIVE_MINUTE.csv","").replace("_ONE_HOUR.csv","").replace("_ONE_DAY.csv","")
    sz = os.path.getsize(f"{d}/{f}")
    try:
        total_rows = sum(1 for _ in open(f"{d}/{f}")) - 1
    except:
        total_rows = 0
    rows.append({"file": f, "symbol": sym, "size_mb": round(sz/1024/1024, 1), "rows": total_rows})

df = pd.DataFrame(rows)
summary = df.groupby("symbol").agg({"size_mb": "sum", "rows": "sum"}).sort_values("symbol")
print("=" * 70)
print(f"ALL 50 NIFTY 50 STOCKS - DATA SUMMARY")
print(f"Total files: {len(df)}")
print(f"Total size: {df['size_mb'].sum():.0f} MB")
print(f"Total rows: {df['rows'].sum():,}")
print("=" * 70)
print()
for sym, row in summary.iterrows():
    print(f"{sym:20s} {row['size_mb']:5.1f} MB  {row['rows']:>8,} rows")
print()
print(f"{'TOTAL':20s} {summary['size_mb'].sum():5.1f} MB  {summary['rows'].sum():>8,} rows")
