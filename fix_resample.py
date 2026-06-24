"""Fix missing resampled files for stocks that have 1-min data"""
import os, pandas as pd

OUTPUT_DIR = "comprehensive_data"
fixed = 0
for f in os.listdir(OUTPUT_DIR):
    if not f.endswith("_ONE_MINUTE.csv"):
        continue
    sym = f.replace("_ONE_MINUTE.csv", "")
    need_resample = False
    for tf in ["FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "ONE_DAY"]:
        if not os.path.exists(f"{OUTPUT_DIR}/{sym}_{tf}.csv"):
            need_resample = True
            break
    if not need_resample:
        continue
    
    print(f"Resampling {sym}...", end=" ", flush=True)
    df1 = pd.read_csv(f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv")
    df1["datetime"] = pd.to_datetime(df1["datetime"])
    df = df1.set_index("datetime")
    
    for name, rule in [("FIVE_MINUTE","5min"),("FIFTEEN_MINUTE","15min"),("ONE_HOUR","1h"),("ONE_DAY","1D")]:
        td = df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
        td.to_csv(f"{OUTPUT_DIR}/{sym}_{name}.csv", index=False)
    
    print("done")
    fixed += 1

print(f"Fixed {fixed} stocks")
