"""Phase 2: resample all 1-min CSVs to higher timeframes"""
import os, pandas as pd

OUTPUT_DIR = "comprehensive_data"
NIFY_DIR = "nifty50_full_history"

for d in [OUTPUT_DIR, NIFY_DIR]:
    if not os.path.exists(d): continue
    for f in os.listdir(d):
        if not f.endswith("_ONE_MINUTE.csv"): continue
        sym = f.replace("_ONE_MINUTE.csv", "")
        
        # Check if all resampled files already exist
        all_done = True
        for tf in ["FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "ONE_DAY"]:
            if not os.path.exists(f"{d}/{sym}_{tf}.csv"):
                all_done = False
                break
        if all_done:
            continue
        
        print(f"Resampling {sym}...", end=" ", flush=True)
        df1 = pd.read_csv(f"{d}/{sym}_ONE_MINUTE.csv")
        df1["datetime"] = pd.to_datetime(df1["datetime"])
        df = df1.set_index("datetime")
        
        for name, rule in [("FIVE_MINUTE","5min"),("FIFTEEN_MINUTE","15min"),("ONE_HOUR","1h"),("ONE_DAY","1D")]:
            td = df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
            td.to_csv(f"{d}/{sym}_{name}.csv", index=False)
        
        print("done")

print("All done!")
