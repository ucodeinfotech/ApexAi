"""Generate missing timeframes for nifty50_full_history stocks"""
import os
import pandas as pd

SRC_DIR = "nifty50_full_history"

for f in os.listdir(SRC_DIR):
    if not f.endswith("_ONE_MINUTE.csv"):
        continue
    sym = f.replace("_ONE_MINUTE.csv", "")
    base = f"{SRC_DIR}/{sym}"
    
    # Check what's missing
    needed = []
    for tf in ["FIVE_MINUTE", "ONE_HOUR", "ONE_DAY"]:
        if not os.path.exists(f"{base}_{tf}.csv"):
            needed.append(tf)
    
    if not needed:
        continue
    
    print(f"{sym}: generating {needed}")
    df = pd.read_csv(f"{base}_ONE_MINUTE.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    
    if "FIVE_MINUTE" in needed:
        res = df.resample("5min").agg(ohlc).dropna()
        res.to_csv(f"{base}_FIVE_MINUTE.csv", index_label="datetime")
    if "ONE_HOUR" in needed:
        res = df.resample("60min").agg(ohlc).dropna()
        res.to_csv(f"{base}_ONE_HOUR.csv", index_label="datetime")
    if "ONE_DAY" in needed:
        res = df.resample("D").agg(ohlc).dropna()
        res.to_csv(f"{base}_ONE_DAY.csv", index_label="datetime")

print("Done generating missing timeframes.")
