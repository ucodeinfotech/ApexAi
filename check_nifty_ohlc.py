"""Check NIFTY50 OHLC inconsistencies in detail"""
import pandas as pd
import os

d = "nifty50_full_history"
for f in ["NIFTY50_FIVE_MINUTE.csv", "NIFTY50_ONE_HOUR.csv"]:
    df = pd.read_csv(f"{d}/{f}")
    bad = df[(df["low"] > df["open"]) | (df["low"] > df["close"]) | 
             (df["high"] < df["open"]) | (df["high"] < df["close"])]
    print(f"\n{f}: {len(bad)} OHLC issues out of {len(df):,} rows")
    if len(bad) > 0:
        print(f"  Max deviation: low-open={max(bad['low']-bad['open']):.4f}, high-close={max(bad['high']-bad['close']):.4f}")
        print(f"  Sample rows:")
        for _, r in bad.head(5).iterrows():
            print(f"    {r['datetime']} O={r['open']} H={r['high']} L={r['low']} C={r['close']}")
