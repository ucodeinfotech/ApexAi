import pandas as pd
import os

DATA_DIR = "nifty50_full_history"
INDICES = ["NIFTY50", "BANKNIFTY", "SENSEX"]
SRC_TF = "FIVE_MINUTE"
TARGET_TFS = {
    "FIFTEEN_MINUTE": "15min",
    "THIRTY_MINUTE": "30min",
    "ONE_HOUR": "1h",
    "ONE_DAY": "D",
}

def resample_file(sym, src_tf, target_name, rule):
    src_path = f"{DATA_DIR}/{sym}_{src_tf}.csv"
    dst_path = f"{DATA_DIR}/{sym}_{target_name}.csv"
    if not os.path.exists(src_path):
        print(f"  SKIP: {src_path} not found")
        return
    # Check if target already exists (skip)
    if os.path.exists(dst_path):
        print(f"  EXISTS: {dst_path}")
        return
    print(f"  Resampling {sym} {src_tf} -> {target_name} ({rule})")
    df = pd.read_csv(src_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    resampled.to_csv(dst_path, index=False)
    print(f"    Saved {len(resampled)} rows -> {dst_path}")

def main():
    for sym in INDICES:
        print(f"\n=== {sym} ===")
        for tf_name, rule in TARGET_TFS.items():
            resample_file(sym, SRC_TF, tf_name, rule)

if __name__ == "__main__":
    main()
