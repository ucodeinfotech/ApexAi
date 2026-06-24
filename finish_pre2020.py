"""Fetch pre-2020 1-min data for remaining stocks"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "nifty50_full_history"

print("Logging in...")
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    exit()
print("OK")
ist = timezone(timedelta(hours=5, minutes=30))
time.sleep(2)

TOKENS = {"RELIANCE":"2885","SBILIFE":"21808","SBIN":"3045","SUNPHARMA":"3351","TATACONSUM":"3432",
          "TATAMOTORS":"3456","TATASTEEL":"3499","TCS":"11536","TECHM":"13538","TITAN":"3506",
          "TRENT":"1964","ULTRACEMCO":"11532","WIPRO":"3787","HDFCLIFE":"467"}

def fetch_chunk(smartApi, token, interval, from_date, to_date, retries=5):
    for attempt in range(retries):
        try:
            params = {"exchange":"NSE","symboltoken":str(token),"interval":interval,
                "fromdate":from_date.strftime("%Y-%m-%d %H:%M"),
                "todate":to_date.strftime("%Y-%m-%d %H:%M")}
            candles = smartApi.getCandleData(params)
            if candles["status"] and candles["data"]:
                df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            return pd.DataFrame()
        except:
            if attempt < retries-1: time.sleep(10)
    return pd.DataFrame()

total = len(TOKENS)
for idx, (sym, token) in enumerate(TOKENS.items(), 1):
    print(f"\n[{idx}/{total}] {sym}")
    end = datetime(2020,1,1,tzinfo=ist)
    earliest = datetime(2016,10,1,tzinfo=ist)
    chunks = []
    empty = 0
    while end > earliest:
        start = end - timedelta(days=35)
        if start < earliest: start = earliest
        df = fetch_chunk(smartApi, token, "ONE_MINUTE", start, end)
        if df.empty:
            empty += 1
            if empty >= 3: break
            end = start
            time.sleep(2)
            continue
        empty = 0
        chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        time.sleep(2)
    
    if not chunks:
        print("  No pre-2020 data")
        continue
    
    new = pd.concat(chunks).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"  Pre-2020: {len(new):,} rows | {new['datetime'].min().date()} to {new['datetime'].max().date()}")
    
    exist_path = f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv"
    exist = pd.read_csv(exist_path)
    exist["datetime"] = pd.to_datetime(exist["datetime"])
    print(f"  Existing: {len(exist):,} rows | {exist['datetime'].min().date()} to {exist['datetime'].max().date()}")
    
    merged = pd.concat([exist, new]).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    merged.to_csv(exist_path, index=False)
    print(f"  Merged:  {len(merged):,} rows | {merged['datetime'].min().date()} to {merged['datetime'].max().date()}")

print("\nDone!")
