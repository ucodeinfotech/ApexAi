"""Re-fetch 1-min data for all 50 stocks from Oct 2016 to Jan 2020 (the missing period)"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
import time
import os
import sys
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "nifty50_full_history"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Logging in...")
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print("OK")
time.sleep(2)

# Get token mapping
print("Downloading Scrip Master...")
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
token_map = {}
for item in resp.json():
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

NIFTY_50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BEL","BHARTIARTL",
    "CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL",
    "GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
    "HINDUNILVR","ICICIBANK","INDIGO","INFY","ITC",
    "JIOFIN","JSWSTEEL","KOTAKBANK","LT","M&M",
    "MARUTI","MAXHEALTH","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBILIFE","SBIN","SHRIRAMFIN",
    "SUNPHARMA","TATACONSUM","TATASTEEL","TATAMOTORS","TCS",
    "TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO"
]
tokens = {}
for sym in NIFTY_50:
    t = token_map.get(sym)
    if t:
        tokens[sym] = t
tokens["TATAMOTORS"] = "3456"
print(f"Mapped {len(tokens)} stocks")

def fetch_chunk(smartApi, token, interval, from_date, to_date, retries=5):
    for attempt in range(retries):
        try:
            params = {
                "exchange": "NSE", "symboltoken": str(token), "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            candles = smartApi.getCandleData(params)
            if candles["status"] and candles["data"]:
                df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            return pd.DataFrame()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            continue
    return pd.DataFrame()

ist = timezone(timedelta(hours=5, minutes=30))

def fetch_missing_1min(smartApi, token, sym):
    """Fetch 1-min data from Oct 2016 to Jan 2020 only, then merge with existing."""
    print(f"  Fetching 1-min pre-2020 data for {sym}...")
    
    # Step backwards from 2020-01-01 to Oct 2016 in 35-day steps
    end = datetime(2020, 1, 1, tzinfo=ist)
    earliest = datetime(2016, 10, 1, tzinfo=ist)
    chunks = []
    empty_count = 0
    
    while end > earliest:
        start = end - timedelta(days=35)
        if start < earliest:
            start = earliest
        
        df = fetch_chunk(smartApi, token, "ONE_MINUTE", start, end)
        if df.empty:
            empty_count += 1
            if empty_count >= 3:
                break
            end = start
            time.sleep(2)
            continue
        empty_count = 0
        chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        time.sleep(2)
    
    if not chunks:
        print(f"    No pre-2020 data found")
        return False
    
    new_df = pd.concat(chunks)
    new_df = new_df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"    Pre-2020: {len(new_df):,} rows | {new_df['datetime'].min().date()} to {new_df['datetime'].max().date()}")
    
    # Merge with existing
    existing_path = f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv"
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        existing["datetime"] = pd.to_datetime(existing["datetime"])
        print(f"    Existing: {len(existing):,} rows | {existing['datetime'].min().date()} to {existing['datetime'].max().date()}")
        
        merged = pd.concat([new_df, existing])
        merged = merged.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        print(f"    Merged: {len(merged):,} rows | {merged['datetime'].min().date()} to {merged['datetime'].max().date()}")
        merged.to_csv(existing_path, index=False)
    else:
        new_df.to_csv(existing_path, index=False)
        print(f"    Saved {len(new_df):,} rows (no existing file)")
    
    return True

start_time = time.time()
total = len(tokens)

for idx, (sym, token) in enumerate(tokens.items(), 1):
    print(f"\n[{idx}/{total}] {sym}")
    fetch_missing_1min(smartApi, token, sym)
    elapsed = time.time() - start_time
    eta = (elapsed / idx) * (total - idx)
    print(f"  [ETA: {eta/60:.0f} min]")

print(f"\nDone! Total time: {(time.time()-start_time)/60:.1f} minutes")
