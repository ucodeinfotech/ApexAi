"""
Continue fetching remaining stocks.
"""
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
print(f"Logged in as {CLIENT_ID}")
time.sleep(2)

print("Downloading Scrip Master...")
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

TOKEN_MAPPING = {}
remaining = [
    "MAXHEALTH","NESTLEIND","NTPC","ONGC","POWERGRID","SBILIFE","SBIN","SHRIRAMFIN",
    "SUNPHARMA","TATACONSUM","TATASTEEL","TATAMOTORS","TCS","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO"
]
for sym in remaining:
    t = token_map.get(sym)
    if t:
        TOKEN_MAPPING[sym] = t
TOKEN_MAPPING["TATAMOTORS"] = "3456"
print(f"Will fetch {len(TOKEN_MAPPING)} stocks")

def fetch_chunk(smartApi, token, interval, from_date, to_date, retries=3):
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

def fetch_full_history(smartApi, token, interval, step_days, start_year=2016):
    all_chunks = []
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    earliest = datetime(start_year, 1, 1, tzinfo=ist)
    chunk_no = 0
    empty_count = 0
    while end > earliest:
        chunk_no += 1
        start = end - timedelta(days=step_days)
        if start < earliest:
            start = earliest
        df = fetch_chunk(smartApi, token, interval, start, end)
        if df.empty:
            empty_count += 1
            if empty_count >= 3:
                break
            end = start
            time.sleep(1.5)
            continue
        empty_count = 0
        all_chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        if chunk_no == 1:
            print(f"    -> {len(df)} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        time.sleep(1.5)
    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks)
    return result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

start_time = time.time()
total = len(TOKEN_MAPPING)

for idx, (symbol, token) in enumerate(TOKEN_MAPPING.items(), 1):
    print(f"\n[{idx}/{total}] {symbol} (token: {token})")
    
    # 15-min
    print(f"  15-min...")
    df15 = fetch_full_history(smartApi, token, "FIFTEEN_MINUTE", 150, 2016)
    if not df15.empty:
        df15.to_csv(f"{OUTPUT_DIR}/{symbol}_FIFTEEN_MINUTE.csv", index=False)
        print(f"    -> {len(df15):,} rows | {df15['datetime'].min().date()} to {df15['datetime'].max().date()}")
    time.sleep(2)
    
    # 1-min
    print(f"  1-min...")
    df1 = fetch_full_history(smartApi, token, "ONE_MINUTE", 90, 2020)
    if not df1.empty:
        df1.to_csv(f"{OUTPUT_DIR}/{symbol}_ONE_MINUTE.csv", index=False)
        print(f"    -> {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}")
    time.sleep(2)
    
    # Verify
    if not df1.empty and not df15.empty:
        resampled = df1.set_index("datetime").resample("15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
        df15_m = df15.copy()
        df15_m["rounded"] = df15_m["datetime"].dt.floor("15min")
        resampled["rounded"] = resampled["datetime"].dt.floor("15min")
        merged = pd.merge(resampled, df15_m, on="rounded", how="inner", suffixes=("_r","_n"))
        matched = (abs(merged["close_r"] - merged["close_n"]) <= 0.05).sum()
        print(f"  Verify: {matched}/{len(merged)} bars match")
    
    elapsed = time.time() - start_time
    per_stock = elapsed / idx
    remaining_eta = per_stock * (total - idx)
    print(f"  [ETA: {remaining_eta/60:.0f} min]")

print(f"\nDone! Total: {(time.time()-start_time)/60:.1f} min")
