"""Fetch Nifty, BankNifty, Sensex - 5min and 1hr spot data"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import time
import sys
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
    sys.exit(1)
print("OK")
time.sleep(2)

INDICES = {
    "NIFTY50":     {"token": "99926000", "exchange": "NSE"},
    "BANKNIFTY":   {"token": "99926009", "exchange": "NSE"},
    "SENSEX":      {"token": "99919000", "exchange": "BSE"},
}

ist = timezone(timedelta(hours=5, minutes=30))

def fetch_chunk(smartApi, token, exchange, interval, from_date, to_date, retries=5):
    for attempt in range(retries):
        try:
            params = {
                "exchange": exchange,
                "symboltoken": str(token),
                "interval": interval,
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

def fetch_index(smartApi, token, exchange, interval, step_days, label):
    """Fetch all available data going backwards."""
    print(f"\n  Fetching {interval} {label}...")
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    earliest = datetime(2015, 1, 1, tzinfo=ist)
    chunks = []
    empty_count = 0
    chunk_no = 0
    
    while end > earliest:
        chunk_no += 1
        start = end - timedelta(days=step_days)
        if start < earliest:
            start = earliest
        
        df = fetch_chunk(smartApi, token, exchange, interval, start, end)
        if df.empty:
            empty_count += 1
            if empty_count >= 3:
                break
            end = start
            time.sleep(1.5)
            continue
        empty_count = 0
        chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        
        if chunk_no == 1:
            print(f"    -> {len(df)} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        time.sleep(1.5)
    
    if not chunks:
        print(f"    NO DATA")
        return pd.DataFrame()
    
    result = pd.concat(chunks)
    result = result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"    Total: {len(result):,} rows | {result['datetime'].min().date()} to {result['datetime'].max().date()}")
    return result

start_time = time.time()

for name, info in INDICES.items():
    print(f"\n{'='*60}")
    print(f"{name} (token={info['token']}, ex={info['exchange']})")
    print(f"{'='*60}")
    
    # 5-min
    df5 = fetch_index(smartApi, info["token"], info["exchange"], "FIVE_MINUTE", 90, "5-min")
    if not df5.empty:
        df5.to_csv(f"{OUTPUT_DIR}/{name}_FIVE_MINUTE.csv", index=False)
    time.sleep(2)
    
    # 1-hour
    df1h = fetch_index(smartApi, info["token"], info["exchange"], "ONE_HOUR", 300, "1-hour")
    if not df1h.empty:
        df1h.to_csv(f"{OUTPUT_DIR}/{name}_ONE_HOUR.csv", index=False)
    time.sleep(2)

print(f"\nDone! Time: {(time.time()-start_time)/60:.1f} min")
print(f"\nFiles saved in {OUTPUT_DIR}/:")
import os
for f in sorted(os.listdir(OUTPUT_DIR)):
    if any(x in f for x in ["NIFTY50_", "BANKNIFTY_", "SENSEX_"]):
        sz = os.path.getsize(f"{OUTPUT_DIR}/{f}")
        print(f"  {f:45s} {sz/1024/1024:.1f} MB")
