"""
Chunked fetch going back to 2016 to find the REAL data boundary
"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import sys

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed")
    sys.exit(1)
print("Logged in")
time.sleep(2)

resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip = resp.json()
token = None
for item in scrip:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ") and item["name"] == "RELIANCE":
        token = item["token"]
        break

def fetch_chunk(smartApi, token, interval, fromdate, todate, retries=3):
    for attempt in range(retries):
        try:
            params = {
                "exchange": "NSE",
                "symboltoken": str(token),
                "interval": interval,
                "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
                "todate": todate.strftime("%Y-%m-%d %H:%M")
            }
            candles = smartApi.getCandleData(params)
            if candles["status"] and candles["data"]:
                return candles["data"]
            return []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            continue
    return None

def chunked_fetch(interval, step_days, start_year=2016):
    """Fetch in chunks going backwards and track when data stops."""
    print(f"\n{'='*70}")
    print(f"CHUNKED FETCH: {interval} | step={step_days} days")
    print(f"{'='*70}")
    
    now = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    chunk_no = 0
    total_rows = 0
    first_date = None
    
    while True:
        chunk_no += 1
        start = end - timedelta(days=step_days)
        if start.year < start_year - 1:
            start = datetime(start_year - 1, 1, 1)
        
        data = fetch_chunk(smartApi, token, interval, start, end)
        if data is None:
            print(f"  Chunk {chunk_no}: ERROR (rate limit) - retrying...")
            time.sleep(10)
            continue
        elif len(data) == 0:
            if first_date:
                print(f"  Chunk {chunk_no}: EMPTY - REACHED END")
            else:
                print(f"  Chunk {chunk_no}: EMPTY - no data at all")
            break
        
        total_rows += len(data)
        chunk_first = data[0][0][:10]
        chunk_last = data[-1][0][:10]
        if first_date is None:
            first_date = chunk_first
        
        print(f"  Chunk {chunk_no}: {start.date()} to {end.date():15s} -> {len(data):4d} rows | {chunk_first} to {chunk_last}")
        
        end = start
        if end.year < start_year:
            break
        time.sleep(1.5)
    
    print(f"\n  TOTAL: {total_rows} rows | First data: {first_date}")

# Test with chunked fetching for each interval
chunked_fetch("ONE_DAY", step_days=365, start_year=2016)       # 1-year chunks for daily
time.sleep(3)
chunked_fetch("FIFTEEN_MINUTE", step_days=60, start_year=2016)  # 60-day chunks for 15-min
time.sleep(3)
chunked_fetch("ONE_MINUTE", step_days=15, start_year=2016)      # 15-day chunks for 1-min

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
