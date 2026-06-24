"""
Deep probe: find exact max historical data depth for ALL intervals
Uses chunking + retries to bypass rate limiting
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

# Get token
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip = resp.json()
token = None
for item in scrip:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ") and item["name"] == "RELIANCE":
        token = item["token"]
        break

def fetch_safe(smartApi, token, interval, fromdate, todate, retries=3):
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
    return None  # None = error after retries

def probe_depth(interval, label, start_year, end_year, step_months=6):
    """Probe data availability by stepping backwards in large chunks."""
    print(f"\n--- {label} ({interval}) ---")
    now = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Step 1: Find the boundary using binary-search style probing
    for year in range(end_year, start_year - 1, -1):
        start = datetime(year, 1, 1)
        if interval == "ONE_DAY":
            s = start.strftime("%Y-%m-%d 00:00")
            e = now.strftime("%Y-%m-%d 00:00")
        else:
            s = start.strftime("%Y-%m-%d 09:15")
            e = now.strftime("%Y-%m-%d 15:30")
        
        data = fetch_safe(smartApi, token, interval, start, now)
        if data is None:
            print(f"  {year}: ERROR (rate limited)")
            time.sleep(5)
            continue
        elif len(data) == 0:
            print(f"  {year}: EMPTY - data boundary found")
            # Try the midpoint to narrow down
            mid_start = start + timedelta(days=182)  # ~6 months in
            data2 = fetch_safe(smartApi, token, interval, mid_start, now)
            if data2 and len(data2) > 0:
                print(f"    -> Data available from ~{mid_start.date()}")
                # Now probe the exact boundary
                probe_boundary(interval, mid_start, start, now)
            break
        else:
            # Count unique dates
            dates = set(d[0][:10] for d in data if d and d[0])
            print(f"  {year}: {len(data):5d} rows, {len(dates)} trading days | {data[0][0][:10]} to {data[-1][0][:10]}")
        time.sleep(2)

def probe_boundary(interval, high, low, end_date):
    """Binary search to find exact start date of available data."""
    for _ in range(5):  # 5 iterations = ~1 month precision
        mid = low + (high - low) // 2
        data = fetch_safe(smartApi, token, interval, mid, end_date)
        if data and len(data) > 0:
            high = mid
        else:
            low = mid
        time.sleep(2)
    data = fetch_safe(smartApi, token, interval, high, end_date)
    if data and len(data) > 0:
        print(f"    Exact start: {data[0][0][:10]}")

print("=" * 70)
print("PROBING MAX HISTORICAL DATA DEPTH FOR RELIANCE")
print("=" * 70)

# Test ONE_DAY from 2015 to 2026
probe_depth("ONE_DAY", "DAILY", 2015, 2026)

# Test FIFTEEN_MINUTE from 2020 to 2026
probe_depth("FIFTEEN_MINUTE", "15-MIN", 2020, 2026)

# Test ONE_HOUR from 2020 to 2026
probe_depth("ONE_HOUR", "1-HOUR", 2020, 2026)

# Test ONE_MINUTE from 2025 to 2026
probe_depth("ONE_MINUTE", "1-MIN", 2025, 2026)

print("\n" + "=" * 70)
print("DONE PROBING")
print("=" * 70)
