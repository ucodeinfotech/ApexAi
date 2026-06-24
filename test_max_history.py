"""
Test max historical data availability for each interval type.
"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed")
    exit()
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
print(f"RELIANCE token: {token}")
time.sleep(2)

def fetch(smartApi, token, interval, fromdate, todate):
    params = {
        "exchange": "NSE",
        "symboltoken": str(token),
        "interval": interval,
        "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
        "todate": todate.strftime("%Y-%m-%d %H:%M")
    }
    try:
        candles = smartApi.getCandleData(params)
        if candles["status"] and candles["data"]:
            return len(candles["data"]), candles["data"][0][0], candles["data"][-1][0]
        return 0, None, None
    except Exception as e:
        return -1, str(e)[:50], None

print("\n" + "=" * 70)
print("TEST 1: Probing 1-min data depth (stepping backwards)")
print("=" * 70)

now = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
for days_back in [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]:
    start = now - timedelta(days=days_back)
    n, first, last = fetch(smartApi, token, "ONE_MINUTE", start, now)
    status = "OK" if n > 0 else ("EMPTY" if n == 0 else "ERROR")
    if n > 0:
        print(f"  {days_back:2d} day range: {n:4d} candles | {str(first)[:16]} to {str(last)[:16]} [{status}]")
    else:
        print(f"  {days_back:2d} day range: {status}")
    time.sleep(1.5)

print("\n" + "=" * 70)
print("TEST 2: Probing 15-min data depth (stepping backwards aggressively)")
print("=" * 70)

for days_back in [30, 60, 90, 180, 365, 400, 500, 730, 1095, 1460, 1825]:
    start = now - timedelta(days=days_back)
    n, first, last = fetch(smartApi, token, "FIFTEEN_MINUTE", start, now)
    status = "OK" if n > 0 else ("EMPTY" if n == 0 else "ERROR")
    if n > 0:
        print(f"  {days_back:4d} day range (~{days_back//365}y {days_back%365//30}m): {n:4d} candles | {str(first)[:16]} to {str(last)[:16]} [{status}]")
    else:
        print(f"  {days_back:4d} day range (~{days_back//365}y {days_back%365//30}m): {status}")
    time.sleep(2)

print("\n" + "=" * 70)
print("TEST 3: Testing ONE_DAY interval (max data)")
print("=" * 70)

for days_back in [365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3650]:
    start = now - timedelta(days=days_back)
    n, first, last = fetch(smartApi, token, "ONE_DAY", start, now)
    status = "OK" if n > 0 else ("EMPTY" if n == 0 else "ERROR")
    if n > 0:
        print(f"  {days_back:4d} day range (~{days_back//365}y): {n:4d} candles | {str(first)[:16]} to {str(last)[:16]} [{status}]")
    else:
        print(f"  {days_back:4d} day range (~{days_back//365}y): {status}")
    time.sleep(2)

print("\n" + "=" * 70)
print("TEST 4: Testing ONE_HOUR interval")
print("=" * 70)

for days_back in [60, 180, 365, 400, 500]:
    start = now - timedelta(days=days_back)
    n, first, last = fetch(smartApi, token, "ONE_HOUR", start, now)
    status = "OK" if n > 0 else ("EMPTY" if n == 0 else "ERROR")
    if n > 0:
        print(f"  {days_back:4d} day range: {n:4d} candles | {str(first)[:16]} to {str(last)[:16]} [{status}]")
    else:
        print(f"  {days_back:4d} day range: {status}")
    time.sleep(2)
