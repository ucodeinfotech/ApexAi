"""Test: download 1-min data for 10 new stocks, show results"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
import time
import sys
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

print("Logging in...")
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print("OK")
time.sleep(2)

# 10 test stocks with their tokens
test_stocks = [
    ("MANKIND", "15380"),
    ("LUPIN", "10440"),
    ("MARICO", "4067"),
    ("POLYCAB", "9590"),
    ("PERSISTENT", "18365"),
    ("SRF", "3273"),
    ("SUPREMEIND", "3363"),
    ("VOLTAS", "3718"),
    ("YESBANK", "11915"),
    ("FORTIS", "14592"),
]

ist = timezone(timedelta(hours=5, minutes=30))

def fetch_chunk(smartApi, token, from_date, to_date):
    params = {
        "exchange": "NSE", "symboltoken": str(token), "interval": "ONE_MINUTE",
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M")
    }
    try:
        candles = smartApi.getCandleData(params)
        if candles["status"] and candles["data"]:
            df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
    except Exception as e:
        pass
    return pd.DataFrame()

print(f"\n{'='*80}")
print(f"{'Symbol':<15} {'Recent 60d':>10} {'2016-1mo':>10} {'2020-1mo':>10} {'Earliest Date':>15}")
print(f"{'='*80}")

now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)

for sym, token in test_stocks:
    # Test 1: recent 60 days
    df_now = fetch_chunk(smartApi, token, now - timedelta(days=60), now)
    
    # Test 2: Oct 2016 (earliest known)
    df_2016 = fetch_chunk(smartApi, token, 
        datetime(2016, 10, 1, tzinfo=ist),
        datetime(2016, 11, 15, tzinfo=ist))
    
    # Test 3: Jan 2020
    df_2020 = fetch_chunk(smartApi, token,
        datetime(2020, 1, 1, tzinfo=ist),
        datetime(2020, 2, 1, tzinfo=ist))
    
    # Find earliest date by probing backwards
    earliest = None
    probe_dates = [2016, 2017, 2018, 2019, 2020, 2021]
    for yr in probe_dates:
        df = fetch_chunk(smartApi, token,
            datetime(yr, 1, 1, tzinfo=ist),
            datetime(yr, 2, 1, tzinfo=ist))
        if not df.empty:
            earliest = str(df["datetime"].min().date())
            break
    
    print(f"{sym:<15} {len(df_now):>10} {len(df_2016):>10} {len(df_2020):>10} {str(earliest or 'N/A'):>15}")
    time.sleep(1.5)

print(f"{'='*80}")
print("Done!")
