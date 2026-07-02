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

print("Logging in...")
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print("OK")
time.sleep(2)

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

ist = timezone(timedelta(hours=5, minutes=30))

# Test 1: SBIN - 1-min from 2016-10 to 2020-01
print("\n=== SBIN 1-min: Oct 2016 to Jan 2020 ===")
df = fetch_chunk(smartApi, "3045", "ONE_MINUTE",
    datetime(2016, 10, 3, tzinfo=ist),
    datetime(2020, 1, 31, tzinfo=ist))
if not df.empty:
    print(f"  {len(df):,} rows | {df['datetime'].min()} to {df['datetime'].max()}")
else:
    print("  EMPTY")
time.sleep(2)

# Test 2: Try smaller chunks to probe
print("\n=== SBIN 1-min: Chunk probes backwards ===")
for yr in [2019, 2018, 2017, 2016]:
    df = fetch_chunk(smartApi, "3045", "ONE_MINUTE",
        datetime(yr, 6, 1, tzinfo=ist),
        datetime(yr, 12, 31, tzinfo=ist))
    if not df.empty:
        print(f"  {yr}: {len(df):,} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    else:
        print(f"  {yr}: EMPTY")
    time.sleep(2)

# Test 3: SBIN - 5-min from 2016 to see if 5-min has more depth than 1-min
print("\n=== SBIN 5-min: Oct 2016 to Jan 2020 ===")
df5 = fetch_chunk(smartApi, "3045", "FIVE_MINUTE",
    datetime(2016, 10, 3, tzinfo=ist),
    datetime(2020, 1, 31, tzinfo=ist))
if not df5.empty:
    print(f"  {len(df5):,} rows | {df5['datetime'].min()} to {df5['datetime'].max()}")
else:
    print("  EMPTY")
time.sleep(2)

# Test 4: RELIANCE 1-min before 2020
print("\n=== RELIANCE 1-min: Oct 2016 to Jan 2020 ===")
df = fetch_chunk(smartApi, "2885", "ONE_MINUTE",
    datetime(2016, 10, 3, tzinfo=ist),
    datetime(2020, 1, 31, tzinfo=ist))
if not df.empty:
    print(f"  {len(df):,} rows | {df['datetime'].min()} to {df['datetime'].max()}")
else:
    print("  EMPTY")
