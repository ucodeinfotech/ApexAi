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
ist = timezone(timedelta(hours=5, minutes=30))
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

# Test: can we get 1-min data from a specific month in the past?
print("\n=== SBIN 1-min: Targeted month queries ===")
for month_pair in [("2019-01-01","2019-02-01"), ("2019-06-01","2019-07-01"), 
                    ("2018-01-01","2018-02-01"), ("2018-06-01","2018-07-01"),
                    ("2017-01-01","2017-02-01"), ("2017-06-01","2017-07-01"),
                    ("2016-10-03","2016-11-03")]:
    fr = datetime.strptime(month_pair[0], "%Y-%m-%d").replace(tzinfo=ist)
    to = datetime.strptime(month_pair[1], "%Y-%m-%d").replace(tzinfo=ist)
    df = fetch_chunk(smartApi, "3045", "ONE_MINUTE", fr, to)
    if not df.empty:
        print(f"  {month_pair[0]}: {len(df):,} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    else:
        print(f"  {month_pair[0]}: EMPTY")
    time.sleep(2)

# Test: Full year 2019 in 1-min - can we get all of it?
print("\n=== SBIN 1-min: Full year 2019 in 1 chunk ===")
df = fetch_chunk(smartApi, "3045", "ONE_MINUTE",
    datetime(2019, 1, 1, tzinfo=ist),
    datetime(2020, 1, 1, tzinfo=ist))
if not df.empty:
    print(f"  {len(df):,} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
else:
    print("  EMPTY")
