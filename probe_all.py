from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime
import time

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed")
    exit()
print("Login OK")
time.sleep(2)

def get_historical_data(smartApi, symboltoken, from_date, to_date=None, exchange="NSE", interval="FIVE_MINUTE"):
    try:
        if to_date is None:
            to_date = datetime.now()
        params = {
            "exchange": exchange,
            "symboltoken": str(symboltoken),
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
        candles = smartApi.getCandleData(params)
        if not candles.get("status", False):
            return pd.DataFrame()
        df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

# Test 1: Probe FIVE_MINUTE depth for SBIN - how far back per call
print("=== FIVE_MINUTE: How far back per call? ===")
for yr in [2016, 2017, 2018, 2019, 2020, 2021]:
    df = get_historical_data(smartApi, "3045", datetime(yr,1,1), datetime(yr+1,1,31))
    if not df.empty:
        print(f"  From {yr}: {len(df):5d} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    else:
        print(f"  From {yr}: EMPTY")
    time.sleep(2)

# Test 2: ONE_MINUTE depth - probe backwards
print("\n=== ONE_MINUTE: How far back? ===")
for yr in [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019]:
    df = get_historical_data(smartApi, "3045", datetime(yr,1,1), datetime(yr+1,6,1) if yr < 2026 else datetime.now(), interval="ONE_MINUTE")
    if not df.empty:
        print(f"  From {yr}: {len(df):6d} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    else:
        print(f"  From {yr}: EMPTY")
    time.sleep(2)

# Test 3: FIFTEEN_MINUTE depth - probe backwards  
print("\n=== FIFTEEN_MINUTE: How far back? ===")
for yr in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
    df = get_historical_data(smartApi, "3045", datetime(yr,1,1), datetime(yr+1,6,1) if yr < 2026 else datetime.now(), interval="FIFTEEN_MINUTE")
    if not df.empty:
        print(f"  From {yr}: {len(df):5d} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    else:
        print(f"  From {yr}: EMPTY")
    time.sleep(2)
