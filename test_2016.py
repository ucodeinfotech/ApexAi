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

def get_historical_data(
    smartApi,
    symboltoken,
    from_date,
    to_date=None,
    exchange="NSE",
    interval="FIVE_MINUTE"
):
    try:
        if to_date is None:
            to_date = datetime.now()
        historicParam = {
            "exchange": exchange,
            "symboltoken": str(symboltoken),
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
        candles = smartApi.getCandleData(historicParam)
        if not candles.get("status", False):
            print("Historical Data Error:", candles)
            return pd.DataFrame()
        df = pd.DataFrame(
            candles["data"],
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception as e:
        print("Error:", e)
        return pd.DataFrame()

print("\n=== TEST 1: SBIN (3045) FIVE_MINUTE 2016-2017 ===")
df = get_historical_data(
    smartApi=smartApi,
    symboltoken="3045",
    from_date=datetime(2016, 1, 1),
    to_date=datetime(2017, 1, 31),
    interval="FIVE_MINUTE"
)
if not df.empty:
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(df.head(3))
    print(df.tail(3))
else:
    print("EMPTY - No data returned")

print("\n=== TEST 2: SBIN (3045) FIVE_MINUTE 2018-2019 ===")
df2 = get_historical_data(
    smartApi=smartApi,
    symboltoken="3045",
    from_date=datetime(2018, 1, 1),
    to_date=datetime(2019, 6, 30),
    interval="FIVE_MINUTE"
)
if not df2.empty:
    print(f"Rows: {len(df2)}")
    print(f"Date range: {df2['datetime'].min()} to {df2['datetime'].max()}")
else:
    print("EMPTY")

print("\n=== TEST 3: SBIN (3045) FIVE_MINUTE 2020-2021 ===")
df3 = get_historical_data(
    smartApi=smartApi,
    symboltoken="3045",
    from_date=datetime(2020, 1, 1),
    to_date=datetime(2021, 6, 30),
    interval="FIVE_MINUTE"
)
if not df3.empty:
    print(f"Rows: {len(df3)}")
    print(f"Date range: {df3['datetime'].min()} to {df3['datetime'].max()}")
else:
    print("EMPTY")

print("\n=== TEST 4: RELIANCE (2885) FIVE_MINUTE 2016 ===")
df4 = get_historical_data(
    smartApi=smartApi,
    symboltoken="2885",
    from_date=datetime(2016, 1, 1),
    to_date=datetime(2017, 1, 31),
    interval="FIVE_MINUTE"
)
if not df4.empty:
    print(f"Rows: {len(df4)}")
    print(f"Date range: {df4['datetime'].min()} to {df4['datetime'].max()}")
else:
    print("EMPTY")
