"""Test ONE_MINUTE row limit at different historical periods"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
print("Logged in\n")

ist = timezone(timedelta(hours=5, minutes=30))
token = "474"

periods = [
    ("Recent 30d",  "2026-05-21 09:15", "2026-06-20 15:30"),
    ("Mid 30d",     "2023-01-01 09:15", "2023-02-01 15:30"),
    ("Old 30d",     "2018-01-01 09:15", "2018-02-01 15:30"),
    ("Old 60d",     "2018-01-01 09:15", "2018-03-01 15:30"),
    ("Old 120d",    "2018-01-01 09:15", "2018-05-01 15:30"),
    ("Old 365d",    "2018-01-01 09:15", "2019-01-01 15:30"),
]

for label, fr, to in periods:
    print(f"{label}: {fr} to {to}", end="")
    try:
        c = smartApi.getCandleData({"exchange":"NSE","symboltoken":token,"interval":"ONE_MINUTE",
            "fromdate":fr,"todate":to})
        if c["status"] and c["data"]:
            d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
            print(f"  ->  {len(d):5d} rows  ({d['datetime'].iloc[0][:10]} to {d['datetime'].iloc[-1][:10]})")
        else:
            print(f"  ->  EMPTY")
    except Exception as e:
        print(f"  ->  ERROR: {str(e)[:80]}")
    time.sleep(1.5)
