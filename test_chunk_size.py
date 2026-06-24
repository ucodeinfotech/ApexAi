"""Test max chunk size for ONE_DAY interval"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
print("Logged in")

ist = timezone(timedelta(hours=5, minutes=30))

tests = [
    ("Full 10yr (2016-2026)", "2016-01-01 00:00", "2026-06-20 15:30"),
    ("5yr (2016-2021)", "2016-01-01 00:00", "2021-01-01 00:00"),
    ("4yr (2016-2020)", "2016-01-01 00:00", "2020-01-01 00:00"),
    ("3yr (2016-2019)", "2016-01-01 00:00", "2019-01-01 00:00"),
]

for label, fr, to in tests:
    print(f"\n{label}: {fr} to {to}")
    try:
        c = smartApi.getCandleData({"exchange":"NSE","symboltoken":"474","interval":"ONE_DAY",
            "fromdate":fr,"todate":to})
        if c["status"] and c["data"]:
            d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
            print(f"  OK: {len(d)} rows, {d['datetime'].iloc[0][:10]} to {d['datetime'].iloc[-1][:10]}")
        else:
            print(f"  FAIL: status={c['status']}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:100]}")
    time.sleep(1)
