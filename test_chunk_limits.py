"""Test ONE_MINUTE chunk size limits"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
print("Logged in\n")

ist = timezone(timedelta(hours=5, minutes=30))
token = "474"  # 3MINDIA

chunks = [30, 60, 90, 120, 150, 180, 365, 730]

for days in chunks:
    end = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    start = end - timedelta(days=days)
    print(f"Chunk: {days:3d} days  |  {start.date()} to {end.date()}", end="")
    try:
        c = smartApi.getCandleData({"exchange":"NSE","symboltoken":token,"interval":"ONE_MINUTE",
            "fromdate":start.strftime("%Y-%m-%d %H:%M"),
            "todate":end.strftime("%Y-%m-%d %H:%M")})
        if c["status"] and c["data"]:
            d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
            print(f"  ->  {len(d):5d} rows  ({d['datetime'].iloc[0][:10]} to {d['datetime'].iloc[-1][:10]})")
        else:
            print(f"  ->  EMPTY (status={c['status']})")
    except Exception as e:
        print(f"  ->  ERROR: {str(e)[:80]}")
    time.sleep(1.5)
