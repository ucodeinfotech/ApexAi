"""Update newly copied Nifty 50 stocks from 2026-06-16 to present"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time, os, json
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "comprehensive_data"

with open("nse_tokens.json") as f: TOKEN_MAP = json.load(f)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]: print("Login Failed!"); exit(1)
print("Logged in.")

SYMS = ["ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJAJFINSV","BAJFINANCE","BEL","BHARTIARTL","CIPLA","COALINDIA","DRREDDY",
    "EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
    "HINDUNILVR","ICICIBANK","INDIGO","INFY","ITC","JIOFIN","JSWSTEEL","KOTAKBANK",
    "LT","M&M","MARUTI","MAXHEALTH","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE",
    "SBILIFE","SBIN","SHRIRAMFIN","SUNPHARMA","TATACONSUM","TATASTEEL","TCS","TECHM",
    "TITAN","TRENT","ULTRACEMCO","WIPRO"]

def fetch_chunk(token, fr, to):
    for _ in range(2):
        try:
            c = smartApi.getCandleData({"exchange":"NSE","symboltoken":str(token),"interval":"ONE_MINUTE",
                "fromdate":fr.strftime("%Y-%m-%d %H:%M"),"todate":to.strftime("%Y-%m-%d %H:%M")})
            if c["status"] and c["data"]:
                d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
                d["datetime"] = pd.to_datetime(d["datetime"]); return d
            return pd.DataFrame()
        except: time.sleep(2)
    return pd.DataFrame()

now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))

for sym in SYMS:
    token = TOKEN_MAP.get(sym)
    if not token:
        print(f"{sym}: NO TOKEN, skip"); continue
    
    csv_path = f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv"
    if not os.path.exists(csv_path):
        print(f"{sym}: CSV not found, skip"); continue

    old = pd.read_csv(csv_path)
    old["datetime"] = pd.to_datetime(old["datetime"])
    last_dt = old["datetime"].max()
    
    start = last_dt + timedelta(minutes=1)
    if start > now_ist:
        print(f"{sym}: already current ({last_dt.date()})"); continue

    print(f"{sym}: updating from {last_dt.date()}...", end=" ", flush=True)
    chunks = []
    while start < now_ist:
        end = start + timedelta(days=30)
        if end > now_ist: end = now_ist
        df = fetch_chunk(token, start, end)
        if df.empty:
            start = end + timedelta(minutes=1); time.sleep(0.3); continue
        chunks.append(df)
        start = df["datetime"].max() + timedelta(minutes=1)
        time.sleep(0.3)
    
    if not chunks: print("no new data"); continue
    
    new_data = pd.concat(chunks).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    combined = pd.concat([old, new_data]).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"{len(new_data)} rows added -> {len(combined):,} total")

print("\nDone!")
