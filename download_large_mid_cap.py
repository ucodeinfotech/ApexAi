"""Download 30 major large/mid cap stocks missing from comprehensive_data"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time, os, json
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
DIR = "comprehensive_data"

with open("nse_tokens.json") as f: TOKEN_MAP = json.load(f)

SYMS = ["ABSLAMC","AMBER","ANGELONE","ASTRAMICRO","AURIONPRO","AWL","BSOFT",
    "GPTINFRA","ICICIAMC","IREDA","KALPATARU","KPITTECH","LATENTVIEW","MAZDA",
    "MUTHOOTCAP","NAM-INDIA","NUVOCO","NYKAA","OLAELEC","PATANJALI","PAYTM",
    "POLICYBZR","RVNL","SAFARI","SONACOMS","SWIGGY","TCI","TCIEXP","TRIVENI","UTIAMC"]

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]: print("Login Failed!"); exit(1)
print("Logged in.")

def fetch_chunk(token, fr, to):
    for _ in range(2):
        try:
            c = smartApi.getCandleData({"exchange":"NSE","symboltoken":str(token),"interval":"ONE_MINUTE",
                "fromdate":fr.strftime("%Y-%m-%d %H:%M"),"todate":to.strftime("%Y-%m-%d %H:%M")})
            if c["status"] and c["data"]:
                d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
                d["datetime"] = pd.to_datetime(d["datetime"]); return d
            return pd.DataFrame()
        except Exception as e:
            if "AB1021" in str(e): smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
            time.sleep(2)
    return pd.DataFrame()

now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))

# Known listing dates for recently-IPOed stocks (approximate)
LISTING_DATES = {
    "PAYTM": "2021-11-18", "NYKAA": "2021-11-10", "POLICYBZR": "2021-11-15",
    "ANGELONE": "2020-10-07", "SWIGGY": "2024-11-13", "OLAELEC": "2024-08-09",
    "IREDA": "2023-11-29", "RVNL": "2009-01-01", "KPITTECH": "2020-01-01",
    "BSOFT": "2010-01-01", "AMBER": "2018-01-01", "SONACOMS": "2021-01-01",
    "ABSLAMC": "2018-01-01", "ICICIAMC": "2018-01-01", "UTIAMC": "2018-01-01",
    "NAM-INDIA": "2017-01-01", "ASTRAMICRO": "2017-01-01", "LATENTVIEW": "2020-01-01",
    "KALPATARU": "2015-01-01", "AWL": "2015-01-01", "PATANJALI": "2015-01-01",
    "NUVOCO": "2021-01-01", "GPTINFRA": "2017-01-01", "SAFARI": "2018-01-01",
    "MAZDA": "2015-01-01", "MUTHOOTCAP": "2015-01-01", "AURIONPRO": "2015-01-01",
    "TRIVENI": "2015-01-01", "TCI": "2015-01-01", "TCIEXP": "2015-01-01"
}

def find_earliest_data(token, sym):
    """Quick scan from estimated listing date to find actual first data"""
    est = LISTING_DATES.get(sym, "2016-01-01")
    cursor = max(datetime.strptime(est, "%Y-%m-%d").replace(tzinfo=timezone(timedelta(hours=5, minutes=30))), now_ist - timedelta(days=365*10))
    cursor = cursor - timedelta(days=60)  # buffer
    if cursor < now_ist - timedelta(days=365*10): cursor = now_ist - timedelta(days=365*10)
    
    # Try a 60-day window first
    df = fetch_chunk(token, cursor, cursor + timedelta(days=60))
    if not df.empty: 
        return df["datetime"].min()
    # Scan forward in 60-day jumps
    while cursor < now_ist:
        df = fetch_chunk(token, cursor, cursor + timedelta(days=60))
        if not df.empty: return df["datetime"].min()
        cursor += timedelta(days=60)
        time.sleep(0.3)
    return now_ist

for sym in SYMS:
    token = TOKEN_MAP.get(sym)
    if not token: print(f"{sym}: NO TOKEN, skip"); continue

    csv_path = f"{DIR}/{sym}_ONE_MINUTE.csv"
    
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path); old["datetime"] = pd.to_datetime(old["datetime"])
        start = old["datetime"].max() + timedelta(minutes=1)
        if start > now_ist: print(f"{sym}: already current ({old['datetime'].max().date()})"); continue
        print(f"{sym}: resuming from {old['datetime'].max().date()}...", end=" ", flush=True)
    else:
        print(f"{sym}: finding earliest data...", end=" ", flush=True)
        earliest = find_earliest_data(token, sym)
        if earliest >= now_ist:
            print(f"{sym}: no data available, skip"); continue
        old = pd.DataFrame()
        start = earliest
        print(f"starting {earliest.date()}...", end=" ", flush=True)

    chunks = []; last_save = time.time()
    while start < now_ist:
        end = min(start + timedelta(days=30), now_ist)
        df = fetch_chunk(token, start, end)
        if not df.empty:
            chunks.append(df)
            start = df["datetime"].max() + timedelta(minutes=1)
        else:
            start = end + timedelta(minutes=1)
        time.sleep(0.3)
        
        # Save intermediate every 5 min to avoid total loss on timeout
        if time.time() - last_save > 300 and chunks:
            combined = pd.concat([old] + chunks).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            combined.to_csv(csv_path, index=False)
            print(f"[{len(combined):,} rows so far]", end=" ", flush=True)
            last_save = time.time()

    if not chunks:
        if not old.empty: 
            print(f"{sym}: no new data (ends {old['datetime'].max().date()})")
        continue

    combined = pd.concat([old] + chunks).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"{len(combined):,} rows")

print("\nAll downloads complete!")
