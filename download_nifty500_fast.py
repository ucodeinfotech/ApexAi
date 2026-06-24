"""Fast download: 1-day data only for missing Nifty 500 stocks"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time, os, sys, json
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "comprehensive_data"

with open("nse_tokens.json") as f:
    TOKEN_MAP = json.load(f)
with open("nifty500_batch.json") as f:
    ALL_STOCKS = json.load(f)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]: print("Login Failed!"); sys.exit(1)
print(f"Logged in. {len(ALL_STOCKS)} stocks to process")

# Progress file
PROGRESS_FILE = "dl_nifty500_progress.json"
done_set = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        done_set = set(json.load(f))
    print(f"Already done: {len(done_set)}")

def fc_daily(token, fr, to):
    for _ in range(3):
        try:
            c = smartApi.getCandleData({"exchange":"NSE","symboltoken":str(token),"interval":"ONE_DAY",
                "fromdate":fr.strftime("%Y-%m-%d %H:%M"),"todate":to.strftime("%Y-%m-%d %H:%M")})
            if c["status"] and c["data"]:
                d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
                d["datetime"] = pd.to_datetime(d["datetime"])
                return d
            return pd.DataFrame()
        except Exception as e:
            print(f"    retry: {e}")
            time.sleep(3)
    return pd.DataFrame()

# Process in small batches
pending = [s for s in ALL_STOCKS if s not in done_set]
print(f"Pending: {len(pending)}")

for idx, sym in enumerate(pending):
    if os.path.exists(f"{OUTPUT_DIR}/{sym}_ONE_DAY.csv"):
        print(f"[{idx+1}/{len(pending)}] {sym}: exists, skip")
        done_set.add(sym)
        with open(PROGRESS_FILE, "w") as f: json.dump(list(done_set), f)
        continue

    token = TOKEN_MAP.get(sym)
    if not token:
        print(f"[{idx+1}/{len(pending)}] {sym}: NO TOKEN")
        done_set.add(sym)
        with open(PROGRESS_FILE, "w") as f: json.dump(list(done_set), f)
        continue

    print(f"[{idx+1}/{len(pending)}] {sym} (token={token})...", end=" ", flush=True)
    
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    # ONE_DAY interval: use 4-year chunks (fits ~1500 rows, well under ~2000 limit)
    df_parts = []
    for yr_start in range(2016, 2027, 4):
        st = datetime(yr_start, 1, 1, tzinfo=ist)
        en = datetime(min(yr_start + 4, 2027), 1, 1, tzinfo=ist)
        if en > now: en = now
        part = fc_daily(token, st, en)
        if not part.empty:
            df_parts.append(part)
        time.sleep(0.5)
    
    if df_parts:
        df = pd.concat(df_parts).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        print(f"{len(df)} rows, {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        df.to_csv(f"{OUTPUT_DIR}/{sym}_ONE_DAY.csv", index=False)
    else:
        print("NO DATA")
    
    done_set.add(sym)
    with open(PROGRESS_FILE, "w") as f: json.dump(list(done_set), f)

print(f"\nDone! Completed {len(done_set)} stocks")
