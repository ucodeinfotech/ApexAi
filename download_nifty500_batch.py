"""Download missing Nifty 500 stocks in batches"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, requests, time, os, sys, json
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "comprehensive_data"; os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load token map
with open("nse_tokens.json") as f:
    TOKEN_MAP = json.load(f)

# Load download list
with open("nifty500_batch.json") as f:
    ALL_STOCKS = json.load(f)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]: print("Login Failed!"); sys.exit(1)
print(f"Logged in. {len(ALL_STOCKS)} stocks to process")

PROGRESS_FILE = "download_nifty500_progress.json"
done_set = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        done_set = set(json.load(f).get("done", []))
    print(f"Already done: {len(done_set)}")

def fc(token, fr, to):
    for _ in range(3):
        try:
            c = smartApi.getCandleData({"exchange":"NSE","symboltoken":str(token),"interval":"ONE_MINUTE",
                "fromdate":fr.strftime("%Y-%m-%d %H:%M"),"todate":to.strftime("%Y-%m-%d %H:%M")})
            if c["status"] and c["data"]:
                d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
                d["datetime"] = pd.to_datetime(d["datetime"]); return d
            return pd.DataFrame()
        except: time.sleep(5)
    return pd.DataFrame()

def fetch_all(token, sy=2016):
    all_c = []; ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15,minute=30,second=0,microsecond=0)
    end = now; early = datetime(sy,1,1,tzinfo=ist); ec = 0
    while end > early:
        st = end - timedelta(days=60)
        if st < early: st = early
        df = fc(token, st, end)
        if df.empty:
            ec += 1
            if ec >= 5: break
            end = st; time.sleep(1.5); continue
        ec = 0; all_c.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        print(f"  chunk {len(all_c)}: {len(df)} rows, up to {df['datetime'].max().date()}", flush=True)
        time.sleep(1.5)
    if not all_c: return pd.DataFrame()
    r = pd.concat(all_c)
    return r.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

def resamp(df1):
    if df1.empty: return {}
    df = df1.set_index("datetime")
    return {n: df.resample(r).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
            for n,r in [("FIVE_MINUTE","5min"),("FIFTEEN_MINUTE","15min"),("ONE_HOUR","1h"),("ONE_DAY","1D")]}

batch_size = 30
total = len(ALL_STOCKS)

for batch_start in range(0, total, batch_size):
    batch = ALL_STOCKS[batch_start:batch_start + batch_size]
    # Skip if all in batch are done
    batch_pending = [s for s in batch if s not in done_set]
    if not batch_pending:
        print(f"Batch {batch_start//batch_size + 1}: all done already")
        continue
    
    print(f"\nBatch {batch_start//batch_size + 1}/{(total+batch_size-1)//batch_size} ({len(batch_pending)} pending)")
    for idx, sym in enumerate(batch_pending):
        if os.path.exists(f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv"):
            print(f"  [{idx+1}/{len(batch_pending)}] {sym}: file exists, skipping")
            done_set.add(sym)
            continue
        
        token = TOKEN_MAP.get(sym)
        if not token:
            print(f"  [{idx+1}/{len(batch_pending)}] {sym}: NO TOKEN")
            done_set.add(sym)
            with open(PROGRESS_FILE, "w") as f:
                json.dump({"done": list(done_set)}, f)
            continue
        
        print(f"  [{idx+1}/{len(batch_pending)}] Downloading {sym} (token={token})...", flush=True)
        df1 = fetch_all(token, 2016)
        if not df1.empty:
            print(f"  Total: {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}", flush=True)
            df1.to_csv(f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv", index=False)
            for tn, td in resamp(df1).items():
                if not td.empty:
                    td.to_csv(f"{OUTPUT_DIR}/{sym}_{tn}.csv", index=False)
                    print(f"    {tn}: {len(td)} rows", flush=True)
            print(f"  {sym} done!", flush=True)
        else:
            print(f"  {sym}: NO DATA", flush=True)
        
        done_set.add(sym)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"done": list(done_set)}, f)

print(f"\nAll done! Completed {len(done_set)} stocks.")
