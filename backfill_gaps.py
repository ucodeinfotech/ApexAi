"""Resume-able backfill for stocks with data gaps"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, requests, time, os, sys, json
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]: print("Login Failed!"); sys.exit(1)
print("Logged in\n")

# Load tokens
with open("nse_tokens.json") as f:
    TOKENS = json.load(f)

def fc(token, fr, to):
    for _ in range(3):
        try:
            c = smartApi.getCandleData({"exchange":"NSE","symboltoken":str(token),"interval":"ONE_MINUTE",
                "fromdate":fr.strftime("%Y-%m-%d %H:%M"),"todate":to.strftime("%Y-%m-%d %H:%M")})
            if c["status"] and c["data"]:
                d = pd.DataFrame(c["data"],columns=["datetime","open","high","low","close","volume"])
                d["datetime"] = pd.to_datetime(d["datetime"]); return d
            return pd.DataFrame()
        except Exception as e:
            time.sleep(5)
    return pd.DataFrame()

def resamp(df1):
    if df1.empty: return {}
    df = df1.set_index("datetime")
    return {n: df.resample(r).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
            for n,r in [("FIVE_MINUTE","5min"),("FIFTEEN_MINUTE","15min"),("ONE_HOUR","1h"),("ONE_DAY","1D")]}

# Progress tracking
PROGRESS_FILE = "backfill_progress.json"
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    print(f"Resuming from progress file. Already processed: {len(progress.get('done',[]))} stocks")
else:
    progress = {"done": [], "skipped": []}

ist = timezone(timedelta(hours=5, minutes=30))
total_filled = 0

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]
all_stocks = set()
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_MINUTE.csv"):
                all_stocks.add(f.replace("_ONE_MINUTE.csv", ""))
all_stocks = sorted(all_stocks)

for sym in all_stocks:
    if sym in progress.get("done", []) or sym in progress.get("skipped", []):
        continue
    
    # Find CSV
    fpath = None
    for d in ALL_DIRS:
        p = f"{d}/{sym}_ONE_MINUTE.csv"
        if os.path.exists(p):
            fpath = p; break
    if not fpath: continue
    
    # Quick gap detection from file
    try:
        df = pd.read_csv(fpath, usecols=["datetime"])
    except:
        print(f"{sym}: read error, skipping", flush=True)
        progress.setdefault("skipped", []).append(sym)
        with open(PROGRESS_FILE, "w") as f: json.dump(progress, f)
        continue
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    days = sorted(set(d.date() for d in df["datetime"]))
    
    gaps = []
    for i in range(1, len(days)):
        gap = (days[i] - days[i-1]).days
        if gap > 10:
            gaps.append((days[i-1], days[i]))
    
    if not gaps:
        progress.setdefault("done", []).append(sym)
        continue
    
    token = TOKENS.get(sym)
    if not token:
        print(f"{sym}: NO TOKEN, skipping", flush=True)
        progress.setdefault("skipped", []).append(sym)
        with open(PROGRESS_FILE, "w") as f: json.dump(progress, f)
        continue
    
    print(f"\n{sym} (token={token}): {len(gaps)} gap(s)", flush=True)
    any_filled = False
    
    for gap_start, gap_end in gaps:
        print(f"  Gap: {gap_start} to {gap_end} ({(gap_end-gap_start).days}d)", flush=True)
        
        fetch_from = gap_start - timedelta(days=5)
        fetch_to = gap_end + timedelta(days=5)
        
        # Fetch in 60-day chunks
        chunks = []
        end = datetime.combine(fetch_to, datetime.min.time()).replace(tzinfo=ist) + timedelta(hours=15, minutes=30)
        st = datetime.combine(fetch_from, datetime.min.time()).replace(tzinfo=ist)
        while end > st:
            cs = end - timedelta(days=60)
            if cs < st: cs = st
            df_chunk = fc(token, cs, end)
            if not df_chunk.empty:
                chunks.append(df_chunk)
                print(f"    chunk got {len(df_chunk)} rows", flush=True)
            end = cs - timedelta(minutes=1)
            time.sleep(1.5)
        
        if not chunks:
            print(f"    No data", flush=True)
            continue
        
        new_data = pd.concat(chunks).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        print(f"    Got {len(new_data)} rows: {new_data['datetime'].min().date()} to {new_data['datetime'].max().date()}", flush=True)
        
        # Merge
        full = pd.read_csv(fpath)
        full["datetime"] = pd.to_datetime(full["datetime"])
        
        merged = pd.concat([full, new_data])
        merged = merged.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        
        added = len(merged) - len(full)
        print(f"    Added {added} rows ({len(full):,} -> {len(merged):,})", flush=True)
        
        if added > 0:
            any_filled = True
            total_filled += 1
            merged.to_csv(fpath, index=False)
            base = fpath.replace("_ONE_MINUTE.csv", "")
            for tn, td in resamp(merged).items():
                td.to_csv(f"{base}_{tn}.csv", index=False)
        
        time.sleep(2)
    
    if any_filled:
        print(f"  {sym}: DONE + backfilled!", flush=True)
    else:
        print(f"  {sym}: no gaps filled", flush=True)
    
    if any_filled:
        progress.setdefault("done", []).append(sym)
    else:
        progress.setdefault("skipped", []).append(sym)
    
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

print(f"\n{'='*50}")
print(f"Backfill complete. Gaps filled: {total_filled}")
print(f"Done: {len(progress.get('done',[]))}, Skipped: {len(progress.get('skipped',[]))}")
print("Run 'python quick_verify.py' to check results.")
