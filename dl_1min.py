"""Phase 1: download 1-min only with per-stock incremental resume"""
from SmartApi import SmartConnect
import pyotp, pandas as pd, time, os, sys, json
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"; CLIENT_ID = "D52359454"; PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "comprehensive_data"
PART_DIR = os.path.join(OUTPUT_DIR, "_part")

with open("nse_tokens.json") as f:
    TOKEN_MAP = json.load(f)
with open("nifty500_batch.json") as f:
    ALL_STOCKS = json.load(f)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]: print("Login Failed!"); sys.exit(1)
print(f"Logged in. {len(ALL_STOCKS)} stocks to process")
os.makedirs(PART_DIR, exist_ok=True)

PROGRESS_FILE = "dl_1min_progress.json"
done_set = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        done_set = set(json.load(f))
    print(f"Already done: {len(done_set)}")

def fc(token, fr, to):
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

pending = [s for s in ALL_STOCKS if s not in done_set]
print(f"Pending: {len(pending)}")

for idx, sym in enumerate(pending):
    csv_path = f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv"
    if os.path.exists(csv_path):
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

    print(f"[{idx+1}/{len(pending)}] Downloading {sym} (token={token})...", flush=True)
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15,minute=30,second=0,microsecond=0)
    early = datetime(2016,1,1,tzinfo=ist)

    # Check for partial progress
    part_path = os.path.join(PART_DIR, f"{sym}.csv")
    if os.path.exists(part_path):
        old = pd.read_csv(part_path)
        old["datetime"] = pd.to_datetime(old["datetime"])
        end = old["datetime"].min() - timedelta(minutes=1)
        all_c = [old]
        print(f"  Resuming from {old['datetime'].min().date()}, {len(old)} rows already saved")
    else:
        end = now
        all_c = []

    ec = 0
    last_save = 0
    while end > early:
        st = end - timedelta(days=60)
        if st < early: st = early
        df = fc(token, st, end)
        if df.empty:
            ec += 1
            if ec >= 5: break
            end = st; time.sleep(0.3); continue
        ec = 0; all_c.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        if len(all_c) % 10 == 0:
            print(f"  chunk {len(all_c)}: {len(df)} rows, up to {df['datetime'].max().date()}", flush=True)
        # Save incrementally every 10 chunks to allow resume
        if len(all_c) - last_save >= 10:
            pd.concat(all_c).drop_duplicates(subset=["datetime"]).sort_values("datetime").to_csv(part_path, index=False)
            last_save = len(all_c)
        time.sleep(0.3)

    if all_c:
        full = pd.concat(all_c).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        print(f"  Total: {len(full):,} rows | {full['datetime'].min().date()} to {full['datetime'].max().date()}", flush=True)
        full.to_csv(csv_path, index=False)
        print(f"  {sym} saved!", flush=True)
    else:
        print(f"  {sym}: NO DATA", flush=True)

    if os.path.exists(part_path):
        os.remove(part_path)
    done_set.add(sym)
    with open(PROGRESS_FILE, "w") as f: json.dump(list(done_set), f)

print(f"\nDone! Completed {len(done_set)} stocks")
