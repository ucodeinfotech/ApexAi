"""Download remaining stocks only"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
import time
import os
import sys
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "comprehensive_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All stocks in combined master list
ALL_STOCKS = [
    # Nifty Next 50
    "ABB","ADANIENSOL","ADANIGREEN","ADANIPOWER","AMBUJACEM",
    "BAJAJHLDNG","BANKBARODA","BOSCHLTD","BPCL","BRITANNIA",
    "CANBK","CGPOWER","CHOLAFIN","CUMMINSIND","DIVISLAB",
    "DLF","DMART","GAIL","GODREJCP","HDFCAMC",
    "HAL","HINDZINC","HYUNDAI","INDHOTEL","IOC",
    "IRFC","JINDALSTEL","LODHA","MAZDOCK","MOTHERSON",
    "MUTHOOTFIN","PFC","PIDILITIND","PNB","RECLTD",
    "SHREECEM","SIEMENS","SOLARINDS","TATACAP","TATAPOWER",
    "TORNTPHARM","TVSMOTOR","UNIONBANK","UNITDSPR","VBL",
    "VEDL","ZYDUSLIFE","ENRIN","MOTHERSUMI",
    # Midcap 100
    "ABCAPITAL","ALKEM","APLAPOLLO","ASHOKLEY","ASTRAL",
    "AUBANK","AUROPHARMA","BANKINDIA","BHARATFORG","BHEL",
    "BIOCON","BSE","DABUR","DIXON","FEDERALBNK",
    "FORTIS","FSL","GMRINFRA","GODREJPROP","HDFCAMC",
    "HEROMOTOCO","HINDPETRO","HUDCO","ICICIPRULI","IDBI",
    "IDFCFIRSTB","INDUSINDBK","JBCHEPHARM","JSWENERGY","JUBLFOOD",
    "KALYANKJIL","LICI","LUPIN","MANKIND","MARICO",
    "MUTHOOTFIN","NATIONALUM","NHPC","NMDC","OIL",
    "PAGEIND","PERSISTENT","PFC","PIIND","POLYCAB",
    "RECLTD","SAIL","SBICARD","SRF","STARHEALTH",
    "SUNTV","SUPREMEIND","SYNGENE","TATACOMM","TORNTPOWER",
    "TRIDENT","UBL","VOLTAS","WESTLIFE","YESBANK",
    "BANDHANBNK","NBCC","PIIND","POLYCAB",
    # Sensex extra (non-Nifty50)
    "INDUSINDBK","HINDALCO",
    # Bank Nifty extra
    "BANDHANBNK",
]

# Get already downloaded
downloaded = set()
for d in ["nifty50_full_history", "comprehensive_data"]:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_MINUTE.csv"):
                downloaded.add(f.replace("_ONE_MINUTE.csv", ""))

still_needed = []
seen = set()
for sym in ALL_STOCKS:
    if sym not in seen and sym not in downloaded:
        seen.add(sym)
        still_needed.append(sym)

print(f"Already downloaded: {len(downloaded)} stocks")
print(f"Still needed: {len(still_needed)} stocks")
if still_needed:
    print(f"Remaining: {still_needed}")

if not still_needed:
    print("All stocks already downloaded!")
    sys.exit(0)

# Login
print("\nLogging in...")
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print("OK")
time.sleep(2)

# Get tokens
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

ALT_NAMES = {
    "BAJAJ-AUTO": "BAJAJ_AUTO",
    "M&M": "M_M",
    "INDIGO": "INTERGLOBE_AVIATION",
    "JIOFIN": "JIO_FINANCIAL_SERVICES",
    "MAXHEALTH": "MAX_HEALTHCARE_INSTITUTE",
    "SHRIRAMFIN": "SHRIRAM_FINANCE",
    "ETERNAL": "ETERNAL",
    "INDUSINDBK": "INDUSIND_BANK",
    "IDFCFIRSTB": "IDFC_FIRST_BANK",
    "FEDERALBNK": "FEDERAL_BANK",
    "HEROMOTOCO": "HERO_MOTOCORP",
    "BHARATFORG": "BHARAT_FORGE",
    "JBCHEPHARM": "JB_CHEMICALS",
    "KALYANKJIL": "KALYAN_JEWELLERS",
    "WESTLIFE": "WESTLIFE_DEVELOPMENT",
    "GMRINFRA": "GMRP&UI",
    "MOTHERSUMI": "MOTHERSON",
    "HINDPETRO": "HINDPETRO",
    "SBICARD": "SBI_CARDS_AND_PAYMENT_SERVICES",
    "NATIONALUM": "NATIONAL_ALUMINIUM",
    "ICICIPRULI": "ICICI_PRUDENTIAL",
    "FSL": "FSN_E_COMMERCE",
}

# Build token mapping for remaining stocks
TOKEN_MAP = {}
not_found = []
for sym in still_needed:
    name = ALT_NAMES.get(sym, sym)
    token = token_map.get(name)
    if token:
        TOKEN_MAP[sym] = token
    else:
        not_found.append(sym)

if not_found:
    print(f"\nNo tokens found for: {not_found}")

print(f"Stocks to download: {len(TOKEN_MAP)}")
if not TOKEN_MAP:
    print("Nothing to download!")
    sys.exit(0)

# Fetch function
def fetch_chunk(smartApi, token, from_date, to_date):
    params = {
        "exchange": "NSE", "symboltoken": str(token), "interval": "ONE_MINUTE",
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M")
    }
    for attempt in range(3):
        try:
            candles = smartApi.getCandleData(params)
            if candles["status"] and candles["data"]:
                df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            return pd.DataFrame()
        except:
            time.sleep(5)
    return pd.DataFrame()

def fetch_full(smartApi, token, start_year=2016):
    all_chunks = []
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    earliest = datetime(start_year, 1, 1, tzinfo=ist)
    step_days = 60
    empty_count = 0

    while end > earliest:
        start = end - timedelta(days=step_days)
        if start < earliest:
            start = earliest
        df = fetch_chunk(smartApi, token, start, end)
        if df.empty:
            empty_count += 1
            if empty_count >= 5:
                break
            end = start
            time.sleep(1.5)
            continue
        empty_count = 0
        all_chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        time.sleep(1.5)

    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks)
    return result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

def resample_all(df1):
    result = {}
    if df1.empty:
        return result
    df = df1.set_index("datetime")
    for name, rule in [("FIVE_MINUTE","5min"),("FIFTEEN_MINUTE","15min"),("ONE_HOUR","1h"),("ONE_DAY","1D")]:
        r = df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
        result[name] = r
    return result

# Main loop
print(f"\n{'='*60}")
print(f"DOWNLOADING {len(TOKEN_MAP)} REMAINING STOCKS")
print(f"{'='*60}")

start = time.time()
for idx, (sym, token) in enumerate(TOKEN_MAP.items(), 1):
    print(f"\n[{idx}/{len(TOKEN_MAP)}] {sym} (token: {token})")
    
    df1 = fetch_full(smartApi, token, 2016)
    if df1.empty:
        print(f"  -> NO DATA")
        continue
    
    print(f"  -> {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}")
    
    # Save 1-min
    df1.to_csv(f"{OUTPUT_DIR}/{sym}_ONE_MINUTE.csv", index=False)
    
    # Resample and save
    tfs = resample_all(df1)
    for tf_name, tf_df in tfs.items():
        if not tf_df.empty:
            tf_df.to_csv(f"{OUTPUT_DIR}/{sym}_{tf_name}.csv", index=False)
            print(f"  {tf_name}: {len(tf_df)} rows")
    
    elapsed = (time.time() - start) / 60
    rate = idx / elapsed if elapsed > 0 else 0
    remaining = (len(TOKEN_MAP) - idx) / rate if rate > 0 else 0
    print(f"  [Elapsed: {elapsed:.1f}min | ETA: {remaining:.1f}min]")

print(f"\n{'='*60}")
print(f"DONE! Time: {(time.time()-start)/60:.1f} min")
print(f"{'='*60}")
