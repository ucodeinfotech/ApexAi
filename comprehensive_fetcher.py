"""
Comprehensive Historical Data Fetcher
- Downloads 1-min data from earliest available to present
- Resamples to 5-min, 15-min, 1-hour, 1-day
- Covers: Nifty 50, Nifty Next 50, Nifty Midcap 100, Sensex, Bank Nifty
"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
import time
import os
import sys
from datetime import datetime, timedelta, timezone

# =========================
# CONFIG
# =========================
API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "comprehensive_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SKIP STOCKS ALREADY DOWNLOADED
# =========================
ALREADY_DOWNLOADED = set()
for existing_dir in ["nifty50_full_history", "comprehensive_data"]:
    if os.path.exists(existing_dir):
        for f in os.listdir(existing_dir):
            if f.endswith("_ONE_MINUTE.csv"):
                sym = f.replace("_ONE_MINUTE.csv", "")
                ALREADY_DOWNLOADED.add(sym)
print(f"Found {len(ALREADY_DOWNLOADED)} stocks with existing 1-min data")

# =========================
# STOCK LISTS BY INDEX
# =========================

NIFTY_50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BEL","BHARTIARTL",
    "CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL",
    "GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
    "HINDUNILVR","ICICIBANK","INDIGO","INFY","ITC",
    "JIOFIN","JSWSTEEL","KOTAKBANK","LT","M&M",
    "MARUTI","MAXHEALTH","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBILIFE","SBIN","SHRIRAMFIN",
    "SUNPHARMA","TATACONSUM","TATASTEEL","TATAMOTORS","TCS",
    "TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO"
]

NIFTY_NEXT_50 = [
    "ABB","ADANIENSOL","ADANIGREEN","ADANIPOWER","AMBUJACEM",
    "BAJAJHLDNG","BANKBARODA","BOSCHLTD","BPCL","BRITANNIA",
    "CANBK","CGPOWER","CHOLAFIN","CUMMINSIND","DIVISLAB",
    "DLF","DMART","GAIL","GODREJCP","HDFCAMC",
    "HAL","HINDZINC","HYUNDAI","INDHOTEL","IOC",
    "IRFC","JINDALSTEL","LODHA","LTIM","MAZDOCK",
    "MOTHERSUMI","MUTHOOTFIN","PFC","PIDILITIND","PNB",
    "RECLTD","SHREECEM","SIEMENS","SOLARINDS","TATACAP",
    "TATAPOWER","TORNTPHARM","TVSMOTOR","UNIONBANK","UNITDSPR",
    "VBL","VEDL","ZYDUSLIFE","ENRIN","MOTHERSON"
]

NIFTY_MIDCAP_100 = [
    "ABB","ABCAPITAL","ADANIENSOL","ADANIGREEN","ADANIPOWER",
    "ALKEM","APLAPOLLO","ASHOKLEY","ASTRAL","AUBANK",
    "AUROPHARMA","BAJAJHLDNG","BANKBARODA","BANKINDIA","BHARATFORG",
    "BHEL","BIOCON","BRITANNIA","BSE","CANBK",
    "CGPOWER","CHOLAFIN","CUMMINSIND","DABUR","DIXON",
    "DLF","DMART","FEDERALBNK","FORTIS","FSL",
    "GAIL","GMRINFRA","GODREJCP","GODREJPROP","HAL",
    "HDFCAMC","HEROMOTOCO","HINDZINC","HINDPETRO","HUDCO",
    "HYUNDAI","ICICIPRULI","IDBI","IDFCFIRSTB",
    "INDHOTEL","INDUSINDBK","IOC","IRFC","JBCHEPHARM",
    "JINDALSTEL","JSWENERGY","JUBLFOOD","KALYANKJIL",
    "LICI","LODHA","LTIM","LUPIN","MANKIND",
    "MARICO","MAZDOCK","MOTHERSON","MUTHOOTFIN",
    "NATIONALUM","NHPC","NMDC","OIL","PAGEIND",
    "PERSISTENT","PFC","PIDILITIND","PIIND","PNB",
    "POLYCAB","POWERGRID","RECLTD","SAIL","SBICARD",
    "SHREECEM","SIEMENS","SOLARINDS","SRF","STARHEALTH",
    "SUNTV","SUPREMEIND","SYNGENE","TATACAP","TATACOMM",
    "TATAPOWER","TORNTPHARM","TORNTPOWER","TRIDENT","TVSMOTOR",
    "UBL","UNIONBANK","UNITDSPR","VBL","VEDL",
    "VOLTAS","WESTLIFE","YESBANK","ZYDUSLIFE"
]

SENSEX = [
    "ASIANPAINT","AXISBANK","BAJAJFINSV","BAJFINANCE","BHARTIARTL",
    "HCLTECH","HDFCBANK","HINDUNILVR","ICICIBANK","INDUSINDBK",
    "INFY","ITC","JSWSTEEL","KOTAKBANK","LT",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBIN","SUNPHARMA","TATACONSUM",
    "TATAMOTORS","TATASTEEL","TCS","TITAN","WIPRO",
    "BAJAJ-AUTO","BEL","DRREDDY","EICHERMOT","GRASIM",
    "HINDALCO","SBILIFE","TECHM","TRENT","ULTRACEMCO"
]

BANK_NIFTY = [
    "AXISBANK","BANKBARODA","CANBK","FEDERALBNK","HDFCBANK",
    "ICICIBANK","INDUSINDBK","KOTAKBANK","PNB","SBIN",
    "YESBANK","BANDHANBNK","IDFCFIRSTB"
]

def build_master_list():
    """Combine all stocks into a single deduplicated list, skipping already downloaded."""
    all_stocks = []
    seen = set()
    for sym in NIFTY_NEXT_50 + NIFTY_MIDCAP_100 + SENSEX + BANK_NIFTY:
        if sym not in seen and sym not in ALREADY_DOWNLOADED:
            seen.add(sym)
            all_stocks.append(sym)
    print(f"  Skipping {len(ALREADY_DOWNLOADED & seen)} already-downloaded stocks")
    return all_stocks

# ALT_NAMES for Scrip Master lookup
ALT_NAMES = {
    "BAJAJ-AUTO": "BAJAJ_AUTO",
    "M&M": "M_M",
    "INDIGO": "INTERGLOBE_AVIATION",
    "JIOFIN": "JIO_FINANCIAL_SERVICES",
    "MAXHEALTH": "MAX_HEALTHCARE_INSTITUTE",
    "SHRIRAMFIN": "SHRIRAM_FINANCE",
    "ETERNAL": "ETERNAL",
    "INDUSINDBK": "INDUSIND_BANK",
    "BHARATFORG": "BHARAT_FORGE",
    "HEROMOTOCO": "HERO_MOTOCORP",
    "FEDERALBNK": "FEDERAL_BANK",
    "SBICARD": "SBI_CARDS_AND_PAYMENT_SERVICES",
    "NATIONALUM": "NATIONAL_ALUMINIUM",
    "IDFCFIRSTB": "IDFC_FIRST_BANK",
    "ICICIPRULI": "ICICI_PRUDENTIAL",
    "JBCHEPHARM": "JB_CHEMICALS",
    "KALYANKJIL": "KALYAN_JEWELLERS",
    "WESTLIFE": "WESTLIFE_DEVELOPMENT",
    "GMRINFRA": "GMRP&UI",
    "MOTHERSUMI": "MOTHERSON",
    "HINDPETRO": "HINDPETRO",
    "FSL": "FSN_E_COMMERCE",
    "WAAREE": "WAAREEINDO",
}

MANUAL_TOKENS = {
    "TATAMOTORS": "3456",
}

# =========================
# LOGIN
# =========================
print("=" * 70)
print("COMPREHENSIVE HISTORICAL DATA FETCHER")
print("=" * 70)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print(f"Logged in as {CLIENT_ID}")
time.sleep(2)

# =========================
# GET TOKEN MAPPING
# =========================
print("\nDownloading Scrip Master...")
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
print(f"Downloaded {len(scrip_master)} instruments")

# Build lookup dict
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

print(f"Found {len(token_map)} NSE EQ symbols")

# Also build name -> symbol mapping for reverse lookup
name_to_symbol = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        name_to_symbol[item["name"]] = item["symbol"].replace("-EQ", "")

# =========================
# BUILD FINAL STOCK LIST
# =========================
all_stocks = build_master_list()
print(f"\nTotal unique stocks across all indices: {len(all_stocks)}")

TOKEN_MAPPING = {}
not_found = []

for symbol in all_stocks:
    # Check manual tokens first
    if symbol in MANUAL_TOKENS:
        TOKEN_MAPPING[symbol] = MANUAL_TOKENS[symbol]
        continue

    name = ALT_NAMES.get(symbol, symbol)
    token = token_map.get(name)
    
    if token:
        TOKEN_MAPPING[symbol] = token
    else:
        # Try removing -EQ from symbol for lookup
        token = token_map.get(symbol)
        if token:
            TOKEN_MAPPING[symbol] = token
        else:
            not_found.append(symbol)

if not_found:
    print(f"\nCould not find tokens for {len(not_found)} stocks:")
    print(f"  {not_found}")
    # Try probing Scrip Master for these
    print(f"\nSearching Scrip Master for alternate names...")
    for sym in not_found[:10]:
        for item in scrip_master:
            if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
                if sym.upper() in item["name"].upper() or item["name"].upper() in sym.upper():
                    print(f"  Potential match: {sym} -> {item['name']} (symbol: {item['symbol']})")
                    break

print(f"\nMapped {len(TOKEN_MAPPING)}/{len(all_stocks)} stocks")
time.sleep(2)

# =========================
# FETCH FUNCTIONS
# =========================
def fetch_chunk(smartApi, token, interval, from_date, to_date, retries=3):
    for attempt in range(retries):
        try:
            params = {
                "exchange": "NSE",
                "symboltoken": str(token),
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            candles = smartApi.getCandleData(params)
            if candles["status"] and candles["data"]:
                df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            return pd.DataFrame()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            continue
    return pd.DataFrame()

def fetch_1min_full_history(smartApi, token, start_year=2016):
    """Fetch all available 1-min data going backwards from now."""
    all_chunks = []
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    earliest = datetime(start_year, 1, 1, tzinfo=ist)
    step_days = 60  # 1-min: ~30 market days per chunk, use 60 calendar days as step
    chunk_no = 0
    empty_count = 0

    while end > earliest:
        chunk_no += 1
        start = end - timedelta(days=step_days)
        if start < earliest:
            start = earliest

        print(f"      Chunk {chunk_no}: {start.date()} to {end.date()}...", end=" ")
        sys.stdout.flush()

        df = fetch_chunk(smartApi, token, "ONE_MINUTE", start, end)
        if df.empty:
            empty_count += 1
            print("EMPTY")
            if empty_count >= 5:
                break
            end = start
            time.sleep(1.5)
            continue
        empty_count = 0
        print(f"{len(df)} rows ({df['datetime'].min().date()} to {df['datetime'].max().date()})")
        all_chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        time.sleep(1.5)

    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks)
    result = result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return result

def resample_timeframes(df1):
    """Resample 1-min data to 5-min, 15-min, 1-hour, 1-day."""
    result = {}
    if df1.empty:
        return result
    df = df1.set_index("datetime")
    rules = {
        "FIVE_MINUTE": "5min",
        "FIFTEEN_MINUTE": "15min",
        "ONE_HOUR": "1h",
        "ONE_DAY": "1D",
    }
    for name, rule in rules.items():
        resampled = df.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
        result[name] = resampled
    return result

def load_existing(output_dir, symbol, timeframe):
    """Load existing data for resume capability."""
    fname = f"{output_dir}/{symbol}_{timeframe}.csv"
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname, parse_dates=["datetime"])
            print(f"      Loaded existing {timeframe}: {len(df)} rows ({df['datetime'].min().date()} to {df['datetime'].max().date()})")
            return df
        except:
            pass
    return None

# =========================
# MAIN FETCH LOOP
# =========================
print("\n" + "=" * 60)
print(f"FETCHING 1-MIN DATA + RESAMPLING FOR {len(TOKEN_MAPPING)} STOCKS")
print("=" * 60)

start_time = time.time()
total = len(TOKEN_MAPPING)
summary = []

for idx, (symbol, token) in enumerate(TOKEN_MAPPING.items(), 1):
    print(f"\n[{idx}/{total}] {symbol} (token: {token})")
    
    # Check if already downloaded
    existing_1min = load_existing(OUTPUT_DIR, symbol, "ONE_MINUTE")
    existing_final = f"{OUTPUT_DIR}/{symbol}_ONE_MINUTE.csv"
    
    if existing_1min is not None:
        # Check if we have recent data (within 2 days)
        latest = existing_1min["datetime"].max()
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)
        days_since = (now - latest).days
        if days_since <= 3:
            print(f"  Already have data up to {latest.date()}. Skipping.")
            # Still resample if needed
            print(f"  Resampling from existing 1-min data...")
            timeframes = resample_timeframes(existing_1min)
            for tf_name, tf_df in timeframes.items():
                tf_file = f"{OUTPUT_DIR}/{symbol}_{tf_name}.csv"
                tf_df.to_csv(tf_file, index=False)
                print(f"    {tf_name}: {len(tf_df)} rows ({tf_df['datetime'].min().date()} to {tf_df['datetime'].max().date()})")
            summary.append({"symbol": symbol, "status": "SKIP", "1min": len(existing_1min)})
            continue
    
    # Fetch 1-min data from 2016
    print(f"  Fetching 1-min data (from 2016)...")
    df1 = fetch_1min_full_history(smartApi, token, 2016)
    
    if df1.empty:
        print(f"  NO DATA for {symbol}")
        summary.append({"symbol": symbol, "status": "NO_DATA", "1min": 0})
        continue
    
    print(f"  -> {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}")
    
    # Save 1-min data
    df1.to_csv(f"{OUTPUT_DIR}/{symbol}_ONE_MINUTE.csv", index=False)
    
    # Resample
    print(f"  Resampling...")
    timeframes = resample_timeframes(df1)
    for tf_name, tf_df in timeframes.items():
        tf_file = f"{OUTPUT_DIR}/{symbol}_{tf_name}.csv"
        if not tf_df.empty:
            tf_df.to_csv(tf_file, index=False)
            print(f"    {tf_name}: {len(tf_df)} rows ({tf_df['datetime'].min().date()} to {tf_df['datetime'].max().date()})")
        else:
            print(f"    {tf_name}: EMPTY")
    
    summary.append({"symbol": symbol, "status": "OK", "1min": len(df1)})
    
    # ETA
    elapsed = time.time() - start_time
    per_stock = elapsed / idx
    remaining = per_stock * (total - idx)
    print(f"  [ETA: {remaining/60:.1f} min]")

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"Time: {(time.time()-start_time)/60:.1f} min")
print(f"Processed: {len(summary)}/{total}")
ok = sum(1 for s in summary if s["status"] == "OK")
skip = sum(1 for s in summary if s["status"] == "SKIP")
nodata = sum(1 for s in summary if s["status"] == "NO_DATA")
print(f"  Downloaded: {ok}")
print(f"  Already existed: {skip}")
print(f"  No data: {nodata}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{OUTPUT_DIR}/_FETCH_SUMMARY.csv", index=False)

print(f"\nAll files in {OUTPUT_DIR}/")
print("\nTo update data later, re-run: it will skip stocks with recent data")
