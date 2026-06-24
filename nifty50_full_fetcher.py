"""
Nifty 50 Full Historical Fetcher - Oct 2016 to Present
Chunked correctly based on actual API limits:
  5-min: ~5000 candles/call = ~3 months per chunk
  15-min: ~3400 candles/call = ~6 months per chunk
  1-min: ~8000 candles/call = ~1 month per chunk
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
OUTPUT_DIR = "nifty50_full_history"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunk sizes per interval (safe limits)
CHUNK_CONFIG = {
    "ONE_MINUTE":     90,     # ~1 month per chunk
    "FIFTEEN_MINUTE": 150,    # ~5 months per chunk
    "ONE_HOUR":       300,    # ~10 months per chunk
    "ONE_DAY":        365,    # ~1 year per chunk
}

# =========================
# LOGIN
# =========================
print("=" * 70)
print("NIFTY 50 FULL HISTORICAL DATA FETCHER (Oct 2016 - Present)")
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
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]
print(f"Found {len(token_map)} NSE EQ tokens")
time.sleep(2)

# =========================
# NIFTY 50 STOCKS (all 50)
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

# Scrip Master uses exact names for these
# (BAJAJ-AUTO, M&M, INDIGO, JIOFIN, MAXHEALTH, SHRIRAMFIN are already correct)
ALT_NAMES = {}
# TATAMOTORS - check if it exists under another name
# If not in Scrip Master, it will be skipped

TOKEN_MAPPING = {}
for sym in NIFTY_50:
    name = ALT_NAMES.get(sym, sym)
    t = token_map.get(name)
    if t:
        TOKEN_MAPPING[sym] = t

# Manual overrides for stocks not in scrip master
TOKEN_MAPPING["TATAMOTORS"] = "3456"

print(f"Mapped {len(TOKEN_MAPPING)}/{len(NIFTY_50)} stocks")
missing = [s for s in NIFTY_50 if s not in TOKEN_MAPPING]
if missing:
    print(f"  Missing: {missing}")

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

def fetch_full_history(smartApi, token, interval, step_days, start_year=2016):
    """Fetch all available data in chunks, going backwards from now."""
    all_chunks = []
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    earliest = datetime(start_year, 1, 1, tzinfo=ist)
    chunk_no = 0
    empty_count = 0

    while end > earliest:
        chunk_no += 1
        start = end - timedelta(days=step_days)
        if start < earliest:
            start = earliest

        df = fetch_chunk(smartApi, token, interval, start, end)
        if df.empty:
            empty_count += 1
            if empty_count >= 3:  # 3 consecutive empty = end of data
                break
            end = start
            time.sleep(1.5)
            continue
        empty_count = 0
        
        all_chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        
        if chunk_no == 1:
            print(f"    -> {len(df)} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        time.sleep(1.5)

    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks)
    result = result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return result

# =========================
# VERIFY FUNCTION
# =========================
def verify_data(df1, df15, symbol):
    """Compare resampled 1-min vs native 15-min."""
    if df1.empty or df15.empty:
        return {"status": "SKIP"}
    resampled = df1.set_index("datetime").resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    df15_m = df15.copy()
    df15_m["rounded"] = df15_m["datetime"].dt.floor("15min")
    resampled["rounded"] = resampled["datetime"].dt.floor("15min")
    merged = pd.merge(
        resampled[["rounded","open","high","low","close","volume"]],
        df15_m[["rounded","open","high","low","close","volume"]],
        on="rounded", how="inner", suffixes=("_resampled","_native")
    )
    if merged.empty:
        return {"status": "FAIL", "reason": "No matching bars"}
    merged["close_diff"] = abs(merged["close_resampled"] - merged["close_native"])
    mismatches = (merged["close_diff"] > 0.05).sum()
    total = len(merged)
    return {
        "status": "PASS" if mismatches == 0 else "NEAR",
        "total_bars": total, "exact_match": total - mismatches,
        "mismatches": mismatches, "max_diff": round(merged["close_diff"].max(), 2)
    }

# =========================
# MAIN LOOP - ALL 50 STOCKS
# =========================
print("\n" + "=" * 70)
print("STARTING FULL HISTORY FETCH - ALL 50 NIFTY STOCKS")
print("=" * 70)

start_time = time.time()
total_stocks = len(TOKEN_MAPPING)
summary_rows = []

for idx, (symbol, token) in enumerate(TOKEN_MAPPING.items(), 1):
    print(f"\n[{idx}/{total_stocks}] {symbol} (token: {token})")
    stock_info = {"symbol": symbol}
    
    # Fetch 15-min data
    print(f"  15-min...")
    df15 = fetch_full_history(smartApi, token, "FIFTEEN_MINUTE", 150, 2016)
    if not df15.empty:
        stock_info["15min_rows"] = len(df15)
        stock_info["15min_from"] = str(df15["datetime"].min().date())
        stock_info["15min_to"] = str(df15["datetime"].max().date())
        df15.to_csv(f"{OUTPUT_DIR}/{symbol}_FIFTEEN_MINUTE.csv", index=False)
        print(f"    -> {len(df15):,} rows | {df15['datetime'].min().date()} to {df15['datetime'].max().date()}")
    else:
        print(f"    -> NO DATA")
    
    time.sleep(2)
    
    # Fetch 1-min data
    print(f"  1-min...")
    df1 = fetch_full_history(smartApi, token, "ONE_MINUTE", 90, 2020)
    if not df1.empty:
        stock_info["1min_rows"] = len(df1)
        stock_info["1min_from"] = str(df1["datetime"].min().date())
        stock_info["1min_to"] = str(df1["datetime"].max().date())
        df1.to_csv(f"{OUTPUT_DIR}/{symbol}_ONE_MINUTE.csv", index=False)
        print(f"    -> {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}")
    else:
        print(f"    -> NO DATA")
    
    time.sleep(2)
    
    # Verify
    v = verify_data(df1, df15, symbol)
    stock_info["verify"] = v["status"]
    print(f"  Verify: {v['status']} ({v.get('exact_match',0)}/{v.get('total_bars',0)} bars)")
    
    summary_rows.append(stock_info)
    
    # Progress estimate
    elapsed = time.time() - start_time
    per_stock = elapsed / idx
    remaining = per_stock * (total_stocks - idx)
    print(f"  [ETA: {remaining/60:.0f} min remaining]")

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nTime elapsed: {(time.time()-start_time)/60:.1f} minutes")
print(f"Stocks fetched: {len(summary_rows)}/{total_stocks}")

summary_df = pd.DataFrame(summary_rows)
summary_csv = f"{OUTPUT_DIR}/_FETCH_SUMMARY.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary saved to: {summary_csv}")
print(f"\nFiles in {OUTPUT_DIR}/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(f"{OUTPUT_DIR}/{f}")
    print(f"  {f:50s} {sz/1024/1024:.1f} MB")
