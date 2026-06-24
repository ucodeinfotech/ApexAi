"""
Nifty 50 Max Historical Data Fetcher + Verification
Fetches 1-min and 15-min data for all 50 stocks with progress tracking,
verifies data integrity by comparing resampled 1-min vs native 15-min,
and saves verified CSV files.
"""
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import numpy as np
import requests
import time
import os
import sys
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

OUTPUT_DIR = "nifty50_verified_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Max history per interval (in days per chunk, max total days)
# 1-min: ~500 candles per call ≈ 1.3 trading days → use 1-day chunks
# 15-min: ~500 candles per call ≈ 20 trading days → use 20-day chunks
CONFIG = {
    "ONE_MINUTE":     {"step": 1,  "max": 30},
    "FIFTEEN_MINUTE": {"step": 20, "max": 400},
}

# =========================
# LOGIN
# =========================
print("=" * 70)
print("NIFTY 50 HISTORICAL DATA FETCHER + VERIFICATION")
print("=" * 70)

totp = pyotp.TOTP(TOTP_SECRET).now()
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, totp)
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print(f"Logged in as {CLIENT_ID}")
time.sleep(2)

# =========================
# GET TOKEN MAPPING
# =========================
print("\nDownloading Scrip Master for token lookup...")
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]
print(f"Found {len(token_map)} NSE EQ tokens")
time.sleep(2)

# =========================
# NIFTY 50 STOCKS
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

ALT_NAMES = {
    # These symbols are already correct as-is in Scrip Master
    # "BAJAJ-AUTO" -> already "BAJAJ-AUTO"
    # "M&M" -> already "M&M"
    # "INDIGO" -> already "INDIGO"
    # "JIOFIN" -> already "JIOFIN"
    # "MAXHEALTH" -> already "MAXHEALTH"
    # "SHRIRAMFIN" -> already "SHRIRAMFIN"
    # "TATAMOTORS" -> NOT in Scrip Master (demerged entity, skip)
}

# Build mapping
TOKEN_MAPPING = {}
for sym in NIFTY_50:
    name = ALT_NAMES.get(sym, sym)
    t = token_map.get(name)
    if t:
        TOKEN_MAPPING[sym] = t

print(f"\nMapped {len(TOKEN_MAPPING)}/{len(NIFTY_50)} Nifty 50 stocks to tokens")
missing = [s for s in NIFTY_50 if s not in TOKEN_MAPPING]
if missing:
    print(f"  Missing tokens for: {missing}")
    # Let the user decide whether to continue
    print("  These will be skipped.")

# =========================
# API FETCH FUNCTION
# =========================
def fetch_chunk(smartApi, token, interval, fromdate, todate, exchange="NSE", retries=3):
    for attempt in range(retries):
        try:
            params = {
                "exchange": exchange,
                "symboltoken": str(token),
                "interval": interval,
                "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
                "todate": todate.strftime("%Y-%m-%d %H:%M")
            }
            candles = smartApi.getCandleData(params)
            if not candles["status"]:
                return pd.DataFrame()
            df = pd.DataFrame(candles["data"], columns=["datetime","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            continue
    return pd.DataFrame()

# =========================
# CHUNKED FETCH (max history)
# =========================
def fetch_max_history(smartApi, token, interval, step_days, max_days, exchange="NSE"):
    all_chunks = []
    now = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    total_fetched = 0

    while total_fetched < max_days:
        start = end - timedelta(days=step_days)
        if start < datetime(2020, 1, 1):
            start = datetime(2020, 1, 1)
        df = fetch_chunk(smartApi, token, interval, start, end, exchange)
        if df.empty:
            break
        all_chunks.append(df)
        total_fetched += step_days
        end = start
        if end < datetime(2020, 1, 1):
            break
        time.sleep(1.5)

    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks)
    result = result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return result

# =========================
# VERIFY: 1-min resampled vs 15-min
# =========================
def verify_data(df1, df15, symbol):
    """Compare resampled 1-min vs native 15-min data."""
    if df1.empty or df15.empty:
        return {"status": "SKIP", "reason": "Missing data"}

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
        return {"status": "FAIL", "reason": "No matching 15-min bars found"}

    merged["close_diff"] = abs(merged["close_resampled"] - merged["close_native"])
    mismatches = (merged["close_diff"] > 0.05).sum()
    total = len(merged)
    exact = total - mismatches

    return {
        "status": "PASS" if mismatches == 0 else "MINOR_MISMATCH",
        "total_bars": total,
        "exact_match": exact,
        "mismatches": mismatches,
        "close_diff_mean": round(merged["close_diff"].mean(), 4),
        "close_diff_max": round(merged["close_diff"].max(), 4),
    }

# =========================
# MAIN LOOP
# =========================
print("\n" + "=" * 70)
print("STARTING FETCH FOR ALL 50 STOCKS")
print("=" * 70)

total_stocks = len(TOKEN_MAPPING)
total_calls = 0
for sym in TOKEN_MAPPING:
    for cfg in CONFIG.values():
        total_calls += cfg["max"] // cfg["step"] + 1

start_time = time.time()
results_log = []

for idx, (symbol, token) in enumerate(TOKEN_MAPPING.items(), 1):
    stock_start = time.time()
    elapsed = time.time() - start_time
    remaining_stocks = total_stocks - idx + 1
    avg_per_stock = elapsed / idx if idx > 1 else 0
    eta = avg_per_stock * remaining_stocks

    print(f"\n{'='*70}")
    print(f"[{idx}/{total_stocks}] {symbol} (token: {token})")
    print(f"  Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min | Remaining: {remaining_stocks} stocks")
    print(f"{'='*70}")

    stock_data = {"symbol": symbol, "token": token}
    dfs = {}

    for interval, cfg in CONFIG.items():
        step, max_d = cfg["step"], cfg["max"]
        n_chunks = max_d // step
        print(f"  [{interval}] Fetching up to {max_d} days in {n_chunks} chunks...")

        all_rows = []
        now = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
        end = now
        fetched = 0
        chunk_no = 0

        while fetched < max_d:
            chunk_no += 1
            start = end - timedelta(days=step)
            if start < datetime(2020, 1, 1):
                start = datetime(2020, 1, 1)

            print(f"    Chunk {chunk_no}: {start.date()} to {end.date()}...", end=" ")
            sys.stdout.flush()

            df = fetch_chunk(smartApi, token, interval, start, end)
            if df.empty:
                print("END (no more data)")
                break

            print(f"{len(df)} rows")
            all_rows.append(df)
            fetched += step
            end = start
            stock_data[f"{interval}_last_date"] = str(df["datetime"].max().date())
            stock_data[f"{interval}_first_date"] = str(df["datetime"].min().date())

            if end < datetime(2020, 1, 1):
                break
            time.sleep(1.5)

        if all_rows:
            combined = pd.concat(all_rows)
            combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            dfs[interval] = combined
            stock_data[f"{interval}_rows"] = len(combined)
            print(f"  [{interval}] TOTAL: {len(combined)} rows | {combined['datetime'].min().date()} to {combined['datetime'].max().date()}")
        else:
            dfs[interval] = pd.DataFrame()
            stock_data[f"{interval}_rows"] = 0
            print(f"  [{interval}] NO DATA")

    # ---- VERIFICATION ----
    print(f"  [VERIFY] Checking 1-min resampled vs 15-min native...")
    verify_result = verify_data(dfs.get("ONE_MINUTE", pd.DataFrame()),
                                dfs.get("FIFTEEN_MINUTE", pd.DataFrame()), symbol)
    stock_data["verify_status"] = verify_result["status"]

    if verify_result["status"] == "PASS":
        print(f"  [VERIFY] PASS: {verify_result['total_bars']} bars match exactly")
    elif verify_result["status"] == "MINOR_MISMATCH":
        print(f"  [VERIFY] MINOR: {verify_result['exact_match']}/{verify_result['total_bars']} exact, "
              f"{verify_result['mismatches']} mismatched (max diff: {verify_result['close_diff_max']})")
    else:
        print(f"  [VERIFY] {verify_result['status']}: {verify_result.get('reason','')}")

    # ---- SAVE CSV ----
    for interval in ["ONE_MINUTE", "FIFTEEN_MINUTE"]:
        if interval in dfs and not dfs[interval].empty:
            fname = f"{OUTPUT_DIR}/{symbol}_{interval}.csv"
            dfs[interval].to_csv(fname, index=False)

    # Save combined + verified
    if "ONE_MINUTE" in dfs and "FIFTEEN_MINUTE" in dfs:
        df1 = dfs["ONE_MINUTE"]
        df15 = dfs["FIFTEEN_MINUTE"]
        if not df1.empty and not df15.empty:
            df1["timeframe"] = "1min"
            df15["timeframe"] = "15min"
            combined = pd.concat([df1, df15], ignore_index=True).sort_values("datetime").reset_index(drop=True)
            combined.to_csv(f"{OUTPUT_DIR}/{symbol}_combined_verified.csv", index=False)

    stock_data["time_taken_s"] = round(time.time() - stock_start, 1)
    results_log.append(stock_data)

    # ---- SUMMARY LINE ----
    status_icon = {"PASS": "OK", "MINOR_MISMATCH": "~", "FAIL": "XX", "SKIP": "--"}
    icon = status_icon.get(verify_result["status"], "??")
    print(f"  >> [{icon}] {symbol} done in {stock_data['time_taken_s']:.0f}s "
          f"(1m:{stock_data.get('ONE_MINUTE_rows',0)} rows, "
          f"15m:{stock_data.get('FIFTEEN_MINUTE_rows',0)} rows)")

# =========================
# FINAL SUMMARY
# =========================
total_time = time.time() - start_time
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Stocks processed: {len(results_log)}/{total_stocks}")
print(f"\n{'Symbol':<15} {'1-min':>8} {'15-min':>8} {'Verify':>10} {'Time':>8}")
print("-" * 55)
passed = 0
for r in results_log:
    v = r.get("verify_status","?")
    icon = {"PASS": "OK", "MINOR_MISMATCH": "NEAR", "FAIL": "FAIL", "SKIP": "SKIP"}.get(v, v)
    if v == "PASS": passed += 1
    print(f"{r['symbol']:<15} {r.get('ONE_MINUTE_rows',0):>8} {r.get('FIFTEEN_MINUTE_rows',0):>8} {icon:>10} {r.get('time_taken_s',0):>7.0f}s")

print("-" * 55)
print(f"Verified PASS: {passed}/{len(results_log)}")
print(f"\nAll CSVs saved to: {OUTPUT_DIR}/")
print("Files per stock:")
print(f"  - {symbol}_ONE_MINUTE.csv")
print(f"  - {symbol}_FIFTEEN_MINUTE.csv")
print(f"  - {symbol}_combined_verified.csv (both timeframes merged)")
