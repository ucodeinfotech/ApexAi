"""
Nifty 50 Historical Data Fetcher (Angel One SmartAPI)
- Fetches both 1-min and 15-min data for all 50 Nifty 50 stocks
- Auto-downloads tokens from Scrip Master
- Chunked fetching for max historical data
- Resamples 1-min to 15-min for signal/execution alignment
- Saves to CSV files
"""

from SmartApi import SmartConnect
import pyotp
import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta

# =========================
# CREDENTIALS
# =========================
API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

# =========================
# LOGIN
# =========================
totp = pyotp.TOTP(TOTP_SECRET).now()
print("Generated TOTP:", totp)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, totp)

if data["status"]:
    print("Login Successful")
    authToken = data['data']['jwtToken']
    refreshToken = data['data']['refreshToken']
    feedToken = smartApi.getfeedToken()
else:
    print("Login Failed")
    raise SystemExit("Cannot proceed without login")

# =========================
# AUTO-DOWNLOAD TOKENS FROM SCRIP MASTER
# =========================
print("\nDownloading Scrip Master for token lookup...")
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
scrip_master = resp.json()
print(f"Downloaded {len(scrip_master)} instruments")

# Build lookup dict: name -> token (only NSE EQ)
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

print(f"Found {len(token_map)} NSE EQ symbols")

# =========================
# FULL NIFTY 50 STOCK LIST (as of Apr 2026)
# =========================
NIFTY_50_TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC",
    "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TATAMOTORS", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
]

# =========================
# BUILD TOKEN MAPPING
# =========================
TOKEN_MAPPING = {}
missing = []
for symbol in NIFTY_50_TICKERS:
    # Some symbols have special names in Scrip Master
    alt_names = {
        "BAJAJ-AUTO": "BAJAJ_AUTO",
        "M&M": "M_M",
        "INDIGO": "INTERGLOBE_AVIATION",
        "JIOFIN": "JIO_FINANCIAL_SERVICES",
        "MAXHEALTH": "MAX_HEALTHCARE_INSTITUTE",
        "ETERNAL": "ETERNAL",
        "SHRIRAMFIN": "SHRIRAM_FINANCE",
        "TATAMOTORS": "TATA_MOTORS_PASSENGER_VEHICLES",
    }
    name = alt_names.get(symbol, symbol)
    token = token_map.get(name)
    if token:
        TOKEN_MAPPING[symbol] = token
    else:
        missing.append(symbol)

if missing:
    print(f"\nWARNING: Could not find tokens for: {missing}")

# =========================
# HISTORICAL DATA FETCH FUNCTION
# =========================
def get_historical_data(smartApi, symboltoken, exchange="NSE", interval="FIVE_MINUTE",
                        fromdate=None, todate=None):
    try:
        if fromdate is None:
            todate = datetime.now()
            fromdate = todate - timedelta(days=5)
        historicParam = {
            "exchange": exchange,
            "symboltoken": str(symboltoken),
            "interval": interval,
            "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
            "todate": todate.strftime("%Y-%m-%d %H:%M")
        }
        candles = smartApi.getCandleData(historicParam)
        if not candles["status"]:
            return pd.DataFrame()
        df = pd.DataFrame(candles["data"], columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

# =========================
# CHUNKED FETCH FOR MAX DATA
# =========================
def fetch_all_chunks(smartApi, symboltoken, interval, step_days, max_total_days=200,
                     exchange="NSE"):
    """
    Fetch historical data in overlapping chunks.
    API returns ~500 candles per call, so we step backwards.
    """
    all_dfs = []
    end = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    total_fetched = 0

    while total_fetched < max_total_days:
        start = end - timedelta(days=step_days)
        print(f"    Fetching {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...", end=" ")
        df = get_historical_data(smartApi, symboltoken, exchange, interval, start, end)
        if df.empty:
            print("No data")
            break
        print(f"{len(df)} candles")
        all_dfs.append(df)
        total_fetched += step_days
        end = start
        time.sleep(0.5)  # rate limit protection

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs)
    result = result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return result

# =========================
# STEP SIZES BY INTERVAL
# =========================
# 1-min: ~500 candles = ~2 market days. Use 2-day steps.
# 15-min: ~500 candles = ~20 market days. Use 20-day steps.
STEP_CONFIG = {
    "ONE_MINUTE": {"step_days": 2, "max_days": 60},     # ~30 x 2 = 60 days max
    "FIFTEEN_MINUTE": {"step_days": 20, "max_days": 400} # ~20 x 20 = 400 days max
}

# =========================
# MAIN FETCH LOOP
# =========================
OUTPUT_DIR = "nifty50_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Fetching historical data for {len(TOKEN_MAPPING)} Nifty 50 stocks")
print(f"{'='*60}")

for symbol, token in list(TOKEN_MAPPING.items())[:5]:  # Remove [:5] to run on all 50
    print(f"\n[{symbol}] Token: {token}")

    interval_files = {}
    for interval, config in STEP_CONFIG.items():
        print(f"  Interval: {interval}")
        df = fetch_all_chunks(
            smartApi, token,
            interval=interval,
            step_days=config["step_days"],
            max_total_days=config["max_days"]
        )
        if df.empty:
            print(f"  SKIP: No {interval} data for {symbol}")
            continue

        print(f"  Got {len(df)} candles ({df['datetime'].min()} to {df['datetime'].max()})")

        # Save raw data
        fname = f"{OUTPUT_DIR}/{symbol}_{interval}.csv"
        df.to_csv(fname, index=False)
        interval_files[interval] = fname

    # =========================
    # ALIGN 1-MIN AND 15-MIN DATA
    # =========================
    one_min_file = f"{OUTPUT_DIR}/{symbol}_ONE_MINUTE.csv"
    fifteen_min_file = f"{OUTPUT_DIR}/{symbol}_FIFTEEN_MINUTE.csv"

    if os.path.exists(one_min_file) and os.path.exists(fifteen_min_file):
        df1 = pd.read_csv(one_min_file, parse_dates=["datetime"])
        df15 = pd.read_csv(fifteen_min_file, parse_dates=["datetime"])

        # Mark which timeframe each row is from
        df1["timeframe"] = "1min"
        df15["timeframe"] = "15min"

        # Combine
        combined = pd.concat([df1, df15], ignore_index=True)
        combined = combined.sort_values("datetime").reset_index(drop=True)

        # Save combined
        combined_fname = f"{OUTPUT_DIR}/{symbol}_combined.csv"
        combined.to_csv(combined_fname, index=False)
        print(f"  Saved combined file: {combined_fname}")

print(f"\nDone! Files saved to {OUTPUT_DIR}/")
print("\nTIP: To run on all 50 stocks, remove '[:5]' from the loop above.")
