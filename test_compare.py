from SmartApi import SmartConnect
import pyotp
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

totp = pyotp.TOTP(TOTP_SECRET).now()
print("Generated TOTP:", totp)

smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, totp)
if data["status"]:
    print("Login Successful")
else:
    print("Login Failed")
    exit()
time.sleep(2)

print("Downloading Scrip Master...")
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
scrip_master = resp.json()
token_map = {}
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]
print(f"Found {len(token_map)} NSE EQ tokens")
time.sleep(2)

symbol = "RELIANCE"
token = token_map.get("RELIANCE")
print(f"{symbol} token: {token}")

def fetch(smartApi, token, interval, fromdate, todate, retries=3):
    for attempt in range(retries):
        try:
            params = {
                "exchange": "NSE",
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
            print(f"  Attempt {attempt+1} failed: {str(e)[:60]}")
            time.sleep(3)
    return pd.DataFrame()

# Fetch 1-min data - smaller date range to avoid hitting limits
end = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
start = end - timedelta(days=3)

print(f"Fetching 1-min data: {start.date()} to {end.date()}...")
df1 = fetch(smartApi, token, "ONE_MINUTE", start, end)
print(f"  1-min rows: {len(df1)}")
time.sleep(2)

print(f"Fetching 15-min data: {start.date()} to {end.date()}...")
df15 = fetch(smartApi, token, "FIFTEEN_MINUTE", start, end)
print(f"  15-min rows: {len(df15)}")

if df1.empty or df15.empty:
    print("ERROR: No data received")
    exit()

# Also try fetching a wider range for 15-min data
print(f"\nFetching MORE 15-min data: {(end - timedelta(days=30)).date()} to {end.date()}...")
df15_wide = fetch(smartApi, token, "FIFTEEN_MINUTE", end - timedelta(days=30), end)
print(f"  15-min rows (30 days): {len(df15_wide)}")

# Resample 1-min to 15-min
df1_resampled = df1.set_index("datetime").resample("15min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()

# Compare
df15_compare = df15.copy()
df15_compare["rounded"] = df15_compare["datetime"].dt.floor("15min")
df1_resampled["rounded"] = df1_resampled["datetime"].dt.floor("15min")

merged = pd.merge(
    df1_resampled[["rounded","open","high","low","close","volume"]].rename(
        columns={"open":"o1","high":"h1","low":"l1","close":"c1","volume":"v1"}),
    df15_compare[["rounded","open","high","low","close","volume"]].rename(
        columns={"open":"o15","high":"h15","low":"l15","close":"c15","volume":"v15"}),
    on="rounded", how="inner"
)

merged["cdiff"] = abs(merged["c1"] - merged["c15"])
merged["odiff"] = abs(merged["o1"] - merged["o15"])
merged["hdiff"] = abs(merged["h1"] - merged["h15"])
merged["ldiff"] = abs(merged["l1"] - merged["l15"])
merged["vdiff"] = abs(merged["v1"] - merged["v15"])

print("\n" + "=" * 75)
print("COMPARISON: 1-min Resampled -> 15-min vs Native API 15-min")
print("=" * 75)
print(f"1-min data range:     {df1['datetime'].min()} to {df1['datetime'].max()} ({len(df1)} rows)")
print(f"15-min data range:    {df15['datetime'].min()} to {df15['datetime'].max()} ({len(df15)} rows)")
print(f"Matched 15-min bars:  {len(merged)}")
print(f"\n--- DIFFERENCE STATS ---")
print(f"  Open  -> exact match: {(merged['odiff']==0).sum()}/{len(merged)}  | mean diff: {merged['odiff'].mean():.4f}")
print(f"  High  -> exact match: {(merged['hdiff']==0).sum()}/{len(merged)}  | mean diff: {merged['hdiff'].mean():.4f}")
print(f"  Low   -> exact match: {(merged['ldiff']==0).sum()}/{len(merged)}  | mean diff: {merged['ldiff'].mean():.4f}")
print(f"  Close -> exact match: {(merged['cdiff']==0).sum()}/{len(merged)}  | mean diff: {merged['cdiff'].mean():.4f}")
print(f"  Vol   -> exact match: {(merged['vdiff']==0).sum()}/{len(merged)}")

print(f"\n--- FIRST 8 CANDLES (1mResampled vs 15mAPI) ---")
print(f"{'Time':<8} {'Open_1mR':>8} {'Open_15A':>8} {'High_1mR':>8} {'High_15A':>8} {'Low_1mR':>8} {'Low_15A':>8} {'Close_1mR':>8} {'Close_15A':>8} {'Vol_1mR':>7} {'Vol_15A':>7}")
print("-" * 95)
for _, r in merged.head(8).iterrows():
    print(f"{str(r['rounded'].time())[:5]:<8} {r['o1']:>8.1f} {r['o15']:>8.1f} {r['h1']:>8.1f} {r['h15']:>8.1f} {r['l1']:>8.1f} {r['l15']:>8.1f} {r['c1']:>8.1f} {r['c15']:>8.1f} {int(r['v1']):>7} {int(r['v15']):>7}")

print(f"\n--- MISMATCHES (close diff > 0.05) ---")
bad = merged[merged["cdiff"] > 0.05]
if len(bad) == 0:
    print("  NONE - All close prices match within 0.05!")
else:
    for _, r in bad.iterrows():
        print(f"  {r['rounded']}  close: {r['c1']:.2f} vs {r['c15']:.2f}  diff={r['cdiff']:.2f}")

total_exact = (merged["cdiff"] == 0).sum()
print(f"\n{'='*75}")
print(f"CONCLUSION: {total_exact}/{len(merged)} 15-min bars match EXACTLY")
if total_exact == len(merged):
    print(">> 1-min resampled data = IDENTICAL to native 15-min API data <<")
    print(">> You only need to fetch 1-min data and resample to 15-min! <<")
else:
    print(">> Minor discrepancies exist (possibly trade aggregation differences) <<")

# Show what data is available in wider 15-min
if not df15_wide.empty:
    print(f"\n--- 15-min data availability (30 days) ---")
    print(f"  Total rows: {len(df15_wide)}")
    print(f"  Date range: {df15_wide['datetime'].min()} to {df15_wide['datetime'].max()}")
