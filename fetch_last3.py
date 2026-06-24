from SmartApi import SmartConnect
import pyotp
import pandas as pd
import time
import sys
from datetime import datetime, timedelta, timezone

API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"
OUTPUT_DIR = "nifty50_full_history"

def fetch_chunk(smartApi, token, interval, from_date, to_date, retries=5):
    for attempt in range(retries):
        try:
            params = {
                "exchange": "NSE", "symboltoken": str(token), "interval": interval,
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
            print(f"    retry {attempt+1}: {type(e).__name__}")
            time.sleep(10)
    return pd.DataFrame()

def fetch_full_history(smartApi, token, interval, step_days, start_year=2016):
    all_chunks = []
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    end = now
    earliest = datetime(start_year, 1, 1, tzinfo=ist)
    empty_count = 0
    while end > earliest:
        start = end - timedelta(days=step_days)
        if start < earliest:
            start = earliest
        df = fetch_chunk(smartApi, token, interval, start, end)
        if df.empty:
            empty_count += 1
            if empty_count >= 3:
                break
            end = start
            time.sleep(2)
            continue
        empty_count = 0
        all_chunks.append(df)
        end = df["datetime"].min() - timedelta(minutes=1)
        time.sleep(2)
    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks)
    return result.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

# Login
print("Logging in...")
smartApi = SmartConnect(api_key=API_KEY)
data = smartApi.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not data["status"]:
    print("Login Failed!")
    sys.exit(1)
print("OK")
time.sleep(2)

# 1) ULTRACEMCO - 1-min
print("\n[1/3] ULTRACEMCO - 1-min only...")
df = fetch_full_history(smartApi, "11532", "ONE_MINUTE", 90, 2020)
if not df.empty:
    df.to_csv(f"{OUTPUT_DIR}/ULTRACEMCO_ONE_MINUTE.csv", index=False)
    print(f"  {len(df):,} rows | {df['datetime'].min().date()} to {df['datetime'].max().date()}")
else:
    print("  FAILED")
time.sleep(3)

# 2) TATAMOTORS
print("\n[2/3] TATAMOTORS (token 3456)...")
print("  15-min...")
df15 = fetch_full_history(smartApi, "3456", "FIFTEEN_MINUTE", 150, 2016)
if not df15.empty:
    df15.to_csv(f"{OUTPUT_DIR}/TATAMOTORS_FIFTEEN_MINUTE.csv", index=False)
    print(f"  {len(df15):,} rows | {df15['datetime'].min().date()} to {df15['datetime'].max().date()}")
else:
    print("  FAILED")
time.sleep(3)
print("  1-min...")
df1 = fetch_full_history(smartApi, "3456", "ONE_MINUTE", 90, 2020)
if not df1.empty:
    df1.to_csv(f"{OUTPUT_DIR}/TATAMOTORS_ONE_MINUTE.csv", index=False)
    print(f"  {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}")
else:
    print("  FAILED")
time.sleep(3)

# 3) WIPRO (token 3787)
print("\n[3/3] WIPRO (token 3787)...")
print("  15-min...")
df15 = fetch_full_history(smartApi, "3787", "FIFTEEN_MINUTE", 150, 2016)
if not df15.empty:
    df15.to_csv(f"{OUTPUT_DIR}/WIPRO_FIFTEEN_MINUTE.csv", index=False)
    print(f"  {len(df15):,} rows | {df15['datetime'].min().date()} to {df15['datetime'].max().date()}")
else:
    print("  FAILED")
time.sleep(3)
print("  1-min...")
df1 = fetch_full_history(smartApi, "3787", "ONE_MINUTE", 90, 2020)
if not df1.empty:
    df1.to_csv(f"{OUTPUT_DIR}/WIPRO_ONE_MINUTE.csv", index=False)
    print(f"  {len(df1):,} rows | {df1['datetime'].min().date()} to {df1['datetime'].max().date()}")
else:
    print("  FAILED")

print("\nDONE!")
