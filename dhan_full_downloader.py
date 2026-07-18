"""
====================================================================
  DHAN API — FULL STOCK DATA DOWNLOADER
  All-in-one: login → security map → multi-TF data → save
  Just copy-paste and run (fill credentials below)
====================================================================

REQUIREMENTS:
  pip install dhanhq pandas numpy

Note: The `security_id_list.csv` file must exist. The script searches
for it at a few common locations automatically, or you can set the path.

OUTPUT:
  - data/{symbol}/1min.csv, 5min.csv, 15min.csv, 30min.csv, daily.csv
  - data/_summary.csv  — metadata for all downloaded stocks
====================================================================
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  FILL YOUR CREDENTIALS HERE                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

YOUR_DHAN_CLIENT_ID = "1102461741"          # ← your Dhan client ID
YOUR_DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwic3ViIjoiQVBJIiwiYXVkIjoiQVBJIiwiY2xpZW50X2lkIjoiMTEwMjQ2MTc0MSIsIm5iZiI6MTc0NDA4OTU4MCwiZXhwIjoxNzU0MTc5NTgwLCJpYXQiOjE3NDQwODk1ODAsImp0aSI6IjUxOTBiMTI0LWM5ZGYtNDFjOS04MmY0LWQwOTI0OGNkMjRiNSJ9.HTTi5IXKbWxLb2VBLJb2NFLgNuzvSq8-8YHFhAJvMGsLqGF5f6CjgCM3BPE1se_3opqC8Xf5g9TjWgOVMdAMvA"  # ← paste your token

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STOCK LIST (edit: add/remove symbols below)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

STOCK_SYMBOLS = [
    # ── NIFTY 50 ──
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT",
    "WIPRO", "AXISBANK", "TITAN", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "HCLTECH", "NTPC", "ONGC", "POWERGRID", "M&M", "TATAMOTORS",
    "ULTRACEMCO", "BAJAJFINSV", "TATASTEEL", "ADANIPORTS", "JSWSTEEL",
    "TECHM", "GRASIM", "NESTLEIND", "DRREDDY", "BRITANNIA", "TRENT",
    "HINDALCO", "COALINDIA", "SBILIFE", "EICHERMOT", "CIPLA",
    "BAJAJ_AUTO", "APOLLOHOSP", "DIVISLAB", "ADANIENT", "BEL",
    "INDUSINDBK", "HDFCLIFE", "BPCL", "SHRIRAMFIN", "HEROMOTOCO",

    # ── Additional stocks ──
    "DIXON", "AEGISLOG", "SWANCORP", "EIDPARRY", "MANAPPURAM",
    "SIGNATURE", "PWL", "RADICO", "ITCHOTELS", "LTF",
    "PHOENIXLTD", "LODHA", "CLEAN", "UNITDSPR", "ABBOTINDIA",
    "HDFCBANK", "CUB", "WELCORP", "ZENTEC", "BANDHANBNK",
    "CAPLIPOINT", "HEG", "MUTHOOTFIN", "CGCL", "LAOPATHLAB",
    "CARTRADE", "POONAWALLA", "CEMPRO", "CANFINHOME", "CHOLAFIN",
    "GODREJPROP", "GLENMARK", "BRIGADE", "FSL", "GABRIEL",
    "ENGINERSIN", "THERMAX", "DEEPAKNTR", "GRAVITA", "GVT&D",
    "ADANIENSOL", "AEGISVOPAK", "THELEELA", "POWERINDIA", "HFCL",
    "KARURVYSYA", "ABDL", "ENRIN", "CGPOWER",
]

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

RATE_LIMIT_SECONDS = 0.7     # delay between API calls (1.4 req/sec)
DAYS_TO_FETCH = 10           # how many days of intraday to fetch
OUTPUT_DIR = "dhan_full_data"

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SCRIPT (no changes needed below)                               ║
# ╚══════════════════════════════════════════════════════════════════╝

import os, sys, time, json, csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dhanhq import dhanhq, DhanContext

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime.now(IST)
TODAY_STR = NOW.strftime("%Y-%m-%d")
FROM_STR = (NOW - timedelta(days=DAYS_TO_FETCH)).strftime("%Y-%m-%d")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOGIN ─────────────────────────────────────────────────────────
print("=" * 70)
print("  DHAN API FULL DATA DOWNLOADER")
print("=" * 70)
print(f"\n  Client ID: {YOUR_DHAN_CLIENT_ID}")
print(f"  Token:     {YOUR_DHAN_ACCESS_TOKEN[:20]}...")
print(f"  Date:      {FROM_STR} → {TODAY_STR}")
print(f"  Stocks:    {len(STOCK_SYMBOLS)}")
print(f"  Output:    {OUTPUT_DIR}/")
print()

ctx = DhanContext(client_id=YOUR_DHAN_CLIENT_ID, access_token=YOUR_DHAN_ACCESS_TOKEN)
api = dhanhq(ctx)

# ─── SECURITY MAP ──────────────────────────────────────────────────
def find_security_csv():
    candidates = [
        "security_id_list.csv",
        os.path.join("ai-pattern-screener", "security_id_list.csv"),
        os.path.join("..", "ai-pattern-screener", "security_id_list.csv"),
        os.path.join("institutional-scanner", "ai-pattern-screener", "security_id_list.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"  Found security CSV: {c}")
            return c
    # Ask user
    p = input("\n  security_id_list.csv not found. Enter full path: ").strip()
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Cannot find security_id_list.csv")

csv_path = find_security_csv()
nse_eq = pd.read_csv(csv_path, low_memory=False)
nse_eq = nse_eq[(nse_eq['SEM_EXM_EXCH_ID'] == 'NSE') & (nse_eq['SEM_INSTRUMENT_NAME'] == 'EQUITY')]
sec_map = {}
for _, r in nse_eq.iterrows():
    sec_map[str(r['SEM_TRADING_SYMBOL']).strip().upper()] = str(r['SEM_SMST_SECURITY_ID']).strip()
print(f"  Loaded {len(sec_map)} symbols from security map")

# ─── DATA FETCHING ─────────────────────────────────────────────────
def fetch_intraday(sec_id, from_date, to_date, interval=1):
    """Fetch intraday minute data with interval (1, 5, 15, 30, 60)."""
    try:
        r = api.intraday_minute_data(sec_id, "NSE_EQ", "EQUITY", from_date, to_date, interval=interval)
        if r.get("status") != "success":
            return None
        d = r.get("data", {})
        if not isinstance(d, dict) or not d.get("open"):
            return None
        candles = []
        for i in range(len(d["timestamp"])):
            ts = datetime.fromtimestamp(d["timestamp"][i], tz=IST)
            candles.append({
                "datetime": ts,
                "open": float(d["open"][i]),
                "high": float(d["high"][i]),
                "low": float(d["low"][i]),
                "close": float(d["close"][i]),
                "volume": int(d["volume"][i]),
            })
        return candles
    except Exception as e:
        return None

def fetch_daily(sec_id, from_date, to_date):
    """Fetch daily historical data."""
    try:
        r = api.historical_daily_data(sec_id, "NSE_EQ", "EQUITY", from_date, to_date)
        if r.get("status") != "success":
            return None
        d = r.get("data", {})
        if not isinstance(d, dict) or not d.get("open"):
            return None
        candles = []
        for i in range(len(d["timestamp"])):
            ts = datetime.fromtimestamp(d["timestamp"][i], tz=IST)
            candles.append({
                "datetime": ts,
                "open": float(d["open"][i]),
                "high": float(d["high"][i]),
                "low": float(d["low"][i]),
                "close": float(d["close"][i]),
                "volume": int(d["volume"][i]) if d.get("volume") else 0,
            })
        return candles
    except Exception as e:
        return None

# ─── MAIN LOOP ─────────────────────────────────────────────────────
summary = []
success = 0; fail = 0; total = len(STOCK_SYMBOLS)

for idx, sym in enumerate(STOCK_SYMBOLS, 1):
    sym_clean = sym.replace("&", "_").replace("-", "_")
    sym_dir = os.path.join(OUTPUT_DIR, sym_clean)
    os.makedirs(sym_dir, exist_ok=True)

    sec_id = sec_map.get(sym.upper())
    if not sec_id:
        # Try alternate match
        alt = sym.upper().replace("_", "-")
        sec_id = sec_map.get(alt)
    if not sec_id:
        print(f"  [{idx:3d}/{total}] {sym:>16s} → NO SECURITY ID (skipping)")
        fail += 1
        continue

    print(f"  [{idx:3d}/{total}] {sym:>16s} (ID: {sec_id:>6s})", end="", flush=True)

    # Fetch all timeframes
    tfs = {"1min": 1, "5min": 5, "15min": 15, "30min": 30}
    results = {}
    all_ok = True

    for tf_name, interval in tfs.items():
        candles = fetch_intraday(sec_id, FROM_STR, TODAY_STR, interval=interval)
        time.sleep(RATE_LIMIT_SECONDS)
        if candles and len(candles) > 0:
            df = pd.DataFrame(candles)
            df.to_csv(os.path.join(sym_dir, f"{tf_name}.csv"), index=False)
            results[tf_name] = len(df)
        else:
            results[tf_name] = 0
            all_ok = False

    # Fetch daily
    candles_d = fetch_daily(sec_id, FROM_STR, TODAY_STR)
    time.sleep(RATE_LIMIT_SECONDS)
    if candles_d and len(candles_d) > 0:
        df_d = pd.DataFrame(candles_d)
        df_d.to_csv(os.path.join(sym_dir, "daily.csv"), index=False)
        results["daily"] = len(df_d)
    else:
        results["daily"] = 0

    # Print result
    detail = " | ".join(f"{k}={v}" for k, v in results.items())
    print(f"  {detail}")
    if all_ok and results.get("daily", 0) > 0:
        success += 1
    else:
        fail += 1

    summary.append({
        "symbol": sym, "sec_id": sec_id,
        **{f"{k}_bars": v for k, v in results.items()},
    })

# ─── SUMMARY ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  DONE: {success} OK, {fail} failed / {total} total")
print(f"{'='*70}\n")

# Save summary
summary_file = os.path.join(OUTPUT_DIR, "_summary.csv")
pd.DataFrame(summary).to_csv(summary_file, index=False)
print(f"  Summary saved: {summary_file}")

# Quick stats
df_sum = pd.DataFrame(summary)
print(f"\n  Per-stock bar counts:")
for tf in ["1min", "5min", "15min", "30min", "daily"]:
    col = f"{tf}_bars"
    if col in df_sum.columns:
        vals = df_sum[df_sum[col] > 0][col]
        print(f"    {tf:>6s}: avg {vals.mean():.0f}  min {vals.min():.0f}  max {vals.max():.0f}")

print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}/")
print(f"  Each stock has its own folder with CSV files.")
print()
