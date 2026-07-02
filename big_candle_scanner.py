"""
Big Candle + Consolidation Scanner
- Scans 493 stocks on 1D timeframe for the pattern
"""
import pandas as pd, numpy as np, os, glob, json, time
from datetime import datetime
from collections import defaultdict

DATA = "C:/Users/pc/Downloads/stock hist data/comprehensive_data"
OUTPUT = "C:/Users/pc/Downloads/stock hist data/backtest_results"
os.makedirs(OUTPUT, exist_ok=True)

# ─── PARAMS (tunable) ───
PARAMS = {
    "big_body_mult": 2.0,      # trigger body > 2x avg body
    "wick_pct": 0.20,          # upper wick < 20% of total range
    "vol_mult": 1.5,           # volume > 1.5x avg vol
    "avg_period": 20,          # lookback for avgs
    "consol_min": 3,           # min consolidation candles
    "consol_body_pct": 0.30,   # small body < 30% of range
    "consol_max_range_pct": 0.05,  # consol range within 5% of trigger close
}

def detect_big_candle(row, avg_body, avg_vol):
    """Check if a candle is a 'big candle' trigger"""
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["close"], row["open"])
    lower_wick = min(row["close"], row["open"]) - row["low"]
    total_range = row["high"] - row["low"]
    
    if total_range == 0 or avg_body == 0 or avg_vol == 0:
        return False
    
    wick_ratio = upper_wick / total_range
    is_big = body > avg_body * PARAMS["big_body_mult"]
    is_small_wick = wick_ratio < PARAMS["wick_pct"]
    is_high_vol = row["volume"] > avg_vol * PARAMS["vol_mult"]
    
    return is_big and is_small_wick and is_high_vol

def is_small_body(row):
    """Check if candle has small body relative to its range"""
    body = abs(row["close"] - row["open"])
    total_range = row["high"] - row["low"]
    if total_range == 0:
        return True
    return body / total_range < PARAMS["consol_body_pct"]

def scan_stock(df):
    """Scan one stock for patterns. Returns list of pattern dicts."""
    patterns = []
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    if n < PARAMS["avg_period"] + PARAMS["consol_min"] + 5:
        return patterns
    
    # Precompute rolling averages
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(PARAMS["avg_period"], min_periods=PARAMS["avg_period"]).mean().shift(1)
    df["avg_vol"] = df["volume"].rolling(PARAMS["avg_period"], min_periods=PARAMS["avg_period"]).mean().shift(1)
    
    for i in range(PARAMS["avg_period"], n):
        row = df.iloc[i]
        avg_body = row["avg_body"]
        avg_vol = row["avg_vol"]
        
        if pd.isna(avg_body) or pd.isna(avg_vol):
            continue
        
        if not detect_big_candle(row, avg_body, avg_vol):
            continue
        
        # Big candle found at index i
        trigger_date = row["datetime"]
        trigger_close = row["close"]
        trigger_type = "BULLISH" if row["close"] > row["open"] else "BEARISH"
        trigger_body = row["body"]
        trigger_range = row["range"]
        trigger_upper_wick = row["high"] - max(row["close"], row["open"])
        trigger_vol = row["volume"]
        
        # Look for consolidation after trigger
        consol_count = 0
        consol_high = row["high"]
        consol_low = row["low"]
        consol_data = []
        
        for j in range(i + 1, n):
            c = df.iloc[j]
            consol_high = max(consol_high, c["high"])
            consol_low = min(consol_low, c["low"])
            
            # Check if consolidation is within range
            consol_range = (consol_high - consol_low) / trigger_close
            if consol_range > PARAMS["consol_max_range_pct"]:
                # Broke out of consolidation range - stop looking
                if consol_count >= PARAMS["consol_min"]:
                    break
                else:
                    consol_count = 0
                    consol_high = c["high"]
                    consol_low = c["low"]
                    consol_data = []
                    continue
            
            if is_small_body(c):
                consol_count += 1
                consol_data.append({
                    "date": str(c["datetime"].date()),
                    "open": c["open"], "high": c["high"],
                    "low": c["low"], "close": c["close"],
                    "body_pct": abs(c["close"] - c["open"]) / c["range"] * 100 if c["range"] > 0 else 0
                })
            else:
                # Non-small body resets count but keeps range
                if consol_count >= PARAMS["consol_min"]:
                    break
                consol_count = 0
                if consol_data:
                    consol_data = []
                consol_high = c["high"]
                consol_low = c["low"]
        
        if consol_count >= PARAMS["consol_min"]:
            # Valid pattern found
            pattern_end = consol_data[-1]["date"]
            last_close = consol_data[-1]["close"]
            
            # Determine current status
            if last_close > trigger_close * 1.02:
                status = "BROKEN UP (bullish)"
            elif last_close < trigger_close * 0.98:
                status = "BROKEN DOWN (bearish)"
            else:
                status = "CONSOLIDATING"
            
            patterns.append({
                "trigger_date": str(trigger_date.date()),
                "trigger_type": trigger_type,
                "trigger_close": trigger_close,
                "trigger_body": trigger_body,
                "trigger_range": trigger_range,
                "trigger_upper_wick_pct": trigger_upper_wick / trigger_range * 100 if trigger_range > 0 else 0,
                "trigger_vol": trigger_vol,
                "trigger_vol_ratio": trigger_vol / avg_vol if avg_vol > 0 else 0,
                "consol_count": consol_count,
                "consol_high": consol_high,
                "consol_low": consol_low,
                "consol_range_pct": (consol_high - consol_low) / trigger_close * 100,
                "pattern_end_date": pattern_end,
                "last_close": last_close,
                "status": status,
                "days_since_pattern": (df.iloc[-1]["datetime"] - pd.Timestamp(pattern_end).tz_localize(df.iloc[-1]["datetime"].tz)).days,
                "consol_candles": consol_data,
            })
    
    return patterns


# ─── SCAN ALL STOCKS ───
files = sorted(glob.glob(f"{DATA}/*_ONE_DAY.csv"))
total = len(files)
print(f"Scanning {total} stocks...")

all_patterns = []
stock_stats = []
start = time.time()

for idx, f in enumerate(files):
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    patterns = scan_stock(df)
    n_found = len(patterns)
    n_total = len(df)
    
    stock_stats.append({
        "symbol": sym,
        "rows": n_total,
        "patterns": n_found,
        "first_date": str(df["datetime"].min().date()),
        "last_date": str(df["datetime"].max().date()),
    })
    
    for p in patterns:
        p["symbol"] = sym
        all_patterns.append(p)
    
    if (idx + 1) % 10 == 0 or idx == total - 1 or idx == 0:
        elapsed = time.time() - start
        print(f"  [{idx+1}/{total}] {sym} - {n_found} patterns found, {n_total} rows ({elapsed:.0f}s)", flush=True)

elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"SCAN COMPLETE: {total} stocks in {elapsed:.0f}s")
print(f"Total patterns found: {len(all_patterns)}")
print(f"Stocks with at least 1 pattern: {sum(1 for s in stock_stats if s['patterns'] > 0)}")

# ─── SAVE RESULTS ───
pattern_df = pd.DataFrame(all_patterns)
if len(pattern_df) > 0:
    pattern_df.to_csv(os.path.join(OUTPUT, "big_candle_patterns.csv"), index=False)
    print(f"Saved patterns to big_candle_patterns.csv")

stats_df = pd.DataFrame(stock_stats)
stats_df.to_csv(os.path.join(OUTPUT, "big_candle_stats.csv"), index=False)
print(f"Saved stats to big_candle_stats.csv")

# ─── PRINT SUMMARY ───
print(f"\n{'='*60}")
print("PATTERN SUMMARY")
print(f"{'='*60}")

if len(pattern_df) > 0:
    print(f"\nBy Trigger Type:")
    print(pattern_df["trigger_type"].value_counts().to_string())
    
    print(f"\nBy Status:")
    print(pattern_df["status"].value_counts().to_string())
    
    print(f"\nBy Consolidation Count:")
    print(pattern_df["consol_count"].value_counts().sort_index().to_string())
    
    print(f"\nTop 20 Stocks by Pattern Frequency:")
    top = pattern_df["symbol"].value_counts().head(20)
    for sym, cnt in top.items():
        print(f"  {sym:20s}: {cnt}")
    
    print(f"\nRecent patterns (last 30 days):")
    recent = pattern_df[pattern_df["days_since_pattern"] <= 30].sort_values("days_since_pattern")
    for _, r in recent.head(20).iterrows():
        print(f"  {r['symbol']:20s} | {r['trigger_date']} | {r['trigger_type']:8s} | "
              f"Close:{r['trigger_close']:>8.0f} | Consol:{r['consol_count']}d | "
              f"Status: {r['status']}")

print(f"\nDone.")
