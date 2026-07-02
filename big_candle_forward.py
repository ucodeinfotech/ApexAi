"""Forward performance analysis after Big Candle + Consolidation pattern"""
import pandas as pd, numpy as np, os, glob
from datetime import datetime

DATA = "C:/Users/pc/Downloads/stock hist data/comprehensive_data"
OUTPUT = "C:/Users/pc/Downloads/stock hist data/backtest_results"

pattern_df = pd.read_csv(os.path.join(OUTPUT, "big_candle_patterns.csv"))
print(f"Loaded {len(pattern_df)} patterns")

# ─── Load raw data for each stock and measure forward returns ───
PARAMS = {
    "big_body_mult": 2.0,
    "wick_pct": 0.20,
    "vol_mult": 1.5,
    "avg_period": 20,
    "consol_min": 3,
    "consol_body_pct": 0.30,
    "consol_max_range_pct": 0.05,
}

def scan_stock_for_patterns(df):
    """Same as scanner - returns trigger indices"""
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    triggers = []
    
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(PARAMS["avg_period"], min_periods=PARAMS["avg_period"]).mean().shift(1)
    df["avg_vol"] = df["volume"].rolling(PARAMS["avg_period"], min_periods=PARAMS["avg_period"]).mean().shift(1)
    
    for i in range(PARAMS["avg_period"], n):
        row = df.iloc[i]
        avg_body, avg_vol = row["avg_body"], row["avg_vol"]
        if pd.isna(avg_body) or pd.isna(avg_vol):
            continue
        
        body = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["close"], row["open"])
        total_range = row["high"] - row["low"]
        if total_range == 0 or avg_body == 0 or avg_vol == 0:
            continue
        
        wick_ratio = upper_wick / total_range
        is_big = body > avg_body * PARAMS["big_body_mult"]
        is_small_wick = wick_ratio < PARAMS["wick_pct"]
        is_high_vol = row["volume"] > avg_vol * PARAMS["vol_mult"]
        
        if not (is_big and is_small_wick and is_high_vol):
            continue
        
        trigger_close = row["close"]
        trigger_type = "BULLISH" if row["close"] > row["open"] else "BEARISH"
        
        consol_count = 0
        consol_high = row["high"]
        consol_low = row["low"]
        pattern_end_idx = -1
        
        for j in range(i + 1, n):
            c = df.iloc[j]
            consol_high = max(consol_high, c["high"])
            consol_low = min(consol_low, c["low"])
            consol_range = (consol_high - consol_low) / trigger_close
            
            if consol_range > PARAMS["consol_max_range_pct"]:
                if consol_count >= PARAMS["consol_min"]:
                    pattern_end_idx = j - 1
                    break
                else:
                    consol_count = 0
                    consol_high = c["high"]
                    consol_low = c["low"]
                    continue
            
            cb = abs(c["close"] - c["open"])
            cr = c["high"] - c["low"]
            is_small = (cb / cr < PARAMS["consol_body_pct"]) if cr > 0 else True
            
            if is_small:
                consol_count += 1
            else:
                if consol_count >= PARAMS["consol_min"]:
                    pattern_end_idx = j - 1
                    break
                consol_count = 0
                consol_high = c["high"]
                consol_low = c["low"]
        
        if consol_count >= PARAMS["consol_min"] and pattern_end_idx < 0:
            pattern_end_idx = j if j < n else n - 1
        
        if pattern_end_idx > 0:
            triggers.append({
                "trigger_idx": i,
                "trigger_close": trigger_close,
                "trigger_type": trigger_type,
                "pattern_end_idx": pattern_end_idx,
            })
    
    return triggers, df


# ─── Measure forward returns ───
horizons = [1, 3, 5, 10, 20, 60]
results = []

files = sorted(glob.glob(f"{DATA}/*_ONE_DAY.csv"))
print(f"Scanning {len(files)} stocks for forward analysis...")

for idx, f in enumerate(files):
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    triggers, df_raw = scan_stock_for_patterns(df)
    
    for tr in triggers:
        end_idx = tr["pattern_end_idx"]
        close_at_end = df.iloc[end_idx]["close"]
        trigger_close = tr["trigger_close"]
        trigger_type = tr["trigger_type"]
        
        for h in horizons:
            fwd_idx = end_idx + h
            if fwd_idx < len(df):
                fwd_close = df.iloc[fwd_idx]["close"]
                fwd_return = (fwd_close - close_at_end) / close_at_end * 100
            else:
                fwd_return = np.nan
            
            results.append({
                "symbol": sym,
                "trigger_type": trigger_type,
                "trigger_close": trigger_close,
                "close_at_end": close_at_end,
                "horizon": h,
                "fwd_return": fwd_return,
            })
    
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"  [{idx+1}/{len(files)}] {sym} - {len(triggers)} patterns", flush=True)

print(f"\nTotal observations: {len(results)}")

# ─── Analyze ───
rdf = pd.DataFrame(results)
rdf = rdf.dropna(subset=["fwd_return"])

print(f"\n{'='*60}")
print("FORWARD PERFORMANCE AFTER PATTERN END")
print(f"{'='*60}")

for ttype in ["BULLISH", "BEARISH"]:
    print(f"\n--- {ttype} Triggers ---")
    print(f"{'Horizon':>8s} {'Count':>8s} {'Avg Ret%':>10s} {'Win Rate':>10s} {'Med Ret%':>10s}")
    for h in horizons:
        sub = rdf[(rdf["trigger_type"] == ttype) & (rdf["horizon"] == h)]
        if len(sub) == 0:
            continue
        avg_ret = sub["fwd_return"].mean()
        med_ret = sub["fwd_return"].median()
        win_rate = (sub["fwd_return"] > 0).mean() * 100
        print(f"  {h:>4d} days  {len(sub):>8d} {avg_ret:>+9.2f}% {win_rate:>8.1f}% {med_ret:>+9.2f}%")

# ─── Save detailed results ───
rdf.to_csv(os.path.join(OUTPUT, "big_candle_forward.csv"), index=False)
print(f"\nSaved forward results to big_candle_forward.csv")

print(f"\nDone.")

# ─── BEST/WORST performing stocks ───
print(f"\n{'='*60}")
print("TOP/BOTTOM STOCKS (5-day forward after BULLISH pattern)")
print(f"{'='*60}")

sub5 = rdf[(rdf["horizon"] == 5) & (rdf["trigger_type"] == "BULLISH")]
stock_perf = sub5.groupby("symbol")["fwd_return"].agg(["mean", "count", lambda x: (x > 0).mean()*100])
stock_perf = stock_perf[stock_perf["count"] >= 10].sort_values("mean")
print(f"\nWorst 10:")
for sym, row in stock_perf.head(10).iterrows():
    print(f"  {sym:20s} avg_ret={row['mean']:+.2f}% count={int(row['count']):3d} win_rate={row['<lambda_0>']:.0f}%")
print(f"\nBest 10:")
for sym, row in stock_perf.tail(10).iloc[::-1].iterrows():
    print(f"  {sym:20s} avg_ret={row['mean']:+.2f}% count={int(row['count']):3d} win_rate={row['<lambda_0>']:.0f}%")
