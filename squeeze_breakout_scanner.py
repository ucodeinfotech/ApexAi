"""
Squeeze Breakout Scanner v2 (backward scan for consolidation from each big candle)
"""
import pandas as pd, numpy as np, glob, os, time

DATA = "C:/Users/pc/Downloads/stock hist data/comprehensive_data"
OUTPUT = "C:/Users/pc/Downloads/stock hist data/backtest_results"
os.makedirs(OUTPUT, exist_ok=True)

LOOKBACK = 60

def scan_squeeze(df):
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1)
    df["avg_vol"] = df["volume"].rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1)
    
    patterns = []
    
    for i in range(LOOKBACK, n):
        row = df.iloc[i]
        avg_body = row["avg_body"]
        avg_vol = row["avg_vol"]
        if pd.isna(avg_body) or pd.isna(avg_vol) or avg_body == 0 or avg_vol == 0:
            continue
        
        body = row["body"]
        total_range = row["range"]
        upper_wick = row["high"] - max(row["close"], row["open"])
        wick_ratio = upper_wick / total_range if total_range > 0 else 1
        
        # Is this a potential breakout candle?
        if not (body > avg_body * 2.0 and row["volume"] > avg_vol * 1.5 and wick_ratio < 0.25):
            continue
        
        # Scan backward for consolidation
        consol_start = -1
        consol_set = False
        consol_high = 0.0
        consol_low = 0.0
        consol_vols = []
        consol_bodies = []
        consol_count = 0
        
        for j in range(i - 1, -1, -1):
            c = df.iloc[j]
            cb = abs(c["close"] - c["open"])
            cr = c["high"] - c["low"]
            cb_ratio = cb / cr if cr > 0 else 0
            
            is_small = cb_ratio < 0.30 and cb < avg_body * 1.2
            if not is_small:
                if consol_count >= 3:
                    consol_start = j + 1
                    break
                consol_count = 0
                consol_set = False
                consol_vols = []
                consol_bodies = []
                continue
            
            if not consol_set:
                consol_high = c["high"]
                consol_low = c["low"]
                consol_set = True
            else:
                consol_high = max(consol_high, c["high"])
                consol_low = min(consol_low, c["low"])
            
            total_consol_range = (consol_high - consol_low) / row["close"]
            
            if total_consol_range > 0.06:
                break
            
            consol_count += 1
            consol_vols.append(c["volume"])
            consol_bodies.append(cb)
            
            if consol_count >= 5:
                consol_start = j
                break
        
        if consol_start < 0 or consol_count < 3:
            continue
        
        # Determine breakout direction
        if row["close"] > row["open"] and row["close"] > consol_high:
            btype = "BULLISH"
        elif row["close"] < row["open"] and row["close"] < consol_low:
            btype = "BEARISH"
        else:
            continue  # Didn't break out of consolidation range
        
        # Volume decline check
        avg_consol_vol = np.mean(consol_vols) if consol_vols else 0
        
        patterns.append({
            "type": btype,
            "symbol": "",
            "close": row["close"],
            "body": body,
            "range": total_range,
            "body_ratio": round(body / avg_body, 2),
            "vol_ratio": round(row["volume"] / avg_vol, 2),
            "consol_days": consol_count,
            "consol_range_pct": round((consol_high - consol_low) / row["close"] * 100, 2),
            "consol_vol_ratio": round(avg_consol_vol / avg_vol, 2) if avg_vol > 0 else 0,
            "idx": i,
            "consol_start": consol_start,
        })
    
    return patterns


# ─── SCAN ───
files = sorted(glob.glob(f"{DATA}/*_ONE_DAY.csv"))
print(f"Scanning {len(files)} stocks for squeeze breakout patterns...")

all_pats = []
start = time.time()

for idx, f in enumerate(files):
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    pats = scan_squeeze(df)
    for p in pats:
        p["symbol"] = sym
    all_pats.extend(pats)
    
    if (idx + 1) % 100 == 0:
        dur = time.time() - start
        print(f"  [{idx+1}/{len(files)}] {sym:20s} -> {len(pats):>3d} patterns [{dur:.0f}s]", flush=True)

elapsed = time.time() - start
print(f"\nComplete in {elapsed:.0f}s")

df_out = pd.DataFrame(all_pats)
print(f"Total patterns: {len(df_out)} across {df_out['symbol'].nunique()} stocks")
if len(df_out) > 0:
    print(f"Bullish: {(df_out['type']=='BULLISH').sum()} | Bearish: {(df_out['type']=='BEARISH').sum()}")
else:
    print("No patterns found.")
    exit()

# ─── Forward returns ───
files_map = {}
for f in files:
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    d = pd.read_csv(f)
    d["datetime"] = pd.to_datetime(d["datetime"])
    files_map[sym] = d.sort_values("datetime").reset_index(drop=True)

print("Measuring forward returns...")
fwd = []
for _, p in df_out.iterrows():
    sym = p["symbol"]
    idx = p["idx"]
    close = p["close"]
    dl_df = files_map.get(sym)
    if dl_df is None or idx >= len(dl_df)-1: continue
    for h in [1,3,5,10,20,60]:
        fwd_idx = idx + h
        if fwd_idx < len(dl_df):
            ret = (dl_df.iloc[fwd_idx]["close"] - close) / close * 100
        else:
            ret = np.nan
        fwd.append({"symbol":sym,"type":p["type"],"horizon":h,"fwd_return":ret})
fwd = pd.DataFrame(fwd)
print(f"Forward observations: {len(fwd)}")

# ─── RESULTS ───
bull = df_out[df_out["type"]=="BULLISH"]
bear = df_out[df_out["type"]=="BEARISH"]
print(f"\n{'='*70}")
print(f"{'SQUEEZE BREAKOUT PATTERN - RESULTS':^70}")
print(f"{'='*70}")
print(f"Total: {len(df_out)} | Bull: {len(bull)} ({len(bull)/len(df_out)*100:.1f}%) | Bear: {len(bear)} ({len(bear)/len(df_out)*100:.1f}%)")
print(f"Avg consol days: {df_out['consol_days'].mean():.1f}")
print(f"Avg consol range: {df_out['consol_range_pct'].mean():.2f}%")
print(f"Avg body ratio: {df_out['body_ratio'].mean():.2f}x")

# 1d directional accuracy
dir_res = []
for _, p in df_out.iterrows():
    sym = p["symbol"]
    idx = p["idx"]
    d = files_map.get(sym)
    if d is None or idx >= len(d)-1: continue
    fwd_close = d.iloc[idx+1]["close"]
    correct = (p["type"]=="BULLISH" and fwd_close>p["close"]) or (p["type"]=="BEARISH" and fwd_close<p["close"])
    dir_res.append({"type":p["type"],"correct":correct})
if dir_res:
    dir_df = pd.DataFrame(dir_res)
    for t in ["BULLISH","BEARISH"]:
        sub = dir_df[dir_df["type"]==t]
        if len(sub) > 0:
            acc = sub["correct"].mean()*100
            print(f"  {t:>8s} 1d accuracy: {acc:.1f}% (n={len(sub)})")

# Forward returns
print(f"\n{'='*70}")
for h in [1,3,5,10,20,60]:
    print(f"\n--- {h}-day Forward ---")
    print(f"{'Type':>10s} {'Count':>8s} {'AvgRet':>10s} {'Median':>10s} {'Win%':>8s}")
    for ttype in ["BULLISH","BEARISH"]:
        sub = fwd[(fwd["horizon"]==h) & (fwd["type"]==ttype)]
        if len(sub)==0: continue
        avg = sub["fwd_return"].mean()
        med = sub["fwd_return"].median()
        wr = (sub["fwd_return"]>0).mean()*100
        print(f"{ttype:>10s} {len(sub):>8d} {avg:>+9.2f}% {med:>+9.2f}% {wr:>7.1f}%")

# Stock-level
print(f"\n{'='*70}")
fwd5 = fwd[fwd["horizon"]==5]
if len(fwd5) > 5:
    stocks = fwd5.groupby("symbol")["fwd_return"].agg(["mean","count",lambda x: (x>0).mean()*100])
    stocks = stocks[stocks["count"]>=3].sort_values("mean")
    if len(stocks) > 0:
        print("WORST 10 (5d forward):")
        for sym, r in stocks.head(10).iterrows():
            print(f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}% wr={r['<lambda_0>']:.0f}%")
        print("BEST 10:")
        for sym, r in stocks.tail(10).iloc[::-1].iterrows():
            print(f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}% wr={r['<lambda_0>']:.0f}%")
    print(f"\nStocks with 100% WR (n>=3):")
    always_win = stocks[stocks['<lambda_0>']==100]
    for sym, r in always_win.head(15).iterrows():
        print(f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}%")
    print(f"\nStocks with 0% WR (n>=3):")
    always_lose = stocks[stocks['<lambda_0>']==0]
    for sym, r in always_lose.head(15).iterrows():
        print(f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}%")

# Save
df_out.to_csv(os.path.join(OUTPUT, "squeeze_breakout_patterns.csv"), index=False)
fwd.to_csv(os.path.join(OUTPUT, "squeeze_breakout_fwd.csv"), index=False)
print(f"\nSaved to {OUTPUT}/squeeze_breakout_*.csv")
