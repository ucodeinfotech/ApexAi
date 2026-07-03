"""
Improved Big Candle + Consolidation Scanner with additional filters
"""
import pandas as pd, numpy as np, os, glob, time

DATA = "C:/Users/pc/Downloads/stock hist data/comprehensive_data"
OUTPUT = "C:/Users/pc/Downloads/stock hist data/backtest_results"
os.makedirs(OUTPUT, exist_ok=True)

# ─── PARAM SETS (base vs improved) ───
BASE = {
    "big_body_mult": 2.0, "wick_pct": 0.20, "vol_mult": 1.5,
    "avg_period": 20, "consol_min": 3, "consol_body_pct": 0.30,
    "consol_max_range_pct": 0.05,
}

IMPROVED = {
    "big_body_mult": 2.0,
    "big_atr_mult": 0,
    "wick_pct": 0.20,
    "vol_mult": 1.5,
    "avg_period": 20,
    "consol_min": 3,
    "consol_body_pct": 0.30,
    "consol_max_range_pct": 0.05,
    "vol_decline_pct": 0.6,      # NEW: consol avg vol < 60% of trigger vol
    "rsi_reverse": True,         # NEW: RSI < 50 for bullish, > 50 for bearish
    "min_avg_vol_14d": 100000,   # NEW: minimum liquidity filter
    "min_close": 10,             # NEW: minimum price filter
}

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def scan_stock(df, params, name=""):
    """Scan one stock. Returns patterns dict."""
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    patterns = []
    
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(params["avg_period"], min_periods=params["avg_period"]).mean().shift(1)
    df["avg_vol"] = df["volume"].rolling(params["avg_period"], min_periods=params["avg_period"]).mean().shift(1)
    df["avg_vol_14"] = df["volume"].rolling(14, min_periods=14).mean().shift(1)
    df["atr"] = df["range"].rolling(14, min_periods=14).mean().shift(1)
    df["rsi_val"] = rsi(df["close"], 14).shift(1)
    
    for i in range(params["avg_period"] + 14, n):
        row = df.iloc[i]
        avg_body = row["avg_body"]
        avg_vol = row["avg_vol"]
        if pd.isna(avg_body) or pd.isna(avg_vol):
            continue
        
        body = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["close"], row["open"])
        total_range = row["high"] - row["low"]
        if total_range == 0 or avg_body == 0 or avg_vol == 0:
            continue
        
        # ─── Trigger checks ───
        wick_ratio = upper_wick / total_range
        ok_big = body > avg_body * params["big_body_mult"]
        ok_wick = wick_ratio < params["wick_pct"]
        ok_vol = row["volume"] > avg_vol * params["vol_mult"]
        
        if "big_atr_mult" in params:
            atr = row["atr"]
            ok_atr = total_range > atr * params["big_atr_mult"] if not pd.isna(atr) and atr > 0 else False
        else:
            ok_atr = True
        
        if "rsi_reverse" in params:
            r = row["rsi_val"]
            if row["close"] > row["open"]:
                ok_rsi = not pd.isna(r) and r < 50  # Bullish: NOT overbought
            else:
                ok_rsi = not pd.isna(r) and r > 50  # Bearish: NOT oversold
        elif "rsi_threshold" in params:
            r = row["rsi_val"]
            if row["close"] > row["open"]:
                ok_rsi = not pd.isna(r) and r > params["rsi_threshold"]
            else:
                ok_rsi = not pd.isna(r) and r < params["rsi_threshold"]
        else:
            ok_rsi = True
        
        ok_liquidity = True
        if "min_avg_vol_14d" in params:
            ok_liquidity = row["avg_vol_14"] > params["min_avg_vol_14d"] if not pd.isna(row["avg_vol_14"]) else False
        ok_price = True
        if "min_close" in params:
            ok_price = row["close"] > params["min_close"]
        
        if not (ok_big and ok_wick and ok_vol and ok_atr and ok_rsi and ok_liquidity and ok_price):
            continue
        
        trigger_type = "BULLISH" if row["close"] > row["open"] else "BEARISH"
        trigger_close = row["close"]
        trigger_vol = row["volume"]
        
        # ─── Scan consolidation ───
        consol_count = 0
        consol_high = row["high"]
        consol_low = row["low"]
        consol_vols = []
        pattern_end_idx = -1
        
        for j in range(i + 1, n):
            c = df.iloc[j]
            consol_high = max(consol_high, c["high"])
            consol_low = min(consol_low, c["low"])
            consol_range = (consol_high - consol_low) / trigger_close
            
            if consol_range > params["consol_max_range_pct"]:
                if consol_count >= params["consol_min"]:
                    pattern_end_idx = j - 1
                    break
                consol_count = 0
                consol_high = c["high"]
                consol_low = c["low"]
                consol_vols = []
                continue
            
            cb = abs(c["close"] - c["open"])
            cr = c["high"] - c["low"]
            is_small = (cb / cr < params["consol_body_pct"]) if cr > 0 else True
            
            if is_small:
                consol_count += 1
                consol_vols.append(c["volume"])
            else:
                if consol_count >= params["consol_min"]:
                    pattern_end_idx = j - 1
                    break
                consol_count = 0
                consol_high = c["high"]
                consol_low = c["low"]
                consol_vols = []
        
        if consol_count >= params["consol_min"] and pattern_end_idx < 0:
            pattern_end_idx = min(j, n - 1)
        
        if pattern_end_idx < 0:
            continue
        
        # ─── Volume decline filter ───
        if "vol_decline_pct" in params and consol_vols:
            avg_consol_vol = np.mean(consol_vols)
            if avg_consol_vol > trigger_vol * params["vol_decline_pct"]:
                continue  # Volume didn't decline enough during consolidation
        
        # ─── Pattern valid ───
        last_idx = pattern_end_idx
        last_close = df.iloc[last_idx]["close"]
        
        status = "CONSOLIDATING"
        if last_close > trigger_close * 1.02:
            status = "BROKEN UP"
        elif last_close < trigger_close * 0.98:
            status = "BROKEN DOWN"
        
        patterns.append({
            "trigger_type": trigger_type,
            "trigger_close": trigger_close,
            "trigger_body": body,
            "trigger_range": total_range,
            "trigger_wick_pct": round(wick_ratio * 100, 1),
            "trigger_vol_ratio": round(row["volume"] / avg_vol, 2) if avg_vol > 0 else 0,
            "atr_mult": round(total_range / row["atr"], 2) if not pd.isna(row["atr"]) and row["atr"] > 0 else 0,
            "rsi_at_trigger": round(row["rsi_val"], 1) if not pd.isna(row["rsi_val"]) else 0,
            "consol_count": consol_count,
            "consol_range_pct": round((consol_high - consol_low) / trigger_close * 100, 2),
            "status": status,
            "pattern_end_idx": pattern_end_idx,
            "trigger_idx": i,
            "last_close": last_close,
        })
    
    return patterns


# ─── SCAN ALL STOCKS FOR BOTH PARAM SETS ───
files = sorted(glob.glob(f"{DATA}/*_ONE_DAY.csv"))
print(f"Scanning {len(files)} stocks...")

all_base = []
all_imp = []
start = time.time()

for idx, f in enumerate(files):
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    base_p = scan_stock(df, BASE, "BASE")
    for p in base_p: p["symbol"] = sym
    all_base.extend(base_p)
    
    imp_p = scan_stock(df, IMPROVED, "IMPROVED")
    for p in imp_p: p["symbol"] = sym
    all_imp.extend(imp_p)
    
    if (idx + 1) % 100 == 0:
        print(f"  [{idx+1}/{len(files)}] {sym} - Base:{len(base_p)} Imp:{len(imp_p)} ({time.time()-start:.0f}s)", flush=True)

elapsed = time.time() - start
print(f"\nComplete in {elapsed:.0f}s")

# ─── COMPARE ───
bd = pd.DataFrame(all_base)
idf = pd.DataFrame(all_imp)
print(f"\nBASE:   {len(bd)} patterns across {bd['symbol'].nunique()} stocks")
print(f"IMPROVED: {len(idf)} patterns across {idf['symbol'].nunique()} stocks")

# Forward analysis for both
files_map = {}
for f in files:
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    files_map[sym] = f

def measure_forward(patterns_df, df_map):
    results = []
    for _, p in patterns_df.iterrows():
        sym = p["symbol"]
        end_idx = p["pattern_end_idx"]
        if sym not in df_map:
            continue
        df = df_map[sym]
        close_end = df.iloc[end_idx]["close"]
        for h in [1, 3, 5, 10, 20, 60]:
            fwd = end_idx + h
            if fwd < len(df):
                ret = (df.iloc[fwd]["close"] - close_end) / close_end * 100
            else:
                ret = np.nan
            results.append({
                "symbol": sym, "trigger_type": p["trigger_type"],
                "horizon": h, "fwd_return": ret,
                "trigger_close": p["trigger_close"],
            })
    return pd.DataFrame(results)

print("Measuring forward returns for BASE...")
df_map = {}
for f in files:
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    d = pd.read_csv(f)
    d["datetime"] = pd.to_datetime(d["datetime"])
    df_map[sym] = d.sort_values("datetime").reset_index(drop=True)

base_fwd = measure_forward(bd, df_map)
imp_fwd = measure_forward(idf, df_map)

print(f"BASE forward obs: {len(base_fwd)}")
print(f"IMPROVED forward obs: {len(imp_fwd)}")

# ─── PRINT COMPARISON ───
print(f"\n{'='*70}")
print(f"{'COMPARISON: BASE vs IMPROVED':^70}")
print(f"{'='*70}")

for label, pf in [("BASE", bd), ("IMPROVED", idf)]:
    b = pf[pf["trigger_type"]=="BULLISH"]
    be = pf[pf["trigger_type"]=="BEARISH"]
    b_ok = (b["last_close"] > b["trigger_close"]).sum() if len(b)>0 else 0
    be_ok = (be["last_close"] < be["trigger_close"]).sum() if len(be)>0 else 0
    print(f"\n{label}:")
    print(f"  Patterns: {len(pf)} | Bull:{len(b)} Bear:{len(be)}")
    print(f"  Bull success: {b_ok}/{len(b)} = {b_ok/len(b)*100:.1f}%" if len(b)>0 else "  Bull: N/A")
    print(f"  Bear success: {be_ok}/{len(be)} = {be_ok/len(be)*100:.1f}%" if len(be)>0 else "  Bear: N/A")

# Forward comparison
for h in [1, 5, 20, 60]:
    print(f"\n--- {h}-day Forward ---")
    print(f"{'Set':>10s} {'Type':>8s} {'Count':>8s} {'AvgRet':>10s} {'Win%':>8s}")
    for label, ff in [("BASE", base_fwd), ("IMPROVED", imp_fwd)]:
        for ttype in ["BULLISH", "BEARISH"]:
            sub = ff[(ff["horizon"]==h) & (ff["trigger_type"]==ttype)]
            if len(sub)==0: continue
            avg = sub["fwd_return"].mean()
            wr = (sub["fwd_return"]>0).mean()*100
            print(f"{label:>10s} {ttype:>8s} {len(sub):>8d} {avg:>+9.2f}% {wr:>7.1f}%")

# Save
bd.to_csv(os.path.join(OUTPUT, "big_candle_base.csv"), index=False)
idf.to_csv(os.path.join(OUTPUT, "big_candle_improved.csv"), index=False)
base_fwd.to_csv(os.path.join(OUTPUT, "big_candle_base_fwd.csv"), index=False)
imp_fwd.to_csv(os.path.join(OUTPUT, "big_candle_improved_fwd.csv"), index=False)
print(f"\nSaved.")

# Quick stock-level improved winners
print(f"\n{'='*70}")
print("BEST STOCKS (IMPROVED, 5-day forward)")
imp5 = imp_fwd[imp_fwd["horizon"]==5].groupby("symbol")["fwd_return"].agg(["mean","count",lambda x: (x>0).mean()*100])
imp5 = imp5[imp5["count"]>=5].sort_values("mean")
print("Worst:")
for sym, r in imp5.head(10).iterrows():
    print(f"  {sym:20s} avg={r['mean']:+.2f}% cnt={int(r['count'])} wr={r['<lambda_0>']:.0f}%")
print("Best:")
for sym, r in imp5.tail(10).iloc[::-1].iterrows():
    print(f"  {sym:20s} avg={r['mean']:+.2f}% cnt={int(r['count'])} wr={r['<lambda_0>']:.0f}%")
