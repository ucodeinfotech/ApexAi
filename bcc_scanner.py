"""
BCC Scanner — Big Candle + Consolidation pattern detector
Run manually whenever you want. Tracks seen patterns to avoid re-alerts.

Usage: python bcc_scanner.py
"""
import pandas as pd, numpy as np, glob, os, hashlib
from datetime import datetime, timezone

DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data"
OUT_DIR = "C:/Users/pc/Downloads/stock hist data/backtest_results"
SEEN_FILE = os.path.join(OUT_DIR, "seen_patterns_bcc.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── PARAMS ───
BODY_MULT = 2.0
VOL_MULT = 1.5
WICK_PCT = 0.20
AVG_PERIOD = 20
CONSOL_MIN = 3
CONSOL_BODY_PCT = 0.30
CONSOL_MAX_RANGE_PCT = 0.05  # 5% of trigger close

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def scan_stock(df, symbol, seen_set):
    """Scan one stock for new BCC patterns. Returns list of new patterns."""
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(AVG_PERIOD, min_periods=AVG_PERIOD).mean().shift(1)
    df["avg_vol"] = df["volume"].rolling(AVG_PERIOD, min_periods=AVG_PERIOD).mean().shift(1)
    df["rsi_val"] = rsi(df["close"], 14).shift(1)
    
    new = []
    
    for i in range(AVG_PERIOD, n):
        row = df.iloc[i]
        avg_body = row["avg_body"]
        avg_vol = row["avg_vol"]
        if pd.isna(avg_body) or pd.isna(avg_vol) or avg_body == 0 or avg_vol == 0:
            continue
        
        body = row["body"]
        total_range = row["range"]
        upper_wick = row["high"] - max(row["close"], row["open"])
        if total_range == 0:
            continue
        wick_ratio = upper_wick / total_range
        
        # ─── Big candle check ───
        if not (body > avg_body * BODY_MULT and row["volume"] > avg_vol * VOL_MULT and wick_ratio < WICK_PCT):
            continue
        
        trigger_type = "BULLISH" if row["close"] > row["open"] else "BEARISH"
        trigger_close = row["close"]
        trigger_date = pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d")
        
        # ─── Generate unique ID ───
        raw_id = f"{symbol}_{trigger_date}"
        pat_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]
        
        if pat_id in seen_set:
            continue
        
        # ─── Scan forward for consolidation ───
        consol_count = 0
        consol_high = row["high"]
        consol_low = row["low"]
        consol_end_idx = -1
        consol_vols = []
        
        for j in range(i + 1, n):
            c = df.iloc[j]
            consol_high = max(consol_high, c["high"])
            consol_low = min(consol_low, c["low"])
            consol_range_pct = (consol_high - consol_low) / trigger_close
            
            if consol_range_pct > CONSOL_MAX_RANGE_PCT:
                if consol_count >= CONSOL_MIN:
                    consol_end_idx = j - 1
                    break
                consol_count = 0
                consol_high = c["high"]
                consol_low = c["low"]
                consol_vols = []
                continue
            
            cb = abs(c["close"] - c["open"])
            cr = c["high"] - c["low"]
            is_small = (cb / cr < CONSOL_BODY_PCT) if cr > 0 else True
            
            if is_small:
                consol_count += 1
                consol_vols.append(c["volume"])
            else:
                if consol_count >= CONSOL_MIN:
                    consol_end_idx = j - 1
                    break
                consol_count = 0
                consol_high = c["high"]
                consol_low = c["low"]
                consol_vols = []
        
        if consol_count >= CONSOL_MIN and consol_end_idx < 0:
            consol_end_idx = min(j, n - 1)
        
        if consol_end_idx < 0:
            continue
        
        last_close = df.iloc[consol_end_idx]["close"]
        change_pct = (last_close - trigger_close) / trigger_close * 100
        
        if change_pct > 2.0:
            status = "BROKEN UP"
        elif change_pct < -2.0:
            status = "BROKEN DOWN"
        else:
            status = "CONSOLIDATING"
        
        avg_consol_vol = np.mean(consol_vols) if consol_vols else 0
        
        rsi_val = row["rsi_val"]
        rsi_str = f"{rsi_val:.0f}" if not pd.isna(rsi_val) else "N/A"
        
        new.append({
            "pattern_id": pat_id,
            "symbol": symbol,
            "trigger_date": trigger_date,
            "trigger_type": trigger_type,
            "trigger_close": round(trigger_close, 2),
            "trigger_body": round(body, 2),
            "body_vs_avg": round(body / avg_body, 2),
            "vol_vs_avg": round(row["volume"] / avg_vol, 2),
            "wick_pct": round(wick_ratio * 100, 1),
            "rsi": rsi_str,
            "consol_candles": consol_count,
            "consol_range_pct": round((consol_high - consol_low) / trigger_close * 100, 2),
            "consol_vol_vs_trigger": round(avg_consol_vol / row["volume"], 2) if row["volume"] > 0 else 0,
            "status": status,
            "last_close": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "detected_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    return new


def print_alert(p, show_header=False):
    """Pretty-print a new pattern alert."""
    color = "\033[92m" if p["trigger_type"] == "BULLISH" else "\033[91m"
    reset = "\033[0m"
    bold = "\033[1m"
    sym = p["symbol"]
    ttype = p["trigger_type"]
    close = p["trigger_close"]
    date = p["trigger_date"]
    consol = p["consol_candles"]
    crange = p["consol_range_pct"]
    status = p["status"]
    body_r = p["body_vs_avg"]
    vol_r = p["vol_vs_avg"]
    rsi = p["rsi"]
    change = p["change_pct"]
    
    status_color = {
        "CONSOLIDATING": "\033[93m",
        "BROKEN UP": "\033[92m",
        "BROKEN DOWN": "\033[91m",
    }.get(status, "")
    
    line = (f"{color}{bold}[BCC]{reset} {sym:16s} {color}{ttype:>8s}{reset} @ "
            f"{close:>9.2f} | {date} | "
            f"Body:{body_r:.1f}x Vol:{vol_r:.1f}x RSI:{rsi} | "
            f"Consol:{consol}d ({crange:.1f}%) | "
            f"{status_color}{status}{reset} (chg{change:+.2f}%)")
    print(line)


# ─── MAIN ───
def main():
    files = sorted(glob.glob(f"{DATA_DIR}/*_ONE_DAY.csv"))
    print(f"{'='*70}")
    print(f"  BCC SCANNER — Big Candle + Consolidation")
    print(f"  {len(files)} stocks | Params: body>{BODY_MULT}x, vol>{VOL_MULT}x, "
          f"consol>{CONSOL_MIN}d/{CONSOL_MAX_RANGE_PCT*100:.0f}%")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Load seen patterns
    seen = set()
    if os.path.exists(SEEN_FILE):
        seen_df = pd.read_csv(SEEN_FILE)
        seen = set(seen_df["pattern_id"].tolist())
        print(f"  Loaded {len(seen)} previously seen patterns from {SEEN_FILE}\n")
    else:
        print(f"  No seen_patterns file found — starting fresh\n")
        # Create header
        pd.DataFrame(columns=[
            "pattern_id","symbol","trigger_date","trigger_type","trigger_close",
            "trigger_body","body_vs_avg","vol_vs_avg","wick_pct","rsi",
            "consol_candles","consol_range_pct","consol_vol_vs_trigger",
            "status","last_close","change_pct","detected_date"
        ]).to_csv(SEEN_FILE, index=False)
    
    # Scan
    all_new = []
    scan_count = 0
    
    for idx, f in enumerate(files):
        sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
        df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        pats = scan_stock(df, sym, seen)
        for p in pats:
            print_alert(p)
        all_new.extend(pats)
        
        if pats:
            scan_count += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(files)} stocks scanned...", flush=True)
    
    print(f"\n{'='*70}")
    if all_new:
        # Save
        new_df = pd.DataFrame(all_new)
        new_df.to_csv(SEEN_FILE, mode="a", header=False, index=False)
        print(f"  {len(all_new)} NEW patterns found across {scan_count} stocks")
        print(f"  Saved to {SEEN_FILE}")
    else:
        print(f"  No new patterns found. ({len(files)} stocks scanned)")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
