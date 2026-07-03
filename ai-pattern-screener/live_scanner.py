"""Live scanner: uses cached daily candle data and detects BCC/Squeeze patterns.
Falls back to Dhan API if cache is missing/stale."""

import sys
import json
import os
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dhan_cache_manager import (
    load_from_cache, save_to_cache, is_cache_valid, get_cached_candles,
    compute_indicators, fetch_stock_candles, login as dhan_login
)

BODY_MULT = 2.0
VOL_MULT = 1.5
WICK_PCT = 0.20
AVG_PERIOD = 20
CONSOL_MIN = 3
CONSOL_BODY_PCT = 0.30
CONSOL_MAX_RANGE_PCT = 0.05

def get_candles(symbol):
    """Get candles from cache first, fallback to Dhan API."""
    cached = get_cached_candles(symbol)
    if cached and len(cached) >= 22 and is_cache_valid(symbol):
        compute_indicators(cached)
        return cached

    api = dhan_login()
    from dhan_cache_manager import build_dhan_security_map
    dhan_map = build_dhan_security_map()
    sec_id = dhan_map.get(symbol)
    if not sec_id:
        return None
    _, candles, err = fetch_stock_candles(api, sec_id, symbol)
    if candles and len(candles) >= 22:
        compute_indicators(candles)
        return candles
    return None


LOOKBACK = [10]  # mutable list so --lookback arg can override

def detect_bcc_patterns(candles, squeeze_mode=False, lookback=None):
    if lookback is None: lookback = LOOKBACK[0]
    """Detect BCC or Squeeze Breakout patterns from last `lookback` candles."""
    patterns = []
    min_candles = AVG_PERIOD + 2
    if len(candles) < min_candles:
        return patterns

    start_idx = max(AVG_PERIOD, len(candles) - lookback)

    for i in range(start_idx, len(candles)):
        c = candles[i]
        avg_body = c.get("avgBody20", 0)
        avg_vol = c.get("avgVol20", 0)
        if not avg_body or not avg_vol:
            continue
        if c["body"] < avg_body * BODY_MULT:
            continue
        if c["volume"] < avg_vol * VOL_MULT:
            continue
        wick = (c["high"] - max(c["close"], c["open"])) / c["range"]
        if wick >= WICK_PCT:
            continue

        t_type = "BULLISH" if c["close"] > c["open"] else "BEARISH"
        consol_count = 0
        consol_high = c["high"]
        consol_low = c["low"]
        consol_end = -1

        for j in range(i + 1, len(candles)):
            n = candles[j]
            consol_high = max(consol_high, n["high"])
            consol_low = min(consol_low, n["low"])
            rng_pct = (consol_high - consol_low) / c["close"]
            if rng_pct > CONSOL_MAX_RANGE_PCT:
                if consol_count >= 2:
                    consol_end = j - 1
                    break
                consol_count = 0
                consol_high = n["high"]
                consol_low = n["low"]
                continue
            if n["body"] / n["range"] < CONSOL_BODY_PCT:
                consol_count += 1
            else:
                if consol_count >= 2:
                    consol_end = j - 1
                    break
                consol_count = 0
                consol_high = n["high"]
                consol_low = n["low"]

        if consol_count >= 2 and consol_end < 0:
            consol_end = len(candles) - 1

        if consol_end < 0:
            patterns.append({
                "triggerDate": c.get("date", ""), "triggerTime": c["time"],
                "triggerType": t_type, "triggerClose": c["close"],
                "bodyVsAvg": round(c["body"] / avg_body, 2),
                "volVsAvg": round(c["volume"] / avg_vol, 2),
                "rsi": round(c.get("rsi14", 50)),
                "consolCount": 0, "consolHigh": c["high"], "consolLow": c["low"],
                "consolRangePct": 0, "consolEndTime": c["time"],
                "status": "FRESH TRIGGER", "changePct": 0,
            })
            continue

        last_close = candles[consol_end]["close"]
        change_pct = (last_close - c["close"]) / c["close"] * 100
        status = "BROKEN UP" if change_pct > 2 else "BROKEN DOWN" if change_pct < -2 else "CONSOLIDATING"
        patterns.append({
            "triggerDate": c.get("date", ""), "triggerTime": c["time"],
            "triggerType": t_type, "triggerClose": c["close"],
            "triggerHigh": c["high"], "triggerLow": c["low"],
            "bodyVsAvg": round(c["body"] / avg_body, 2),
            "volVsAvg": round(c["volume"] / avg_vol, 2),
            "rsi": round(c.get("rsi14", 50)),
            "consolCount": consol_count, "consolHigh": round(consol_high, 2),
            "consolLow": round(consol_low, 2),
            "consolRangePct": round((consol_high - consol_low) / c["close"] * 100, 2),
            "consolEndTime": candles[consol_end]["time"],
            "status": status, "changePct": round(change_pct, 2),
        })

    # Squeeze breakout (reverse pattern)
    if not squeeze_mode:
        for i in range(start_idx + 2, len(candles)):
            c = candles[i]
            avg_body = c.get("avgBody20", 0)
            avg_vol = c.get("avgVol20", 0)
            if not avg_body or not avg_vol:
                continue
            consol_before = 0
            ch, cl = float("-inf"), float("inf")
            for j in range(i - 1, max(i - 8, start_idx - 1), -1):
                n = candles[j]
                ch = max(ch, n["high"])
                cl = min(cl, n["low"])
                if n["body"] / n["range"] < CONSOL_BODY_PCT:
                    consol_before += 1
                else:
                    if consol_before >= 2:
                        break
                    consol_before = 0
                    ch, cl = float("-inf"), float("inf")
            if consol_before >= 2 and c["body"] >= avg_body * BODY_MULT and c["volume"] >= avg_vol * VOL_MULT:
                wick = (c["high"] - max(c["close"], c["open"])) / c["range"]
                if wick < WICK_PCT:
                    b_type = "BULLISH" if c["close"] > c["open"] else "BEARISH"
                    prev_close = candles[i - 1]["close"] if i > 0 else c["close"]
                    patterns.append({
                        "triggerDate": c.get("date", ""), "triggerTime": c["time"],
                        "triggerType": b_type, "triggerClose": c["close"],
                        "triggerHigh": c["high"], "triggerLow": c["low"],
                        "bodyVsAvg": round(c["body"] / avg_body, 2),
                        "volVsAvg": round(c["volume"] / avg_vol, 2),
                        "rsi": round(c.get("rsi14", 50)),
                        "consolCount": consol_before, "consolHigh": 0, "consolLow": 0,
                        "consolRangePct": 0, "consolEndTime": c["time"],
                        "status": "SQUEEZE BREAKOUT",
                        "changePct": round((c["close"] - prev_close) / prev_close * 100, 2),
                    })

    return patterns


def compute_ai(p):
    if not p:
        return 0
    score = 30
    bv = p.get("bodyVsAvg", 0)
    vv = p.get("volVsAvg", 0)
    if bv > 2:
        score += min((bv - 2) * 8, 20)
    if vv > 1.5:
        score += min((vv - 1.5) * 10, 15)
    rsi = p.get("rsi", 50)
    if 30 <= rsi <= 70:
        score += 10
    cc = p.get("consolCount", 0)
    if cc >= 2:
        score += 5
    if cc >= 5:
        score += 5
    st = p.get("status", "")
    if "BROKEN" in st or "SQUEEZE" in st:
        score += 10
    if "FRESH" in st:
        score += 5
    cp = abs(p.get("changePct", 0))
    if cp < 2:
        score += 5
    return round(min(score, 98))


def process_stock(symbol):
    candles = get_candles(symbol)
    if not candles or len(candles) < 22:
        return None

    patterns = detect_bcc_patterns(candles, lookback=LOOKBACK[0])
    current = candles[-1] if candles else None
    prev = candles[-2] if len(candles) > 1 else None
    change = ((current["close"] - prev["close"]) / prev["close"] * 100) if current and prev else 0

    pattern_str = "No Pattern"
    ai_score = 0
    if patterns:
        best = max(patterns, key=compute_ai)
        pattern_str = f"{best['triggerType']} {best['status']} ({best['consolCount']}c)" if best["consolCount"] > 0 else f"{best['triggerType']} FRESH TRIGGER"
        ai_score = compute_ai(best)

    avg_vol_20 = 0
    if len(candles) >= 20:
        vols = [p["volume"] for p in candles[-21:-1]]
        avg_vol_20 = sum(vols) / len(vols) if vols else 0

    rsi_val = round(current.get("rsi14", 50)) if current else 50
    trend = "Strong Up" if change > 2 else "Up" if change > 0.5 else "Strong Down" if change < -2 else "Down" if change < -0.5 else "Sideways"
    bv = patterns[-1]["bodyVsAvg"] if patterns else 0
    vv = patterns[-1]["volVsAvg"] if patterns else 0

    sector_map = {"RELIANCE":"Energy","TCS":"IT","HDFCBANK":"Banking","INFY":"IT","ICICIBANK":"Banking","SBIN":"Banking","HINDUNILVR":"FMCG","ITC":"FMCG","BHARTIARTL":"Telecom","KOTAKBANK":"Banking","BAJFINANCE":"Banking","LT":"Infra","WIPRO":"IT","AXISBANK":"Banking","TITAN":"Consumer","ADANIENT":"Energy","MARUTI":"Auto","SUNPHARMA":"Pharma","HCLTECH":"IT","NTPC":"Power","ONGC":"Energy","POWERGRID":"Power","ULTRACEMCO":"Infra","M&M":"Auto","JSWSTEEL":"Metal","TATASTEEL":"Metal","HINDALCO":"Metal","TATAMOTORS":"Auto","DMART":"Consumer","PAYTM":"IT","IRCTC":"Infra","IEX":"Power","PNB":"Banking","YESBANK":"Banking","IDFCFIRSTB":"Banking","BEL":"Infra","HAL":"Infra","IRFC":"Infra","NHPC":"Power","HUDCO":"Infra","HDFCLIFE":"Insurance","SBILIFE":"Insurance","ICICIGI":"Insurance","ICICIPRULI":"Insurance","BAJAJFINSV":"Banking","HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTOR":"Auto","BRITANNIA":"FMCG","NESTLEIND":"FMCG","DABUR":"FMCG","COLPAL":"FMCG","BERGEPAINT":"FMCG","PIDILITIND":"Chemicals","GRASIM":"Infra","SHREECEM":"Infra","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma","LUPIN":"Pharma","APOLLOHOSP":"Healthcare","TRENT":"Consumer","POLYCAB":"Infra","HAVELLS":"Consumer","SIEMENS":"Infra","BANKBARODA":"Banking","CANBK":"Banking","RECLTD":"Power","PFC":"Power","COALINDIA":"Metal","GODREJCP":"FMCG","MARICO":"FMCG","DIXON":"Consumer","PERSISTENT":"IT","LTTS":"IT","MPHASIS":"IT","COFORGE":"IT","ASHOKLEY":"Auto","JUBLFOOD":"Consumer","INDIGO":"Infra","MUTHOOTFIN":"Banking","ADANIGREEN":"Energy","ADANIPORTS":"Infra","GAIL":"Energy","IGL":"Energy","MGL":"Energy","PETRONET":"Energy","BPCL":"Energy","IOC":"Energy","HINDPETRO":"Energy"}
    sector = sector_map.get(symbol, "Other")

    return {
        "ticker": symbol,
        "price": round(current["close"], 2) if current else 0,
        "volume": current["volume"] if current else 0,
        "change": round(change, 2),
        "relVolume": round(current["volume"] / avg_vol_20, 1) if avg_vol_20 > 0 else 0,
        "pattern": pattern_str,
        "aiScore": ai_score,
        "similarity": round(min(bv / 4 * 100, 99)),
        "confidence": round(min(vv / 3 * 100, 98)),
        "rsi": rsi_val,
        "trend": trend,
        "sector": sector,
        "atr": round((current["high"] - current["low"]) / current["close"] * 100, 2) if current else 0,
        "alerted": len(patterns) > 0,
    }


def main():
    csv_dir = r"C:\Users\pc\Downloads\stock hist data\comprehensive_data"
    stock_list = []
    if os.path.exists(csv_dir):
        for fname in os.listdir(csv_dir):
            if fname.endswith("_ONE_DAY.csv"):
                sym = fname.replace("_ONE_DAY.csv", "")
                stock_list.append(sym)

    if not stock_list:
        print(json.dumps({"error": "No stocks found", "stocks": []}))
        sys.exit(1)

    results = []
    total = len(stock_list)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_stock, sym): sym for sym in stock_list}
        done = 0
        success = 0
        patterns_found = 0

        for f in as_completed(futures):
            done += 1
            if done % 25 == 0:
                print(json.dumps({"progress": {"total": total, "current": done, "success": success, "patterns": patterns_found}}), file=sys.stderr)
            result = f.result()
            if result:
                success += 1
                if result.get("pattern") != "No Pattern" and result.get("aiScore", 0) > 30:
                    patterns_found += 1
                    results.append(result)

    print(json.dumps({"progress": {"total": total, "current": total, "success": success, "patterns": patterns_found}}), file=sys.stderr)
    results.sort(key=lambda r: r.get("aiScore", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    print(json.dumps({"stocks": results, "total": len(results), "totalScanned": total}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=10)
    args = parser.parse_args()
    LOOKBACK[0] = args.lookback
    main()
