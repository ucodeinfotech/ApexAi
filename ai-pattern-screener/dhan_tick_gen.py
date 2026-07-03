"""Generates live tick data from Dhan daily cache for the LiveTicker component.
Reads cached daily candles, computes gainers/losers from latest vs previous day.
Runs continuously at a configurable interval."""

import json
import os
import time
import sys
import glob

CACHE_DIR = r"C:\Users\pc\Downloads\stock hist data\ai-pattern-screener\daily_cache"
TICKS_FILE = r"C:\Users\pc\Downloads\stock hist data\ai-pattern-screener\ticks.jsonl"
POLL_INTERVAL = 30  # seconds

def compute_gainers_losers():
    ticks = []
    for fpath in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            candles = data.get("candles", [])
            if len(candles) < 2:
                continue
            last = candles[-1]
            prev = candles[-2]
            ltp = last["close"]
            prev_close = prev["close"]
            change_pct = round((ltp - prev_close) / prev_close * 100, 2) if prev_close else 0
            volume = last.get("volume", 0)
            ticks.append({
                "token": os.path.basename(fpath).replace(".json", ""),
                "symbol": data.get("symbol", os.path.basename(fpath).replace(".json", "")),
                "ltp": ltp,
                "change": round(ltp - prev_close, 2),
                "changePct": change_pct,
                "volume": volume,
                "timestamp": int(time.time()),
            })
        except:
            continue
    return ticks

def write_ticks(ticks):
    os.makedirs(os.path.dirname(TICKS_FILE), exist_ok=True)
    with open(TICKS_FILE, "w") as f:
        for t in ticks[-200:]:
            f.write(json.dumps(t) + "\n")

def get_summary(ticks):
    advancers = sum(1 for t in ticks if t["changePct"] > 0)
    decliners = sum(1 for t in ticks if t["changePct"] < 0)
    unchanged = sum(1 for t in ticks if t["changePct"] == 0)
    total_volume = sum(t.get("volume", 0) for t in ticks)
    return {"advancers": advancers, "decliners": decliners, "unchanged": unchanged, "totalVolume": total_volume}

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else POLL_INTERVAL
    while True:
        ticks = compute_gainers_losers()
        write_ticks(ticks)
        summary = get_summary(ticks)
        status = {
            "event": "tick",
            "active": True,
            "uniqueStocks": len(ticks),
            "totalVolume": summary["totalVolume"],
            "advancers": summary["advancers"],
            "decliners": summary["decliners"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(json.dumps(status), file=sys.stderr)
        time.sleep(interval)
