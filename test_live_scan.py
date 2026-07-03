"""Test live scanner with just a few cached stocks."""
import sys, json, os
sys.path.insert(0, r"C:\Users\pc\Downloads\stock hist data\ai-pattern-screener")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\ai-pattern-screener")

from live_scanner import get_candles, detect_bcc_patterns, compute_ai, login, LOOKBACK
from cache_manager import load_tokens, cache_status

# Check cache status
status = cache_status()
print(f"Cache: {status['fresh']} fresh, {status['cached']} cached, {status['total']} total")

# Login
print("Logging into Angel One...")
try:
    smartApi = login()
except Exception as e:
    print(f"Login failed: {e}")
    sys.exit(1)

# Test RELIANCE (should be cached)
print("\n--- RELIANCE (cached) ---")
candles = get_candles(smartApi, load_tokens()["RELIANCE"], "RELIANCE")
if candles:
    print(f"Candles: {len(candles)}, {candles[0]['date']} -> {candles[-1]['date']}")
    patterns = detect_bcc_patterns(candles, lookback=LOOKBACK)
    print(f"Patterns in last {LOOKBACK} candles: {len(patterns)}")
    for p in patterns:
        print(f"  {p['triggerDate']} {p['triggerType']} body={p['bodyVsAvg']}x vol={p['volVsAvg']}x consol={p['consolCount']}d status={p['status']} ai={compute_ai(p)}")
else:
    print("No candles")

# Test HDFCBANK (may not be cached)
print("\n--- HDFCBANK (checking) ---")
tokens = load_tokens()
if "HDFCBANK" in tokens:
    candles2 = get_candles(smartApi, tokens["HDFCBANK"], "HDFCBANK")
    if candles2:
        print(f"Candles: {len(candles2)}")
        patterns2 = detect_bcc_patterns(candles2, lookback=LOOKBACK)
        print(f"Patterns in last {LOOKBACK} candles: {len(patterns2)}")
        for p in patterns2:
            print(f"  {p['triggerDate']} {p['triggerType']} body={p['bodyVsAvg']}x ai={compute_ai(p)}")

# Test TCS (may not be cached)
print("\n--- TCS (checking) ---")
if "TCS" in tokens:
    candles3 = get_candles(smartApi, tokens["TCS"], "TCS")
    if candles3:
        print(f"Candles: {len(candles3)}")
        patterns3 = detect_bcc_patterns(candles3, lookback=LOOKBACK)
        print(f"Patterns in last {LOOKBACK} candles: {len(patterns3)}")
        for p in patterns3:
            print(f"  {p['triggerDate']} {p['triggerType']} body={p['bodyVsAvg']}x ai={compute_ai(p)}")

print("\nDone!")
