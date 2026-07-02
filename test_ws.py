"""Test Angel One WebSocket V2 - fetches scrip master, builds token map, ready for WS"""
import requests, json, struct, os, time, sys

# ─── 1. Fetch scrip master (token mapping) ───
URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE = "OpenAPIScripMaster.json"

if os.path.exists(CACHE):
    print("Loading cached scrip master...")
    with open(CACHE, "r") as f:
        scrip = json.load(f)
else:
    print("Downloading scrip master (~25MB)...")
    resp = requests.get(URL, timeout=120)
    scrip = resp.json()
    with open(CACHE, "w") as f:
        json.dump(scrip, f)

print(f"Total tokens: {len(scrip)}")

# ─── 2. Build lookup by symbol ───
token_map = {}
for item in scrip:
    symbol = item.get("symbol", "").upper()
    exch = item.get("exch_seg", "")
    token = item.get("token", "")
    if symbol and exch and token:
        key = f"{exch}:{symbol}"
        token_map[key] = {"token": token, "exch": exch, "symbol": symbol, "name": item.get("name", "")}

# ─── 3. Find some test tokens ───
print("\n--- Sample NSE CM tokens ---")
test_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "TCS", "INFY"]
for sym in test_symbols:
    key = f"NSE:{sym}"
    if key in token_map:
        t = token_map[key]
        print(f"  {sym:12s} -> token={t['token']:>8s}  name={t['name']}")
    else:
        # Try with different exchange prefix
        for exch in ["NSE", "BSE", "NFO", "CDS"]:
            k = f"{exch}:{sym}"
            if k in token_map:
                t = token_map[k]
                print(f"  {sym:12s} ({exch}) -> token={t['token']:>8s}  name={t['name']}")
                break
        else:
            print(f"  {sym:12s} -> NOT FOUND in scrip master")

# ─── 4. Show WebSocket subscription payload ───
print("\n--- WebSocket Subscription Ready ---")
print("\nTo subscribe, you'd run:")
print("""
from SmartApi import smartWebSocketV2 as ws

socket = ws.SmartWebSocketV2(
    auth_token="YOUR_JWT_TOKEN",
    api_key="YOUR_API_KEY",
    client_code="YOUR_CLIENT_CODE",
    feed_token="YOUR_FEED_TOKEN"
)

def on_data(ws, data):
    ltp = data.get("last_traded_price", 0) / 100
    token = data.get("token", "")
    print(f"{token}: {ltp}")

socket.on_data = on_data
socket.on_open = lambda ws: print("Connected!")
socket.on_error = lambda ws, e: print(f"Error: {e}")
socket.on_close = lambda ws: print("Closed")

socket.connect()

# After connect, subscribe:
socket.subscribe(
    correlation_id="test123456",
    mode=2,  # QUOTE
    token_list=[{"exchangeType": 1, "tokens": ["3045", "26009"]}]
)
""")

# ─── 5. Show exchange type mapping ───
print("\n--- Exchange Type Mapping ---")
print("  1 = NSE CM (cash)")
print("  2 = NSE FO (F&O)")
print("  3 = BSE CM")
print("  4 = BSE FO")
print("  5 = MCX FO")

# ─── 6. If we had credentials, test WebSocket ───
print("\n--- Ready. Provide credentials to connect live. ---")
