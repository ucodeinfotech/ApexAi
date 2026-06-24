"""Pre-save all stock tokens from Scrip Master to avoid repeated downloads"""
import requests, json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
print("Downloading Scrip Master...", flush=True)
r = requests.get(url, timeout=60)
scrips = r.json()
print(f"Got {len(scrips)} entries", flush=True)

# Build NSE token map
tokens = {}
for s in scrips:
    if s.get("exch_seg") == "NSE" and s.get("symbol"):
        tokens[s["symbol"]] = s["token"]

with open("nse_tokens.json", "w") as f:
    json.dump(tokens, f)
print(f"Saved {len(tokens)} NSE tokens to nse_tokens.json")

# Also check specific stocks we care about
ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]
all_stocks = set()
for d in ALL_DIRS:
    import os
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_MINUTE.csv"):
                all_stocks.add(f.replace("_ONE_MINUTE.csv", ""))

missing_tokens = [s for s in sorted(all_stocks) if s not in tokens]
print(f"\nStocks without NSE tokens: {missing_tokens}")
