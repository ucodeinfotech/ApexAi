import requests, json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
data = resp.json()

# Search for TATAMOTORS and related
print("=== Searching for TATA MOTORS ===")
for item in data:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        name = item["name"]
        if "TATA" in name and "MOTOR" in name:
            print(f"  {name:45s} -> token: {item['token']:8s}  symbol: {item['symbol']}")

print("\n=== Fix ALT_NAMES ===")
# Fix the alt names based on actual Scrip Master names
fixed_alt = {
    "BAJAJ-AUTO": "BAJAJ-AUTO",    # not BAJAJ_AUTO
    "M&M": "M&M",                   # not M_M
    "INDIGO": "INDIGO",             # not INTERGLOBE_AVIATION
    "JIOFIN": "JIOFIN",             # not JIO_FINANCIAL_SERVICES
    "MAXHEALTH": "MAXHEALTH",       # not MAX_HEALTHCARE_INSTITUTE
    "SHRIRAMFIN": "SHRIRAMFIN",     # not SHRIRAM_FINANCE
}

token_map = {}
for item in data:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

for sym, name in fixed_alt.items():
    tok = token_map.get(name)
    print(f"  {sym:15s} -> {name:25s} -> token: {tok}")

# Check TATASTEEL and TATACONSUM too
print("\n=== Verify existing tokens ===")
verify = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "TATASTEEL", "TATACONSUM"]
for name in verify:
    tok = token_map.get(name)
    print(f"  {name:20s} -> token: {tok}")
