import requests, json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
data = resp.json()

# Search patterns for the 7 missing stocks
searches = ["BAJAJ", "INTERGLOBE", "JIO", "MAX_HEALTH", "SHRIRAM_FINANCE", "TATA_MOTORS"]
found = set()
for item in data:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        name = item["name"]
        if any(s in name for s in searches):
            if name not in found:
                found.add(name)
                print(f"  {name:45s} -> token: {item['token']}")

# Also check what the original ALT_NAMES resolve to
print("\n--- Checking ALT_NAMES ---")
alt = {
    "BAJAJ-AUTO": "BAJAJ_AUTO",
    "M&M": "M_M",
    "INDIGO": "INTERGLOBE_AVIATION",
    "JIOFIN": "JIO_FINANCIAL_SERVICES",
    "MAXHEALTH": "MAX_HEALTHCARE_INSTITUTE",
    "SHRIRAMFIN": "SHRIRAM_FINANCE",
    "TATAMOTORS": "TATA_MOTORS_PASSENGER_VEHICLES",
}
token_map = {}
for item in data:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        token_map[item["name"]] = item["token"]

for sym, alt_name in alt.items():
    tok = token_map.get(alt_name)
    print(f"  {sym:15s} -> {alt_name:35s} -> token: {tok if tok else 'NOT FOUND'}")

# Try direct name search for each missing
print("\n--- Direct search for each missing symbol ---")
direct = ["BAJAJ-AUTO", "INDIGO", "JIOFIN", "M&M", "MAXHEALTH", "SHRIRAMFIN", "TATAMOTORS"]
for item in data:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        name = item["name"]
        if name in direct or name in [a.replace("_","") for a in alt.values()]:
            print(f"  FOUND: {name:35s} -> token: {item['token']}")
