import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
# Find all TATA-related NSE stocks
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ"):
        if item["name"].startswith("TATA") or "TATA" in item["name"]:
            print(f"name='{item['name']}' symbol='{item['symbol']}' token={item['token']}")

print("\n--- Searching for specific missing stocks ---")
for s in ["TATAMOTORS", "M&M", "INDIGO", "JIOFIN", "MAXHEALTH", "SHRIRAMFIN", "ETERNAL"]:
    found = False
    for item in scrip_master:
        if item["exch_seg"] == "NSE" and item["name"] == s:
            print(f"  {s}: token={item['token']} symbol='{item['symbol']}'")
            found = True
            break
    if not found:
        print(f"  {s}: NOT FOUND")
