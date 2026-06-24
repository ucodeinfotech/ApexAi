import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
# Search ALL entries for TATAMOTORS (any field)
matches = []
for item in scrip_master:
    for key, val in item.items():
        if val and "TATAMOTORS" in str(val):
            matches.append(item)
            break
print(f"Found {len(matches)} matches for TATAMOTORS:")
for m in matches:
    print(f"  exch={m['exch_seg']} symbol='{m['symbol']}' name='{m['name']}' token={m['token']}")

# Also check if maybe token 3456 exists
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["symbol"].endswith("-EQ") and item["token"] == "3456":
        print(f"token 3456: symbol='{item['symbol']}' name='{item['name']}'")
