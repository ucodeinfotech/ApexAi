import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
# Check NFO segment for TATAMOTORS
for item in scrip_master:
    if item["exch_seg"] in ("NFO", "NSE"):
        if "TATAMOTOR" in item["symbol"] or "TATAMOTOR" in item["name"] or item.get("expiry") == "":
            pass  # too many
    if item["exch_seg"] == "NFO" and "TATAMOTOR" in item["symbol"]:
        print(f"NFO: symbol='{item['symbol']}' name='{item['name']}' token={item['token']}")

# Also get unique names starting with TATA in NSE
seen = set()
for item in scrip_master:
    if item["exch_seg"] == "NSE" and item["name"].startswith("TATA") and item["name"] not in seen:
        seen.add(item["name"])
        print(f"NSE: name='{item['name']}' symbol='{item['symbol']}' token={item['token']}")
