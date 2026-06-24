import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
# Search everything
for item in scrip_master:
    if "MOTORS" in item["symbol"] or "MOTORS" in item["name"]:
        print(f"symbol='{item['symbol']}' name='{item['name']}' token={item['token']} ex={item['exch_seg']}")
