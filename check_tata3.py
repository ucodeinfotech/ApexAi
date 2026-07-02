import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
for item in scrip_master:
    if item["exch_seg"] == "NSE":
        if "TATAMOTORS" in item["symbol"] or "TATAMOTORS" in item["name"]:
            print(f"symbol='{item['symbol']}' name='{item['name']}' -> token {item['token']}")
        if item["symbol"] == "TATAMOTORS-EQ":
            print(f"EXACT: symbol='{item['symbol']}' name='{item['name']}' -> token {item['token']}")
        if item["name"] == "TATAMOTORS":
            print(f"BY NAME: symbol='{item['symbol']}' name='{item['name']}' -> token {item['token']}")
