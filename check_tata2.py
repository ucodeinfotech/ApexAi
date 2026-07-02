import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()
for item in scrip_master:
    if item["exch_seg"] == "NSE" and "TATAM" in item["name"]:
        print(f"'{item['name']}' -> token {item['token']}")
    if item["exch_seg"] == "NSE" and "MOTORS" in item["name"]:
        print(f"'{item['name']}' -> token {item['token']}")
    if item["exch_seg"] == "NSE" and item["name"] == "TATAMOTORS":
        print(f"FOUND: '{item['name']}' -> token {item['token']}")
