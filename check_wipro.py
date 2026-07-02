import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
for item in resp.json():
    if item["exch_seg"] == "NSE" and "WIPRO" in item["name"]:
        print(f"{item['name']} -> token {item['token']}")
