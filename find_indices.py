import requests
resp = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
scrip_master = resp.json()

print("=== INDICES (instrumenttype=AMXIDX or INDEX) ===")
for item in scrip_master:
    if item.get("instrumenttype") in ("AMXIDX", "INDEX") or "INDX" in item.get("instrumenttype",""):
        if "NIFTY" in item["name"].upper() or "SENSEX" in item["name"].upper() or "BANK" in item["name"].upper():
            print(f"  exch={item['exch_seg']:5s} token={item['token']:>10s} name='{item['name']:15s}' symbol='{item['symbol']}' instr={item['instrumenttype']}")

print("\n=== All tokens starting with 999 ===")
for item in scrip_master:
    if item["token"].startswith("999"):
        print(f"  exch={item['exch_seg']:5s} token={item['token']:>10s} name='{item['name']:20s}' symbol='{item['symbol']:20s}' instr={item['instrumenttype']}")
