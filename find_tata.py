import requests, json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
data = resp.json()

# Check raw data for TATA
print("=== All NSE entries with 'TATA' in name ===")
for item in data:
    if item.get("exch_seg") == "NSE" and "TATA" in item.get("name", "").upper():
        print(f"  name={item['name']:40s} symbol={item['symbol']:25s} token={item['token']:8s}")

# Also search NFO for TATAMOTORS (futures/options)
print("\n=== NFO entries with TATAMOTORS ===")
for item in data:
    if item.get("exch_seg") == "NFO" and "TATAMOTOR" in item.get("symbol", "").upper():
        print(f"  symbol={item['symbol']:30s} token={item['token']:8s} name={item.get('name',''):25s}")
        break  # just first one

# Check CDS and other segments
print("\n=== All segments with TATAMOTORS ===")
for item in data:
    if "TATAMOTOR" in json.dumps(item).upper():
        print(f"  token={item['token']:8s} exch_seg={item['exch_seg']:8s} symbol={item['symbol']:30s}")
print("--- END ---")
