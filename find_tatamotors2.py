import requests, json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
data = resp.json()

# Search TATAMOTORS across ALL fields
print("=== Searching 'TATAMOTORS' across all fields ===")
for item in data:
    for key, val in item.items():
        if isinstance(val, str) and "TATAMOTORS" in val.upper():
            print(f"  {key}={val}  exch_seg={item.get('exch_seg','')}  token={item.get('token','')}")
            break

# Search for symbol ending with TATAMOTORS-EQ
print("\n=== EQ symbols with TATAMOTORS ===")
for item in data:
    symbol = item.get("symbol", "")
    name = item.get("name", "")
    if "TATAMOTOR" in symbol.upper() and item.get("exch_seg") == "NSE":
        print(f"  symbol={symbol:30s} name={name:30s} token={item['token']} exch_seg={item['exch_seg']}")

# Check if it exists at all 
print("\n=== Any token with 'TATAMOTORS' ===")
for item in data:
    if "TATAMOTORS" in json.dumps(item).upper():
        print(f"  token={item['token']:8s} symbol={item['symbol']:30s} name={item.get('name',''):30s} exch_seg={item['exch_seg']:8s}")
