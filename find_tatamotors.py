import requests, json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
resp = requests.get(url)
data = resp.json()

# Search ALL entries for TATA MOTORS (not just EQ)
print("=== ALL TATA MOTORS entries ===")
for item in data:
    name = item.get("name", "")
    if "TATA" in name.upper() and "MOTOR" in name.upper():
        print(f"  name={name:40s} symbol={item['symbol']:30s} exch_seg={item['exch_seg']:8s} token={item['token']:8s}")

# Search for PASSENGER
print("\n=== PASSENGER entries ===")
for item in data:
    name = item.get("name", "")
    if "PASSENGER" in name.upper():
        print(f"  name={name:40s} symbol={item['symbol']:30s} exch_seg={item['exch_seg']:8s} token={item['token']:8s}")
