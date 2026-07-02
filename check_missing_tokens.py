"""Check ZOMATO and VODAFONEIDEA in Scrip Master raw data"""
import json, urllib.request

url = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())

print("Searching for ZOMATO, VODAFONEIDEA...")
for s in data:
    sym = s.get('symbol', '')
    if 'ZOMATO' in sym:
        print(f"ZOMATO: symbol={sym} exch={s.get('exch_seg')} token={s.get('token')} name={s.get('name')}")
    if 'VODAFONE' in sym or sym == 'IDEA-EQ':
        if sym.endswith('-EQ') or 'VODAFONE' in sym:
            print(f"VODAFONE/IDEA: symbol={sym} exch={s.get('exch_seg')} token={s.get('token')} name={s.get('name')}")
    # Also check BSE segment
    if 'ZOMATO' in sym and s.get('exch_seg') == 'BSE':
        print(f"ZOMATO BSE: symbol={sym} token={s.get('token')}")

print("\nAll ZOMATO occurrences:")
for s in data:
    if 'ZOMATO' in s.get('symbol', ''):
        print(f"  {s.get('symbol')} | {s.get('exch_seg')} | {s.get('token')} | {s.get('name')}")
