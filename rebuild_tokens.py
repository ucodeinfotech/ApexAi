"""Rebuild NSE EQ token map from Scrip Master"""
import json
import urllib.request
import time

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
print("Downloading Scrip Master...", flush=True)

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=120) as resp:
    scrips = json.loads(resp.read().decode())

print(f"Total entries: {len(scrips)}", flush=True)

# Build NSE EQ token map
nse_tokens = {}
for s in scrips:
    exch = s.get("exch_seg", "")
    sym = s.get("symbol", "")
    tok = s.get("token", "")
    name = s.get("name", "")
    if exch == "NSE" and sym.endswith("-EQ"):
        # Strip -EQ for clean lookup
        clean = sym[:-3]
        nse_tokens[clean] = tok

with open("nse_tokens.json", "w") as f:
    json.dump(nse_tokens, f, indent=1)
print(f"Saved {len(nse_tokens)} NSE EQ tokens to nse_tokens.json", flush=True)

# Verify some known stocks
for s in ["RELIANCE","TCS","INFY","HDFCBANK","ABB","SBIN","M&M","WIPRO","ADANIENT"]:
    tok = nse_tokens.get(s, "NOT FOUND")
    print(f"  {s}: {tok}", flush=True)
