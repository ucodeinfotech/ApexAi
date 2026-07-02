import json, os
with open("nifty500_batch.json") as f:
    batch = json.load(f)
existing_1min = [s for s in batch if os.path.exists(f"comprehensive_data/{s}_ONE_MINUTE.csv")]
print("New batch stocks:", len(batch))
print("Already have 1-min:", len(existing_1min))
if existing_1min:
    print("First few:", existing_1min[:10])
try:
    d = json.load(open("dl_nifty500_progress.json"))
    print("Fast (1-day only) progress:", len(d), "done")
except:
    print("No fast progress file")
try:
    d = json.load(open("download_nifty500_progress.json"))
    print("Full batch progress:", len(d.get("done",[])), "done")
except:
    print("No batch progress file")
oneday = [f for f in os.listdir("comprehensive_data") if f.endswith("_ONE_DAY.csv")]
print("Total ONE_DAY in comprehensive_data:", len(oneday))
