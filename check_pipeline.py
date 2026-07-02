import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")
new_30 = ["ABSLAMC","AMBER","ANGELONE","ASTRAMICRO","AURIONPRO","AWL","BSOFT",
    "GPTINFRA","ICICIAMC","IREDA","KALPATARU","KPITTECH","LATENTVIEW","MAZDA",
    "MUTHOOTCAP","NAM-INDIA","NUVOCO","NYKAA","OLAELEC","PATANJALI","PAYTM",
    "POLICYBZR","RVNL","SAFARI","SONACOMS","SWIGGY","TCI","TCIEXP","TRIVENI","UTIAMC"]

for tbl in ["feature_store", "market_structure"]:
    existing = set(r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM {tbl}").fetchall())
    missing = [s for s in new_30 if s not in existing]
    print(f"{tbl}: {len(existing)} total, {len(new_30)-len(missing)}/30 new done, {len(missing)} still missing")
    if missing:
        print("  Still missing:", ", ".join(missing[:10]))
        if len(missing) > 10: print(f"  ... and {len(missing)-10} more")
con.close()
