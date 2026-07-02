"""Final comprehensive check of all tables for 30 new stocks"""
import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")

new_30 = ["ABSLAMC","AMBER","ANGELONE","ASTRAMICRO","AURIONPRO","AWL","BSOFT",
    "GPTINFRA","ICICIAMC","IREDA","KALPATARU","KPITTECH","LATENTVIEW","MAZDA",
    "MUTHOOTCAP","NAM-INDIA","NUVOCO","NYKAA","OLAELEC","PATANJALI","PAYTM",
    "POLICYBZR","RVNL","SAFARI","SONACOMS","SWIGGY","TCI","TCIEXP","TRIVENI","UTIAMC"]

tables = ["raw_market", "feature_store", "market_structure", "pattern_occurrences"]

print("=" * 60)
print("FINAL COVERAGE REPORT")
print("=" * 60)
print(f"{'Table':<25} {'Total Syms':<12} {'30 New':<10}")
print("-" * 50)

all_ok = True
for tbl in tables:
    total = con.execute(f"SELECT COUNT(DISTINCT symbol) FROM {tbl}").fetchone()[0]
    existing = set(r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM {tbl}").fetchall())
    ok_count = sum(1 for s in new_30 if s in existing)
    missing = [s for s in new_30 if s not in existing]
    status = "OK" if ok_count == 30 else f"MISSING {len(missing)}"
    if status != "OK": all_ok = False
    print(f"{tbl:<25} {total:<12} {ok_count}/30 {status}")
    if missing:
        print(f"  {'':<25} Missing: {', '.join(missing)}")

print("-" * 50)
print(f"\nRAW DATA QUALITY (post-fix):")
print(f"  High<low: {con.execute('SELECT COUNT(*) FROM raw_market WHERE high < low').fetchone()[0]}")
print(f"  OOC range: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open < low OR open > high OR close < low OR close > high').fetchone()[0]}")
print(f"  Negative: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open < 0').fetchone()[0]}")
print(f"  NULLs: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open IS NULL').fetchone()[0]}")
print(f"  Duplicates: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0] - con.execute('SELECT COUNT(*) FROM (SELECT DISTINCT symbol,timeframe,datetime FROM raw_market)').fetchone()[0]}")
print(f"  Total rows: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")

print(f"\nOVERALL: {'ALL CLEAN ✓' if all_ok else 'ISSUES REMAIN'}")
con.close()
