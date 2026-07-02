import duckdb, os

con = duckdb.connect("warehouse/market_data.duckdb")
db_syms = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market").fetchall())
csv_syms = set(f.replace("_ONE_MINUTE.csv","") for f in os.listdir("comprehensive_data") if f.endswith("_ONE_MINUTE.csv"))

missing_from_db = csv_syms - db_syms
missing_from_csv = db_syms - csv_syms

print(f"In DB: {len(db_syms):,}")
print(f"In CSVs: {len(csv_syms):,}")
print(f"In CSVs but NOT in DB: {len(missing_from_db)}")
for s in sorted(missing_from_db):
    print(f"  {s}")

print(f"\nIn DB but NOT in CSVs: {len(missing_from_csv)}")
for s in sorted(missing_from_csv):
    print(f"  {s}")
