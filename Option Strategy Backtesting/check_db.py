"""Check DuckDB options_data_clean contents."""
import duckdb, pandas as pd
DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
con = duckdb.connect(DB_PATH)

tables = con.execute("SHOW TABLES").fetchdf()
print("=== TABLES ===")
print(tables.to_string(index=False))

for tbl in tables["name"]:
    cnt = con.execute(f"SELECT count(*) FROM \"{tbl}\"").fetchone()[0]
    print(f"\n=== {tbl} ===")
    print(f"Rows: {cnt:,}")
    
    cols = con.execute(f"DESCRIBE \"{tbl}\"").fetchdf()
    print("\nColumns:")
    print(cols.to_string(index=False))
    
    if cnt > 0:
        print("\nDate range:")
        dt_cols = [c for c in cols["column_name"] if "time" in c.lower() or "date" in c.lower()]
        for c in dt_cols:
            r = con.execute(f"SELECT min({c}), max({c}) FROM \"{tbl}\"").fetchone()
            print(f"  {c}: {r[0]}  ->  {r[1]}")
        
        print("\nNULL counts:")
        for c in cols["column_name"]:
            n = con.execute(f"SELECT count(*) FROM \"{tbl}\" WHERE {c} IS NULL").fetchone()[0]
            if n > 0:
                print(f"  {c}: {n:,} NULL")
        
        print("\nSample distinct values:")
        for c in ["option_type", "expiry_code", "expiry_flag", "atm_distance"]:
            if c in cols["column_name"].values:
                vals = con.execute(f"SELECT {c}, count(*) FROM \"{tbl}\" GROUP BY {c} ORDER BY 2 DESC").fetchdf()
                print(f"  {c}:")
                print(f"    {vals.to_string(index=False, header=False)}")

con.close()
