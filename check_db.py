import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")
tables = con.execute("""
    SELECT table_name, 
           (SELECT COUNT(*) FROM information_schema.columns WHERE table_name=t.table_name AND table_schema='main') as cols,
           (SELECT estimated_size FROM duckdb_tables() WHERE table_name=t.table_name) as est_size
    FROM information_schema.tables t 
    WHERE t.table_schema='main' 
    ORDER BY t.table_name
""").df()
print(tables.to_string())

# Check a sample of each table
for tbl in tables["table_name"]:
    row = con.execute(f"SELECT COUNT(*) as cnt FROM \"{tbl}\"").fetchone()
    print(f"  {tbl}: {row[0]:,} rows")
    # show last date
    try:
        last = con.execute(f"SELECT MAX(datetime) FROM \"{tbl}\" WHERE datetime IS NOT NULL").fetchone()
        if last[0]: print(f"    last: {last[0]}")
    except: pass
