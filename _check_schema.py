import duckdb
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')
for r in con.execute("PRAGMA table_info('raw_market')").fetchall():
    print(r)
con.close()
