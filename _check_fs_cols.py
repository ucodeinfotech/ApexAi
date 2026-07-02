import duckdb
con = duckdb.connect(r'C:\Users\pc\Downloads\stock hist data\warehouse\market_data.duckdb')
cols = con.execute("SELECT COUNT(*) FROM pragma_table_info('feature_store')").fetchone()[0]
print(f'feature_store columns: {cols}')
# Sample column names
cols2 = [r[1] for r in con.execute("SELECT * FROM pragma_table_info('feature_store') LIMIT 20").fetchall()]
print('First 20 cols:', cols2)
# Check for BB_pct_b
has_bb = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='feature_store' AND column_name='bb_pct_b'").fetchone()
print(f'bb_pct_b exists: {has_bb is not None}')
con.close()
