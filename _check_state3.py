import duckdb
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')
# Check market_structure
has_ms = con.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='market_structure')").fetchone()[0]
if has_ms:
    syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM market_structure").fetchone()[0]
    print(f'market_structure: {syms} symbols')
else:
    print('market_structure: NOT FOUND')

# Check feature_store
has_fs = con.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='feature_store')").fetchone()[0]
if has_fs:
    syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM feature_store").fetchone()[0]
    rows = con.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
    print(f'feature_store: {syms} symbols, {rows:,} rows')
else:
    print('feature_store: NOT FOUND')
con.close()
