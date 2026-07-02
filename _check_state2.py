import duckdb
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')
# Current state
syms = con.execute("SELECT COUNT(DISTINCT symbol), COUNT(*) FROM raw_market").fetchone()
print(f'raw_market: {syms[0]} symbols, {syms[1]:,} rows')
# Per timeframe
for r in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f'  {r[0]}: {r[1]:,} rows, {r[2]} symbols')
# Date range per timeframe
for r in con.execute("SELECT timeframe, MIN(datetime)::DATE, MAX(datetime)::DATE FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f'  {r[0]}: {r[1]} to {r[2]}')
con.close()
