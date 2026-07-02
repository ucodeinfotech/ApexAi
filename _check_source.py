import duckdb
con = duckdb.connect(r'C:\Users\pc\Downloads\stock hist data\warehouse\market_data.duckdb')
# Check source distribution for 60min
r = con.execute("SELECT source, COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='60min' GROUP BY source").fetchall()
print('60min source groups:', r)
# Check if RELIANCE was in original 60min data
r2 = con.execute("SELECT COUNT(*) FROM raw_market WHERE timeframe='60min' AND symbol='RELIANCE'").fetchone()[0]
print(f'RELIANCE 60min rows: {r2}')
# Sample 10 symbols with 60min data
r3 = con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='60min' ORDER BY symbol LIMIT 10").fetchall()
print('Sample 60min symbols:', [x[0] for x in r3])
con.close()
