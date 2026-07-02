import duckdb
con = duckdb.connect(r'C:\Users\pc\Downloads\stock hist data\warehouse\market_data.duckdb')
r = con.execute("SELECT timeframe, COUNT(*) FROM raw_market WHERE symbol='RELIANCE' GROUP BY timeframe").fetchall()
print('RELIANCE timeframes:', r)

# Check what happened to original 60min data
orig_60 = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='60min' AND source IS NULL").fetchone()[0]
both_60 = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='60min'").fetchone()[0]
print(f'60min symbols: {both_60} total, {orig_60} without source tag')
con.close()
