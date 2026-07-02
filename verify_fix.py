import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")
print(f"High less than low: {con.execute('SELECT COUNT(*) FROM raw_market WHERE high < low').fetchone()[0]}")
print(f"OOC range: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open < low OR open > high OR close < low OR close > high').fetchone()[0]}")
print(f"Negative: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open < 0 OR high < 0 OR low < 0 OR close < 0').fetchone()[0]}")
print(f"NULLs: {con.execute('SELECT COUNT(*) FROM raw_market WHERE open IS NULL').fetchone()[0]}")
print(f"Total: {con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]:,}")
cnt = con.execute('SELECT COUNT(*) FROM raw_market').fetchone()[0]
dist = con.execute('SELECT COUNT(*) FROM (SELECT DISTINCT symbol, timeframe, datetime FROM raw_market)').fetchone()[0]
print(f"Duplicates: {cnt - dist}")
con.close()
