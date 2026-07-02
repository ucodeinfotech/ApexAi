import duckdb
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')
# Check for index symbols
for sym in ['^NSEI', 'NIFTY50', '^BSESN', 'SENSEX', 'BANKNIFTY']:
    r = con.execute("SELECT symbol, timeframe, COUNT(*) FROM raw_market WHERE symbol=? GROUP BY symbol, timeframe ORDER BY timeframe", [sym]).fetchall()
    if r:
        print(f'{sym}: {len(r)} timeframes, {sum(x[2] for x in r):,} rows')
        for tf, _, cnt in r:
            print(f'  {tf}: {cnt:,}')
    else:
        print(f'{sym}: NOT FOUND')
con.close()
