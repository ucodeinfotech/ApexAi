"""Import comprehensive_data CSVs into warehouse raw_market."""
import duckdb, pandas as pd, os, time, re
from pathlib import Path

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
CSV_DIR = BASE / 'comprehensive_data'

TF_MAP = {
    'ONE_DAY': '1day', 'ONE_HOUR': '60min',
    'FIFTEEN_MINUTE': '15min', 'FIVE_MINUTE': '5min', 'ONE_MINUTE': '1min'
}

con = duckdb.connect(str(DB))

# Get existing raw_market symbols
existing = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market").fetchall())
print(f'Existing raw_market symbols: {len(existing)}')

# Get all CSV files
files = [f for f in os.listdir(str(CSV_DIR)) if f.endswith('.csv') and not f.startswith('_')]
print(f'Found {len(files)} CSV files to import')

# Parse filename: SYMBOL_TIMEFRAME.csv
imported = 0; skipped = 0; errors = 0; total_rows = 0
t0 = time.time()

for fname in sorted(files):
    # Parse symbol and timeframe
    m = re.match(r'(.+)_(ONE_DAY|ONE_HOUR|FIFTEEN_MINUTE|FIVE_MINUTE|ONE_MINUTE)\.csv$', fname)
    if not m:
        # Could be _part/ files with different naming
        continue
    sym = m.group(1)
    tf_raw = m.group(2)
    tf = TF_MAP[tf_raw]
    
    fp = CSV_DIR / fname
    try:
        df = pd.read_csv(fp)
        if len(df) == 0: continue
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['symbol'] = sym
        df['timeframe'] = tf
        df = df[['symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Insert only rows that don't already exist (keep existing data)
        con.register('df_sub', df)
        already = con.execute("SELECT datetime FROM raw_market WHERE symbol=? AND timeframe=?", [sym, tf]).fetchdf()
        if len(already) > 0:
            existing_dts = set(pd.to_datetime(already['datetime']))
            df_new = df[~df['datetime'].isin(existing_dts)]
            if len(df_new) > 0:
                con.register('df_new', df_new)
                con.execute("INSERT INTO raw_market (symbol,timeframe,datetime,open,high,low,close,volume) SELECT symbol,timeframe,datetime,open,high,low,close,volume FROM df_new")
                con.unregister('df_new')
        else:
            con.execute("INSERT INTO raw_market (symbol,timeframe,datetime,open,high,low,close,volume) SELECT symbol,timeframe,datetime,open,high,low,close,volume FROM df_sub")
        con.unregister('df_sub')
        imported += 1
        total_rows += len(df)
        
        if imported % 50 == 0:
            print(f'  {imported}/{len(files)} files ({time.time()-t0:.0f}s, {total_rows:,} rows)')
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f'  ERROR {fname}: {e}')

t1 = time.time()
print(f'\nImport complete: {imported} files, {total_rows:,} rows, {errors} errors in {t1-t0:.0f}s')

# Summary
print('\n=== Post-import summary ===')
syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market").fetchone()[0]
rows = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()[0]
print(f'Total symbols: {syms}')
print(f'Total rows: {rows:,}')
print('\nRows per timeframe:')
for r in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f'  {r[0]}: {r[1]:,} rows, {r[2]} symbols')

con.close()
print('Done')
