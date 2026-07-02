"""Import NEW symbols from comprehensive_data into raw_market."""
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

# Existing symbols in raw_market (don't reimport these)
existing_syms = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market").fetchall())
print(f'Existing raw_market symbols: {len(existing_syms)}')

# All symbols available in comprehensive_data
all_csv = [f for f in os.listdir(str(CSV_DIR)) if f.endswith('.csv') and not f.startswith('_')]
comp_syms = set()
for f in all_csv:
    m = re.match(r'(.+)_(ONE_DAY|ONE_HOUR|FIFTEEN_MINUTE|FIVE_MINUTE|ONE_MINUTE)\.csv$', f)
    if m: comp_syms.add(m.group(1))
print(f'Comprehensive data symbols: {len(comp_syms)}')

# NEW symbols to import
new_syms = comp_syms - existing_syms
print(f'New symbols to import: {len(new_syms)}')
print(f'Sample: {sorted(list(new_syms))[:10]}')

# Collect all files for NEW symbols
files_to_import = []
for f in all_csv:
    m = re.match(r'(.+)_(ONE_DAY|ONE_HOUR|FIFTEEN_MINUTE|FIVE_MINUTE|ONE_MINUTE)\.csv$', f)
    if m and m.group(1) in new_syms:
        files_to_import.append((f, m.group(1), TF_MAP[m.group(2)]))

print(f'Files to import: {len(files_to_import)}')
print(f'Per timeframe:')
for tf in ['1day', '60min', '15min', '5min', '1min']:
    cnt = sum(1 for _, _, t in files_to_import if t == tf)
    print(f'  {tf}: {cnt} files')

imported = 0; total_rows = 0; errors = 0; t0 = time.time()
for fname, sym, tf in sorted(files_to_import):
    try:
        df = pd.read_csv(CSV_DIR / fname)
        if len(df) == 0: continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.columns = [c.lower() for c in df.columns]
        df['symbol'] = sym; df['timeframe'] = tf; df['source'] = 'comprehensive'
        # Match DuckDB table column order: datetime,open,high,low,close,volume,symbol,timeframe,source
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'source']]
        
        con.register('df_sub', df)
        con.execute("INSERT INTO raw_market SELECT datetime,open,high,low,close,volume,symbol,timeframe,source FROM df_sub")
        con.unregister('df_sub')
        imported += 1; total_rows += len(df)
        if imported % 200 == 0:
            rate = total_rows / (time.time() - t0) if time.time() > t0 else 0
            print(f'  {imported}/{len(files_to_import)} files ({time.time()-t0:.0f}s, {total_rows:,} rows, {rate:,.0f} rows/s)')
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f'  ERROR {fname}: {e}')

t1 = time.time()
print(f'\nDone: {imported} files, {total_rows:,} rows, {errors} errors in {t1-t0:.0f}s')
print(f'\n=== Post-import summary ===')
syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market").fetchone()[0]
rows = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()[0]
print(f'Total symbols: {syms}, Total rows: {rows:,}')
for r in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f'  {r[0]}: {r[1]:,} rows, {r[2]} symbols')
for r in con.execute("SELECT timeframe, MIN(datetime)::DATE, MAX(datetime)::DATE FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f'  {r[0]}: {r[1]} to {r[2]}')
con.close()
