import duckdb, time, sys
sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
from src.features.indicators import compute_all_features

con = duckdb.connect(r'C:\Users\pc\Downloads\stock hist data\warehouse\market_data.duckdb')

t0 = time.time()
df = con.execute("SELECT datetime,open,high,low,close,volume FROM raw_market WHERE symbol='RELIANCE' AND timeframe='1day' ORDER BY datetime").fetchdf()
print(f'RELIANCE 1day: {len(df)} rows')
t1 = time.time()
feat = compute_all_features(df)
t2 = time.time()
print(f'  Features: {t2-t1:.2f}s ({len(feat.columns)} cols)')

df60 = con.execute("SELECT datetime,open,high,low,close,volume FROM raw_market WHERE symbol='RELIANCE' AND timeframe='60min' ORDER BY datetime").fetchdf()
print(f'RELIANCE 60min: {len(df60)} rows')
t3 = time.time()
feat60 = compute_all_features(df60)
t4 = time.time()
print(f'  Features: {t4-t3:.2f}s ({len(feat60.columns)} cols)')

df5 = con.execute("SELECT datetime,open,high,low,close,volume FROM raw_market WHERE symbol='RELIANCE' AND timeframe='5min' ORDER BY datetime").fetchdf()
print(f'RELIANCE 5min: {len(df5)} rows')
t5 = time.time()
feat5 = compute_all_features(df5)
t6 = time.time()
print(f'  Features: {t6-t5:.2f}s ({len(feat5.columns)} cols)')

# Check how many symbols have 1day data
syms = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='1day'").fetchone()[0]
print(f'\nSymbols with 1day: {syms}')
est = (t2-t1) * syms
print(f'Estimated 1day rebuild: {est:.0f}s ({est/60:.0f}min)')

# Check symbols with 60min
syms60 = con.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market WHERE timeframe='60min'").fetchone()[0]
print(f'Symbols with 60min: {syms60}')
est60 = (t4-t3) * syms60
print(f'Estimated 60min rebuild: {est60:.0f}s ({est60/60:.0f}min)')
con.close()
