import duckdb, pandas as pd, numpy as np
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')

# Check each table for duplicate keys used in merge
# 1. MTF: check if 60min feature_store has duplicate symbol-date pairs
m = con.execute("SELECT symbol,datetime,high,low,close,rsi_14,bb_width,macd_hist FROM feature_store WHERE timeframe='60min' ORDER BY datetime").fetchdf()
md = pd.to_datetime(m['datetime'])
m['datetime'] = (md.dt.tz_localize(None).astype('datetime64[us]') if md.dt.tz is not None else md.astype('datetime64[us]'))
m['date'] = pd.to_datetime(m['datetime']).dt.normalize()
print(f"60min raw rows: {len(m):,}")
dup = m.groupby(['symbol','date']).size()
print(f"Unique symbol-date pairs: {len(dup):,}")
print(f"Duplicate pairs (count>1): {(dup>1).sum()}")

# 2. Delivery data
dv = con.execute("SELECT symbol,date FROM delivery_data ORDER BY symbol,date").fetchdf()
dv['date'] = pd.to_datetime(dv['date']).astype('datetime64[us]')
print(f"\nDelivery rows: {len(dv):,}")
dup_dv = dv.groupby(['symbol','date']).size()
print(f"Unique symbol-date pairs: {len(dup_dv):,}")
print(f"Duplicate pairs: {(dup_dv>1).sum()}")

# 3. RS (market_structure)
ms = con.execute("SELECT symbol,datetime FROM market_structure WHERE timeframe='1day'").fetchdf()
msd = pd.to_datetime(ms['datetime'])
ms['datetime'] = (msd.dt.tz_localize(None).astype('datetime64[us]') if msd.dt.tz is not None else msd.astype('datetime64[us]'))
print(f"\nMarket structure rows: {len(ms):,}")
dup_ms = ms.groupby(['symbol','datetime']).size()
print(f"Unique symbol-datetime pairs: {len(dup_ms):,}")
print(f"Duplicate pairs: {(dup_ms>1).sum()}")

# 4. Feature_store 1day (base): check for duplicates
fs = con.execute("SELECT symbol,datetime FROM feature_store WHERE timeframe='1day'").fetchdf()
fsd = pd.to_datetime(fs['datetime'])
fs['datetime'] = (fsd.dt.tz_localize(None).astype('datetime64[us]') if fsd.dt.tz is not None else fsd.astype('datetime64[us]'))
print(f"\nFeature store 1day rows: {len(fs):,}")
dup_fs = fs.groupby(['symbol','datetime']).size()
print(f"Unique pairs: {len(dup_fs):,}")
print(f"Duplicate pairs: {(dup_fs>1).sum()}")

con.close()
