import duckdb, pandas as pd, numpy as np
from pathlib import Path
con = duckdb.connect(str(Path(r'C:\Users\pc\Downloads\stock hist data\warehouse\market_data.duckdb')), read_only=True)
df = con.execute("SELECT symbol, datetime::DATE as date, open, high, low, close, volume FROM raw_market WHERE timeframe='1day' ORDER BY symbol, date").fetchdf()
print(f'Shape: {df.shape}')
print(f'Symbols: {df["symbol"].nunique()}')
print(f'High<Low: {(df["high"]<df["low"]).sum()}')
print(f'Close<Low: {(df["close"]<df["low"]).sum()}')
print(f'Close>High: {(df["close"]>df["high"]).sum()}')
print(f'Volume<=0: {(df["volume"]<=0).sum()}')
n_dup = df.duplicated(subset=['symbol','date']).sum()
print(f'Duplicates: {n_dup}')
stale = 0
for sym in df['symbol'].unique()[:10]:
    sd = df[df['symbol']==sym].sort_values('date')
    stale += int((sd['close']==sd['close'].shift()).sum())
print(f'Stale (10 syms): {stale}')
con.close()
