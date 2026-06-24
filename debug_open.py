import duckdb, pandas as pd, numpy as np
con = duckdb.connect('warehouse/market_data.duckdb')

# Check if open != close in source data
df = con.execute("SELECT symbol,datetime,open,close FROM feature_store WHERE timeframe='1day' AND symbol='RELIANCE' ORDER BY datetime LIMIT 5").fetchdf()
print('Source data sample:')
print(df.to_string())

diff = (df['open'] != df['close']).sum()
print(f'\nopen != close: {diff}/{len(df)}')

# Load predictions and check columns
rd = pd.read_csv('return_prediction_report_v5/predictions_v5.csv')
print(f'\nPredictions columns: {list(rd.columns)}')
print(f'act sample: {rd["act"].head(3).values}')
print(f'act_open sample: {rd["act_open"].head(3).values}')
print(f'act dtype: {rd["act"].dtype}, act_open dtype: {rd["act_open"].dtype}')
print(f'Same values check: {(rd["act"] == rd["act_open"]).sum()} / {len(rd)}')

# The issue: fwd_open_ret_1d computed as close(T+1)/open(T+1)-1
# but maybe open column was overwritten? Let's trace through the computation
# Load the original data and recreate the computation
df2 = con.execute("SELECT symbol,datetime,open,close FROM feature_store WHERE timeframe='1day' ORDER BY datetime").fetchdf()
ng = df2.groupby('symbol')
fwd_ret = (ng['close'].shift(-1) / df2['close'] - 1) * 100
fwd_open = (ng['close'].shift(-1) / ng['open'].shift(-1) - 1) * 100
same = (fwd_ret == fwd_open).sum() if len(fwd_ret) == len(fwd_open) else -1
print(f'\nDirect computation - same values: {same} / {len(fwd_ret)}')
print(f'fwd_ret sample: {fwd_ret.head(5).values}')
print(f'fwd_open sample: {fwd_open.head(5).values}')

# Check if ng['open'].shift(-1) is different from df2['close']
print(f'\nclose(T) first 5: {df2["close"].head(5).values}')
print(f'open(T+1) first 5: {ng["open"].shift(-1).head(5).values}')
print(f'close(T+1) first 5: {ng["close"].shift(-1).head(5).values}')
con.close()
