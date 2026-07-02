import pandas as pd
df = pd.read_csv(r'C:\Users\pc\Downloads\stock hist data\comprehensive_data\3MINDIA_FIFTEEN_MINUTE.csv')
print('Columns:', list(df.columns))
print('Head:')
print(df.head(2))
print('Dtypes:', dict(df.dtypes))
