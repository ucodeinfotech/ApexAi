import duckdb
conn = duckdb.connect('warehouse/market_data.duckdb')
tables = conn.execute("SELECT table_name, table_type FROM information_schema.tables").fetchdf()
print('Tables:')
print(tables)

schema = conn.execute("DESCRIBE raw_market").fetchdf()
print('\nraw_market schema:')
print(schema)

schema2 = conn.execute("DESCRIBE feature_store").fetchdf()
print('\nfeature_store schema:')
print(schema2)

cnt = conn.execute("SELECT COUNT(*) FROM raw_market WHERE timeframe='1day'").fetchone()[0]
print(f'\nraw_market 1day rows: {cnt}')

feat_names = schema2['column_name'].tolist()
swing_feats = [c for c in feat_names if 'swing' in c.lower() or 'high' in c.lower() or 'low' in c.lower()]
print('Swing-related columns:', swing_feats)
print()
print('All feature_store columns:', feat_names)

# Check if market_regimes exists
tables_list = tables['table_name'].tolist()
print('\nAll tables:', tables_list)
conn.close()
