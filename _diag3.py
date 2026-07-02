import duckdb, pandas as pd, numpy as np, pickle, sys
sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')

# 1. Feature_store 1day year distribution
print("=== RAW feature_store 1day ===")
df = con.execute("SELECT symbol,datetime FROM feature_store WHERE timeframe='1day'").fetchdf()
df['year'] = pd.to_datetime(df['datetime']).dt.year
print(f'Total rows: {len(df):,}')
yrs = df['year'].value_counts().sort_index()
for y, c in yrs.items():
    print(f'  {y}: {c:>8,}')

# 2. Check feature count
core_cols_query = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='feature_store' AND table_schema='main'").fetchdf()
all_cols = sorted(core_cols_query['column_name'].tolist())
print(f'\nTotal columns in feature_store: {len(all_cols)}')
print(f'Columns: {all_cols[:15]}...')

# 3. v5 results - year distribution
print('\n=== v5 results (from pickle) ===')
class V5Loader(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == 'cost_rt': return lambda s: s * 0.001
        return super().find_class(mod, name)
with open(BASE + r'\return_prediction_report_v5\results_v5.pkl', 'rb') as f:
    v5 = V5Loader(f).load()
rd5 = v5['rd']
rd5['dt'] = pd.to_datetime(rd5['dt'])
rd5['year'] = rd5['dt'].dt.year
yrs5 = rd5['year'].value_counts().sort_index()
print(f'Total predictions: {len(rd5):,}')
for y, c in yrs5.items():
    print(f'  {y}: {c:>8,}')

# 4. v6 results - year distribution
print('\n=== v6 results (from pickle) ===')
with open(BASE + r'\return_prediction_report_v6\results_v6.pkl', 'rb') as f:
    v6 = pickle.load(f)
rd6 = v6['rd']
rd6['dt'] = pd.to_datetime(rd6['dt'])
rd6['year'] = rd6['dt'].dt.year
yrs6 = rd6['year'].value_counts().sort_index()
print(f'Total predictions: {len(rd6):,}')
for y, c in yrs6.items():
    print(f'  {y}: {c:>8,}')

print(f'\n=== Summary ===')
print(f'v5: {len(rd5):,} predictions, date range {rd5["dt"].min().date()} to {rd5["dt"].max().date()}, {rd5["sym"].nunique()} symbols')
print(f'v6: {len(rd6):,} predictions, date range {rd6["dt"].min().date()} to {rd6["dt"].max().date()}, {rd6["sym"].nunique()} symbols')

# 5. Compare key metrics
print(f'\n=== v5 metrics ===')
for c in v5['bt'].columns:
    if '_net' not in c: continue
    s = v5['bt'][c].dropna()
    if len(s) < 10: continue
    cagr = (1+s/100).prod()**(252/len(s))-1
    print(f'  {c:20s} CAGR={cagr*100:+8.1f}%')

print(f'\nv5 features: {v5["features"]}')
print(f'v6 features: {v6["features"]}')

# What are the differences in features?
print(f'\n=== Feature set differences ===')
f5 = set(v5['features'])
f6 = set(v6['features'])
print(f'v5 has {len(f5)} features, v6 has {len(f6)} features')
print(f'Common: {len(f5 & f6)}')
print(f'Only v5: {sorted(f5 - f6)[:20]}')
print(f'Only v6: {sorted(f6 - f5)[:20]}')

con.close()
