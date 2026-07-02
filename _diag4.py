import duckdb, pandas as pd, numpy as np, pickle, sys, time
sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')
from src.features.indicators import compute_all_features

# Define same feature lists as train_v6.py
BASE_F = ['sma_5','sma_10','sma_20','sma_50','ema_5','ema_10','ema_20','ema_50',
    'rsi_7','rsi_14','rsi_21','macd_line','macd_signal','macd_hist','adx',
    'plus_di','minus_di','atr_7','atr_14','atr_21','bb_pct_b','bb_width',
    'kc_width','dc_width','obv','cmf','stoch_k','stoch_d','williams_r',
    'mfi','uo','cci','trix','roc_5','roc_10','roc_20','zscore_20',
    'skew_20','kurt_20','hv_10','hv_20','hv_30','eom','fi','vpt']
EXTRA_F = ['ret_1d','ret_5d','ret_10d','ret_20d','log_ret_1d','log_ret_5d',
    'log_ret_10d','log_ret_20d','close_vs_sma_10','close_vs_sma_20',
    'close_vs_sma_50','close_vs_sma_200','body_ratio_5','body_ratio_10',
    'body_ratio_20','aroon_up','aroon_down','aroon_osc','serial_corr_20',
    'vol_ratio_5','vol_ratio_10','vol_ratio_20','swing_high','swing_low',
    'pivot','r1','r2','s1','s2','psar','range_5','range_10','range_20',
    'ad_line','bb_lower','bb_middle','bb_upper','dc_lower','dc_mid','dc_upper',
    'kc_lower','kc_upper','ema_200','sma_200','wma_10','wma_20']
core_cols = ','.join(f'"{f}"' for f in (BASE_F + EXTRA_F))

# Load 1day feature_store
print("Loading feature_store 1day...")
t0 = time.time()
df = con.execute(f"SELECT symbol,datetime,{core_cols},open,high,low,close,volume FROM feature_store WHERE timeframe='1day' ORDER BY datetime").fetchdf()
print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")
ds = pd.to_datetime(df['datetime'])
df['datetime'] = (ds.dt.tz_localize(None).astype('datetime64[us]') if ds.dt.tz is not None else ds.astype('datetime64[us]'))
df['year'] = pd.to_datetime(df['datetime']).dt.year

# Check which features have NaN
base_extras = BASE_F + EXTRA_F
na_counts = df[base_extras].isna().sum()
print(f"\nFeatures with >0 NaN:")
for c in na_counts[na_counts > 0].sort_values(ascending=False).index:
    print(f"  {c}: {na_counts[c]:,} NaN ({na_counts[c]/len(df)*100:.1f}%)")

# Check how many rows survive NaN filter
mask = df[base_extras].notna().all(axis=1)
print(f"\nRows surviving base feature NaN filter: {mask.sum():,} / {len(df):,} ({mask.sum()/len(df)*100:.1f}%)")

# Per year
df['valid'] = mask
yr_valid = df.groupby('year').agg(rows=('valid','size'), valid=('valid','sum'), pct=('valid','mean'))
print(f"\nPer year valid rows:")
for y, r in yr_valid.iterrows():
    print(f"  {int(y)}: {r['valid']:,}/{r['rows']:,} ({r['pct']*100:.1f}%)")

# Check v6 close-close predictions
print("\n=== v6 results (from direct data check) ===")
try:
    with open(BASE + r'\return_prediction_report_v6\results_v6.pkl', 'rb') as f:
        v6 = pickle.load(f)
    rd6 = v6['rd']
    print(f"v6 predictions: {len(rd6):,}, cols: {rd6.columns.tolist()[:10]}")
    print(f"v6 dates: {pd.to_datetime(rd6['dt']).min()} to {pd.to_datetime(rd6['dt']).max()}")
    yrs6 = pd.to_datetime(rd6['dt']).dt.year.value_counts().sort_index()
    for y, c in yrs6.items():
        print(f"  {int(y)}: {c:,}")
except Exception as e:
    print(f"Cannot load v6 results: {e}")
    # Check window info from optuna files
    import glob
    opt_files = sorted(glob.glob(BASE + r'\return_prediction_report_v6\optuna_window_*.txt'))
    for f in opt_files:
        with open(f) as fh:
            for line in fh.readlines()[:3]:
                print(line.strip())

con.close()
