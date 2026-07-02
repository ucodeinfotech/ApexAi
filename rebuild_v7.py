"""v7: Expanded universe (446+ symbols) + all fixed indicators"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math
import xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from pathlib import Path
from datetime import datetime, timedelta
import optuna, shap
import torch
import torch.nn as nn
warnings.filterwarnings('ignore'); np.random.seed(42)
torch.manual_seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_v7'
OUT.mkdir(exist_ok=True)
t0 = datetime.now()

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    total = brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
    return total

CPS = cost_rt(TOTAL_POS)
print(f'Cost/stock (single): {CPS*100:.3f}%')

# ─── Step 1: Rebuild 1day feature_store for ALL symbols ───
print('\n=== Step 1: Rebuilding 1day feature_store (all 446+ symbols) ===')
from src.features.indicators import compute_all_features
con = duckdb.connect(str(DB))
syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day' ORDER BY symbol").fetchall()]
print(f'Rebuilding features for {len(syms)} symbols...')
t1 = time.time()
fs_rows = 0
for i, sym in enumerate(syms):
    df = con.execute("SELECT datetime,open,high,low,close,volume FROM raw_market WHERE symbol=? AND timeframe='1day' ORDER BY datetime", [sym]).fetchdf()
    if len(df) < 200: continue
    df = compute_all_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df['symbol'] = sym; df['timeframe'] = '1day'
    con.execute("DELETE FROM feature_store WHERE symbol=? AND timeframe='1day'", [sym])
    table_cols = [r[1] for r in con.execute("PRAGMA table_info('feature_store')").fetchall()]
    df_cols = [c for c in table_cols if c in df.columns]
    con.register('df', df[df_cols])
    con.execute("INSERT INTO feature_store SELECT * FROM df")
    con.unregister('df')
    fs_rows += len(df)
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(syms)} symbols ({time.time()-t1:.0f}s, {fs_rows:,} rows)')
print(f'Feature store: {fs_rows:,} rows in {time.time()-t1:.0f}s')

# ─── Step 2: Rebuild market_structure for ALL symbols ───
print('\n=== Step 2: Rebuilding market_structure (all symbols) ===')
from src.patterns.market_structure import compute_all_structure
nifty = con.execute("SELECT datetime, close FROM raw_market WHERE symbol='NIFTY50' AND timeframe='5min' ORDER BY datetime").fetchdf()
if len(nifty) > 0:
    nifty_close = nifty.set_index('datetime')['close']
    nifty_close = nifty_close[~nifty_close.index.duplicated(keep='first')]
else:
    nifty_close = None

ms_syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day' AND symbol NOT IN ('NIFTY50','BANKNIFTY') ORDER BY symbol").fetchall()]
t2 = time.time()
ms_cnt = 0
for i, sym in enumerate(ms_syms):
    df = con.execute("SELECT datetime,open,high,low,close,volume FROM raw_market WHERE symbol=? AND timeframe='1day' ORDER BY datetime", [sym]).fetchdf()
    if len(df) < 100: continue
    df = compute_all_structure(df, market_close=nifty_close)
    df = df.replace([np.inf, -np.inf], np.nan)
    df['symbol'] = sym; df['timeframe'] = '1day'
    con.execute("DELETE FROM market_structure WHERE symbol=? AND timeframe='1day'", [sym])
    table_cols = [r[1] for r in con.execute("PRAGMA table_info('market_structure')").fetchall()]
    avail = [c for c in table_cols if c in df.columns]
    con.register('df', df[avail])
    con.execute("INSERT INTO market_structure SELECT * FROM df")
    con.unregister('df')
    ms_cnt += 1
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(ms_syms)} symbols ({time.time()-t2:.0f}s)')
print(f'Market structure: {ms_cnt} symbols in {time.time()-t2:.0f}s')

# ─── Step 3: Rebuild market_regimes ───
print('\n=== Step 3: Rebuilding market_regimes (fixed expanding quantile) ===')
con.execute("DELETE FROM market_regimes")
nifty = con.execute("SELECT datetime, close FROM raw_market WHERE symbol='NIFTY50' AND timeframe='5min' ORDER BY datetime").fetchdf()
if len(nifty) > 500:
    nifty['datetime'] = pd.to_datetime(nifty['datetime'])
    daily = nifty.set_index('datetime').resample('D').agg({'close':'last'}).dropna()
    daily['returns'] = daily['close'].pct_change() * 100
    daily = daily.dropna()
    daily['ma50'] = daily['close'].rolling(50, min_periods=30).mean()
    daily['above_ma'] = daily['close'] > daily['ma50']
    daily['ret_20d'] = daily['close'].pct_change(20) * 100
    daily['bull'] = daily['above_ma'] & (daily['ret_20d'] > 0)
    daily['bear'] = (~daily['above_ma']) & (daily['ret_20d'] < -5)
    daily['sideways'] = ~daily['bull'] & ~daily['bear']
    daily['vol_20d'] = daily['returns'].rolling(20).std()
    daily['high_vol'] = daily['vol_20d'] >= daily['vol_20d'].expanding().quantile(0.7)
    regime_map = {'bull': 1, 'bear': -1, 'sideways': 0}
    rows = []
    for idx, row in daily.iterrows():
        regime = 'bull' if row['bull'] else ('bear' if row['bear'] else 'sideways')
        rows.append(('1day', pd.Timestamp(idx), regime, regime_map[regime], 'high_vol' if row['high_vol'] else 'normal_vol'))
    con.register('reg', pd.DataFrame(rows, columns=['timeframe','datetime','regime_label','regime_id','volatility_regime']))
    con.execute("INSERT INTO market_regimes SELECT * FROM reg")
    con.unregister('reg')
    summary = con.execute("SELECT regime_label, COUNT(1) FROM market_regimes GROUP BY regime_label").fetchall()
    print(f'  Regimes: {dict(summary)}')
con.close()

print(f'\nPreprocessing done in {(datetime.now()-t0).total_seconds():.0f}s')
