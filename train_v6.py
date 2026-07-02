"""v6: Fixed swing detection + transformer model + retrain"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, math
import xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from pathlib import Path
from datetime import datetime, timedelta
import optuna, shap
from scipy import stats
import torch
import torch.nn as nn
warnings.filterwarnings('ignore'); np.random.seed(42)
torch.manual_seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_v6'
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

# ─── Step 1: Rebuild feature_store 1day with fixed indicators ───
print('\n=== Step 1: Rebuilding 1day feature_store with FIXED indicators ===')
from src.features.indicators import compute_all_features
con = duckdb.connect(str(DB))
syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day' ORDER BY symbol").fetchall()]
print(f'Rebuilding features for {len(syms)} symbols (1day)...')
t1 = time.time()
total_rows = 0
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
    total_rows += len(df)
    if (i+1) % 20 == 0:
        print(f'  {i+1}/{len(syms)} symbols ({time.time()-t1:.0f}s, {total_rows:,} rows)')
print(f'Feature store rebuild complete: {total_rows:,} rows in {time.time()-t1:.0f}s')

# ─── Step 2: Rebuild market_structure for RS features ───
print('\n=== Step 2: Rebuilding market_structure ===')
from src.patterns.market_structure import compute_all_structure, STRUCTURE_COLUMNS
nifty = con.execute("SELECT datetime, close FROM raw_market WHERE symbol='NIFTY50' AND timeframe='5min' ORDER BY datetime").fetchdf()
if len(nifty) > 0:
    nifty_close = nifty.set_index('datetime')['close']
    nifty_close = nifty_close[~nifty_close.index.duplicated(keep='first')]
else:
    nifty_close = None

ms_syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market WHERE timeframe='1day' AND symbol NOT IN ('NIFTY50','BANKNIFTY') ORDER BY symbol").fetchall()]
t2 = time.time()
ms_rows = 0
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
    ms_rows += 1
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(ms_syms)} symbols ({time.time()-t2:.0f}s)')
print(f'Market structure rebuild: {ms_rows} symbols in {time.time()-t2:.0f}s')

# ─── Step 3: Rebuild market_regimes with fixed volatility_threshold ───
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
    # FIXED: expanding().quantile() instead of full-history quantile()
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

# ─── Step 4: Transformer model definition ───
print('\n=== Step 4: Building Transformer model ===')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockTransformer(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2, seq_len=20):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.decoder(x).squeeze(-1)

# ─── Step 5: Load features ───
print('\n=== Step 5: Loading features ===')
con = duckdb.connect(str(DB))
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
RS_F = ['rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector',
    'rs_momentum_10','rs_momentum_20']
CAL_F = ['dow','month','is_month_end','is_quarter_end','is_thursday']
VIX_F = ['vix_close','vix_change','vix_range','vix_ma_5','vix_ma_20',
    'vix_zscore_20','vix_ma_5_r','vix_ma_20_r','vix_high_r']
DV_F = ['delivery_pct','delivery_pct_ma5','delivery_pct_ma20','delivery_delta']
MTF_F = ['intra_rsi_mean','intra_rsi_std','intra_vol_std','intra_range_sum',
    'intra_range_max','intra_bb_width_mean','intra_macd_std',
    'intra_rsi_mean_ma5','intra_range_sum_ma5','intra_vol_std_ma5']
RNG_F = ['range_pct']
ALL_FEATS = BASE_F + EXTRA_F + RNG_F + CAL_F + VIX_F + DV_F + MTF_F + RS_F
RANK_FEATS = ['rsi_7','rsi_14','rsi_21','atr_7','atr_14','atr_21','bb_pct_b',
    'bb_width','kc_width','dc_width','stoch_k','stoch_d','williams_r',
    'mfi','uo','cci','trix','roc_5','roc_10','roc_20','zscore_20',
    'skew_20','kurt_20','hv_10','hv_20','hv_30','eom','fi','vpt',
    'ret_1d','ret_5d','ret_10d','ret_20d','log_ret_1d','log_ret_5d',
    'log_ret_10d','log_ret_20d','close_vs_sma_10','close_vs_sma_20',
    'close_vs_sma_50','close_vs_sma_200','body_ratio_5','body_ratio_10',
    'body_ratio_20','aroon_up','aroon_down','aroon_osc','serial_corr_20',
    'vol_ratio_5','vol_ratio_10','vol_ratio_20','swing_high','swing_low',
    'range_pct','range_5','range_10','range_20','vix_close','vix_change',
    'vix_range','vix_ma_5_r','vix_ma_20_r','rs_vs_market','rs_vs_sector',
    'rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20',
    'delivery_pct','delivery_pct_ma5','delivery_delta']

core_cols = ','.join(f'"{f}"' for f in (BASE_F + EXTRA_F))
df = con.execute(f'SELECT symbol,datetime,{core_cols},open,high,low,close,volume '
    "FROM feature_store WHERE timeframe='1day' ORDER BY datetime").fetchdf()
ds = pd.to_datetime(df['datetime'])
df['datetime'] = (ds.dt.tz_localize(None).astype('datetime64[us]')
                  if ds.dt.tz is not None else ds.astype('datetime64[us]'))
df['range_pct'] = (df['high']-df['low'])/df['close']*100
dc = pd.to_datetime(df['datetime'])
df['year']=dc.dt.year; df['dow']=dc.dt.dayofweek; df['month']=dc.dt.month
df['is_month_end']=dc.dt.is_month_end.astype(int); df['is_quarter_end']=dc.dt.is_quarter_end.astype(int)
df['is_thursday']=(df['dow']==3).astype(int)
print(f'Base: {len(df):,} rows from {df["symbol"].nunique()} symbols')

v = con.execute('SELECT datetime,vix_close,vix_change,vix_range,vix_ma_5,vix_ma_20,vix_zscore_20 FROM vix_data ORDER BY datetime').fetchdf()
vd=pd.to_datetime(v['datetime']); v['datetime']=(vd.dt.tz_localize(None).astype('datetime64[us]') if vd.dt.tz is not None else vd.astype('datetime64[us]'))
v['vix_ma_5_r']=v['vix_close']/v['vix_ma_5'].replace(0,np.nan)-1; v['vix_ma_20_r']=v['vix_close']/v['vix_ma_20'].replace(0,np.nan)-1
v['vix_high_r']=0.0; v=v.fillna(0)
df=pd.merge_asof(df.sort_values('datetime'),v.sort_values('datetime'),on='datetime',direction='backward')

dv=con.execute('SELECT symbol,date,delivery_pct FROM delivery_data ORDER BY symbol,date').fetchdf()
dv['date']=pd.to_datetime(dv['date']).astype('datetime64[us]')
dv['delivery_pct_ma5']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(5,min_periods=2).mean())
dv['delivery_pct_ma20']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(20,min_periods=5).mean())
dv['delivery_delta']=dv['delivery_pct']-dv['delivery_pct_ma5']; dv=dv.fillna(0)
dv=dv.drop_duplicates(subset=['symbol','date'])
df['date_m']=pd.to_datetime(df['datetime']).dt.normalize()
df=df.merge(dv,left_on=['symbol','date_m'],right_on=['symbol','date'],how='left')
for c in DV_F: df[c]=df[c].fillna(0)

m=con.execute("SELECT symbol,datetime,high,low,close,rsi_14,bb_width,macd_hist FROM feature_store WHERE timeframe='60min' ORDER BY datetime").fetchdf()
md=pd.to_datetime(m['datetime']); m['datetime']=(md.dt.tz_localize(None).astype('datetime64[us]') if md.dt.tz is not None else md.astype('datetime64[us]'))
m['date']=pd.to_datetime(m['datetime']).dt.normalize(); m['r']=(m['high']-m['low'])/m['close']*100
mtf=m.groupby(['symbol','date']).agg(intra_rsi_mean=('rsi_14','mean'),intra_rsi_std=('rsi_14','std'),
    intra_vol_std=('close',lambda x:float(np.std(np.diff(x.values))/(np.mean(x)+1e-12)*100) if len(x)>1 else 0),
    intra_range_sum=('r','sum'),intra_range_max=('r','max'),intra_bb_width_mean=('bb_width','mean'),
    intra_macd_std=('macd_hist','std')).reset_index()
mtf=mtf.drop_duplicates(subset=['symbol','date'])
for c in ['intra_rsi_mean','intra_range_sum','intra_vol_std']:
    mtf[f'{c}_ma5']=mtf.groupby('symbol')[c].transform(lambda x:x.rolling(5,min_periods=2).mean())
df=df.merge(mtf,left_on=['symbol','date_m'],right_on=['symbol','date'],how='left')
for c in MTF_F: df[c]=df[c].fillna(0)

ms=con.execute("SELECT symbol,datetime,rs_vs_market,rs_vs_sector,rs_ratio_market,rs_ratio_sector,rs_momentum_10,rs_momentum_20 FROM market_structure WHERE timeframe='1day' ORDER BY datetime").fetchdf()
msd=pd.to_datetime(ms['datetime']); ms['datetime']=(msd.dt.tz_localize(None).astype('datetime64[us]') if msd.dt.tz is not None else msd.astype('datetime64[us]'))
ms=ms.drop_duplicates(subset=['symbol','datetime'])
df=df.merge(ms,on=['symbol','datetime'],how='left')
for c in RS_F: df[c]=df[c].fillna(0)

df = df.sort_values(['datetime','symbol']).reset_index(drop=True)
for feat in RANK_FEATS:
    if feat in df.columns:
        df[f'rank_{feat}'] = df.groupby('datetime')[feat].rank(pct=True).fillna(0.5)
CS_FEATS = [f'rank_{f}' for f in RANK_FEATS if f'rank_{f}' in df.columns]

breadth = con.execute("""SELECT DATE_TRUNC('day',datetime)::DATE as day, COUNT(*) FILTER (WHERE close>open) as adv,
    COUNT(*) FILTER (WHERE close<open) as dec, COUNT(*) as tot FROM 
    (SELECT datetime,close,open FROM raw_market WHERE timeframe='1day'
     AND symbol IN (SELECT DISTINCT symbol FROM feature_store WHERE timeframe='1day')) sub GROUP BY day""").fetchdf()
breadth['day']=pd.to_datetime(breadth['day']).astype('datetime64[us]')
breadth['adv_dec_ratio']=breadth['adv']/breadth['dec'].replace(0,1); breadth['adv_dec_diff']=breadth['adv']-breadth['dec']
breadth['brd_pct']=breadth['adv']/breadth['tot']
for c in ['adv_dec_ratio','adv_dec_diff','brd_pct']:
    breadth[f'{c}_ma5']=breadth[c].rolling(5,min_periods=2).mean(); breadth[f'{c}_ma20']=breadth[c].rolling(20,min_periods=5).mean()
BRD_F=[c for c in breadth.columns if c not in ['day']]; df=df.merge(breadth,left_on='date_m',right_on='day',how='left')
for c in BRD_F: df[c]=df[c].fillna(0)

reg=con.execute('SELECT datetime,regime_label,regime_id FROM market_regimes ORDER BY datetime').fetchdf()
rdt=pd.to_datetime(reg['datetime']); reg['datetime']=(rdt.dt.tz_localize(None).astype('datetime64[us]') if rdt.dt.tz is not None else rdt.astype('datetime64[us]'))
df=df.merge(reg,on='datetime',how='left'); df['regime_label']=df['regime_label'].fillna('sideways')
df['regime_id']=df['regime_id'].fillna(0).astype(int)
con.close()

ng=df.groupby('symbol')
RET_COL='fwd_return_1d'
df[RET_COL]=(ng['close'].shift(-1)/df['close']-1)*100
df['fwd_open_ret_1d'] = (ng['close'].shift(-1) / ng['open'].shift(-1) - 1) * 100
TARGETS_MT=[RET_COL]
for nd in [3,5,10,20]:
    c=f'fwd_return_{nd}d'
    df[c]=(ng['close'].shift(-nd)/df['close']-1)*100
    TARGETS_MT.append(c)
df['triple_barrier']=np.where(df[RET_COL]>=2.0,1,np.where(df[RET_COL]<=-2.0,-1,0))

rl=df[RET_COL].quantile(0.005); ru=df[RET_COL].quantile(0.995); df[RET_COL]=df[RET_COL].clip(rl,ru)
ALL_F=[f for f in ALL_FEATS if f in df.columns]+CS_FEATS+BRD_F
corr_sample=df[ALL_F].sample(min(10000,len(df)),random_state=42).corr().abs()
upper=corr_sample.where(np.triu(np.ones(corr_sample.shape),k=1).astype(bool))
to_drop=set()
for col in upper.columns:
    if col in to_drop: continue
    hi=list(upper.index[upper[col]>0.95])
    to_drop.update(hi)
if to_drop:
    print(f'Dropping {len(to_drop)} high-correlation features (>0.95)')
    ALL_F=[f for f in ALL_F if f not in to_drop]
pre_nan=df[ALL_F].isna().sum().sum(); nan_feats=df[ALL_F].isna().any()
if pre_nan>0:
    print(f'Imputing {pre_nan:,} NaN values across {nan_feats.sum()} features per-symbol...')
    for c in ALL_F:
        if df[c].isna().any():
            df[c]=df.groupby('symbol',group_keys=False)[c].transform(lambda s:s.ffill().bfill().fillna(0))
post_nan=df[ALL_F].isna().sum().sum()
if post_nan>0:
    print(f'  {post_nan} NaN remaining (fallback fill=0)')
    df[ALL_F]=df[ALL_F].fillna(0)
df_clean=df.dropna(subset=[RET_COL]).copy()
print(f'Features: {len(ALL_F)}, Total rows: {len(df_clean):,}, Symbols: {df_clean["symbol"].nunique()}, NaN after impute: {df_clean[ALL_F].isna().sum().sum()}')
print(f'\nPreprocessing done in {(datetime.now()-t0).total_seconds():.0f}s')

# ─── Step 6: Walkforward with Transformer ───
years=sorted(df_clean['year'].unique())
windows=[(years[:i],years[i]) for i in range(2,len(years))]
print(f'\n{len(windows)} walkforward windows')

def get_shap_imp(model, X, feat_names):
    e = shap.TreeExplainer(model)
    sv = e.shap_values(X)
    return dict(zip(feat_names, np.abs(sv).mean(axis=0)))

def calc_metrics(s, n=252):
    if len(s)<5 or s.std()==0: return 0,0,0,0
    cagr=(1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh=s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr=(s>0).mean()
    dd=((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd

def make_transformer_sequences(df_sym, feature_cols, target_col, seq_len=20):
    """Create sequences for a single symbol."""
    vals = df_sym[feature_cols].values.astype(np.float32)
    tgt = df_sym[target_col].values.astype(np.float32)
    dates = df_sym['datetime'].values
    X_seqs, y_seqs, d_seqs = [], [], []
    for i in range(len(vals) - seq_len):
        if np.isnan(vals[i:i+seq_len]).any() or np.isnan(tgt[i+seq_len]):
            continue
        X_seqs.append(vals[i:i+seq_len])
        y_seqs.append(tgt[i+seq_len])
        d_seqs.append(dates[i+seq_len])
    return np.array(X_seqs), np.array(y_seqs), d_seqs

all_results = []; all_models_dict = {}; fi_over_time = {}

for wi, (ty, test_yr) in enumerate(windows):
    tr_raw = df_clean[df_clean['year'].isin(ty)].copy()
    test = df_clean[df_clean['year']==test_yr].copy()
    if len(test)<50: continue
    embargo_start = test['datetime'].min()-timedelta(days=7)
    train = tr_raw[tr_raw['datetime']<embargo_start].copy()
    print(f'\n[{wi+1:2d}/{len(windows)}] Test {test_yr}: train={len(train):,}, test={len(test):,}')

    tr_nona = train[ALL_F].dropna(axis=1,how='any')
    valid = [c for c in tr_nona.columns if tr_nona[c].std()>1e-8]
    keep = [RET_COL,'fwd_open_ret_1d','triple_barrier','symbol','datetime','regime_label']+valid
    train = train[[c for c in keep if c in train.columns]].copy()
    test = test[[c for c in keep if c in test.columns]].copy()
    train=train.dropna(subset=valid).reset_index(drop=True)
    test=test.dropna(subset=valid+[RET_COL]).reset_index(drop=True)
    valid2=[c for c in valid if c in train.columns and c in test.columns]
    if len(valid2)<5 or len(train)<100 or len(test)<5: continue

    # Optuna
    def obj(trial):
        p={'n_estimators':trial.suggest_int('n',80,150),'max_depth':trial.suggest_int('d',3,6),
           'learning_rate':trial.suggest_float('lr',0.02,0.08,log=True),
           'subsample':trial.suggest_float('ss',0.6,1.0),'colsample_bytree':trial.suggest_float('cs',0.6,1.0),
           'reg_alpha':trial.suggest_float('al',1e-8,1.0,log=True),'reg_lambda':trial.suggest_float('la',1e-8,1.0,log=True)}
        tr_sorted=train.sort_values('datetime').reset_index(drop=True)
        sp=int(len(tr_sorted)*0.8); tr,val=tr_sorted.iloc[:sp],tr_sorted.iloc[sp:]
        if len(val)<20: return -999
        s=StandardScaler(); m=xgb.XGBRegressor(random_state=42,n_jobs=-1,verbosity=0,**p)
        m.fit(s.fit_transform(tr[valid2].values),tr[RET_COL].values)
        pr=m.predict(s.transform(val[valid2].values))
        return r2_score(val[RET_COL].values,pr) if not np.isnan(pr).any() else -999
    st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj,n_trials=10,show_progress_bar=False)
    best_hp=st.best_params
    xgb_hp={k.replace('lr','learning_rate').replace('n','n_estimators').replace('d','max_depth').replace('ss','subsample').replace('cs','colsample_bytree').replace('al','reg_alpha').replace('la','reg_lambda'):v for k,v in best_hp.items()}

    # SHAP feature selection
    ss=StandardScaler(); Xs=ss.fit_transform(train[valid2].values)
    ms=xgb.XGBRegressor(n_estimators=100,max_depth=4,random_state=42,n_jobs=-1,verbosity=0)
    ms.fit(Xs,train[RET_COL].values)
    try:
        si=get_shap_imp(ms,Xs[:min(2000,len(Xs))],valid2)
        sf=sorted(si.items(),key=lambda x:-x[1])
        ci=np.cumsum([s[1] for s in sf])
        if ci[-1] <= 0:
            tf=valid2[:min(30, len(valid2))]
        else:
            ti=np.searchsorted(ci,ci[-1]*0.8)+1
            tf=[s[0] for s in sf[:max(ti,15)]]
        fi_over_time[test_yr]=si
    except:
        tf=valid2

    scaler=StandardScaler(); X_tr=scaler.fit_transform(train[tf].values); X_te=scaler.transform(test[tf].values)
    y_tr=train[RET_COL].values; y_te=test[RET_COL].values

    train['dr']=train.groupby('datetime')[RET_COL].rank('dense',ascending=True).astype(int)-1
    max_r=train['dr'].max(); train['dr_cap']=(train['dr']/(max_r/30+1)).astype(int)
    train['gid']=train.groupby('datetime').ngroup()
    y_rank=train['dr_cap'].values; grps=train.groupby('gid').size().values

    # 7 base models
    m_xgb=xgb.XGBRegressor(random_state=42,n_jobs=-1,verbosity=0,**xgb_hp).fit(X_tr,y_tr)
    m_rank=xgb.XGBRanker(n_estimators=120,max_depth=4,learning_rate=0.05,random_state=42,n_jobs=-1,verbosity=0,objective='rank:ndcg',ndcg_exp_gain=False).fit(X_tr,y_rank,group=grps)
    m_lgb=lgb.LGBMRegressor(n_estimators=best_hp['n'],max_depth=best_hp['d'],learning_rate=best_hp['lr'],subsample=best_hp['ss'],colsample_bytree=best_hp['cs'],reg_alpha=best_hp['al'],reg_lambda=best_hp['la'],random_state=42,n_jobs=-1,verbosity=-1).fit(X_tr,y_tr)
    m_lgb_r=lgb.LGBMRanker(n_estimators=100,max_depth=4,learning_rate=0.05,random_state=42,n_jobs=-1,verbosity=-1,max_position=30).fit(X_tr,y_rank,group=grps)
    m_cb=cb.CatBoostRegressor(n_estimators=best_hp['n'],learning_rate=best_hp['lr'],random_seed=42,verbose=0,allow_writing_files=False).fit(X_tr,y_tr)
    m_rf=RandomForestRegressor(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1).fit(X_tr,y_tr)
    m_et=ExtraTreesRegressor(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1).fit(X_tr,y_tr)

    # Transformer: train on sequences
    print(f'  Training Transformer (seq_len=20, {len(tf)} feats)...')
    seq_len = 20
    X_seq_all, y_seq_all = [], []
    for sym in train['symbol'].unique():
        sym_tr = train[train['symbol']==sym].sort_values('datetime')
        if len(sym_tr) < seq_len + 5: continue
        Xs_s, ys_s, _ = make_transformer_sequences(sym_tr, tf, RET_COL, seq_len)
        if len(Xs_s) > 0:
            X_seq_all.append(Xs_s); y_seq_all.append(ys_s)
    if len(X_seq_all) > 0:
        X_seq = np.concatenate(X_seq_all)
        y_seq = np.concatenate(y_seq_all)
        # Normalize per-feature across the sequence dimension
        X_seq_t = torch.from_numpy(X_seq).float()
        # Scale per feature using training stats
        seq_mean = X_seq_t.mean(dim=(0,1), keepdim=True)
        seq_std = X_seq_t.std(dim=(0,1), keepdim=True) + 1e-8
        X_seq_t = (X_seq_t - seq_mean) / seq_std

        tf_model = StockTransformer(n_features=len(tf), d_model=32, nhead=4, num_layers=2, seq_len=seq_len)
        optimizer = torch.optim.AdamW(tf_model.parameters(), lr=0.001, weight_decay=1e-5)
        tf_model.train()
        n_epochs = 15
        batch_size = 256
        y_seq_t = torch.from_numpy(y_seq).float()
        for epoch in range(n_epochs):
            perm = torch.randperm(len(X_seq_t))
            epoch_loss = 0
            n_batches = 0
            for start in range(0, len(X_seq_t), batch_size):
                idx = perm[start:start+batch_size]
                opt = optimizer.zero_grad()
                pred = tf_model(X_seq_t[idx])
                loss = nn.MSELoss()(pred, y_seq_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tf_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch+1) % 5 == 0:
                print(f'    Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/n_batches:.6f}')

        # Predict on test set
        tf_model.eval()
        p_transformer = np.zeros(len(test))
        with torch.no_grad():
            for sym in test['symbol'].unique():
                sym_te = test[test['symbol']==sym].sort_values('datetime')
                sym_tr_full = train[train['symbol']==sym].sort_values('datetime')
                # Combine train + test for sequence creation, but only predict test periods
                combined = pd.concat([sym_tr_full, sym_te]).drop_duplicates(subset='datetime').sort_values('datetime')
                if len(combined) < seq_len + 1: continue
                Xs_s, _, ds_s = make_transformer_sequences(combined, tf, RET_COL, seq_len)
                if len(Xs_s) == 0: continue
                Xt = torch.from_numpy(Xs_s).float()
                Xt = (Xt - seq_mean) / seq_std
                preds = tf_model(Xt).numpy()
                for pi, dt_pred in enumerate(ds_s):
                    mask = (test['symbol']==sym) & (test['datetime'].values == dt_pred)
                    p_transformer[mask] = preds[pi]
    else:
        p_transformer = np.zeros(len(test))
        print('  WARNING: No sequences for transformer, using zeros')

    p_xgb=m_xgb.predict(X_te); p_rank=m_rank.predict(X_te); p_lgb=m_lgb.predict(X_te)
    p_lgb_r=m_lgb_r.predict(X_te); p_cb=m_cb.predict(X_te); p_rf=m_rf.predict(X_te); p_et=m_et.predict(X_te)

    if len(all_results) > 1000:
        hist = pd.DataFrame(all_results).sort_values('dt')
        meta_X = hist[['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer']].values
        meta_y = hist['act'].values
        meta = Ridge(alpha=1.0, random_state=42)
        meta.fit(meta_X, meta_y)
        p_stack = meta.predict(np.column_stack([p_xgb, p_rank, p_lgb, p_lgb_r, p_cb, p_rf, p_et, p_transformer]))
    else:
        p_stack = np.mean([p_xgb, p_rank, p_lgb, p_lgb_r, p_cb, p_rf, p_et, p_transformer], axis=0)
    p_avg = np.mean([p_xgb, p_rank, p_lgb, p_lgb_r, p_cb, p_rf, p_et, p_transformer], axis=0)

    for name, p in [('XGB',p_xgb),('Ranker',p_rank),('LGB',p_lgb),('LGBR',p_lgb_r),
                    ('CatB',p_cb),('RF',p_rf),('ET',p_et),('TF',p_transformer),('Avg',p_avg),('Stack',p_stack)]:
        if np.isnan(p).any(): continue
        r2=r2_score(y_te,p)
        corr=np.corrcoef(p,y_te)[0,1] if np.std(p)>1e-12 and np.std(y_te)>1e-12 else 0
        da=((p>0)==(y_te>0)).mean()
        print(f'  {name:6s} R²={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

    all_models_dict[test_yr] = {'xgb':m_xgb,'ranker':m_rank,'lgb':m_lgb,'lgb_r':m_lgb_r,'cb':m_cb,'rf':m_rf,'et':m_et,'transformer':tf_model if len(X_seq_all)>0 else None,'features':tf,'scaler':scaler}
    for i in range(len(test)):
        all_results.append(dict(zip(
            ['dt','sym','act','act_open','xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack','regime','tb'],
            [test['datetime'].iloc[i],test['symbol'].iloc[i],y_te[i],
             test['fwd_open_ret_1d'].iloc[i] if 'fwd_open_ret_1d' in test.columns else y_te[i],
             p_xgb[i],p_rank[i],p_lgb[i],p_lgb_r[i],p_cb[i],p_rf[i],p_et[i],p_transformer[i],p_avg[i],p_stack[i],
             test['regime_label'].iloc[i] if 'regime_label' in test.columns else '?',
             test['triple_barrier'].iloc[i] if 'triple_barrier' in test.columns else 0])))

if len(all_results) == 0:
    print('ERROR: no results'); exit(1)
rd = pd.DataFrame(all_results)
rd.to_pickle(OUT/'results_raw.pkl')

print(f'\n{"="*50}')
print(f'v6 PERFORMANCE (fixed features + transformer)')
print(f'{"="*50}')
print(f'Total predictions: {len(rd):,}')
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']:
    if col not in rd.columns: continue
    r2=r2_score(rd['act'],rd[col]) if len(rd['act'])==len(rd[col]) else 0
    corr=np.corrcoef(rd['act'],rd[col])[0,1] if len(rd['act'])==len(rd[col]) and np.std(rd[col])>1e-12 and np.std(rd['act'])>1e-12 else 0
    da=((rd[col]>0)==(rd['act']>0)).mean()
    print(f'{col:12s} R²={r2:+.4f}  Corr={corr:+.4f}  DirAcc={da:.1%}')

# ─── Backtest ───
print(f'\n{"="*50}')
print(f'BACKTEST (open-close PnL)')
print(f'{"="*50}')
rd['win']=(rd['act_open']>0).astype(int)
rd=rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm']=rd['dt'].dt.normalize()

meta_p=[]; mf=['avg','xgb','lgb','cb','stack']
for d in sorted(rd['dt_norm'].unique()):
    past=rd[rd['dt_norm']<d]; today=rd[rd['dt_norm']==d]
    if len(past)<500 or len(today)<5:
        meta_p.extend(today['avg'].clip(0,1).tolist()); continue
    s=StandardScaler(); clf=LogisticRegression(random_state=42,max_iter=500,C=1.0)
    clf.fit(s.fit_transform(past[mf].values),past['win'].values)
    meta_p.extend(clf.predict_proba(s.transform(today[mf].values))[:,1].tolist())
rd['mc']=meta_p

strategy_cols=['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']
dates=sorted(rd['dt_norm'].unique())
prev_models={}; prev_picks={}; bt=[]

def compute_turnover(prev_set, curr_set):
    if prev_set is None: return 1.0
    if not prev_set or not curr_set: return 1.0
    ch=len(curr_set-prev_set)+len(prev_set-curr_set)
    return ch/max(len(curr_set|prev_set),1)

for d in dates:
    day=rd[rd['dt_norm']==d]
    if len(day)<5: continue
    picks={}
    for col in strategy_cols:
        picks[col]=day.sort_values(col,ascending=False).iloc[0]['sym']
    ranked=day.sort_values('stack',ascending=False)['sym'].tolist()
    t1_pick=ranked[0]; t3_picks=set(ranked[:3])
    t5_picks=set(ranked[:5]); t10_picks=set(ranked[:10])
    t3mc=[s for s in ranked if day[day['sym']==s]['mc'].values[0]>=0.5][:3]
    if not t3mc: t3mc=[t1_pick]
    t3m_picks=set(t3mc)

    def pick_ret(syms):
        r=day[day['sym'].isin(syms)]['act_open'].values
        return np.mean(r) if len(r)>0 else day.iloc[0]['act_open']

    t1_ret=pick_ret({t1_pick}); t3_ret=pick_ret(t3_picks)
    t5_ret=pick_ret(t5_picks); t10_ret=pick_ret(t10_picks); t3m_ret=pick_ret(t3m_picks)
    model_rets={}; model_tos={}
    for col in strategy_cols:
        mr=day[day['sym']==picks[col]]['act_open'].values
        model_rets[col]=mr[0] if len(mr)>0 else 0.0
        pm=prev_models.get(col)
        model_tos[col]=0.0 if pm is not None and picks[col]==pm else 1.0
        prev_models[col]=picks[col]
    t1_to=1.0; t3_to=1.0; t5_to=1.0; t10_to=1.0; t3m_to=1.0
    if prev_picks:
        t1_to=0.0 if t1_pick==prev_picks.get('t1') else 1.0
        t3_to=compute_turnover(prev_picks.get('t3'),t3_picks)
        t5_to=compute_turnover(prev_picks.get('t5'),t5_picks)
        t10_to=compute_turnover(prev_picks.get('t10'),t10_picks)
        t3m_to=compute_turnover(prev_picks.get('t3m'),t3m_picks)
    prev_picks={'t1':t1_pick,'t3':t3_picks,'t5':t5_picks,'t10':t10_picks,'t3m':t3m_picks}
    t1_cost=cost_rt(TOTAL_POS)*t1_to*100
    t3_cost=cost_rt(TOTAL_POS/3)*t3_to*100; t5_cost=cost_rt(TOTAL_POS/5)*t5_to*100
    t10_cost=cost_rt(TOTAL_POS/10)*t10_to*100; t3m_cost=cost_rt(TOTAL_POS/len(t3m_picks))*t3m_to*100
    rec={'d':d,'t1_ret':t1_ret,'t3_ret':t3_ret,'t5_ret':t5_ret,'t10_ret':t10_ret,'t3m_ret':t3m_ret,
         't1_to':t1_to,'t3_to':t3_to,'t5_to':t5_to,'t10_to':t10_to,'t3m_to':t3m_to,
         't1_cost':t1_cost,'t3_cost':t3_cost,'t5_cost':t5_cost,'t10_cost':t10_cost,'t3m_cost':t3m_cost,
         't3_n':len(t3_picks),'t5_n':5,'t10_n':10,'t3m_n':len(t3m_picks)}
    for col in strategy_cols:
        rec[f'{col}_ret']=model_rets[col]; rec[f'{col}_to']=model_tos[col]
    bt.append(rec)

bt=pd.DataFrame(bt)
for s in ['t1','t3','t5','t10','t3m']:
    bt[f'{s}_net']=bt[f'{s}_ret']-bt[f'{s}_cost']
for col in strategy_cols:
    bt[f'{col}_net']=bt[f'{col}_ret']-cost_rt(TOTAL_POS)*bt[f'{col}_to']*100

STRATS=[('Top-1','t1_ret','t1_net','t1_to'),('Top-3','t3_ret','t3_net','t3_to'),
         ('Top-5','t5_ret','t5_net','t5_to'),('Top-10','t10_ret','t10_net','t10_to'),
         ('Top-3+Meta','t3m_ret','t3m_net','t3m_to')]
print(f'\nBacktest days: {len(bt)}')
for sn,_,_,tc in STRATS:
    print(f'Avg {sn} turnover: {bt[tc].mean():.1%}', end='  ')
print(f'\n{"Strategy":20s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
print(f'{"-"*84}')
for sn,rc,nc,tc in STRATS:
    g=bt[rc].dropna(); n=bt[nc].dropna()
    if len(g)<10: continue
    gc,gs,gw,gdd=calc_metrics(g)
    nac,nas,naw,nadd=calc_metrics(n)
    print(f'{sn:20s} {gc:>+11.1f}% {nac:>+11.1f}% {nas:>7.2f} {naw:>7.1f}% {nadd:>7.1f}% {bt[tc].mean():>7.1%}')

print(f'\n{"Model":12s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
print(f'{"-"*80}')
for col in strategy_cols:
    rc=f'{col}_ret'; nc=f'{col}_net'; tc=f'{col}_to'
    g=bt[rc].dropna(); n=bt[nc].dropna()
    if len(g)<10: continue
    gc,gs,gw,gdd=calc_metrics(g)
    nac,nas,naw,nadd=calc_metrics(n)
    print(f'{col:12s} {gc:>+11.1f}% {nac:>+11.1f}% {nas:>7.2f} {naw:>7.1f}% {nadd:>7.1f}% {bt[tc].mean():>7.1%}')

# ─── Close-close vs open-close ───
print(f'\n{"="*60}')
print(f'CLOSE-CLOSE vs OPEN-CLOSE comparison')
print(f'{"="*60}')
rd2 = rd.copy()
for method, col in [('CLOSE-CLOSE','act'), ('OPEN-CLOSE ','act_open')]:
    cc_dates = sorted(rd2['dt_norm'].unique())
    cc_bt = []
    for d in cc_dates:
        day = rd2[rd2['dt_norm']==d]
        if len(day) < 5: continue
        ranked = day.sort_values('stack',ascending=False)
        pick = ranked.iloc[0]
        r = pick[col]
        cc_bt.append(r)
    cc_bt = np.array(cc_bt)
    cagr = (1+cc_bt/100).prod()**(252/len(cc_bt))-1
    sh = cc_bt.mean()/cc_bt.std()*np.sqrt(252) if cc_bt.std()>0 else 0
    da = ((rd2['stack']>0)==(rd2[col]>0)).mean()
    print(f'  {method}: Gross CAGR={cagr*100:+8.1f}% Sharpe={sh:.2f} DirAcc={da:.1%} mean={cc_bt.mean():+.4f}%')
del rd2

# ─── Save ───
output = {'bt':bt,'rd':rd,'features':ALL_F,'fi':fi_over_time,'cps':CPS,
          'n_symbols':df_clean['symbol'].nunique(),'n_rows':len(df_clean),'time':(datetime.now()-t0).total_seconds()}
with open(OUT/'results_v6.pkl','wb') as f: pickle.dump(output,f)
bt.to_csv(OUT/'backtest_v6.csv',index=False)
rd.to_csv(OUT/'predictions_v6.csv',index=False)
torch.save(all_models_dict, OUT/'models_v6.pth')
print(f'\nSaved to {OUT}')
print(f'Total time: {(datetime.now()-t0).total_seconds():.0f}s')
