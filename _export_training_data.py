"""Export training-ready DataFrame from warehouse to pickle."""
import duckdb, pandas as pd, numpy as np, warnings, sys, os
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_v7'
OUT.mkdir(exist_ok=True)

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

con = duckdb.connect(str(DB), config={'access_mode': 'READ_ONLY'})
core_cols = ','.join(f'"{f}"' for f in (BASE_F + EXTRA_F))
print('Loading feature_store...')
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

print('Loading VIX...')
v = con.execute('SELECT datetime,vix_close,vix_change,vix_range,vix_ma_5,vix_ma_20,vix_zscore_20 FROM vix_data ORDER BY datetime').fetchdf()
vd=pd.to_datetime(v['datetime']); v['datetime']=(vd.dt.tz_localize(None).astype('datetime64[us]') if vd.dt.tz is not None else vd.astype('datetime64[us]'))
v['vix_ma_5_r']=v['vix_close']/v['vix_ma_5'].replace(0,np.nan)-1; v['vix_ma_20_r']=v['vix_close']/v['vix_ma_20'].replace(0,np.nan)-1
v['vix_high_r']=0.0; v=v.fillna(0)
df=pd.merge_asof(df.sort_values('datetime'),v.sort_values('datetime'),on='datetime',direction='backward')

print('Loading delivery data...')
dv=con.execute('SELECT symbol,date,delivery_pct FROM delivery_data ORDER BY symbol,date').fetchdf()
dv['date']=pd.to_datetime(dv['date']).astype('datetime64[us]')
dv['delivery_pct_ma5']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(5,min_periods=2).mean())
dv['delivery_pct_ma20']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(20,min_periods=5).mean())
dv['delivery_delta']=dv['delivery_pct']-dv['delivery_pct_ma5']; dv=dv.fillna(0)
dv=dv.drop_duplicates(subset=['symbol','date'])
df['date_m']=pd.to_datetime(df['datetime']).dt.normalize()
df=df.merge(dv,left_on=['symbol','date_m'],right_on=['symbol','date'],how='left')
for c in DV_F: df[c]=df[c].fillna(0)

print('Loading MTF (60min) features...')
m=con.execute("SELECT symbol,datetime,high,low,close,rsi_14,bb_width,macd_hist FROM feature_store WHERE timeframe='60min' ORDER BY datetime").fetchdf()
if len(m) > 0:
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
else:
    for c in MTF_F: df[c]=0.0
    print('WARNING: No 60min data for MTF')

print('Loading market structure...')
ms=con.execute("SELECT symbol,datetime,rs_vs_market,rs_vs_sector,rs_ratio_market,rs_ratio_sector,rs_momentum_10,rs_momentum_20 FROM market_structure WHERE timeframe='1day' ORDER BY datetime").fetchdf()
msd=pd.to_datetime(ms['datetime']); ms['datetime']=(msd.dt.tz_localize(None).astype('datetime64[us]') if msd.dt.tz is not None else msd.astype('datetime64[us]'))
ms=ms.drop_duplicates(subset=['symbol','datetime'])
df=df.merge(ms,on=['symbol','datetime'],how='left')
for c in RS_F: df[c]=df[c].fillna(0)

print('Computing cross-sectional ranks...')
df = df.sort_values(['datetime','symbol']).reset_index(drop=True)
for feat in RANK_FEATS:
    if feat in df.columns:
        df[f'rank_{feat}'] = df.groupby('datetime')[feat].rank(pct=True).fillna(0.5)
CS_FEATS = [f'rank_{f}' for f in RANK_FEATS if f'rank_{f}' in df.columns]

print('Loading breadth data...')
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

print('Loading market regimes...')
reg=con.execute('SELECT datetime,regime_label,regime_id FROM market_regimes ORDER BY datetime').fetchdf()
rdt=pd.to_datetime(reg['datetime']); reg['datetime']=(rdt.dt.tz_localize(None).astype('datetime64[us]') if rdt.dt.tz is not None else rdt.astype('datetime64[us]'))
df=df.merge(reg,on='datetime',how='left'); df['regime_label']=df['regime_label'].fillna('sideways')
df['regime_id']=df['regime_id'].fillna(0).astype(int)
con.close()

print('Computing targets...')
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

print('Imputing NaN per-symbol...')
pre_nan=df[ALL_F].isna().sum().sum()
if pre_nan>0:
    nan_feats=df[ALL_F].isna().any()
    print(f'Imputing {pre_nan:,} NaN values across {nan_feats.sum()} features per-symbol...')
    for c in ALL_F:
        if df[c].isna().any():
            df[c]=df.groupby('symbol',group_keys=False)[c].transform(lambda s:s.ffill().bfill().fillna(0))
    post_nan=df[ALL_F].isna().sum().sum()
    if post_nan>0:
        print(f'  {post_nan} NaN remaining (fallback fill=0)')
        df[ALL_F]=df[ALL_F].fillna(0)

df_clean=df.dropna(subset=[RET_COL]).copy()
print(f'Features: {len(ALL_F)}, Rows: {len(df_clean):,}, Symbols: {df_clean["symbol"].nunique()}')

data = {'df': df_clean, 'ALL_F': ALL_F, 'RET_COL': RET_COL}
import pickle
with open(OUT/'training_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print(f'Saved to training_data.pkl ({os.path.getsize(OUT/"training_data.pkl")/1e9:.2f}GB)')
