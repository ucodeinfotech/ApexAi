import duckdb, pandas as pd, numpy as np, pickle, sys, time
sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
BASE = r'C:\Users\pc\Downloads\stock hist data'
con = duckdb.connect(BASE + r'\warehouse\market_data.duckdb')

# Exact same feature definitions as train_v6.py
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
df = con.execute(f"SELECT symbol,datetime,{core_cols},open,high,low,close,volume FROM feature_store WHERE timeframe='1day' ORDER BY datetime").fetchdf()
ds = pd.to_datetime(df['datetime'])
df['datetime'] = (ds.dt.tz_localize(None).astype('datetime64[us]') if ds.dt.tz is not None else ds.astype('datetime64[us]'))
df['range_pct'] = (df['high']-df['low'])/df['close']*100
dc = pd.to_datetime(df['datetime'])
df['year']=dc.dt.year; df['dow']=dc.dt.dayofweek; df['month']=dc.dt.month
df['is_month_end']=dc.dt.is_month_end.astype(int); df['is_quarter_end']=dc.dt.is_quarter_end.astype(int)
df['is_thursday']=(df['dow']==3).astype(int)
print(f'Base: {len(df):,} rows from {df["symbol"].nunique()} symbols')

# VIX
v = con.execute('SELECT datetime,vix_close,vix_change,vix_range,vix_ma_5,vix_ma_20,vix_zscore_20 FROM vix_data ORDER BY datetime').fetchdf()
vd=pd.to_datetime(v['datetime']); v['datetime']=(vd.dt.tz_localize(None).astype('datetime64[us]') if vd.dt.tz is not None else vd.astype('datetime64[us]'))
v['vix_ma_5_r']=v['vix_close']/v['vix_ma_5'].replace(0,np.nan)-1; v['vix_ma_20_r']=v['vix_close']/v['vix_ma_20'].replace(0,np.nan)-1
v['vix_high_r']=0.0; v=v.fillna(0)
df=pd.merge_asof(df.sort_values('datetime'),v.sort_values('datetime'),on='datetime',direction='backward')

# Delivery
dv=con.execute('SELECT symbol,date,delivery_pct FROM delivery_data ORDER BY symbol,date').fetchdf()
dv['date']=pd.to_datetime(dv['date']).astype('datetime64[us]')
dv['delivery_pct_ma5']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(5,min_periods=2).mean())
dv['delivery_pct_ma20']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(20,min_periods=5).mean())
dv['delivery_delta']=dv['delivery_pct']-dv['delivery_pct_ma5']; dv=dv.fillna(0)
df['date_m']=pd.to_datetime(df['datetime']).dt.normalize()
df=df.merge(dv,left_on=['symbol','date_m'],right_on=['symbol','date'],how='left')
for c in DV_F: df[c]=df[c].fillna(0)

# MTF 60min
m=con.execute("SELECT symbol,datetime,high,low,close,rsi_14,bb_width,macd_hist FROM feature_store WHERE timeframe='60min' ORDER BY datetime").fetchdf()
md=pd.to_datetime(m['datetime']); m['datetime']=(md.dt.tz_localize(None).astype('datetime64[us]') if md.dt.tz is not None else md.astype('datetime64[us]'))
m['date']=pd.to_datetime(m['datetime']).dt.normalize(); m['r']=(m['high']-m['low'])/m['close']*100
mtf=m.groupby(['symbol','date']).agg(intra_rsi_mean=('rsi_14','mean'),intra_rsi_std=('rsi_14','std'),
    intra_vol_std=('close',lambda x:float(np.std(np.diff(x.values))/(np.mean(x)+1e-12)*100) if len(x)>1 else 0),
    intra_range_sum=('r','sum'),intra_range_max=('r','max'),intra_bb_width_mean=('bb_width','mean'),
    intra_macd_std=('macd_hist','std')).reset_index()
for c in ['intra_rsi_mean','intra_range_sum','intra_vol_std']:
    mtf[f'{c}_ma5']=mtf.groupby('symbol')[c].transform(lambda x:x.rolling(5,min_periods=2).mean())
df=df.merge(mtf,left_on=['symbol','date_m'],right_on=['symbol','date'],how='left')
for c in MTF_F: df[c]=df[c].fillna(0)

# RS
ms=con.execute("SELECT symbol,datetime,rs_vs_market,rs_vs_sector,rs_ratio_market,rs_ratio_sector,rs_momentum_10,rs_momentum_20 FROM market_structure WHERE timeframe='1day' ORDER BY datetime").fetchdf()
msd=pd.to_datetime(ms['datetime']); ms['datetime']=(msd.dt.tz_localize(None).astype('datetime64[us]') if msd.dt.tz is not None else msd.astype('datetime64[us]'))
df=df.merge(ms,on=['symbol','datetime'],how='left')
for c in RS_F: df[c]=df[c].fillna(0)

# Cross-sectional ranks
df = df.sort_values(['datetime','symbol']).reset_index(drop=True)
print(f'After all merges: {len(df):,} rows')
for feat in RANK_FEATS:
    if feat in df.columns:
        df[f'rank_{feat}'] = df.groupby('datetime')[feat].rank(pct=True).fillna(0.5)
CS_FEATS = [f'rank_{f}' for f in RANK_FEATS if f'rank_{f}' in df.columns]
print(f'Rank features: {len(CS_FEATS)}')

# Breadth
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
print(f'After breadth merge: {len(df):,} rows')

# Regimes
reg=con.execute('SELECT datetime,regime_label,regime_id FROM market_regimes ORDER BY datetime').fetchdf()
rdt=pd.to_datetime(reg['datetime']); reg['datetime']=(rdt.dt.tz_localize(None).astype('datetime64[us]') if rdt.dt.tz is not None else rdt.astype('datetime64[us]'))
df=df.merge(reg,on='datetime',how='left'); df['regime_label']=df['regime_label'].fillna('sideways')
df['regime_id']=df['regime_id'].fillna(0).astype(int)
print(f'After regime merge: {len(df):,} rows')
con.close()

# Target
ng=df.groupby('symbol')
RET_COL='fwd_return_1d'
df[RET_COL]=(ng['close'].shift(-1)/df['close']-1)*100
df['fwd_open_ret_1d'] = (ng['close'].shift(-1) / ng['open'].shift(-1) - 1) * 100
rl=df[RET_COL].quantile(0.005); ru=df[RET_COL].quantile(0.995); df[RET_COL]=df[RET_COL].clip(rl,ru)

# Assemble ALL_F
ALL_F=[f for f in ALL_FEATS if f in df.columns]+CS_FEATS+BRD_F
print(f'ALL_F count: {len(ALL_F)}')

# Check NaN counts
na_counts = df[ALL_F].isna().sum()
na_cols = na_counts[na_counts > 0].sort_values(ascending=False)
print(f'\nALL_F NaN counts (non-zero only):')
for c in na_cols.index:
    print(f"  {c}: {na_cols[c]:,} NaN ({na_cols[c]/len(df)*100:.1f}%)")

# Final clean
clean_mask=df[ALL_F].notna().all(axis=1)
df_clean=df[clean_mask].copy(); df_clean=df_clean.dropna(subset=[RET_COL])
print(f'\nClean rows: {len(df_clean):,}')
print(f'Clean year distribution:')
yrs = df_clean['year'].value_counts().sort_index()
for y, c in yrs.items():
    print(f"  {int(y)}: {c:,}")

# Now check walkforward windows
years=sorted(df_clean['year'].unique())
print(f'\nYears: {years}')
print(f'Range(2, {len(years)}) = {list(range(2, len(years)))}')
for i in range(2, len(years)):
    ty = years[:i]
    test_yr = years[i]
    test = df_clean[df_clean['year']==test_yr]
    embargo = test['datetime'].min()-pd.Timedelta(days=7)
    train = df_clean[df_clean['year'].isin(ty)][lambda x: x['datetime'] < embargo]
    print(f'  i={i}: Train={len(train):,} (years {ty[0]}-{ty[-1]}), Test={test_yr} ({len(test):,} rows)')
