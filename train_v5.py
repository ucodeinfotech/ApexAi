"""v5: Train with 141 stocks (90 original + 51 new heavyweights)"""
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
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'return_prediction_report_v5'
OUT.mkdir(exist_ok=True)
t0 = datetime.now()

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000

def cost_rt(pos_size):
    """Round-trip cost for a position of given size (decimal fraction).
    GST applies to brokerage + exchange + SEBI only (NOT STT).
    Min brokerage (Rs20) applies per trade side."""
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    total = brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
    return total

CPS = cost_rt(TOTAL_POS)  # default: single position
print(f'Cost/stock (single): {CPS*100:.3f}%')
print(f'Cost/stock (3-pos): {cost_rt(TOTAL_POS/3)*100:.3f}%')
con = duckdb.connect(str(DB))

# Feature definitions
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
print(f'{len(ALL_FEATS)} base + rank features defined')

# 1. Load base features
print('\nLoading base features...')
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

# 2. VIX
v = con.execute('SELECT datetime,vix_close,vix_change,vix_range,vix_ma_5,vix_ma_20,vix_zscore_20 FROM vix_data ORDER BY datetime').fetchdf()
vd=pd.to_datetime(v['datetime']); v['datetime']=(vd.dt.tz_localize(None).astype('datetime64[us]') if vd.dt.tz is not None else vd.astype('datetime64[us]'))
v['vix_ma_5_r']=v['vix_close']/v['vix_ma_5'].replace(0,np.nan)-1; v['vix_ma_20_r']=v['vix_close']/v['vix_ma_20'].replace(0,np.nan)-1
v['vix_high_r']=0.0; v=v.fillna(0)
df=pd.merge_asof(df.sort_values('datetime'),v.sort_values('datetime'),on='datetime',direction='backward')

# 3. Delivery
dv=con.execute('SELECT symbol,date,delivery_pct FROM delivery_data ORDER BY symbol,date').fetchdf()
dv['date']=pd.to_datetime(dv['date']).astype('datetime64[us]')
dv['delivery_pct_ma5']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(5,min_periods=2).mean())
dv['delivery_pct_ma20']=dv.groupby('symbol')['delivery_pct'].transform(lambda x:x.shift(1).rolling(20,min_periods=5).mean())
dv['delivery_delta']=dv['delivery_pct']-dv['delivery_pct_ma5']; dv=dv.fillna(0)
df['date_m']=pd.to_datetime(df['datetime']).dt.normalize()
df=df.merge(dv,left_on=['symbol','date_m'],right_on=['symbol','date'],how='left')
for c in DV_F: df[c]=df[c].fillna(0)

# 4. Intraday 60min features
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

# 5. RS features from market_structure
ms=con.execute("SELECT symbol,datetime,rs_vs_market,rs_vs_sector,rs_ratio_market,rs_ratio_sector,rs_momentum_10,rs_momentum_20 FROM market_structure WHERE timeframe='1day' ORDER BY datetime").fetchdf()
msd=pd.to_datetime(ms['datetime']); ms['datetime']=(msd.dt.tz_localize(None).astype('datetime64[us]') if msd.dt.tz is not None else msd.astype('datetime64[us]'))
df=df.merge(ms,on=['symbol','datetime'],how='left')
for c in RS_F: df[c]=df[c].fillna(0)
print('Features loaded:', len(df))

# 6. Cross-sectional ranks
df = df.sort_values(['datetime','symbol']).reset_index(drop=True)
for feat in RANK_FEATS:
    if feat in df.columns:
        df[f'rank_{feat}'] = df.groupby('datetime')[feat].rank(pct=True).fillna(0.5)
CS_FEATS = [f'rank_{f}' for f in RANK_FEATS if f'rank_{f}' in df.columns]

# 7. Breadth
breadth = con.execute("""SELECT DATE_TRUNC('day',datetime)::DATE as day, SUM(CASE WHEN close>open THEN 1 ELSE 0 END) as adv,
    SUM(CASE WHEN close<open THEN 1 ELSE 0 END) as dec, COUNT(*) as tot FROM 
    (SELECT datetime,close,open FROM raw_market WHERE timeframe='1day'
     AND symbol IN (SELECT DISTINCT symbol FROM feature_store WHERE timeframe='1day')) sub GROUP BY day""").fetchdf()
breadth['day']=pd.to_datetime(breadth['day']).astype('datetime64[us]')
breadth['adv_dec_ratio']=breadth['adv']/breadth['dec'].replace(0,1); breadth['adv_dec_diff']=breadth['adv']-breadth['dec']
breadth['brd_pct']=breadth['adv']/breadth['tot']
for c in ['adv_dec_ratio','adv_dec_diff','brd_pct']:
    breadth[f'{c}_ma5']=breadth[c].rolling(5,min_periods=2).mean(); breadth[f'{c}_ma20']=breadth[c].rolling(20,min_periods=5).mean()
BRD_F=[c for c in breadth.columns if c not in ['day']]; df=df.merge(breadth,left_on='date_m',right_on='day',how='left')
for c in BRD_F: df[c]=df[c].fillna(0)

# 8. Regimes
reg=con.execute('SELECT datetime,regime_label,regime_id FROM market_regimes ORDER BY datetime').fetchdf()
rdt=pd.to_datetime(reg['datetime']); reg['datetime']=(rdt.dt.tz_localize(None).astype('datetime64[us]') if rdt.dt.tz is not None else rdt.astype('datetime64[us]'))
df=df.merge(reg,on='datetime',how='left'); df['regime_label']=df['regime_label'].fillna('sideways')
df['regime_id']=df['regime_id'].fillna(0).astype(int)
print(f'CS: {len(CS_FEATS)} features, Breadth: {len(BRD_F)}, Regimes available')

# 9. Targets
con.close()
df=df.sort_values(['symbol','datetime']).reset_index(drop=True)
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
# Correlation filtering: remove one from each pair >0.95
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
clean_mask=df[ALL_F].notna().all(axis=1); df_clean=df[clean_mask].copy(); df_clean=df_clean.dropna(subset=[RET_COL])
print(f'Features: {len(ALL_F)}, Clean rows: {len(df_clean):,}, Symbols: {df_clean["symbol"].nunique()}')

print(f'\nPreprocessing done in {(datetime.now()-t0).total_seconds():.0f}s')
print(f'Symbols in dataset: {sorted(df_clean["symbol"].unique())[:5]}...')

# 10. Walkforward
years=sorted(df_clean['year'].unique())
windows=[(years[:i],years[i]) for i in range(2,len(years))]
print(f'\n{len(windows)} walkforward windows')
if len(windows) == 0:
    print('ERROR: <5 years of data, no walkforward windows'); exit(1)

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
    print(f'  Optuna ({len(valid2)} feats)...')
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
    print(f'  HPO R²={st.best_value:.4f}')

    # SHAP feature selection
    print(f'  SHAP...')
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

    # Train all 7 models
    scaler=StandardScaler(); X_tr=scaler.fit_transform(train[tf].values); X_te=scaler.transform(test[tf].values)
    y_tr=train[RET_COL].values; y_te=test[RET_COL].values

    train['dr']=train.groupby('datetime')[RET_COL].rank('dense',ascending=True).astype(int)-1
    max_r=train['dr'].max(); train['dr_cap']=(train['dr']/(max_r/30+1)).astype(int)
    train['gid']=train.groupby('datetime').ngroup()
    y_rank=train['dr_cap'].values; grps=train.groupby('gid').size().values

    m_xgb=xgb.XGBRegressor(random_state=42,n_jobs=-1,verbosity=0,**xgb_hp).fit(X_tr,y_tr)
    m_rank=xgb.XGBRanker(n_estimators=120,max_depth=4,learning_rate=0.05,random_state=42,n_jobs=-1,verbosity=0,objective='rank:ndcg',ndcg_exp_gain=False).fit(X_tr,y_rank,group=grps)
    m_lgb=lgb.LGBMRegressor(n_estimators=best_hp['n'],max_depth=best_hp['d'],learning_rate=best_hp['lr'],subsample=best_hp['ss'],colsample_bytree=best_hp['cs'],reg_alpha=best_hp['al'],reg_lambda=best_hp['la'],random_state=42,n_jobs=-1,verbosity=-1).fit(X_tr,y_tr)
    m_lgb_r=lgb.LGBMRanker(n_estimators=100,max_depth=4,learning_rate=0.05,random_state=42,n_jobs=-1,verbosity=-1,max_position=30).fit(X_tr,y_rank,group=grps)
    m_cb=cb.CatBoostRegressor(n_estimators=best_hp['n'],learning_rate=best_hp['lr'],random_seed=42,verbose=0,allow_writing_files=False).fit(X_tr,y_tr)
    m_rf=RandomForestRegressor(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1).fit(X_tr,y_tr)
    m_et=ExtraTreesRegressor(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1).fit(X_tr,y_tr)

    p_xgb=m_xgb.predict(X_te); p_rank=m_rank.predict(X_te); p_lgb=m_lgb.predict(X_te)
    p_lgb_r=m_lgb_r.predict(X_te); p_cb=m_cb.predict(X_te); p_rf=m_rf.predict(X_te); p_et=m_et.predict(X_te)

    if len(all_results) > 1000:
        hist = pd.DataFrame(all_results).sort_values('dt')
        meta_X = hist[['xgb','ranker','lgb','lgb_r','cb','rf','et']].values
        meta_y = hist['act'].values
        meta = Ridge(alpha=1.0, random_state=42)
        meta.fit(meta_X, meta_y)
        p_stack = meta.predict(np.column_stack([p_xgb, p_rank, p_lgb, p_lgb_r, p_cb, p_rf, p_et]))
    else:
        p_stack = np.mean([p_xgb, p_rank, p_lgb, p_lgb_r, p_cb, p_rf, p_et], axis=0)
    p_avg = np.mean([p_xgb, p_rank, p_lgb, p_lgb_r, p_cb, p_rf, p_et], axis=0)

    for name, p in [('XGB',p_xgb),('Ranker',p_rank),('LGB',p_lgb),('LGBR',p_lgb_r),
                    ('CatB',p_cb),('RF',p_rf),('ET',p_et),('Avg',p_avg),('Stack',p_stack)]:
        if np.isnan(p).any(): continue
        r2=r2_score(y_te,p)
        corr=np.corrcoef(p,y_te)[0,1] if np.std(p)>1e-12 and np.std(y_te)>1e-12 else 0
        da=((p>0)==(y_te>0)).mean()
        print(f'  {name:6s} R²={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

    all_models_dict[test_yr] = {'xgb':m_xgb,'ranker':m_rank,'lgb':m_lgb,'lgb_r':m_lgb_r,'cb':m_cb,'rf':m_rf,'et':m_et,'features':tf,'scaler':scaler}
    for i in range(len(test)):
        all_results.append(dict(zip(
            ['dt','sym','act','act_open','xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack','regime','tb'],
            [test['datetime'].iloc[i],test['symbol'].iloc[i],y_te[i],
             test['fwd_open_ret_1d'].iloc[i] if 'fwd_open_ret_1d' in test.columns else y_te[i],
             p_xgb[i],p_rank[i],p_lgb[i],p_lgb_r[i],p_cb[i],p_rf[i],p_et[i],p_avg[i],p_stack[i],
             test['regime_label'].iloc[i] if 'regime_label' in test.columns else '?',
             test['triple_barrier'].iloc[i] if 'triple_barrier' in test.columns else 0])))

if len(all_results) == 0:
    print('ERROR: no results generated from walkforward'); exit(1)
rd = pd.DataFrame(all_results)
rd.to_pickle(OUT/'results_raw.pkl')

# 11. Overall performance report
print(f'\n{"="*50}')
print(f'FINAL PERFORMANCE — v5 (141 stocks)')
print(f'{"="*50}')
print(f'Time elapsed: {(datetime.now()-t0).total_seconds():.0f}s')
print(f'Total predictions: {len(rd):,}')
print(f'Models: {len(all_models_dict)} years')
print(f'{"Model":8s} {"R²":>8s} {"Corr":>8s} {"DirAcc":>8s}')
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']:
    if col not in rd.columns: continue
    r2=r2_score(rd['act'],rd[col])
    corr=np.corrcoef(rd['act'],rd[col])[0,1] if np.std(rd[col])>1e-12 and np.std(rd['act'])>1e-12 else 0
    da=((rd[col]>0)==(rd['act']>0)).mean()
    print(f'{col:8s} {r2:+.4f}  {corr:+.4f}  {da:.1%}')

# 12. Turnover-aware backtest with position tracking
print(f'\n{"="*50}')
print(f'BACKTEST: position tracking + turnover-aware costs [OPEN-CLOSE PnL]')
print(f'{"="*50}')
rd['win']=(rd['act_open']>0).astype(int)
rd=rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm']=rd['dt'].dt.normalize()

# Meta-classifier for win probability
meta_p=[]; mf=['avg','xgb','lgb','cb','stack']
for d in sorted(rd['dt_norm'].unique()):
    past=rd[rd['dt_norm']<d]; today=rd[rd['dt_norm']==d]
    if len(past)<500 or len(today)<5:
        # Fallback: use ensemble avg direction prob instead of hard 0.5
        meta_p.extend(today['avg'].clip(0,1).tolist()); continue
    s=StandardScaler(); clf=LogisticRegression(random_state=42,max_iter=500,C=1.0)
    clf.fit(s.fit_transform(past[mf].values),past['win'].values)
    meta_p.extend(clf.predict_proba(s.transform(today[mf].values))[:,1].tolist())
rd['mc']=meta_p

strategy_cols=['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']
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

# Report
STRATS=[('Top-1','t1_ret','t1_net','t1_to'),('Top-3','t3_ret','t3_net','t3_to'),
         ('Top-5','t5_ret','t5_net','t5_to'),('Top-10','t10_ret','t10_net','t10_to'),
         ('Top-3+Meta','t3m_ret','t3m_net','t3m_to')]
print(f'\nBacktest days: {len(bt)}')
for sn,_,_,tc in STRATS:
    print(f'Avg {sn} turnover: {bt[tc].mean():.1%}', end='  ')
print('\n')
print(f'{"Strategy":20s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
print(f'{"-"*84}')
for sn,rc,nc,tc in STRATS:
    g=bt[rc].dropna(); n=bt[nc].dropna()
    if len(g)<10: continue
    gc,gs,gw,gdd=calc_metrics(g)
    nac,nas,naw,nadd=calc_metrics(n)
    print(f'{sn:20s} {gc:>+11.1f}% {nac:>+11.1f}% {nas:>7.2f} {naw:>7.1f}% {nadd:>7.1f}% {bt[tc].mean():>7.1%}')

print(f'\n{"Model":10s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
print(f'{"-"*74}')
for col in strategy_cols:
    rc=f'{col}_ret'; nc=f'{col}_net'; tc=f'{col}_to'
    g=bt[rc].dropna(); n=bt[nc].dropna()
    if len(g)<10: continue
    gc,gs,gw,gdd=calc_metrics(g)
    nac,nas,naw,nadd=calc_metrics(n)
    print(f'{col:10s} {gc:>+11.1f}% {nac:>+11.1f}% {nas:>7.2f} {naw:>7.1f}% {nadd:>7.1f}% {bt[tc].mean():>7.1%}')

# --- Close-close vs open-close comparison ---
print(f'\n{"="*60}')
print(f'CLOSE-CLOSE vs OPEN-CLOSE comparison (Top-1 gross)')
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

# 13. Institutional validation
# --- Deflated Sharpe Ratio (all portfolio strategies) ---
port_sc=[f'{s}_net' for s in ['t1','t3','t5','t10','t3m']]
M_port=len(port_sc)
for ps in port_sc:
    sr_=bt[ps].mean()/bt[ps].std()*np.sqrt(252) if bt[ps].std()>0 else 0
    T_obs=len(bt); em=math.sqrt(2*math.log(M_port))
    num=sr_*math.sqrt(T_obs-1)-em; den=math.sqrt(1+(3-1)/4*sr_**2) if sr_>0 else 1
    dsr=stats.norm.cdf(num/den) if den>0 else 0
    print(f'\n{ps:10s} Sharpe={sr_:.2f}  DSR={dsr:.4f} (M={M_port})')

# --- White's Reality Check with block bootstrap ---
all_sc=[f'{c}_net' for c in strategy_cols]+port_sc
all_returns=bt[all_sc].values
T_obs,M_strat=all_returns.shape
mr=all_returns.mean(axis=0); sr_=all_returns.std(axis=0)
sr_[sr_<1e-12]=1e-12
t_stats=np.sqrt(T_obs)*mr/sr_
V_obs=t_stats.max(); best_s=all_sc[np.argmax(t_stats)]

# Block bootstrap (block size = 21 trading days ~ 1 month)
np.random.seed(42)
block_size=21; n_blocks=int(np.ceil(T_obs/block_size))
boot_max=np.zeros(5000)
for b in range(5000):
    bs_returns=np.zeros((T_obs,M_strat))
    for bi in range(n_blocks):
        bi_start=np.random.randint(0,max(1,T_obs-block_size))
        bi_end=min(bi_start+block_size,T_obs)
        blen=bi_end-bi_start
        if bi*block_size+blen<=T_obs:
            bs_returns[bi*block_size:bi*block_size+blen]=all_returns[bi_start:bi_end]
    bm=bs_returns.mean(axis=0); bt_=np.sqrt(T_obs)*bm/sr_
    boot_max[b]=bt_.max()
p_wrc=(boot_max>=V_obs).mean()
print(f'\nWhite RC (block bootstrap, block=21): p={p_wrc:.4f} (best={best_s}, t={V_obs:.2f})')

# --- Slippage sensitivity ---
print(f'\n{"="*60}')
print(f'SLIPPAGE SENSITIVITY')
print(f'{"="*60}')
for label,sv,n_pos,desc in [('Top-1 5bp',0.0005,1,''),('Top-1 10bp',0.0010,1,''),
    ('Top-1 20bp',0.0020,1,''),('Top-1 30bp',0.0030,1,''),
    ('Top-5 5bp',0.0005,5,''),('Top-5 10bp',0.0010,5,''),
    ('Top-5 20bp',0.0020,5,''),('Top-5 30bp',0.0030,5,''),
    ('Top-10 5bp',0.0005,10,''),('Top-10 10bp',0.0010,10,''),
    ('Top-10 20bp',0.0020,10,''),('Top-10 30bp',0.0030,10,'')]:
    ps=TOTAL_POS/n_pos; bs=max(BRK*ps,MIN_BRK)/ps; bt2=bs*2; gb=bt2+EXCH*2+SEBI*2
    cost_r=bt2+STT+EXCH*2+SEBI*2+STAMP+gb*GST+sv*2
    ncol=f't{n_pos}_' if n_pos>1 else 't1_'
    tcost=bt[f'{ncol}to'] if n_pos>1 else bt['t1_to']
    net_=bt[f'{ncol}ret']-tcost*cost_r*100
    cagr_=((1+net_/100).prod()**(252/len(net_))-1)*100; sh_=net_.mean()/net_.std()*math.sqrt(252) if net_.std()>0 else 0
    print(f'  {label:15s}: Net CAGR={cagr_:+.1f}%  Sharpe={sh_:.2f}')

# --- Multi-day holding backtest ---
print(f'\n{"="*60}')
print(f'MULTI-DAY HOLDING BACKTEST (stack-based)')
print(f'{"="*60}')
for HOLD in [1,3,5,10]:
    hold_bt=[]; cur5=None; cur10=None; cur3m=None
    for di,d in enumerate(dates):
        day=rd[rd['dt_norm']==d]
        if len(day)<5: continue
        ranked=day.sort_values('stack',ascending=False)['sym'].tolist()
        if HOLD==1 or di % HOLD == 0:
            new5=set(ranked[:5]); new10=set(ranked[:10])
            new3m=set([s for s in ranked if day[day['sym']==s]['mc'].values[0]>=0.5][:3])
            if not new3m: new3m={ranked[0]}
        else:
            new5=cur5 if cur5 is not None else set(ranked[:5])
            new10=cur10 if cur10 is not None else set(ranked[:10])
            new3m=cur3m if cur3m is not None else {ranked[0]}
        to5=compute_turnover(cur5,new5); to10=compute_turnover(cur10,new10); to3m=compute_turnover(cur3m,new3m)
        r5=day[day['sym'].isin(new5)]['act_open'].mean() if new5 else 0
        r10=day[day['sym'].isin(new10)]['act_open'].mean() if new10 else 0
        r3m=day[day['sym'].isin(new3m)]['act_open'].mean() if new3m else 0
        c5=cost_rt(TOTAL_POS/5)*to5*100; c10=cost_rt(TOTAL_POS/10)*to10*100; c3m=cost_rt(TOTAL_POS/len(new3m))*to3m*100
        hold_bt.append({'d':d,'r5':r5,'r10':r10,'r3m':r3m,'to5':to5,'to10':to10,'to3m':to3m,'c5':c5,'c10':c10,'c3m':c3m})
        cur5=new5; cur10=new10; cur3m=new3m
    hbt=pd.DataFrame(hold_bt)
    for s in ['5','10','3m']:
        hbt[f'n{s}']=hbt[f'r{s}']-hbt[f'c{s}']
    n5c,_,_,n5dd=[calc_metrics(hbt['n5'].dropna())[i] for i in range(4)]
    n10c,_,_,n10dd=[calc_metrics(hbt['n10'].dropna())[i] for i in range(4)]
    n3c,_,_,n3dd=[calc_metrics(hbt['n3m'].dropna())[i] for i in range(4)]
    print(f'  Hold {HOLD:2d}d: Top-5 Net CAGR={n5c:>+8.1f}% MaxDD={n5dd:>6.1f}% TO={hbt["to5"].mean():.0%}  '
          f'Top-10 Net CAGR={n10c:>+8.1f}% TO={hbt["to10"].mean():.0%}  '
          f'Top-3+Meta Net CAGR={n3c:>+8.1f}% TO={hbt["to3m"].mean():.0%}')

# 14. Save
output = {'bt':bt,'rd':rd,'models':all_models_dict,'features':ALL_F,'fi':fi_over_time,'cps':CPS,
          'cost_rt': cost_rt, 'total_pos': TOTAL_POS,
          'n_symbols':df_clean['symbol'].nunique(),'n_rows':len(df_clean),'time':(datetime.now()-t0).total_seconds()}
with open(OUT/'results_v5.pkl','wb') as f: pickle.dump(output,f)
bt.to_csv(OUT/'backtest_v5.csv',index=False)
rd.to_csv(OUT/'predictions_v5.csv',index=False)
print(f'\nSaved to {OUT}')
print(f'Total time: {(datetime.now()-t0).total_seconds():.0f}s')
