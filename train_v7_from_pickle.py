"""v7: Train from exported pickle (446 symbols, no DB dependency)"""
import pandas as pd, numpy as np, warnings, pickle, os
import xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import optuna, shap
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'return_prediction_report_v7')
os.makedirs(OUT, exist_ok=True)
t0 = datetime.now()

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000
def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

CPS = cost_rt(TOTAL_POS)
print(f'Cost/stock: {CPS*100:.3f}%')

# Load pre-exported training data
print('\n=== Loading training data ===')
with open(os.path.join(OUT, 'training_data.pkl'), 'rb') as f:
    data = pickle.load(f)
df_clean = data['df']
ALL_F = data['ALL_F']
RET_COL = data['RET_COL']
del data
print(f'Loaded: {len(df_clean):,} rows, {df_clean["symbol"].nunique()} symbols, {len(ALL_F)} features')

# Walkforward
years = sorted(df_clean['year'].unique())
windows = [(years[:i], years[i]) for i in range(2, len(years))]
print(f'\n{len(windows)} walkforward windows')

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
    st.optimize(obj,n_trials=5,show_progress_bar=False)
    best_hp=st.best_params
    xgb_hp={k.replace('lr','learning_rate').replace('n','n_estimators').replace('d','max_depth').replace('ss','subsample').replace('cs','colsample_bytree').replace('al','reg_alpha').replace('la','reg_lambda'):v for k,v in best_hp.items()}

    ss=StandardScaler(); Xs=ss.fit_transform(train[valid2].values)
    ms=xgb.XGBRegressor(n_estimators=100,max_depth=4,random_state=42,n_jobs=-1,verbosity=0)
    ms.fit(Xs,train[RET_COL].values)
    try:
        e = shap.TreeExplainer(ms)
        sv = e.shap_values(Xs[:min(2000,len(Xs))])
        si = dict(zip(valid2, np.abs(sv).mean(axis=0)))
        sf=sorted(si.items(),key=lambda x:-x[1])
        ci=np.cumsum([s[1] for s in sf])
        if ci[-1] <= 0: tf=valid2[:min(30,len(valid2))]
        else: tf=[s[0] for s in sf[:max(np.searchsorted(ci,ci[-1]*0.8)+1,15)]]
        fi_over_time[test_yr]=si
    except: tf=valid2

    scaler=StandardScaler(); X_tr=scaler.fit_transform(train[tf].values); X_te=scaler.transform(test[tf].values)
    y_tr=train[RET_COL].values; y_te=test[RET_COL].values

    train['dr']=train.groupby('datetime')[RET_COL].rank('dense',ascending=True).astype(int)-1
    max_r=train['dr'].max(); train['dr_cap']=(train['dr']/(max_r/30+1)).astype(int)
    train['gid']=train.groupby('datetime').ngroup()
    y_rank=train['dr_cap'].values; grps=train.groupby('gid').size().values

    m_xgb=xgb.XGBRegressor(random_state=42,n_jobs=-1,verbosity=0,**xgb_hp).fit(X_tr,y_tr)
    m_rank=xgb.XGBRanker(n_estimators=120,max_depth=4,learning_rate=0.05,random_state=42,n_jobs=-1,verbosity=0,objective='rank:ndcg',ndcg_exp_gain=False).fit(X_tr,y_rank,group=grps)
    m_lgb=lgb.LGBMRegressor(**best_hp,random_state=42,n_jobs=-1,verbosity=-1).fit(X_tr,y_tr)
    m_lgb_r=lgb.LGBMRanker(n_estimators=100,max_depth=4,learning_rate=0.05,random_state=42,n_jobs=-1,verbosity=-1,max_position=30).fit(X_tr,y_rank,group=grps)
    m_cb=cb.CatBoostRegressor(n_estimators=best_hp['n'],learning_rate=best_hp['lr'],random_seed=42,verbose=0,allow_writing_files=False).fit(X_tr,y_tr)
    m_rf=RandomForestRegressor(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1).fit(X_tr,y_tr)
    m_et=ExtraTreesRegressor(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1).fit(X_tr,y_tr)

    p_xgb=m_xgb.predict(X_te); p_rank=m_rank.predict(X_te); p_lgb=m_lgb.predict(X_te)
    p_lgb_r=m_lgb_r.predict(X_te); p_cb=m_cb.predict(X_te); p_rf=m_rf.predict(X_te); p_et=m_et.predict(X_te)
    p_transformer=np.zeros(len(test))

    if len(all_results)>1000:
        hist=pd.DataFrame(all_results).sort_values('dt')
        meta=Ridge(alpha=1.0).fit(hist[['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer']].values,hist['act'].values)
        p_stack=meta.predict(np.column_stack([p_xgb,p_rank,p_lgb,p_lgb_r,p_cb,p_rf,p_et,p_transformer]))
    else:
        p_stack=np.mean([p_xgb,p_rank,p_lgb,p_lgb_r,p_cb,p_rf,p_et,p_transformer],axis=0)
    p_avg=np.mean([p_xgb,p_rank,p_lgb,p_lgb_r,p_cb,p_rf,p_et,p_transformer],axis=0)

    for name,p in [('XGB',p_xgb),('Ranker',p_rank),('LGB',p_lgb),('LGBR',p_lgb_r),('CatB',p_cb),('RF',p_rf),('ET',p_et),('Avg',p_avg),('Stack',p_stack)]:
        if np.isnan(p).any(): continue
        r2=r2_score(y_te,p)
        corr=np.corrcoef(p,y_te)[0,1] if np.std(p)>1e-12 and np.std(y_te)>1e-12 else 0
        da=((p>0)==(y_te>0)).mean()
        if wi==len(windows)-1:
            print(f'  {name:6s} R²={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

    all_models_dict[test_yr]={'xgb':m_xgb,'ranker':m_rank,'lgb':m_lgb,'lgb_r':m_lgb_r,'cb':m_cb,'rf':m_rf,'et':m_et,'features':tf,'scaler':scaler}
    for i in range(len(test)):
        all_results.append(dict(zip(
            ['dt','sym','act','act_open','xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack','regime','tb'],
            [test['datetime'].iloc[i],test['symbol'].iloc[i],y_te[i],
             test['fwd_open_ret_1d'].iloc[i] if 'fwd_open_ret_1d' in test.columns else y_te[i],
             p_xgb[i],p_rank[i],p_lgb[i],p_lgb_r[i],p_cb[i],p_rf[i],p_et[i],p_transformer[i],p_avg[i],p_stack[i],
             test['regime_label'].iloc[i] if 'regime_label' in test.columns else '?',
             test['triple_barrier'].iloc[i] if 'triple_barrier' in test.columns else 0])))

if len(all_results)==0: print('ERROR: no results'); exit(1)
rd=pd.DataFrame(all_results)

print(f'\n{"="*50}')
print(f'v7 PERFORMANCE (446 symbols, fixed features)')
print(f'{"="*50}')
print(f'Total predictions: {len(rd):,}')
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']:
    if col not in rd.columns: continue
    r2=r2_score(rd['act'],rd[col]) if len(rd['act'])==len(rd[col]) else 0
    corr=np.corrcoef(rd['act'],rd[col])[0,1] if len(rd['act'])==len(rd[col]) and np.std(rd[col])>1e-12 and np.std(rd['act'])>1e-12 else 0
    da=((rd[col]>0)==(rd['act']>0)).mean()
    print(f'{col:12s}  R²={r2:+.4f}  Corr={corr:+.4f}  DirAcc={da:.1%}')

# Quick top-1 backtest
print(f'\n{"="*50}')
print('QUICK TOP-1 BACKTEST (open-close)')
print(f'{"="*50}')
rd['win']=(rd['act_open']>0).astype(int)
rd=rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm']=rd['dt'].dt.normalize()
meta_p=[]; mf=['avg','xgb','lgb','cb','stack']
for d in sorted(rd['dt_norm'].unique()):
    past=rd[rd['dt_norm']<d]; today=rd[rd['dt_norm']==d]
    if len(past)<500 or len(today)<5:
        meta_p.extend(today['avg'].clip(0,1).tolist()); continue
    s=StandardScaler(); clf=LogisticRegression(C=1.0,random_state=42,max_iter=500)
    clf.fit(s.fit_transform(past[mf].values),past['win'].values)
    meta_p.extend(clf.predict_proba(s.transform(today[mf].values))[:,1].tolist())
rd['mc']=meta_p

prev={}; bt_daily=[]
for d in sorted(rd['dt_norm'].unique()):
    day=rd[rd['dt_norm']==d]; ranked=day.sort_values('stack',ascending=False)
    if len(day)<5: continue
    t1=ranked.iloc[0]; r=t1['act_open']; to=0.0 if prev.get('t1')==t1['sym'] else 1.0
    prev['t1']=t1['sym']; cost=cost_rt(TOTAL_POS)*to*100
    bt_daily.append({'ret':r,'net':r-cost,'to':to})
bt=pd.DataFrame(bt_daily)
print(f'Days: {len(bt)}')
gc,gs,gw,gdd=calc_metrics(bt['ret'])
nac,nas,naw,nadd=calc_metrics(bt['net'])
print(f'Top-1 Gross: {gc:+8.1f}% Net: {nac:+8.1f}% Sharpe: {nas:.2f} WinRate: {naw:.1f}% MaxDD: {nadd:.1f}%')

# Save
output={'rd':rd,'features':ALL_F,'fi':fi_over_time,'cps':CPS,'n_symbols':df_clean['symbol'].nunique(),
        'n_rows':len(df_clean),'time':(datetime.now()-t0).total_seconds(),'version':'v7','universe_size':446}
with open(os.path.join(OUT,'results_v7.pkl'),'wb') as f: pickle.dump(output,f)
rd.to_csv(os.path.join(OUT,'predictions_v7.csv'),index=False)
with open(os.path.join(OUT,'models_v7.pkl'),'wb') as f: pickle.dump(all_models_dict,f)
print(f'\nSaved to {OUT}')
print(f'Total time: {(datetime.now()-t0).total_seconds():.0f}s')
