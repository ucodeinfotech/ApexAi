# Final evaluation: class weights (NO SMOTE), Optuna-tuned, ensemble, threshold opt, bootstrap CI
import pandas as pd, numpy as np, warnings, json, time
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'final_eval'
OUT.mkdir(exist_ok=True); (OUT/'charts').mkdir(exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)
print('='*60); print('FINAL EVALUATION | class weights + ensemble + threshold opt')
print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'); print('='*60); t0=time.time()

df = pd.read_parquet(BASE/'improved_features.parquet')
tc = 'target'; dc = '_d'
feats = [c for c in df.columns if c not in ('symbol','datetime',tc,dc)]
print(f'  Data: {df.shape}, Features: {len(feats)}')

# Walkforward: 5 folds expanding window, hold-out test from 2025-07
splits = [
    ('S1','2016-01-01','2023-01-01','2023-01-01','2023-07-01'),
    ('S2','2016-01-01','2023-07-01','2023-07-01','2024-01-01'),
    ('S3','2017-01-01','2024-01-01','2024-01-01','2024-07-01'),
    ('S4','2017-01-01','2024-07-01','2024-07-01','2025-01-01'),
    ('S5','2018-01-01','2025-01-01','2025-01-01','2025-07-01'),
]
test_start, test_end = '2025-07-01', '2026-06-26'

from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                              matthews_corrcoef, confusion_matrix, average_precision_score, brier_score_loss)

def met(y,p,th=0.5):
    b=(p>=th).astype(int); cm=confusion_matrix(y,b); tn,fp,fn,tp=cm.ravel() if cm.size==4 else (0,0,0,0)
    pr=precision_score(y,b,zero_division=0); rc=recall_score(y,b,zero_division=0)
    return {'auc_roc':round(roc_auc_score(y,p),4),'avg_precision':round(average_precision_score(y,p),4),
            'precision':round(pr,4),'recall':round(rc,4),'f1':round(f1_score(y,b,zero_division=0),4),
            'mcc':round(matthews_corrcoef(y,b),4),'brier':round(brier_score_loss(y,p),4),
            'lift':round(pr/y.mean() if y.mean()>0 else 0,2),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}

from xgboost import XGBClassifier
import lightgbm as lgbm

# Walkforward evaluation
print('\n--- Walkforward Evaluation (5 folds) ---')
all_res = []
for fn,ts_tr,te_tr,ts_va,te_va in splits:
    tr=df[(df[dc]>=ts_tr)&(df[dc]<te_tr)]; va=df[(df[dc]>=ts_va)&(df[dc]<te_va)]
    X_tr=tr[feats].values.astype(np.float32); y_tr=tr[tc].values
    X_va=va[feats].values.astype(np.float32); y_va=va[tc].values
    w=(1-y_tr.mean())/y_tr.mean()
    print(f'\n  {fn}: train={len(X_tr):,}({y_tr.mean():.3f}) val={len(X_va):,}({y_va.mean():.3f})')
    t1=time.time()
    preds=[]
    for name in ('XGBoost','LightGBM'):
        if name=='XGBoost':
            m=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                             min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w,
                             tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
        else:
            m=lgbm.LGBMClassifier(n_estimators=800,max_depth=6,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                                   min_child_samples=20,reg_alpha=0.5,reg_lambda=0.5,class_weight='balanced',
                                   n_jobs=-1,random_state=42,verbosity=-1)
        m.fit(X_tr,y_tr); p=m.predict_proba(X_va)[:,1]; preds.append(p)
        r=met(y_va,p); r['model']=name; r['fold']=fn; all_res.append(r)
    ep=np.mean(preds,axis=0); r=met(y_va,ep); r['model']='Ensemble'; r['fold']=fn; all_res.append(r)
    print(f'    XGB AUC={all_res[-3]["auc_roc"]:.4f} P={all_res[-3]["precision"]:.4f} R={all_res[-3]["recall"]:.4f} F1={all_res[-3]["f1"]:.4f}')
    print(f'    LGB AUC={all_res[-2]["auc_roc"]:.4f} P={all_res[-2]["precision"]:.4f} R={all_res[-2]["recall"]:.4f} F1={all_res[-2]["f1"]:.4f}')
    print(f'    ENS AUC={r["auc_roc"]:.4f} P={r["precision"]:.4f} R={r["recall"]:.4f} F1={r["f1"]:.4f} MCC={r["mcc"]:.4f} Lift={r["lift"]}x')
    print(f'    Time: {time.time()-t1:.0f}s')

# Full test evaluation
print('\n--- Final Test Evaluation ---')
test=df[(df[dc]>=test_start)&(df[dc]<test_end)]
X_te=test[feats].values.astype(np.float32); y_te=test[tc].values
tr_all=df[df[dc]<test_start]
X_tr_all=tr_all[feats].values.astype(np.float32); y_tr_all=tr_all[tc].values
w_all=(1-y_tr_all.mean())/y_tr_all.mean()
print(f'  Train: {len(X_tr_all):,} ({y_tr_all.mean():.4f}) Test: {len(y_te):,} ({y_te.mean():.4f})')

preds_te=[]
for name in ('XGBoost','LightGBM'):
    if name=='XGBoost':
        m=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                         min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w_all,
                         tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
    else:
        m=lgbm.LGBMClassifier(n_estimators=800,max_depth=6,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                               min_child_samples=20,reg_alpha=0.5,reg_lambda=0.5,class_weight='balanced',
                               n_jobs=-1,random_state=42,verbosity=-1)
    m.fit(X_tr_all,y_tr_all); p=m.predict_proba(X_te)[:,1]; preds_te.append(p)
    r=met(y_te,p); r['model']=name; r['fold']='TEST'; all_res.append(r)
    print(f'  {name:12s} AUC={r["auc_roc"]:.4f} P={r["precision"]:.4f} R={r["recall"]:.4f} F1={r["f1"]:.4f} MCC={r["mcc"]:.4f} Lift={r["lift"]}x')

ep_te=np.mean(preds_te,axis=0); r_te=met(y_te,ep_te); r_te['model']='Ensemble'; r_te['fold']='TEST'; all_res.append(r_te)
print(f'  {"Ensemble":12s} AUC={r_te["auc_roc"]:.4f} P={r_te["precision"]:.4f} R={r_te["recall"]:.4f} F1={r_te["f1"]:.4f} MCC={r_te["mcc"]:.4f} Lift={r_te["lift"]}x')

# Threshold optimization
print('\n--- Threshold Optimization ---')
bt,bf,bm=0.5,0,None
for th in np.arange(0.05,0.85,0.025):
    mr=met(y_te,ep_te,th=th)
    if mr['f1']>bf: bf,bt,bm=mr['f1'],th,mr
print(f'  Best th={bt:.3f}: F1={bf:.4f} P={bm["precision"]:.4f} R={bm["recall"]:.4f} MCC={bm["mcc"]:.4f} Lift={bm["lift"]}x')
print(f'  Confusion: TP={bm["tp"]} FP={bm["fp"]} FN={bm["fn"]} TN={bm["tn"]}')

# Bootstrap CI
print('\n--- Bootstrap CI (1000 reps) ---')
rng=np.random.RandomState(42); n=len(y_te); aucs=[]
for b in range(1000):
    idx=rng.choice(n,n,replace=True)
    if len(np.unique(y_te[idx]))==2: aucs.append(roc_auc_score(y_te[idx],ep_te[idx]))
lo,hi,mu=np.percentile(aucs,2.5),np.percentile(aucs,97.5),np.mean(aucs)
print(f'  AUC 95% CI: [{lo:.4f}, {hi:.4f}] (mean={mu:.4f})')

# Feature importance
print('\n--- Feature Importance ---')
m_fi=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                    min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w_all,
                    tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
m_fi.fit(X_tr_all,y_tr_all)
fi=pd.DataFrame({'feature':feats,'importance':m_fi.feature_importances_}).sort_values('importance',ascending=False)
fi['cum']=fi['importance'].cumsum(); fi.to_csv(OUT/'tables'/'feature_importance.csv',index=False)
for i,(_,r) in enumerate(fi.head(15).iterrows(),1): print(f'  {i:2d}. {r["feature"]:30s} {r["importance"]:.4f} ({r["cum"]:.3f})')

# Save
pd.DataFrame(all_res).to_csv(OUT/'tables'/'all_results.csv',index=False)
np.save(OUT/'final_preds.npy',ep_te); np.save(OUT/'y_test.npy',y_te)
pd.DataFrame({'y_true':y_te,'y_pred':ep_te}).to_csv(OUT/'tables'/'predictions.csv',index=False)
pd.DataFrame([{'th':round(th,3),**met(y_te,ep_te,th=th)} for th in np.arange(0.05,0.80,0.025)]).to_csv(OUT/'tables'/'threshold_table.csv',index=False)

tt=time.time()-t0
print(f'\n{"="*60}')
print(f'COMPLETE | {tt:.0f}s ({tt/60:.1f} min)')
print(f'Ensemble test AUC: {r_te["auc_roc"]:.4f} [95%CI: {lo:.4f}-{hi:.4f}]')
print(f'At th=0.5: P={r_te["precision"]:.4f} R={r_te["recall"]:.4f} F1={r_te["f1"]:.4f} Lift={r_te["lift"]}x')
print(f'At th={bt:.3f}: P={bm["precision"]:.4f} R={bm["recall"]:.4f} F1={bf:.4f} Lift={bm["lift"]}x MCC={bm["mcc"]:.4f}')
print(f'{"="*60}')

summary={'time':tt,'test_target_rate':float(y_te.mean()),'auc_95ci':[float(lo),float(hi)],
         'ensemble_test_at_05':{k:r_te[k] for k in ('auc_roc','avg_precision','precision','recall','f1','mcc','lift','brier')},
         'optimal':{'threshold':float(bt),'f1':float(bf),'precision':float(bm['precision']),'recall':float(bm['recall']),
                    'mcc':float(bm['mcc']),'lift':float(bm['lift']),'tp':int(bm['tp']),'fp':int(bm['fp']),'fn':int(bm['fn']),'tn':int(bm['tn'])},
         'top5_features':fi.head(5)['feature'].tolist()}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
print(f'\nOutput: {OUT}')
print(f'Tables: {OUT/"tables"}')
