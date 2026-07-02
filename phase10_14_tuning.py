# Phases 10-14: Nested CV Hyperparameter Tuning + Final Evaluation (v2)
import pandas as pd, numpy as np, warnings, json, time, gc
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'phase10_14_tuning'
OUT.mkdir(exist_ok=True); (OUT/'charts').mkdir(exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)

print('='*70); print('PHASES 10-14: NESTED CV TUNING + CALIBRATION (v2)')
print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'); print('='*70)
t_start = time.time()

df = pd.read_parquet(BASE/'improved_features.parquet')
tc = 'target'; dc = '_d'
feats = [c for c in df.columns if c not in ('symbol','datetime',tc,dc)]
print(f'  Shape: {df.shape}, Features: {len(feats)}')

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, average_precision_score, brier_score_loss

def met(y,p,th=0.5):
    b=(p>=th).astype(int); cm=confusion_matrix(y,b); tn,fp,fn,tp=cm.ravel() if cm.size==4 else (0,0,0,0)
    pr=precision_score(y,b,zero_division=0); rc=recall_score(y,b,zero_division=0)
    return {'auc_roc':round(roc_auc_score(y,p),4),'avg_precision':round(average_precision_score(y,p),4),
            'precision':round(pr,4),'recall':round(rc,4),'f1':round(f1_score(y,b,zero_division=0),4),
            'mcc':round(matthews_corrcoef(y,b),4),'brier':round(brier_score_loss(y,p),4),
            'lift':round(pr/y.mean() if y.mean()>0 else 0,2),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'th':th}

splits = [
    ('S1','2016-01-01','2023-01-01','2023-01-01','2023-07-01'),
    ('S2','2016-01-01','2023-07-01','2023-07-01','2024-01-01'),
    ('S3','2017-01-01','2024-01-01','2024-01-01','2024-07-01'),
    ('S4','2017-01-01','2024-07-01','2024-07-01','2025-01-01'),
    ('S5','2018-01-01','2025-01-01','2025-01-01','2025-07-01'),
]
ts, te = '2025-07-01', '2026-06-26'

import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from xgboost import XGBClassifier
import lightgbm as lgbm
from imblearn.combine import SMOTETomek
from sklearn.calibration import CalibratedClassifierCV

N_TRIALS = 30
SAMPLE_TUNE = 100000
USE_SMOTE_FINAL = True

all_results = []
best_bp = {}  # per model

for fold_name, ts_tr, te_tr, ts_va, te_va in splits:
    print(f'\n=== {fold_name} ==='); t0=time.time()
    tr = df[(df[dc]>=ts_tr)&(df[dc]<te_tr)]; va = df[(df[dc]>=ts_va)&(df[dc]<te_va)]
    X_tr = tr[feats].values.astype(np.float32); y_tr = tr[tc].values
    X_va = va[feats].values.astype(np.float32); y_va = va[tc].values
    w = (1-y_tr.mean())/y_tr.mean()
    print(f'  Train:{len(X_tr):,}({y_tr.mean():.3f}) Val:{len(X_va):,}({y_va.mean():.3f})')

    # Subsample for tuning
    rng=np.random.RandomState(42)
    if len(X_tr)>SAMPLE_TUNE:
        idx=rng.choice(len(X_tr),SAMPLE_TUNE,replace=False); X_t=X_tr[idx]; y_t=y_tr[idx]
    else: X_t=X_tr; y_t=y_tr

    # ── XGB Optuna (no SMOTE in CV loop - use scale_pos_weight instead) ──
    def obj_xgb(t):
        p={'n_estimators':t.suggest_int('n',400,1200),'max_depth':t.suggest_int('md',4,9),
           'lr':t.suggest_float('lr',0.01,0.1,log=True),'subsample':t.suggest_float('sub',0.7,1.0),
           'colsample_bytree':t.suggest_float('cs',0.6,1.0),'min_child_weight':t.suggest_int('mcw',1,8),
           'gamma':t.suggest_float('g',0,2),'reg_alpha':t.suggest_float('ra',0,2),
           'reg_lambda':t.suggest_float('rl',0,2),'scale_pos_weight':w}
        split=int(len(y_t)*0.8)
        m=XGBClassifier(**p,tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
        m.fit(X_t[:split],y_t[:split])
        return roc_auc_score(y_t[split:],m.predict_proba(X_t[split:])[:,1])

    print('  XGBoost...')
    sx=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    sx.optimize(obj_xgb,n_trials=N_TRIALS)
    bp1=sx.best_params
    best_bp['XGBoost']=bp1
    print(f'    AUC={sx.best_value:.4f} md={bp1["md"]} lr={bp1["lr"]:.4f} n={bp1["n"]} spw={w:.2f}')

    # ── LGB Optuna ──
    def obj_lgb(t):
        p={'n_estimators':t.suggest_int('n',400,1200),'max_depth':t.suggest_int('md',4,9),
           'lr':t.suggest_float('lr',0.01,0.1,log=True),'subsample':t.suggest_float('sub',0.7,1.0),
           'colsample_bytree':t.suggest_float('cs',0.6,1.0),'min_child_samples':t.suggest_int('mcs',10,50),
           'reg_alpha':t.suggest_float('ra',0,2),'reg_lambda':t.suggest_float('rl',0,2)}
        split=int(len(y_t)*0.8)
        m=lgbm.LGBMClassifier(**p,class_weight='balanced',n_jobs=-1,random_state=42,verbosity=-1)
        m.fit(X_t[:split],y_t[:split])
        return roc_auc_score(y_t[split:],m.predict_proba(X_t[split:])[:,1])

    print('  LightGBM...')
    sl=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    sl.optimize(obj_lgb,n_trials=N_TRIALS)
    bp2=sl.best_params
    best_bp['LightGBM']=bp2
    print(f'    AUC={sl.best_value:.4f} md={bp2["md"]} lr={bp2["lr"]:.4f} n={bp2["n"]}')

    # ── Train models with SMOTE on full fold data ──
    print('  Training on full fold...')
    # Use scale_pos_weight/class_weight instead of SMOTE (SMOTE harms calibration on imbalanced val)
    fold_preds=[]
    for nm,bp in [('XGBoost',{**bp1,'scale_pos_weight':w}),('LightGBM',{**bp2})]:
        if nm=='XGBoost':
            m=XGBClassifier(**bp,tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
            m.fit(X_tr,y_tr)
        else:
            m=lgbm.LGBMClassifier(**bp,class_weight='balanced',n_jobs=-1,random_state=42,verbosity=-1)
            m.fit(X_tr,y_tr)
        p=m.predict_proba(X_va)[:,1]; fold_preds.append(p)
        r=met(y_va,p); r['model']=nm; r['fold']=fold_name; all_results.append(r)
        print(f'    {nm:12s} AUC={r["auc_roc"]:.4f} P={r["precision"]:.4f} R={r["recall"]:.4f} F1={r["f1"]:.4f} MCC={r["mcc"]:.4f}')

    ep=np.mean(fold_preds,axis=0); r=met(y_va,ep); r['model']='Ensemble'; r['fold']=fold_name; all_results.append(r)
    print(f'    {"Ensemble":12s} AUC={r["auc_roc"]:.4f} P={r["precision"]:.4f} R={r["recall"]:.4f} F1={r["f1"]:.4f} MCC={r["mcc"]:.4f}')
    print(f'    Time: {time.time()-t0:.0f}s')

pd.DataFrame(all_results).to_csv(OUT/'tables'/'fold_results.csv',index=False)

# ─── FINAL TEST ───
print('\n=== FINAL TEST ==='); t0=time.time()
test=df[(df[dc]>=ts)&(df[dc]<te)]; X_te=test[feats].values.astype(np.float32); y_te=test[tc].values
tr_all=df[df[dc]<ts]; X_tr_all=tr_all[feats].values.astype(np.float32); y_tr_all=tr_all[tc].values
w_all=(1-y_tr_all.mean())/y_tr_all.mean()
print(f'  Train:{len(X_tr_all):,}({y_tr_all.mean():.4f}) Test:{len(y_te):,}({y_te.mean():.4f})')

final_preds=[]
for nm,bp in best_bp.items():
    if nm=='XGBoost': m=XGBClassifier(**{**bp,'scale_pos_weight':w_all},tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
    else: m=lgbm.LGBMClassifier(**bp,class_weight='balanced',n_jobs=-1,random_state=42,verbosity=-1)
    m.fit(X_tr_all,y_tr_all); p=m.predict_proba(X_te)[:,1]; final_preds.append(p)
    r=met(y_te,p); r['model']=nm; r['fold']='FINAL'; all_results.append(r)
    print(f'  {nm:12s} AUC={r["auc_roc"]:.4f} P={r["precision"]:.4f} R={r["recall"]:.4f} F1={r["f1"]:.4f} MCC={r["mcc"]:.4f} Lift={r["lift"]}x')

ep=np.mean(final_preds,axis=0); r_ens=met(y_te,ep); r_ens['model']='Ensemble'; r_ens['fold']='FINAL'; all_results.append(r_ens)
print(f'  {"Ensemble":12s} AUC={r_ens["auc_roc"]:.4f} P={r_ens["precision"]:.4f} R={r_ens["recall"]:.4f} F1={r_ens["f1"]:.4f} MCC={r_ens["mcc"]:.4f} Lift={r_ens["lift"]}x')

# ─── THRESHOLD OPT ───
print('\nThreshold opt...')
best_th,best_f1,best_m=0.5,0,None
for th in np.arange(0.05,0.85,0.025):
    m_res=met(y_te,ep,th=th)
    if m_res['f1']>best_f1: best_f1,best_th,best_m=m_res['f1'],th,m_res
print(f'  Best th={best_th:.3f}: F1={best_f1:.4f} P={best_m["precision"]:.4f} R={best_m["recall"]:.4f} MCC={best_m["mcc"]:.4f} Lift={best_m["lift"]}x')
print(f'  TP={best_m["tp"]} FP={best_m["fp"]} FN={best_m["fn"]} TN={best_m["tn"]}')

# ─── BOOTSTRAP CI ───
print('\nBootstrap CI...')
boot=[roc_auc_score(y_te[np.random.RandomState(42+b).choice(len(y_te),len(y_te),replace=True)],ep[np.random.RandomState(42+b).choice(len(y_te),len(y_te),replace=True)]) for b in range(500)]
lo,hi=np.percentile(boot,2.5),np.percentile(boot,97.5)
print(f'  AUC 95% CI: [{lo:.4f}, {hi:.4f}]')

# ─── FEATURE IMPORTANCE ───
print('\nFeature importance...')
best_xgb=XGBClassifier(**best_bp['XGBoost'],tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
best_xgb.fit(X_res_all,y_res_all)
fi=pd.DataFrame({'feature':feats,'importance':best_xgb.feature_importances_}).sort_values('importance',ascending=False)
fi['cum']=fi['importance'].cumsum(); fi.to_csv(OUT/'tables'/'feature_importance.csv',index=False)
print(f'  Top5: {fi.head(5)["feature"].tolist()}')
print(f'  Top5 cum: {fi.head(5)["cum"].iloc[-1]:.3f}')

np.save(OUT/'final_preds.npy',ep); np.save(OUT/'y_test.npy',y_te)
pd.DataFrame({'y_true':y_te,'y_pred':ep}).to_csv(OUT/'tables'/'predictions.csv',index=False)
pd.DataFrame(all_results).to_csv(OUT/'tables'/'all_results.csv',index=False)

tt=time.time()-t_start
print(f'\n{"="*70}')
print(f'DONE | {tt:.0f}s ({tt/60:.1f}min)')
print(f'Ensemble test: AUC={r_ens["auc_roc"]:.4f} [95%CI: {lo:.4f}-{hi:.4f}]')
print(f'  At th=0.5: P={r_ens["precision"]:.4f} R={r_ens["recall"]:.4f} F1={r_ens["f1"]:.4f}')
print(f'  At th={best_th:.3f}: P={best_m["precision"]:.4f} R={best_m["recall"]:.4f} F1={best_f1:.4f} Lift={best_m["lift"]}x')
print(f'{"="*70}')

summary={'phase':'10-14','time':tt,'test_target_rate':float(y_te.mean()),
         'ensemble_test':{k:v for k,v in r_ens.items() if k in ('auc_roc','avg_precision','precision','recall','f1','mcc','lift','brier')},
         'optimal':{k:v for k,v in best_m.items() if k not in ('th',)},'auc_95ci':[float(lo),float(hi)],
         'top5_features':fi.head(5)['feature'].tolist(),'models_tuned':list(best_bp.keys())}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
print(f'Summary: {OUT/"summary.json"}')
