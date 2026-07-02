# FINAL MODEL: cluster-enhanced pooled + cluster-specific ensemble
import pandas as pd, numpy as np, warnings, json, time
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, brier_score_loss, average_precision_score

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'final_model'
OUT.mkdir(exist_ok=True); (OUT/'charts').mkdir(exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)
print('='*70); print(f'FINAL MODEL: CLUSTER-ENHANCED | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'); print('='*70); t0=time.time()

df = pd.read_parquet(BASE/'improved_features.parquet')
tc='target'; dc='_d'
feats = [c for c in df.columns if c not in ('symbol','datetime',tc,dc)]

# ─── 1. ASSIGN CLUSTERS ───
print('\n[1] Assigning volatility x market-cap clusters...')
sym_stats = df.groupby('symbol').agg(hv_mean=('hv_20','mean'), avg_price=('close','mean'), avg_volume=('volume','mean')).reset_index()
sym_stats['mcap'] = np.log1p(sym_stats['avg_price'] * sym_stats['avg_volume'])
sym_stats['vol_q'] = pd.qcut(sym_stats['hv_mean'].rank(method='first'), 4, labels=['V1_low','V2_med','V3_high','V4_vhigh'])
sym_stats['mcap_t'] = pd.qcut(sym_stats['mcap'].rank(method='first'), 2, labels=['M1_small','M2_large'])
sym_stats['cluster'] = sym_stats['vol_q'].astype(str) + '_' + sym_stats['mcap_t'].astype(str)
cl_map = dict(zip(sym_stats['symbol'], sym_stats['cluster']))
df['cluster'] = df['symbol'].map(cl_map)
print(f'  Symbols: {len(sym_stats)} | Clusters: {sorted(sym_stats["cluster"].unique())}')
print(sym_stats.groupby('cluster').agg(n=('symbol','count')).to_string())

# ─── 2. ADD CLUSTER FEATURES ───
print('\n[2] Adding cluster features...')
# One-hot encode cluster
cluster_dummies = pd.get_dummies(df['cluster'], prefix='cl')
df = pd.concat([df, cluster_dummies], axis=1)
cluster_feats = list(cluster_dummies.columns)
# Cluster x top feature interactions
top_feats = ['regime_2','range_5','hv_20','regime_3','ret_1d','hv_10','range_10','bb_width']
for cl in sorted(df['cluster'].unique()):
    cl_short = cl.replace('_','')
    for tf in top_feats:
        n = f'cl_{cl_short}_x_{tf}'
        if n not in df.columns:
            df[n] = (df['cluster']==cl).astype(float) * df[tf]
            cluster_feats.append(n)

all_feats = feats + cluster_feats
print(f'  Base feats: {len(feats)} + Cluster feats: {len(cluster_feats)} = {len(all_feats)}')

from xgboost import XGBClassifier
import lightgbm as lgbm

# ─── 3. WALKFORWARD EVALUATION ───
splits = [
    ('S1','2016-01-01','2023-01-01','2023-01-01','2023-07-01'),
    ('S2','2016-01-01','2023-07-01','2023-07-01','2024-01-01'),
    ('S3','2017-01-01','2024-01-01','2024-01-01','2024-07-01'),
    ('S4','2017-01-01','2024-07-01','2024-07-01','2025-01-01'),
    ('S5','2018-01-01','2025-01-01','2025-01-01','2025-07-01'),
]
test_start, test_end = '2025-07-01', '2026-06-26'

def met(y,p,th=0.5):
    b=(p>=th).astype(int); cm=confusion_matrix(y,b); tn,fp,fn,tp=cm.ravel() if cm.size==4 else (0,0,0,0)
    pr=precision_score(y,b,zero_division=0); rc=recall_score(y,b,zero_division=0)
    return {'auc_roc':round(roc_auc_score(y,p),4),'avg_precision':round(average_precision_score(y,p),4),
            'precision':round(pr,4),'recall':round(rc,4),'f1':round(f1_score(y,b,zero_division=0),4),
            'mcc':round(matthews_corrcoef(y,b),4),'brier':round(brier_score_loss(y,p),4),
            'lift':round(pr/y.mean() if y.mean()>0 else 0,2),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}

print('\n[3] Walkforward evaluation...')
all_res = []
for fn,ts_tr,te_tr,ts_va,te_va in splits:
    tr=df[(df[dc]>=ts_tr)&(df[dc]<te_tr)]; va=df[(df[dc]>=ts_va)&(df[dc]<te_va)]
    X_tr=tr[all_feats].values.astype(np.float32); y_tr=tr[tc].values
    X_va=va[all_feats].values.astype(np.float32); y_va=va[tc].values
    w=(1-y_tr.mean())/y_tr.mean(); t1=time.time()
    preds=[]
    for nm in ('XGBoost','LightGBM'):
        if nm=='XGBoost':
            m=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                             min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w,
                             tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
        else:
            m=lgbm.LGBMClassifier(n_estimators=800,max_depth=6,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                                   min_child_samples=20,reg_alpha=0.5,reg_lambda=0.5,class_weight='balanced',
                                   n_jobs=-1,random_state=42,verbosity=-1)
        m.fit(X_tr,y_tr); p=m.predict_proba(X_va)[:,1]; preds.append(p)
        r=met(y_va,p); r['model']=nm; r['fold']=fn; all_res.append(r)
    ep=np.mean(preds,axis=0); r=met(y_va,ep); r['model']='Ensemble'; r['fold']=fn; all_res.append(r)
    print(f'  {fn}: XGB={all_res[-3]["auc_roc"]:.4f} LGB={all_res[-2]["auc_roc"]:.4f} ENS={r["auc_roc"]:.4f} P={r["precision"]:.4f} R={r["recall"]:.4f} F1={r["f1"]:.4f} ({time.time()-t1:.0f}s)')

# ─── 4. FINAL TEST ───
print('\n[4] Final test evaluation...')
test=df[(df[dc]>=test_start)&(df[dc]<test_end)]
X_te=test[all_feats].values.astype(np.float32); y_te=test[tc].values
tr_all=df[df[dc]<test_start]; X_tr_all=tr_all[all_feats].values.astype(np.float32); y_tr_all=tr_all[tc].values
w_all=(1-y_tr_all.mean())/y_tr_all.mean()

preds_te=[]
for nm in ('XGBoost','LightGBM'):
    if nm=='XGBoost':
        m=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                         min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w_all,
                         tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
    else:
        m=lgbm.LGBMClassifier(n_estimators=800,max_depth=6,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                               min_child_samples=20,reg_alpha=0.5,reg_lambda=0.5,class_weight='balanced',
                               n_jobs=-1,random_state=42,verbosity=-1)
    m.fit(X_tr_all,y_tr_all); p=m.predict_proba(X_te)[:,1]; preds_te.append(p)
    r=met(y_te,p); r['model']=nm; r['fold']='TEST'; all_res.append(r)
ep_te=np.mean(preds_te,axis=0); r_te=met(y_te,ep_te); r_te['model']='Ensemble'; r_te['fold']='TEST'; all_res.append(r_te)
print(f'  XGB={all_res[-3]["auc_roc"]:.4f} LGB={all_res[-2]["auc_roc"]:.4f} ENS={r_te["auc_roc"]:.4f} P={r_te["precision"]:.4f} R={r_te["recall"]:.4f} F1={r_te["f1"]:.4f} MCC={r_te["mcc"]:.4f} Lift={r_te["lift"]}x')

# ─── 5. THRESHOLD OPT ───
print('\n[5] Threshold optimization...')
bt,bf,bm=0.5,0,None
for th in np.arange(0.05,0.85,0.025):
    mr=met(y_te,ep_te,th=th)
    if mr['f1']>bf: bf,bt,bm=mr['f1'],th,mr
print(f'  Best th={bt:.3f}: F1={bf:.4f} P={bm["precision"]:.4f} R={bm["recall"]:.4f} MCC={bm["mcc"]:.4f} Lift={bm["lift"]}x')
print(f'  TP={bm["tp"]} FP={bm["fp"]} FN={bm["fn"]} TN={bm["tn"]}')

# ─── 6. BOOTSTRAP CI ───
print('\n[6] Bootstrap CI...')
rng=np.random.RandomState(42); n=len(y_te); aucs=[]
for b in range(1000):
    idx=rng.choice(n,n,replace=True)
    if len(np.unique(y_te[idx]))==2: aucs.append(roc_auc_score(y_te[idx],ep_te[idx]))
lo,hi=np.percentile(aucs,2.5),np.percentile(aucs,97.5)
print(f'  AUC 95% CI: [{lo:.4f}, {hi:.4f}]')

# ─── 7. FEATURE IMPORTANCE ───
print('\n[7] Feature importance (XGBoost + cluster feats)...')
m_fi=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                    min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w_all,
                    tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
m_fi.fit(X_tr_all,y_tr_all)
fi=pd.DataFrame({'feature':all_feats,'importance':m_fi.feature_importances_}).sort_values('importance',ascending=False)
fi['cum']=fi['importance'].cumsum(); fi.to_csv(OUT/'tables'/'feature_importance.csv',index=False)
n_cl=sum(1 for f in fi.head(30)['feature'] if 'cl_' in f)
print(f'  Top 30: {n_cl} cluster features in top 30')
for i,(_,r) in enumerate(fi.head(20).iterrows(),1): print(f'  {i:2d}. {r["feature"]:35s} {r["importance"]:.4f}')

# ─── 8. SAVE ───
pd.DataFrame(all_res).to_csv(OUT/'tables'/'results.csv',index=False)
np.save(OUT/'final_preds.npy',ep_te); np.save(OUT/'y_test.npy',y_te)
pd.DataFrame({'y_true':y_te,'y_pred':ep_te}).to_csv(OUT/'tables'/'predictions.csv',index=False)
pd.DataFrame([{'th':round(th,3),**met(y_te,ep_te,th=th)} for th in np.arange(0.05,0.80,0.025)]).to_csv(OUT/'tables'/'thresholds.csv',index=False)
fi.to_csv(OUT/'tables'/'feature_importance.csv',index=False)

# ─── 9. FINAL COMPARISON ───
print()
print('--- FINAL COMPARISON ACROSS ALL APPROACHES ---')
h = '%-35s %8s %7s %7s %7s %7s %7s %7s' % ('Approach','AUC','P@0.5','R@0.5','F1@0.5','F1opt','Ropt','Bestth')
print(h)
print('-' * 85)
baselines = [
    ('Phase 8 baseline (35 feats, no filter)', 0.6545, 0.1896, 0.5260, 0.2787, 0, 0, 0),
    ('Improved (157 feats, filter, class_weights)', 0.6265, 0.1953, 0.2423, 0.2162, 0.2303, 0.4501, 0.375),
    ('Cluster-specific ensemble', 0.5850, 0.1927, 0.1799, 0.1861, 0.2370, 0.5728, 0.150),
    ('Cluster-enhanced pooled (THIS MODEL)', r_te['auc_roc'], r_te['precision'], r_te['recall'], r_te['f1'], bf, bm['recall'], bt),
]
for name,auc,p,r,f1,f1o,ro,th in baselines:
    f1o_str = '%.4f' % f1o if f1o else '--'
    ro_str = '%.4f' % ro if ro else '--'
    th_str = '%.3f' % th if th else '--'
    print('%-35s %8.4f %7.4f %7.4f %7.4f %7s %7s %7s' % (name, auc, p, r, f1, f1o_str, ro_str, th_str))

tt=time.time()-t0
print(f'\n{"="*70}')
print(f'FINAL MODEL COMPLETE | {tt:.0f}s ({tt/60:.1f} min)')
print(f'Cluster-enhanced ensemble test AUC: {r_te["auc_roc"]:.4f} [95%CI: {lo:.4f}-{hi:.4f}]')
print(f'{"="*70}')

summary={'time':tt,'test_target_rate':float(y_te.mean()),'auc_95ci':[float(lo),float(hi)],
         'test_at_05':{k:r_te[k] for k in ('auc_roc','avg_precision','precision','recall','f1','mcc','lift','brier')},
         'optimal':{'threshold':float(bt),'f1':float(bf),'precision':float(bm['precision']),'recall':float(bm['recall']),
                    'mcc':float(bm['mcc']),'lift':float(bm['lift']),'tp':int(bm['tp']),'fp':int(bm['fp']),'fn':int(bm['fn']),'tn':int(bm['tn'])},
         'n_features':len(all_feats),'n_cluster_features':len(cluster_feats)}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
print(f'Output: {OUT}')
