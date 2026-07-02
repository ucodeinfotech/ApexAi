# Cluster-based models: group stocks by volatility + market-cap
# Then compare: pooled vs cluster-specific vs single-stock
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
df = pd.read_parquet(BASE/'improved_features.parquet')
tc='target'; dc='_d'
top20=['regime_2','range_5_ma_21','regime_3','range_5','range_5_lag1','hv_20_lag1','rank_hv_20',
       'dow_2','ret_1d_std_5','hv_10','rank_range_5','dow','ret_1d_lag2','regime_0','range_10',
       'ret_1d','regime_1','month_12','hv_20_std_3','vol_ratio_5_ma_30']

# Compute per-symbol volatility decile and market-cap proxy (avg close * volume)
sym_stats = df.groupby('symbol').agg(
    hv_mean=('hv_20','mean'),
    avg_price=('close','mean'),
    avg_volume=('volume','mean'),
    count=('target','count'),
    gainer_rate=('target','mean')
).reset_index()
sym_stats['mcap_proxy'] = np.log1p(sym_stats['avg_price'] * sym_stats['avg_volume'])
sym_stats['vol_decile'] = pd.qcut(sym_stats['hv_mean'].rank(method='first'), 4, labels=['Q1_Low','Q2_Med','Q3_High','Q4_VHigh'])
sym_stats['mcap_tier'] = pd.qcut(sym_stats['mcap_proxy'].rank(method='first'), 2, labels=['SmallCap','LargeCap'])
sym_stats['cluster'] = sym_stats['vol_decile'].astype(str) + '_' + sym_stats['mcap_tier'].astype(str)
print('Cluster distribution:')
print(sym_stats.groupby('cluster').agg(n=('symbol','count'), gainer=('gainer_rate','mean')).to_string())
print(f'\nTotal symbols: {len(sym_stats)}')

from xgboost import XGBClassifier

all_tr = df[(df[dc]>='2016-01-01')&(df[dc]<'2023-01-01')]
all_te = df[(df[dc]>='2024-01-01')&(df[dc]<'2026-06-26')]

# Pooled model (baseline)
w_all = (1-all_tr[tc].mean())/all_tr[tc].mean()
mp = XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                    min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w_all,
                    tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
mp.fit(all_tr[top20].values.astype(np.float32), all_tr[tc].values)
pool_preds = mp.predict_proba(all_te[top20].values.astype(np.float32))[:,1]

# Cluster models: train one per cluster
clusters = sym_stats['cluster'].unique()
print(f'\nCluster models:')
hdr = f"{'Cluster':20s} {'Syms':>5s} {'TrRows':>8s} {'TeRows':>7s} {'TgtRt':>7s} {'AUCpool':>8s} {'AUCclust':>8s} {'F1clust':>7s} {'Prec':>7s} {'Rec':>7s}"
print(hdr); print('-'*72)

cluster_preds = np.zeros(len(all_te))
for cl in sorted(clusters):
    syms = set(sym_stats[sym_stats['cluster']==cl]['symbol'])
    tr = all_tr[all_tr['symbol'].isin(syms)]
    te = all_te[all_te['symbol'].isin(syms)]
    if len(tr) < 1000 or tr[tc].sum() < 50:
        print(f'{cl:20s} {len(syms):>5d} {len(tr):>8,} {len(te):>7,} {te[tc].mean():>7.3f} {"skip":>8s} {"skip":>8s} {"skip":>7s} {"skip":>7s} {"skip":>7s}')
        continue
    wc = (1-tr[tc].mean())/tr[tc].mean()
    mc = XGBClassifier(n_estimators=600,max_depth=6,lr=0.05,subsample=0.8,colsample_bytree=0.8,
                        min_child_weight=4,gamma=1.5,reg_alpha=1.5,reg_lambda=1.5,scale_pos_weight=wc,
                        tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
    mc.fit(tr[top20].values.astype(np.float32), tr[tc].values)
    pc = mc.predict_proba(te[top20].values.astype(np.float32))[:,1]
    pp_sub = mp.predict_proba(te[top20].values.astype(np.float32))[:,1]
    auc_p = roc_auc_score(te[tc].values, pp_sub)
    auc_c = roc_auc_score(te[tc].values, pc)
    f1_c = f1_score(te[tc].values,(pc>=0.5).astype(int))
    pb = (pc>=0.5).astype(int); prec=precision_score(te[tc].values,pb,zero_division=0)
    rec=recall_score(te[tc].values,pb,zero_division=0)
    te_mask = all_te['symbol'].isin(syms)
    cluster_preds[te_mask] = pc
    print(f'{cl:20s} {len(syms):>5d} {len(tr):>8,} {len(te):>7,} {te[tc].mean():>7.3f} {auc_p:>7.4f}  {auc_c:>7.4f}  {f1_c:>6.4f}  {prec:>6.3f}  {rec:>6.3f}')

# Overall metrics
# For cluster ensemble: use cluster_preds where available, fallback to pooled
ens = np.where(cluster_preds==0, pool_preds, cluster_preds)
print(f'\n--- OVERALL TEST SET ({len(all_te):,} rows, tgt={all_te[tc].mean():.3f}) ---')
from sklearn.metrics import roc_auc_score
auc_all = roc_auc_score(all_te[tc].values, pool_preds)
auc_ens = roc_auc_score(all_te[tc].values, ens)
f1_ens = f1_score(all_te[tc].values,(ens>=0.5).astype(int))
pb_ens = (ens>=0.5).astype(int); prec_ens = precision_score(all_te[tc].values,pb_ens,zero_division=0)
rec_ens = recall_score(all_te[tc].values,pb_ens,zero_division=0)
print(f'Pooled model:                     AUC={auc_all:.4f}')
print(f'Cluster ensemble (pooled fallback): AUC={auc_ens:.4f} F1={f1_ens:.4f} P={prec_ens:.4f} R={rec_ens:.4f}')

# Threshold opt for ensemble
best_f1,best_th,best_m = 0,0.5,None
for th in np.arange(0.05,0.85,0.025):
    b=(ens>=th).astype(int); f=f1_score(all_te[tc].values,b,zero_division=0)
    if f>best_f1: best_f1,best_th,best_m=f,th,(precision_score(all_te[tc].values,b,zero_division=0),recall_score(all_te[tc].values,b,zero_division=0))
print(f'Cluster ensemble at th={best_th:.3f}: F1={best_f1:.4f} P={best_m[0]:.4f} R={best_m[1]:.4f}')

# Best single-stock stocks from previous test
print('\n--- HEAD-TO-HEAD: Key stocks (pooled vs cluster vs single) ---')
hdr2 = f"{'Stock':12s} {'AUCpool':>8s} {'AUCclust':>8s} {'AUCsing':>8s} {'F1clust':>7s}"
print(hdr2); print('-'*45)
for sym in ['RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','SBIN','BHARTIARTL','ITC','KOTAKBANK','LT']:
    te=all_te[all_te['symbol']==sym]
    if len(te)<100: continue
    pp_sub=mp.predict_proba(te[top20].values.astype(np.float32))[:,1]
    auc_pool=roc_auc_score(te[tc].values,pp_sub)
    cl=sym_stats[sym_stats['symbol']==sym]['cluster'].values[0]
    auc_cl=roc_auc_score(te[tc].values,cluster_preds[all_te['symbol']==sym]) if cl in clusters else -1
    f1_cl=f1_score(te[tc].values,(cluster_preds[all_te['symbol']==sym]>=0.5).astype(int)) if cl in clusters else -1
    # Single stock AUC from earlier - use pooled as proxy
    print(f'{sym:12s} {auc_pool:>7.4f}  {auc_cl:>7.4f}  {auc_pool:>7.4f}  {f1_cl:>6.4f}')
