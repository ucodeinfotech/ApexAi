# Phase 8: Deep Comprehensive Feature Selection (Optimized)
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, warnings, time, gc, os, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'feature_selection_results'
OUT.mkdir(exist_ok=True)
(OUT/'charts').mkdir(exist_ok=True)
(OUT/'tables').mkdir(exist_ok=True)

print('='*70)
print('PHASE 8: DEEP COMPREHENSIVE FEATURE SELECTION (OPTIMIZED)')
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('='*70)
t_start = time.time()

# ─── 0. LOAD DATA ───
print('\n[0/12] Loading cleaned features...')
t0 = time.time()
df = pd.read_parquet(BASE/'cleaned_features.parquet')
print(f'  Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB')

target_col = 'target'
exclude_cols = {'symbol', 'date', target_col, 'symbol_date', 'Date', 'Symbol', 'datetime'}
date_col = 'datetime'
feature_cols = [c for c in df.columns if c not in exclude_cols and c not in ('symbol','date','Date','Symbol','symbol_date','datetime')]
# Exclude leak features (future info that would be used to compute target)
leak_patterns = ['target_ret', 'next_', 'fwd_', '_fwd', 'future_']
feature_cols = [c for c in feature_cols if not any(p in c for p in leak_patterns)]
print(f'  After removing leaks: {len(feature_cols)} features')
print(f'  Candidate features: {len(feature_cols)}, Target rate: {df[target_col].mean():.4f}')

# Temporal split
df['_d'] = pd.to_datetime(df[date_col])
train_m = df['_d'] < '2023-01-01'
val_m = (df['_d'] >= '2023-01-01') & (df['_d'] < '2024-01-01')
test_m = df['_d'] >= '2024-01-01'
X_train = df.loc[train_m, feature_cols].copy()
y_train = df.loc[train_m, target_col].values
X_val = df.loc[val_m, feature_cols]
y_val = df.loc[val_m, target_col].values
print(f'  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {test_m.sum():,}')

# Subsample map for stages
rng = np.random.RandomState(42)
S = lambda n, m=50000: rng.choice(n, min(m, n), replace=False)
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 1. MI RANKING ───
print('\n[1/12] Mutual Information...')
t0 = time.time()
from sklearn.feature_selection import mutual_info_classif
mi = np.zeros(len(feature_cols))
for i in range(0, len(feature_cols), 20):
    b = feature_cols[i:i+20]
    mi[i:i+len(b)] = mutual_info_classif(X_train[b].values, y_train, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi}).sort_values('mi_score', ascending=False)
mi_df['mi_rank'] = range(1, len(mi_df)+1)
mi_df.to_csv(OUT/'tables'/'mi_ranking.csv', index=False)
print(f'  Top 5: {mi_df.head(5)["feature"].tolist()} | Time: {time.time()-t0:.0f}s')

# ─── 2. VIF (correlation matrix method, fast) ───
print('\n[2/12] VIF (correlation matrix method)...')
t0 = time.time()
from sklearn.preprocessing import StandardScaler

# Skip dummies
vif_skip = {f for f in feature_cols if any(f.startswith(p) for p in ('month_','dow_','quarter'))}
vif_candidates = [f for f in feature_cols if f not in vif_skip]
print(f'  VIF candidates: {len(vif_candidates)} ({len(vif_skip)} dummies skipped)')

# Sample for VIF
vif_n = min(30000, len(X_train))
vif_idx = S(len(X_train), vif_n)
X_vif = X_train.iloc[vif_idx][vif_candidates]
sc = StandardScaler()
Xv = sc.fit_transform(X_vif)

# Iterative VIF via correlation matrix inverse
remaining = vif_candidates.copy()
vif_hist = []
for it in range(40):
    if len(remaining) <= 2: break
    # Get columns for remaining features
    cols = [vif_candidates.index(f) for f in remaining]
    X_sub = Xv[:, cols]
    corr = np.corrcoef(X_sub.T)
    try:
        inv = np.linalg.inv(corr)
        vifs = np.diag(inv)
    except:
        break
    max_v = vifs.max()
    if max_v < 10: break
    mx = remaining[np.argmax(vifs)]
    vif_hist.append({'iter': it+1, 'feature': mx, 'vif': float(max_v)})
    remaining.remove(mx)
    if it < 10 or (it+1) % 10 == 0:
        print(f'  Iter {it+1}: {mx} (VIF={max_v:.1f})')

vif_df = pd.DataFrame(vif_hist)
vif_df.to_csv(OUT/'tables'/'vif_elimination.csv', index=False)
survived_vif = list(vif_skip) + remaining
print(f'  Survived: {len(survived_vif)} (removed {len(vif_hist)}) | Time: {time.time()-t0:.0f}s')

# ─── 3. XGB IMPORTANCE ───
print('\n[3/12] XGBoost importance...')
t0 = time.time()
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_n = min(300000, len(X_train))
xgb_idx = S(len(X_train), xgb_n)
X_xgb = X_train.iloc[xgb_idx]
y_xgb = y_train[xgb_idx]
w = (1-y_xgb.mean())/y_xgb.mean()

try:
    xgb = XGBClassifier(n_estimators=300, max_depth=7, lr=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=w, tree_method='hist', device='cuda', random_state=42, verbosity=0)
    xgb.fit(X_xgb, y_xgb)
    print('  GPU OK')
except:
    xgb = XGBClassifier(n_estimators=300, max_depth=7, lr=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=w, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0)
    xgb.fit(X_xgb, y_xgb)
    print('  CPU fallback')

xgb_imp = pd.DataFrame({'feature': feature_cols, 'xgb_importance': xgb.feature_importances_}).sort_values('xgb_importance', ascending=False)
xgb_imp['xgb_rank'] = range(1, len(xgb_imp)+1)
xgb_imp.to_csv(OUT/'tables'/'xgb_importance.csv', index=False)
xgb_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1])
print(f'  AUC: {xgb_auc:.4f} | Top 5: {xgb_imp.head(5)["feature"].tolist()} | Time: {time.time()-t0:.0f}s')

# ─── 4. LGB IMPORTANCE ───
print('\n[4/12] LightGBM importance...')
t0 = time.time()
import lightgbm as lgb
try:
    lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=7, lr=0.05, subsample=0.8, colsample_bytree=0.8,
                                class_weight='balanced', device='gpu', gpu_platform_id=0, gpu_device_id=0, random_state=42, verbosity=-1)
    lgbm.fit(X_xgb, y_xgb)
    print('  GPU OK')
except:
    lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=7, lr=0.05, subsample=0.8, colsample_bytree=0.8,
                                class_weight='balanced', n_jobs=-1, random_state=42, verbosity=-1)
    lgbm.fit(X_xgb, y_xgb)
    print('  CPU fallback')

lgb_imp = pd.DataFrame({'feature': feature_cols, 'lgb_importance': lgbm.feature_importances_}).sort_values('lgb_importance', ascending=False)
lgb_imp['lgb_rank'] = range(1, len(lgb_imp)+1)
lgb_imp.to_csv(OUT/'tables'/'lgb_importance.csv', index=False)
lgb_auc = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:,1])
print(f'  AUC: {lgb_auc:.4f} | Top 5: {lgb_imp.head(5)["feature"].tolist()} | Time: {time.time()-t0:.0f}s')

# ─── 5. PERMUTATION IMPORTANCE ───
print('\n[5/12] Permutation importance...')
t0 = time.time()
from sklearn.inspection import permutation_importance
perm_n = 50000
perm_idx = S(len(X_train), perm_n)
X_p = X_train.iloc[perm_idx]; y_p = y_train[perm_idx]
pm = XGBClassifier(n_estimators=200, max_depth=6, lr=0.05, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=(1-y_p.mean())/y_p.mean())
pm.fit(X_p, y_p)
pr = permutation_importance(pm, X_p, y_p, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.DataFrame({'feature': feature_cols, 'perm_importance': pr.importances_mean, 'perm_std': pr.importances_std}).sort_values('perm_importance', ascending=False)
perm_imp['perm_rank'] = range(1, len(perm_imp)+1)
perm_imp.to_csv(OUT/'tables'/'permutation_importance.csv', index=False)
print(f'  Top 5: {perm_imp.head(5)["feature"].tolist()} | Time: {time.time()-t0:.0f}s')

# ─── 6. FORWARD SELECTION ───
print('\n[6/12] Greedy forward selection...')
t0 = time.time()
# Consensus top 40 candidates
mi_t = set(mi_df.head(40)['feature'])
xgb_t = set(xgb_imp.head(40)['feature'])
lgb_t = set(lgb_imp.head(40)['feature'])
perm_t = set(perm_imp.head(40)['feature'])
candidates = list(mi_t | xgb_t | lgb_t | perm_t)
print(f'  Candidates: {len(candidates)}')

fs_n = 40000
fs_idx = S(len(X_train), fs_n)
fs_split = int(fs_n * 0.8)
X_fs = X_train.iloc[fs_idx][candidates]; y_fs = y_train[fs_idx]
X_fs_tr = X_fs.iloc[:fs_split]; y_fs_tr = y_fs[:fs_split]
X_fs_va = X_fs.iloc[fs_split:]; y_fs_va = y_fs[fs_split:]
fsw = (1-y_fs_tr.mean())/y_fs_tr.mean()
sel, rem = [], candidates.copy()
fwd_hist = []
for step in range(min(30, len(candidates))):
    best_a, best_f = 0, None
    for f in rem:
        s = sel + [f]
        m = XGBClassifier(n_estimators=100, max_depth=5, lr=0.05, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=fsw)
        m.fit(X_fs_tr[s].values, y_fs_tr)
        a = roc_auc_score(y_fs_va, m.predict_proba(X_fs_va[s].values)[:,1])
        if a > best_a: best_a, best_f = a, f
    if best_f is None: break
    sel.append(best_f); rem.remove(best_f)
    fwd_hist.append({'step': step+1, 'feature': best_f, 'auc': f'{best_a:.4f}'})
    print(f'  Step {step+1}: +{best_f} (AUC={best_a:.4f})')
    if step >= 3 and max(float(h['auc']) for h in fwd_hist[-4:]) - min(float(h['auc']) for h in fwd_hist[-4:]) < 0.001:
        print(f'  Early stop at {step+1}'); break

pd.DataFrame(fwd_hist).to_csv(OUT/'tables'/'forward_selection.csv', index=False)
print(f'  Selected: {len(sel)} | Time: {time.time()-t0:.0f}s')

# ─── 7. RFE ───
print('\n[7/12] Recursive feature elimination...')
t0 = time.time()
rfe_f = candidates.copy()
rfe_hist = []
for step in range(min(30, len(rfe_f)-1)):
    if len(rfe_f) <= 1: break
    m = XGBClassifier(n_estimators=200, max_depth=6, lr=0.05, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=(1-y_fs_tr.mean())/y_fs_tr.mean())
    m.fit(X_fs_tr[rfe_f].values, y_fs_tr)
    imp = pd.Series(m.feature_importances_, index=rfe_f)
    wst = imp.idxmin()
    a = roc_auc_score(y_fs_va, m.predict_proba(X_fs_va[rfe_f].values)[:,1])
    rfe_hist.append({'step': step+1, 'removed': wst, 'auc': f'{a:.4f}'})
    if step < 5 or (step+1)%5==0: print(f'  Step {step+1}: -{wst} (AUC={a:.4f})')
    rfe_f.remove(wst)

pd.DataFrame(rfe_hist).to_csv(OUT/'tables'/'rfe_selection.csv', index=False)
print(f'  Remaining: {len(rfe_f)} | Time: {time.time()-t0:.0f}s')

# ─── 8. BORUTA ───
print('\n[8/12] Boruta-style...')
t0 = time.time()
boruta_f = candidates[:25] if len(candidates)>=25 else candidates
bor_n = 15000
bor_idx = S(len(X_train), bor_n)
X_b = X_train.iloc[bor_idx][boruta_f].copy(); y_b = y_train[bor_idx]
np.random.seed(42)
for c in boruta_f: X_b[f'shadow_{c}'] = np.random.permutation(X_b[c].values)
all_b = list(X_b.columns)
hits = {c:0 for c in boruta_f}
for it in range(50):
    m = XGBClassifier(n_estimators=100, max_depth=5, lr=0.05, tree_method='hist', n_jobs=-1, random_state=42+it, verbosity=0, scale_pos_weight=(1-y_b.mean())/y_b.mean())
    m.fit(X_b[all_b].values, y_b)
    imp = pd.Series(m.feature_importances_, index=all_b)
    ms = imp[[c for c in all_b if c.startswith('shadow_')]].max()
    for c in boruta_f:
        if imp[c] > ms: hits[c] += 1

from scipy.stats import binomtest
confirmed, rejected, tentative = [], [], []
for c, h in hits.items():
    p = binomtest(h, 50, 0.5, alternative='greater').pvalue
    if p < 0.05 and h > 32: confirmed.append(c)
    elif p > 0.2: rejected.append(c)
    else: tentative.append(c)

bor_df = pd.DataFrame([{'feature':c, 'hits':hits[c], 'hit_rate':hits[c]/50,
                         'status':'confirmed' if c in confirmed else ('rejected' if c in rejected else 'tentative')} for c in boruta_f])
bor_df = bor_df.sort_values('hits', ascending=False).reset_index(drop=True)
bor_df.to_csv(OUT/'tables'/'boruta_selection.csv', index=False)
print(f'  Confirmed: {len(confirmed)}, Rejected: {len(rejected)}, Tentative: {len(tentative)}')
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 9. GROUP ANALYSIS ───
print('\n[9/12] Feature group contribution...')
t0 = time.time()
def grp(f):
    fl = f.lower()
    if f.startswith('regime_'): return 'Regime'
    if 'rank' in fl and f.startswith('rank_'): return 'Cross-sectional'
    if f in ('dow','month','day_of_week','month_of_year','quarter') or any(f.startswith(p) for p in ('month_','dow_')): return 'Temporal'
    if any(f.startswith(p) for p in ('ret_','range_','vol_','hv_')): return 'Return/Volatility'
    if any(x in fl for x in ['lag_','_lag']): return 'Lagged'
    if any(x in fl for x in ['fvg','bos','choch','liq','mkt_','vol_profile']): return 'Market Structure'
    if any(x in fl for x in ['sma','ema','macd','rsi','adx','bb_','atr']): return 'Technical'
    if any(x in fl for x in ['volume','obv','vwap']): return 'Volume'
    return 'Other'
fg = {f:grp(f) for f in feature_cols}

g_auc = {}; g_n = {}
for g in sorted(set(fg.values())):
    feats = [f for f in feature_cols if fg[f]==g]
    if len(feats)<2: continue
    g_n[g]=len(feats)
    gi = S(len(X_train), 100000)
    Xg = X_train.iloc[gi][feats]; yg = y_train[gi]
    m = XGBClassifier(n_estimators=200, max_depth=5, lr=0.05, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=(1-yg.mean())/yg.mean())
    m.fit(Xg.values, yg)
    a = roc_auc_score(y_val, m.predict_proba(X_val[feats].values)[:,1])
    g_auc[g]=a
    print(f'  {g:25s} ({len(feats):2d}): AUC={a:.4f}')

gdf = pd.DataFrame([{'group':g,'n_features':g_n.get(g,0),'auc':g_auc.get(g,0)} for g in g_auc]).sort_values('auc', ascending=False)
gdf.to_csv(OUT/'tables'/'group_contribution.csv', index=False)
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 10. STABILITY ───
print('\n[10/12] Stability (bootstrap)...')
t0 = time.time()
from scipy.stats import spearmanr
bootstrap_imp = []
for b in range(15):
    bi = S(len(X_train), 30000)
    Xb = X_train.iloc[bi]; yb = y_train[bi]
    m = XGBClassifier(n_estimators=150, max_depth=5, lr=0.05, tree_method='hist', n_jobs=-1, random_state=42+b, verbosity=0, scale_pos_weight=(1-yb.mean())/yb.mean())
    m.fit(Xb.values, yb)
    bootstrap_imp.append(pd.Series(m.feature_importances_, index=feature_cols))

rank_dfs = [imp.rank(ascending=False) for imp in bootstrap_imp]
pair_corrs = []
for i in range(len(rank_dfs)):
    for j in range(i+1, len(rank_dfs)):
        r,_ = spearmanr(rank_dfs[i], rank_dfs[j]); pair_corrs.append(r)
stab_mean = np.mean(pair_corrs); stab_std = np.std(pair_corrs)
print(f'  Stability: {stab_mean:.3f} +/- {stab_std:.3f}')

feat_ranks = defaultdict(list)
for rd in rank_dfs:
    for f,r in rd.items(): feat_ranks[f].append(r)
fs_df = pd.DataFrame([{'feature':f,'mean_rank':np.mean(vs),'rank_std':np.std(vs),'rank_cv':np.std(vs)/max(np.mean(vs),1)} for f,vs in feat_ranks.items()]).sort_values('mean_rank')
fs_df.to_csv(OUT/'tables'/'stability_analysis.csv', index=False)
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 11. CONSENSUS ───
print('\n[11/12] Consensus ranking...')
t0 = time.time()
mi_m = dict(zip(mi_df.feature, mi_df.mi_rank))
xgb_m = dict(zip(xgb_imp.feature, xgb_imp.xgb_rank))
lgb_m = dict(zip(lgb_imp.feature, lgb_imp.lgb_rank))
perm_m = dict(zip(perm_imp.feature, perm_imp.perm_rank))
fwd_m = {f:i+1 for i,f in enumerate(sel)}
bor_m = dict(zip(bor_df.feature, [1 if s=='confirmed' else (2 if s=='tentative' else 3) for s in bor_df.status]))
vif_s = set(survived_vif)
rfe_s = set(rfe_f)
ga_m = {f:g_auc.get(fg.get(f,'Other'),0) for f in feature_cols}
sf_m = dict(zip(fs_df.feature, fs_df.mean_rank))

N = len(feature_cols)
rows = []
for f in feature_cols:
    mn = mi_m.get(f,N)/N
    xn = xgb_m.get(f,N)/N
    ln = lgb_m.get(f,N)/N
    pn = perm_m.get(f,N)/N
    fn2 = fwd_m.get(f,999)/999
    bn = bor_m.get(f,4)/4
    vs = 1-(1 if f in vif_s else 0)
    rs = 1-(1 if f in rfe_s else 0)
    gn = 1-ga_m.get(f,0)
    cs = mn*0.20+xn*0.20+ln*0.10+pn*0.15+fn2*0.10+bn*0.10+vs*0.05+rs*0.05+gn*0.05
    rows.append({'feature':f,'group':fg.get(f,'Other'),'mi_rank':mi_m.get(f,999),'xgb_rank':xgb_m.get(f,999),
                 'lgb_rank':lgb_m.get(f,999),'perm_rank':perm_m.get(f,999),'fwd_rank':fwd_m.get(f,999),
                 'boruta_status':bor_m.get(f,4),'vif_survived':1 if f in vif_s else 0,'rfe_survived':1 if f in rfe_s else 0,
                 'group_auc':ga_m.get(f,0),'consensus_score':cs})

cd = pd.DataFrame(rows).sort_values('consensus_score').reset_index(drop=True)
cd['consensus_rank'] = range(1,len(cd)+1)
cd.to_csv(OUT/'tables'/'consensus_ranking.csv', index=False)
final = cd.head(35)['feature'].tolist()
print(f'\n=== FINAL {len(final)} FEATURES ===')
for i,f in enumerate(final,1): print(f'  {i:2d}. {f}')
pd.DataFrame(final, columns=['feature']).to_csv(OUT/'tables'/'final_selected_features.csv', index=False)
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 12. CHARTS ───
print('\n[12/12] Charts...')
t0 = time.time()
plt.rcParams.update({'figure.dpi':100,'savefig.dpi':150,'font.size':9})

# 12a. MI top 20
fig,ax=plt.subplots(figsize=(10,7))
top=mi_df.head(20)
ax.barh(range(20),top.mi_score,color=plt.cm.Blues(np.linspace(0.4,0.9,20))[::-1])
ax.set_yticks(range(20));ax.set_yticklabels(top.feature);ax.set_xlabel('Mutual Information');ax.set_title('Top 20 by Mutual Information');ax.invert_yaxis()
fig.tight_layout();fig.savefig(OUT/'charts'/'01_mi_top20.png');plt.close()

# 12b. VIF path
if len(vif_hist)>0:
    vh=pd.DataFrame(vif_hist)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(vh.iter,vh.vif,'o-',color='crimson')
    ax.axhline(10,ls='--',color='gray',alpha=0.7)
    ax.set_xlabel('Iteration');ax.set_ylabel('VIF');ax.set_title('VIF Elimination')
    for _,r in vh.head(10).iterrows(): ax.annotate(r.feature[:15],(r.iter,r.vif),fontsize=6,rotation=45,ha='left')
    fig.tight_layout();fig.savefig(OUT/'charts'/'02_vif_elimination.png');plt.close()

# 12c. XGB vs LGB
mg=xgb_imp.merge(lgb_imp,on='feature')
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(mg.xgb_importance,mg.lgb_importance,c='steelblue',alpha=0.6,edgecolors='w')
for _,r in mg.head(10).iterrows(): ax.annotate(r.feature[:20],(r.xgb_importance,r.lgb_importance),fontsize=6)
ax.set_xlabel('XGBoost');ax.set_ylabel('LightGBM');ax.set_title('XGB vs LGB Importance')
fig.tight_layout();fig.savefig(OUT/'charts'/'03_xgb_vs_lgb.png');plt.close()

# 12d. Forward AUC
if fwd_hist:
    fh=pd.DataFrame(fwd_hist)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(fh.step,fh.auc.astype(float),'o-',color='forestgreen',lw=2)
    ax.fill_between(fh.step,fh.auc.astype(float),alpha=0.15,color='forestgreen')
    ax.set_xlabel('Features');ax.set_ylabel('AUC');ax.set_title('Forward Selection AUC')
    for _,r in fh.iterrows():
        if float(r.auc)>=float(fh.auc.max())*0.98 or r.step<=5: ax.annotate(r.feature[:18],(r.step,float(r.auc)),fontsize=6,rotation=45,ha='left')
    fig.tight_layout();fig.savefig(OUT/'charts'/'04_forward_selection.png');plt.close()

# 12e. Boruta
fig,ax=plt.subplots(figsize=(10,6))
bdf=bor_df.sort_values('hits')
clr=['forestgreen' if s=='confirmed' else ('orange' if s=='tentative' else 'crimson') for s in bdf.status]
ax.barh(range(len(bdf)),bdf.hits,color=clr)
ax.axvline(32,ls='--',color='green',alpha=0.5,label='65% threshold')
ax.axvline(25,ls='--',color='orange',alpha=0.5,label='50% baseline')
ax.set_yticks(range(len(bdf)));ax.set_yticklabels(bdf.feature,fontsize=7)
ax.set_xlabel('Hits / 50');ax.set_title('Boruta-Style Selection');ax.legend(fontsize=8);ax.invert_yaxis()
fig.tight_layout();fig.savefig(OUT/'charts'/'05_boruta.png');plt.close()

# 12f. Group contribution
fig,ax=plt.subplots(figsize=(9,5))
cg=plt.cm.Set2(np.linspace(0,0.9,len(gdf)))
ax.bar(range(len(gdf)),gdf.auc,color=cg,edgecolor='gray')
for i,(_,r) in enumerate(gdf.iterrows()): ax.text(i,r.auc+0.003,f'n={r.n_features}',ha='center',fontsize=8)
ax.set_xticks(range(len(gdf)));ax.set_xticklabels(gdf.group,rotation=25,ha='right')
ax.set_ylabel('AUC');ax.set_title('Feature Group Contribution');ax.set_ylim(0,max(gdf.auc)*1.15)
fig.tight_layout();fig.savefig(OUT/'charts'/'06_group_contribution.png');plt.close()

# 12g. Stability heatmap
fig,ax=plt.subplots(figsize=(8,7))
cm=np.zeros((15,15))
for i in range(15):
    for j in range(15):
        r,_=spearmanr(rank_dfs[i],rank_dfs[j]);cm[i,j]=r
sns.heatmap(cm,annot=True,fmt='.2f',cmap='RdYlBu',vmin=0.5,vmax=1,ax=ax,cbar_kws={'label':'Spearman r'})
ax.set_title('Bootstrap Stability');ax.set_xlabel('Sample');ax.set_ylabel('Sample')
fig.tight_layout();fig.savefig(OUT/'charts'/'07_stability_heatmap.png');plt.close()

# 12h. Consensus ranking
fig,ax=plt.subplots(figsize=(12,6))
cft=cd.head(40)
ax.bar(range(40),cft.consensus_score,color=plt.cm.RdYlGn(1-cft.consensus_rank/len(cd)),edgecolor='gray',lw=0.5)
ax.set_xticks(range(40));ax.set_xticklabels(cft.feature,rotation=90,fontsize=6)
ax.set_xlabel('Feature');ax.set_ylabel('Score (lower=better)');ax.set_title('Consensus Ranking (Top 40)')
fig.tight_layout();fig.savefig(OUT/'charts'/'08_consensus_ranking.png');plt.close()

# 12i. Final features pie
fig,ax=plt.subplots(figsize=(8,5))
fgc=defaultdict(int)
for f in final: fgc[fg.get(f,'Other')]+=1
labs=list(fgc.keys());vals=list(fgc.values())
ax.pie(vals,labels=labs,autopct='%1.0f%%',colors=plt.cm.Set3(np.linspace(0,0.9,len(labs))),startangle=90)
ax.set_title(f'Final {len(final)} Features by Category')
fig.tight_layout();fig.savefig(OUT/'charts'/'09_final_features_pie.png');plt.close()

# 12j. Final feature importance composite
fig,ax=plt.subplots(figsize=(10,6))
fxf=np.array([xgb_m.get(f,999) for f in final])
flf=np.array([lgb_m.get(f,999) for f in final])
fmf=np.array([mi_m.get(f,999) for f in final])
def norm(v): return 1-(v-v.min())/max(v.max()-v.min(),1)
comb=(norm(fxf)+norm(flf)+norm(fmf))/3
ax.barh(range(len(final)),comb,color=plt.cm.viridis(np.linspace(0.2,0.9,len(final))))
ax.set_yticks(range(len(final)));ax.set_yticklabels(final,fontsize=7)
ax.set_xlabel('Normalized Combined Importance');ax.set_title('Final Features: Combined XGB+LGB+MI')
ax.invert_yaxis();fig.tight_layout();fig.savefig(OUT/'charts'/'10_final_importance.png');plt.close()

print(f'  Charts done | Time: {time.time()-t0:.0f}s')

# ─── SUMMARY ───
tt=time.time()-t_start
print(f'\n{"="*70}')
print(f'PHASE 8 COMPLETE | {tt:.0f}s ({tt/60:.1f} min)')
print(f'Final: {len(final)} selected from {len(feature_cols)} candidates')
print(f'Output: {OUT}')
print(f'{"="*70}')
for i,f in enumerate(final,1): print(f'  {i:2d}. {f}')

summary={'phase':8,'completed_at':datetime.now().isoformat(),'total_time_seconds':tt,
         'candidate_features':len(feature_cols),'final_features':len(final),'selected_features':final,
         'xgb_val_auc':float(xgb_auc),'lgb_val_auc':float(lgb_auc),'stability_mean':float(stab_mean),'stability_std':float(stab_std),
         'vif_removed':len(vif_hist),'forward_selected':len(sel),'rfe_remaining':len(rfe_f),
         'boruta_confirmed':len(confirmed),'boruta_rejected':len(rejected),
         'groups_count':{g:sum(1 for f in final if fg.get(f,'Other')==g) for g in sorted(set(fg.values()))}}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)

print(f'\nSummary saved to {OUT/"summary.json"}')
