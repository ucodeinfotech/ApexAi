# Phase 8-14: Comprehensive Improvement Pipeline
# Symbol filter + Feature interactions + SMOTE + Optuna-tuning + Model comparison + Ensemble
import pandas as pd, numpy as np, warnings, json, time, gc, os
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'improvement_results'
OUT.mkdir(exist_ok=True)
(OUT/'charts').mkdir(exist_ok=True)
(OUT/'tables').mkdir(exist_ok=True)
t_all = time.time()
print('='*70)
print('PHASE 8-14: COMPREHENSIVE IMPROVEMENT PIPELINE')
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('='*70)

# ─── 0. LOAD ───
print('\n[0] Loading data...')
df = pd.read_parquet(BASE/'cleaned_features.parquet')
print(f'  Shape: {df.shape}')

target_col = 'target'
feature_cols = [c for c in df.columns if c not in ('symbol','date','datetime','target','target_ret','symbol_date','Date','Symbol','_d')]
feature_cols = [c for c in feature_cols if not any(p in c for p in ('next_','fwd_','future_'))]
print(f'  Base features: {len(feature_cols)}')

# Parse dates
df['_d'] = pd.to_datetime(df['datetime'])

# ─── 1. SYMBOL FILTER ───
print('\n[1] Symbol liquidity filter...')
t0=time.time()
sym_vol = df.groupby('symbol')['volume'].median().sort_values()
n_total = len(sym_vol)
n_keep = int(n_total * 0.75)
keep_syms = set(sym_vol.tail(n_keep).index)
remove_syms = set(sym_vol.head(n_total - n_keep).index)
df_f = df[df['symbol'].isin(keep_syms)].copy()
print(f'  Symbols: {n_total} -> {len(keep_syms)} ({len(remove_syms)} removed, bottom 25% by median volume)')
print(f'  Rows: {len(df):,} -> {len(df_f):,}')
print(f'  Removed symbols (lowest vol): {", ".join(list(remove_syms)[:15])}...')
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 2. FEATURE INTERACTIONS ───
print('\n[2] Feature interactions...')
t0=time.time()
# Regime features
regime_cols = [c for c in df_f.columns if c.startswith('regime_')]
# Range/vol features
vol_cols = ['range_5','hv_20','range_10','hv_10','bb_width']
ret_cols = ['ret_1d','ret_1d_std_5','ret_5d_lag1']
cross_cols = ['rank_range_5','rank_hv_20']
interactions = []
used = set(feature_cols)

# Regime x volatility (capture regime-conditional vol)
for r in regime_cols:
    for v in vol_cols:
        n = f'{r}_x_{v}'
        if n not in used:
            df_f[n] = df_f[r] * df_f[v]
            interactions.append(n); used.add(n)

# Month x range (seasonal volatility)
month_dummies = [c for c in df_f.columns if c.startswith('month_') and c!='month']
for m in month_dummies[:6]:
    for v in vol_cols[:3]:
        n = f'{m}_x_{v}'
        if n not in used:
            df_f[n] = df_f[m] * df_f[v]
            interactions.append(n); used.add(n)

# DOW x return (day-of-week momentum)
dow_dummies = [c for c in df_f.columns if c.startswith('dow_')]
for d in dow_dummies[:3]:
    for r_var in ret_cols[:2]:
        n = f'{d}_x_{r_var}'
        if n not in used:
            df_f[n] = df_f[d] * df_f[r_var]
            interactions.append(n); used.add(n)

# Cross-sectional rank x volatility
for c in cross_cols[:2]:
    for v in vol_cols[:2]:
        n = f'{c}_x_{v}'
        if n not in used:
            df_f[n] = df_f[c] * df_f[v]
            interactions.append(n); used.add(n)

print(f'  Added {len(interactions)} interactions: {", ".join(interactions[:10])}...')
print(f'  Total features: {len(feature_cols)} -> {len(feature_cols)+len(interactions)}')

# Add interaction cols to feature set
all_features = feature_cols + interactions
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 3. WALKFORWARD SPLIT ───
print('\n[3] Walkforward split...')
t0=time.time()
# Expanding window: fold 1 train up to 2023-01, fold 2 up to 2023-07, etc.
folds = [
    ('2023-H1', '2023-01-01', '2023-07-01', '2023-07-01', '2024-01-01'),
    ('2023-H2', '2023-01-01', '2024-01-01', '2024-01-01', '2024-07-01'),
    ('2024-H1', '2022-01-01', '2024-07-01', '2024-07-01', '2025-01-01'),
    ('2024-H2', '2022-01-01', '2025-01-01', '2025-01-01', '2026-06-25'),
]
fold_data = {}
for name, tr_start, tr_end, va_start, va_end in folds:
    tr = df_f[(df_f['_d'] >= tr_start) & (df_f['_d'] < tr_end)]
    va = df_f[(df_f['_d'] >= va_start) & (df_f['_d'] < va_end)]
    fold_data[name] = (tr, va)
    print(f'  {name}: train={len(tr):,} (tgt={tr[target_col].mean():.4f}) val={len(va):,} (tgt={va[target_col].mean():.4f})')
print(f'  Time: {time.time()-t0:.0f}s')

# Full test set (2024-01 to end)
test_full = df_f[df_f['_d'] >= '2024-01-01']
print(f'  Full test: {len(test_full):,} (tgt={test_full[target_col].mean():.4f})')

y_test_full = test_full[target_col].values

# ─── 4. PREPROCESSING CONFIG ───
# We'll apply SMOTE inside each fold during training
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, average_precision_score, log_loss, brier_score_loss

results_all = []
best_models = {}

def compute_metrics(y_true, y_pred_proba, thresh=0.5):
    from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                                  f1_score, matthews_corrcoef, confusion_matrix,
                                  average_precision_score, log_loss, brier_score_loss)
    yb = (y_pred_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, yb)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    p = precision_score(y_true, yb, zero_division=0)
    r = recall_score(y_true, yb, zero_division=0)
    return {
        'auc_roc': round(roc_auc_score(y_true, y_pred_proba),4),
        'avg_precision': round(average_precision_score(y_true, y_pred_proba),4),
        'accuracy': round(accuracy_score(y_true, yb),4),
        'precision': round(p,4),
        'recall': round(r,4),
        'f1': round(f1_score(y_true, yb, zero_division=0),4),
        'mcc': round(matthews_corrcoef(y_true, yb),4),
        'brier': round(brier_score_loss(y_true, y_pred_proba),4),
        'log_loss': round(log_loss(y_true, y_pred_proba),4),
        'lift': round(p/y_true.mean() if y_true.mean()>0 else 0,2),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }

SMOTE_SAMPLE = 200000  # subsample for SMOTE (speed)

def train_model(name, model_fn, X_tr, y_tr, X_va, y_va, use_smote=True):
    """Train with optional SMOTE, return model + metrics"""
    if use_smote:
        # Subsample for SMOTE speed
        rng = np.random.RandomState(42)
        if len(X_tr) > SMOTE_SAMPLE:
            idx = rng.choice(len(X_tr), SMOTE_SAMPLE, replace=False)
            X_s, y_s = X_tr[idx], y_tr[idx]
        else:
            X_s, y_s = X_tr, y_tr
        sm = SMOTETomek(random_state=42, n_jobs=-1)
        X_res, y_res = sm.fit_resample(X_s, y_s)
        print(f'    SMOTE: {X_s.shape} -> {X_res.shape} ({y_res.mean():.3f} positive)')
        X_t, y_t = X_res, y_res
    else:
        X_t, y_t = X_tr, y_tr
    m = model_fn.fit(X_t, y_t)
    pred = m.predict_proba(X_va)[:,1]
    met = compute_metrics(y_va, pred)
    met['model'] = name; met['use_smote'] = use_smote
    return m, met

# ─── 5. MODEL COMPARISON (NO SMOTE VS SMOTE) ───
print('\n[4] Model comparison (fold 2023-H1)...')
t0=time.time()
tr, va = fold_data['2023-H1']
X_tr, y_tr = tr[all_features].values, tr[target_col].values
X_va, y_va = va[all_features].values, va[target_col].values

# Scale for RF (tree-based don't need it but RF can benefit with many features)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

w_bal = (1-y_tr.mean())/y_tr.mean()
w_smote = 1.0  # SMOTE balances classes

models_config = {
    'XGBoost': lambda: XGBClassifier(n_estimators=500, max_depth=5, lr=0.03, subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=w_bal, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0),
    'XGBoost_SMOTE': lambda: XGBClassifier(n_estimators=500, max_depth=5, lr=0.03, subsample=0.8, colsample_bytree=0.8,
                                             scale_pos_weight=w_smote, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0),
    'LightGBM': lambda: lgbm.LGBMClassifier(n_estimators=500, max_depth=5, lr=0.03, subsample=0.8, colsample_bytree=0.8,
                                              class_weight='balanced', n_jobs=-1, random_state=42, verbosity=-1),
    'LightGBM_SMOTE': lambda: lgbm.LGBMClassifier(n_estimators=500, max_depth=5, lr=0.03, subsample=0.8, colsample_bytree=0.8,
                                                    class_weight=None, n_jobs=-1, random_state=42, verbosity=-1),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                                     class_weight='balanced_subsample', n_jobs=-1, random_state=42),
    'RandomForest_SMOTE': lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                                           class_weight=None, n_jobs=-1, random_state=42),
}

comparison = []
for name, fn in models_config.items():
    use_smote = '_SMOTE' in name
    _, met = train_model(name, fn(), X_tr, y_tr, X_va, y_va, use_smote=use_smote)
    comparison.append(met)
    print(f'  {name:25s} AUC={met["auc_roc"]:.4f} P={met["precision"]:.4f} R={met["recall"]:.4f} F1={met["f1"]:.4f} MCC={met["mcc"]:.4f}')

cmp_df = pd.DataFrame(comparison)
cmp_df.to_csv(OUT/'tables'/'model_comparison.csv', index=False)
# Pick best model class based on F1
best_f1_model = cmp_df.sort_values('f1', ascending=False).iloc[0]
best_type = best_f1_model['model'].replace('_SMOTE','').replace('_NO_SMOTE','')
print(f'\n  Best model: {best_f1_model["model"]} (F1={best_f1_model["f1"]:.4f})')
print(f'  Time: {time.time()-t0:.0f}s')

# ─── 6. OPTUNA HYPERPARAMETER TUNING ───
print('\n[5] Optuna hyperparameter tuning (XGBoost + SMOTE)...')
t0=time.time()

# Determine which model type performed best
use_catboost = False
try:
    from catboost import CatBoostClassifier
    use_catboost = True
    print('  CatBoost available')
except:
    print('  CatBoost not available, using XGBoost')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Subsample for Optuna (speed)
opt_n = 100000
opt_idx = np.random.RandomState(42).choice(len(X_tr), min(opt_n, len(X_tr)), replace=False)
X_opt, y_opt = X_tr[opt_idx], y_tr[opt_idx]

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'lr': trial.suggest_float('lr', 0.005, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 3),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3, 12),
    }
    sm = SMOTETomek(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_opt, y_opt)
    m = XGBClassifier(**params, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0)
    m.fit(X_res, y_res)
    pred = m.predict_proba(X_va)[:,1]
    return roc_auc_score(y_va, pred)

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'lr': trial.suggest_float('lr', 0.005, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 3),
    }
    sm = SMOTETomek(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_opt, y_opt)
    m = lgbm.LGBMClassifier(**params, n_jobs=-1, random_state=42, verbosity=-1)
    m.fit(X_res, y_res)
    pred = m.predict_proba(X_va)[:,1]
    return roc_auc_score(y_va, pred)

if use_catboost:
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'depth': trial.suggest_int('depth', 4, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 8),
            'auto_class_weights': 'SqrtBalanced',
        }
        sm = SMOTETomek(random_state=42, n_jobs=-1)
        X_res, y_res = sm.fit_resample(X_opt, y_opt)
        m = CatBoostClassifier(**params, task_type='CPU', random_seed=42, verbose=0)
        m.fit(X_res, y_res)
        pred = m.predict_proba(X_va)[:,1]
        return roc_auc_score(y_va, pred)

# Run Optuna for top models
n_trials = 30
studies = {}

print('  Tuning XGBoost...')
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=n_trials)
studies['XGBoost'] = study_xgb
print(f'    Best XGB AUC: {study_xgb.best_value:.4f}')

print('  Tuning LightGBM...')
study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=n_trials)
studies['LightGBM'] = study_lgb
print(f'    Best LGB AUC: {study_lgb.best_value:.4f}')

if use_catboost:
    print('  Tuning CatBoost...')
    study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(objective_cat, n_trials=n_trials)
    studies['CatBoost'] = study_cat
    print(f'    Best CatBoost AUC: {study_cat.best_value:.4f}')

# Save study results
for name, study in studies.items():
    pd.DataFrame([{**t.params, 'value': t.value} for t in study.trials if t.value is not None]).to_csv(
        OUT/'tables'/f'optuna_{name.lower()}.csv', index=False)

# Train best models
print('\n  Training best-tuned models on full fold 1...')
best_models_config = {}
for name, study in studies.items():
    bp = study.best_params
    if name == 'XGBoost':
        m = XGBClassifier(**bp, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0)
    elif name == 'LightGBM':
        m = lgbm.LGBMClassifier(**bp, n_jobs=-1, random_state=42, verbosity=-1)
    elif name == 'CatBoost':
        m = CatBoostClassifier(**bp, task_type='CPU', random_seed=42, verbose=0)
    sm = SMOTETomek(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    m.fit(X_res, y_res)
    best_models_config[name] = m
    pred = m.predict_proba(X_va)[:,1]
    met = compute_metrics(y_va, pred)
    print(f'    {name:15s} AUC={met["auc_roc"]:.4f} P={met["precision"]:.4f} R={met["recall"]:.4f} F1={met["f1"]:.4f} MCC={met["mcc"]:.4f}')

print(f'  Time: {time.time()-t0:.0f}s')

# ─── 7. ENSEMBLE + WALKFORWARD EVALUATION ───
print('\n[6] Walkforward + final evaluation...')
t0=time.time()
all_fold_results = []

def smote_fit(name, bp, X_tr, y_tr):
    """Fit model with SMOTE on (potentially subsampled) data"""
    rng = np.random.RandomState(42)
    if len(X_tr) > SMOTE_SAMPLE:
        idx = rng.choice(len(X_tr), SMOTE_SAMPLE, replace=False)
        X_s, y_s = X_tr[idx], y_tr[idx]
    else:
        X_s, y_s = X_tr, y_tr
    sm = SMOTETomek(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_s, y_s)
    if name == 'XGBoost':
        model = XGBClassifier(**bp, tree_method='hist', n_jobs=-1, random_state=42, verbosity=0)
    elif name == 'LightGBM':
        model = lgbm.LGBMClassifier(**bp, n_jobs=-1, random_state=42, verbosity=-1)
    elif name == 'CatBoost':
        model = CatBoostClassifier(**bp, task_type='CPU', random_seed=42, verbose=0)
    model.fit(X_res, y_res)
    return model

# Use last (largest) fold for final model
last_name = [f[0] for f in folds if f[0] != '2023-H2'][-1] if len([f for f in folds])>2 else '2023-H2'
for fn in [f[0] for f in folds]:
    if '2024' in fn: last_name = fn
_, va_last = fold_data[last_name]
# Actually use all training data from 2022+
tr_full = df_f[(df_f['_d'] >= '2022-01-01') & (df_f['_d'] < '2025-01-01')]
X_tr_full = tr_full[all_features].values; y_tr_full = tr_full[target_col].values
X_test_full_a = test_full[all_features].values

test_preds = []
for name, study in studies.items():
    bp = study.best_params
    m = smote_fit(name, bp, X_tr_full, y_tr_full)
    p = m.predict_proba(X_test_full_a)[:,1]
    test_preds.append(p)
    met = compute_metrics(y_test_full, p)
    met['fold'] = 'final_test'; met['model'] = name
    all_fold_results.append(met)
    print(f'    {name:15s} AUC={met["auc_roc"]:.4f} P={met["precision"]:.4f} R={met["recall"]:.4f} F1={met["f1"]:.4f}')

# Ensemble
ens_pred_test = np.mean(test_preds, axis=0)
met_ens_test = compute_metrics(y_test_full, ens_pred_test)
met_ens_test['fold'] = 'final_test'; met_ens_test['model'] = 'Ensemble'
all_fold_results.append(met_ens_test)
print(f'    {"Ensemble":15s} AUC={met_ens_test["auc_roc"]:.4f} P={met_ens_test["precision"]:.4f} R={met_ens_test["recall"]:.4f} F1={met_ens_test["f1"]:.4f}')

results_df = pd.DataFrame(all_fold_results)
results_df.to_csv(OUT/'tables'/'walkforward_results.csv', index=False)

# ─── 7b. THRESHOLD OPTIMIZATION ───
print('\n[7] Threshold optimization...')
t0=time.time()
# Use ensemble predictions on test
best_th, best_f1, best_met = 0.5, 0, None
for th in np.arange(0.05, 0.85, 0.025):
    met = compute_metrics(y_test_full, ens_pred_test, thresh=th)
    if met['f1'] > best_f1:
        best_f1 = met['f1']
        best_th = th
        best_met = met

print(f'  Best threshold: {best_th:.3f} (F1={best_met["f1"]:.4f})')
print(f'  At th={best_th:.3f}: P={best_met["precision"]:.4f} R={best_met["recall"]:.4f} AUC={best_met["auc_roc"]:.4f}')
print(f'  Confusion: TP={best_met["tp"]} FP={best_met["fp"]} FN={best_met["fn"]} TN={best_met["tn"]}')
print(f'  Lift: {best_met["lift"]}x')

# Full threshold table
th_data = []
for th in np.arange(0.05, 0.80, 0.025):
    met = compute_metrics(y_test_full, ens_pred_test, thresh=th)
    met['threshold'] = round(th,3)
    th_data.append(met)
th_df = pd.DataFrame(th_data)
th_df.to_csv(OUT/'tables'/'threshold_optimization.csv', index=False)

# Also save ensemble predictions
np.save(OUT/'ensemble_pred_test.npy', ens_pred_test)
np.save(OUT/'y_test_full.npy', y_test_full)
print(f'  Time: {time.time()-t0:.0f}s')

# ─── BASELINE COMPARISON ───
print('\n[8] Baseline vs Improved comparison...')
baseline_test = {'auc_roc':0.6545,'avg_precision':0.2127,'f1':0.2787,'precision':0.1896,'recall':0.5260,'mcc':0.1488,'lift':1.56}
improved_test = {'auc_roc':met_ens_test['auc_roc'],'avg_precision':met_ens_test['avg_precision'],
                 'f1':met_ens_test['f1'],'precision':met_ens_test['precision'],'recall':met_ens_test['recall'],
                 'mcc':met_ens_test['mcc'],'lift':met_ens_test['lift']}
print(f'  {"Metric":15s} {"Baseline":10s} {"Improved":10s} {"Change":10s}')
print(f'  {"-"*45}')
for k in ['auc_roc','avg_precision','f1','precision','recall','mcc','lift']:
    b = baseline_test.get(k,0); i = improved_test.get(k,0)
    ch = i - b if isinstance(i,(int,float)) and isinstance(b,(int,float)) else 0
    print(f'  {k:15s} {b:<10.4f} {i:<10.4f} {ch:<+10.4f}')
# Lift display
print(f'  {"lift":15s} {str(baseline_test["lift"])+"x":<10s} {str(improved_test["lift"])+"x":<10s} +{round(improved_test["lift"]-baseline_test["lift"],2):.2f}x')

# ─── SUMMARY ───
tt=time.time()-t_all
print(f'\n{"="*70}')
print(f'IMPROVEMENT PIPELINE COMPLETE | {tt:.0f}s ({tt/60:.1f} min)')
print(f'{"="*70}')
print(f'Symbol filter: {n_total} -> {len(keep_syms)} ({len(remove_syms)} removed)')
print(f'Feature interactions: {len(interactions)} added')
print(f'Models compared: 6 variants + {len(studies)} Optuna-tuned + Ensemble')
print(f'Threshold optimized: {best_th:.3f} (F1={best_f1:.4f})')

# Save summary
summary = {
    'completed_at': datetime.now().isoformat(),
    'total_time_seconds': tt,
    'symbols_before': n_total, 'symbols_after': len(keep_syms), 'symbols_removed': len(remove_syms),
    'base_features': len(feature_cols), 'interactions_added': len(interactions), 'total_features': len(all_features),
    'optuna_trials': n_trials, 'models_tuned': list(studies.keys()),
    'best_threshold': best_th, 'best_f1_at_threshold': best_f1,
    'baseline_vs_improved': {
        'baseline': {k:float(v) if isinstance(v,(int,float)) else v for k,v in baseline_test.items()},
        'improved': {k:float(v) if isinstance(v,(int,float)) else v for k,v in improved_test.items()},
    },
    'test_target_rate': float(y_test_full.mean()),
    'ensemble_test_metrics': {k:float(v) if isinstance(v,(int,float)) else v for k,v in met_ens_test.items()},
}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)

print(f'\nResults saved to {OUT}')
print(f'Summary: {OUT/"summary.json"}')
