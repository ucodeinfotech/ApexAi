# Phase 8 Feature Set Evaluation: Full Metrics
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'feature_selection_results'
final_features = pd.read_csv(OUT/'tables'/'final_selected_features.csv')['feature'].tolist()
print(f'Loading {len(final_features)} selected features...')

df = pd.read_parquet(BASE/'cleaned_features.parquet')
target = 'target'
# Temporal split
df['_d'] = pd.to_datetime(df['datetime'])
train = df[df['_d'] < '2023-01-01']
val   = df[(df['_d'] >= '2023-01-01') & (df['_d'] < '2024-01-01')]
test  = df[df['_d'] >= '2024-01-01']
print(f'Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')

X_tr, y_tr = train[final_features].values, train[target].values
X_va, y_va = val[final_features].values, val[target].values
X_te, y_te = test[final_features].values, test[target].values
w = (1-y_tr.mean())/y_tr.mean()
print(f'Target rate: train={y_tr.mean():.4f} val={y_va.mean():.4f} test={y_te.mean():.4f}')
print(f'Weight: {w:.2f}')

from xgboost import XGBClassifier
import lightgbm as lgbm
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, confusion_matrix, classification_report,
                              average_precision_score, log_loss, brier_score_loss)

models = {}
# XGBoost
print('\n--- XGBoost ---')
try:
    xgb = XGBClassifier(n_estimators=2000, max_depth=5, lr=0.03, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=w, tree_method='hist', device='cuda',
                         early_stopping_rounds=100, eval_metric='auc', random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print('  GPU OK')
except:
    xgb = XGBClassifier(n_estimators=2000, max_depth=5, lr=0.03, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=w, tree_method='hist', n_jobs=-1,
                         early_stopping_rounds=100, eval_metric='auc', random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print('  CPU fallback')
models['XGBoost'] = xgb
best_iter = xgb.best_iteration if hasattr(xgb,'best_iteration') else 2000
print(f'  Best iteration: {best_iter}')

# LightGBM
print('\n--- LightGBM ---')
try:
    lg_model = lgbm.LGBMClassifier(n_estimators=500, max_depth=7, lr=0.05, subsample=0.8, colsample_bytree=0.8,
                                    class_weight='balanced', device='gpu', gpu_platform_id=0, gpu_device_id=0,
                                    random_state=42, verbosity=-1)
    lg_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgbm.early_stopping(50)])
    print('  GPU OK')
except:
    lg_model = lgbm.LGBMClassifier(n_estimators=500, max_depth=7, lr=0.05, subsample=0.8, colsample_bytree=0.8,
                                    class_weight='balanced', n_jobs=-1, random_state=42, verbosity=-1)
    lg_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgbm.early_stopping(50)])
    print('  CPU fallback')
models['LightGBM'] = lg_model

def evaluate_model(name, model, X_va, y_va, X_te, y_te):
    for dataset, X, y, label in [('Validation', X_va, y_va, 'Val'), ('Test', X_te, y_te, 'Test')]:
        pred = model.predict_proba(X)[:,1]
        pred_bin = (pred >= 0.5).astype(int)
        cm = confusion_matrix(y, pred_bin)
        tn, fp, fn, tp = cm.ravel()
        best_thresh = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.05):
            pb = (pred >= th).astype(int)
            f = f1_score(y, pb, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = th
        pred_bin_opt = (pred >= best_thresh).astype(int)
        cm_opt = confusion_matrix(y, pred_bin_opt)
        metrics = {
            'dataset': label,
            'model': name,
            'auc_roc': round(roc_auc_score(y, pred), 4),
            'avg_precision': round(average_precision_score(y, pred), 4),
            'log_loss': round(log_loss(y, pred), 4),
            'brier': round(brier_score_loss(y, pred), 4),
            'accuracy': round(accuracy_score(y, pred_bin), 4),
            'precision': round(precision_score(y, pred_bin, zero_division=0), 4),
            'recall': round(recall_score(y, pred_bin, zero_division=0), 4),
            'f1': round(f1_score(y, pred_bin, zero_division=0), 4),
            'mcc': round(matthews_corrcoef(y, pred_bin), 4),
            'best_thresh': round(best_thresh, 2),
            'f1_optimal': round(best_f1, 4),
            'precision_opt': round(precision_score(y, pred_bin_opt, zero_division=0), 4),
            'recall_opt': round(recall_score(y, pred_bin_opt, zero_division=0), 4),
            'mcc_opt': round(matthews_corrcoef(y, pred_bin_opt), 4),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'lift': round((tp/(tp+fp))/(y.mean()) if (tp+fp)>0 else 0, 2),
            'gain': round(tp/(tp+fn), 4),
            'target_rate': round(y.mean(), 4),
        }
        yield metrics

results = []
for name, model in models.items():
    for r in evaluate_model(name, model, X_va, y_va, X_te, y_te):
        results.append(r)
        print(f'\n=== {r["model"]} - {r["dataset"]} ===')
        print(f'  AUC-ROC: {r["auc_roc"]:.4f} | Avg Precision: {r["avg_precision"]:.4f}')
        print(f'  Accuracy: {r["accuracy"]:.4f} | Precision: {r["precision"]:.4f} | Recall: {r["recall"]:.4f} | F1: {r["f1"]:.4f}')
        print(f'  MCC: {r["mcc"]:.4f} | LogLoss: {r["log_loss"]:.4f} | Brier: {r["brier"]:.4f}')
        print(f'  Confusion: TN={r["tn"]} FP={r["fp"]} FN={r["fn"]} TP={r["tp"]}')
        print(f'  Optimal threshold: {r["best_thresh"]:.2f} (F1={r["f1_optimal"]:.4f}, P={r["precision_opt"]:.4f}, R={r["recall_opt"]:.4f})')
        print(f'  Lift: {r["lift"]:.2f}x | Gain: {r["gain"]:.4f}')

# Combined summary table
print('\n\n=== COMBINED METRICS TABLE ===')
print(f'{"Model":12s} {"Set":6s} {"AUC":8s} {"AP":8s} {"Acc":8s} {"Prec":8s} {"Recall":8s} {"F1":8s} {"MCC":8s} {"Lift":8s} {"Brier":8s}')
print('-'*84)
for r in results:
    print(f'{r["model"]:12s} {r["dataset"]:6s} {r["auc_roc"]:.4f}  {r["avg_precision"]:.4f}  {r["accuracy"]:.4f}  {r["precision"]:.4f}  {r["recall"]:.4f}  {r["f1"]:.4f}  {r["mcc"]:.4f}  {r["lift"]:.2f}x  {r["brier"]:.4f}')

# Save
pd.DataFrame(results).to_csv(OUT/'tables'/'model_evaluation_metrics.csv', index=False)
print(f'\nResults saved to {OUT/"tables"/"model_evaluation_metrics.csv"}')

# Threshold analysis
print('\n\n=== THRESHOLD ANALYSIS (XGBoost Test) ===')
pred_test = models['XGBoost'].predict_proba(X_te)[:,1]
print(f'{"Thresh":8s} {"Acc":8s} {"Prec":8s} {"Recall":8s} {"F1":8s} {"MCC":8s} {"TP":6s} {"FP":6s} {"FN":6s} {"TN":6s} {"Lift":8s}')
print('-'*78)
for th in np.arange(0.05, 0.75, 0.05):
    pb = (pred_test >= th).astype(int)
    cm = confusion_matrix(y_te, pb)
    tn, fp, fn, tp = cm.ravel()
    p = precision_score(y_te, pb, zero_division=0)
    r = recall_score(y_te, pb, zero_division=0)
    f = f1_score(y_te, pb, zero_division=0)
    m = matthews_corrcoef(y_te, pb)
    a = accuracy_score(y_te, pb)
    l = (p/y_te.mean()) if p>0 else 0
    print(f'{th:.2f}     {a:.4f}  {p:.4f}  {r:.4f}  {f:.4f}  {m:.4f}  {tp:5d} {fp:5d} {fn:5d} {tn:5d}  {l:.2f}x')

print('\n=== FEATURE IMPORTANCE (XGBoost) ===')
fi = pd.DataFrame({'feature': final_features, 'importance': models['XGBoost'].feature_importances_}).sort_values('importance', ascending=False)
fi['cumulative'] = fi['importance'].cumsum()
print(f'{"Rank":5s} {"Feature":28s} {"Importance":12s} {"Cumulative":12s}')
print('-'*57)
for i, (_, r) in enumerate(fi.iterrows(), 1):
    print(f'{i:4d}. {r["feature"]:28s} {r["importance"]:.4f}      {r["cumulative"]:.4f}')
fi.to_csv(OUT/'tables'/'final_feature_importance.csv', index=False)

# Summary JSON
summary = {
    'n_features': len(final_features),
    'n_train': len(X_tr), 'n_val': len(X_va), 'n_test': len(X_te),
    'target_rate_train': float(y_tr.mean()), 'target_rate_val': float(y_va.mean()), 'target_rate_test': float(y_te.mean()),
    'results': {f'{r["model"]}_{r["dataset"]}': {k:v for k,v in r.items() if k not in ('dataset','model')} for r in results}
}
with open(OUT/'evaluation_summary.json','w') as f: json.dump(summary,f,indent=2)
print(f'\nFull summary: {OUT/"evaluation_summary.json"}')
