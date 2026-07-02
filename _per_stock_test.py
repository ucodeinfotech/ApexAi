import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
df = pd.read_parquet(BASE/'improved_features.parquet')
tc='target'; dc='_d'
top20=['regime_2','range_5_ma_21','regime_3','range_5','range_5_lag1','hv_20_lag1','rank_hv_20',
       'dow_2','ret_1d_std_5','hv_10','rank_range_5','dow','ret_1d_lag2','regime_0','range_10',
       'ret_1d','regime_1','month_12','hv_20_std_3','vol_ratio_5_ma_30']
stocks=['RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','SBIN','BHARTIARTL','ITC','KOTAKBANK','LT']

all_tr=df[(df[dc]>='2016-01-01')&(df[dc]<'2023-01-01')]
w_all=(1-all_tr[tc].mean())/all_tr[tc].mean()
mp=XGBClassifier(n_estimators=800,max_depth=7,lr=0.06,subsample=0.85,colsample_bytree=0.8,
                  min_child_weight=3,gamma=1,reg_alpha=0.5,reg_lambda=0.5,scale_pos_weight=w_all,
                  tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
mp.fit(all_tr[top20].values.astype(np.float32),all_tr[tc].values)

hdr = f"{'Stock':12s} {'Rows':>7s} {'TrRows':>7s} {'TeRows':>6s} {'TgtRt':>7s} {'AUCpool':>9s} {'AUCsing':>9s} {'F1sing':>8s} {'Prec':>7s} {'Rec':>7s}"
print(hdr); print('-'*72)
for sym in stocks:
    sd=df[df['symbol']==sym].copy()
    tr=sd[(sd[dc]>='2016-01-01')&(sd[dc]<'2023-01-01')]
    te=sd[(sd[dc]>='2024-01-01')&(sd[dc]<'2026-06-26')]
    pp=mp.predict_proba(te[top20].values.astype(np.float32))[:,1]
    auc_p=roc_auc_score(te[tc].values,pp)
    if len(tr)>200 and tr[tc].sum()>15 and te[tc].sum()>5:
        ws=(1-tr[tc].mean())/tr[tc].mean()
        ms=XGBClassifier(n_estimators=200,max_depth=4,lr=0.03,subsample=0.8,colsample_bytree=0.8,
                          min_child_weight=5,gamma=2,reg_alpha=2,reg_lambda=2,scale_pos_weight=ws,
                          tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
        ms.fit(tr[top20].values.astype(np.float32),tr[tc].values)
        ps=ms.predict_proba(te[top20].values.astype(np.float32))[:,1]
        auc_s=roc_auc_score(te[tc].values,ps)
        f1_s=f1_score(te[tc].values,(ps>=0.5).astype(int))
        pred_b=(ps>=0.5).astype(int)
        prec=te[tc].values[pred_b==1].mean() if pred_b.sum()>0 else 0.0
        rec=te[tc].values[pred_b==1].sum()/te[tc].sum() if te[tc].sum()>0 else 0.0
    else:
        auc_s,f1_s,prec,rec=-1,-1,0,0
    print(f'{sym:12s} {len(sd):>7,} {len(tr):>7,} {len(te):>6,} {te[tc].mean():>7.3f} {auc_p:>8.4f}  {auc_s:>8.4f}  {f1_s:>7.4f}  {prec:>6.3f}  {rec:>6.3f}')
