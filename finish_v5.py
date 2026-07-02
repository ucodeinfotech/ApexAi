"""Compute backtest and final results from results_raw.pkl"""
import pickle, pandas as pd, numpy as np, math, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'return_prediction_report_v5'
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size
    brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    total = brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2
    return total

def calc_metrics(s, n=252):
    if len(s)<5 or s.std()==0: return 0,0,0,0
    cagr=(1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh=s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr=(s>0).mean()
    dd=((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd

with open(OUT/'results_raw.pkl','rb') as f:
    all_results = pickle.load(f)
rd = pd.DataFrame(all_results)

print(f'Total predictions: {len(rd):,}')
cols = ['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']
avail = [c for c in cols if c in rd.columns]
print(f'Models: {avail}')

print(f'\n{"="*50}')
print(f'FINAL PERFORMANCE - v5 (fixed features + GPU)')
print(f'{"="*50}')
print(f'{"Model":8s} {"R2":>8s} {"Corr":>8s} {"DirAcc":>8s}')
for col in avail:
    r2 = r2_score(rd['act'], rd[col])
    corr = np.corrcoef(rd['act'], rd[col])[0,1] if np.std(rd[col])>1e-12 and np.std(rd['act'])>1e-12 else 0
    da = ((rd[col]>0)==(rd['act']>0)).mean()
    print(f'{col:8s} {r2:+.4f}  {corr:+.4f}  {da:.1%}')

# Backtest
print(f'\n{"="*50}')
print(f'BACKTEST: position tracking + turnover-aware costs')
print(f'{"="*50}')
rd['win']=(rd['act_open']>0).astype(int)
rd = rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()

meta_p=[]; mf=['avg','xgb','lgb','cb','stack']
for d in sorted(rd['dt_norm'].unique()):
    past=rd[rd['dt_norm']<d]; today=rd[rd['dt_norm']==d]
    if len(past)<500 or len(today)<5:
        meta_p.extend(today['avg'].clip(0,1).tolist()); continue
    s=StandardScaler(); clf=LogisticRegression(random_state=42,max_iter=500,C=1.0)
    clf.fit(s.fit_transform(past[mf].values),past['win'].values)
    meta_p.extend(clf.predict_proba(s.transform(today[mf].values))[:,1].tolist())
rd['mc']=meta_p

strategy_cols=['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']
dates=sorted(rd['dt_norm'].unique())
prev_models={}; prev_picks={}; bt=[]

def turnover(prev_s, curr_s):
    if prev_s is None: return 1.0
    if not prev_s or not curr_s: return 1.0
    ch=len(curr_s-prev_s)+len(prev_s-curr_s)
    return ch/max(len(curr_s|prev_s),1)

for d in dates:
    day=rd[rd['dt_norm']==d]
    if len(day)<5: continue
    picks={}
    for col in strategy_cols:
        picks[col]=day.sort_values(col,ascending=False).iloc[0]['sym']
    ranked=day.sort_values('stack',ascending=False)['sym'].tolist()
    t1_pick=ranked[0]; t3_picks=set(ranked[:3]); t5_picks=set(ranked[:5]); t10_picks=set(ranked[:10])
    t3mc=[s for s in ranked if day[day['sym']==s]['mc'].values[0]>=0.5][:3]
    if not t3mc: t3mc=[t1_pick]
    t3m_picks=set(t3mc)

    def pr(syms):
        r=day[day['sym'].isin(syms)]['act_open'].values
        return np.mean(r) if len(r)>0 else day.iloc[0]['act_open']

    t1_ret=pr({t1_pick}); t3_ret=pr(t3_picks); t5_ret=pr(t5_picks)
    t10_ret=pr(t10_picks); t3m_ret=pr(t3m_picks)
    mr={}; mt={}
    for col in strategy_cols:
        mr[col]=day[day['sym']==picks[col]]['act_open'].values[0] if len(day[day['sym']==picks[col]])>0 else 0.0
        pm=prev_models.get(col)
        mt[col]=0.0 if pm is not None and picks[col]==pm else 1.0
        prev_models[col]=picks[col]
    t1_to=1.0; t3_to=1.0; t5_to=1.0; t10_to=1.0; t3m_to=1.0
    if prev_picks:
        t1_to=0.0 if t1_pick==prev_picks.get('t1') else 1.0
        t3_to=turnover(prev_picks.get('t3'),t3_picks)
        t5_to=turnover(prev_picks.get('t5'),t5_picks)
        t10_to=turnover(prev_picks.get('t10'),t10_picks)
        t3m_to=turnover(prev_picks.get('t3m'),t3m_picks)
    prev_picks={'t1':t1_pick,'t3':t3_picks,'t5':t5_picks,'t10':t10_picks,'t3m':t3m_picks}
    t1c=cost_rt(TOTAL_POS)*t1_to*100; t3c=cost_rt(TOTAL_POS/3)*t3_to*100
    t5c=cost_rt(TOTAL_POS/5)*t5_to*100; t10c=cost_rt(TOTAL_POS/10)*t10_to*100
    t3mc=cost_rt(TOTAL_POS/len(t3m_picks))*t3m_to*100
    rec={'d':d,'t1_ret':t1_ret,'t3_ret':t3_ret,'t5_ret':t5_ret,'t10_ret':t10_ret,'t3m_ret':t3m_ret,
         't1_to':t1_to,'t3_to':t3_to,'t5_to':t5_to,'t10_to':t10_to,'t3m_to':t3m_to,
         't1_cost':t1c,'t3_cost':t3c,'t5_cost':t5c,'t10_cost':t10c,'t3m_cost':t3mc,
         't3_n':len(t3_picks),'t5_n':5,'t10_n':10,'t3m_n':len(t3m_picks)}
    for col in strategy_cols:
        rec[f'{col}_ret']=mr[col]; rec[f'{col}_to']=mt[col]
    bt.append(rec)

bt=pd.DataFrame(bt)
for s in ['t1','t3','t5','t10','t3m']:
    bt[f'{s}_net']=bt[f'{s}_ret']-bt[f'{s}_cost']
for col in strategy_cols:
    bt[f'{col}_net']=bt[f'{col}_ret']-cost_rt(TOTAL_POS)*bt[f'{col}_to']*100

# Portfolio strategies
STRATS=[('Top-1','t1_ret','t1_net','t1_to'),('Top-3','t3_ret','t3_net','t3_to'),
        ('Top-5','t5_ret','t5_net','t5_to'),('Top-10','t10_ret','t10_net','t10_to'),
        ('Top-3+Meta','t3m_ret','t3m_net','t3m_to')]
print(f'\nBacktest days: {len(bt)}')
print(f'\n{"Strategy":20s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
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

# Save
output = {'bt':bt,'rd':rd,'models':{},'features':[],'fi':{},'cps':cost_rt(TOTAL_POS),
          'cost_rt':cost_rt,'total_pos':TOTAL_POS,
          'n_symbols':rd['sym'].nunique(),'n_rows':len(rd),'time':0}
with open(OUT/'results_v5.pkl','wb') as f: pickle.dump(output,f)
bt.to_csv(OUT/'backtest_v5.csv',index=False)
rd.to_csv(OUT/'predictions_v5.csv',index=False)
print(f'\nSaved to {OUT}')
