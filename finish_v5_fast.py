"""Fast backtest from results_raw.pkl"""
import pickle, pandas as pd, numpy as np, warnings
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

print('Loading...')
with open(OUT/'results_raw.pkl','rb') as f:
    rd = pickle.load(f)
print(f'Loaded: {len(rd):,} predictions, {rd["sym"].nunique()} symbols')

cols = ['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']
avail = [c for c in cols if c in rd.columns]

print(f'\n{"="*50}')
print(f'PERFORMANCE - v5 (fixed features, GPU)')
print(f'{"="*50}')
print(f'{"Model":8s} {"R2":>8s} {"Corr":>8s} {"DirAcc":>8s}')
for col in avail:
    r2 = r2_score(rd['act'], rd[col])
    corr = np.corrcoef(rd['act'], rd[col])[0,1] if np.std(rd[col])>1e-12 and np.std(rd['act'])>1e-12 else 0
    da = ((rd[col]>0)==(rd['act']>0)).mean()
    print(f'{col:8s} {r2:+.4f}  {corr:+.4f}  {da:.1%}')

print(f'\n{"="*50}')
print(f'BACKTEST [OPEN-CLOSE PnL]')
print(f'{"="*50}')
rd = rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm'] = rd['dt'].dt.normalize()

# Meta classifier (fast: fit once per date)
rd['win'] = (rd['act_open'] > 0).astype(int)
print('Meta classifier...')
mf = ['avg','xgb','lgb','cb','stack']
rd['mc'] = rd['avg'].clip(0,1)  # default
dates = sorted(rd['dt_norm'].unique())
for i, d in enumerate(dates):
    past = rd[rd['dt_norm'] < d]
    today_mask = rd['dt_norm'] == d
    today_idx = rd.index[today_mask]
    if len(past) >= 500 and len(today_idx) >= 5:
        s = StandardScaler()
        clf = LogisticRegression(random_state=42, max_iter=500, C=1.0)
        clf.fit(s.fit_transform(past[mf].values), past['win'].values)
        rd.loc[today_idx, 'mc'] = clf.predict_proba(s.transform(rd.loc[today_idx, mf].values))[:, 1]

# Backtest
print('Backtest...')
strategy_cols = avail

# Pre-compute daily top picks
def daily_top(day):
    picks = {}
    for col in strategy_cols:
        picks[col] = day.sort_values(col, ascending=False).iloc[0]['sym']
    ranked = day.sort_values('stack', ascending=False)
    tops = ranked['sym'].tolist()
    mc_ok = ranked[ranked['mc'] >= 0.5]['sym'].tolist()
    return picks, tops, mc_ok

def pick_ret(day, syms):
    mask = day['sym'].isin(syms)
    return day.loc[mask, 'act_open'].mean() if mask.any() else day.iloc[0]['act_open']

def turnover(prev, curr):
    if prev is None or not prev or not curr: return 1.0
    ch = len(curr - prev) + len(prev - curr)
    return ch / max(len(curr | prev), 1)

bt = []
prev_models = {}
prev_picks = None

grouped = list(rd.groupby('dt_norm'))
for i, (d, day) in enumerate(grouped):
    if len(day) < 5: continue
    day = day.reset_index(drop=True)
    picks, ranked, mc_ok = daily_top(day)
    t1 = ranked[0]; t3 = set(ranked[:3]); t5 = set(ranked[:5]); t10 = set(ranked[:10])
    t3m_cand = [s for s in ranked if s in mc_ok][:3]
    t3m = set(t3m_cand) if t3m_cand else {t1}

    t1r = pick_ret(day, {t1}); t3r = pick_ret(day, t3)
    t5r = pick_ret(day, t5); t10r = pick_ret(day, t10); t3mr = pick_ret(day, t3m)

    mrets = {}; mtos = {}
    for col in strategy_cols:
        mrets[col] = day.loc[day['sym'] == picks[col], 'act_open'].values
        mrets[col] = mrets[col][0] if len(mrets[col]) > 0 else 0.0
        mtos[col] = 0.0 if picks[col] == prev_models.get(col) else 1.0
        prev_models[col] = picks[col]

    t1to = 0.0 if prev_picks and t1 == prev_picks['t1'] else 1.0
    t3to = turnover(prev_picks.get('t3') if prev_picks else None, t3)
    t5to = turnover(prev_picks.get('t5') if prev_picks else None, t5)
    t10to = turnover(prev_picks.get('t10') if prev_picks else None, t10)
    t3mto = turnover(prev_picks.get('t3m') if prev_picks else None, t3m)
    prev_picks = {'t1': t1, 't3': t3, 't5': t5, 't10': t10, 't3m': t3m}

    rec = {'d': d, 't1_ret': t1r, 't3_ret': t3r, 't5_ret': t5r, 't10_ret': t10r, 't3m_ret': t3mr,
           't1_to': t1to, 't3_to': t3to, 't5_to': t5to, 't10_to': t10to, 't3m_to': t3mto,
           't1_cost': cost_rt(TOTAL_POS) * t1to * 100,
           't3_cost': cost_rt(TOTAL_POS/3) * t3to * 100,
           't5_cost': cost_rt(TOTAL_POS/5) * t5to * 100,
           't10_cost': cost_rt(TOTAL_POS/10) * t10to * 100,
           't3m_cost': cost_rt(TOTAL_POS/len(t3m)) * t3mto * 100,
           't3_n': len(t3), 't5_n': 5, 't10_n': 10, 't3m_n': len(t3m)}
    for col in strategy_cols:
        rec[f'{col}_ret'] = mrets[col]; rec[f'{col}_to'] = mtos[col]
    bt.append(rec)

bt = pd.DataFrame(bt)
for s in ['t1','t3','t5','t10','t3m']:
    bt[f'{s}_net'] = bt[f'{s}_ret'] - bt[f'{s}_cost']
for col in strategy_cols:
    bt[f'{col}_net'] = bt[f'{col}_ret'] - cost_rt(TOTAL_POS) * bt[f'{col}_to'] * 100

STRATS = [('Top-1','t1_ret','t1_net','t1_to'),('Top-3','t3_ret','t3_net','t3_to'),
          ('Top-5','t5_ret','t5_net','t5_to'),('Top-10','t10_ret','t10_net','t10_to'),
          ('Top-3+Meta','t3m_ret','t3m_net','t3m_to')]
print(f'Backtest days: {len(bt)}')
print(f'\n{"Strategy":20s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
print('-'*84)
for sn,rc,nc,tc in STRATS:
    g = bt[rc].dropna(); n = bt[nc].dropna()
    if len(g) < 10: continue
    gc,gs,gw,gdd = calc_metrics(g)
    nac,nas,naw,nadd = calc_metrics(n)
    print(f'{sn:20s} {gc:>+11.1f}% {nac:>+11.1f}% {nas:>7.2f} {naw:>7.1f}% {nadd:>7.1f}% {bt[tc].mean():>7.1%}')

print(f'\n{"Model":10s} {"Gross CAGR":>12s} {"Net CAGR":>12s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"AvgTO":>8s}')
print('-'*74)
for col in strategy_cols:
    rc = f'{col}_ret'; nc = f'{col}_net'; tc = f'{col}_to'
    g = bt[rc].dropna(); n = bt[nc].dropna()
    if len(g) < 10: continue
    gc,gs,gw,gdd = calc_metrics(g)
    nac,nas,naw,nadd = calc_metrics(n)
    print(f'{col:10s} {gc:>+11.1f}% {nac:>+11.1f}% {nas:>7.2f} {naw:>7.1f}% {nadd:>7.1f}% {bt[tc].mean():>7.1%}')

output = {'bt':bt,'rd':rd,'models':{},'features':[],'fi':{},'cps':cost_rt(TOTAL_POS),
          'cost_rt':cost_rt,'total_pos':TOTAL_POS,
          'n_symbols':rd['sym'].nunique(),'n_rows':len(rd),'time':0}
with open(OUT/'results_v5.pkl','wb') as f: pickle.dump(output,f)
bt.to_csv(OUT/'backtest_v5.csv',index=False)
rd.to_csv(OUT/'predictions_v5.csv',index=False)
print(f'\nSaved to {OUT}')
