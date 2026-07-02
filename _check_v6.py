import pickle, pandas as pd, numpy as np
BASE = r'C:\Users\pc\Downloads\stock hist data'

# Load v6 results
with open(BASE + r'\return_prediction_report_v6\results_v6.pkl', 'rb') as f:
    v6 = pickle.load(f)

print(f'v6 results keys: {list(v6.keys())}')
print(f'v6 predictions: {len(v6["rd"]):,}')
print(f'v6 features: {v6["features"]}')
print(f'v6 n_symbols: {v6["n_symbols"]}')
print(f'v6 time: {v6["time"]/60:.1f} min')

# Backtest metrics
bt = v6['bt']
rd = v6['rd']

STRATS = [('Top-1','t1_ret','t1_net','t1_to'),('Top-3','t3_ret','t3_net','t3_to'),
          ('Top-5','t5_ret','t5_net','t5_to'),('Top-10','t10_ret','t10_net','t10_to'),
          ('Top-3M','t3m_ret','t3m_net','t3m_to')]

print(f'\n{"Strategy":15s} {"Gross CAGR":>10s} {"Net CAGR":>10s} {"Sharpe":>7s} {"WinRate":>8s} {"MaxDD":>7s}')
print('-'*65)
for sn, rc, nc, tc in STRATS:
    g = bt[rc].dropna()
    n = bt[nc].dropna()
    if len(g) < 10: continue
    gc = (1+g/100).prod()**(252/len(g))-1
    nac = (1+n/100).prod()**(252/len(n))-1
    sh = n.mean()/n.std()*np.sqrt(252)
    wr = (n>0).mean()
    dd = ((1+n/100).cumprod()/(1+n/100).cumprod().cummax()-1).min()*100
    print(f'{sn:15s} {gc*100:>+9.1f}% {nac*100:>+9.1f}% {sh:>6.2f} {wr:>7.1%} {dd:>6.1f}%')

# Per model comparison
print(f'\n{"Model":12s} {"Net CAGR":>10s} {"Sharpe":>7s} {"WinRate":>8s} {"MaxDD":>7s}')
print('-'*50)
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']:
    nc = f'{col}_net'
    if nc not in bt.columns: continue
    n = bt[nc].dropna()
    if len(n) < 10: continue
    cagr = (1+n/100).prod()**(252/len(n))-1
    sh = n.mean()/n.std()*np.sqrt(252)
    wr = (n>0).mean()
    dd = ((1+n/100).cumprod()/(1+n/100).cumprod().cummax()-1).min()*100
    print(f'{col:12s} {cagr*100:>+9.1f}% {sh:>6.2f} {wr:>7.1%} {dd:>6.1f}%')

# Directional accuracy per model
print(f'\nDirectional Accuracy (Close-Close):')
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']:
    if col not in rd.columns: continue
    da_cc = ((rd[col]>0)==(rd['act']>0)).mean()
    da_oc = ((rd[col]>0)==(rd['act_open']>0)).mean()
    print(f'  {col:12s} CC-DirAcc={da_cc:.1%}  OC-DirAcc={da_oc:.1%}')

# Year range
rd['dt'] = pd.to_datetime(rd['dt'])
print(f'\nDate range: {rd["dt"].min()} to {rd["dt"].max()}')
print(f'Trading days: {rd["dt"].dt.normalize().nunique()}')
print(f'Symbols: {rd["sym"].nunique()}')
