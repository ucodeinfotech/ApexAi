import pickle, numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'C:\Users\pc\Downloads\stock hist data')

# v5
v5 = pickle.load(open(BASE/'return_prediction_report_v5'/'results_v5.pkl','rb'))
print('=== v5 (original) ===')
print(f'BT rows: {len(v5["bt"])}')
a5=v5['bt']['stack_ret'].dropna(); b5=v5['bt']['stack_net'].dropna()
cagr5=(1+a5/100).prod()**(252/len(a5))-1; net5=(1+b5/100).prod()**(252/len(b5))-1
print(f'Stack Gross CAGR: {cagr5*100:+.1f}%')
print(f'Stack Net CAGR: {net5*100:+.1f}%')
print(f'Stack DirAcc: {((v5["rd"]["stack"]>0)==(v5["rd"]["act_open"]>0)).mean():.1%}')

# v6
try:
    v6 = pickle.load(open(BASE/'return_prediction_report_v6'/'results_v6.pkl','rb'))
    print('\n=== v6 (fixed) ===')
    print(f'BT rows: {len(v6["bt"])}')
    t1=v6['bt']['t1_ret'].dropna()
    cagr_t1=(1+t1/100).prod()**(252/len(t1))-1
    print(f'Top-1 Gross CAGR: {cagr_t1*100:+.1f}%')
    n1=v6['bt']['t1_net'].dropna()
    cagr_n1=(1+n1/100).prod()**(252/len(n1))-1
    print(f'Top-1 Net CAGR: {cagr_n1*100:+.1f}%')
    # Compare target distributions
    r5=v5['rd']; r6=v6['rd']
    print(f'\nv5 target range: [{r5["act"].min():+.2f}%, {r5["act"].max():+.2f}%]')
    print(f'v6 target range: [{r6["act"].min():+.2f}%, {r6["act"].max():+.2f}%]')
    print(f'v5 act_open range: [{r5["act_open"].min():+.2f}%, {r5["act_open"].max():+.2f}%]')
    print(f'v6 act_open range: [{r6["act_open"].min():+.2f}%, {r6["act_open"].max():+.2f}%]')
    print(f'v5 act_open mean: {r5["act_open"].mean():+.4f}%')
    print(f'v6 act_open mean: {r6["act_open"].mean():+.4f}%')
    # Compare predictions
    print(f'\nv5 stack pred range: [{r5["stack"].min():+.4f}, {r5["stack"].max():+.4f}]')
    print(f'v6 stack pred range: [{r6["stack"].min():+.4f}, {r6["stack"].max():+.4f}]')
    print(f'\nv5 predictions: {len(r5)}, v6 predictions: {len(r6)}')
    print(f'v5 symbols: {r5["sym"].nunique()}, v6 symbols: {r6["sym"].nunique()}')
    # v5 stats
    print(f'\n--- v5 per-model CAGR ---')
    for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']:
        nc=f'{col}_net'
        if nc not in v5['bt'].columns: continue
        n=v5['bt'][nc].dropna()
        if len(n)<10: continue
        cagr=(1+n/100).prod()**(252/len(n))-1
        to=v5['bt'][f'{col}_to'].mean()*100 if f'{col}_to' in v5['bt'].columns else 0
        print(f'{col:10s} CAGR={cagr*100:+8.1f}% TO={to:.1f}%')
    print(f'\n--- v6 per-model CAGR ---')
    for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']:
        nc=f'{col}_net'
        if nc not in v6['bt'].columns: continue
        n=v6['bt'][nc].dropna()
        if len(n)<10: continue
        cagr=(1+n/100).prod()**(252/len(n))-1
        to=v6['bt'][f'{col}_to'].mean()*100 if f'{col}_to' in v6['bt'].columns else 0
        print(f'{col:10s} CAGR={cagr*100:+8.1f}% TO={to:.1f}%')
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
