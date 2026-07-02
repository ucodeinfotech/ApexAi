import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
class V5Loader(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == 'cost_rt': return lambda s: s * 0.001
        return super().find_class(mod, name)
BASE = r'C:\Users\pc\Downloads\stock hist data'

# v5
with open(BASE + r'\return_prediction_report_v5\results_v5.pkl','rb') as f:
    v5 = V5Loader(f).load()
print('=== v5 ===')
for c in v5['bt'].columns:
    if '_net' not in c: continue
    s = v5['bt'][c].dropna()
    if len(s) < 10: continue
    cagr = (1+s/100).prod()**(252/len(s))-1
    print(f'{c:20s} CAGR={cagr*100:+8.1f}%')

print(f'\nStack DirAcc (act): {((v5["rd"]["stack"]>0)==(v5["rd"]["act"]>0)).mean()*100:.1f}%')
print(f'Stack DirAcc (act_open): {((v5["rd"]["stack"]>0)==(v5["rd"]["act_open"]>0)).mean()*100:.1f}%')
print(f'Num features: {len(v5["features"])}')

# v6
with open(BASE + r'\return_prediction_report_v6\results_v6.pkl','rb') as f:
    v6 = pickle.load(f)
print('\n=== v6 ===')
for c in v6['bt'].columns:
    if '_net' not in c: continue
    s = v6['bt'][c].dropna()
    if len(s) < 10: continue
    cagr = (1+s/100).prod()**(252/len(s))-1
    print(f'{c:20s} CAGR={cagr*100:+8.1f}%')

print(f'\nStack DirAcc (act): {((v6["rd"]["stack"]>0)==(v6["rd"]["act"]>0)).mean()*100:.1f}%')
print(f'Top-1 DirAcc (act): {((v6["bt"]["t1_ret"]>0)==True).mean()*100:.1f}%')
print(f'Num features: {len(v6["features"])}')
print(f'v6 BT len: {len(v6["bt"])}')
print(f'v6 RD len: {len(v6["rd"])}')
print(f'v6 Symbols: {v6["n_symbols"]}')

# Compare RD dates
print(f'\nv5 date range: {v5["rd"]["dt"].min()} to {v5["rd"]["dt"].max()}')
print(f'v6 date range: {v6["rd"]["dt"].min()} to {v6["rd"]["dt"].max()}')
print(f'v5 unique dates: {v5["rd"]["dt"].nunique()}')
print(f'v6 unique dates: {v6["rd"]["dt"].nunique()}')
print(f'v5 unique syms: {v5["rd"]["sym"].nunique()}')
print(f'v6 unique syms: {v6["rd"]["sym"].nunique()}')
