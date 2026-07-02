"""Analyze all directional strategies from CSV"""
import pandas as pd, numpy as np, math
df = pd.read_csv(r'C:\Users\pc\Downloads\stock hist data\directional_backtest.csv')

def metrics(s, n=252):
    s = s.dropna()
    if len(s) < 5 or s.std() == 0:
        return {'n':len(s),'CAGR':0,'Sharpe':0,'WinRate':0,'MaxDD':0,'Mean':0,'Std':0}
    cagr = ((1+s/100).prod()**(n/len(s))-1)*100
    sh = s.mean()/s.std()*math.sqrt(n)
    wr = (s>0).mean()*100
    dd = ((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return {'n':len(s),'CAGR':cagr,'Sharpe':sh,'WinRate':wr,'MaxDD':dd,'Mean':s.mean(),'Std':s.std()}
    
ret_cols = [c for c in df.columns if c.endswith('_ret')]
print(f'{"Strategy":25s} {"Days":>5s} {"CAGR":>10s} {"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} {"Mean Ret":>10s}')
print('-'*76)
for c in ret_cols:
    name = c.replace('_ret','')
    m = metrics(df[c])
    print(f'{name:25s} {m["n"]:5d} {m["CAGR"]:>+9.1f}% {m["Sharpe"]:>7.2f} {m["WinRate"]:>7.1f}% {m["MaxDD"]:>7.1f}% {m["Mean"]:>+9.3f}%')
