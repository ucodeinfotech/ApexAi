"""Approach 3+4: Test Rs50 filter + regime-conditional on existing predictions"""
import pickle, pandas as pd, numpy as np, warnings
from sklearn.metrics import r2_score
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'return_prediction_report_v5'
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

def calc_metrics(s, n=252):
    if len(s)<5 or s.std()==0: return 0,0,0,0
    cagr=(1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh=s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr=(s>0).mean()
    dd=((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd

print('Loading predictions...')
with open(OUT/'results_raw.pkl','rb') as f:
    rd = pickle.load(f)
print(f'Total: {len(rd):,} predictions, {rd["sym"].nunique()} symbols')

# Identify penny symbols
# Get average close from feature_store
import duckdb
con = duckdb.connect(str(BASE/'warehouse'/'market_data.duckdb'), read_only=True)
avg_close = con.execute("SELECT symbol, AVG(close) as avg_c FROM feature_store WHERE timeframe='1day' AND datetime >= '2024-06-24' GROUP BY symbol").fetchdf()
con.close()
penny_syms = set(avg_close[avg_close['avg_c'] < 50]['symbol'].tolist())
print(f'Penny symbols (< Rs 50 avg): {len(penny_syms)}')

# --- Approach 3: Rs50 filter ---
print(f'\n{"="*55}')
print(f'APPROACH 3: Rs50 filter — remove penny stocks')
print(f'{"="*55}')
rd_filt = rd[~rd['sym'].isin(penny_syms)].copy()
print(f'Rows after filter: {len(rd_filt):,} ({len(rd)-len(rd_filt):,} removed)')

for col in ['xgb','ranker','lgb','cb','rf','et','avg','stack']:
    if col not in rd_filt.columns: continue
    r2 = r2_score(rd_filt['act'], rd_filt[col])
    corr = np.corrcoef(rd_filt['act'], rd_filt[col])[0,1] if np.std(rd_filt[col])>1e-12 and np.std(rd_filt['act'])>1e-12 else 0
    da = ((rd_filt[col]>0)==(rd_filt['act']>0)).mean()
    print(f'  {col:8s} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

# --- Approach 4: Regime-conditional ---
print(f'\n{"="*55}')
print(f'APPROACH 4: Regime-conditional performance')
print(f'{"="*55}')
if 'regime' in rd.columns:
    regimes = rd['regime'].unique()
    print(f'Regimes found: {regimes}')
    for reg in ['bull', 'sideways', 'bear']:
        sub = rd[rd['regime'] == reg]
        if len(sub) < 100: continue
        print(f'\n  --- {reg.upper()} regime ({len(sub):,} predictions) ---')
        for col in ['xgb','ranker','lgb','cb','rf','et','avg','stack']:
            if col not in sub.columns: continue
            r2 = r2_score(sub['act'], sub[col])
            corr = np.corrcoef(sub[col], sub['act'])[0,1] if np.std(sub[col])>1e-12 and np.std(sub['act'])>1e-12 else 0
            da = ((sub[col]>0)==(sub['act']>0)).mean()
            print(f'    {col:8s} R2={r2:+.4f} Corr={corr:+.4f} DirAcc={da:.1%}')

# Quick backtest per regime (Top-1 by stack)
rd = rd.sort_values('dt').reset_index(drop=True)
rd['dt_norm'] = pd.to_datetime(rd['dt']).dt.normalize() if hasattr(pd.to_datetime(rd['dt']),'dt') else rd['dt']

if 'regime' in rd.columns:
    print(f'\n{"="*55}')
    print(f'Regime-conditional Top-1 backtest')
    print(f'{"="*55}')
    for reg in ['bull', 'sideways', 'bear']:
        sub = rd[rd['regime'] == reg]
        if len(sub) < 100: continue
        dates = sorted(sub['dt_norm'].unique())
        bt = []; prev = None
        for d in dates:
            day = sub[sub['dt_norm'] == d]
            if len(day) < 5: continue
            pick = day.sort_values('stack', ascending=False).iloc[0]
            ret = pick['act_open'] if 'act_open' in pick else pick['act']
            to = 0.0 if prev == pick['sym'] else 1.0
            cost = cost_rt(TOTAL_POS) * to * 100
            bt.append({'ret': ret, 'net': ret - cost})
            prev = pick['sym']
        btdf = pd.DataFrame(bt)
        if len(btdf) < 5: continue
        gc, gs, gw, gdd = calc_metrics(btdf['ret'])
        nc, ns, nw, ndd = calc_metrics(btdf['net'])
        print(f'  {reg:10s}: Gross CAGR={gc:>+8.1f}% Net CAGR={nc:>+8.1f}% Sharpe={gs:.2f} WinRate={gw:.1f}%')
