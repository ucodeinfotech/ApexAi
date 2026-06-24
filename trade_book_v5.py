"""Full trade book: CSV + PDF with proper formatting"""
import pickle, math, time, numpy as np, pandas as pd, duckdb, warnings
from pathlib import Path; from datetime import datetime, date as dt_date
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from tqdm import tqdm
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'return_prediction_report_v5'
FIGS = OUT / 'trade_figs'; FIGS.mkdir(exist_ok=True)

STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size<=0: return 1.0
    b=max(BRK*pos_size,MIN_BRK)/pos_size*2; g=b+EXCH*2+SEBI*2
    return b+STT+EXCH*2+SEBI*2+STAMP+g*GST+SLIP*2

def calc_metrics(s,n=252):
    if len(s)<5 or s.std()==0: return (0,0,0,0,0,0)
    cagr=(1+s/100).prod()**(n/len(s))-1; sh=s.mean()/s.std()*np.sqrt(n)
    wr=(s>0).mean(); eq=np.cumprod(1+s/100); dd=(eq/np.maximum.accumulate(eq)-1).min()*100
    tr=((1+s/100).prod()-1)*100; return(cagr*100,sh,wr*100,dd,tr,s.mean())

def style_table(tbl, fontsize=7):
    for (i,j),cell in tbl.get_celld().items():
        if i==0: cell.set_facecolor('#1a1a2e'); cell.set_text_props(color='white',fontweight='bold',fontsize=fontsize)
        else: cell.set_facecolor('#f0f0f8' if i%2==0 else 'white'); cell.set_text_props(fontsize=fontsize-1)
        cell.PAD = 0.02

plt.rcParams.update({'font.size':9,'axes.titlesize':12,'axes.labelsize':10,'figure.dpi':120,'savefig.dpi':120})

t0 = time.time(); print('Loading data...')
with open(OUT/'results_v5.pkl','rb') as f: res = pickle.load(f)
bt=res['bt']; rd=res['rd']; n_sym=res['n_symbols']; n_rows=res['n_rows']; print(f'  {time.time()-t0:.1f}s')

rd['dt']=pd.to_datetime(rd['dt'])
rd=rd.sort_values(['dt','sym']).reset_index(drop=True)

print('Loading OHLC...')
t0 = time.time()
con=duckdb.connect(str(BASE/'warehouse'/'market_data.duckdb'))
ohlc=con.execute("SELECT symbol,datetime::DATE as date,open,close FROM feature_store WHERE timeframe='1day' ORDER BY symbol,date").fetchdf()
con.close()
# Pre-build fast price lookup: {(symbol, date_str) -> (open, close)}
price_lookup = {}
for _,r in tqdm(ohlc.iterrows(), total=len(ohlc), desc='Building price lookup'):
    d = r['date']; d = d.date() if hasattr(d, 'date') else d
    price_lookup[(r['symbol'], d)] = (float(r['open']), float(r['close']))
print(f'  {time.time()-t0:.1f}s, {len(price_lookup):,} entries')

def lookup_fast(sym, d):
    sd=pd.to_datetime(d)
    for offset in range(10):
        nxt = sd + pd.Timedelta(days=1+offset)
        key = (sym, nxt.date())
        if key in price_lookup:
            return price_lookup[key]
    return (np.nan, np.nan)

# ── Reconstruct trade book ──
print('Reconstructing trades...')
t0 = time.time()
sc=['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']
dates=sorted(rd['dt_norm'].unique())
trades=[]; pm={}; pp={}
for d in tqdm(dates, desc='Trades'):
    day=rd[rd['dt_norm']==d]
    if len(day)<5: continue
    ranked=day.sort_values('stack',ascending=False); syms=ranked['sym'].tolist()
    mv=dict(zip(ranked['sym'],ranked['mc']))
    t1=syms[0]; t3=syms[:3]; t5=syms[:5]; t10=syms[:10]
    t3m=[s for s in syms if mv.get(s,0)>=0.5][:3]
    if not t3m: t3m=[t1]
    mp={c:day.sort_values(c,ascending=False).iloc[0]['sym'] for c in sc}
    for sn,sl in [('Top-1',[t1]),('Top-3',t3),('Top-5',t5),('Top-10',t10),('Top-3+Meta',t3m)]+[(c,[mp[c]]) for c in sc]:
        prev=pp.get(sn); n_pos=len(sl); cr=cost_rt(TOTAL_POS/n_pos)
        for sym in sl:
            ep,xp=lookup_fast(sym,d)
            ret=(xp/ep-1)*100 if ep and xp and not(np.isnan(ep)or np.isnan(xp)) else np.nan
            row=day[day['sym']==sym]
            if len(row)>0:
                r0=row.iloc[0]
                sig={c:float(r0[c]) for c in sc if c in r0}
                sig['stack']=float(r0['stack']); sig['avg']=float(r0['avg']); sig['mc']=float(r0['mc'])
                sig['act']=float(r0['act']); sig['act_o']=float(r0['act_open']); sig['reg']=str(r0.get('regime','?'))
            else:
                sig={c:np.nan for c in sc+['avg','stack','mc','act','act_o']}; sig['reg']='?'
            to=0.0 if sn in sc and pm.get(sn)==sym else (1.0 if sn in sc else 1.0)
            if sn not in sc:
                pset=pp.get(sn)
                if pset is not None:
                    ch=len(set(sl)-pset)+len(pset-set(sl)); to=ch/max(len(set(sl)|pset),1)
            cost=cr*to*100; net=(ret-cost) if not np.isnan(ret) else np.nan
            trades.append({'Date':str(d)[:10],'Strategy':sn,'Symbol':sym,'Entry':ep,'Exit':xp,
                'Return':ret,'Cost':cost,'Net':net,'Win':1 if net and net>0 else(0 if net and net<0 else''),
                'TO':to,'N':n_pos,'stack':sig.get('stack'),'avg':sig.get('avg'),'mc':sig.get('mc'),
                'act':sig.get('act'),'act_o':sig.get('act_o'),'reg':sig.get('reg')})
        pp[sn]=set(sl)
        if sn in sc: pm[sn]=mp.get(sn)
tf=pd.DataFrame(trades)
tf.to_csv(OUT/'trade_book_v5.csv',index=False); print(f'Trades: {len(tf):,}  ({time.time()-t0:.1f}s)')
print(f'CSV: {OUT/"trade_book_v5.csv"}')
print(f'Done in {time.time()-t0:.1f}s')
