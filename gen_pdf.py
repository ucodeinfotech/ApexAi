"""Generate PDF from saved CSV trade book"""
import pickle, math, time, numpy as np, pandas as pd, warnings
from pathlib import Path; from datetime import datetime
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'return_prediction_report_v5'
FIGS = OUT / 'trade_figs'; FIGS.mkdir(exist_ok=True)
PDF_PATH = OUT / 'v5_trade_book.pdf'

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

t0 = time.time()
print('Loading trade book CSV...')
tf = pd.read_csv(OUT / 'trade_book_v5.csv')
print(f'  {len(tf):,} trades loaded ({time.time()-t0:.1f}s)')
tf['Net'] = pd.to_numeric(tf['Net'], errors='coerce')
tf['Return'] = pd.to_numeric(tf['Return'], errors='coerce')

with open(OUT/'results_v5.pkl','rb') as f: res = pickle.load(f)
bt=res['bt']; rd=res['rd']; n_sym=res['n_symbols']; n_rows=res['n_rows']
dates=sorted(rd['dt_norm'].unique())
sc=['xgb','ranker','lgb','lgb_r','cb','rf','et','avg','stack']

# ── BUILD PDF ──
print('Building PDF...')
pdf=PdfPages(PDF_PATH)

def new_page(orient='portrait'):
    return plt.figure(figsize=(11,8.5) if orient=='landscape' else (8.5,11))

def header(ax, text, y=0.97):
    ax.text(0.5,y,text,fontsize=14,fontweight='bold',ha='center',transform=ax.transAxes)

# ═══ COVER ═══
fig=plt.figure(figsize=(8.5,11)); ax=fig.add_axes([0,0,1,1],facecolor='#1a1a2e')
ax.text(0.5,0.88,'v5 Trade Book',fontsize=28,fontweight='bold',color='white',ha='center')
ax.text(0.5,0.82,'Full Position-Level Backtest (Open-Close PnL)',fontsize=14,color='#a0a0c0',ha='center')
ax.text(0.5,0.76,f'{datetime.now().strftime("%Y-%m-%d %H:%M")}  |  {n_sym} symbols  |  {len(dates)} trading days',fontsize=11,color='#808090',ha='center')
lines=[]
for sn in ['Top-1','Top-3+Meta','Top-5','Top-10']:
    sub=tf[tf['Strategy']==sn]['Net'].dropna()
    if len(sub)<10: continue
    c,s,wr,dd,tr,av=calc_metrics(sub.values)
    lines.append(f'{sn:12s}  Net CAGR={c:>+7.1f}%  Sharpe={s:.2f}  Win={wr:.1f}%  DD={dd:.1f}%')
for c in ['lgb','cb','avg','stack','xgb','ranker']:
    sub=tf[tf['Strategy']==c]['Net'].dropna()
    if len(sub)<10: continue
    c2,s2,wr2,dd2,tr2,av2=calc_metrics(sub.values)
    lines.append(f'{c:12s}  Net CAGR={c2:>+7.1f}%  Sharpe={s2:.2f}  Win={wr2:.1f}%  DD={dd2:.1f}%')
ax.text(0.5,0.46,'\n'.join(lines[:16]),fontsize=8,color='#c0c0d0',ha='center',va='center',fontfamily='monospace')
ax.axis('off'); pdf.savefig(fig); plt.close()

# ═══ TRADE LOGS (landscape, 10 cols, FIRST 90 TRADES ONLY) ═══
print('  Trade logs...')
cols_s=['Date','Symbol','Entry','Exit','Return','Cost','Net','Win','stack','reg']
col_w=[0.12,0.10,0.10,0.10,0.09,0.07,0.09,0.05,0.10,0.08]
MAX_PAGES = 3  # limit per strategy
for sn in ['Top-1','Top-3+Meta','Top-5','Top-10']:
    sub=tf[tf['Strategy']==sn]
    if len(sub)<5: continue
    for ci in range(0, min(len(sub), MAX_PAGES*30), 30):
        chunk=sub.iloc[ci:ci+30]
        fig=new_page('landscape'); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
        header(ax,f'{sn} Trade Log ({ci+1}-{min(ci+30,len(sub))} of {len(sub)})',0.97)
        cells=[list(cols_s)]
        for _,r in chunk.iterrows():
            cells.append([str(r['Date'])[:10],str(r['Symbol']),
                f'{r["Entry"]:.1f}' if not np.isnan(r["Entry"]) else 'N/A',
                f'{r["Exit"]:.1f}' if not np.isnan(r["Exit"]) else 'N/A',
                f'{r["Return"]:+.2f}' if not np.isnan(r["Return"]) else 'N/A',
                f'{r["Cost"]:.3f}' if not np.isnan(r["Cost"]) else 'N/A',
                f'{r["Net"]:+.2f}' if not np.isnan(r["Net"]) else 'N/A',
                'W' if r['Win']==1 else('L' if r['Win']==0 else ''),
                f'{r["stack"]:.4f}' if not np.isnan(r["stack"]) else 'N/A',
                str(r.get('reg','?'))])
        tbl=ax.table(cellText=cells,loc='upper center',cellLoc='center',colWidths=col_w)
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
        style_table(tbl); pdf.savefig(fig); plt.close()
    # Add note about remaining
    if len(sub) > MAX_PAGES*30:
        fig=new_page('landscape'); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
        header(ax,f'{sn} Trade Log (cont.)',0.97)
        ax.text(0.5,0.5,f'Remaining {len(sub)-MAX_PAGES*30} trades omitted.\nFull trade log available in trade_book_v5.csv',
            fontsize=14,ha='center',va='center',fontfamily='monospace',color='#666')
        pdf.savefig(fig); plt.close()

for sn in ['lgb','cb','avg','stack','xgb','ranker','rf','et']:
    sub=tf[tf['Strategy']==sn]
    if len(sub)<5: continue
    for ci in range(0, min(len(sub), MAX_PAGES*30), 30):
        chunk=sub.iloc[ci:ci+30]
        fig=new_page('landscape'); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
        header(ax,f'{sn.upper()} Trade Log ({ci+1}-{min(ci+30,len(sub))} of {len(sub)})',0.97)
        cells=[list(cols_s)]
        for _,r in chunk.iterrows():
            cells.append([str(r['Date'])[:10],str(r['Symbol']),
                f'{r["Entry"]:.1f}' if not np.isnan(r["Entry"]) else 'N/A',
                f'{r["Exit"]:.1f}' if not np.isnan(r["Exit"]) else 'N/A',
                f'{r["Return"]:+.2f}' if not np.isnan(r["Return"]) else 'N/A',
                f'{r["Cost"]:.3f}' if not np.isnan(r["Cost"]) else 'N/A',
                f'{r["Net"]:+.2f}' if not np.isnan(r["Net"]) else 'N/A',
                'W' if r['Win']==1 else('L' if r['Win']==0 else ''),
                f'{r["stack"]:.4f}' if not np.isnan(r["stack"]) else 'N/A',
                str(r.get('reg','?'))])
        tbl=ax.table(cellText=cells,loc='upper center',cellLoc='center',colWidths=col_w)
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
        style_table(tbl); pdf.savefig(fig); plt.close()
    if len(sub) > MAX_PAGES*30:
        fig=new_page('landscape'); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
        header(ax,f'{sn.upper()} Trade Log (cont.)',0.97)
        ax.text(0.5,0.5,f'Remaining {len(sub)-MAX_PAGES*30} trades omitted.\nFull trade log available in trade_book_v5.csv',
            fontsize=14,ha='center',va='center',fontfamily='monospace',color='#666')
        pdf.savefig(fig); plt.close()

# ═══ SUMMARY TABLE ═══
print('  Summary...')
fig=new_page(); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
header(ax,'Complete Strategy Summary')
cols=['Strategy','Trades','Gross CAGR','Net CAGR','Sharpe','WinRate','MaxDD','AvgRet','AvgTO']
rows=[cols]
for sn in ['Top-1','Top-3','Top-5','Top-10','Top-3+Meta']+sc:
    sub=tf[tf['Strategy']==sn]
    g=sub['Return'].dropna(); n=sub['Net'].dropna()
    if len(g)<10: continue
    gs=calc_metrics(g.values); ns=calc_metrics(n.values); to=sub['TO'].mean()
    rows.append([sn[:10],str(len(g)),f'{gs[0]:+7.1f}%',f'{ns[0]:+7.1f}%',f'{ns[1]:.2f}',f'{ns[2]:.1f}%',f'{ns[3]:.1f}%',f'{ns[5]:+.3f}%',f'{to:.0%}'])
tbl=ax.table(cellText=rows,loc='upper center',cellLoc='center',colWidths=[0.10]*9)
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
style_table(tbl); pdf.savefig(fig); plt.close()

# ═══ EQUITY CURVES ═══
print('  Charts...')
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for ax,(sn,cols,colors) in zip(axes,[
    ('Top-1 / Top-3+Meta',['t1_net','t3m_net'],['#1f77b4','#d62728']),
    ('Top-5 / Top-10',['t5_net','t10_net'],['#2ca02c','#ff7f0e']),
    ('Models',['lgb','cb','avg'],['#9467bd','#8c564b','#7f7f7f'])]):
    for ci,col in enumerate(cols):
        if col in bt.columns:
            s=bt[col].dropna(); eq=(1+s/100).cumprod(); c2=calc_metrics(s.values)[0]
            ax.plot(eq.index,eq,lw=1.2,color=colors[ci],label=f'{col} ({c2:+.1f}% CAGR)')
    ax.set_ylabel('Equity'); ax.legend(fontsize=8,loc='upper left')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0)); ax.grid(alpha=0.3)
axes[-1].set_xlabel('Trading Day')
fig.suptitle('Equity Curves (Open-Close PnL)',fontsize=13,fontweight='bold')
plt.tight_layout(); fig.savefig(FIGS/'equity.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ DRAWDOWN + DISTRIBUTION ═══
fig,axes=plt.subplots(2,2,figsize=(10,8))
for ax,sn,col in [(axes[0,0],'Top-1','t1_net'),(axes[0,1],'Top-3+Meta','t3m_net')]:
    s=bt[col].dropna(); eq=(1+s/100).cumprod(); dd=eq/eq.cummax()-1
    ax.fill_between(dd.index,dd*100,0,color='#d62728',alpha=0.5,lw=0)
    ax.set_title(f'{sn} Drawdown',fontweight='bold'); ax.grid(alpha=0.3)
for ax,sn,col in [(axes[1,0],'Top-1','t1_net'),(axes[1,1],'Top-5','t5_net')]:
    s=bt[col].dropna()
    ax.hist(s,bins=80,color='#1f77b4',alpha=0.7,edgecolor='white',linewidth=0.5)
    ax.axvline(s.mean(),color='red',ls='--',label=f'Mean={s.mean():+.3f}%')
    ax.axvline(0,color='black',ls='-',lw=0.5)
    ax.set_title(f'{sn} Return Distribution',fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'dd.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ MONTHLY RETURNS ═══
bt_idx=bt.set_index('d'); bt_idx.index=pd.to_datetime(bt_idx.index)
for label,col in [('Top-1','t1_net'),('Top-3+Meta','t3m_net')]:
    monthly=bt_idx[col].groupby([bt_idx.index.year,bt_idx.index.month]).sum().unstack(0)
    fig,ax=plt.subplots(figsize=(12,4))
    if not monthly.empty:
        vm=max(abs(monthly.min().min()),abs(monthly.max().max()))
        im=ax.imshow(monthly.values,cmap='RdYlGn',aspect='auto',vmin=-vm,vmax=vm)
        ax.set_xticks(range(len(monthly.columns))); ax.set_xticklabels(monthly.columns)
        ax.set_yticks(range(12)); ax.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        plt.colorbar(im,ax=ax,label='%')
        for i in range(monthly.shape[0]):
            for j in range(monthly.shape[1]):
                v=monthly.values[i,j]
                if not np.isnan(v): ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=6,color='white' if abs(v)>vm*0.7 else 'black')
    ax.set_title(f'{label} Monthly Returns',fontweight='bold')
    plt.tight_layout(); fig.savefig(FIGS/f'monthly_{label}.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ PER SYMBOL ═══
sym_st=tf[tf['Strategy']=='Top-1'].groupby('Symbol').agg(Trades=('Net','count'),WR=('Win',lambda x:np.mean([v for v in x if v!=''])*100),AvgRet=('Net','mean'))
sym_st=sym_st.sort_values('AvgRet',ascending=False)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,col,title,c in [(axes[0],'AvgRet','Best Avg Return','#2ca02c'),(axes[1],'WR','Best Win Rate','#1f77b4')]:
    n=min(20,len(sym_st)); vals=sym_st[col].head(n).values
    ax.barh(range(n),vals,color=c)
    ax.set_yticks(range(n)); ax.set_yticklabels(sym_st.index[:n],fontsize=7)
    ax.set_title(f'Top-1: {title}',fontweight='bold'); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'sym.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ SLIPPAGE ═══
fig,ax=plt.subplots(figsize=(10,5))
for npos,lab,col in [(1,'Top-1','#1f77b4'),(5,'Top-5','#2ca02c'),(10,'Top-10','#d62728')]:
    svals=[0.0005,0.001,0.002,0.003,0.005,0.01]; cags=[]
    for sv in svals:
        ps=TOTAL_POS/npos; bs=max(BRK*ps,MIN_BRK)/ps*2; gb=bs+EXCH*2+SEBI*2; cr=bs+STT+EXCH*2+SEBI*2+STAMP+gb*GST+sv*2
        nc='t1_'; ne=bt['t1_ret']; tc=bt['t1_to']
        if npos>1: nc=f't{npos}_'; ne=bt[f'{nc}ret']; tc=bt[f'{nc}to']
        net_=ne-tc*cr*100; cag_=((1+net_/100).prod()**(252/len(net_))-1)*100
        cags.append(cag_)
    ax.plot([s*10000 for s in svals],cags,marker='o',lw=2,label=lab,color=col)
    for x,y in zip([s*10000 for s in svals],cags): ax.text(x,y,f'{y:.0f}%',fontsize=7,ha='center',va='bottom')
ax.axhline(0,color='black',ls='--',lw=0.5); ax.set_xlabel('Slippage (bp)'); ax.set_ylabel('Net CAGR %')
ax.set_title('Slippage Sensitivity',fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'slip.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ MULTI-DAY ═══
fig,ax=plt.subplots(figsize=(10,5))
hd=[1,2,3,5,10,20]
for lab,np_,col in [('Top-5',5,'#2ca02c'),('Top-3+Meta','3m','#ff7f0e'),('Top-10',10,'#d62728')]:
    cags=[]
    for h in hd:
        hbt=[]; cur=None
        for di,d in enumerate(dates):
            day=rd[rd['dt_norm']==d]
            if len(day)<5: continue
            rk=day.sort_values('stack',ascending=False)['sym'].tolist()
            if h==1 or di%h==0:
                if lab=='Top-3+Meta':
                    new=set([s for s in rk if day[day['sym']==s]['mc'].values[0]>=0.5][:3])
                    if not new: new={rk[0]}
                else: new=set(rk[:np_])
            else: new=cur if cur is not None else set(rk[:np_])
            to=0.0 if cur is not None and new==cur else 1.0
            r=day[day['sym'].isin(new)]['act_open'].mean() if new else 0
            c=cost_rt(TOTAL_POS/len(new))*to*100
            hbt.append(r-c); cur=new
        hbt=np.array(hbt)
        if len(hbt)>5: cags.append(((1+hbt/100).prod()**(252/len(hbt))-1)*100)
        else: cags.append(0)
    ax.plot(hd[:len(cags)],cags,marker='o',lw=2,label=lab,color=col)
    for x,y in zip(hd[:len(cags)],cags): ax.text(x,y,f'{y:.0f}%',fontsize=7,ha='center',va='bottom')
ax.axhline(0,color='black',ls='--',lw=0.5); ax.set_xlabel('Hold Days'); ax.set_ylabel('Net CAGR %')
ax.set_title('Multi-Day Holding',fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'hold.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ MODEL COMPARISON ═══
mr=[]
for c in sc:
    sub=tf[tf['Strategy']==c]['Net'].dropna()
    if len(sub)<10: continue
    c2,s2,wr2,dd2,tr2,av2=calc_metrics(sub.values); mr.append({'M':c,'CAGR':c2,'Sharpe':s2,'Win':wr2,'DD':dd2})
mrd=pd.DataFrame(mr).set_index('M')
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].bar(range(len(mrd)),mrd['CAGR'],color=['#1f77b4' if v>=0 else '#d62728' for v in mrd['CAGR']])
axes[0].set_xticks(range(len(mrd))); axes[0].set_xticklabels(mrd.index,fontsize=8,rotation=45)
axes[0].set_title('Model Net CAGR',fontweight='bold'); axes[0].grid(alpha=0.3)
axes[1].bar(range(len(mrd)),mrd['Sharpe'],color='#2ca02c')
axes[1].set_xticks(range(len(mrd))); axes[1].set_xticklabels(mrd.index,fontsize=8,rotation=45)
axes[1].set_title('Model Sharpe',fontweight='bold'); axes[1].grid(alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'model.png',bbox_inches='tight',dpi=120); pdf.savefig(fig); plt.close()

# ═══ WHITE'S RC ═══
print('  White RC...')
all_sc=[f'{c}_net' for c in sc]+[f'{s}_net' for s in ['t1','t3','t5','t10','t3m']]
avail=[c for c in all_sc if c in bt.columns]
all_ret=bt[avail].values; T,M=all_ret.shape
mr_=all_ret.mean(axis=0); sr=all_ret.std(axis=0); sr[sr<1e-12]=1e-12
ts=np.sqrt(T)*mr_/sr; Vo=ts.max(); bi=np.argmax(ts)
block_size=21; n_blocks=int(np.ceil(T/block_size)); bmax=np.zeros(2000)
for b in range(2000):
    br=np.zeros((T,M))
    for bi2 in range(n_blocks):
        st=np.random.randint(0,max(1,T-block_size)); en=min(st+block_size,T); blen=en-st
        if bi2*block_size+blen<=T: br[bi2*block_size:bi2*block_size+blen]=all_ret[st:en]
    bm=br.mean(axis=0); bt_=np.sqrt(T)*bm/sr; bmax[b]=bt_.max()
pw=(bmax>=Vo).mean()
fig=new_page(); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
header(ax,'Institutional Risk Metrics')
txt=[f'Strategies: {M} ({len(sc)} model + 5 portfolio)',f'Best: {avail[bi]} (t={Vo:.2f})',f'White RC p={pw:.4f}']
txt+=['','DSR (5 portfolio):']
for k in ['t1_net','t3m_net','t5_net','t10_net']:
    if k in bt.columns:
        s_=bt[k]; txt.append(f'  {k:10s} Sharpe={s_.mean()/s_.std()*np.sqrt(252):.2f}')
y=0.9
for t in txt: ax.text(0.08,y,t,fontsize=10,fontfamily='monospace'); y-=0.03
axi=fig.add_axes([0.55,0.1,0.35,0.3])
axi.hist(bmax,bins=50,color='#1f77b4',alpha=0.7,edgecolor='white')
axi.axvline(Vo,color='red',ls='--',lw=2,label=f'Obs t={Vo:.1f}')
axi.set_xlabel('Max t'); axi.legend(fontsize=7); axi.set_title('White RC Bootstrap (B=2000)',fontsize=9)
pdf.savefig(fig); plt.close()

# ═══ COST ANALYSIS ═══
fig=new_page(); ax=fig.add_axes([0.02,0.02,0.96,0.96]); ax.axis('off')
header(ax,'Cost Analysis')
txt=['Round-trip cost:']
for n in [1,3,5,10]: txt.append(f'  {n} pos (Rs{TOTAL_POS//n:,}): {cost_rt(TOTAL_POS/n)*100:.3f}%')
txt+=['','Avg turnover:']
for sn in ['Top-1','Top-3+Meta','Top-5','Top-10']: txt.append(f'  {sn}: {tf[tf["Strategy"]==sn]["TO"].mean():.1%}')
txt+=['','Cost components (single pos):']
cr=cost_rt(TOTAL_POS)
br=max(BRK*TOTAL_POS,MIN_BRK)/TOTAL_POS*2; comp=[('Brokerage',br),('STT',STT),('Exch x2',EXCH*2),('SEBI x2',SEBI*2),('Stamp',STAMP),('GST',(br+EXCH*2+SEBI*2)*GST),('Slippage',SLIP*2)]
for k,v in comp: txt.append(f'  {k:10s}: {v*100:.3f}% ({v/cr*100:.1f}%)')
y=0.92
for t in txt: ax.text(0.08,y,t,fontsize=10,fontfamily='monospace'); y-=0.022
pdf.savefig(fig); plt.close()

pdf.close()
print(f'PDF saved: {PDF_PATH}')
print(f'Total time: {time.time()-t0:.1f}s')
