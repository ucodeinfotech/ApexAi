"""
Full rupee backtest (FIXED): 1 lot NIFTY, 1L capital, same-strike tracking
Generates comprehensive PDF with trade book
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, calendar
from datetime import timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
from option_fixed_engine import load_fixed_data, get_entry_strike_idx, exit_tp_maxd
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
LOT=50
plt.rcParams.update({"font.size":8,"axes.titlesize":11,"axes.labelsize":9,"figure.dpi":120,"savefig.dpi":150,"font.family":"sans-serif"})

def atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

print("Building trades...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
a5=atr(m5); me=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()
trades=[]; b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; rr=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
    if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
    lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
    idx=np.searchsorted(me,ts.asm8.view("int64"),side="right")
    if idx>=len(m5["close"]): continue
    bi=idx
    while bi<len(m5["close"]) and m5["close"].iloc[bi]<=lv: bi+=1
    if bi>=len(m5["close"])-1: continue
    ri=bi+1
    while ri<len(m5["close"]):
        if m5["low"].iloc[ri]<lv and m5["close"].iloc[ri]>lv and pd.Series(m5["datetime"]).dt.time.iloc[ri]<CUT: break
        ri+=1
    if ri>=len(m5["close"]): continue
    ed=m5["datetime"].iloc[ri]; ep_=m5["close"].iloc[ri]
    if ep_-m5["low"].iloc[ri]<=0: continue
    he=ep_
    for j in range(ri,len(m5["close"])):
        ca=a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"]=trades["exit_dt"].dt.tz_localize(None)
exp_w_l=[]
for _,r in trades.iterrows():
    d_=r["ed_naive"].date()
    da=(3-d_.weekday())%7; da=da if da>0 else 7
    exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
trades["exp_w"]=exp_w_l
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Trades: {len(trades_pre)}")

print("Loading fixed option data...")
atm_ts_arr, atm_cl_arr, atm_st_arr, strike_cache, lookup = load_fixed_data(trades_pre)

# Run fixed backtest: TP30_Max7d + Prem<130
TP_VAL=30; MAX_D=7; PREM_MAX=130
pnls=[]; trade_book=[]
for i in range(len(trades_pre)):
    ed=trades_pre.iloc[i]["ed_naive"]
    info=get_entry_strike_idx(ed, lookup, atm_ts_arr, strike_cache)
    if info is None: continue
    if info["ep"]>PREM_MAX: continue
    xp,xts=exit_tp_maxd(info["stk_data"],info["s_idx"],TP_VAL,MAX_D)
    if xp is None: continue
    pnl=round(xp-info["ep"],1)
    pnls.append(pnl)
    trade_book.append({
        "entry":ed,"exit":pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts,np.datetime64) else xts,
        "strike":info["strike"],"entry_prem":round(info["ep"],1),"exit_prem":round(xp,1),"pnl":pnl,
        "yr":ed.year,"mo":ed.month,"weekday":ed.weekday()
    })
print(f"Trades: {len(pnls)} | PnL: {sum(pnls):+,.0f} pts (Rs {sum(pnls)*LOT:+,.0f})")
print(f"WR: {sum(1 for p in pnls if p>0)/len(pnls)*100:.1f}%")

# === GENERATE PDF ===
PDF_NAME="Option_1L_Backtest_Report_FIXED.pdf"
with PdfPages(PDF_NAME) as pdf:
    # PAGE 1: Summary
    fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
    p=pd.Series(pnls); n=len(p); net=p.sum(); wr=(p>0).mean()
    aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); mdd=(mx-cum).max()
    sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    calmar=net/mdd if mdd>0 else 999
    capital=100000; lot_cost=sum(pnls)
    trade_pnl_rs=[p*LOT for p in pnls]
    net_rs=sum(trade_pnl_rs); final_rs=capital+net_rs
    y = lambda ys: 0.92 - ys*0.035
    def t(txt,ys,fs=11,wt="bold"):
        ax.text(0.08,y(ys),txt,fontsize=fs,fontweight=wt,transform=ax.transAxes,verticalalignment="top")
    t("NIFTY50 OPTION STRATEGY - FIXED (Same-Strike Tracking)",0,fs=14)
    t("Strategy: Bullish Engulfing + TP30 Max7d + Premium<130",2,fs=12)
    t("Exit: Take profit +30 pts on option, or time stop at 7 days",3)
    t(f"Trades: {n}",5); t(f"Net PnL: {net:+,.0f} pts (Rs {net_rs:+,.0f})",6)
    t(f"Win Rate: {wr:.1%}",7); t(f"Avg Trade: {p.mean():+.1f} pts",8)
    t(f"Avg Win: {aw:+.1f} | Avg Loss: {al:+.1f}",9)
    t(f"W/L Ratio: {aw/abs(al) if al!=0 else 999:.1f}x",10)
    t(f"Max DD: {mdd:,.0f} pts (Rs {mdd*LOT:,.0f})",11)
    t(f"Calmar: {calmar:.1f}x",12); t(f"Sharpe: {sharpe:.2f}",13)
    t(f"Max Win: {max(pnls):+.1f} | Max Loss: {min(pnls):+.1f}",14)
    t(f"Capital: Rs {capital:,}",15)
    t(f"Final Capital: Rs {final_rs:,} ({((final_rs/capital)-1)*100:+.1f}%)",16)
    t(f"LOT_SIZE: {LOT} (NIFTY)",17)
    t(f"Data: options_data_dedup (deduped, 9.99M rows)",18)
    t(f"Entry Filter: Premium <= {PREM_MAX} pts",19)
    pdf.savefig(fig); plt.close()

    # PAGE 2: Equity Curve
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.27,11.69),gridspec_kw={"height_ratios":[2,1]})
    cum_rs=np.cumsum(trade_pnl_rs)+capital
    ax1.plot(cum_rs,color="green",lw=1)
    ax1.axhline(y=capital,color="gray",ls="--",alpha=0.5)
    ax1.fill_between(range(len(cum_rs)),capital,cum_rs,where=(cum_rs>=capital),color="green",alpha=0.1)
    ax1.fill_between(range(len(cum_rs)),capital,cum_rs,where=(cum_rs<capital),color="red",alpha=0.1)
    ax1.set_title("Equity Curve (Rs)"); ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}"))
    ax1.grid(alpha=0.3)
    dd_rs=mx*np.nan_to_num(np.array([LOT]*len(cum))); dd_curve=(mx-cum)*LOT
    ax2.fill_between(range(len(dd_curve)),0,dd_curve,color="red",alpha=0.3)
    ax2.set_title("Drawdown (Rs)"); ax2.set_ylabel("Drawdown (Rs)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax2.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # PAGE 3: Yearly breakdown
    fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
    tb=pd.DataFrame(trade_book)
    yr=tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean())).sort_index()
    t("Yearly Breakdown",1,fs=14,wt="bold")
    col_w=[0.08,0.08,0.12,0.08,0.08]
    hds=["Year","Trades","Net (pts)","Net (Rs)","WR"]
    xs=[sum(col_w[:j])+0.05 for j in range(len(col_w))]
    for j,h in enumerate(hds):
        ax.text(xs[j],0.83,h,fontsize=10,fontweight="bold",transform=ax.transAxes)
    for k,(y_,r) in enumerate(yr.iterrows()):
        yy=0.80-k*0.035; vals=[str(int(y_)),str(int(r["trades"])),f"{r['net']:+,.0f}",f"Rs{r['net']*LOT:+,.0f}",f"{r['wr']:.0%}"]
        for j,v in enumerate(vals):
            ax.text(xs[j],yy,v,fontsize=9,transform=ax.transAxes)
    t(f"Total: {len(tb)} trades | {tb['pnl'].sum():+,.0f} pts | Rs{tb['pnl'].sum()*LOT:+,.0f} | WR={(tb['pnl']>0).mean():.0%}", 0.80-(len(yr)+1)*0.035+0.035, fs=10, wt="bold")
    
    # Monthly breakdown
    mo=tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    t("Monthly Breakdown",0.80-(len(yr)+4)*0.035,fs=14,wt="bold")
    bm=0.80-(len(yr)+6)*0.035
    for j,h in enumerate(["Month","Trades","Net (pts)","Net (Rs)","WR"]):
        ax.text(xs[j],bm,h,fontsize=10,fontweight="bold",transform=ax.transAxes)
    for k,(m_,r) in enumerate(mo.iterrows()):
        yy=bm-(k+1)*0.035
        ax.text(xs[0],yy,MONTHS[int(m_)-1],fontsize=9,transform=ax.transAxes)
        ax.text(xs[1],yy,str(int(r["trades"])),fontsize=9,transform=ax.transAxes)
        ax.text(xs[2],yy,f"{r['net']:+,.0f}",fontsize=9,transform=ax.transAxes)
        ax.text(xs[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=9,transform=ax.transAxes)
        ax.text(xs[4],yy,f"{r['wr']:.0%}",fontsize=9,transform=ax.transAxes)
    pdf.savefig(fig); plt.close()

    # PAGE 4: Trade book (up to 40 trades per page)
    trades_per_page=40
    for page_start in range(0,len(trade_book),trades_per_page):
        fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
        chunk=trade_book[page_start:page_start+trades_per_page]
        t(f"Trade Book ({page_start+1}-{page_start+len(chunk)} of {len(trade_book)})",1,fs=12,wt="bold")
        t(f"TP30_MAX7D | LOT={LOT} | Prem<={PREM_MAX}",2,fs=10)
        hds=["#","Entry","Exit","Strike","Prem","ExitPrem","PnL(pts)","PnL(Rs)"]
        x_pos=[0.02,0.10,0.28,0.42,0.50,0.58,0.66,0.74]
        for j,h in enumerate(hds):
            ax.text(x_pos[j],0.90,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
        for k,tb_ in enumerate(chunk):
            yy=0.88-k*0.021
            ax.text(x_pos[0],yy,str(page_start+k+1),fontsize=6,transform=ax.transAxes)
            ax.text(x_pos[1],yy,tb_["entry"].strftime("%Y-%m-%d %H:%M"),fontsize=6,transform=ax.transAxes)
            ax.text(x_pos[2],yy,tb_["exit"].strftime("%Y-%m-%d %H:%M"),fontsize=6,transform=ax.transAxes)
            ax.text(x_pos[3],yy,str(tb_["strike"]),fontsize=6,transform=ax.transAxes)
            ax.text(x_pos[4],yy,f"{tb_['entry_prem']:.1f}",fontsize=6,transform=ax.transAxes)
            ax.text(x_pos[5],yy,f"{tb_['exit_prem']:.1f}",fontsize=6,transform=ax.transAxes)
            pnl_c="green" if tb_["pnl"]>0 else "red"
            ax.text(x_pos[6],yy,f"{tb_['pnl']:+.1f}",fontsize=6,color=pnl_c,transform=ax.transAxes)
            ax.text(x_pos[7],yy,f"Rs{tb_['pnl']*LOT:+,.0f}",fontsize=6,color=pnl_c,transform=ax.transAxes)
        pdf.savefig(fig); plt.close()

    # PAGE 5+: Trade book continuation + stats pages
    # Distribution chart
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    ax.hist(pnls,bins=30,color="steelblue",edgecolor="white",alpha=0.7)
    ax.axvline(x=0,color="red",ls="--",lw=1)
    ax.axvline(x=np.mean(pnls),color="green",ls="--",lw=1,label=f"Mean={np.mean(pnls):+.1f}")
    ax.set_title("PnL Distribution"); ax.set_xlabel("PnL (pts)"); ax.set_ylabel("Frequency")
    ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # Monthly net bar chart
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    mo_bar=tb.groupby(["yr","mo"])["pnl"].sum().reset_index()
    mo_bar["label"]=mo_bar["yr"].astype(str)+"-"+mo_bar["mo"].apply(lambda x:f"{int(x):02d}")
    colors=["green" if v>0 else "red" for v in mo_bar["pnl"]]
    ax.bar(range(len(mo_bar)),mo_bar["pnl"],color=colors,alpha=0.7)
    ax.set_title("Monthly Net PnL"); ax.set_xticks(range(len(mo_bar)))
    ax.set_xticklabels(mo_bar["label"],rotation=90,fontsize=6)
    ax.axhline(y=0,color="black",lw=0.5); ax.grid(alpha=0.3,axis="y")
    pdf.savefig(fig); plt.close()

    # Win rate by year
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    yr_wr=tb.groupby("yr")["pnl"].agg(lambda x:(x>0).mean())
    yr_count=tb.groupby("yr")["pnl"].count()
    ax.bar(yr_wr.index,yr_wr.values,color="steelblue",alpha=0.7)
    ax.set_title("Win Rate by Year"); ax.set_ylabel("Win Rate")
    ax.set_ylim(0,1); ax.axhline(y=0.5,color="red",ls="--",alpha=0.5)
    for y_,v in yr_wr.items():
        ax.text(y_,v+0.02,f"{v:.0%}",ha="center",fontsize=9)
    ax.grid(alpha=0.3,axis="y")
    pdf.savefig(fig); plt.close()

    # PAGE: Consecutive wins/losses
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    seqs=[1 if p>0 else 0 for p in pnls]
    runs=[]; cur=seqs[0]; cnt=1
    for s in seqs[1:]:
        if s==cur: cnt+=1
        else: runs.append((cur,cnt)); cur=s; cnt=1
    runs.append((cur,cnt))
    win_runs=[c for w,c in runs if w==1]
    loss_runs=[c for w,c in runs if w==0]
    ax.bar(range(len(win_runs)),win_runs,color="green",alpha=0.5,label="Win Streaks")
    if loss_runs:
        ax.bar(range(len(loss_runs)),loss_runs,color="red",alpha=0.5,label="Loss Streaks")
    ax.set_title("Win/Loss Streaks"); ax.set_ylabel("Consecutive Trades"); ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

print(f"PDF saved: {PDF_NAME}")

print(f"\nFIXED RESULTS (same-strike tracking):")
print(f"  Trades: {len(pnls)}")
print(f"  Net: {sum(pnls):+,.0f} pts (Rs {sum(pnls)*LOT:+,.0f})")
print(f"  WR: {sum(1 for p in pnls if p>0)/len(pnls)*100:.1f}%")
print(f"  Avg: {np.mean(pnls):+.1f}")
print(f"  Max: {max(pnls):+.1f} | Min: {min(pnls):+.1f}")
print(f"  Sharpe: {np.mean(pnls)/np.std(pnls)*np.sqrt(252):.2f}" if np.std(pnls)>0 else "")
print(f"  Final Capital: Rs {100000+sum(pnls)*LOT:,} (+{(100000+sum(pnls)*LOT)/100000*100-100:.1f}%)")
