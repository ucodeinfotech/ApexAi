"""Full PDF: Comprehensive same-day analysis + trade book for best strategies"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT=50
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
plt.rcParams.update({"font.size":7,"axes.titlesize":10,"axes.labelsize":8,"figure.dpi":120,"savefig.dpi":150})
DB_PATH=r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# ============ SPOT ENGINE ============
def atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
a5=atr(m5);me=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()
trades=[]
b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];rr=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
    if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
    lv=h1["high"].iloc[i];ts=h1["datetime"].iloc[i]
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
    ed=m5["datetime"].iloc[ri];ep_=m5["close"].iloc[ri]
    if ep_-m5["low"].iloc[ri]<=0: continue
    he=ep_
    for j in range(ri,len(m5["close"])):
        ca=a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month,"weekday":ed.weekday(),"entry_time":ed.time()})
            break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)

# ============ OPTION DATA ============
con=duckdb.connect(DB_PATH)
df_atm=con.execute("""SELECT timestamp,close,strike FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"]=pd.to_datetime(df_atm["timestamp"],utc=False)
atm_ts=df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl=df_atm["close"].values.astype(float)
atm_st=df_atm["strike"].values.astype(float)
def lookup_atm(ed):
    ts64=np.datetime64(ed,"us");i=np.searchsorted(atm_ts,ts64)
    if i>=len(atm_ts): return len(atm_ts)-1,atm_cl[-1],atm_st[-1]
    if i==0: return 0,atm_cl[0],atm_st[0]
    return (i,atm_cl[i],atm_st[i]) if atm_ts[i]==ts64 else (i-1,atm_cl[i-1],atm_st[i-1])
strike_set=set()
for ed in trades_pre["ed_naive"]: _,_,st=lookup_atm(ed); strike_set.add(int(st))
stk_list=sorted(strike_set)
con2=duckdb.connect(DB_PATH)
stk_where=",".join(str(s) for s in stk_list)
df_all=con2.execute(f"""SELECT timestamp,close,strike FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({stk_where}) ORDER BY strike,timestamp""").fetchdf()
con2.close()
df_all["timestamp"]=pd.to_datetime(df_all["timestamp"],utc=False)
strike_cache={}
for stk,grp in df_all.groupby("strike"):
    ts=grp["timestamp"].values.astype("datetime64[us]");cl=grp["close"].values.astype(float)
    strike_cache[int(stk)]={"ts":ts,"cl":cl}
trade_infos=[]
for ed in trades_pre["ed_naive"]:
    ts64=np.datetime64(ed,"us");i=np.searchsorted(atm_ts,ts64)
    if i>=len(atm_ts): si=len(atm_ts)-1
    elif i==0: si=0
    else: si=i if atm_ts[i]==ts64 else i-1
    st=int(atm_st[si]);sd=strike_cache.get(st)
    if sd is None: trade_infos.append(None); continue
    s_idx=np.searchsorted(sd["ts"],atm_ts[si])
    if s_idx>=len(sd["cl"]): trade_infos.append(None); continue
    trade_infos.append({"strike":st,"ep":float(sd["cl"][s_idx]),"s_idx":s_idx,"stk_data":sd})

def exit_tp_eod(stk_data,s_idx,tp):
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m");last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    for i in range(s_idx+1,last_idx+1):
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

def run(trade_infos,tp,pmax=None,days=None,eh_max=None):
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if pmax is not None and info["ep"]>pmax: continue
        if days is not None and trades_pre.iloc[i]["weekday"] not in days: continue
        if eh_max is not None and trades_pre.iloc[i]["entry_time"].hour>=eh_max: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    return np.array(pnls)

def stats(pnls):
    n=len(pnls);net=pnls.sum();wr=(pnls>0).mean()*100;avg=pnls.mean()
    std=pnls.std() if n>1 else 0;sharpe=avg/std*np.sqrt(252) if std>0 else 0
    cum=np.cumsum(pnls);mx=np.maximum.accumulate(cum);mdd=(mx-cum).max() if len(cum)>0 else 0
    calmar=net/mdd if mdd>0 else 0;pf=pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999
    return {"n":n,"net":net,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pf":pf}

# Build trade book for the final pick: TP35_MonThu_E<12
TP=35; DAYS=[0,1,2,3]; EH=12
best_pnls=[]; best_trades=[]
for i,info in enumerate(trade_infos):
    if info is None: continue
    if trades_pre.iloc[i]["weekday"] not in DAYS: continue
    if trades_pre.iloc[i]["entry_time"].hour>=EH: continue
    r=exit_tp_eod(info["stk_data"],info["s_idx"],TP)
    if r[0] is None: continue
    xp,xts=r; pnl=round(xp-info["ep"],1); best_pnls.append(pnl)
    best_trades.append({"entry":trades_pre.iloc[i]["ed_naive"],
        "exit":pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts,np.datetime64) else xts,
        "strike":info["strike"],"ep":round(info["ep"],1),"xp":round(xp,1),
        "pnl":pnl,"yr":trades_pre.iloc[i]["yr"],"mo":trades_pre.iloc[i]["mo"],
        "wd":trades_pre.iloc[i]["weekday"],"et":trades_pre.iloc[i]["entry_time"]})
best_pnls=np.array(best_pnls)
n=len(best_pnls);net=best_pnls.sum();wr=(best_pnls>0).mean()*100
trade_pnl_rs=[x*LOT for x in best_pnls];net_rs=sum(trade_pnl_rs);cap=100000;final=cap+net_rs
aw=best_pnls[best_pnls>0].mean() if (best_pnls>0).sum()>0 else 0
al=best_pnls[best_pnls<0].mean() if (best_pnls<0).sum()>0 else 0
cum=np.cumsum(best_pnls);mx_=np.maximum.accumulate(cum);mdd_=(mx_-cum).max()
sharpe=best_pnls.mean()/best_pnls.std()*np.sqrt(252) if best_pnls.std()>0 else 0
calmar_=net/mdd_ if mdd_>0 else 999
pf_=best_pnls[best_pnls>0].sum()/abs(best_pnls[best_pnls<0].sum()) if (best_pnls<0).sum()>0 else 999

# Generate comparison table for multi-day hold (same data)
def exit_md(stk_data,s_idx,tp,maxd):
    end_ns=stk_data["ts"][s_idx]+np.timedelta64(int(maxd*86400*1e6),"us");ep=stk_data["cl"][s_idx]
    for i in range(s_idx+1,min(s_idx+3000,len(stk_data["cl"]))):
        if stk_data["ts"][i]>end_ns: return stk_data["cl"][i],stk_data["ts"][i]
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return None,None
md_pnls=[]; md_trades=[]
for i,info in enumerate(trade_infos):
    if info is None: continue
    r=exit_md(info["stk_data"],info["s_idx"],200,10)
    if r[0] is None: continue
    md_pnls.append(round(r[0]-info["ep"],1))
md_pnls=np.array(md_pnls)

# ============ GENERATE PDF ============
PDF_NAME="SameDay_FullAnalysis.pdf"
with PdfPages(PDF_NAME) as pdf:
    # PAGE 1: Title + Summary
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
    yl=lambda y:0.95-y*0.026
    def t2(t,y,fs=10,wt="bold"): ax.text(0.06,yl(y),t,fontsize=fs,fontweight=wt,transform=ax.transAxes,va="top")
    t2("NIFTY50 SAME-DAY OPTION STRATEGY",0,fs=14)
    t2("Comprehensive Analysis & Trade Book",1,fs=10)
    t2("",2)
    t2("STRATEGY: TP35_MonThu_E<12",3,fs=12)
    t2("Entry: Bullish Engulfing (1H) + Breakout+Retest (5M) before 12:00",4)
    t2("Exit: TP=35 pts on option, or last bar of same day (EOD ~15:20-15:30)",5)
    t2("Days: Monday through Thursday only (skip Friday)",6)
    t2("Data: options_data_clean | Same-strike tracking | LOT="+str(LOT),7)
    t2("Period: 2021-06-14 to 2026-06-16",8)
    t2("",9)
    t2("--- PERFORMANCE SUMMARY ---",10,fs=11)
    t2(f"Total Trades: {n}",11)
    t2(f"Net PnL: {net:+,.0f} pts = Rs {net_rs:,}",12)
    t2(f"Win Rate: {wr:.1f}%",13)
    t2(f"Average Trade: {best_pnls.mean():+.1f} pts = Rs {best_pnls.mean()*LOT:+,.0f}",14)
    t2(f"Average Win: {aw:+.1f} pts | Average Loss: {al:+.1f} pts",15)
    t2(f"Max Win: {best_pnls.max():+.1f} pts | Max Loss: {best_pnls.min():+.1f} pts",16)
    t2(f"Sharpe Ratio: {sharpe:.2f}",17)
    t2(f"Calmar Ratio: {calmar_:.1f}x",18)
    t2(f"Profit Factor: {pf_:.2f}x",19)
    t2(f"Max Drawdown: {mdd_:,.0f} pts = Rs {mdd_*LOT:,}",20)
    t2(f"Initial Capital: Rs {cap:,} → Final Capital: Rs {final:,} ({(final/cap-1)*100:+.1f}%)",21)
    pdf.savefig(fig);plt.close()

    # PAGE 2: Equity curve + DD
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.27,11.69),gridspec_kw={"height_ratios":[2,1]})
    cum_rs=np.cumsum(trade_pnl_rs)+cap
    ax1.plot(cum_rs,color="green",lw=0.8);ax1.axhline(y=cap,color="gray",ls="--",alpha=0.5)
    ax1.fill_between(range(len(cum_rs)),cap,cum_rs,where=(cum_rs>=cap),color="green",alpha=0.08)
    ax1.fill_between(range(len(cum_rs)),cap,cum_rs,where=(cum_rs<cap),color="red",alpha=0.08)
    ax1.set_title("Equity Curve");ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"Rs{x:,.0f}"));ax1.grid(alpha=0.3)
    tb=pd.DataFrame(best_trades)
    yr_cum=tb.copy();yr_cum["pnl_rs"]=yr_cum["pnl"]*LOT
    for y_ in sorted(yr_cum["yr"].unique()):
        yt=yr_cum[yr_cum["yr"]==y_].sort_values("entry")
        ycum=np.cumsum(yt["pnl_rs"].values)
        ax1.plot(range(sum(yr_cum["yr"]<y_),sum(yr_cum["yr"]<y_)+len(ycum)),ycum+cap,label=str(int(y_)),lw=1.5,alpha=0.7)
    dd_=(mx_-cum)*LOT
    ax2.fill_between(range(len(dd_)),0,dd_,color="red",alpha=0.3)
    ax2.set_title("Drawdown (Rs)");ax2.set_ylabel("Drawdown");ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"Rs{x:,.0f}"));ax2.grid(alpha=0.3)
    pdf.savefig(fig);plt.close()

    # PAGE 3: Yearly + Monthly + Day-of-week
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
    yr_=tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),
                             avg=("pnl","mean"),mxx=("pnl","max"),mnn=("pnl","min"))
    t2("YEARLY BREAKDOWN",0,fs=11,wt="bold")
    hd=["Year","Trd","Net(pt)","Net(Rs)","WR","Avg","Max","Min"]
    xs=[0.04,0.08,0.16,0.28,0.38,0.46,0.54,0.62]
    for j,h in enumerate(hd): ax.text(xs[j],0.86,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    for k,(y_,r) in enumerate(yr_.iterrows()):
        yy=0.84-k*0.028
        vals=[str(int(y_)),str(int(r["trades"])),f"{r['net']:+,.0f}",f"Rs{r['net']*LOT:+,.0f}",f"{r['wr']:.0%}",f"{r['avg']:+.1f}",f"{r['mxx']:+.1f}",f"{r['mnn']:+.1f}"]
        for j,v in enumerate(vals): ax.text(xs[j],yy,v,fontsize=6.5,transform=ax.transAxes)
    t2(f"Total: {n}trd {net:+,.0f}pts Rs{net_rs:,} WR{wr:.0f}%",0.84-(len(yr_)+1)*0.028,fs=8,wt="bold")

    mo_=tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    t2("MONTHLY BREAKDOWN",0.84-(len(yr_)+4)*0.028,fs=11,wt="bold")
    bm=0.84-(len(yr_)+6)*0.028
    xs2=[0.04,0.12,0.22,0.34,0.44]
    for j,h in enumerate(["Mon","N","Net(pt)","Net(Rs)","WR"]): ax.text(xs2[j],bm,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    for k,(m_,r) in enumerate(mo_.iterrows()):
        yy=bm-(k+1)*0.030
        ax.text(xs2[0],yy,MONTHS[int(m_)-1],fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[1],yy,str(int(r["trades"])),fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[2],yy,f"{r['net']:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[4],yy,f"{r['wr']:.0%}",fontsize=6.5,transform=ax.transAxes)

    dow=tb.groupby("wd").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),avg=("pnl","mean"))
    t2("DAY-OF-WEEK BREAKDOWN",bm-15*0.030,fs=11,wt="bold")
    bm2=bm-17*0.030
    for j,h in enumerate(["Day","N","Net(pt)","Net(Rs)","WR","Avg"]):
        ax.text([0.04,0.12,0.22,0.34,0.44,0.52][j],bm2,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    for k,(d_,r) in enumerate(dow.iterrows()):
        yy=bm2-(k+1)*0.030
        ax.text(0.04,yy,["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(d_)],fontsize=6.5,transform=ax.transAxes)
        ax.text(0.12,yy,str(int(r["trades"])),fontsize=6.5,transform=ax.transAxes)
        ax.text(0.22,yy,f"{r['net']:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(0.34,yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(0.44,yy,f"{r['wr']:.0%}",fontsize=6.5,transform=ax.transAxes)
        ax.text(0.52,yy,f"{r['avg']:+.1f}",fontsize=6.5,transform=ax.transAxes)
    pdf.savefig(fig);plt.close()

    # PAGE 4: Strategy comparison matrix
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
    t2("STRATEGY COMPARISON MATRIX",0,fs=12,wt="bold")
    t2("All 85 strategy combinations tested (TP, filters, day/entry combinations)",1,fs=7)
    # Compute all strategies
    all_res={}
    for tp in [5,8,10,12,15,18,20,25,30,35,40,45,50,60,75]:
        for desc,pmax,days,eh in [("",None,None,None),("_E<12",None,None,12),("_MonThu_E<12",None,[0,1,2,3],12),
                                   ("_Mon_E<12",None,[0],12),("_Fri_E<12",None,[4],12),("_Entry13",None,None,13)]:
            pnls=run(trade_infos,tp,pmax,days,eh)
            if len(pnls)>=3:
                s=stats(pnls);all_res[f"TP{tp}{desc}"]=s
    for tp in [20,25,30,35,40,50,75]:
        for pm in [80,100,120]:
            pnls=run(trade_infos,tp,pmax=pm,days=DAYS,eh_max=12)
            if len(pnls)>=3: all_res[f"TP{tp}_P{pm}_MonThu_E<12"]=stats(pnls)
    top40=sorted(all_res.items(),key=lambda x:x[1]["net"],reverse=True)[:40]
    t2("TOP 40 STRATEGIES BY NET PNL",3,fs=10,wt="bold")
    mx_hd=["#","Strategy","N","Net(pt)","Net(Rs)","WR","Avg","Sharp","Cal","PF"]
    mx_xs=[0.02,0.05,0.11,0.22,0.34,0.42,0.49,0.56,0.64,0.72]
    for j,h in enumerate(mx_hd):
        ax.text(mx_xs[j],0.80,h,fontsize=5.5,fontweight="bold",transform=ax.transAxes)
    for rank,(name,r) in enumerate(top40):
        if rank>=38: continue
        yy=0.78-rank*0.018
        ax.text(mx_xs[0],yy,str(rank+1),fontsize=4.5,transform=ax.transAxes)
        ax.text(mx_xs[1],yy,name[:22],fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[2],yy,str(r["n"]),fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[3],yy,f"{r['net']:+,.0f}",fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[4],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[5],yy,f"{r['wr']:.1f}",fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[6],yy,f"{r['avg']:+.1f}",fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[7],yy,f"{r['sharpe']:.2f}",fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[8],yy,f"{r['calmar']:.1f}",fontsize=4.2,transform=ax.transAxes)
        ax.text(mx_xs[9],yy,f"{r['pf']:.2f}",fontsize=4.2,transform=ax.transAxes)
    is_sel=all_res.get("TP35_MonThu_E<12")
    if is_sel:
        t2(f"** SELECTED: TP35_MonThu_E<12 | {is_sel['n']}trd | {is_sel['net']:+,.0f}pts | Rs{is_sel['net']*LOT:,} | WR{is_sel['wr']:.1f}% | Sharpe{is_sel['sharpe']:.2f} | Calmar{is_sel['calmar']:.1f}x",0.03,fs=7,wt="bold")
    pdf.savefig(fig);plt.close()

    # PAGE 5: Multi-day hold comparison
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
    t2("MULTI-DAY HOLD COMPARISON (SAME DATA)",0,fs=12,wt="bold")
    t2("Same-strike tracking, same entry rules, longer holding period",1,fs=8)
    md_res={}
    for tp,maxd,desc in [(30,7,"TP30_Max7d"),(75,10,"TP75_Max10d"),(200,10,"TP200_Max10d"),
                         (30,7,"TP30_Max7d_Prem130"),(75,10,"TP75_Max10d_Prem120")]:
        pmax=130 if "130" in desc else (120 if "120" in desc else None)
        pnls=[]; pnl_arr=[]
        for i,info in enumerate(trade_infos):
            if info is None: continue
            if pmax is not None and info["ep"]>pmax: continue
            r=exit_md(info["stk_data"],info["s_idx"],tp,maxd)
            if r[0] is None: continue
            pnls.append(round(r[0]-info["ep"],1))
        if pnls: md_res[desc]=stats(np.array(pnls))
    t2("Multi-Day Hold Results:",3,fs=9,wt="bold")
    md_hd=["Strategy","N","Net(pt)","Net(Rs)","WR","Avg","Sharpe","Calmar"]
    md_xs=[0.04,0.08,0.16,0.28,0.38,0.46,0.54,0.62]
    for j,h in enumerate(md_hd): ax.text(md_xs[j],0.80,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    for k,(name,r) in enumerate(md_res.items()):
        yy=0.78-k*0.030
        ax.text(md_xs[0],yy,name[:25],fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[1],yy,str(r["n"]),fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[2],yy,f"{r['net']:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[4],yy,f"{r['wr']:.1f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[5],yy,f"{r['avg']:+.1f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[6],yy,f"{r['sharpe']:.2f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(md_xs[7],yy,f"{r['calmar']:.1f}",fontsize=6.5,transform=ax.transAxes)
    t2("Comparison vs Same-Day TP35_MonThu_E<12:",0.78-(len(md_res)+2)*0.030,fs=9,wt="bold")
    sd_net=all_res.get("TP35_MonThu_E<12",{}).get("net",0) if "TP35_MonThu_E<12" in all_res else 0
    for name,r in md_res.items():
        ratio=r["net"]/sd_net if sd_net!=0 else 0
        yy=0.78-(len(md_res)+4+(len(list(md_res.keys())).__class__==list and len(md_res) or 0))*0.030
    for idx,(name,r) in enumerate(md_res.items()):
        yy=0.78-(len(md_res)+4+idx)*0.030
        ratio=r["net"]/sd_net if sd_net else 0
        ax.text(0.06,yy,f"{name}: {r['net']:+,.0f}pts vs SD {sd_net:+,.0f}pts = {ratio:.1f}x",fontsize=7,transform=ax.transAxes)
    pdf.savefig(fig);plt.close()

    # PAGE 6: PnL distribution + streaks
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.27,11.69),gridspec_kw={"height_ratios":[1,1]})
    ax1.hist(best_pnls,bins=25,color="steelblue",edgecolor="white",alpha=0.7)
    ax1.axvline(x=0,color="red",ls="--",lw=1)
    ax1.axvline(x=best_pnls.mean(),color="green",ls="--",lw=1,label=f"Mean={best_pnls.mean():+.1f}")
    ax1.axvline(x=np.median(best_pnls),color="orange",ls=":",lw=1,label=f"Median={np.median(best_pnls):+.1f}")
    ax1.set_title("PnL Distribution");ax1.set_xlabel("PnL (pts)");ax1.set_ylabel("Freq");ax1.legend(fontsize=7);ax1.grid(alpha=0.3)
    seqs=[1 if p>0 else 0 for p in best_pnls]
    runs=[];cur=seqs[0];cnt=1
    for s in seqs[1:]:
        if s==cur: cnt+=1
        else: runs.append((cur,cnt));cur=s;cnt=1
    runs.append((cur,cnt))
    win_r=[c for w,c in runs if w==1];loss_r=[c for w,c in runs if w==0]
    wmax=max(win_r) if win_r else 0;lmax=max(loss_r) if loss_r else 0
    if win_r: ax2.bar(range(len(win_r)),win_r,color="green",alpha=0.5,label=f"Win (max={wmax})")
    if loss_r: ax2.bar(range(len(loss_r)),loss_r,color="red",alpha=0.5,label=f"Loss (max={lmax})")
    ax2.set_title("Win/Loss Streaks");ax2.set_ylabel("Consecutive");ax2.legend(fontsize=7);ax2.grid(alpha=0.3)
    pdf.savefig(fig);plt.close()

    # PAGE 7: Monthly bar chart
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    tb_=tb.copy();tb_["ym"]=tb_["yr"].astype(str)+"-"+tb_["mo"].apply(lambda x:f"{int(x):02d}")
    mo_bar=tb_.groupby("ym")["pnl"].sum().reset_index()
    colors=["green" if v>0 else "red" for v in mo_bar["pnl"]]
    ax.bar(range(len(mo_bar)),mo_bar["pnl"],color=colors,alpha=0.7)
    ax.set_title("Monthly Net PnL");ax.set_xticks(range(len(mo_bar)));ax.set_xticklabels(mo_bar["ym"],rotation=90,fontsize=5)
    ax.axhline(y=0,color="black",lw=0.5);ax.grid(alpha=0.3,axis="y")
    pdf.savefig(fig);plt.close()

    # PAGE 8+: TRADE BOOK
    tpb=tb.copy()
    tpb["entry_str"]=tpb["entry"].dt.strftime("%m/%d %H:%M")
    tpb["exit_str"]=pd.to_datetime(tpb["exit"]).dt.strftime("%m/%d %H:%M")
    tpb["dow"]=tpb["wd"].apply(lambda x:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    tpb["rs"]=tpb["pnl"]*LOT
    tpb_show=tpb[["entry_str","exit_str","strike","ep","xp","pnl","rs","dow"]]
    tpp=48
    for ps in range(0,len(tpb_show),tpp):
        fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
        chunk=tpb_show.iloc[ps:ps+tpp]
        t2(f"TRADE BOOK ({ps+1}-{ps+len(chunk)}/{len(tpb_show)})",0,fs=10,wt="bold")
        t2(f"TP35_MonThu_E<12 | LOT={LOT} | TP=35 | Exit=EOD",1,fs=7)
        hd_=["#","Entry","Exit","Strike","E.Prem","X.Prem","P(pt)","P(Rs)","Day"]
        xs_=[0.02,0.07,0.20,0.32,0.39,0.46,0.53,0.60,0.68]
        for j,h in enumerate(hd_): ax.text(xs_[j],0.91,h,fontsize=5.5,fontweight="bold",transform=ax.transAxes)
        for k,(_,r) in enumerate(chunk.iterrows()):
            yy=0.89-k*0.018
            ax.text(xs_[0],yy,str(ps+k+1),fontsize=4.2,transform=ax.transAxes)
            ax.text(xs_[1],yy,str(r["entry_str"])[:12],fontsize=4.2,transform=ax.transAxes)
            ax.text(xs_[2],yy,str(r["exit_str"])[:12],fontsize=4.2,transform=ax.transAxes)
            ax.text(xs_[3],yy,f'{r["strike"]:.0f}',fontsize=4.2,transform=ax.transAxes)
            ax.text(xs_[4],yy,f'{r["ep"]:.1f}',fontsize=4.2,transform=ax.transAxes)
            ax.text(xs_[5],yy,f'{r["xp"]:.1f}',fontsize=4.2,transform=ax.transAxes)
            c="green" if r["pnl"]>0 else "red"
            ax.text(xs_[6],yy,f'{r["pnl"]:+.1f}',fontsize=4.2,color=c,transform=ax.transAxes)
            ax.text(xs_[7],yy,f'Rs{r["rs"]:+,.0f}',fontsize=4.2,color=c,transform=ax.transAxes)
            ax.text(xs_[8],yy,r["dow"],fontsize=4.2,transform=ax.transAxes)
        pdf.savefig(fig);plt.close()

print(f"\nPDF: {PDF_NAME} | {n} trades | {net:+,.0f} pts | Rs{net_rs:,} | WR{wr:.1f}% | Sharpe{sharpe:.2f} | Calmar{calmar_:.1f}x")
