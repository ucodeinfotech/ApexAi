"""Final report: TP35_Mon_Entry<12 same-day strategy"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT = 50
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
plt.rcParams.update({"font.size":8,"axes.titlesize":11,"axes.labelsize":9,"figure.dpi":120,"savefig.dpi":150})
DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# Spot engine
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

# Load option data
con=duckdb.connect(DB_PATH)
df_atm=con.execute("""SELECT timestamp,close,strike FROM options_data_clean
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"]=pd.to_datetime(df_atm["timestamp"],utc=False)
atm_ts=df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl=df_atm["close"].values.astype(float)
atm_st=df_atm["strike"].values.astype(float)

def lookup_atm(ed):
    ts64=np.datetime64(ed,"us"); i=np.searchsorted(atm_ts,ts64)
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
    ts=grp["timestamp"].values.astype("datetime64[us]"); cl=grp["close"].values.astype(float)
    strike_cache[int(stk)]={"ts":ts,"cl":cl}

trade_infos=[]
for ed in trades_pre["ed_naive"]:
    ts64=np.datetime64(ed,"us"); i=np.searchsorted(atm_ts,ts64)
    if i>=len(atm_ts): si=len(atm_ts)-1
    elif i==0: si=0
    else: si=i if atm_ts[i]==ts64 else i-1
    st=int(atm_st[si]); sd=strike_cache.get(st)
    if sd is None: trade_infos.append(None); continue
    s_idx=np.searchsorted(sd["ts"],atm_ts[si])
    if s_idx>=len(sd["cl"]): trade_infos.append(None); continue
    trade_infos.append({"strike":st,"ep":float(sd["cl"][s_idx]),"s_idx":s_idx,"stk_data":sd})

def exit_tp_eod(stk_data,s_idx,tp):
    ep=stk_data["cl"][s_idx]; entry_ns=stk_data["ts"][s_idx]
    entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m")
    last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    for i in range(s_idx+1,last_idx+1):
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

# Strategy: TP35, Monday, Entry<12
TP=35; DAYS=[0]; ENTRY_HOUR=12
pnls=[]; best_trades=[]
for i,info in enumerate(trade_infos):
    if info is None: continue
    if trades_pre.iloc[i]["weekday"] not in DAYS: continue
    if trades_pre.iloc[i]["entry_time"].hour>=ENTRY_HOUR: continue
    r=exit_tp_eod(info["stk_data"],info["s_idx"],TP)
    if r[0] is None: continue
    xp,xts=r; pnl=round(xp-info["ep"],1); pnls.append(pnl)
    best_trades.append({"entry":trades_pre.iloc[i]["ed_naive"],
        "exit":pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts,np.datetime64) else xts,
        "strike":info["strike"],"entry_prem":round(info["ep"],1),"exit_prem":round(xp,1),
        "pnl":pnl,"yr":trades_pre.iloc[i]["yr"],"mo":trades_pre.iloc[i]["mo"],
        "wd":trades_pre.iloc[i]["weekday"],"et":trades_pre.iloc[i]["entry_time"]})
pnls=np.array(pnls); n=len(pnls); net=pnls.sum()
trade_pnl_rs=[x*LOT for x in pnls]; net_rs=sum(trade_pnl_rs); capital=100000; final_rs=capital+net_rs
wr=(pnls>0).mean(); aw=pnls[pnls>0].mean() if (pnls>0).sum()>0 else 0
al=pnls[pnls<0].mean() if (pnls<0).sum()>0 else 0
cum=np.cumsum(pnls); mx_=np.maximum.accumulate(cum); mdd_=(mx_-cum).max()
sharpe=pnls.mean()/pnls.std()*np.sqrt(252) if pnls.std()>0 else 0
calmar_=net/mdd_ if mdd_>0 else 999
pf_=pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999

# Generate PDF
PDF_NAME="SameDay_TP35_Mon_Entry12.pdf"
print(f"Trades: {n}, Net: {net:+,.0f}pts Rs{net_rs:+,.0f}, WR:{wr:.1%}")
with PdfPages(PDF_NAME) as pdf:
    # Summary
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
    yl=lambda y:0.94-y*0.027
    def t2(t,y,fs=11,wt="bold"): ax.text(0.06,yl(y),t,fontsize=fs,fontweight=wt,transform=ax.transAxes,verticalalignment="top")
    t2("NIFTY50 SAME-DAY OPTION - TP35 MONDAY ENTRY<12",0,fs=13)
    t2(f"Trades: {n} | Net: {net:+,.0f}pts = Rs{net_rs:+,.0f}",2)
    t2(f"WR: {wr:.1%} | Avg: {pnls.mean():+.1f}pts = Rs{pnls.mean()*LOT:+,.0f}",3)
    t2(f"AvgWin: {aw:+.1f} | AvgLoss: {al:+.1f} | MaxW: {pnls.max():+.1f} | MaxL: {pnls.min():+.1f}",4)
    t2(f"Sharpe: {sharpe:.2f} | Calmar: {calmar_:.1f}x | MDD: {mdd_:,.0f}pts = Rs{mdd_*LOT:,.0f}",5)
    t2(f"Profit Factor: {pf_:.2f}x",6)
    t2(f"Capital: Rs{capital:,} -> Rs{final_rs:,} ({(final_rs/capital-1)*100:+.1f}%)",7)
    t2(f"Entry: BE(1H) + Bro/Ret(5M) | Exit: TP{TP} or EOD last bar",8)
    t2(f"Filter: Monday only, entry before 12:00 | Same-strike | LOT={LOT}",9)
    pdf.savefig(fig);plt.close()

    # Equity
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.27,11.69),gridspec_kw={"height_ratios":[2,1]})
    cum_rs=np.cumsum(trade_pnl_rs)+capital
    ax1.plot(cum_rs,color="green",lw=1);ax1.axhline(y=capital,color="gray",ls="--",alpha=0.5)
    ax1.fill_between(range(len(cum_rs)),capital,cum_rs,where=(cum_rs>=capital),color="green",alpha=0.1)
    ax1.fill_between(range(len(cum_rs)),capital,cum_rs,where=(cum_rs<capital),color="red",alpha=0.1)
    ax1.set_title(f"Equity - TP35 Mon Entry<12");ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"Rs{x:,.0f}"));ax1.grid(alpha=0.3)
    dd_=(mx_-cum)*LOT
    ax2.fill_between(range(len(dd_)),0,dd_,color="red",alpha=0.3)
    ax2.set_title("Drawdown");ax2.set_ylabel("Drawdown (Rs)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"Rs{x:,.0f}"));ax2.grid(alpha=0.3)
    pdf.savefig(fig);plt.close()

    # Yearly + Monthly
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
    tb=pd.DataFrame(best_trades)
    yr=tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),
                            avg=("pnl","mean"),mxx=("pnl","max"),mnn=("pnl","min"))
    t2("Yearly",0,fs=12,wt="bold")
    hd=["Year","N","Net(pt)","Net(Rs)","WR","Avg","Max","Min"]
    xs=[0.04,0.08,0.16,0.28,0.38,0.46,0.54,0.62]
    for j,h in enumerate(hd): ax.text(xs[j],0.88,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    for k,(y_,r) in enumerate(yr.iterrows()):
        yy=0.86-k*0.028
        ax.text(xs[0],yy,str(int(y_)),fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[1],yy,str(int(r["trades"])),fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[2],yy,f"{r['net']:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[4],yy,f"{r['wr']:.0%}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[5],yy,f"{r['avg']:+.1f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[6],yy,f"{r['mxx']:+.1f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs[7],yy,f"{r['mnn']:+.1f}",fontsize=6.5,transform=ax.transAxes)
    t2(f"Total: {n}trd {net:+,.0f}pts Rs{net_rs:+,.0f} WR{wr:.0%}",0.86-(len(yr)+1)*0.028,fs=8,wt="bold")
    mo=tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    t2("Monthly",0.86-(len(yr)+4)*0.028,fs=12,wt="bold")
    bm=0.86-(len(yr)+6)*0.028
    xs2=[0.04,0.12,0.22,0.34,0.44]
    for j,h in enumerate(["Mon","N","Net(pt)","Net(Rs)","WR"]):
        ax.text(xs2[j],bm,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    for k,(m_,r) in enumerate(mo.iterrows()):
        yy=bm-(k+1)*0.030
        ax.text(xs2[0],yy,MONTHS[int(m_)-1],fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[1],yy,str(int(r["trades"])),fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[2],yy,f"{r['net']:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=6.5,transform=ax.transAxes)
        ax.text(xs2[4],yy,f"{r['wr']:.0%}",fontsize=6.5,transform=ax.transAxes)
    pdf.savefig(fig);plt.close()

    # Trade book
    tpb=tb.copy()
    tpb["entry_str"]=tpb["entry"].dt.strftime("%m/%d %H:%M")
    tpb["exit_str"]=pd.to_datetime(tpb["exit"]).dt.strftime("%m/%d %H:%M")
    tpb["dow"]=tpb["wd"].apply(lambda x:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    tpb["rs"]=tpb["pnl"]*LOT
    tpb_show=tpb[["entry_str","exit_str","strike","entry_prem","exit_prem","pnl","rs","dow"]]
    tpp=45
    for ps in range(0,len(tpb_show),tpp):
        fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis("off")
        chunk=tpb_show.iloc[ps:ps+tpp]
        t2(f"Trade Book ({ps+1}-{ps+len(chunk)}/{len(tpb_show)})",0,fs=10,wt="bold")
        t2(f"TP35 Mon Entry<12 LOT={LOT}",1,fs=7)
        hd_=["#","Entry","Exit","Strike","E.P","X.P","P(pt)","P(Rs)","Day"]
        xs_=[0.02,0.07,0.20,0.32,0.39,0.46,0.53,0.60,0.68]
        for j,h in enumerate(hd_): ax.text(xs_[j],0.91,h,fontsize=5.5,fontweight="bold",transform=ax.transAxes)
        for k,(_,r) in enumerate(chunk.iterrows()):
            yy=0.89-k*0.018
            ax.text(xs_[0],yy,str(ps+k+1),fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[1],yy,str(r["entry_str"])[:12],fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[2],yy,str(r["exit_str"])[:12],fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[3],yy,f'{r["strike"]:.0f}',fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[4],yy,f'{r["entry_prem"]:.1f}',fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[5],yy,f'{r["exit_prem"]:.1f}',fontsize=4.5,transform=ax.transAxes)
            c="green" if r["pnl"]>0 else "red"
            ax.text(xs_[6],yy,f'{r["pnl"]:+.1f}',fontsize=4.5,color=c,transform=ax.transAxes)
            ax.text(xs_[7],yy,f'Rs{r["rs"]:+,.0f}',fontsize=4.5,color=c,transform=ax.transAxes)
            ax.text(xs_[8],yy,r["dow"],fontsize=4.5,transform=ax.transAxes)
        pdf.savefig(fig);plt.close()

    # PnL histogram
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    ax.hist(pnls,bins=25,color="steelblue",edgecolor="white",alpha=0.7)
    ax.axvline(x=0,color="red",ls="--",lw=1);ax.axvline(x=pnls.mean(),color="green",ls="--",lw=1,label=f"Mean={pnls.mean():+.1f}")
    ax.axvline(x=np.median(pnls),color="orange",ls=":",lw=1,label=f"Median={np.median(pnls):+.1f}")
    ax.set_title("PnL Distribution - TP35 Mon Entry<12");ax.set_xlabel("PnL (pts)");ax.set_ylabel("Freq");ax.legend();ax.grid(alpha=0.3)
    pdf.savefig(fig);plt.close()

print(f"\nPDF: {PDF_NAME}")
print(f"Trades: {n} | Net: {net:+,.0f}pts = Rs{net_rs:+,.0f} | WR: {wr:.1%} | Sharpe: {sharpe:.2f} | Calmar: {calmar_:.1f}x")
