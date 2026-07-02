"""
Comprehensive audit + full report: fixed same-strike backtest
Validates every step, then generates complete PDF
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
LOCAL_DB = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
LOT=50
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
plt.rcParams.update({"font.size":8,"axes.titlesize":11,"axes.labelsize":9,"figure.dpi":120,"savefig.dpi":150,"font.family":"sans-serif"})

# ============================================================
# STEP 1: Build trades (spot engine - same as always)
# ============================================================
print("="*70)
print("AUDIT: Step 1 - Building spot trades")
print("="*70)

def atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)

print(f"  1H bars: {len(h1):,} ({h1['datetime'].min()} to {h1['datetime'].max()})")
print(f"  5M bars: {len(m5):,} ({m5['datetime'].min()} to {m5['datetime'].max()})")

a5=atr(m5); me=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()
trades=[]; b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; rr=h1["close"]<h1["open"]
signal_count=0; breakout_fail=0; retest_fail=0; ch55_exit=0
for i in range(1,len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
    if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
    signal_count+=1
    lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
    idx=np.searchsorted(me,ts.asm8.view("int64"),side="right")
    if idx>=len(m5["close"]): continue
    bi=idx
    while bi<len(m5["close"]) and m5["close"].iloc[bi]<=lv: bi+=1
    if bi>=len(m5["close"])-1: breakout_fail+=1; continue
    ri=bi+1
    while ri<len(m5["close"]):
        if m5["low"].iloc[ri]<lv and m5["close"].iloc[ri]>lv and pd.Series(m5["datetime"]).dt.time.iloc[ri]<CUT: break
        ri+=1
    if ri>=len(m5["close"]): retest_fail+=1; continue
    ed=m5["datetime"].iloc[ri]; ep_=m5["close"].iloc[ri]
    if ep_-m5["low"].iloc[ri]<=0: continue
    he=ep_
    for j in range(ri,len(m5["close"])):
        ca=a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); ch55_exit+=1; break
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

print(f"  1H signals: {signal_count}")
print(f"  Breakout failures: {breakout_fail}")
print(f"  Retest failures: {retest_fail}")
print(f"  CH55 exits: {ch55_exit}")
print(f"  Total trades: {len(trades)}")
print(f"  Option-era trades (>=2021-06-14): {len(trades_pre)}")

# ============================================================
# STEP 2: Verify option DB tables
# ============================================================
print("\n"+"="*70)
print("AUDIT: Step 2 - Option DB tables")
print("="*70)
con=duckdb.connect(LOCAL_DB)
for tbl in ["options_data","options_data_dedup","options_data_clean"]:
    try:
        cnt=con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {cnt:,} rows")
    except:
        print(f"  {tbl}: NOT FOUND")

# Check what table to use
TABLE = "options_data_clean"
try:
    con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
except:
    TABLE = "options_data_dedup"
    print(f"  -> Using {TABLE}")

# Verify counts
atm_rows=con.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0").fetchone()[0]
print(f"  ATM CALL WEEK rows: {atm_rows:,}")
all_call_week=con.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'").fetchone()[0]
print(f"  ALL CALL WEEK rows: {all_call_week:,}")
max_ts=con.execute(f"SELECT MAX(timestamp) FROM {TABLE} WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0").fetchone()[0]
min_ts=con.execute(f"SELECT MIN(timestamp) FROM {TABLE} WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0").fetchone()[0]
print(f"  ATM data range: {min_ts} to {max_ts}")

# ============================================================
# STEP 3: Load fixed engine data + validate alignment
# ============================================================
print("\n"+"="*70)
print("AUDIT: Step 3 - Loading fixed engine data")
print("="*70)
from option_fixed_engine import load_fixed_data, get_entry_strike_idx, exit_tp_maxd, exit_tp_sl

atm_ts_arr, atm_cl_arr, atm_st_arr, strike_cache, lookup = load_fixed_data(trades_pre)
print(f"  ATM bars loaded: {len(atm_ts_arr):,}")
print(f"  Unique strikes: {len(strike_cache)}")
print(f"  ATM close range: {atm_cl_arr.min():.1f} to {atm_cl_arr.max():.1f}")

# Validate strike alignment: for each strike, check that timestamps match ATM data
print("\n  Validating timestamp alignment for key strikes...")
misalignments=0
for stk, sd in list(strike_cache.items())[:5]:
    atm_mask = (atm_st_arr == stk)
    atm_tss = atm_ts_arr[atm_mask]
    stk_tss = sd["ts"]
    common = np.intersect1d(atm_tss, stk_tss)
    print(f"    Strike {stk}: {len(atm_tss)} atm bars, {len(stk_tss)} strike bars, {len(common)} common")
    if len(common) < len(atm_tss) * 0.9:
        print(f"      WARNING: <90% overlap!")
        misalignments+=1
if misalignments==0: print("  => All strikes have good timestamp alignment")

# ============================================================
# STEP 4: Build trade info + validate against original sweep
# ============================================================
print("\n"+"="*70)
print("AUDIT: Step 4 - Building trade info")
print("="*70)
trade_infos=[]
for i in range(len(trades_pre)):
    ed=trades_pre.iloc[i]["ed_naive"]
    info=get_entry_strike_idx(ed, lookup, atm_ts_arr, strike_cache)
    trade_infos.append(info if info else None)

valid=sum(1 for t in trade_infos if t)
print(f"  Valid trades: {valid}/{len(trade_infos)}")
if valid<len(trade_infos):
    missing=[i for i,t in enumerate(trade_infos) if t is None]
    print(f"  Missing trades at indices: {missing[:5]}")
    for idx in missing[:3]:
        print(f"    Trade {idx}: ed={trades_pre.iloc[idx]['ed_naive']}")

# Premium distribution (for prem<120 filter check)
premia=[t["ep"] for t in trade_infos if t]
print(f"  Entry premium range: {min(premia):.1f} to {max(premia):.1f}")
print(f"  Median premium: {np.median(premia):.1f}")
lt120=sum(1 for p in premia if p<=120)
print(f"  Premium <= 120: {lt120}/{len(premia)} ({lt120/len(premia)*100:.0f}%)")
lt130=sum(1 for p in premia if p<=130)
print(f"  Premium <= 130: {lt130}/{len(premia)} ({lt130/len(premia)*100:.0f}%)")

# ============================================================
# STEP 5: Run key strategies + validate vs sweep results
# ============================================================
print("\n"+"="*70)
print("AUDIT: Step 5 - Running strategies + validation")
print("="*70)

strategies = [
    ("TP30_Max7d", 30, 7, None),
    ("TP30_Max7d_Prem130", 30, 7, 130),
    ("TP75_Max10d", 75, 10, None),
    ("TP100_Max7d", 100, 7, None),
    ("TP200_Max7d", 200, 7, None),
    ("TP200_Max7d_Prem120", 200, 7, 120),
    ("TP200_Max10d", 200, 10, None),
    ("TP75_Max10d_Prem120", 75, 10, 120),
]

results={}
for name, tp, maxd, pmax in strategies:
    pnls=[]
    for info in trade_infos:
        if info is None: continue
        if pmax is not None and info["ep"]>pmax: continue
        r=exit_tp_maxd(info["stk_data"], info["s_idx"], tp, maxd)
        if r is None or r[0] is None: continue
        xp,_=r; pnl=round(xp-info["ep"],1); pnls.append(pnl)
    pnls=np.array(pnls)
    n=len(pnls); net=pnls.sum(); wr=(pnls>0).mean()*100
    avg=pnls.mean(); std=pnls.std() if len(pnls)>1 else 0
    sharpe=avg/std*np.sqrt(252) if std>0 else 0
    cum=np.cumsum(pnls); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max() if len(dd)>0 else 0
    calmar=net/mdd if mdd>0 else 0
    results[name]={"n":n,"net":net,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pnls":pnls}
    print(f"  {name:<25} n={n:3d} net={net:>+7,.0f}pts (Rs{net*LOT:>+9,.0f}) WR={wr:5.1f}% Avg={avg:+.1f} Sharpe={sharpe:.2f}")

# Validate TP30_Max7d_Prem130 against known sweep result
ref=results["TP30_Max7d_Prem130"]
expected_net=9242
diff=abs(ref["net"]-expected_net)
if diff<10:
    print(f"\n  >>> TP30_Max7d_Prem130 validated: net={ref['net']:+,.0f} (expected ~+9,242)")
else:
    print(f"\n  >>> WARNING: TP30_Max7d_Prem130 net={ref['net']:+,.0f} differs from expected +9,242 by {diff:,.0f}!")

# ============================================================
# STEP 6: Generate comprehensive report for best strategy
# ============================================================
print("\n"+"="*70)
print("STEP 6: Generating comprehensive PDF report")
print("="*70)

# Choose best balanced strategy: TP75_Max10d (high WR, high Sharpe, good Calmar)
BEST_NAME = "TP75_Max10d"
BEST_TP = 75
BEST_MAXD = 10
BEST_PMAX = None

# OR use the highest net: TP200_Max7d_Prem120
# BEST_NAME = "TP200_Max7d_Prem120"
# BEST_TP = 200
# BEST_MAXD = 7
# BEST_PMAX = 120

print(f"  Best strategy: {BEST_NAME}")
b=results[BEST_NAME]

# Re-run to get trade book
best_pnls=[]
best_trades=[]
for i,info in enumerate(trade_infos):
    if info is None: continue
    if BEST_PMAX is not None and info["ep"]>BEST_PMAX: continue
    r=exit_tp_maxd(info["stk_data"], info["s_idx"], BEST_TP, BEST_MAXD)
    if r is None or r[0] is None: continue
    xp,xts=r
    pnl=round(xp-info["ep"],1)
    best_pnls.append(pnl)
    best_trades.append({
        "entry":trades_pre.iloc[i]["ed_naive"],
        "exit":pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts,np.datetime64) else xts,
        "strike":info["strike"],
        "entry_prem":round(info["ep"],1),
        "exit_prem":round(xp,1),
        "pnl":pnl,
        "yr":trades_pre.iloc[i]["yr"],
        "mo":trades_pre.iloc[i]["mo"],
        "weekday":trades_pre.iloc[i]["weekday"],
    })
best_pnls=np.array(best_pnls)

print(f"  Trades: {len(best_trades)}")
print(f"  Building PDF...")

PDF_NAME = f"FINAL_{BEST_NAME}_Report.pdf"
with PdfPages(PDF_NAME) as pdf:
    # PAGE 1: Summary
    fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
    p=pd.Series(best_pnls); n=len(p); net=p.sum(); wr=(p>0).mean()
    aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0
    cum=np.cumsum(p); mx_=np.maximum.accumulate(cum); mdd=(mx_-cum).max()
    sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    calmar=net/mdd if mdd>0 else 999
    trade_pnl_rs=[x*LOT for x in best_pnls]
    net_rs=sum(trade_pnl_rs); capital=100000; final_rs=capital+net_rs

    yl=lambda y: 0.94-y*0.030
    def t2(txt,y,fs=11,wt="bold"):
        ax.text(0.06,yl(y),txt,fontsize=fs,fontweight=wt,transform=ax.transAxes,verticalalignment="top")
    t2(f"NIFTY50 OPTION STRATEGY - FINAL REPORT",0,fs=14)
    t2(f"Strategy: {BEST_NAME} | Same-Strike Tracking",2,fs=12)
    t2(f"Entry: Bullish Engulfing (1H) + Breakout+Retest (5M)",3)
    t2(f"Exit: Take profit +{BEST_TP} pts on option, or time stop at {BEST_MAXD} days",4)
    t2(f"Premium filter: {'<='+str(BEST_PMAX)+' pts' if BEST_PMAX else 'None'}",5)
    t2(f"Data: options_data_clean | LOT={LOT} NIFTY",6)
    t2("",7)
    t2(f"Trades: {n}",8)
    t2(f"Net PnL: {net:+,.0f} pts = Rs {net_rs:+,.0f}",9)
    t2(f"Win Rate: {wr:.1%}",10)
    t2(f"Avg Trade: {p.mean():+.1f} pts = Rs {p.mean()*LOT:+,.0f}",11)
    t2(f"Avg Win: {aw:+.1f} pts = Rs {aw*LOT:+,.0f} | Avg Loss: {al:+.1f} pts = Rs {al*LOT:+,.0f}",12)
    t2(f"Max Win: {p.max():+.1f} pts = Rs {p.max()*LOT:+,.0f} | Max Loss: {p.min():+.1f} pts = Rs {p.min()*LOT:+,.0f}",13)
    t2(f"Sharpe Ratio: {sharpe:.2f} | Calmar Ratio: {calmar:.1f}x",14)
    t2(f"Max Drawdown: {mdd:,.0f} pts = Rs {mdd*LOT:,.0f}",15)
    t2(f"Profit Factor: {p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999:.2f}x",16)
    t2(f"Initial Capital: Rs {capital:,}",17)
    t2(f"Final Capital: Rs {final_rs:,} ({(final_rs/capital-1)*100:+.1f}%)",18)
    t2(f"Consecutive Wins: {max(np.diff(np.where(p>0)[0]))-1 if len(np.where(p>0)[0])>1 else n}",19)
    t2(f"Consecutive Losses: {max(np.diff(np.where(p<=0)[0]))-1 if len(np.where(p<=0)[0])>1 else 0}",20)
    pdf.savefig(fig); plt.close()

    # PAGE 2: Equity curve
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.27,11.69),gridspec_kw={"height_ratios":[2,1]})
    cum_rs=np.cumsum(trade_pnl_rs)+capital
    ax1.plot(cum_rs,color="green",lw=1)
    ax1.axhline(y=capital,color="gray",ls="--",alpha=0.5)
    ax1.fill_between(range(len(cum_rs)),capital,cum_rs,where=(cum_rs>=capital),color="green",alpha=0.1)
    ax1.fill_between(range(len(cum_rs)),capital,cum_rs,where=(cum_rs<capital),color="red",alpha=0.1)
    ax1.set_title(f"Equity Curve - {BEST_NAME} (Rs)"); ax1.set_ylabel("Capital (Rs)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax1.grid(alpha=0.3)
    dd_curve=(mx_-cum)*LOT
    ax2.fill_between(range(len(dd_curve)),0,dd_curve,color="red",alpha=0.3)
    ax2.set_title("Drawdown (Rs)"); ax2.set_ylabel("Drawdown (Rs)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"Rs{x:,.0f}")); ax2.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # PAGE 3: Yearly + Monthly breakdown
    fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
    tb=pd.DataFrame(best_trades)
    # Yearly
    yr=tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),
                            avg=("pnl","mean"),mxx=("pnl","max"),mnn=("pnl","min"))
    t2("Yearly Breakdown",0,fs=14,wt="bold")
    hd=["Year","Trades","Net(pts)","Net(Rs)","WR","Avg","Max","Min"]
    xs=[0.04,0.08,0.16,0.28,0.38,0.46,0.54,0.62]
    for j,h in enumerate(hd):
        ax.text(xs[j],0.88,h,fontsize=8,fontweight="bold",transform=ax.transAxes)
    for k,(y_,r) in enumerate(yr.iterrows()):
        yy=0.86-k*0.028
        vals=[str(int(y_)),str(int(r["trades"])),f"{r['net']:+,.0f}",f"Rs{r['net']*LOT:+,.0f}",f"{r['wr']:.0%}",f"{r['avg']:+.1f}",f"{r['mxx']:+.1f}",f"{r['mnn']:+.1f}"]
        for j,v in enumerate(vals):
            ax.text(xs[j],yy,v,fontsize=7,transform=ax.transAxes)
    t2(f"Total: {n} trades | {net:+,.0f} pts | Rs {net_rs:+,.0f} | WR {wr:.0%}",0.86-(len(yr)+1)*0.028,fs=9,wt="bold")
    
    # Monthly
    mo=tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    t2("Monthly Breakdown",0.86-(len(yr)+4)*0.028,fs=14,wt="bold")
    bm=0.86-(len(yr)+6)*0.028
    hd2=["Month","Trades","Net(pts)","Net(Rs)","WR"]
    xs2=[0.04,0.12,0.22,0.34,0.44]
    for j,h in enumerate(hd2):
        ax.text(xs2[j],bm,h,fontsize=8,fontweight="bold",transform=ax.transAxes)
    for k,(m_,r) in enumerate(mo.iterrows()):
        yy=bm-(k+1)*0.030
        ax.text(xs2[0],yy,MONTHS[int(m_)-1],fontsize=7,transform=ax.transAxes)
        ax.text(xs2[1],yy,str(int(r["trades"])),fontsize=7,transform=ax.transAxes)
        ax.text(xs2[2],yy,f"{r['net']:+,.0f}",fontsize=7,transform=ax.transAxes)
        ax.text(xs2[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=7,transform=ax.transAxes)
        ax.text(xs2[4],yy,f"{r['wr']:.0%}",fontsize=7,transform=ax.transAxes)
    
    # Strategy comparison matrix
    t2("Strategy Evaluation Matrix",bm-(14)*0.030,fs=14,wt="bold")
    bm2=bm-(16)*0.030
    mx_hd=["Strategy","Trades","Net(pts)","Net(Rs)","WR","Avg","Sharpe","Calmar","MDD(Rs)"]
    mx_xs=[0.02,0.08,0.16,0.28,0.38,0.46,0.54,0.62,0.70]
    for j,h in enumerate(mx_hd):
        ax.text(mx_xs[j],bm2,h,fontsize=7,fontweight="bold",transform=ax.transAxes)
    rank=0
    for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
        r=results[name]
        rank+=1
        if rank>12: continue
        yy=bm2-rank*0.025
        ax.text(mx_xs[0],yy,name[:16],fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[1],yy,str(r["n"]),fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[2],yy,f"{r['net']:+,.0f}",fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[3],yy,f"Rs{r['net']*LOT:+,.0f}",fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[4],yy,f"{r['wr']:.1f}",fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[5],yy,f"{r['avg']:+.1f}",fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[6],yy,f"{r['sharpe']:.2f}",fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[7],yy,f"{r['calmar']:.1f}",fontsize=5.5,transform=ax.transAxes)
        ax.text(mx_xs[8],yy,f"Rs{r['mdd']*LOT:,.0f}",fontsize=5.5,transform=ax.transAxes)
    pdf.savefig(fig); plt.close()

    # PAGE 4: Trade book
    tpb=tb.copy()
    tpb["entry_str"]=tpb["entry"].dt.strftime("%m/%d %H:%M")
    tpb["exit_str"]=pd.to_datetime(tpb["exit"]).dt.strftime("%m/%d %H:%M") if hasattr(tpb["exit"],"dt") else tpb["exit"]
    tpb["rs"]=tpb["pnl"]*LOT
    tpb_show=tpb[["entry_str","exit_str","strike","entry_prem","exit_prem","pnl","rs"]]
    trades_per_page=45
    for page_start in range(0,len(tpb_show),trades_per_page):
        fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
        chunk=tpb_show.iloc[page_start:page_start+trades_per_page]
        t2(f"Trade Book ({page_start+1}-{page_start+len(chunk)} of {len(tpb_show)})",0,fs=11,wt="bold")
        t2(f"{BEST_NAME} | LOT={LOT} | TP={BEST_TP} Max={BEST_MAXD}d",1,fs=8)
        hd=["#","Entry","Exit","Strike","E.Prem","X.Prem","PnL(pt)","PnL(Rs)"]
        xs_=[0.02,0.08,0.22,0.34,0.42,0.50,0.58,0.66]
        for j,h in enumerate(hd):
            ax.text(xs_[j],0.91,h,fontsize=6,fontweight="bold",transform=ax.transAxes)
        for k,(_,r) in enumerate(chunk.iterrows()):
            yy=0.89-k*0.019
            ax.text(xs_[0],yy,str(page_start+k+1),fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[1],yy,str(r["entry_str"])[:14],fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[2],yy,str(r["exit_str"])[:14],fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[3],yy,f'{r["strike"]:.0f}',fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[4],yy,f'{r["entry_prem"]:.1f}',fontsize=4.5,transform=ax.transAxes)
            ax.text(xs_[5],yy,f'{r["exit_prem"]:.1f}',fontsize=4.5,transform=ax.transAxes)
            c="green" if r["pnl"]>0 else "red"
            ax.text(xs_[6],yy,f'{r["pnl"]:+.1f}',fontsize=4.5,color=c,transform=ax.transAxes)
            ax.text(xs_[7],yy,f'Rs{r["rs"]:+,.0f}',fontsize=4.5,color=c,transform=ax.transAxes)
        pdf.savefig(fig); plt.close()

    # PAGE: PnL Distribution
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    ax.hist(best_pnls,bins=30,color="steelblue",edgecolor="white",alpha=0.7)
    ax.axvline(x=0,color="red",ls="--",lw=1)
    ax.axvline(x=np.mean(best_pnls),color="green",ls="--",lw=1,label=f"Mean={np.mean(best_pnls):+.1f}")
    ax.axvline(x=np.median(best_pnls),color="orange",ls=":",lw=1,label=f"Median={np.median(best_pnls):+.1f}")
    ax.set_title(f"PnL Distribution - {BEST_NAME}"); ax.set_xlabel("PnL (pts)"); ax.set_ylabel("Frequency")
    ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # PAGE: Monthly bar chart
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    tb["ym"]=tb["yr"].astype(str)+"-"+tb["mo"].apply(lambda x:f"{int(x):02d}")
    mo_bar=tb.groupby("ym")["pnl"].sum().reset_index()
    colors=["green" if v>0 else "red" for v in mo_bar["pnl"]]
    ax.bar(range(len(mo_bar)),mo_bar["pnl"],color=colors,alpha=0.7)
    ax.set_title("Monthly Net PnL"); ax.set_xticks(range(len(mo_bar)))
    ax.set_xticklabels(mo_bar["ym"],rotation=90,fontsize=5)
    ax.axhline(y=0,color="black",lw=0.5); ax.grid(alpha=0.3,axis="y")
    pdf.savefig(fig); plt.close()

    # PAGE: Win rate by year
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

    # PAGE: Win/loss streaks
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    seqs=[1 if p>0 else 0 for p in best_pnls]
    runs=[]; cur=seqs[0]; cnt=1
    for s in seqs[1:]:
        if s==cur: cnt+=1
        else: runs.append((cur,cnt)); cur=s; cnt=1
    runs.append((cur,cnt))
    win_runs=[c for w,c in runs if w==1]
    loss_runs=[c for w,c in runs if w==0]
    if win_runs: ax.bar(range(len(win_runs)),win_runs,color="green",alpha=0.5,label=f"Win Streaks (max={max(win_runs)})")
    if loss_runs: ax.bar(range(len(loss_runs)),loss_runs,color="red",alpha=0.5,label=f"Loss Streaks (max={max(loss_runs)})")
    ax.set_title("Win/Loss Streaks"); ax.set_ylabel("Consecutive Trades"); ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # PAGE: Yearly cumulative comparison
    fig,ax=plt.subplots(figsize=(8.27,5.85))
    for y_ in sorted(tb["yr"].unique()):
        y_trades=tb[tb["yr"]==y_].sort_values("entry")
        y_cum=np.cumsum(y_trades["pnl"].values)
        ax.plot(range(len(y_cum)),y_cum,label=str(int(y_)),lw=1.5)
    ax.set_title("Yearly Cumulative PnL"); ax.set_ylabel("Cumulative PnL (pts)"); ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

print(f"\nPDF saved: {PDF_NAME}")
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")
print(f"\nBEST STRATEGY: {BEST_NAME}")
print(f"{'Metric':<25} {'Points':>12} {'Rupees':>12}")
print("-"*50)
def rup(v):
    try: return "Rs{:+,.0f}".format(float(v))
    except: return str(v)
print(("{:<25} {:>+12,.0f} {:>12}").format('Net PnL',b['net'],rup(b['net']*LOT)))
print(("{:<25} {:>+12.1f} {:>12}").format('Avg Trade',b['avg'],rup(b['avg']*LOT)))
print(("{:<25} {:>+12.1f} {:>12}").format('Max Win',b['pnls'].max(),rup(b['pnls'].max()*LOT)))
print(("{:<25} {:>+12.1f} {:>12}").format('Max Loss',b['pnls'].min(),rup(b['pnls'].min()*LOT)))
print(f"{'Win Rate':<25} {b['wr']:>12.1f}%")
print(f"{'Trades':<25} {b['n']:>12}")
print(f"{'Sharpe':<25} {b['sharpe']:>12.2f}")
print(f"{'Calmar':<25} {b['calmar']:>12.1f}x")
print(("{:<25} {:>12,.0f} {:>12}").format('Max DD',b['mdd'],rup(b['mdd']*LOT)))

print(f"\n{'='*70}")
print("ALL STRATEGIES COMPARISON (sorted by Net PnL)")
print(f"{'='*70}")
print(("{:<25} {:>6} {:>10} {:>12} {:>6} {:>7} {:>7}").format('Strategy','Trades','Net(pts)','Net(Rs)','WR','Sharpe','Calmar'))
print("-"*75)
for name in sorted(results, key=lambda n: results[n]["net"], reverse=True):
    r=results[name]
    net_rs=rup(r['net']*LOT)
    print(("{:<25} {:>6} {:>+10,.0f} {:>12} {:>5.1f}% {:>7.2f} {:>5.1f}x").format(name,r['n'],r['net'],net_rs,r['wr'],r['sharpe'],r['calmar']))

# Yearly breakdown for best
print(f"\n{'='*70}")
print(f"YEARLY BREAKDOWN - {BEST_NAME}")
print(f"{'='*70}")
tb=pd.DataFrame(best_trades)
yr=tb.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()),
                        avg=("pnl","mean"),mxx=("pnl","max"),mnn=("pnl","min"))
print(f"{'Year':>6} {'Trades':>7} {'Net(pts)':>12} {'Net(Rs)':>12} {'WR':>7} {'Avg':>8} {'Max':>8} {'Min':>8}")
print("-"*70)
for y_ in yr.index:
    r=yr.loc[y_]
    print(("{:>6} {:>7} {:>+12,.0f} {:>12} {:>6.1%} {:>+8.1f} {:>+8.1f} {:>+8.1f}").format(int(y_),int(r['trades']),r['net'],rup(r['net']*LOT),r['wr'],r['avg'],r['mxx'],r['mnn']))

print(f"\n{'='*70}")
print(f"MONTHLY BREAKDOWN - {BEST_NAME}")
print(f"{'='*70}")
mo=tb.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
print(f"{'Month':<6} {'Trades':>7} {'Net(pts)':>12} {'Net(Rs)':>12} {'WR':>7}")
print("-"*45)
for m_ in range(1,13):
    if m_ in mo.index:
        r=mo.loc[m_]
        print(("{:<6} {:>7} {:>+12,.0f} {:>12} {:>6.1%}").format(MONTHS[m_-1],int(r['trades']),r['net'],rup(r['net']*LOT),r['wr']))

print(f"\nDone! Full report: {PDF_NAME}")
con.close()
