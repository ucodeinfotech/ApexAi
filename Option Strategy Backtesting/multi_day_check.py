"""Multi-day hold strategies with corrected engine. Compare vs same-day."""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT=50; plt.rcParams.update({"font.size":7,"axes.titlesize":10,"axes.labelsize":8,"figure.dpi":120})
DB_PATH=r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# SPOT ENGINE
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

# OPTION DATA
con=duckdb.connect(DB_PATH)
df_atm=con.execute("""SELECT timestamp,close,strike FROM options_data_clean WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 ORDER BY timestamp""").fetchdf()
con.close()
df_atm["timestamp"]=pd.to_datetime(df_atm["timestamp"],utc=False)
atm_ts=df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl=df_atm["close"].values.astype(float);atm_st=df_atm["strike"].values.astype(float)
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
df_all=con2.execute(f"""SELECT timestamp,close,strike FROM options_data_clean WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({stk_where}) ORDER BY strike,timestamp""").fetchdf()
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

# EXIT: Multi-day hold
def exit_md(stk_data,s_idx,tp,maxd):
    end_ns=stk_data["ts"][s_idx]+np.timedelta64(int(maxd*86400*1e6),"us");ep=stk_data["cl"][s_idx]
    for i in range(s_idx+1,min(s_idx+3000,len(stk_data["cl"]))):
        if stk_data["ts"][i]>end_ns: return stk_data["cl"][i],stk_data["ts"][i]
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return None,None

# EXIT: Same-day (for comparison)
def exit_eod(stk_data,s_idx,tp):
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m");last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    for i in range(s_idx+1,last_idx+1):
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

# EXIT: Time cutoff (15:00)
def exit_cut(stk_data,s_idx,tp,cut_h=15,cut_m=0):
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    cut_ns=entry_date+np.timedelta64(cut_h*60+cut_m,"m");cut_idx=np.searchsorted(stk_data["ts"],cut_ns)
    if cut_idx>=len(stk_data["ts"]): cut_idx=len(stk_data["ts"])-1
    if cut_idx<=s_idx: return None,None
    for i in range(s_idx+1,cut_idx+1):
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][cut_idx],stk_data["ts"][cut_idx]

def stats(pnls):
    n=len(pnls);net=pnls.sum();wr=(pnls>0).mean()*100;avg=pnls.mean();std=pnls.std() if n>1 else 0
    sharpe=avg/std*np.sqrt(252) if std>0 else 0;cum=np.cumsum(pnls);mx=np.maximum.accumulate(cum);mdd=(mx-cum).max() if len(cum)>0 else 0
    calmar=net/mdd if mdd>0 else 0;pf=pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999
    return {"n":n,"net":net,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pf":pf}

all_results={}

# ====== MULTI-DAY HOLD (no day filter, no entry time filter) ======
print("="*70)
print("MULTI-DAY HOLD STRATEGIES (corrected engine)")
print("="*70)
md_sweep=[]
for tp in [10,15,20,25,30,40,50,75,100,150,200,300]:
    for maxd in [3,5,7,10,14]:
        pnls=[]
        for i,info in enumerate(trade_infos):
            if info is None: continue
            r=exit_md(info["stk_data"],info["s_idx"],tp,maxd)
            if r[0] is None: continue
            pnls.append(round(r[0]-info["ep"],1))
        if len(pnls)>=3:
            all_results[f"MD_TP{tp}_D{maxd}"]=stats(np.array(pnls))
            s=all_results[f"MD_TP{tp}_D{maxd}"]
            print(f"  MD_TP{tp}_D{maxd}: n={s['n']:3d} net={s['net']:>+7,.0f} WR={s['wr']:5.1f}% Avg={s['avg']:+.1f} Sharp={s['sharpe']:.2f}")

# ====== MULTI-DAY WITH PREMIUM FILTER ======
print("\n"+"="*70)
print("MULTI-DAY + Premium filter")
print("="*70)
for tp,maxd in [(30,7),(50,7),(75,10),(100,10),(200,10),(200,14)]:
    for pmax in [80,100,120,130,150]:
        pnls=[]
        for i,info in enumerate(trade_infos):
            if info is None: continue
            if info["ep"]>pmax: continue
            r=exit_md(info["stk_data"],info["s_idx"],tp,maxd)
            if r[0] is None: continue
            pnls.append(round(r[0]-info["ep"],1))
        if len(pnls)>=3:
            all_results[f"MD_TP{tp}_D{maxd}_P{pmax}"]=stats(np.array(pnls))
            s=all_results[f"MD_TP{tp}_D{maxd}_P{pmax}"]
            print(f"  MD_TP{tp}_D{maxd}_P{pmax}: n={s['n']:3d} net={s['net']:>+7,.0f} WR={s['wr']:5.1f}% Avg={s['avg']:+.1f}")

# ====== TOP MULTI-DAY ======
md_results={k:v for k,v in all_results.items() if k.startswith("MD_")}
top_md=sorted(md_results.items(),key=lambda x:x[1]["net"],reverse=True)[:30]
print(f"\n{'='*70}")
print(f"TOP 30 MULTI-DAY STRATEGIES")
print(f"{'='*70}")
print(f"{'Strategy':<28} {'N':>4} {'Net(pt)':>9} {'Net(Rs)':>11} {'WR':>5} {'Avg':>6} {'Sharp':>6} {'Cal':>5} {'PF':>5}")
print("-"*85)
for name,r in top_md:
    rs="Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<28} {r['n']:>4} {r['net']:>+8,.0f} {rs:>11} {r['wr']:>4.1f}% {r['avg']:>+6.1f} {r['sharpe']:>5.2f} {r['calmar']:>4.1f}x {r['pf']:>4.2f}x")

# ====== ADD SAME-DAY FOR COMPARISON ======
print("\n"+"="*70)
print("BEST SAME-DAY STRATEGIES (for comparison)")
print("="*70)
for tp in [28,30,35,40,50]:
    for desc,fn,params in [("EOD",exit_eod,(tp,)),("Cut15",exit_cut,(tp,15,0))]:
        pnls=[]
        for i,info in enumerate(trade_infos):
            if info is None: continue
            if trades_pre.iloc[i]["weekday"] in [4]: continue  # skip Fri
            if trades_pre.iloc[i]["entry_time"].hour>=12: continue
            r=fn(info["stk_data"],info["s_idx"],*params)
            if r[0] is None: continue
            pnls.append(round(r[0]-info["ep"],1))
        if len(pnls)>=3:
            all_results[f"SD_TP{tp}_{desc}_MTE12"]=stats(np.array(pnls))
            s=all_results[f"SD_TP{tp}_{desc}_MTE12"]
            print(f"  SD_TP{tp}_{desc}_MTE12: n={s['n']:3d} net={s['net']:>+6,.0f} WR={s['wr']:5.1f}% Avg={s['avg']:+.1f}")

# ====== GRAND TOP 50 ======
print(f"\n{'='*70}")
print(f"GRAND TOP 50 (Multi-day + Same-day)")
print(f"{'='*70}")
print(f"{'Strategy':<28} {'N':>4} {'Net(pt)':>9} {'Net(Rs)':>11} {'WR':>5} {'Avg':>6} {'Sharp':>6} {'Cal':>5} {'PF':>5}")
print("-"*85)
top50=sorted(all_results.items(),key=lambda x:x[1]["net"],reverse=True)[:50]
for name,r in top50:
    rs="Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<28} {r['n']:>4} {r['net']:>+8,.0f} {rs:>11} {r['wr']:>4.1f}% {r['avg']:>+6.1f} {r['sharpe']:>5.2f} {r['calmar']:>4.1f}x {r['pf']:>4.2f}x")
