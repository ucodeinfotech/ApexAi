"""SL & optimization refinement: tight SL, BE trail, %SL, time SL, TP/SL ratio"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

# EXIT FUNCTIONS
def exit_tp_sl(stk_data,s_idx,tp,sl):
    """TP or SL, whichever hits first. Exit at EOD if neither."""
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m");last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    for i in range(s_idx+1,last_idx+1):
        pnl=stk_data["cl"][i]-ep
        if pnl>=tp: return stk_data["cl"][i],stk_data["ts"][i]
        if sl is not None and pnl<=-abs(sl): return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

def exit_tp_be(stk_data,s_idx,tp,be_at):
    """TP exit. Once profit >= be_at, SL moves to breakeven (ep)."""
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m");last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    be_active=False
    for i in range(s_idx+1,last_idx+1):
        pnl=stk_data["cl"][i]-ep
        if not be_active and pnl>=be_at: be_active=True
        if be_active and pnl<=0: return stk_data["cl"][i],stk_data["ts"][i]  # stopped at BE
        if pnl>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

def exit_tp_time(stk_data,s_idx,tp,cut_hour,cut_min=0):
    """TP exit with hard time cutoff (e.g., exit at 14:00 if TP not hit)."""
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    cut_ns=entry_date+np.timedelta64(cut_hour*60+cut_min,"m")
    cut_idx=np.searchsorted(stk_data["ts"],cut_ns)
    if cut_idx>=len(stk_data["ts"]): cut_idx=len(stk_data["ts"])-1
    if cut_idx<=s_idx: return None,None
    for i in range(s_idx+1,cut_idx+1):
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][cut_idx],stk_data["ts"][cut_idx]

def exit_tp_pct_sl(stk_data,s_idx,tp_pct,sl_pct):
    """TP/SL as % of entry premium."""
    ep=stk_data["cl"][s_idx];tp=ep*tp_pct;sl=-ep*sl_pct
    return exit_tp_sl(stk_data,s_idx,tp,sl)

def exit_tp_sl_eod_fixed(stk_data,s_idx,tp,sl,exit_hour=15,exit_min=25):
    """TP/SL with fixed EOD time."""
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    eod_ns=entry_date+np.timedelta64(exit_hour*60+exit_min,"m")
    eod_idx=np.searchsorted(stk_data["ts"],eod_ns)
    if eod_idx>=len(stk_data["ts"]): eod_idx=len(stk_data["ts"])-1
    if eod_idx<=s_idx: return None,None
    for i in range(s_idx+1,eod_idx+1):
        pnl=stk_data["cl"][i]-ep
        if pnl>=tp: return stk_data["cl"][i],stk_data["ts"][i]
        if sl is not None and pnl<=-abs(sl): return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][eod_idx],stk_data["ts"][eod_idx]

def compute_stats(pnls):
    n=len(pnls);net=pnls.sum();wr=(pnls>0).mean()*100;avg=pnls.mean()
    std=pnls.std() if n>1 else 0;sharpe=avg/std*np.sqrt(252) if std>0 else 0
    cum=np.cumsum(pnls);mx=np.maximum.accumulate(cum);mdd=(mx-cum).max() if len(cum)>0 else 0
    calmar=net/mdd if mdd>0 else 0;pf=pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999
    return {"n":n,"net":net,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pf":pf}

def run_tp_sl(tp,sl,pmax=None,days=None,eh_max=None):
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if pmax is not None and info["ep"]>pmax: continue
        if days is not None and trades_pre.iloc[i]["weekday"] not in days: continue
        if eh_max is not None and trades_pre.iloc[i]["entry_time"].hour>=eh_max: continue
        r=exit_tp_sl(info["stk_data"],info["s_idx"],tp,sl)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    return compute_stats(np.array(pnls)) if len(pnls)>=3 else None

def run_fn(fn,params,pmax=None,days=None,eh_max=None):
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if pmax is not None and info["ep"]>pmax: continue
        if days is not None and trades_pre.iloc[i]["weekday"] not in days: continue
        if eh_max is not None and trades_pre.iloc[i]["entry_time"].hour>=eh_max: continue
        r=fn(info["stk_data"],info["s_idx"],*params)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    return compute_stats(np.array(pnls)) if len(pnls)>=3 else None

all_results={}

# 1) Tight TP + tight SL combos (Mon-Thu, Entry<12)
print("="*70)
print("1) TP/SL combos: Mon-Thu Entry<12")
print("="*70)
for tp,sl in [(8,3),(10,3),(10,5),(12,4),(12,5),(15,5),(15,8),(18,5),(18,8),(20,5),(20,8),(20,10),
               (25,8),(25,10),(25,12),(30,8),(30,10),(30,12),(30,15),(35,10),(35,12),(35,15),
               (40,12),(40,15),(40,20),(45,15),(45,20),(50,15),(50,20),(50,25),(60,20),(75,25)]:
    r=run_tp_sl(tp,sl,days=[0,1,2,3],eh_max=12)
    if r:
        all_results[f"TP{tp}_SL{sl}_MTE12"]=r
        print(f"  TP{tp}_SL{sl}_MTE12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f} Cal={r['calmar']:.1f}x PF={r['pf']:.2f}x")

# 2) Breakeven trailing (Mon-Thu, Entry<12)
print("\n"+"="*70)
print("2) Breakeven trail: Mon-Thu Entry<12")
print("="*70)
for tp,be in [(20,5),(25,5),(30,8),(30,10),(35,8),(35,10),(35,12),(40,10),(40,12),(45,10),(50,10),(50,15),(60,15),(75,20)]:
    r=run_fn(exit_tp_be,(tp,be),days=[0,1,2,3],eh_max=12)
    if r:
        all_results[f"TP{tp}_BE{be}_MTE12"]=r
        print(f"  TP{tp}_BE{be}_MTE12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f}")

# 3) %-based SL (Mon-Thu, Entry<12)
print("\n"+"="*70)
print("3) % SL of premium: Mon-Thu Entry<12")
print("="*70)
for tp_pct in [0.3,0.4,0.5,0.6,0.7,0.8,1.0]:
    for sl_pct in [0.2,0.25,0.3,0.4,0.5]:
        r=run_fn(exit_tp_pct_sl,(tp_pct,sl_pct),days=[0,1,2,3],eh_max=12)
        if r:
            tp_str=f"TP{tp_pct*100:.0f}p"; sl_str=f"SL{sl_pct*100:.0f}p"
            all_results[f"{tp_str}_{sl_str}_MTE12"]=r
            print(f"  {tp_str}_{sl_str}_MTE12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 4) Time cutoff exit (Mon-Thu, Entry<12)
print("\n"+"="*70)
print("4) Time cutoff exit (no EOD): Mon-Thu Entry<12")
print("="*70)
for tp in [20,25,30,35,40,50]:
    for h,m in [(14,0),(14,15),(14,30),(14,45),(15,0)]:
        r=run_fn(exit_tp_time,(tp,h,m),days=[0,1,2,3],eh_max=12)
        if r:
            all_results[f"TP{tp}_Cut{h:02d}{m:02d}_MTE12"]=r
            print(f"  TP{tp}_Cut{h:02d}{m:02d}_MTE12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 5) TP/SL with fixed 15:25 exit (Mon-Thu, Entry<12)
print("\n"+"="*70)
print("5) TP/SL fixed 15:25 exit: Mon-Thu Entry<12")
print("="*70)
for tp,sl in [(20,8),(20,10),(25,10),(30,10),(30,12),(35,12),(35,15),(40,15),(50,20)]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["weekday"] not in [0,1,2,3]: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_sl_eod_fixed(info["stk_data"],info["s_idx"],tp,sl,15,25)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    if len(pnls)>=3:
        all_results[f"TP{tp}_SL{sl}_1525_MTE12"]=compute_stats(np.array(pnls))
        r=all_results[f"TP{tp}_SL{sl}_1525_MTE12"]
        print(f"  TP{tp}_SL{sl}_1525_MTE12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 6) Tight TP only (no SL) with Mon-Thu Entry<12
print("\n"+"="*70)
print("6) TP-only (no SL): Mon-Thu Entry<12")
print("="*70)
for tp in [8,10,12,15,18,20,22,25,28,30,32,35,38,40,45,50,55,60]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["weekday"] not in [0,1,2,3]: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_sl(info["stk_data"],info["s_idx"],tp,None)  # No SL
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    if len(pnls)>=3:
        all_results[f"TP{tp}_MTE12"]=compute_stats(np.array(pnls))
        r=all_results[f"TP{tp}_MTE12"]
        print(f"  TP{tp}_MTE12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharp={r['sharpe']:.2f}")

# 7) All days (no day filter) TP/SL
print("\n"+"="*70)
print("7) TP/SL all days Entry<12")
print("="*70)
for tp,sl in [(15,5),(20,8),(20,10),(25,10),(30,10),(30,12),(30,15),(35,12),(35,15),(40,15),(40,20),(50,20),(50,25)]:
    r=run_tp_sl(tp,sl,eh_max=12)
    if r:
        all_results[f"TP{tp}_SL{sl}_E12"]=r
        print(f"  TP{tp}_SL{sl}_E12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 8) TP with breakeven + all days
print("\n"+"="*70)
print("8) Breakeven trail: all days Entry<12")
print("="*70)
for tp,be in [(20,5),(25,5),(30,8),(30,10),(35,10),(40,10),(45,10),(50,15)]:
    r=run_fn(exit_tp_be,(tp,be),eh_max=12)
    if r:
        all_results[f"TP{tp}_BE{be}_E12"]=r
        print(f"  TP{tp}_BE{be}_E12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# TOP 50
print(f"\n{'='*70}")
print(f"TOP 50 (of {len(all_results)})")
print(f"{'='*70}")
print(f"{'Strategy':<28} {'N':>4} {'Net(pt)':>9} {'Net(Rs)':>11} {'WR':>5} {'Avg':>6} {'Sharp':>6} {'Cal':>5} {'PF':>5}")
print("-"*85)
top50=sorted(all_results.items(),key=lambda x:x[1]["net"],reverse=True)[:50]
for name,r in top50:
    rs="Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<28} {r['n']:>4} {r['net']:>+8,.0f} {rs:>11} {r['wr']:>4.1f}% {r['avg']:>+6.1f} {r['sharpe']:>5.2f} {r['calmar']:>4.1f}x {r['pf']:>4.2f}x")
