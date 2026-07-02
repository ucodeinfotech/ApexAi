"""Refine same-day further: TP/Mon/Entry combos, expiry filter, SL, exit time"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT = 50
plt.rcParams.update({"font.size":8,"axes.titlesize":11,"axes.labelsize":9})
DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

# Spot engine (abbreviated)
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

# Option data
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
    ts=grp["timestamp"].values.astype("datetime64[us]");cl=grp["close"].values.astype(float)
    strike_cache[int(stk)]={"ts":ts,"cl":cl}

trade_infos=[]
for idx,ed in enumerate(trades_pre["ed_naive"]):
    ts64=np.datetime64(ed,"us");i=np.searchsorted(atm_ts,ts64)
    if i>=len(atm_ts): si=len(atm_ts)-1
    elif i==0: si=0
    else: si=i if atm_ts[i]==ts64 else i-1
    st=int(atm_st[si]);sd=strike_cache.get(st)
    if sd is None: trade_infos.append(None); continue
    s_idx=np.searchsorted(sd["ts"],atm_ts[si])
    if s_idx>=len(sd["cl"]): trade_infos.append(None); continue
    trade_infos.append({"strike":st,"ep":float(sd["cl"][s_idx]),"s_idx":s_idx,"stk_data":sd})
    # Store expiry week for later filtering
    trades_pre.at[idx,"exp_w"] = trades_pre.iloc[idx]["ed_naive"] + pd.Timedelta(days=(3-trades_pre.iloc[idx]["weekday"])%7 or 7)

# Exit functions
def exit_tp_eod(stk_data,s_idx,tp):
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m");last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    for i in range(s_idx+1,last_idx+1):
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

def exit_tp_sl(stk_data,s_idx,tp,sl):
    ep=stk_data["cl"][s_idx];entry_ns=stk_data["ts"][s_idx];entry_date=entry_ns.astype("datetime64[D]")
    next_day=entry_date+np.timedelta64(24*60,"m");last_idx=np.searchsorted(stk_data["ts"],next_day)-1
    if last_idx<0 or last_idx<=s_idx: return None,None
    for i in range(s_idx+1,last_idx+1):
        pnl=stk_data["cl"][i]-ep
        if pnl>=tp: return stk_data["cl"][i],stk_data["ts"][i]
        if sl is not None and pnl<=-abs(sl): return stk_data["cl"][i],stk_data["ts"][i]
    return stk_data["cl"][last_idx],stk_data["ts"][last_idx]

def compute_stats(pnls):
    n=len(pnls);net=pnls.sum();wr=(pnls>0).mean()*100
    avg=pnls.mean();std=pnls.std() if n>1 else 0;sharpe=avg/std*np.sqrt(252) if std>0 else 0
    cum=np.cumsum(pnls);mx=np.maximum.accumulate(cum);mdd=(mx-cum).max() if len(cum)>0 else 0
    calmar=net/mdd if mdd>0 else 0;pf=pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()) if (pnls<0).sum()>0 else 999
    return {"n":n,"net":net,"wr":wr,"avg":avg,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"pf":pf,"pnls":pnls}

all_results = {}

# 1) ALL TP targets with Mon+Entry<12 combined
print("="*70)
print("1) TP sweep: Monday + Entry<12")
print("="*70)
for tp in [5,8,10,12,15,18,20,25,30,35,40,45,50,60,75]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["weekday"]!=0: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls);n=len(pnls)
    if n>=3:
        all_results[f"TP{tp}_MonE<12"]=compute_stats(pnls)
        r=all_results[f"TP{tp}_MonE<12"]
        print(f"  TP{tp}_MonE<12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharpe={r['sharpe']:.2f} Cal={r['calmar']:.1f}x")

# 2) ALL TP targets with Entry<12 only (no day filter)
print("\n"+"="*70)
print("2) TP sweep: Entry<12 (all days)")
print("="*70)
for tp in [5,8,10,12,15,18,20,25,30,35,40,45,50,60,75]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls);n=len(pnls)
    if n>=3:
        all_results[f"TP{tp}_E<12"]=compute_stats(pnls)
        r=all_results[f"TP{tp}_E<12"]
        print(f"  TP{tp}_E<12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharpe={r['sharpe']:.2f} Cal={r['calmar']:.1f}x")

# 3) TP + SL combinations (Entry<12, all days)
print("\n"+"="*70)
print("3) TP+SL sweep: Entry<12")
print("="*70)
for tp,sl in [(15,5),(20,10),(25,10),(30,15),(35,15),(40,20),(50,25),(75,30)]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_sl(info["stk_data"],info["s_idx"],tp,sl)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls);n=len(pnls)
    if n>=3:
        all_results[f"TP{tp}_SL{sl}_E<12"]=compute_stats(pnls)
        r=all_results[f"TP{tp}_SL{sl}_E<12"]
        print(f"  TP{tp}_SL{sl}_E<12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharpe={r['sharpe']:.2f}")

# 4) Expiry proximity filter: skip last 1-2 days before weekly expiry
print("\n"+"="*70)
print("4) Expiry proximity filter (skip last N days)")
print("="*70)
def days_to_expiry(entry_dt, exp_w):
    return (exp_w.date()-entry_dt.date()).days if hasattr(exp_w,'date') else 999
for skip_days in [1,2,3]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        dte=days_to_expiry(trades_pre.iloc[i]["ed_naive"],trades_pre.iloc[i]["exp_w"])
        if dte<skip_days: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],35)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls);n=len(pnls)
    if n>=3:
        all_results[f"TP35_E<12_Skip{skip_days}d"]=compute_stats(pnls)
        r=all_results[f"TP35_E<12_Skip{skip_days}d"]
        print(f"  Skip {skip_days}d to expiry: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 5) Premium + entry<12 combo for all TP targets
print("\n"+"="*70)
print("5) TP + Premium + Entry<12")
print("="*70)
for tp in [20,25,30,35,40,50]:
    for pm in [80,100,120]:
        pnls=[]
        for i,info in enumerate(trade_infos):
            if info is None: continue
            if trades_pre.iloc[i]["entry_time"].hour>=12: continue
            if info["ep"]>pm: continue
            r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
            if r[0] is None: continue
            pnls.append(round(r[0]-info["ep"],1))
        pnls=np.array(pnls);n=len(pnls)
        if n>=3:
            all_results[f"TP{tp}_P{pm}_E<12"]=compute_stats(pnls)
            r=all_results[f"TP{tp}_P{pm}_E<12"]
            print(f"  TP{tp}_P{pm}_E<12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 6) PM filter only (no entry time)
print("\n"+"="*70)
print("6) TP + Premium (no entry filter)")
print("="*70)
for tp in [20,30,35,50]:
    for pm in [80,100,120]:
        pnls=[]
        for i,info in enumerate(trade_infos):
            if info is None: continue
            if info["ep"]>pm: continue
            r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
            if r[0] is None: continue
            pnls.append(round(r[0]-info["ep"],1))
        pnls=np.array(pnls);n=len(pnls)
        if n>=3:
            all_results[f"TP{tp}_P{pm}"]=compute_stats(pnls)
            r=all_results[f"TP{tp}_P{pm}"]
            print(f"  TP{tp}_P{pm}: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# 7) Tuesday + Wednesday + Thursday filters
print("\n"+"="*70)
print("7) Day filters (TP35, Entry<12)")
print("="*70)
for d,dn in [(0,"Mon"),(1,"Tue"),(2,"Wed"),(3,"Thu"),(4,"Fri")]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["weekday"]!=d: continue
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],35)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls);n=len(pnls)
    if n>=3:
        all_results[f"TP35_{dn}_E<12"]=compute_stats(pnls)
        r=all_results[f"TP35_{dn}_E<12"]
        print(f"  TP35_{dn}_E<12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharpe={r['sharpe']:.2f}")

# 8) All days EXCEPT Friday (Mon-Thu only)
print("\n"+"="*70)
print("8) Mon-Thu only (skip Fri)")
print("="*70)
for tp in [25,30,35,40,50]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        if trades_pre.iloc[i]["weekday"]==4: continue  # skip Fri
        if trades_pre.iloc[i]["entry_time"].hour>=12: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls)
    if len(pnls)>=3:
        all_results[f"TP{tp}_MonThu_E<12"]=compute_stats(pnls)
        r=all_results[f"TP{tp}_MonThu_E<12"]
        print(f"  TP{tp}_MonThu_E<12: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f} Sharpe={r['sharpe']:.2f}")

# 9) Entry time = 13 (afternoon) only
print("\n"+"="*70)
print("9) Afternoon entry (13:00-14:00) only")
print("="*70)
for tp in [20,30,35,50]:
    pnls=[]
    for i,info in enumerate(trade_infos):
        if info is None: continue
        et=trades_pre.iloc[i]["entry_time"]
        if et.hour<13 or et.hour>=14: continue
        r=exit_tp_eod(info["stk_data"],info["s_idx"],tp)
        if r[0] is None: continue
        pnls.append(round(r[0]-info["ep"],1))
    pnls=np.array(pnls)
    if len(pnls)>=3:
        all_results[f"TP{tp}_Entry13"]=compute_stats(pnls)
        r=all_results[f"TP{tp}_Entry13"]
        print(f"  TP{tp}_Entry13: n={r['n']:3d} net={r['net']:>+6,.0f} WR={r['wr']:5.1f}% Avg={r['avg']:+.1f}")

# Print TOP 40
print(f"\n{'='*70}")
print(f"TOP 40 SAME-DAY REFINEMENTS (out of {len(all_results)})")
print(f"{'='*70}")
print(f"{'Strategy':<28} {'N':>4} {'Net(pt)':>9} {'Net(Rs)':>11} {'WR':>5} {'Avg':>6} {'Sharp':>6} {'Cal':>5} {'PF':>5}")
print("-"*85)
top40=sorted(all_results.items(),key=lambda x:x[1]["net"],reverse=True)[:40]
for name,r in top40:
    rs="Rs{:+,.0f}".format(r["net"]*LOT)
    print(f"{name:<28} {r['n']:>4} {r['net']:>+8,.0f} {rs:>11} {r['wr']:>4.1f}% {r['avg']:>+6.1f} {r['sharpe']:>5.2f} {r['calmar']:>4.1f}x {r['pf']:>4.2f}x")
