"""
Fixed combined option improvements: same-strike tracking for all variants
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, calendar, re
from datetime import timedelta
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
from option_fixed_engine import load_fixed_data, get_entry_strike_idx, exit_tp_maxd, exit_maxd, exit_tp_sl
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

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
from datetime import timedelta; import calendar
exp_w_l=[];
for _,r in trades.iterrows():
    d_=r["ed_naive"].date()
    da=(3-d_.weekday())%7; da=da if da>0 else 7
    exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
trades["exp_w"]=exp_w_l
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Trades: {len(trades_pre)}")

print("Loading fixed data...")
atm_ts_arr, atm_cl_arr, atm_st_arr, strike_cache, lookup = load_fixed_data(trades_pre)

# Build trade info list
trade_infos=[]
for i in range(len(trades_pre)):
    ed=trades_pre.iloc[i]["ed_naive"]
    info=get_entry_strike_idx(ed, lookup, atm_ts_arr, strike_cache)
    trade_infos.append(info if info else None)
print(f"Valid: {sum(1 for t in trade_infos if t)}/{len(trade_infos)}")

def test_combo(tp, max_d, prem_max=None, day_filter=None, skip=0, return_idx=False):
    pnls=[]; indices=[]; n_skip=0
    for idx,info in enumerate(trade_infos):
        if info is None: continue
        if prem_max is not None and info["ep"]>prem_max: continue
        if day_filter is not None and trades_pre.iloc[idx]["weekday"] not in day_filter: continue
        if n_skip>0: n_skip-=1; continue
        if tp is None:
            r=exit_maxd(info["stk_data"], info["s_idx"], max_d)
        else:
            r=exit_tp_maxd(info["stk_data"], info["s_idx"], tp, max_d)
        if r is None: continue
        xp,_=r
        if xp is None: continue
        pnl=round(xp-info["ep"],1)
        pnls.append(pnl); indices.append(idx)
        if skip>0 and pnl<0: n_skip=skip
    return (pnls, indices) if return_idx else pnls

def add_result(label, pnls):
    p=pd.Series(pnls); n=len(p)
    if n<3: return
    net=p.sum(); wr=(p>0).mean()
    aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0
    wl=aw/abs(al) if al!=0 else 999; pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999; sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    avg_pnl=net/n
    all_rows.append({"Variant":label,"Trades":n,"Net":net,"WR":wr,"W/L":wl,"PF":pf,
        "MDD":mdd,"Calmar":calmar,"Sharpe":sharpe,"Avg":avg_pnl,
        "NetStr":f"{net:+,.0f}","WRStr":f"{wr:.0%}","AvgStr":f"{avg_pnl:+.1f}"})
    print(f"  {label:<35} {n:>4d} {net:>+8,.0f} {wr:>5.0%} {wl:>4.1f}x {calmar:>5.1f}x {avg_pnl:>+5.1f}")

all_rows=[]

# 1. TP30 ± premium filters
print("\n=== TP30_Max7d + Premium Filters ===")
tp,tmax=30,7
add_result("TP30_Max7d (base)", test_combo(tp,tmax))
for pm in [50,60,70,80,90,100,110,120,130,140,150,200]:
    pnls=test_combo(tp,tmax,prem_max=pm)
    if len(pnls)>=5: add_result(f"+Prem<{pm}", pnls)

# 2. TP30 + Entry Day Filters
print("\n=== TP30_Max7d + Day Filters ===")
for dname,dfilt in [("TueWedThu",[1,2,3]),("MonTueWed",[0,1,2]),("WedThu",[2,3]),
    ("Thu",[3]),("TueWed",[1,2]),("MonTue",[0,1])]:
    pnls=test_combo(tp,tmax,day_filter=dfilt)
    if len(pnls)>=5: add_result(f"+Day{dname}", pnls)

# 3. TP30 + Skip
print("\n=== TP30_Max7d + Skip After Loss ===")
for sk in [1,2,3]:
    pnls=test_combo(tp,tmax,skip=sk)
    if len(pnls)>=5: add_result(f"+Skip{sk}", pnls)

# 4. TP30 + Premium + Day
print("\n=== TP30 + Premium + Day (stacked) ===")
for pm in [80,100,120,150]:
    for dname,dfilt in [("TueWedThu",[1,2,3]),("WedThu",[2,3])]:
        pnls=test_combo(tp,tmax,prem_max=pm,day_filter=dfilt)
        if len(pnls)>=5: add_result(f"+Prem<{pm}+Day{dname}", pnls)

# 5. TP sweep with best filters
print("\n=== TP sweep with Prem<120 ===")
for tp_val in [20,30,40,50,60,75,100,150,200]:
    pnls=test_combo(tp_val,tmax,prem_max=120)
    if len(pnls)>=5: add_result(f"TP{tp_val}_Max7d+Prem<120", pnls)

# 6. Max days sweep
print("\n=== TP30 + Prem<120 + MaxDays sweep ===")
for md in [3,4,5,6,7,8,9,10,14]:
    pnls=test_combo(30,md,prem_max=120)
    if len(pnls)>=5: add_result(f"TP30_Max{md}d+Prem<120", pnls)

# 7. Hold4D + Premium filters
print("\n=== Hold4D + Premium Filters ===")
add_result("Hold4D (base)", test_combo(None,4))
for pm in [80,100,120,150]:
    pnls=test_combo(None,4,prem_max=pm)
    if len(pnls)>=5: add_result(f"Hold4D+Prem<{pm}", pnls)

# 8. TP30 + Prem<120 + Day + Skip
print("\n=== TP30 + Prem<120 + Day + Skip ===")
for dname,dfilt in [("",None),("+TueWedThu",[1,2,3]),("+Thu",[3])]:
    for sk in [0,1]:
        label=f"TP30_Max7d+Prem<120{dname}"
        if sk>0: label+=f"+Skip{sk}"
        pnls=test_combo(30,7,prem_max=120,day_filter=dfilt,skip=sk)
        if len(pnls)>=5: add_result(label, pnls)

# 9. RANKING
rr=pd.DataFrame(all_rows).sort_values("Net",ascending=False).reset_index(drop=True)
print("\n"+"="*140)
print("FIXED COMBINED VARIANT RANKING (Same-Strike)")
print("="*140)
print(f"{'Rank':>4} {'Variant':<45} {'Trades':>6} {'Net':>10} {'WR':>5} {'W/L':>5} {'PF':>5} {'MDD':>8} {'Calmar':>6} {'Sharpe':>6} {'Avg':>7}")
print("-"*105)
for i,(_,r) in enumerate(rr.iterrows()):
    hl="*" if i==0 else " "
    print(f"{i+1:>3}{hl} {r['Variant']:<45} {r['Trades']:>6} {r['NetStr']:>10} {r['WRStr']:>5} {r['W/L']:>4.1f}x {r['PF']:>4.1f}x {r['MDD']:>8,.0f} {r['Calmar']:>5.1f}x {r['Sharpe']:>5.2f} {r['AvgStr']:>7}")

best=rr.iloc[0]
print(f"\n{'='*100}")
print(f"BEST: {best['Variant']} Net={best['NetStr']} WR={best['WRStr']} Avg={best['AvgStr']}")
print(f"{'='*100}")

print("\nDone!")
