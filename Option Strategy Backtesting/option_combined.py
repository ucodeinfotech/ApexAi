"""
Combined option improvements: TP30 + filters stacked
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, bisect
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# === Build trades ===
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
from datetime import timedelta
import calendar
exp_w_l=[]; exp_m_l=[]
for _,r in trades.iterrows():
    d=r["ed_naive"].date()
    da=(3-d.weekday())%7; da=da if da>0 else 7
    exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
    _,ld=calendar.monthrange(d.year,d.month); dd=d.replace(day=ld)
    while dd.weekday()!=3: dd-=timedelta(days=1)
    da=(dd-d).days; da=da if da>0 else 28
    exp_m_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
trades["exp_w"]=exp_w_l; trades["exp_m"]=exp_m_l
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Trades: {len(trades_pre)}")

# === Load WEEKLY option data ===
print("Loading option data...")
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
df_opt=con.execute("SELECT timestamp,close,open,high,low FROM options_data WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp").fetchdf()
con.close()
print(f"Rows: {len(df_opt):,}")
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
cl_arr=df_opt["close"].values.astype(float)
hi_arr=df_opt["high"].values.astype(float)

def lookup(ts):
    i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
    if i==0: return 0,cl_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1]
    return i-1,cl_arr[i-1]

def entry_price(ed):
    _,c=lookup(ed); return c

def exit_tp_maxd(s_idx, tp, max_days):
    """Exit at TP or after max_days, whichever first"""
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
    ep=cl_arr[s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(cl_arr))):
        if ts_arr[i]>end_ns:
            return cl_arr[i]
        if cl_arr[i]-ep>=tp:
            return cl_arr[i]
    return None

def exit_maxd(s_idx, max_days):
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
    for i in range(s_idx+1, min(s_idx+3000, len(cl_arr))):
        if ts_arr[i]>end_ns:
            return cl_arr[i]
    return None

# Pre-compute entry prices
eps=np.array([entry_price(ed) for ed in trades_pre["ed_naive"]])
# Pre-compute start indices
s_idxs=np.array([lookup(ed)[0] for ed in trades_pre["ed_naive"]])

def test_combo(eps_arr, sidx_arr, tp, max_d, prem_max=None, day_filter=None, skip=0, return_idx=False):
    """Test combined variant. Returns pnl list, or (pnls, indices) if return_idx."""
    pnls=[]; indices=[]; n_skip=0
    for i in range(len(eps_arr)):
        if prem_max is not None and eps_arr[i]>prem_max: continue
        if day_filter is not None and trades_pre.iloc[i]["weekday"] not in day_filter: continue
        if n_skip>0: n_skip-=1; continue
        
        if tp is None:
            xp=exit_maxd(sidx_arr[i], max_d)
        else:
            xp=exit_tp_maxd(sidx_arr[i], tp, max_d)
        if xp is None: continue
        pnl=round(xp-eps_arr[i],1)
        pnls.append(pnl); indices.append(i)
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

# ============ 1. TP30 ± premium filters ============
print("\n=== TP30_Max7d + Premium Filters ===")
tp,tmax=30,7
add_result("TP30_Max7d (base)", test_combo(eps,s_idxs,tp,tmax))
for pm in [50,60,70,80,90,100,110,120,130,140,150,200]:
    pnls=test_combo(eps,s_idxs,tp,tmax,prem_max=pm)
    if len(pnls)>=5: add_result(f"+Prem<{pm}", pnls)

# ============ 2. TP30 + Entry Day Filters ============
print("\n=== TP30_Max7d + Day Filters ===")
for dname,dfilt in [("TueWedThu",[1,2,3]),("MonTueWed",[0,1,2]),("WedThu",[2,3]),
    ("Thu",[3]),("TueWed",[1,2]),("MonTue",[0,1])]:
    pnls=test_combo(eps,s_idxs,tp,tmax,day_filter=dfilt)
    if len(pnls)>=5: add_result(f"+Day{dname}", pnls)

# ============ 3. TP30 + Skip ============
print("\n=== TP30_Max7d + Skip After Loss ===")
for sk in [1,2,3]:
    pnls=test_combo(eps,s_idxs,tp,tmax,skip=sk)
    if len(pnls)>=5: add_result(f"+Skip{sk}", pnls)

# ============ 4. TP30 + Premium + Day ============
print("\n=== TP30 + Premium + Day (stacked) ===")
for pm in [80,100,120,150]:
    for dname,dfilt in [("TueWedThu",[1,2,3]),("WedThu",[2,3])]:
        pnls=test_combo(eps,s_idxs,tp,tmax,prem_max=pm,day_filter=dfilt)
        if len(pnls)>=5: add_result(f"+Prem<{pm}+Day{dname}", pnls)

# ============ 5. TP values with best filters ============
print("\n=== TP sweep with Prem<120 ===")
for tp_val in [20,30,40,50,60,75]:
    pnls=test_combo(eps,s_idxs,tp_val,tmax,prem_max=120)
    if len(pnls)>=5: add_result(f"TP{tp_val}_Max7d+Prem<120", pnls)

# ============ 6. Max days sweep with TP30 + Prem<120 ============
print("\n=== TP30 + Prem<120 + MaxDays sweep ===")
for md in [3,4,5,6,7,8,9,10,14]:
    pnls=test_combo(eps,s_idxs,30,md,prem_max=120)
    if len(pnls)>=5: add_result(f"TP30_Max{md}d+Prem<120", pnls)

# ============ 7. Hold4D + Premium filters ============
print("\n=== Hold4D + Premium Filters ===")
add_result("Hold4D (base)", test_combo(eps,s_idxs,None,4))
for pm in [80,100,120,150]:
    pnls=test_combo(eps,s_idxs,None,4,prem_max=pm)
    if len(pnls)>=5: add_result(f"Hold4D+Prem<{pm}", pnls)

# ============ 8. TP30 + Prem<120 + Day + Skip ============
print("\n=== TP30 + Prem<120 + Day + Skip ===")
for dname,dfilt in [("",None),("+TueWedThu",[1,2,3]),("+Thu",[3])]:
    for sk in [0,1]:
        label=f"TP30_Max7d+Prem<120{dname}"
        if sk>0: label+=f"+Skip{sk}"
        pnls=test_combo(eps,s_idxs,30,7,prem_max=120,day_filter=dfilt,skip=sk)
        if len(pnls)>=5: add_result(label, pnls)

# ============ 9. RANKING ============
rr=pd.DataFrame(all_rows).sort_values("Net",ascending=False).reset_index(drop=True)
print("\n"+"="*140)
print("COMBINED VARIANT RANKING")
print("="*140)
print(f"{'Rank':>4} {'Variant':<45} {'Trades':>6} {'Net':>10} {'WR':>5} {'W/L':>5} {'PF':>5} {'MDD':>8} {'Calmar':>6} {'Sharpe':>6} {'Avg':>7}")
print("-"*105)
for i,(_,r) in enumerate(rr.iterrows()):
    hl="*" if i==0 else " "
    print(f"{i+1:>3}{hl} {r['Variant']:<45} {r['Trades']:>6} {r['NetStr']:>10} {r['WRStr']:>5} {r['W/L']:>4.1f}x {r['PF']:>4.1f}x {r['MDD']:>8,.0f} {r['Calmar']:>5.1f}x {r['Sharpe']:>5.2f} {r['AvgStr']:>7}")

# ============ 10. BEST VARIANT DETAILED BREAKDOWN ============
best=rr.iloc[0]
print(f"\n{'='*100}")
print(f"BEST: {best['Variant']} Net={best['NetStr']} WR={best['WRStr']} Avg={best['AvgStr']}")
print(f"{'='*100}")

# Re-run best variant for breakdown
# Parse best variant params
v=best["Variant"]
tp_val=30
md_val=7
pm_val=None; df_val=None; sk_val=0
# Premium filter?
import re
pm_m=re.search(r'Prem<(\d+)',v)
if pm_m: pm_val=int(pm_m.group(1))
# Day filter?
day_map={"TueWedThu":[1,2,3],"MonTueWed":[0,1,2],"WedThu":[2,3],"Thu":[3],"TueWed":[1,2],"MonTue":[0,1]}
for dn,df in day_map.items():
    if dn in v: df_val=df; break
# Skip?
sk_m=re.search(r'Skip(\d)',v)
if sk_m: sk_val=int(sk_m.group(1))
# TP value
tp_m=re.search(r'TP(\d+)',v)
if tp_m: tp_val=int(tp_m.group(1))
# Max days
md_m=re.search(r'Max(\d+)d',v)
if md_m: md_val=int(md_m.group(1))

pnls,indices=test_combo(eps,s_idxs,tp_val,md_val,prem_max=pm_val,day_filter=df_val,skip=sk_val,return_idx=True)
actual_trades=trades_pre.iloc[indices].copy()
actual_trades["pnl"]=pnls

print(f"\nYearly ({len(pnls)} trades):")
yr=actual_trades.groupby("yr")["pnl"].agg(["count","sum",lambda x:(x>0).mean()])
yr.columns=["Trades","Net","WR"]
print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
for y in sorted(yr.index): r=yr.loc[y]; print(f"{int(y):>6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

print(f"\nMonthly:")
mo=actual_trades.groupby("mo")["pnl"].agg(["count","sum",lambda x:(x>0).mean()])
mo.columns=["Trades","Net","WR"]
print(f"{'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>7}")
for m in range(1,13):
    if m in mo.index: r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

# Equity curve
cum=np.cumsum(pnls)
mx=np.maximum.accumulate(cum)
dd=mx-cum
print(f"\nFinal Net: {sum(pnls):+,.0f}")
print(f"Max DD: {dd.max():,.0f}")
print(f"Calmar: {sum(pnls)/dd.max() if dd.max()>0 else 999:.1f}x")
print(f"Avg PnL: {np.mean(pnls):+.1f}")
print(f"Sharpe: {np.mean(pnls)/np.std(pnls)*np.sqrt(252):.2f}")
print(f"Max Win: {max(pnls):+.1f}  Max Loss: {min(pnls):+.1f}")
print(f"Wins: {(pd.Series(pnls)>0).sum()}  Losses: {(pd.Series(pnls)<0).sum()}")
print(f"Consec Wins: {max(np.diff(np.where(np.array(pnls)>0)[0])) if len(np.where(np.array(pnls)>0)[0])>1 else 1}")
print(f"Consec Losses: {max(np.diff(np.where(np.array(pnls)<=0)[0])) if len(np.where(np.array(pnls)<=0)[0])>1 else 1}")

print("\nDone!")
