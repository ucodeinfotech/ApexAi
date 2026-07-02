"""
Deep ITM CALL - fully in-memory after bulk load
Load all CALL option prices into pandas, then do fast dict lookups
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, bisect
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# === Build trades (same engine) ===
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
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month}); break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"]=trades["exit_dt"].dt.tz_localize(None)
trades["f5d_naive"]=trades["ed_naive"]+pd.Timedelta(days=5)
from datetime import timedelta
import calendar
exp_w_l,exp_m_l=[],[]
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
print(f"Option-era trades: {len(trades_pre)}")

# === Bulk load option data into pandas ===
print("Loading option data from DuckDB...")
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
# Load only CALL, expiry_code=1, for all needed ATM distances
atm_list=[0,-1,-2,-3,-4,-5]
atm_str=",".join([str(a) for a in atm_list])
df_opt=con.execute(f"""
    SELECT timestamp, close, expiry_flag, atm_distance, strike
    FROM options_data 
    WHERE option_type='CALL' AND expiry_code=1 
      AND atm_distance IN ({atm_str})
    ORDER BY timestamp
""").fetchdf()
con.close()
print(f"Option rows loaded: {len(df_opt):,}")
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
print(f"  Timestamp range: {df_opt['timestamp'].min()} to {df_opt['timestamp'].max()}")

# Build per-(flag,atm) dicts for fast lookup
flag_map={"WEEK":"WEEK","MONTH":"MONTH"}
opt_dicts={}
for flag in ["WEEK","MONTH"]:
    for atm in atm_list:
        sub=df_opt[(df_opt["expiry_flag"]==flag)&(df_opt["atm_distance"]==atm)]
        # Build dict: timestamp -> close
        # Use nearest neighbor lookup
        ts_arr=sub["timestamp"].values.astype("datetime64[us]")
        cl_arr=sub["close"].values.astype(float)
        opt_dicts[(flag,atm)]=(ts_arr,cl_arr)
        print(f"  {flag} ATM={atm}: {len(sub):,} rows")

def lookup_price(ts, flag, atm):
    """Nearest neighbor lookup in pre-loaded arrays"""
    ts_arr,cl_arr=opt_dicts.get((flag,atm),(np.array([],dtype="datetime64[us]"),np.array([])))
    if len(ts_arr)==0: return None
    ts_ns=np.datetime64(ts,"us")
    i=np.searchsorted(ts_arr,ts_ns)
    if i==0: return float(cl_arr[0]) if not np.isnan(cl_arr[0]) else None
    if i>=len(ts_arr): return float(cl_arr[-1]) if not np.isnan(cl_arr[-1]) else None
    return float(cl_arr[i-1]) if not np.isnan(cl_arr[i-1]) else None

def lookup_price_near(ts, flag, atm, window=np.timedelta64(86400,"s")):
    """Nearest neighbor within time window"""
    ts_arr,cl_arr=opt_dicts.get((flag,atm),(np.array([]),np.array([])))
    if len(ts_arr)==0: return None
    ts_ns=np.datetime64(ts,"us")
    i=np.searchsorted(ts_arr,ts_ns)
    best=None; best_dist=None
    for j in range(max(0,i-5),min(len(ts_arr),i+5)):
        dist=abs(ts_arr[j]-ts_ns)
        if dist>window: continue
        if best_dist is None or dist<best_dist:
            if not np.isnan(cl_arr[j]):
                best=float(cl_arr[j]); best_dist=dist
    return best

# === Run all variants ===
all_rows=[]
for flag in ["WEEK","MONTH"]:
    for atm in atm_list:
        print(f"  Testing {flag} ATM={atm}...",end=" ")
        recs=[]
        for _,row in trades_pre.iterrows():
            ep=lookup_price(row["ed_naive"],flag,atm)
            if ep is None: continue
            xp=lookup_price(row["xd_naive"],flag,atm)
            f5=lookup_price(row["f5d_naive"],flag,atm)
            exp=lookup_price_near(row["exp_w" if flag=="WEEK" else "exp_m"],flag,atm)
            recs.append({"yr":row["yr"],"mo":row["mo"],
                "entry":ep,
                "spot_pnl":round(xp-ep,1) if xp else None,
                "d5_pnl":round(f5-ep,1) if f5 else None,
                "expiry_pnl":round(exp-ep,1) if exp else None})
        print(f"{len(recs)} records")
        
        for ec,en in [("spot_pnl","SpotExit"),("d5_pnl","Hold5Day"),("expiry_pnl","HoldExp")]:
            vals=[r[ec] for r in recs if r[ec] is not None]
            if len(vals)<5: continue
            p=pd.Series(vals); n=len(p); net=p.sum(); wr=(p>0).mean()
            aw=p[p>0].mean() if (p>0).sum()>0 else 0
            al=p[p<0].mean() if (p<0).sum()>0 else 0
            wl=aw/abs(al) if al!=0 else 999; pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
            cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
            calmar=net/mdd if mdd>0 else 999; sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
            ae=np.mean([r["entry"] for r in recs if r[ec] is not None])
            all_rows.append({"Variant":f"ATM{atm}_{en}_{flag}","Net":net,"WR":wr,
                "W/L":wl,"PF":pf,"MDD":mdd,"Calmar":calmar,"Sharpe":sharpe,
                "Trades":n,"NetStr":f"{net:+,.0f}","WRStr":f"{wr:.0%}","AvgEntry":f"{ae:.1f}"})

rr=pd.DataFrame(all_rows).sort_values("Net",ascending=False).reset_index(drop=True)

# === DISPLAY ===
print("\n"+"="*130)
print("DEEP ITM CALL - ALL VARIANTS RANKED (in-memory lookups)")
print("="*130)
print(f"{'Rank':>4} {'Variant':<32} {'Trades':>6} {'Net':>12} {'WR':>5} {'W/L':>6} {'PF':>6} {'MDD':>10} {'Calmar':>8} {'Sharpe':>7}")
print("-"*100)
for i,(_,r) in enumerate(rr.iterrows()):
    hl="*" if i==0 else " "
    if r["Net"]>0 or i<5:
        print(f"{i+1:>3}{hl} {r['Variant']:<32} {r['Trades']:>6} {r['NetStr']:>12} {r['WRStr']:>5} {r['W/L']:>5.1f}x {r['PF']:>5.1f}x {r['MDD']:>8,.0f} {r['Calmar']:>7.1f}x {r['Sharpe']:>6.2f}")

pos=rr[rr["Net"]>0]
print(f"\n=== POSITIVE VARIANTS: {len(pos)} of {len(rr)} ===")
for i,(_,r) in enumerate(pos.iterrows()):
    print(f"  {i+1}. {r['Variant']:<32} Net={r['NetStr']:>10} WR={r['WRStr']:>5} W/L={r['W/L']:>4.1f}x Calmar={r['Calmar']:>6.1f}x")

print("\n"+"="*100)
print("ATM COMPARISON (WEEKLY, Spot Exit)")
print("="*100)
print(f"{'ATM':>6} {'Trades':>7} {'Net':>12} {'WR':>6} {'W/L':>6} {'AvgEntry':>10}")
for atm in atm_list:
    sub=rr[(rr["Variant"]==f"ATM{atm}_SpotExit_WEEK")]
    if len(sub)>0:
        r=sub.iloc[0]; print(f"ATM{atm:>2} {r['Trades']:>7} {r['NetStr']:>12} {r['WRStr']:>6} {r['W/L']:>5.1f}x {r['AvgEntry']:>10}")

# Best detail
pos2=rr[rr["Net"]>0].sort_values("Net",ascending=False)
if len(pos2)>0:
    best=pos2.iloc[0]
    print(f"\n{'='*100}")
    print(f"BEST: {best['Variant']} Net={best['NetStr']} WR={best['WRStr']} W/L={best['W/L']:5.1f}x")
    print(f"{'='*100}")
    bparts=best["Variant"].split("_")
    atm_v=int(bparts[0].replace("ATM",""))
    em_v=bparts[1]; fl_v=bparts[2]
    col_map={"SpotExit":"spot_pnl","Hold5Day":"d5_pnl","HoldExp":"expiry_pnl"}
    ec=col_map[em_v]
    
    recs2=[]
    for _,row in trades_pre.iterrows():
        ep=lookup_price(row["ed_naive"],fl_v,atm_v)
        if ep is None: continue
        if em_v=="SpotExit": xp=lookup_price(row["xd_naive"],fl_v,atm_v)
        elif em_v=="Hold5Day": xp=lookup_price(row["f5d_naive"],fl_v,atm_v)
        else: xp=lookup_price_near(row["exp_w" if fl_v=="WEEK" else "exp_m"],fl_v,atm_v)
        if xp is None: continue
        recs2.append({"yr":row["yr"],"mo":row["mo"],"pnl":round(xp-ep,1)})
    bdf=pd.DataFrame(recs2)
    
    if len(bdf)>0:
        print(f"\nYearly ({len(bdf)} trades):")
        yr=bdf.groupby("yr")["pnl"].agg(["count","sum",lambda x:(x>0).mean()])
        yr.columns=["Trades","Net","WR"]
        print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
        for y in sorted(yr.index): r=yr.loc[y]; print(f"{int(y):>6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")
        print(f"\nMonthly:")
        mo=bdf.groupby("mo")["pnl"].agg(["count","sum",lambda x:(x>0).mean()])
        mo.columns=["Trades","Net","WR"]
        print(f"{'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>7}")
        for m in range(1,13):
            if m in mo.index: r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

print("\nDone!")
