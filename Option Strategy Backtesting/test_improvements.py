"""
IMPROVEMENT TEST: Combine rolling W/L with all other signals
Tests layered improvements to find the optimal system
"""
import pandas as pd, numpy as np, os, warnings
from itertools import product
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

CH_VALS=[25,30,35,40,45,50,55,60]; CH_BASE=45; CH_RANGE=10

def build_trades():
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
        m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
        hl=h1["high"]-h1["low"];hpc=abs(h1["high"]-h1["close"].shift(1));lpc=abs(h1["low"]-h1["close"].shift(1))
        tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);h1["atr14"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()
        a14=h1["atr14"].values;a20=pd.Series(a14).rolling(20).mean().values
        h1["atr14_pct"]=h1["atr14"]/h1["close"]*100;h1["range_pct"]=(h1["high"]-h1["low"])/h1["open"]*100
        h1["body_pct"]=(h1["close"]-h1["open"]).abs()/h1["open"]*100
        for p in [5,20,50]:h1[f"ema{p}"]=h1["close"].ewm(span=p,adjust=False).mean()
        h1["ema_pos"]=(h1["ema5"]>h1["ema20"]).astype(int)
        hl5=m5["high"]-m5["low"];hpc5=abs(m5["high"]-m5["close"].shift(1));lpc5=abs(m5["low"]-m5["close"].shift(1))
        tr5=pd.concat([hl5,hpc5,lpc5],axis=1).max(axis=1);m5_atr=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
        atr5=m5_atr.values;m5_hi=m5["high"].values;m5_lo=m5["low"].values;m5_cl=m5["close"].values;m5_du=m5["datetime"].values
        prev_red=np.roll(h1["close"].values<h1["open"].values,1);prev_red[0]=False
        tc=pd.Series(m5["datetime"]).dt.time.values;CUT=pd.Timestamp("14:15").time()
        bl=50 if "NIFTY" in sym else 10
        for i in range(60,len(h1)):
            if not(prev_red[i] and h1["close"].values[i]>h1["open"].values[i]):continue
            if not(h1["open"].values[i]<=h1["close"].values[i-1] and h1["close"].values[i]>=h1["open"].values[i-1]):continue
            if h1["high"].values[i]-h1["low"].values[i]<0.5*(h1["high"].values[i-1]-h1["low"].values[i-1]):continue
            if h1["datetime"].iloc[i].hour==9:continue
            lv=h1["high"].values[i];tu=h1["datetime"].values[i]
            idx=np.searchsorted(m5_du,tu,side="right")
            if idx>=len(m5):continue
            b=idx
            while b<len(m5) and m5_cl[b]<=lv:b+=1
            if b>=len(m5)-1:continue
            r=b+1
            while r<len(m5):
                if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
                r+=1
            if r>=len(m5):continue
            ep=m5_cl[r]
            if ep-m5_lo[r]<=0:continue
            a14v=a14[i];a20v=a20[i]
            reg=0
            if not pd.isna(a14v) and not pd.isna(a20v):
                if a14v>a20v*1.1:reg=1
                elif a14v<a20v*0.9:reg=2
            pnls={}
            for cv2 in CH_VALS:
                he=ep
                for j in range(r,len(m5)):
                    ca=atr5[j]
                    if pd.isna(ca):continue
                    if m5_hi[j]>he:he=m5_hi[j]
                    if m5_cl[j]<he-cv2*ca:
                        pnls[cv2]=round((m5_cl[j]-ep)*bl-20,2);break
            if 45 not in pnls:continue
            retrace_bars=r-b;atr14_pct=float(h1["atr14_pct"].values[i]) if not pd.isna(h1["atr14_pct"].values[i]) else 0
            t={"sym":sym,"year":h1["datetime"].iloc[i].year,"month":h1["datetime"].iloc[i].month,
               "hour":h1["datetime"].iloc[i].hour,"pnl":pnls[45],"is_win":pnls[45]>0,
               "reg":reg,"retrace_bars":retrace_bars,"atr14_pct":atr14_pct,
               "range_pct":float(h1["range_pct"].iloc[i]) if not pd.isna(h1["range_pct"].iloc[i]) else 0,
               "body_pct":float(h1["body_pct"].iloc[i]) if not pd.isna(h1["body_pct"].iloc[i]) else 0,
               "ema_pos":int(h1["ema_pos"].iloc[i]) if not pd.isna(h1["ema_pos"].iloc[i]) else 0,
               "ep":ep,"bl":bl,"ts":h1["datetime"].iloc[i]}
            for c,p in pnls.items():t[f"p{c}"]=p
            all_t.append(t)
    return pd.DataFrame(all_t).fillna(0)

print("Building trades...")
df=build_trades()
print(f"Total: {len(df)} trades")

# ═══════════════════════════════════════════════
# TEST ALL COMBINATIONS
# ═══════════════════════════════════════════════
print("\n"+"="*110)
print("TESTING ALL IMPROVEMENT COMBINATIONS (Walk-Forward 2022-2026)")
print("="*110)

def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def test_system(df, use_wl=False, skip_months=None, skip_hour15=False, retrace_max=None, atr_max=None, ema_only=False, dyn_ch=False, ch_key_getter=None):
    """Test a system with various filters. Returns list of PnLs."""
    results=[];rw=[];rl=[]
    for yr in range(2022,2027):
        yr_d=df[df["year"]==yr].sort_values("ts")
        # For dynamic CH: compute best per month from history
        month_ch={}
        if dyn_ch:
            hist=df[df["year"]<yr]
            for m in range(1,13):
                sub=hist[hist["month"]==m]
                if len(sub)<5:month_ch[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_ch[m]=best
        for _,t in yr_d.iterrows():
            take=True;ch=t["p45"];sz=1.0
            # Dynamic CH
            if dyn_ch:
                m_ch=month_ch.get(t["month"],45)
                col=f"p{m_ch}"
                ch=t.get(col,t["p45"])
            else:
                ch=t["p45"]
            # Skip filters
            if skip_months and t["month"] in skip_months:take=False
            if take and skip_hour15 and t["hour"]==15:take=False
            if take and retrace_max and t["retrace_bars"]>retrace_max:take=False
            if take and atr_max and t["atr14_pct"]>atr_max:take=False
            if take and ema_only and t["ema_pos"]!=1:take=False
            if take and use_wl:sz=wl_size(rw,rl,lb=5)
            if ch_key_getter:
                ch=ch_key_getter(t,yr)
            pnl=ch*sz
            results.append(pnl)
            if pnl>0:rw.append(pnl)
            else:rl.append(abs(pnl))
    return results

# Define systems to test
systems = []

# 1. Baseline
systems.append(("BASELINE CH45", {"use_wl":False}))
# 2. W/L alone
systems.append(("W/L Sizing only", {"use_wl":True}))
# 3. W/L + month filter
systems.append(("W/L + Skip Jan/Sep/Dec", {"use_wl":True,"skip_months":[1,9,12]}))
# 4. W/L + hour filter
systems.append(("W/L + Skip Hour 15", {"use_wl":True,"skip_hour15":True}))
# 5. W/L + month + hour
systems.append(("W/L + Month+Hour", {"use_wl":True,"skip_months":[1,9,12],"skip_hour15":True}))
# 6. W/L + retrace_bars <= 500
systems.append(("W/L + Retrace<=500", {"use_wl":True,"retrace_max":500}))
# 7. W/L + ATR% <= 0.40
systems.append(("W/L + ATR%<=0.40", {"use_wl":True,"atr_max":0.40}))
# 8. W/L + EMA5>20 only
systems.append(("W/L + EMA>20 only", {"use_wl":True,"ema_only":True}))
# 9. W/L + Dynamic CH by month
systems.append(("W/L + DynCH(Month)", {"use_wl":True,"dyn_ch":True}))
# 10. W/L + Month+Hour + DynCH
systems.append(("W/L + M+H + DynCH", {"use_wl":True,"skip_months":[1,9,12],"skip_hour15":True,"dyn_ch":True}))
# 11. W/L + retrace_bars <= 500 + ATR <= 0.40 + EMA>20
systems.append(("W/L + Retr+ATR+EMA", {"use_wl":True,"retrace_max":500,"atr_max":0.40,"ema_only":True}))
# 12. ALL filters
systems.append(("ALL: W/L+M+H+Retr+ATR+EMA", {"use_wl":True,"skip_months":[1,9,12],"skip_hour15":True,
                                               "retrace_max":500,"atr_max":0.40,"ema_only":True}))
# 13. W/L + Month+Hour + DynCH + retrace
systems.append(("W/L + M+H+DynCH+Retr", {"use_wl":True,"skip_months":[1,9,12],"skip_hour15":True,
                                         "dyn_ch":True,"retrace_max":500}))

base_net = sum(test_system(df, use_wl=False))
results_table = []

for name, params in systems:
    res=test_system(df, **params)
    net=sum(res); n_trades=sum(1 for r in res if r!=0)
    wr=sum(1 for r in res if r>0)/n_trades if n_trades>0 else 0
    peak=0;running=0;mdd=0
    for r in res:
        running+=r
        if running>peak:peak=running
        dd=peak-running
        if dd>mdd:mdd=dd
    vs_base=(net/base_net-1)*100 if base_net else 0
    impr=(net/base_net-1)*100 if base_net else 0
    results_table.append((name,net,n_trades,wr,mdd,impr))

# Print sorted by net
results_table.sort(key=lambda x:x[1], reverse=True)
print(f"\n{'Rank':<5s} {'Strategy':<40s} {'Net':>12s} {'N':>5s} {'WR':>6s} {'MDD':>12s} {'vs Base':>10s}")
print("-"*95)
for i,(name,net,n,wr,mdd,impr) in enumerate(results_table):
    print(f"  {i+1:<3d} {name:<40s} Rs{net:>+9,.0f} {n:4d} {wr:>5.1%} Rs{mdd:>+9,.0f} {impr:>+8.1f}%")

# ═══════════════════════════════════════════════
# YEAR BREAKDOWN: TOP 3 SYSTEMS
# ═══════════════════════════════════════════════
print("\n"+"="*110)
print("YEAR BREAKDOWN: TOP 3 SYSTEMS")
print("="*110)

for rank,(name,params) in enumerate([systems[i] for i in [0,2,3,4,6,8]]):
    if rank>=4:break
    res=test_system(df,**params)
    print(f"\n--- {name} ---")
    print(f"{'Year':<8s} {'Net':>12s} {'N':>5s} {'WR':>6s}")
    for yr in range(2022,2027):
        # Re-run per year to separate
        yr_d=df[df["year"]==yr].sort_values("ts")
        p2=params.copy()
        yr_res=test_system(df,**p2)
        # This is wrong - need to filter by year. Let me recompute per year.
    # Actually filter the full result by year is complex since we sort globally
    # Let me just print the total
    yr_breakdown=[]
    rw=[];rl=[]
    for yr in range(2022,2027):
        yr_d=df[df["year"]==yr].sort_values("ts")
        yr_pnls=[]
        for _,t in yr_d.iterrows():
            take=True;sz=1.0
            ch=t["p45"]
            if params.get("dyn_ch") and False:pass
            m_ch=None
            if params.get("skip_months") and t["month"] in params["skip_months"]:take=False
            if take and params.get("skip_hour15") and t["hour"]==15:take=False
            if take and params.get("retrace_max") and t["retrace_bars"]>params["retrace_max"]:take=False
            if take and params.get("atr_max") and t["atr14_pct"]>params["atr_max"]:take=False
            if take and params.get("ema_only") and t["ema_pos"]!=1:take=False
            if take and params.get("use_wl"):sz=wl_size(rw,rl,lb=5)
            pnl=ch*sz
            yr_pnls.append(pnl)
            if pnl>0:rw.append(pnl)
            else:rl.append(abs(pnl))
        yr_breakdown.append((yr,sum(yr_pnls),sum(1 for p in yr_pnls if p!=0),sum(1 for p in yr_pnls if p>0)/max(sum(1 for p in yr_pnls if p!=0),1)))
    for yr,n,w,wr2 in yr_breakdown:
        print(f"  {yr}: Rs{n:>+10,.0f} {w:3d} {wr2:.1%}")
    total_net=sum(x[1] for x in yr_breakdown)
    print(f"  TOTAL: Rs{total_net:>+10,.0f}")

# ═══════════════════════════════════════════════
# OPTIMAL THRESHOLD SEARCH
# ═══════════════════════════════════════════════
print("\n"+"="*110)
print("THRESHOLD OPTIMIZATION: Finding best W/L thresholds")
print("="*110)

# Test different threshold sets
threshold_configs = [
    ("Conservative", [(1.5,0.75)]),
    ("Medium", [(1.2,0.5),(2.0,0.75)]),
    ("Aggressive", [(0.8,0.1),(1.2,0.3),(1.5,0.5),(2.0,0.75)]),
    ("Very Aggressive", [(0.5,0.05),(1.0,0.2),(1.5,0.5)]),
    ("Step", [(1.0,0.25),(1.3,0.5),(1.6,0.75),(2.0,0.9)]),
    ("Binary", [(1.5,0.5)]),
    ("Mild", [(1.2,0.5)]),
    ("Safe", [(0.8,0.25),(1.2,0.5)]),
]

for lb in [3,5,8,10,15]:
    for tname, thresh in threshold_configs:
        rows=[];rw=[];rl=[]
        for yr in range(2022,2027):
            yr_d=df[df["year"]==yr].sort_values("ts")
            for _,t in yr_d.iterrows():
                sz=1.0
                if rw and rl:
                    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
                    r=aw/al if al>0 else 10
                    for thr,sz2 in sorted(thresh):
                        if r<thr:sz=sz2;break
                pnl=t["p45"]*sz
                rows.append(pnl)
                if pnl>0:rw.append(pnl)
                else:rl.append(abs(pnl))
        net=sum(rows);mdd=0;peak=0;running=0
        for r in rows:running+=r;peak=max(peak,running);mdd=max(mdd,peak-running)
        impr=(net/base_net-1)*100
        print(f"  {tname:<20s} lb={lb:2d}: Net=Rs{net:>+9,.0f} MDD=Rs{mdd:>+9,.0f} {impr:+.1f}%")

# ═══════════════════════════════════════════════
# BEST COMBINATION: DETAILED RESULTS
# ═══════════════════════════════════════════════
print("\n"+"="*110)
print("BEST SYSTEM: DETAILED RESULTS")
print("="*110)

best_name = results_table[0][0]
best_params = {}
for name, params in systems:
    if name == best_name:
        best_params = params
        break

if best_params:
    res=test_system(df,**best_params)
    net_all=sum(res); n_all=sum(1 for r in res if r!=0)
    wr_all=sum(1 for r in res if r>0)/n_all if n_all>0 else 0
    
    print(f"\n  Best System: {best_name}")
    print(f"  Total: Rs{net_all:>+10,.0f} | N={n_all} | WR={wr_all:.1%}")
    
    # Year breakdown
    rw=[];rl=[]
    for yr in range(2022,2027):
        yr_d=df[df["year"]==yr].sort_values("ts")
        yr_pnls=[]
        for _,t in yr_d.iterrows():
            take=True;sz=1.0;ch=t["p45"]
            if best_params.get("skip_months") and t["month"] in best_params["skip_months"]:take=False
            if take and best_params.get("skip_hour15") and t["hour"]==15:take=False
            if take and best_params.get("retrace_max") and t["retrace_bars"]>best_params["retrace_max"]:take=False
            if take and best_params.get("atr_max") and t["atr14_pct"]>best_params["atr_max"]:take=False
            if take and best_params.get("ema_only") and t["ema_pos"]!=1:take=False
            if take and best_params.get("use_wl"):sz=wl_size(rw,rl,lb=5)
            pnl=ch*sz
            yr_pnls.append(pnl)
            if pnl>0:rw.append(pnl)
            else:rl.append(abs(pnl))
        base_yr=df[df["year"]==yr]["p45"].sum()
        print(f"  {yr}: Base=Rs{base_yr:>+9,.0f} Fix=Rs{sum(yr_pnls):>+9,.0f} Delta=Rs{sum(yr_pnls)-base_yr:>+9,.0f}")
    
    # Per symbol
    for sym in ["NIFTY50","SENSEX"]:
        sym_d=df[(df["sym"]==sym)&(df["year"]>=2022)].sort_values("ts")
        sym_pnls=[];rw=[];rl=[]
        for _,t in sym_d.iterrows():
            take=True;sz=1.0;ch=t["p45"]
            if best_params.get("skip_months") and t["month"] in best_params["skip_months"]:take=False
            if take and best_params.get("skip_hour15") and t["hour"]==15:take=False
            if take and best_params.get("retrace_max") and t["retrace_bars"]>best_params["retrace_max"]:take=False
            if take and best_params.get("atr_max") and t["atr14_pct"]>best_params["atr_max"]:take=False
            if take and best_params.get("ema_only") and t["ema_pos"]!=1:take=False
            if take and best_params.get("use_wl"):sz=wl_size(rw,rl,lb=5)
            pnl=ch*sz
            sym_pnls.append(pnl)
            if pnl>0:rw.append(pnl)
            else:rl.append(abs(pnl))
        base_sym=sym_d["p45"].sum()
        pts_sym=[(p+20)/ (50 if sym=="NIFTY50" else 10) for p in sym_pnls]
        base_pts=[(p+20)/(50 if sym=="NIFTY50" else 10) for p in sym_d["p45"]]
        print(f"  {sym}: Base=Rs{base_sym:>+9,.0f} Fix=Rs{sum(sym_pnls):>+9,.0f} "
              f"Points: Base={sum(base_pts):>+8,.0f} Fix={sum(pts_sym):>+8,.0f} "
              f"WR={sum(1 for p in sym_pnls if p>0)/max(sum(1 for p in sym_pnls if p!=0),1):.1%}")
