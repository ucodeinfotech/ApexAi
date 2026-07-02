"""
NEW VARIANTS SWEEP: Month, ret_20h, DOW, volatility CH, partial profit, all-filters combo
Builds on verified infrastructure from all_improvements_combined.py
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def compute_atr20(h1):
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    return tr.rolling(20,min_periods=20).mean()

def compute_adx14(h1):
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    return 100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()

def compute_daily_ema(h1,period=50):
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

def compute_ret_n(h1,n):
    """Return over last n hours before each bar"""
    return h1["close"].pct_change(n)

def compute_n_consec_red(h1):
    """Count consecutive red candles"""
    is_red=(h1["close"]<h1["open"]).astype(int)
    groups=(is_red!=is_red.shift()).cumsum()
    return is_red.groupby(groups).cumsum()

# === LOAD DATA ===
print("Loading data...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    # Pre-compute h1 features
    h1["ret_20h"]=compute_ret_n(h1,20)
    h1["n_consec_red"]=compute_n_consec_red(h1)
    h1["dow"]=h1["datetime"].dt.dayofweek
    DATA[sym]={"h1":h1,"m5":m5,"atr5v":atr5.values,"m5_hi":m5["high"].values,"m5_lo":m5["low"].values,
               "m5_cl":m5["close"].values,"m5_atr":atr5.values,
               "m5_epoch":m5["datetime"].astype('int64').values,"tc":pd.Series(m5["datetime"]).dt.time.values}

CUT=pd.Timestamp("14:15").time()

def find_retest(sym,t,lv):
    d=DATA[sym];m5_epoch=d["m5_epoch"];m5_cl=d["m5_cl"];m5_lo=d["m5_lo"];tc=d["tc"]
    t_ep=t.asm8.view('int64')
    idx=np.searchsorted(m5_epoch,t_ep,side="right")
    if idx>=len(m5_cl):return None
    b=idx
    while b<len(m5_cl) and m5_cl[b]<=lv:b+=1
    if b>=len(m5_cl)-1:return None
    r=b+1
    while r<len(m5_cl):
        if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
        r+=1
    if r>=len(m5_cl):return None
    ep=m5_cl[r];sl=m5_lo[r]
    if ep-sl<=0:return None
    return (r,ep,sl)

def compute_ch_exits(sym,r,ep):
    d=DATA[sym];m5_cl=d["m5_cl"];m5_hi=d["m5_hi"];m5_atr=d["m5_atr"]
    pnls={}
    for cv in CH_VALS:
        he=ep
        for j in range(r,len(m5_cl)):
            ca=m5_atr[j]
            if pd.isna(ca):continue
            if m5_hi[j]>he:he=m5_hi[j]
            if m5_cl[j]<he-cv*ca:
                pnls[cv]=round(m5_cl[j]-ep,1);break
    return pnls

def compute_ch_atr_exits(sym,r,ep,mult):
    """CH = mult x ATR(14) at entry"""
    d=DATA[sym];m5_cl=d["m5_cl"];m5_hi=d["m5_hi"];m5_atr=d["m5_atr"]
    entry_atr=m5_atr[r]
    if pd.isna(entry_atr) or entry_atr==0:return None
    trail_pts=mult*entry_atr
    he=ep
    for j in range(r,len(m5_cl)):
        if m5_hi[j]>he:he=m5_hi[j]
        if m5_cl[j]<he-trail_pts:
            return round(m5_cl[j]-ep,1)
    return None

def compute_partial_exit(sym,r,ep,tight_ch,wide_ch):
    """Take partial profit at tight_ch, trail remainder at wide_ch"""
    d=DATA[sym];m5_cl=d["m5_cl"];m5_hi=d["m5_hi"];m5_atr=d["m5_atr"]
    he=ep;locked=False;partial_pnl=0
    for j in range(r,len(m5_cl)):
        ca=m5_atr[j]
        if pd.isna(ca):continue
        if m5_hi[j]>he:he=m5_hi[j]
        if not locked and m5_cl[j]>ep+tight_ch*ca:
            partial_pnl=tight_ch*ca*0.5  # half position booked
            locked=True
            partial_entry=m5_cl[j]
        if locked:
            trail=he-wide_ch*ca
            if m5_cl[j]<trail:
                remainder=m5_cl[j]-partial_entry
                return round(partial_pnl+remainder,1)
        else:
            if m5_cl[j]<he-wide_ch*ca:
                return round(m5_cl[j]-ep,1)  # full exit at wide stop
    return None

# === ENGULF_RAW SIGNALS ===
def sigs_engulf_raw(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],
                    "yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month,
                    "ret_20h":h1["ret_20h"].iloc[i],"n_consec_red":h1["n_consec_red"].iloc[i],
                    "dow":h1["dow"].iloc[i],"idx":i})
    return out

# === BUILD TRADES ===
print("Building Engulf_Raw trades with features...")
trades=[]
for sym in ["NIFTY50","SENSEX"]:
    for sig in sigs_engulf_raw(sym):
        ret=find_retest(sym,sig["ts"],sig["lv"])
        if ret is None:continue
        r,ep,sl=ret
        pnls=compute_ch_exits(sym,r,ep)
        if 45 not in pnls:continue
        # Compute ATR-based CH exits
        atr_exits={}
        for mult in [2,3,4,5]:
            p=compute_ch_atr_exits(sym,r,ep,mult)
            if p is not None:atr_exits[f"ch_{mult}xatr"]=p
        # Partial exit
        pp=compute_partial_exit(sym,r,ep,25,55)
        t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"ts":sig["ts"],
           "ret_20h":sig["ret_20h"],"n_consec_red":sig["n_consec_red"],"dow":sig["dow"],
           "pts45":pnls[45],"idx":sig["idx"]}
        for c,p in pnls.items():t[f"p{c}"]=p
        for k,v in atr_exits.items():t[k]=v
        if pp is not None:t["pp_25_55"]=pp
        trades.append(t)
df=pd.DataFrame(trades).fillna(0)
print(f"Total: {len(df)} trades")

# === BACKTEST ENGINE ===
def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def run_bt(df, pts_col="pts45", use_wl=False, skip_n=0, size_fn=None,
           mo_skip=None, mo_boost=None, ret20h_min=None, ret20h_max=None,
           dow_allow=None, ncr_min=None, use_dynch=False):
    rows=[]
    for yr in sorted(df["yr"].unique()):
        yr_d=df[df["yr"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        month_best={}
        if use_dynch:
            hist=df[df["yr"]<yr]
            for m in range(1,13):
                sub=hist[hist["mo"]==m]
                if len(sub)<5:month_best[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_best[m]=best
        rw=[];rl=[];skip_c=0
        for _,t in yr_d.iterrows():
            # Skip filter
            if skip_c>0:skip_c-=1;continue
            # Month skip
            if mo_skip and t["mo"] in mo_skip:continue
            # ret_20h filter
            if ret20h_min is not None and t["ret_20h"]<ret20h_min:continue
            if ret20h_max is not None and t["ret_20h"]>ret20h_max:continue
            # Day of week filter
            if dow_allow is not None and t["dow"] not in dow_allow:continue
            # n_consec_red filter
            if ncr_min is not None and t["n_consec_red"]<ncr_min:continue
            # Get PnL
            if pts_col in CH_COLS:
                if use_dynch:
                    m_ch=month_best.get(t["mo"],45)
                    pts=t.get(f"p{m_ch}",t["pts45"])
                else:
                    pts=t.get(pts_col,t["pts45"])
            elif pts_col=="pts45":
                pts=t.get("pts45",0)
            else:
                pts=t.get(pts_col,0)
            # Sizing
            sz=1.0
            if use_wl:sz=wl_size(rw,rl,lb=5)
            if size_fn=="2w1l":
                nw=sum(1 for x in rw[-2:] if x>0);nl=sum(1 for x in rl[-1:] if x<0)
                sz=1.5 if nw>=2 else (0.5 if nl>=1 else 1.0)
            if size_fn=="2w1l_skip2":
                nw=sum(1 for x in rw[-2:] if x>0);nl=sum(1 for x in rl[-1:] if x<0)
                sz=1.5 if nw>=2 else (0.5 if nl>=1 else 1.0)
            # Month boost
            if mo_boost and t["mo"] in mo_boost:sz*=2.0
            pnl=pts*sz
            rows.append({"pts":pnl,"yr":t["yr"],"mo":t["mo"],"sz":sz})
            if pnl>0:rw.append(pnl);skip_c=0
            else:rl.append(abs(pnl));skip_c=skip_n
    return pd.DataFrame(rows)

CH_COLS=set(f"p{v}" for v in CH_VALS)

def compute_metrics(res):
    if len(res)==0:return {"net":0,"n":0,"wr":0,"wl":0,"mdd":0}
    net=res["pts"].sum();n=len(res);wr=(res["pts"]>0).mean()
    w=res[res["pts"]>0]["pts"];l=res[res["pts"]<0]["pts"]
    wl_r=w.mean()/abs(l.mean()) if len(l)>0 and l.mean()!=0 else 999
    cum=res["pts"].cumsum();mx=cum.cummax();mdd=(mx-cum).max()
    return {"net":net,"n":n,"wr":wr,"wl":wl_r,"mdd":mdd}

# === CONFIGURATIONS ===
configs=[
    # (name, pts_col, use_wl, skip_n, size_fn, mo_skip, mo_boost, ret20h_min, ret20h_max, dow_allow, ncr_min, use_dynch)

    # === BASELINES (for comparison) ===
    ("CH45_base","p45",False,0,None,None,None,None,None,None,None,False),
    ("CH55_base","p55",False,0,None,None,None,None,None,None,None,False),
    ("CH55+WL+Skip2","p55",True,2,None,None,None,None,None,None,None,False),
    ("CH55+2w1l","p55",False,0,"2w1l",None,None,None,None,None,None,False),

    # === MONTH-AWARE VARIANTS ===
    ("CH55+WL+Skip2+MoSkip","p55",True,2,None,[1,9],None,None,None,None,None,False),
    ("CH55+WL+Skip2+MoBoostJun","p55",True,2,None,None,[6],None,None,None,None,False),
    ("CH55+WL+Skip2+MoSkip+MoBoost","p55",True,2,None,[1,9],[6],None,None,None,None,False),

    # === ret_20h FILTER VARIANTS ===
    ("CH55+WL+Skip2+ret20h>0","p55",True,2,None,None,None,0,None,None,None,False),
    ("CH55+WL+Skip2+ret20h>0.005","p55",True,2,None,None,None,0.005,None,None,None,False),

    # === DAY-OF-WEEK VARIANTS ===
    ("CH55+WL+Skip2+DOW45","p55",True,2,None,None,None,None,None,[3,4],None,False),
    ("CH55+WL+Skip2+DOW345","p55",True,2,None,None,None,None,None,[2,3,4],None,False),

    # === N_CONSEC_RED VARIANTS ===
    ("CH55+WL+Skip2+NCR>=5","p55",True,2,None,None,None,None,None,None,5,False),
    ("CH55+WL+Skip2+NCR>=3","p55",True,2,None,None,None,None,None,None,3,False),

    # === ATR-BASED CH (volatility-adjusted) ===
    ("CH_2xATR","ch_2xatr",False,0,None,None,None,None,None,None,None,False),
    ("CH_3xATR","ch_3xatr",False,0,None,None,None,None,None,None,None,False),
    ("CH_4xATR","ch_4xatr",False,0,None,None,None,None,None,None,None,False),
    ("CH_5xATR","ch_5xatr",False,0,None,None,None,None,None,None,None,False),
    ("CH_3xATR+WL+Skip2","ch_3xatr",True,2,None,None,None,None,None,None,None,False),
    ("CH_4xATR+WL+Skip2","ch_4xatr",True,2,None,None,None,None,None,None,None,False),

    # === PARTIAL PROFIT ===
    ("PP25_55","pp_25_55",False,0,None,None,None,None,None,None,None,False),
    ("PP25_55+WL+Skip2","pp_25_55",True,2,None,None,None,None,None,None,None,False),

    # === AGGRESSIVE ===
    ("CH55+2w1l+Skip2","p55",False,2,"2w1l",None,None,None,None,None,None,False),

    # === DynCH + improvements ===
    ("DynCH+WL+Skip2+MoSkip","p45",True,2,None,[1,9],None,None,None,None,None,True),
    ("DynCH+WL+Skip2+ret20h>0","p45",True,2,None,None,None,0,None,None,None,True),

    # === ALL-FILTERS COMBO ===
    ("CH55+WL+Skip2+ALL","p55",True,2,None,[1,9],[6],0,None,[3,4],5,False),

    # === ret_20h RANGE ===
    ("CH55+WL+Skip2+ret20h_neg","p55",True,2,None,None,None,None,0,None,None,False),
]

# === RUN ALL ===
print("\n"+"="*130)
print("NEW VARIANTS SWEEP - Full Results")
print("="*130)
print(f"{'#':>3} {'Config':<32} {'Trades':>7} {'Net Pts':>12} {'WR':>6} {'W/L':>7} {'MDD':>9} {'Net/MDD':>8}")
print("-"*130)

results=[]
for cname, col, wl, skip, szfn, moskip, moboost, rmin, rmax, dow, ncr, dyn in configs:
    res=run_bt(df, pts_col=col, use_wl=wl, skip_n=skip, size_fn=szfn,
               mo_skip=moskip, mo_boost=moboost,
               ret20h_min=rmin, ret20h_max=rmax, dow_allow=dow, ncr_min=ncr, use_dynch=dyn)
    m=compute_metrics(res)
    nm=m["net"]/m["mdd"] if m["mdd"]>0 else 999
    results.append((cname,m["net"],m["wr"],m["wl"],m["mdd"],m["n"],nm))
    label="**" if "WL+Skip2" in cname and "ALL" not in cname and cname!="CH55+WL+Skip2" else ""
    print(f"{len(results):>3} {cname:<32} {m['n']:>7} {m['net']:>+11,.0f}  {m['wr']:>5.1%} {m['wl']:>5.1f}x {m['mdd']:>+7,.0f}  {nm:>6.1f}x  {label}")

# Sort by net descending
results.sort(key=lambda r: -r[1])
print("\n"+"="*130)
print("RANKED BY NET PNL")
print("="*130)
print(f"{'#':>3} {'Config':<32} {'Trades':>7} {'Net Pts':>12} {'WR':>6} {'W/L':>7} {'MDD':>9} {'Net/MDD':>8}")
print("-"*130)
for i,(cname,net,wr,wl,mdd,n,nm) in enumerate(results,1):
    print(f"{i:>3} {cname:<32} {n:>7} {net:>+11,.0f}  {wr:>5.1%} {wl:>5.1f}x {mdd:>+7,.0f}  {nm:>6.1f}x")

# Sort by Net/MDD
results.sort(key=lambda r: -r[6])
print("\n"+"="*130)
print("RANKED BY RISK-ADJUSTED (Net/MDD)")
print("="*130)
print(f"{'#':>3} {'Config':<32} {'Trades':>7} {'Net Pts':>12} {'WR':>6} {'W/L':>7} {'Net/MDD':>8}")
print("-"*130)
for i,(cname,net,wr,wl,mdd,n,nm) in enumerate(results,1):
    print(f"{i:>3} {cname:<32} {n:>7} {net:>+11,.0f}  {wr:>5.1%} {wl:>5.1f}x  {nm:>6.1f}x")

# === BEST OF NEW vs BASELINE ===
print("\n"+"="*130)
print("BEST NEW VARIANTS vs BASELINE")
print("="*130)
baseline_net=0
baseline_nm=0
for cname,net,wr,wl,mdd,n,nm in results:
    if cname=="CH55+WL+Skip2":
        baseline_net=net;baseline_nm=nm
        break
print(f"\n  Baseline (CH55+WL+Skip2):    Net={baseline_net:>+11,.0f}  Net/MDD={baseline_nm:>6.1f}x")
# Best new by net
new_results=[r for r in results if r[0] not in ("CH45_base","CH55_base","CH55+WL+Skip2","CH55+2w1l")]
if new_results:
    best_net=max(new_results,key=lambda r:r[1])
    best_nm=max(new_results,key=lambda r:r[6])
    print(f"  Best new by Net:             {best_net[0]:<32} Net={best_net[1]:>+11,.0f}  Net/MDD={best_net[6]:>6.1f}x")
    print(f"  Best new by Net/MDD:         {best_nm[0]:<32} Net={best_nm[1]:>+11,.0f}  Net/MDD={best_nm[6]:>6.1f}x")

# Check if any new variant beats baseline
for cname,net,wr,wl,mdd,n,nm in results:
    if cname in ("CH45_base","CH55_base","CH55+WL+Skip2","CH55+2w1l"):continue
    if net>baseline_net:
        print(f"\n  >>> {cname} BEATS BASELINE by {net-baseline_net:+,.0f} pts!")
    if nm>baseline_nm:
        print(f"  >>> {cname} BEATS BASELINE Net/MDD ({nm:.1f}x vs {baseline_nm:.1f}x)")

print("\n"+"="*130)
print("DONE")
print("="*130)
