"""
Additional testing on Engulfing-only 1-lot - condensed
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20; CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# Pre-load all data once
h1_data={}; m5_data={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv")); m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1_data[sym]=h1.sort_values("datetime").reset_index(drop=True)
    m5_data[sym]=m5.sort_values("datetime").reset_index(drop=True)

def compute(h1,m5):
    h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    return h1,m5

for sym in h1_data:
    h1_data[sym],m5_data[sym]=compute(h1_data[sym],m5_data[sym])

# Pre-compute engulfing signals once
sigs_all={}
for sym in ["NIFTY50","SENSEX"]:
    h1=h1_data[sym]
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        sigs.append({"tt":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"ix":i,"close":h1["close"].iloc[i]})
    sigs_all[sym]=sigs

def run_test_fast(ch_base=45, ch_adj=10, dyn=True, body_min=0.5):
    """Fast version using pre-loaded data"""
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=h1_data[sym]; m5=m5_data[sym]
        sigs=[s for s in sigs_all[sym] if abs(h1["close"].iloc[s["ix"]]-h1["open"].iloc[s["ix"]])>=body_min*abs(h1["close"].iloc[s["ix"]-1]-h1["open"].iloc[s["ix"]-1])]
        sigs=[s for s in sigs if h1["close"].iloc[s["ix"]]>=h1["open"].iloc[s["ix"]-1]]
        
        bl=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=A(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        
        for sig in sigs:
            ix=sig["ix"]; tu=int(pd.to_datetime(sig["tt"]).timestamp()); lv=sig["lv"]
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5): continue
            brk=idx
            while brk<len(m5) and cl[brk]<=lv: brk+=1
            if brk>=len(m5): continue
            rt=brk+1
            while rt<len(m5):
                if lo[rt]<lv and cl[rt]>lv and tc.iloc[rt]<CUTOFF: break
                rt+=1
            if rt>=len(m5): continue
            ep=cl[rt]
            if ep-lo[rt]<=0 or m5["datetime"].iloc[rt].hour==9: continue
            
            atr14_v=h1["atr14"].iloc[ix]; atr_ma_v=h1["atr_ma20"].iloc[ix]
            if dyn and not pd.isna(atr14_v) and not pd.isna(atr_ma_v):
                ch_m=ch_base-ch_adj if atr14_v>atr_ma_v else ch_base+ch_adj
            else:
                ch_m=ch_base
            
            he=ep
            for j in range(rt+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-ch_m*ca:
                    pnl=(cl[j]-ep)*bl*1 - CHG*1
                    et.append({"pnl":pnl,"et":m5["datetime"].iloc[j]})
                    break
    return pd.DataFrame(et).sort_values("et").reset_index(drop=True) if et else pd.DataFrame()

def eval_fast(c,skip=2):
    if len(c)==0: return {}
    c=c.sort_values("et").reset_index(drop=True)
    lc=0; k=np.ones(len(c),dtype=bool)
    for i in range(len(c)):
        if lc>=skip: k[i]=False; lc=0; continue
        if c["pnl"].iloc[i]<=0: lc+=1
        else: lc=0
    c=c[k].reset_index(drop=True) if skip>0 else c
    n=len(c); net=c["pnl"].sum()
    wr=(c["pnl"]>0).sum()/n*100 if n>0 else 0
    aw=c[c["pnl"]>0]["pnl"].mean() if (c["pnl"]>0).sum()>0 else 0
    al=c[c["pnl"]<0]["pnl"].mean() if (c["pnl"]<0).sum()>0 else 0
    pf=(c[c["pnl"]>0]["pnl"].sum()/abs(c[c["pnl"]<0]["pnl"].sum())) if (c["pnl"]<0).sum()!=0 else 99
    eq=c["pnl"].cumsum()+200000 if n>0 else np.array([200000])
    pk=np.maximum.accumulate(eq); mdd=(pk-eq).max()
    mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
    yrs=10
    cagr=((1+net/200000)**(1/yrs)-1)*100
    return {"net":net,"n":n,"wr":wr,"pf":pf,"mdd":mdd,"mdd_p":mdd_p,"cagr":cagr}

# Baseline
bl=run_test_fast(45,10,True,0.5)
BR=eval_fast(bl,2)
print(f"BASELINE (CH45+/-10 skip=2): Rs{BR['net']:+,.0f}  WR={BR['wr']:.1f}%  PF={BR['pf']:.2f}  MDD={BR['mdd_p']:.2f}%  CAGR={BR['cagr']:.1f}%")
BN=BR["net"]

# 1. Fixed CH multipliers
print(f"\n--- FIXED CH ---")
for ch in [15,20,25,30,35,40,45,50,55,60]:
    c=run_test_fast(ch,0,False,0.5); r=eval_fast(c,2)
    print(f"  CH{ch:>2d}:     Rs{r['net']:>+9,.0f}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  +{((r['net']/BN)-1)*100:+5.1f}%")

# 2. Dynamic CH variations
print(f"\n--- DYNAMIC CH ---")
for b,a in [(20,5),(25,8),(30,8),(35,10),(40,10),(45,10),(50,10),(55,10),(50,15),(45,15)]:
    c=run_test_fast(b,a,True,0.5); r=eval_fast(c,2)
    print(f"  CH{b}+/-{a:>2d}: Rs{r['net']:>+9,.0f}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  +{((r['net']/BN)-1)*100:+5.1f}%")

# 3. Skip loss filter
print(f"\n--- SKIP FILTER ---")
for s in [0,1,2,3,4]:
    c=run_test_fast(50,10,True,0.5); r=eval_fast(c,s)
    print(f"  Skip{s}: Rs{r['net']:>+9,.0f}  n={r['n']:4d}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  +{((r['net']/BN)-1)*100:+5.1f}%")

# 4. Body ratio filter
print(f"\n--- BODY RATIO ---")
for br in [0.2,0.3,0.5,0.75,1.0]:
    c=run_test_fast(50,10,True,br); r=eval_fast(c,2)
    print(f"  BR>{br:.2f}: Rs{r['net']:>+9,.0f}  n={r['n']:4d}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  +{((r['net']/BN)-1)*100:+5.1f}%")

# 5. Entry mechanism
print(f"\n--- ENTRY MODE ---")
for entry_desc,ch_b,ch_a in [("Retest",50,10),("Breakout",50,10)]:
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=h1_data[sym]; m5=m5_data[sym]
        sigs=[s for s in sigs_all[sym] if abs(h1["close"].iloc[s["ix"]]-h1["open"].iloc[s["ix"]])>=0.5*abs(h1["close"].iloc[s["ix"]-1]-h1["open"].iloc[s["ix"]-1])]
        sigs=[s for s in sigs if h1["close"].iloc[s["ix"]]>=h1["open"].iloc[s["ix"]-1]]
        bl=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=A(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        for sig in sigs:
            ix=sig["ix"]; lv=sig["lv"]
            tu=int(pd.to_datetime(sig["tt"]).timestamp())
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5): continue
            if entry_desc=="Breakout":
                brk=idx
                while brk<len(m5) and cl[brk]<=lv: brk+=1
                if brk>=len(m5): continue
                entry_idx=brk; ep=cl[brk]
                if entry_idx>=len(m5) or m5["datetime"].iloc[entry_idx].hour==9: continue
            else:
                brk=idx
                while brk<len(m5) and cl[brk]<=lv: brk+=1
                if brk>=len(m5): continue
                rt=brk+1
                while rt<len(m5):
                    if lo[rt]<lv and cl[rt]>lv and tc.iloc[rt]<CUTOFF: break
                    rt+=1
                if rt>=len(m5): continue
                entry_idx=rt; ep=cl[rt]
                if ep-lo[rt]<=0 or m5["datetime"].iloc[rt].hour==9: continue
            atr14_v=h1["atr14"].iloc[ix]; atr_ma_v=h1["atr_ma20"].iloc[ix]
            if not pd.isna(atr14_v) and not pd.isna(atr_ma_v):
                ch_m=ch_b-ch_a if atr14_v>atr_ma_v else ch_b+ch_a
            else: ch_m=ch_b
            he=ep
            for j in range(entry_idx+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-ch_m*ca:
                    pnl=(cl[j]-ep)*bl*1 - CHG*1
                    et.append({"pnl":pnl,"et":m5["datetime"].iloc[j]})
                    break
    c=pd.DataFrame(et).sort_values("et").reset_index(drop=True) if et else pd.DataFrame()
    r=eval_fast(c,2)
    print(f"  {entry_desc:10s}: Rs{r['net']:>+9,.0f}  n={r['n']:4d}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  +{((r['net']/BN)-1)*100:+5.1f}%")

print(f"\nDONE")
