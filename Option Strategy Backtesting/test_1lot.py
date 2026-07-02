"""
Test DynCH45±10 with fixed 1 lot (no anti-martingale)
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20; CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

CH_BASE=45; CH_ADJ=10
et=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
    h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*0.5: continue
        sigs.append({"tt":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"ix":i})
    bl=NLOT if "NIFTY" in sym else SLOT
    tc=m5["datetime"].dt.time; atr5=A(m5,14)
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    for sig in sigs:
        ix=sig["ix"]
        tu=int(pd.to_datetime(sig["tt"]).timestamp()); lv=sig["lv"]
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
        he=ep
        atr14_v=h1["atr14"].iloc[ix]; atr_ma_v=h1["atr_ma20"].iloc[ix]
        ch_m=CH_BASE
        if not pd.isna(atr14_v) and not pd.isna(atr_ma_v):
            if atr14_v>atr_ma_v: ch_m=CH_BASE-CH_ADJ
            else: ch_m=CH_BASE+CH_ADJ
        for j in range(rt+1,len(m5)):
            ca=atr5.iloc[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch_m*ca:
                pnl=(cl[j]-ep)*bl*1 - CHG*1
                et.append({"pnl":pnl,"et":m5["datetime"].iloc[j],"sym":sym})
                break

mt=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
    atr=A(h1,14); hi20=h1["high"].rolling(20).max().shift(1)
    lot=NLOT if "NIFTY" in sym else SLOT
    it=False; ep2=None; he2=None; etm2=None
    for i in range(20,len(h1)):
        if not it:
            if h1["close"].iloc[i]>hi20.iloc[i] and h1["close"].iloc[i]>h1["open"].iloc[i] and h1["datetime"].iloc[i].time()<CUTOFF and h1["datetime"].iloc[i].hour>=9:
                it=True; ep2=float(h1["close"].iloc[i]); he2=ep2; etm2=h1["datetime"].iloc[i]
        else:
            if h1["high"].iloc[i]>he2: he2=h1["high"].iloc[i]
            ca=atr.iloc[i]; ex=False
            if not pd.isna(ca) and h1["close"].iloc[i]<he2-10*ca: ex=True
            if 24>0 and not ex and etm2 is not None:
                hr=(h1["datetime"].iloc[i]-etm2).total_seconds()/3600
                if hr>24 and h1["close"].iloc[i]<=ep2: ex=True
            if ex:
                mt.append({"pnl":(h1["close"].iloc[i]-ep2)*lot-CHG,"et":h1["datetime"].iloc[i]})
                it=False

c=pd.concat([pd.DataFrame(et),pd.DataFrame(mt)],ignore_index=True)
c=c.sort_values("et").reset_index(drop=True)

# Without skip filter
c_nf=c.reset_index(drop=True)
n=len(c_nf); net=c_nf["pnl"].sum()
wr=(c_nf["pnl"]>0).sum()/n*100 if n>0 else 0
aw=c_nf[c_nf["pnl"]>0]["pnl"].mean() if (c_nf["pnl"]>0).sum()>0 else 0
al=c_nf[c_nf["pnl"]<0]["pnl"].mean() if (c_nf["pnl"]<0).sum()>0 else 0
pf=(c_nf[c_nf["pnl"]>0]["pnl"].sum()/abs(c_nf[c_nf["pnl"]<0]["pnl"].sum())) if (c_nf["pnl"]<0).sum()!=0 else 99
eq=c_nf["pnl"].cumsum()+200000 if n>0 else np.array([200000])
pk=np.maximum.accumulate(eq); mdd=(pk-eq).max()
mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
yrs=(c_nf["et"].max()-c_nf["et"].min()).total_seconds()/31536000 if len(c_nf)>1 else 1
cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0

# With skip filter
lc=0; k=np.ones(len(c),dtype=bool)
for i in range(len(c)):
    if lc>=2: k[i]=False; lc=0; continue
    if c["pnl"].iloc[i]<=0: lc+=1
    else: lc=0
c_f=c[k].reset_index(drop=True)

nf=len(c_f); nft=c_f["pnl"].sum()
wr_f=(c_f["pnl"]>0).sum()/nf*100 if nf>0 else 0
aw_f=c_f[c_f["pnl"]>0]["pnl"].mean() if (c_f["pnl"]>0).sum()>0 else 0
al_f=c_f[c_f["pnl"]<0]["pnl"].mean() if (c_f["pnl"]<0).sum()>0 else 0
pf_f=(c_f[c_f["pnl"]>0]["pnl"].sum()/abs(c_f[c_f["pnl"]<0]["pnl"].sum())) if (c_f["pnl"]<0).sum()!=0 else 99
eq_f=c_f["pnl"].cumsum()+200000 if nf>0 else np.array([200000])
pk_f=np.maximum.accumulate(eq_f); mdd_f=(pk_f-eq_f).max()
mdd_pf=mdd_f/pk_f.max()*100 if pk_f.max()>0 else 0
yrs_f=(c_f["et"].max()-c_f["et"].min()).total_seconds()/31536000 if len(c_f)>1 else 1
cagr_f=((1+nft/200000)**(1/yrs_f)-1)*100 if yrs_f>0 else 0

print("="*70)
print(" 1-LOT ONLY: DynCH45+/-10")
print("="*70)
print(f"")
print(f"  {'Metric':25s}  {'No Skip Filter':>18s}  {'With Skip':>18s}")
print(f"  {'-'*25}  {'-'*18}  {'-'*18}")
print(f"  {'Net P&L':25s}  Rs{nft:>+14,.0f}  Rs{net:>+14,.0f}")
print(f"  {'Trades':25s}  {nf:>18d}  {n:>18d}")
print(f"  {'Win Rate':25s}  {wr_f:>17.1f}%  {wr:>17.1f}%")
print(f"  {'Profit Factor':25s}  {pf_f:>17.2f}  {pf:>17.2f}")
print(f"  {'Avg Win':25s}  Rs{aw_f:>+13,.0f}  Rs{aw:>+13,.0f}")
print(f"  {'Avg Loss':25s}  Rs{al_f:>+13,.0f}  Rs{al:>+13,.0f}")
print(f"  {'Max DD (Rs)':25s}  Rs{mdd_f:>+12,.0f}  Rs{mdd:>+12,.0f}")
print(f"  {'Max DD (%)':25s}  {mdd_pf:>17.2f}%  {mdd_p:>17.2f}%")
print(f"  {'CAGR':25s}  {cagr_f:>17.1f}%  {cagr:>17.1f}%")
print(f"  {'Net/MDD':25s}  {nft/mdd_f:>17.1f}x  {net/mdd:>17.1f}x")

# Engulfing only stats
eng_df=c_f[~c_f["sym"].isna()] if "sym" in c_f.columns else c_f
eng_pnl=eng_df["pnl"]
eng_n=len(eng_pnl)
eng_net=eng_pnl.sum()
eng_wr=(eng_pnl>0).sum()/eng_n*100 if eng_n>0 else 0
eng_pf=(eng_pnl[eng_pnl>0].sum()/abs(eng_pnl[eng_pnl<0].sum())) if (eng_pnl<0).sum()!=0 else 99

print(f"")
print(f"  ENGULFING ONLY (1-lot, with skip):")
print(f"    Trades: {eng_n}, Net: Rs{eng_net:+,.0f}, WR: {eng_wr:.1f}%, PF: {eng_pf:.2f}")

# Comparison
print(f"")
print(f"  {'='*60}")
print(f"  COMPARISON with Anti-Martingale 1w1l:")
print(f"  {'='*60}")
print(f"  {'Version':30s}  {'Net P&L':>12s}  {'CAGR':>6s}  {'MDD%':>6s}")
print(f"  {'-'*60}")
print(f"  {'1-LOT (this test)':30s}  Rs{nft:>+10,.0f}  {cagr_f:>5.1f}%  {mdd_pf:>5.2f}%")
print(f"  {'1w1l Anti-Martingale':30s}  Rs+46,514,715  61.7%  1.96%")
print(f"  {'AM boost factor':30s}  {46514715/nft:>10.1f}x  {(61.7/cagr_f if cagr_f>0 else 0):>5.1f}x")
print(f"")
print("DONE")
