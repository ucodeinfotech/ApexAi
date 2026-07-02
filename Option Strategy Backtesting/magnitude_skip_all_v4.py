"""
Magnitude-skip on ALL strategy versions with per-version threshold.
Based on proven v2 trade detection.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# VERSIONS
VERS = {
    "DynCH 25+10":(25,10),"DynCH 30+10":(30,10),"DynCH 30+15":(30,15),
    "DynCH 35+10":(35,10),"DynCH 35+15":(35,15),
    "DynCH 40+5":(40,5),"DynCH 40+10":(40,10),"DynCH 40+12":(40,12),
    "DynCH 45+5":(45,5),"DynCH 45+8":(45,8),"DynCH 45+10":(45,10),
    "DynCH 45+12":(45,12),"DynCH 45+15":(45,15),
    "DynCH 50+8":(50,8),"DynCH 50+10":(50,10),"DynCH 50+12":(50,12),
    "DynCH 55+10":(55,10),"DynCH 55+15":(55,15),
    "DynCH 60+10":(60,10),"DynCH 60+15":(60,15),
}
VN=list(VERS.keys()); CB=[VERS[v][0] for v in VN]; CR=[VERS[v][1] for v in VN]; NV=len(VN)
CH_VALS=list(set(CB))  # unique base CH values

print("Computing trades (slow per-entry method)...")
all_trades=[]; entries_raw=[]

for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean(); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    body=(h1["close"]-h1["open"]).abs(); is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=50 if "NIFTY" in sym else 10
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; tc=m5["datetime"].dt.time
    
    for i in range(60,len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*0.5: continue
        tu=int(pd.to_datetime(h1["datetime"].iloc[i]).timestamp()); lv=h1["high"].iloc[i]
        idx=np.searchsorted(du,tu,side="right")
        if idx>=len(m5): continue; b=idx
        while b<len(m5) and cl[b]<=lv: b+=1
        if b>=len(m5): continue; r=b+1
        while r<len(m5):
            if lo[r]<lv and cl[r]>lv and tc.iloc[r]<CUTOFF: break
            r+=1
        if r>=len(m5): continue; ep=cl[r]
        if ep-lo[r]<=0 or m5["datetime"].iloc[r].hour==9: continue
        
        # Regime
        a14=h1["atr14"].iloc[i]; a20=h1["atr_ma20"].iloc[i]
        reg=0
        if not pd.isna(a14) and not pd.isna(a20) and a14>a20: reg=1
        elif not pd.isna(a14): reg=2
        
        # Pre-compute P&L for all CH values (proper Chandelier exit)
        pnls={}
        for cv in CH_VALS:
            he=ep
            for j in range(r, min(r+2000, len(m5))):
                ca=atr5[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-cv*ca:
                    pnls[cv]=(cl[j]-ep)*bl-20
                    break
        
        if 45 not in pnls: continue
        all_trades.append({
            "dt":h1["datetime"].iloc[i],"sym":sym,"year":h1["datetime"].iloc[i].year,
            "bl":bl,"ep":ep,"reg":reg,"pnls":pnls.copy()
        })

print(f"Total: {len(all_trades)} trades")
cut=pd.Timestamp("2022-01-01").tz_localize("Asia/Kolkata")
train=[t for t in all_trades if t["dt"]<cut]
test=[t for t in all_trades if t["dt"]>=cut]
print(f"Train: {len(train)}, Test: {len(test)}")

# Get P&L for a given CH value
def get_pnl(t, ch_b, ch_r):
    if t["reg"]==1: cv=ch_b-ch_r
    elif t["reg"]==2: cv=ch_b+ch_r
    else: cv=ch_b
    nearest=min(CH_VALS, key=lambda x:abs(x-cv))
    return t["pnls"].get(nearest)

# ═══════════════════════════════════════
# EVALUATE ALL VERSIONS
# ═══════════════════════════════════════
print(f"\n{'='*95}")
print(f"{'VERSION':>40s} {'BASE Net':>12s} {'N':>5s} {'SKIP Net':>12s} {'N':>5s} {'ΔRs':>10s} {'Δ%':>7s}")
print(f"{'='*95}")

results=[]
for v in range(NV):
    vn=VN[v]; cb=CB[v]; cr=CR[v]
    
    # Build test PnL
    base_pnls=[]; base_wins=[]
    for t in test:
        p=get_pnl(t,cb,cr)
        if p is not None: base_pnls.append(p); base_wins.append(p>0)
    base_pnls=np.array(base_pnls)
    base_net=base_pnls.sum(); base_n=len(base_pnls)
    
    # Train loss median (per-version)
    train_losses=[]
    for t in train:
        p=get_pnl(t,cb,cr)
        if p is not None and p<0: train_losses.append(p)
    loss_med=np.median(train_losses) if len(train_losses)>5 else -5000
    
    # Skip
    skip_pnls=[]; prior_l=list(train_losses.copy())
    for i,p in enumerate(base_pnls):
        prior=[x for x in prior_l if x<0]
        th=np.median(prior) if len(prior)>5 else loss_med
        
        sk=(i>0 and not base_wins[i-1] and base_pnls[i-1] < th)
        if not sk: skip_pnls.append(p); prior_l.append(p)
    
    sn=sum(skip_pnls); skip_n=len(skip_pnls)
    delta=sn-base_net; pct=(sn/base_net-1)*100 if base_net!=0 else 0
    results.append((sn, vn, base_net, sn, base_n, skip_n, delta, pct))
    print(f"{vn:>40s} Rs{base_net:>+9,.0f} {base_n:>5d} Rs{sn:>+9,.0f} {skip_n:>5d} Rs{delta:>+8,.0f} {pct:>+7.1f}%")

# ═══════════════════════════════════════
# RANKED
# ═══════════════════════════════════════
print(f"\n{'='*95}")
print(f"{'Rank':>4s} {'VERSION':>40s} {'Base Net':>12s} {'Skip Net':>12s} {'Δ%':>7s}")
results.sort(key=lambda x:-x[0])
for rank,(sn,vn,bn,s,bn_n,sn_n,delta,pct) in enumerate(results,1):
    print(f"{rank:>4d} {vn:>40s} Rs{bn:>+9,.0f} Rs{sn:>+9,.0f} {pct:>+7.1f}%")

# ═══════════════════════════════════════
# WALK-FORWARD TOP 3
# ═══════════════════════════════════════
print(f"\n{'='*95}")
print("WALK-FORWARD for Top 3")
print(f"{'='*95}")

for rank in range(min(3,len(results))):
    sn,vn,bn,s,bn_n,sn_n,delta,pct=results[rank]
    v=VN.index(vn); cb=CB[v]; cr=CR[v]
    
    print(f"\n  [{vn}]")
    print(f"  {'Year':>6s} {'Base':>10s} {'Skip':>10s} {'Δ%':>7s} {'N':>5s}")
    
    for yr in sorted(set(t["year"] for t in test)):
        yr_t=[t for t in test if t["year"]==yr]
        yr_p=[]
        for t in yr_t:
            p=get_pnl(t,cb,cr)
            if p is not None: yr_p.append(p)
        yr_p=np.array(yr_p)
        if len(yr_p)<3: continue
        base_yr=yr_p.sum(); wins=[p>0 for p in yr_p]
        
        # Pre-this-year training
        prior=[t for t in train if t["year"]<yr]
        prior_losses=[]
        for t in prior:
            p=get_pnl(t,cb,cr)
            if p is not None and p<0: prior_losses.append(p)
        th=np.median(prior_losses) if len(prior_losses)>5 else -5000
        
        sp=[]; pl=prior_losses.copy()
        for i,p in enumerate(yr_p):
            pr=[x for x in pl if x<0]; th2=np.median(pr) if len(pr)>5 else th
            sk=False
            if i>0 and not wins[i-1] and yr_p[i-1]<th2: sk=True
            if not sk: sp.append(p); pl.append(p)
        
        sn_yr=sum(sp); imp=(sn_yr/base_yr-1)*100 if base_yr!=0 else 0
        print(f"  {yr:>6d} Rs{base_yr:>+8,.0f} Rs{sn_yr:>+8,.0f} {imp:>+6.1f}% {len(sp):>5d}")
    
    print(f"  {'TOTAL':>6s} Rs{bn:>+8,.0f} Rs{sn:>+8,.0f} {pct:>+6.1f}% {sn_n:>5d}")

print(f"\nDONE")
