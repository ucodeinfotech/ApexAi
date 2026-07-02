"""
Magnitude-based skip on ALL strategy versions with proper per-version thresholds.
One-pass exit sim per entry for all versions.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"

# VERSIONS: name -> (ch_base, ch_range)
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
VN=list(VERS.keys()); CB=[VERS[v][0] for v in VN]; CR=[VERS[v][1] for v in VN]
NV=len(VN)

print("Computing entries with one-pass multi-exit...")
entries=[]

for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean(); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    body=abs(h1["close"]-h1["open"]); is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=50 if "NIFTY" in sym else 10
    du=m5["datetime"].values.astype("int64")//10**6; hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1).rolling(14,min_periods=14).mean().values
    
    for i in range(60,len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*0.5: continue
        tu=int(h1["datetime"].iloc[i].timestamp()); lv=h1["high"].iloc[i]
        idx=np.searchsorted(du,tu,side="right")
        if idx>=len(m5): continue; b=idx
        while b<len(m5) and cl[b]<=lv: b+=1
        if b>=len(m5): continue; r=b+1
        while r<len(m5):
            if lo[r]<lv and cl[r]>lv: break; r+=1
        if r>=len(m5): continue; ep=cl[r]
        if ep-lo[r]<=0: continue
        
        # Compute regime
        a14=h1["atr14"].iloc[i]; a20=h1["atr_ma20"].iloc[i]
        reg=0  # 0=norm,1=high,2=low
        if not pd.isna(a14) and not pd.isna(a20) and a14>a20: reg=1
        elif not pd.isna(a14): reg=2
        
        # One-pass exit sim for all versions
        he=ep; exits=[None]*NV
        for j in range(r, min(r+288, len(m5))):
            ca=atr5[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            for v in range(NV):
                if exits[v] is not None: continue
                cv=CB[v]-CR[v] if reg==1 else (CB[v]+CR[v] if reg==2 else CB[v])
                if cl[j]<he-cv*ca:
                    exits[v]=(cl[j]-ep)
            if all(x is not None for x in exits): break
        
        if all(x is None for x in exits): continue
        
        entries.append({
            "dt":h1["datetime"].iloc[i],"year":h1["datetime"].iloc[i].year,"bl":bl,
            "exits":exits  # list same order as VN, gross per lot
        })

print(f"Entries: {len(entries)}")
cut=pd.Timestamp("2022-01-01").tz_localize("Asia/Kolkata")
train_e=[e for e in entries if e["dt"]<cut]
test_e=[e for e in entries if e["dt"]>=cut]
print(f"Train: {len(train_e)}, Test: {len(test_e)}")

# ═══════════════════════════════════════
# EVALUATE: BASE vs SKIP for each version
# ═══════════════════════════════════════
print(f"\n{'='*110}")
print(f"{'VERSION':>40s} {'BASE Net':>12s} {'N':>5s} {'SKIP Net':>12s} {'N':>5s} {'ΔRs':>10s} {'Δ%':>7s} {'WR b→s':>12s}")
print(f"{'='*110}")

results=[]
for v in range(NV):
    # Build net P&L for this version
    base_net=[]; base_win=[]
    for e in test_e:
        g=e["exits"][v]
        if g is None: continue
        net=g*e["bl"]-20; base_net.append(net); base_win.append(net>0)
    base_net=np.array(base_net); base_n=len(base_net)
    base_total=base_net.sum(); base_wr=np.mean(base_win)*100
    
    # Train loss median for this version
    train_losses=[]
    for e in train_e:
        g=e["exits"][v]
        if g is None: continue; net=g*e["bl"]-20
        if net<0: train_losses.append(net)
    loss_med=np.median(train_losses) if len(train_losses)>5 else -5000
    
    # Skip filter
    skip_pnl=[]; prior_l=list(train_losses.copy())
    for i,e in enumerate(test_e):
        g=e["exits"][v]
        if g is None: continue; net=g*e["bl"]-20
        prior=[x for x in prior_l if x<0]
        th=np.median(prior) if len(prior)>5 else loss_med
        
        sk=(i>0 and base_net[i-1]<0 and base_net[i-1]<th)
        if not sk: skip_pnl.append(net); prior_l.append(net)
    
    sn=sum(skip_pnl); skip_n=len(skip_pnl); swr=np.mean([p>0 for p in skip_pnl])*100 if skip_n>0 else 0
    delta=sn-base_total; pct=(sn/base_total-1)*100 if base_total!=0 else 0
    
    results.append((sn,vn:=VN[v],base_total,sn,base_n,skip_n,delta,pct,base_wr,swr))
    print(f"{vn:>40s} Rs{base_total:>+9,.0f} {base_n:>5d} Rs{sn:>+9,.0f} {skip_n:>5d} Rs{delta:>+8,.0f} {pct:>+6.1f}% {base_wr:>4.1f}→{swr:>4.1f}")

# ═══════════════════════════════════════
# RANKED
# ═══════════════════════════════════════
print(f"\n{'='*110}")
print(f"{'Rank':>4s} {'VERSION':>40s} {'Base Net':>12s} {'Skip Net':>12s} {'Δ%':>7s} {'WR':>10s}")
results.sort(key=lambda x:-x[0])
for rank,(sn,vn,bn,s,bn_n,sn_n,delta,pct,bwr,swr) in enumerate(results,1):
    print(f"{rank:>4d} {vn:>40s} Rs{bn:>+9,.0f} Rs{sn:>+9,.0f} {pct:>+6.1f}% {bwr:>4.1f}→{swr:>4.1f}")

# ═══════════════════════════════════════
# WALK-FORWARD: TOP 3
# ═══════════════════════════════════════
print(f"\n{'='*110}")
print("WALK-FORWARD for Top 3")
print(f"{'='*110}")

for rank in range(min(3,len(results))):
    sn,vn,bn,s,bn_n,sn_n,delta,pct,bwr,swr=results[rank]
    v=VN.index(vn)
    print(f"\n  [{vn}] (Δ{pct:+.1f}%, {bwr:.0f}%→{swr:.0f}%)")
    print(f"  {'Year':>6s} {'Base':>10s} {'Skip':>10s} {'Δ%':>7s} {'N':>5s} {'WR':>7s}")
    
    for yr in sorted(set(e["year"] for e in test_e)):
        yr_e=[e for e in test_e if e["year"]==yr]
        # Pre-this-year training
        prior=[e for e in entries if e["year"]<yr or (e["year"]==yr and e["dt"]<cut)]
        prior_losses=[]
        for e in prior:
            g=e["exits"][v]
            if g is None: continue; net=g*e["bl"]-20
            if net<0: prior_losses.append(net)
        th=np.median(prior_losses) if len(prior_losses)>5 else (np.median(train_losses) if train_losses else -5000)
        
        yr_o=[e for e in yr_e if e["exits"][v] is not None]
        if len(yr_o)<3: continue
        
        bp=[e["exits"][v]*e["bl"]-20 for e in yr_o]
        base_yr=sum(bp); bw=[p>0 for p in bp]
        
        sp=[]; pl=prior_losses.copy()
        for i,e in enumerate(yr_o):
            net=e["exits"][v]*e["bl"]-20
            pr=[x for x in pl if x<0]; th2=np.median(pr) if len(pr)>5 else th
            sk=False
            if i>0 and bp[i-1]<0 and bp[i-1]<th2: sk=True
            if not sk: sp.append(net); pl.append(net)
        
        sn_yr=sum(sp); imp=(sn_yr/base_yr-1)*100 if base_yr!=0 else 0
        swr_yr=np.mean([p>0 for p in sp])*100 if sp else 0
        print(f"  {yr:>6d} Rs{base_yr:>+8,.0f} Rs{sn_yr:>+8,.0f} {imp:>+6.1f}% {len(sp):>5d} {bw.count(True)/len(bw)*100:>4.0f}%→{swr_yr:>4.0f}%")
    
    # Total
    print(f"  TOTAL: Rs{bn:>+9,.0f} → Rs{sn:>+9,.0f} ({pct:+.1f}%)")

print(f"\nDONE")
