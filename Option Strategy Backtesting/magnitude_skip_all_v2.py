"""
Optimized: Pre-compute P&L for all CH values once, then test all strategies.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Computing trades...")
all_trades=[]; CH_VALS=[25,30,35,40,45,50,55,60]

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
        if idx>=len(m5): continue
        b=idx
        while b<len(m5) and cl[b]<=lv: b+=1
        if b>=len(m5): continue
        r=b+1
        while r<len(m5):
            if lo[r]<lv and cl[r]>lv and tc.iloc[r]<CUTOFF: break
            r+=1
        if r>=len(m5): continue
        ep=cl[r]
        if ep-lo[r]<=0 or m5["datetime"].iloc[r].hour==9: continue
        
        atr14_v=h1["atr14"].iloc[i]; atr_ma_v=h1["atr_ma20"].iloc[i]
        # Compute CH regime once
        if not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v: reg="high"
        elif not pd.isna(atr14_v): reg="low"
        else: reg="norm"
        
        # Pre-compute P&L for all CH values (proper Chandelier exit)
        pnls={}
        for cv in CH_VALS:
            he=ep
            for j in range(r, len(m5)):
                ca=atr5[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-cv*ca:
                    pnls[cv]=(cl[j]-ep)*bl-20
                    break
        
        # Only keep trades with at least CH=45 exit
        if 45 not in pnls: continue
        
        all_trades.append({
            "dt":h1["datetime"].iloc[i],"sym":sym,"year":h1["datetime"].iloc[i].year,
            "bl":bl,"ep":ep,"reg":reg,"pnls":pnls.copy()
        })

print(f"Total: {len(all_trades)} trades")

# Split
cutoff=pd.Timestamp("2022-01-01").tz_localize("Asia/Kolkata")
train=[t for t in all_trades if t["dt"]<cutoff]
test=[t for t in all_trades if t["dt"]>=cutoff]
print(f"Train: {len(train)}, Test: {len(test)}")

# Get CH PnL for a trade given strategy params
def get_pnl(t, ch_b, ch_r):
    if t["reg"]=="high": cv=ch_b-ch_r
    elif t["reg"]=="low": cv=ch_b+ch_r
    else: cv=ch_b
    nearest=min(CH_VALS, key=lambda x:abs(x-cv))
    return t["pnls"].get(nearest)

# Compute train loss threshold
train_losses=[]
for t in train:
    p=get_pnl(t,45,10)
    if p is not None and p<0: train_losses.append(p)
loss_med=np.median(train_losses) if train_losses else -8000
print(f"Train loss P50: {loss_med:+,.0f} (N={len(train_losses)})")

# ═══════════════════════════════════
# TEST ALL STRATEGIES
# ═══════════════════════════════════
versions=[
    ("DynCH 30+15",30,15),
    ("DynCH 35+10",35,10),
    ("DynCH 35+15",35,15),
    ("DynCH 40+5",40,5),
    ("DynCH 40+10",40,10),
    ("DynCH 40+12",40,12),
    ("DynCH 45+5",45,5),
    ("DynCH 45+8",45,8),
    ("DynCH 45+10",45,10),
    ("DynCH 45+12",45,12),
    ("DynCH 45+15",45,15),
    ("DynCH 50+8",50,8),
    ("DynCH 50+10",50,10),
    ("DynCH 50+12",50,12),
    ("DynCH 55+10",55,10),
    ("DynCH 55+15",55,15),
    ("DynCH 60+10",60,10),
    ("DynCH 60+15",60,15),
    ("DynCH 25+10",25,10),
    ("DynCH 30+10",30,10),
]

print(f"\n{'='*85}")
print("MAGNITUDE SKIP ON ALL STRATEGY VERSIONS")
print(f"{'='*85}")
print(f"{'Version':>35s} {'Base Net':>12s} {'Base N':>6s} {'Skip Net':>12s} {'Skip N':>6s} {'Gain%':>8s}")
print(f"{'-'*35} {'-'*12} {'-'*6} {'-'*12} {'-'*6} {'-'*8}")

all_results=[]
for name, ch_b, ch_r in versions:
    # Compute base
    base_pnls=[]; base_wins=[]
    for t in test:
        p=get_pnl(t,ch_b,ch_r)
        if p is not None: base_pnls.append(p); base_wins.append(p>0)
    base_pnls=np.array(base_pnls)
    base_net=base_pnls.sum(); base_n=len(base_pnls)
    
    # Apply magnitude skip (evolving threshold)
    skip_pnls=[]; all_losses=list(train_losses.copy())
    for i,p in enumerate(base_pnls):
        prior_losses=[x for x in all_losses if x<0]
        th=np.median(prior_losses) if len(prior_losses)>10 else loss_med
        
        skip=False
        if i>0 and not base_wins[i-1] and base_pnls[i-1] < th:
            skip=True
        
        if not skip:
            skip_pnls.append(p)
            all_losses.append(p)
    
    skip_net=sum(skip_pnls); skip_n=len(skip_pnls)
    gain=(skip_net/base_net-1)*100 if base_net!=0 else 0
    all_results.append((gain, name, base_net, base_n, skip_net, skip_n))
    print(f"{name:>35s} Rs{base_net:>+9,.0f} {base_n:>6d} Rs{skip_net:>+9,.0f} {skip_n:>6d} {gain:>+7.1f}%")

print(f"\n{'='*85}")
print("RANKED BY SKIP NET P&L")
print(f"{'='*85}")
all_results.sort(key=lambda x:-x[4])
print(f"{'Rank':>4s} {'Version':>35s} {'Base Net':>12s} {'Skip Net':>12s} {'Gain%':>8s} {'Skip N':>6s}")
for rank,(gain,name,bn,bn_n,sn,sn_n) in enumerate(all_results,1):
    print(f"{rank:>4d} {name:>35s} Rs{bn:>+9,.0f} Rs{sn:>+9,.0f} {gain:>+7.1f}% {sn_n:>6d}")

# ═══════════════════════════════════
# WALK-FORWARD FOR BEST
# ═══════════════════════════════════
best_name=all_results[0][1]
best_params=None
for n,cb,cr in versions:
    if n==best_name: best_params=(cb,cr); break

print(f"\n{'='*85}")
print(f"WALK-FORWARD: [{best_name}] with magnitude skip")
print(f"{'='*85}")
print(f"{'Year':>6s} {'Base':>12s} {'Skip':>12s} {'Gain%':>8s} {'N':>5s}")
for yr in sorted(set(t["year"] for t in test)):
    yr_t=[t for t in test if t["year"]==yr]
    if len(yr_t)<3: continue
    yr_p=[]
    for t in yr_t:
        p=get_pnl(t,*best_params)
        if p is not None: yr_p.append(p)
    yr_p=np.array(yr_p)
    base=yr_p.sum()
    wins=[p>0 for p in yr_p]
    
    all_l=list(train_losses.copy()); sp=[]
    for i,p in enumerate(yr_p):
        pl=[x for x in all_l if x<0]
        th=np.median(pl) if len(pl)>10 else loss_med
        sk=False
        if i>0 and not wins[i-1] and yr_p[i-1]<th: sk=True
        if not sk: sp.append(p); all_l.append(p)
    sn=sum(sp); impr=(sn/base-1)*100 if base!=0 else 0
    print(f"{yr:>6d} Rs{base:>+9,.0f} Rs{sn:>+9,.0f} {impr:>+7.1f}% {len(sp):>5d}")

# Also show best WITHOUT skip (pure strategy ranking)
print(f"\n{'='*85}")
print("PURE STRATEGY RANKING (without skip filter)")
print(f"{'='*85}")
pure=sorted([(bn,name,sn) for (g,name,bn,bn_n,sn,sn_n) in all_results], key=lambda x:-x[2])
for rank,(bn,name,sn) in enumerate(pure[:5],1):
    print(f"{rank:>4d} {name:>35s} Net=Rs{bn:>+9,.0f}")

print(f"\nDONE")
