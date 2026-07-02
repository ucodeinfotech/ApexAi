"""
Apply magnitude-based skip to ALL strategy versions we've tested.
Compares baseline vs skip for each, identifies best combined version.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Generating all trades with full context...")
trades=[]

for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean(); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    body=(h1["close"]-h1["open"]).abs(); is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    
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
        bl=50 if "NIFTY" in sym else 10
        
        # Store: entry, bl, high/low/close/atr arrays for exit simulation
        # We'll compute P&L for each CH value on the fly
        arr=[]
        for j in range(r, len(m5)):
            arr.append({"h":hi[j],"l":lo[j],"c":cl[j],"a":atr5[j]})
        
        trades.append({
            "dt":h1["datetime"].iloc[i],"sym":sym,"ep":ep,"bl":bl,
            "arr":arr,"atr14":atr14_v,"atr_ma20":atr_ma_v,
            "year":h1["datetime"].iloc[i].year
        })

print(f"Total: {len(trades)} trades")

def sim_pnl(t, ch_base, ch_range, pt=0, be=0, mh=0):
    """Fast P&L simulation."""
    atr14=t["atr14"]; atr_ma=t["atr_ma20"]
    if not pd.isna(atr14) and not pd.isna(atr_ma) and atr14>atr_ma: ch=ch_base-ch_range
    elif not pd.isna(atr14): ch=ch_base+ch_range
    else: ch=ch_base
    
    ep=t["ep"]; he=ep
    for cd in t["arr"]:
        ca=cd["a"]
        if pd.isna(ca): continue
        if cd["h"]>he: he=cd["h"]
        if cd["c"]<he-ch*ca: return (cd["c"]-ep)*t["bl"]-20
        if pt>0 and cd["h"]>=ep+pt*ca: return (pt*ca)*t["bl"]-20
    return None

# ═══════════════════════════════════
# ALL STRATEGY VERSIONS TO TEST
# ═══════════════════════════════════
strategy_versions={
    # (name, ch_base, ch_range, profit_target, breakeven, max_hold)
    "DynCH 35+10": (35, 10, 0, 0, 0),
    "DynCH 40+10": (40, 10, 0, 0, 0),
    "DynCH 45+10": (45, 10, 0, 0, 0),  # baseline
    "DynCH 50+10": (50, 10, 0, 0, 0),
    "DynCH 55+10": (55, 10, 0, 0, 0),
    "DynCH 45+5": (45, 5, 0, 0, 0),
    "DynCH 45+15": (45, 15, 0, 0, 0),
    "DynCH 40+12": (40, 12, 0, 0, 0),
    "DynCH 50+8": (50, 8, 0, 0, 0),
    "DynCH 30+15": (30, 15, 0, 0, 0),
    # With exit modifications
    "DynCH45+10 + PT1": (45, 10, 1, 0, 0),
    "DynCH45+10 + PT2": (45, 10, 2, 0, 0),
    "DynCH45+10 + BE1": (45, 10, 0, 1, 0),
    "DynCH45+10 + BE2": (45, 10, 0, 2, 0),
    "DynCH45+10 + MH24": (45, 10, 0, 0, 24),
    "DynCH45+10 + MH48": (45, 10, 0, 0, 48),
}

# Use walk-forward (train pre-2022, test 2022+)
train_trades=[t for t in trades if t["dt"]<"2022-01-01"]
test_trades=[t for t in trades if t["dt"]>="2022-01-01"]
print(f"Train: {len(train_trades)}, Test: {len(test_trades)}")

# Compute training loss stats for magnitude threshold
train_losses=[]
for t in train_trades:
    pnl=sim_pnl(t, 45, 10)  # use baseline to get loss distribution
    if pnl is not None and pnl<0:
        train_losses.append(pnl)
loss_median=np.median(train_losses) if train_losses else -8000
print(f"Train loss P50: {loss_median:+,.0f} (from {len(train_losses)} losses)")

# ═══════════════════════════════════
# TEST EACH STRATEGY WITH/WITHOUT SKIP
# ═══════════════════════════════════
print(f"\n{'='*85}")
print(f"{'Strategy':>35s} {'Base Net':>12s} {'Base N':>7s} {'Skip Net':>12s} {'Skip N':>7s} {'Impr%':>8s}")
print(f"{'='*85}")

results=[]
for name, ch_b, ch_r, pt, be, mh in strategy_versions.items():
    # Compute P&L for all test trades
    test_pnls=[]; test_wins=[]
    for t in test_trades:
        pnl=sim_pnl(t, ch_b, ch_r, pt, be, mh)
        if pnl is not None:
            test_pnls.append(pnl)
            test_wins.append(pnl>0)
    
    test_pnls=np.array(test_pnls)
    base_net=test_pnls.sum(); base_n=len(test_pnls)
    
    # Apply magnitude-based skip with evolving threshold
    # Uses only training data for initial threshold, then expands
    skip_pnls=[]
    all_losses=list(train_losses.copy())
    
    for i, p in enumerate(test_pnls):
        # Determine threshold from prior losses
        prior_losses=[x for x in all_losses if x<0]
        th=np.median(prior_losses) if len(prior_losses)>10 else loss_median
        
        # Check if prev trade was a loss AND was a small loss
        should_skip=False
        if i>0 and not test_wins[i-1] and test_pnls[i-1]<th:
            should_skip=True
        
        if not should_skip:
            skip_pnls.append(p)
            all_losses.append(p)
    
    skip_net=sum(skip_pnls); skip_n=len(skip_pnls)
    impr=(skip_net/base_net-1)*100 if base_net!=0 else 0
    results.append((impr, name, base_net, base_n, skip_net, skip_n))
    print(f"{name:>35s} Rs{base_net:>+9,.0f} {base_n:>7d} Rs{skip_net:>+9,.0f} {skip_n:>7d} {impr:>+7.1f}%")

# ═══════════════════════════════════
# RANK & REPORT
# ═══════════════════════════════════
print(f"\n{'='*85}")
print("RANKED BY SKIP NET P&L")
print(f"{'='*85}")
results.sort(key=lambda x:-x[4])
print(f"{'Rank':>4s} {'Strategy':>35s} {'Base Net':>12s} {'Skip Net':>12s} {'Impr%':>8s} {'Skip N':>7s}")
for rank,(impr, name, bn, bn_n, sn, sn_n) in enumerate(results,1):
    print(f"{rank:>4d} {name:>35s} Rs{bn:>+9,.0f} Rs{sn:>+9,.0f} {impr:>+7.1f}% {sn_n:>7d}")

# ═══════════════════════════════════
# BEST COMBINATION WALK-FORWARD
# ═══════════════════════════════════
best_name=results[0][1]
best_params=None
for name, ch_b, ch_r, pt, be, mh in strategy_versions:
    if name==best_name:
        best_params=(ch_b, ch_r, pt, be, mh)
        break

print(f"\n{'='*85}")
print(f"WALK-FORWARD: Best = {best_name} with magnitude skip")
print(f"{'='*85}")

for yr in sorted(set(t["year"] for t in test_trades)):
    yr_t=[t for t in test_trades if t["year"]==yr]
    if len(yr_t)<5: continue
    
    yr_pnls=[]
    for t in yr_t:
        p=sim_pnl(t, *best_params)
        if p is not None: yr_pnls.append(p)
    yr_pnls=np.array(yr_pnls)
    base=yr_pnls.sum()
    
    # Skip filter
    all_losses=list(train_losses.copy())
    skip_pnls=[]
    wins=[p>0 for p in yr_pnls]
    for i,p in enumerate(yr_pnls):
        pl=[x for x in all_losses if x<0]
        th=np.median(pl) if len(pl)>10 else loss_median
        
        sk=False
        if i>0 and not wins[i-1] and yr_pnls[i-1]<th:
            sk=True
        if not sk:
            skip_pnls.append(p)
            all_losses.append(p)
    
    sn=sum(skip_pnls)
    impr=(sn/base-1)*100 if base!=0 else 0
    print(f"  {yr}: Base=Rs{base:>+9,.0f} Skip=Rs{sn:>+9,.0f} ({impr:>+.1f}%) N={len(skip_pnls)}")

# Full test overall
print(f"\n  TOTAL (2022+): Base=Rs{sum(r[2] for r in results if r[1]==best_name):+,.0f} Skip=Rs{results[0][4]:+,.0f}")

print(f"\nDONE")
