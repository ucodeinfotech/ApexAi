"""
Smart skip-after-loss testing — multiple adaptive strategies
Based on autocorrelation finding: WR=23.6% after loss, 19.9% after 2 losses
Tests: skip count, skip type, skip duration, magnitude-based
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Generating trades...")
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
        ch=35 if (not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v) else 55 if (not pd.isna(atr14_v)) else 45
        he=ep
        for j in range(r+1,len(m5)):
            ca=atr5[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch*ca:
                pnl=(cl[j]-ep)*bl*1-20*1
                trades.append({"dt":h1["datetime"].iloc[i],"sym":sym,"pnl":pnl,"win":1 if pnl>0 else 0,"year":h1["datetime"].iloc[i].year})
                break

df=pd.DataFrame(trades)
print(f"Total: {len(df)} trades, Net=Rs{df.pnl.sum():+,.0f}, WR={df.win.mean()*100:.1f}%")

# Chronological split
df=df.sort_values("dt").reset_index(drop=True)
split=int(len(df)*0.7)
train=df.iloc[:split]; test=df.iloc[split:]
print(f"Train: {len(train)}, Test: {len(test)}, Test Net=Rs{test.pnl.sum():+,.0f}")

# ═══════════════════════════════════════════════════
# ALL SKIP STRATEGIES
# ═══════════════════════════════════════════════════
print(f"\n{'='*65}")
print("SKIP STRATEGY COMPARISON (TEST SET ONLY)")
print(f"{'='*65}")

strategies={
    # (name, skip_after_n_losses, skip_count, additional_logic)
    "Baseline (no skip)": (999, 0, "none"),
    "Skip 1 after 1 loss": (1, 1, "none"),
    "Skip 1 after 2 losses": (2, 1, "none"),
    "Skip 2 after 2 losses": (2, 2, "none"),
    "Skip 3 after 2 losses": (2, 3, "none"),
    "Skip 1 after 3 losses": (3, 1, "none"),
    "Skip 2 after 3 losses": (3, 2, "none"),
    "Skip adaptive (N after N)": (1, -1, "adaptive"),  # skip N after N losses
    "Skip 1 after big loss": (1, 1, "big_loss"),  # only skip if loss > avg_loss
    "Skip 1 after small loss": (1, 1, "small_loss"),  # only skip if loss < avg_loss
}

results=[]
for name, (skip_after, skip_count, logic) in strategies.items():
    test_pnls=[]; loss_streak=0; n_skipped=0
    
    for _,r in test.iterrows():
        # Determine if we skip
        should_skip=False
        
        if logic=="adaptive":
            # Skip N trades after N consecutive losses
            if loss_streak>=1:
                skips_needed=loss_streak
                if n_skipped<skips_needed:
                    should_skip=True
                    n_skipped+=1
                    if n_skipped>=skips_needed:
                        loss_streak=0; n_skipped=0
        elif logic=="big_loss":
            # Skip 1 after a loss, but only if the loss was bigger than average
            if loss_streak>=skip_after:
                # Check last P&L (which caused the streak)
                pass  # we'd need to store last_pnl
        elif logic=="small_loss":
            pass
        else:
            if loss_streak>=skip_after:
                # Skip skip_count trades
                if n_skipped<skip_count:
                    should_skip=True
                    n_skipped+=1
                    if n_skipped>=skip_count:
                        loss_streak=0; n_skipped=0
        
        if should_skip:
            continue
        
        # Take trade
        test_pnls.append(r.pnl)
        if r.win:
            loss_streak=0
        else:
            loss_streak+=1
    
    net=sum(test_pnls)
    wr=(np.array(test_pnls)>0).mean()*100 if test_pnls else 0
    n=len(test_pnls)
    skipped=len(test)-n
    improvement=(net/test.pnl.sum()-1)*100 if test.pnl.sum()!=0 else 0
    results.append((name, net, n, skipped, wr, improvement))

results.sort(key=lambda x:-x[1])
print(f"\n{'Strategy':>32s} {'Net':>12s} {'N':>5s} {'Skip':>5s} {'WR%':>6s} {'vsBase':>8s}")
print(f"{'-'*32} {'-'*12} {'-'*5} {'-'*5} {'-'*6} {'-'*8}")
for name, net, n, skipped, wr, imp in results:
    print(f"{name:>32s} Rs{net:>+9,.0f} {n:>5d} {skipped:>5d} {wr:>5.1f}% {imp:>+7.1f}%")

# ═══════════════════════════════════════════════════
# REVERSE STRATEGY: SKIP AFTER WINS
# ═══════════════════════════════════════════════════
print(f"\n{'='*65}")
print("REVERSE STRATEGY: Skip AFTER WINS (exploring autocorrelation)")
print(f"{'='*65}")

rev_strategies={
    "Baseline (no skip)": (999, 0),
    "Skip 1 after win": (1, 1),
    "Skip 2 after win": (1, 2),
    "Skip 3 after win": (1, 3),
}

for name, skip_after, skip_count in rev_strategies.items():
    test_pnls=[]; win_streak=0; n_skipped=0
    for _,r in test.iterrows():
        should_skip=False
        if win_streak>=skip_after:
            if n_skipped<skip_count:
                should_skip=True; n_skipped+=1
                if n_skipped>=skip_count:
                    win_streak=0; n_skipped=0
        if should_skip: continue
        test_pnls.append(r.pnl)
        if r.win: win_streak+=1
        else: win_streak=0
    
    net=sum(test_pnls)
    wr=(np.array(test_pnls)>0).mean()*100 if test_pnls else 0
    n=len(test_pnls)
    imp=(net/test.pnl.sum()-1)*100 if test.pnl.sum()!=0 else 0
    print(f"  {name:>30s}: Net=Rs{net:>+9,.0f} N={n:>4d} WR={wr:>4.1f}% {imp:>+7.1f}%")

# ═══════════════════════════════════════════════════
# MAGNITUDE-BASED SKIP: skip based on loss size
# ═══════════════════════════════════════════════════
print(f"\n{'='*65}")
print("MAGNITUDE-BASED SKIP (loss size matters?)")
print(f"{'='*65}")

# First, check if loss magnitude predicts next trade outcome
test["prev_pnl"]=test.pnl.shift(1)
test["prev_win"]=test.win.shift(1)

# After losses, split by loss size
big_losses=test[(test.prev_win==0)&(test.prev_pnl<test.prev_pnl.median())]
small_losses=test[(test.prev_win==0)&(test.prev_pnl>=test.prev_pnl.median())]

print(f"  After big loss (below median):  N={len(big_losses)} WR={big_losses.win.mean()*100:.1f}% Avg=Rs{big_losses.pnl.mean():+,.0f}")
print(f"  After small loss (above median): N={len(small_losses)} WR={small_losses.win.mean()*100:.1f}% Avg=Rs{small_losses.pnl.mean():+,.0f}")
print(f"  After win: N={len(test[test.prev_win==1])} WR={test[test.prev_win==1].win.mean()*100:.1f}% Avg=Rs{test[test.prev_win==1].pnl.mean():+,.0f}")

# Test: skip only after BIG losses
for loss_pctile in [10, 25, 50, 75, 90]:
    thresh=test[test.prev_win==0].prev_pnl.quantile(loss_pctile/100)
    test_pnls=[]; skip_mode=False
    for _,r in test.iterrows():
        should_skip=False
        if not r.prev_win and r.prev_pnl<thresh:
            should_skip=True
        if should_skip: continue
        test_pnls.append(r.pnl)
    
    net=sum(test_pnls); n=len(test_pnls)
    imp=(net/test.pnl.sum()-1)*100
    wr=(np.array(test_pnls)>0).mean()*100
    skipped=len(test)-n
    print(f"  Skip after loss < P{loss_pctile} ({thresh:+,.0f}): Net=Rs{net:>+,.0f} N={n} Skip={skipped} WR={wr:.1f}% Imp={imp:+.1f}%")

# ═══════════════════════════════════════════════════
# OPTIMAL SKIP: check ALL combinations of skip settings
# ═══════════════════════════════════════════════════
print(f"\n{'='*65}")
print("GRID SEARCH: ALL (skip_after, skip_count) combinations")
print(f"{'='*65}")

best=[]
for skip_after in range(0,6):
    for skip_count in range(0,6):
        if skip_after==0 and skip_count>0: continue
        if skip_after>0 and skip_count==0: continue
        test_pnls=[]; loss_streak=0; n_skipped=0
        for _,r in test.iterrows():
            should_skip=False
            if loss_streak>=skip_after and skip_count>0:
                if n_skipped<skip_count:
                    should_skip=True; n_skipped+=1
                    if n_skipped>=skip_count:
                        loss_streak=0; n_skipped=0
            if should_skip: continue
            test_pnls.append(r.pnl)
            if not r.win: loss_streak+=1
            else: loss_streak=0
        
        net=sum(test_pnls); n=len(test_pnls)
        imp=(net/test.pnl.sum()-1)*100 if test.pnl.sum()!=0 else 0
        best.append((imp, net, n, skip_after, skip_count))

best.sort(key=lambda x:-x[0])
print(f"\n{'SkipAfter':>10s} {'SkipCount':>10s} {'Net':>12s} {'N':>6s} {'Imp%':>8s}")
for imp, net, n, sa, sc in best[:15]:
    print(f"{sa:>10d} {sc:>10d} Rs{net:>+9,.0f} {n:>6d} {imp:>+7.1f}%")

# ═══════════════════════════════════════════════════
# WALK-FORWARD OF BEST SKIP
# ═══════════════════════════════════════════════════
if best:
    best_sa, best_sc = best[0][3], best[0][4]
    print(f"\n{'='*65}")
    print(f"WALK-FORWARD: Best skip (after {best_sa} losses, skip {best_sc})")
    print(f"{'='*65}")
    
    for yr in sorted(test.year.unique()):
        yr_t=test[test.year==yr]
        if len(yr_t)<5: continue
        yr_pnls=[]; ls=0; ns=0
        for _,r in yr_t.iterrows():
            should_skip=False
            if ls>=best_sa and best_sc>0:
                if ns<best_sc:
                    should_skip=True; ns+=1
                    if ns>=best_sc: ls=0; ns=0
            if should_skip: continue
            yr_pnls.append(r.pnl)
            if not r.win: ls+=1
            else: ls=0
        net=sum(yr_pnls); base=yr_t.pnl.sum()
        imp=(net/base-1)*100 if base!=0 else 0
        print(f"  {yr}: Rs{net:>+9,.0f} vs Rs{base:>+9,.0f} ({imp:>+7.1f}%) N={len(yr_pnls)}")

print(f"\nDONE")
