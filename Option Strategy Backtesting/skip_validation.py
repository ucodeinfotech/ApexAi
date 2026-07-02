"""
Validation of best skip strategies with proper walk-forward (no look-ahead bias)
Tests: magnitude-based skip, skip-after-N, combined
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
                trades.append({"dt":h1["datetime"].iloc[i],"sym":sym,"pnl":(cl[j]-ep)*bl-20,"year":h1["datetime"].iloc[i].year})
                break

df=pd.DataFrame(trades).sort_values("dt").reset_index(drop=True)
df["win"]=df.pnl>0
print(f"Total: {len(df)} trades, Net=Rs{df.pnl.sum():+,.0f}, WR={df.win.mean()*100:.1f}%")

# Use 2015-2021 as training (to learn thresholds), 2022-2026 as test
train=df[df["dt"]<"2022-01-01"].copy()
test=df[df["dt"]>="2022-01-01"].copy()
test=test.reset_index(drop=True)
test["prev_pnl"]=test.pnl.shift(1)
test["prev_win"]=test.prev_pnl>0 if len(test)>0 else pd.Series(dtype=bool)
print(f"Train: {len(train)} (pre-2022), Test: {len(test)} (2022+)")

# Get loss stats from TRAINING data (no look-ahead)
train_losses=train[train.pnl<0].pnl
loss_median=train_losses.median()
loss_p25=train_losses.quantile(0.25)
loss_p75=train_losses.quantile(0.75)
print(f"Train loss stats: P25={loss_p25:+,.0f} P50={loss_median:+,.0f} P75={loss_p75:+,.0f}")

# ════════════════════════════════════════════
# FAIR WALK-FORWARD TEST (no look-ahead bias)
# ════════════════════════════════════════════
print(f"\n{'='*65}")
print("FAIR WALK-FORWARD (thresholds from TRAINING data ONLY)")
print(f"{'='*65}")

tests={
    "Baseline (all trades)": (None, None, None),
    "Skip 1 after 2 losses": ("count", 2, 1),
    "Skip 2 after 3 losses": ("count", 3, 2),
    "Skip 3 after 3 losses": ("count", 3, 3),
    "Skip if prev loss < P50": ("magnitude", loss_median, None),
    "Skip if prev loss < P75": ("magnitude", loss_p75, None),
    "Skip if prev loss < P25": ("magnitude", loss_p25, None),
    "Combined: skip after 3 losses OR magnitude < P50": ("combined", 3, loss_median),
}

for name, (strategy, param1, param2) in tests.items():
    test_pnls=[]; loss_streak=0; n_skipped=0; skips_by_magnitude=0; skips_by_count=0
    
    for i, r in test.iterrows():
        should_skip=False
        prev_loss = i>0 and not test.iloc[i-1].win
        prev_loss_amt = test.iloc[i-1].pnl if prev_loss else 0
        
        if strategy=="count":
            if loss_streak>=param1:
                loss_streak=0; should_skip=True; skips_by_count+=1
        
        elif strategy=="magnitude":
            if prev_loss and prev_loss_amt < param1:
                should_skip=True; skips_by_magnitude+=1
        
        elif strategy=="combined":
            # Skip if: after N losses OR (after a loss AND loss was below threshold)
            skip_count_version = loss_streak>=3
            skip_mag_version = prev_loss and prev_loss_amt < param2
            if skip_count_version or skip_mag_version:
                should_skip=True
                if skip_count_version: loss_streak=0; skips_by_count+=1
                else: skips_by_magnitude+=1
        
        if should_skip:
            continue
        
        test_pnls.append(r.pnl)
        if r.pnl>0: loss_streak=0
        else: loss_streak+=1
    
    net=sum(test_pnls); n=len(test_pnls)
    base=test.pnl.sum()
    imp=(net/base-1)*100 if base!=0 else 0
    skipped=len(test)-n
    wr=(np.array(test_pnls)>0).mean()*100 if test_pnls else 0
    print(f"  {name:>45s}: Rs{net:>+9,.0f} N={n:>4d} Skip={skipped:>3d} WR={wr:>4.1f}% {imp:>+6.1f}%")

# ════════════════════════════════════════════
# EVOLVING THRESHOLD (learn from expanding data)
# ════════════════════════════════════════════
print(f"\n{'='*65}")
print("EVOLVING THRESHOLD (expanding window — most realistic)")
print(f"{'='*65}")

# Use expanding threshold: after each trade, update the loss median
# and use it for the NEXT trade
test_pnls=[]; loss_streak=0; all_pnls=list(train.pnl.values)
for i, r in test.iterrows():
    # Compute threshold from ALL prior data
    prior_losses=[p for p in all_pnls if p<0]
    if len(prior_losses)>20:
        dyn_thresh=np.median(prior_losses)
    else:
        dyn_thresh=loss_median  # fall back to train median
    
    prev_loss = i>0 and not test.iloc[i-1].win
    prev_loss_amt = test.iloc[i-1].pnl if prev_loss else 0
    
    should_skip=False
    if prev_loss and prev_loss_amt < dyn_thresh:
        should_skip=True
    
    if not should_skip:
        test_pnls.append(r.pnl)
        all_pnls.append(r.pnl)
    
    if r.pnl>0: loss_streak=0
    else: loss_streak+=1

net=sum(test_pnls); n=len(test_pnls)
base=test.pnl.sum()
imp=(net/base-1)*100 if base!=0 else 0
print(f"  {'Evolving P50 magnitude skip':>45s}: Rs{net:>+9,.0f} N={n:>4d} Skip={len(test)-n:>3d} {imp:>+6.1f}%")

# ════════════════════════════════════════════
# YEAR-BY-YEAR WITH BEST STRATEGY
# ════════════════════════════════════════════
print(f"\n{'='*65}")
print("YEAR-BY-YEAR: Skip if prev loss < P50 (from expanding data)")
print(f"{'='*65}")

for yr in sorted(test.year.unique()):
    yr_test=test[test.year==yr].reset_index(drop=True)
    yr_pnls=[]; yr_losses=list(train[train.pnl<0].pnl.values)
    for i, r in yr_test.iterrows():
        prior_losses=[p for p in yr_losses if p<0]
        th=np.median(prior_losses) if len(prior_losses)>20 else loss_median
        
        prev_loss=i>0 and not yr_test.iloc[i-1].win
        prev_amt=yr_test.iloc[i-1].pnl if prev_loss else 0
        
        if prev_loss and prev_amt<th:
            continue
        
        yr_pnls.append(r.pnl)
        yr_losses.append(r.pnl)
    
    net=sum(yr_pnls); base=yr_test.pnl.sum()
    imp=(net/base-1)*100 if base!=0 else 0
    n=len(yr_pnls)
    print(f"  {yr}: Rs{net:>+9,.0f} vs Rs{base:>+9,.0f} ({imp:>+6.1f}%) N={n}")

# ════════════════════════════════════════════
# COMBINED: MAGNITUDE + COUNT SKIP 
# with evolving thresholds
# ════════════════════════════════════════════  
print(f"\n{'='*65}")
print("BEST COMBINED (magnitude < P50 OR skip after 3 losses)")
print("with evolving threshold + expanding window")
print(f"{'='*65}")

test_pnls=[]; all_pnls=list(train.pnl.values); loss_streak=0
for i, r in test.iterrows():
    prior_losses=[p for p in all_pnls if p<0]
    dyn_thresh=np.median(prior_losses) if len(prior_losses)>20 else loss_median
    
    prev_loss=i>0 and not test.iloc[i-1].win
    prev_amt=test.iloc[i-1].pnl if prev_loss else 0
    
    should_skip=False
    if loss_streak>=3:
        loss_streak=0; should_skip=True
    elif prev_loss and prev_amt<dyn_thresh:
        should_skip=True
    
    if should_skip: continue
    test_pnls.append(r.pnl); all_pnls.append(r.pnl)
    if r.pnl>0: loss_streak=0
    else: loss_streak+=1

net=sum(test_pnls); imp=(net/base-1)*100 if base!=0 else 0
print(f"  Combined: Rs{net:>+9,.0f} vs Rs{base:>+9,.0f} ({imp:>+6.1f}%) N={len(test_pnls)}")

# Year-by-year combined
print(f"\n  Year-by-year combined:")
for yr in sorted(test.year.unique()):
    yr_test=test[test.year==yr].reset_index(drop=True)
    yr_pnls=[]; yr_losses=list(train[train.pnl<0].pnl.values); ls=0
    for i, r in yr_test.iterrows():
        pl=[p for p in yr_losses if p<0]
        th=np.median(pl) if len(pl)>20 else loss_median
        pv=i>0 and not yr_test.iloc[i-1].win
        pa=yr_test.iloc[i-1].pnl if pv else 0
        
        sk=False
        if ls>=3: ls=0; sk=True
        elif pv and pa<th: sk=True
        
        if sk: continue
        yr_pnls.append(r.pnl); yr_losses.append(r.pnl)
        if r.pnl>0: ls=0
    else: ls+=1
    
    net=sum(yr_pnls); base=yr_test.pnl.sum()
    imp=(net/base-1)*100 if base!=0 else 0
    print(f"    {yr}: Rs{net:>+9,.0f} vs Rs{base:>+9,.0f} ({imp:>+6.1f}%) N={len(yr_pnls)}")

print(f"\nDONE")
