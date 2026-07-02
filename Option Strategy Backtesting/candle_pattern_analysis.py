"""
Deep dive: what do the engulfing candles and signals actually look like?
Analyses all 1,218 trades for patterns in the candle data itself.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()
RS=42; np.random.seed(RS)

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Loading data...")
h1d={}; m5d={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean()
    h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["ema10"]=h1["close"].ewm(span=10).mean()
    h1["ema50"]=h1["close"].ewm(span=50).mean()
    h1["ema200"]=h1["close"].ewm(span=200).mean()
    h1d[sym]=h1; m5d[sym]=m5

# Collect ALL trade data
records = []
for sym in ["NIFTY50","SENSEX"]:
    h1=h1d[sym]; m5=m5d[sym]
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=50 if "NIFTY" in sym else 10
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; tc=m5["datetime"].dt.time

    for i in range(60, len(h1)):
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
                rec={
                    "sym":sym, "dt":h1["datetime"].iloc[i],
                    # ── SIGNAL CANDLE ──
                    "s_open":h1["open"].iloc[i], "s_high":h1["high"].iloc[i],
                    "s_low":h1["low"].iloc[i], "s_close":h1["close"].iloc[i],
                    "s_body":body.iloc[i], "s_range":h1["high"].iloc[i]-h1["low"].iloc[i],
                    "s_vol":h1["volume"].iloc[i],
                    # ── PRIOR CANDLE (red) ──
                    "p_open":h1["open"].iloc[i-1], "p_close":h1["close"].iloc[i-1],
                    "p_body":body.iloc[i-1], "p_range":h1["high"].iloc[i-1]-h1["low"].iloc[i-1],
                    "p_vol":h1["volume"].iloc[i-1],
                    # ── COMPARISONS ──
                    "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
                    "body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
                    "vol_ratio":h1["volume"].iloc[i]/(h1["volume"].iloc[i-1]+1e-10),
                    # ── CONTEXT ──
                    "hour":h1["datetime"].iloc[i].hour,
                    "dow":h1["datetime"].iloc[i].dayofweek,
                    "month":h1["datetime"].iloc[i].month,
                    "year":h1["datetime"].iloc[i].year,
                    "c_ema50":(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100,
                    "c_ema200":(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100,
                    "atr_pct":atr14_v/(h1["close"].iloc[i]+1e-10)*100 if not pd.isna(atr14_v) else 0,
                    "ch20_ret":(h1["close"].iloc[i]/h1["close"].iloc[i-20]-1)*100 if i>=20 else 0,
                    "ch5_ret":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0,
                    # ── OUTCOME ──
                    "pnl":pnl,
                    "win":1 if pnl>0 else 0,
                    # ── 5 CANDLES BEFORE ──
                    "trend_5":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0,
                    "trend_10":(h1["close"].iloc[i]/h1["close"].iloc[i-10]-1)*100 if i>=10 else 0,
                    "avg_body_5":body.iloc[i-5:i].mean() if i>=5 else 0,
                    "vol_ma20":h1["volume"].iloc[i]/(h1["volume"].iloc[i-20:i].mean()+1e-10),
                }
                # Prev 5 candles: how many were green?
                greens=0
                for k in range(1,6):
                    if i-k>=0 and h1["close"].iloc[i-k]>h1["open"].iloc[i-k]: greens+=1
                rec["green_prev5"]=greens
                rec["red_prev5"]=5-greens
                records.append(rec)
                break

df=pd.DataFrame(records)
print(f"\nTotal trades: {len(df)}")
print(f"Winners: {df.win.sum()} ({df.win.mean()*100:.1f}%)")
print(f"Net P&L: Rs{df.pnl.sum():+,.0f}")

# ══════════════════════════════════════════════
# 1. SIGNAL CANDLE CHARACTERISTICS
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("1. SIGNAL CANDLE CHARACTERISTICS")
print(f"{'='*70}")

print(f"\n  --- Body Size Distribution ---")
for p in [10,25,50,75,90]:
    print(f"  P{p}: {df.s_body.quantile(p/100):.0f} pts (avg {df.s_body.mean():.0f})")

print(f"\n  --- Range Distribution ---")
for p in [10,25,50,75,90]:
    print(f"  P{p}: {df.s_range.quantile(p/100):.0f} pts")

print(f"\n  --- Body % of Range ---")
df["body_pct_range"]=df.s_body/df.s_range*100
print(f"  Mean: {df.body_pct_range.mean():.1f}%")
print(f"  Median: {df.body_pct_range.median():.1f}%")

print(f"\n  --- Gap from prior close to open ---")
print(f"  Gap%: mean={df.gap_pct.mean():+.3f}%, range=[{df.gap_pct.min():+.2f}, {df.gap_pct.max():+.2f}]")
print(f"  Positive gap: {(df.gap_pct>0).sum()}/{len(df)} ({(df.gap_pct>0).mean()*100:.1f}%)")
print(f"  Negative gap: {(df.gap_pct<0).sum()}/{len(df)} ({(df.gap_pct<0).mean()*100:.1f}%)")

print(f"\n  --- Volume Ratio (signal vs prior) ---")
print(f"  Mean: {df.vol_ratio.mean():.2f}x")
print(f"  Median: {df.vol_ratio.median():.2f}x")

print(f"\n  --- Prior red candle body ---")
print(f"  Mean body: {df.p_body.mean():.0f} pts")
print(f"  Body ratio (signal/prior): mean={df.body_ratio.mean():.1f}x, median={df.body_ratio.median():.1f}x")

# ══════════════════════════════════════════════
# 2. MARKET CONTEXT WHEN SIGNAL OCCURS
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("2. MARKET CONTEXT AT SIGNAL TIME")
print(f"{'='*70}")

print(f"\n  --- Hour Distribution ---")
for h in sorted(df.hour.unique()):
    subset=df[df.hour==h]
    print(f"  {h:02d}:00: {len(subset)} trades ({len(subset)/len(df)*100:.1f}%), WR={subset.win.mean()*100:.1f}%, "
          f"Avg PnL=Rs{subset.pnl.mean():+,.0f}, Net=Rs{subset.pnl.sum():+,.0f}")

print(f"\n  --- Day of Week ---")
dow={0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
for d in sorted(df.dow.unique()):
    subset=df[df.dow==d]
    print(f"  {dow[d]}: {len(subset)} trades, WR={subset.win.mean()*100:.1f}%, "
          f"Avg=Rs{subset.pnl.mean():+,.0f}, Net=Rs{subset.pnl.sum():+,.0f}")

print(f"\n  --- Month ---")
for m in sorted(df.month.unique()):
    subset=df[df.month==m]
    print(f"  Month {m}: {len(subset)} trades, WR={subset.win.mean()*100:.1f}%, "
          f"Avg=Rs{subset.pnl.mean():+,.0f}, Net=Rs{subset.pnl.sum():+,.0f}")

print(f"\n  --- Position relative to EMA50 ---")
above_ema50=df[df.c_ema50>0]
below_ema50=df[df.c_ema50<=0]
print(f"  Above EMA50: {len(above_ema50)} trades, WR={above_ema50.win.mean()*100:.1f}%, Net=Rs{above_ema50.pnl.sum():+,.0f}")
print(f"  Below EMA50: {len(below_ema50)} trades, WR={below_ema50.win.mean()*100:.1f}%, Net=Rs{below_ema50.pnl.sum():+,.0f}")

print(f"\n  --- Position relative to EMA200 ---")
above_ema200=df[df.c_ema200>0]
below_ema200=df[df.c_ema200<=0]
print(f"  Above EMA200: {len(above_ema200)} trades, WR={above_ema200.win.mean()*100:.1f}%, Net=Rs{above_ema200.pnl.sum():+,.0f}")
print(f"  Below EMA200: {len(below_ema200)} trades, WR={below_ema200.win.mean()*100:.1f}%, Net=Rs{below_ema200.pnl.sum():+,.0f}")

# ══════════════════════════════════════════════
# 3. PRIOR TREND ANALYSIS
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("3. PRIOR TREND (candles before signal)")
print(f"{'='*70}")

print(f"\n  --- Trend (5-candle return) ---")
print(f"  Mean: {df.trend_5.mean():+.2f}%")
# Bucket
df["trend_bucket"]=pd.cut(df.trend_5, bins=[-10,-3,-1,-0.5,0,0.5,1,3,10])
for b, grp in df.groupby("trend_bucket", observed=True):
    print(f"  {b}: {len(grp)} trades, WR={grp.win.mean()*100:.1f}%, Net=Rs{grp.pnl.sum():+,.0f}")

print(f"\n  --- 10-candle trend ---")
print(f"  Mean: {df.trend_10.mean():+.2f}%")
df["trend10_bucket"]=pd.cut(df.trend_10, bins=[-15,-5,-2,-1,0,1,2,5,15])
for b, grp in df.groupby("trend10_bucket", observed=True):
    print(f"  {b}: {len(grp)} trades, WR={grp.win.mean()*100:.1f}%, Net=Rs{grp.pnl.sum():+,.0f}")

print(f"\n  --- Red candles in prior 5 ---")
df["red_prev5_bucket"]=pd.cut(df.red_prev5, bins=[-1,0,1,2,3,4,5])
for b, grp in df.groupby("red_prev5_bucket", observed=True):
    print(f"  {b} red candles: {len(grp)} trades, WR={grp.win.mean()*100:.1f}%, Net=Rs{grp.pnl.sum():+,.0f}")

print(f"\n  --- Green candles in prior 5 ---")
df["green_prev5_bucket"]=pd.cut(df.green_prev5, bins=[-1,0,1,2,3,4,5])
for b, grp in df.groupby("green_prev5_bucket", observed=True):
    print(f"  {b} green candles: {len(grp)} trades, WR={grp.win.mean()*100:.1f}%, Net=Rs{grp.pnl.sum():+,.0f}")

# ══════════════════════════════════════════════
# 4. BODY RATIO ANALYSIS
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("4. BODY RATIO (how much bigger is signal vs prior candle?)")
print(f"{'='*70}")

for ratio_thresh in [0.5, 1, 2, 3, 5, 10]:
    subset=df[df.body_ratio>=ratio_thresh]
    print(f"  Body ratio >= {ratio_thresh:.0f}x: {len(subset)} trades ({len(subset)/len(df)*100:.1f}%), "
          f"WR={subset.win.mean()*100:.1f}%, Net=Rs{subset.pnl.sum():+,.0f}")

# ══════════════════════════════════════════════
# 5. VOLUME ANALYSIS
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("5. VOLUME PATTERNS")
print(f"{'='*70}")

# Check if volume is even meaningful (many may be 0)
print(f"\n  --- Volume present? ---")
print(f"  Trades with vol=0: {(df.s_vol==0).sum()}/{len(df)}")
print(f"  Trades with prior vol=0: {(df.p_vol==0).sum()}/{len(df)}")
print(f"  Trades with both vol=0: {((df.s_vol==0)&(df.p_vol==0)).sum()}/{len(df)}")

# Volume MA20
print(f"\n  --- Volume vs 20-MA ---")
for p in [0.5,1,1.5,2,3]:
    subset=df[df.vol_ma20>=p]
    print(f"  Vol >= {p:.1f}x MA20: {len(subset)} trades, WR={subset.win.mean()*100:.1f}%, Net=Rs{subset.pnl.sum():+,.0f}")

# ══════════════════════════════════════════════
# 6. WORST LOSS ANALYSIS
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("6. WORST LOSERS DEEP DIVE")
print(f"{'='*70}")

worst=df.nsmallest(20, "pnl")
print(f"\n  Top 20 worst losses:")
for _, r in worst.iterrows():
    print(f"  {r.dt.date()} {r.sym:>7s} | PnL=Rs{r.pnl:>+8,.0f} | "
          f"Body={r.s_body:.0f}/{r.p_body:.0f} | Gap={r.gap_pct:+.2f}% | "
          f"Trend5={r.trend_5:+.2f}% | Vol={r.vol_ratio:.1f}x | "
          f"Hour={r.hour:02d}h | EMA50={'ABV' if r.c_ema50>0 else 'BLW'}")

print(f"\n  --- What do the worst 50 have in common? ---")
worst50=df.nsmallest(50, "pnl")
best50=df.nlargest(50, "pnl")
print(f"  {'Metric':>20s} {'Worst50':>12s} {'Best50':>12s} {'All':>12s}")
for col in ["s_body","p_body","body_ratio","gap_pct","vol_ratio","s_range",
            "trend_5","trend_10","c_ema50","c_ema200","atr_pct"]:
    print(f"  {col:>20s} {worst50[col].mean():>10.2f}  {best50[col].mean():>10.2f}  {df[col].mean():>10.2f}")

# ══════════════════════════════════════════════
# 7. IS THERE A "DEAD ZONE"?
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("7. DEAD ZONE SEARCH: are there ranges where WR is significantly different?")
print(f"{'='*70}")

# Test every feature for WR extremes
for col in ["s_body","s_range","body_ratio","gap_pct","vol_ratio","trend_5","trend_10",
            "c_ema50","c_ema200","atr_pct","green_prev5","red_prev5"]:
    # Bucket into deciles
    df["_bucket"]=pd.qcut(df[col].rank(method="first"), q=10, labels=False, duplicates="drop")
    wr_by_bucket = df.groupby("_bucket")["win"].mean()
    max_wr = wr_by_bucket.max()
    min_wr = wr_by_bucket.min()
    max_bucket = wr_by_bucket.idxmax()
    min_bucket = wr_by_bucket.idxmin()
    span = max_wr - min_wr
    if span > 0.12:  # more than 12% WR difference
        subset_max=df[df["_bucket"]==max_bucket]
        subset_min=df[df["_bucket"]==min_bucket]
        print(f"  {col:>15s}: WR range {min_wr*100:.1f}%-{max_wr*100:.1f}% ({span*100:.1f}% spread) ***")
        print(f"    Best group: n={len(subset_max)}, WR={subset_max.win.mean()*100:.1f}%, Net=Rs{subset_max.pnl.sum():+,.0f}")
        print(f"    Worst group: n={len(subset_min)}, WR={subset_min.win.mean()*100:.1f}%, Net=Rs{subset_min.pnl.sum():+,.0f}")
    del df["_bucket"]

# ══════════════════════════════════════════════
# 8. CAN WE PROFILE WINNERS VS LOSERS?
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("8. WINNER VS LOSER PROFILE COMPARISON")
print(f"{'='*70}")

w=df[df.win==1]
l=df[df.win==0]
print(f"  {'Feature':>20s} {'Winners':>12s} {'Losers':>12s} {'Diff':>10s} {'Sig?':>6s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
for col in ["s_body","s_range","body_pct_range","p_body","body_ratio","gap_pct",
            "vol_ratio","trend_5","trend_10","atr_pct","green_prev5","red_prev5",
            "c_ema50","c_ema200","vol_ma20"]:
    if col not in df.columns:
        # compute body_pct_range
        if col=="body_pct_range":
            wm=w["s_body"].mean()/w["s_range"].mean()*100
            lm=l["s_body"].mean()/l["s_range"].mean()*100
        else:
            continue
    else:
        wm=w[col].mean()
        lm=l[col].mean()
    diff=wm-lm
    # Quick significance: if diff > 2*std_error
    se=np.sqrt(w[col].var()/len(w)+l[col].var()/len(l)) if col in df.columns else 999
    sig="***" if abs(diff)>2*se else "ns"
    print(f"  {col:>20s} {wm:>10.2f}  {lm:>10.2f}  {diff:>+8.2f}  {sig:>6s}")

# ══════════════════════════════════════════════
# 9. PER-TRADE BREAKDOWN (show individual trades)
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("9. FIRST 20 TRADES (raw data)")
print(f"{'='*70}")

for idx in range(min(20, len(df))):
    r = df.iloc[idx]
    print(f"  #{idx+1:3d} | {str(r.dt.date()):>10s} {r.sym:>7s} | "
          f"Signal: O={r.s_open:.0f} H={r.s_high:.0f} L={r.s_low:.0f} C={r.s_close:.0f} | "
          f"Body={r.s_body:.0f}/{r.p_body:.0f} Gap={r.gap_pct:+.2f}% | "
          f"Hour={r.hour:02d}h Trend5={r.trend_5:+.2f}% | "
          f"PnL=Rs{r.pnl:>+8,.0f} {'W' if r.win else 'L'}")

# ══════════════════════════════════════════════
# 10. DISTRIBUTION OF P&L
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("10. P&L DISTRIBUTION")
print(f"{'='*70}")

print(f"  Min:  Rs{df.pnl.min():+,.0f}")
print(f"  Max:  Rs{df.pnl.max():+,.0f}")
print(f"  Mean: Rs{df.pnl.mean():+,.0f}")
print(f"  Median: Rs{df.pnl.median():+,.0f}")
print(f"  Std:  Rs{df.pnl.std():+,.0f}")
percs=df.pnl.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99])
for p,v in percs.items():
    print(f"  P{p*100:.0f}: Rs{v:+,.0f}")

# How many trades drive the P&L?
print(f"\n  --- Concentration of P&L ---")
sorted_pnl = df.pnl.sort_values().values
top10_pct = sorted_pnl[-10:].sum() / df.pnl.sum() * 100
top20_pct = sorted_pnl[-20:].sum() / df.pnl.sum() * 100
top50_pct = sorted_pnl[-50:].sum() / df.pnl.sum() * 100
bottom10_pct = sorted_pnl[:10].sum() / df.pnl.sum() * 100
print(f"  Top 10 trades contribute: {top10_pct:.1f}% of total P&L")
print(f"  Top 20 trades contribute: {top20_pct:.1f}% of total P&L")
print(f"  Top 50 trades contribute: {top50_pct:.1f}% of total P&L")
print(f"  Bottom 10 trades contribute: {bottom10_pct:.1f}% of total P&L")

# ══════════════════════════════════════════════
# 11. WHAT MAKES THE BIG WINNERS DIFFERENT?
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("11. TOP 10 WINNERS - What do they look like?")
print(f"{'='*70}")

top10 = df.nlargest(10, "pnl")
for _, r in top10.iterrows():
    vol_str = f"Vol={r.vol_ratio:.1f}x" if r.s_vol>0 and r.p_vol>0 else "Vol=N/A"
    print(f"  {r.dt.date()} {r.sym:>7s} | PnL={r.pnl:>+8,.0f} | "
          f"Body={r.s_body:.0f}/{r.p_body:.0f} BR={r.body_ratio:.1f}x | "
          f"Gap={r.gap_pct:+.2f}% | Trend5={r.trend_5:+.2f}% | "
          f"{vol_str} | Hour={r.hour:02d}h | "
          f"{'ABV' if r.c_ema50>0 else 'BLW'}50 {'ABV' if r.c_ema200>0 else 'BLW'}200")

print(f"\n  --- Top 10 summary ---")
print(f"  Avg body ratio: {top10.body_ratio.mean():.1f}x (all: {df.body_ratio.mean():.1f}x)")
print(f"  Avg gap: {top10.gap_pct.mean():+.2f}% (all: {df.gap_pct.mean():+.2f}%)")
print(f"  Avg trend5: {top10.trend_5.mean():+.2f}% (all: {df.trend_5.mean():+.2f}%)")
print(f"  % above EMA50: {(top10.c_ema50>0).mean()*100:.0f}% (all: {(df.c_ema50>0).mean()*100:.0f}%)")
print(f"  % above EMA200: {(top10.c_ema200>0).mean()*100:.0f}% (all: {(df.c_ema200>0).mean()*100:.0f}%)")

# ══════════════════════════════════════════════
# 12. CAN WE FIND A SIMPLE RULE?
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("12. SIMPLE RULE SEARCH - brute force filter search")
print(f"{'='*70}")

# Test every combination of simple filters
filters = {
    "body_ratio_ge": lambda df, v: df[df.body_ratio>=v],
    "body_ratio_le": lambda df, v: df[df.body_ratio<=v],
    "gap_ge": lambda df, v: df[df.gap_pct>=v],
    "gap_le": lambda df, v: df[df.gap_pct<=v],
    "trend5_ge": lambda df, v: df[df.trend_5>=v],
    "trend5_le": lambda df, v: df[df.trend_5<=v],
    "above_ema50": lambda df, v: df[df.c_ema50>0] if v else df[df.c_ema50<=0],
    "above_ema200": lambda df, v: df[df.c_ema200>0] if v else df[df.c_ema200<=0],
}

best_rules = []
for fname, ffunc in filters.items():
    if fname in ["above_ema50","above_ema200"]:
        vals = [True, False]
    elif "ratio" in fname or "gap" in fname:
        vals = [0.5, 1, 1.5, 2, 3, 5]
    elif "trend" in fname:
        vals = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    else:
        continue
    
    for v in vals:
        try:
            subset = ffunc(df, v)
        except:
            continue
        if len(subset) < 20 or len(subset) > len(df)-20:
            continue
        wr = subset.win.mean()
        net = subset.pnl.sum()
        all_net = df.pnl.sum()
        improvement = (net / all_net - 1) * 100 if all_net != 0 else 0
        best_rules.append((fname, v, len(subset), wr, net, improvement))

best_rules.sort(key=lambda x: -x[5])
print(f"\n  Top 15 simple filters by P&L improvement:")
print(f"  {'Filter':>25s} {'Value':>8s} {'N':>5s} {'WR%':>6s} {'Net P&L':>12s} {'Improve':>8s}")
print(f"  {'-'*25} {'-'*8} {'-'*5} {'-'*6} {'-'*12} {'-'*8}")
for fname, v, n, wr, net, imp in best_rules[:15]:
    val_str = f"{v}" if isinstance(v, bool) else f"{v:.1f}"
    print(f"  {fname:>25s} {val_str:>8s} {n:>5d} {wr*100:>5.1f}% Rs{net:>+9,.0f} {imp:>+7.1f}%")

# Check: are the best filters consistent over time?
print(f"\n  --- Consistency check: best filter over years ---")
if len(best_rules) > 0:
    best_filter_name, best_val, _, _, _, _ = best_rules[0]
    print(f"  Best filter: {best_filter_name} > {best_val}")
    for yr in sorted(df.year.unique()):
        subset = df[df.year==yr]
        if len(subset) < 10: continue
        # Apply best filter
        try:
            filtered = filters[best_filter_name](subset, best_val)
        except:
            continue
        if len(filtered) < 5: continue
        imp = (filtered.pnl.sum() / subset.pnl.sum() - 1) * 100 if subset.pnl.sum() != 0 else 0
        print(f"    {yr}: all={len(subset)}, filtered={len(filtered)}, "
              f"WR={filtered.win.mean()*100:.1f}% (all={subset.win.mean()*100:.1f}%), "
              f"improvement={imp:+.1f}%")

print(f"\nDONE")
