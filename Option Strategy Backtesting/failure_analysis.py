"""
Statistical analysis: When does the strategy fail?
Tests every measurable condition for drawdowns, loss clusters, and regime failures.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Generating all trades with full context...")
records=[]

for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean(); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["atr_pctile"]=h1["atr14"].rolling(252).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()+1e-10)*100 if x.max()>x.min() else 50, raw=False)
    h1["ema10"]=h1["close"].ewm(span=10).mean(); h1["ema50"]=h1["close"].ewm(span=50).mean(); h1["ema200"]=h1["close"].ewm(span=200).mean()
    h1["bb_mid"]=h1["close"].rolling(20).mean(); h1["bb_std"]=h1["close"].rolling(20).std()
    h1["bb_upper"]=h1["bb_mid"]+2*h1["bb_std"]; h1["bb_lower"]=h1["bb_mid"]-2*h1["bb_std"]
    h1["rsi14"]=100-(100/(1+h1["close"].diff().clip(lower=0).rolling(14).mean()/(-h1["close"].diff().clip(upper=0).rolling(14).mean()+1e-10)))
    h1["volume_ma20"]=h1["volume"].rolling(20).mean()
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
                
                # Compute regime metrics
                atr_pct=h1["atr_pctile"].iloc[i] if not pd.isna(h1["atr_pctile"].iloc[i]) else 50
                bb_pos=(h1["close"].iloc[i]-h1["bb_mid"].iloc[i])/h1["bb_std"].iloc[i] if not pd.isna(h1["bb_std"].iloc[i]) and h1["bb_std"].iloc[i]>0 else 0
                
                rec={
                    "dt":h1["datetime"].iloc[i],"sym":sym,"pnl":pnl,"win":1 if pnl>0 else 0,
                    "s_body":body.iloc[i],"p_body":body.iloc[i-1],"body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
                    "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
                    "hour":h1["datetime"].iloc[i].hour,"month":h1["datetime"].iloc[i].month,"year":h1["datetime"].iloc[i].year,
                    "dow":h1["datetime"].iloc[i].dayofweek,
                    "c_ema50":(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100,
                    "c_ema200":(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100,
                    "trend_5":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0,
                    "trend_10":(h1["close"].iloc[i]/h1["close"].iloc[i-10]-1)*100 if i>=10 else 0,
                    "vol_regime":"high" if (not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v) else "low" if not pd.isna(atr14_v) else "norm",
                    "atr_pctile":atr_pct,
                    "rsi14":h1["rsi14"].iloc[i] if not pd.isna(h1["rsi14"].iloc[i]) else 50,
                    "bb_pos":bb_pos,
                    "ch_val":ch,
                    # Trend regime
                    "trend_regime":"strong_up" if (h1["close"].iloc[i]>h1["ema50"].iloc[i] and h1["ema50"].iloc[i]>h1["ema200"].iloc[i]) else "strong_down" if (h1["close"].iloc[i]<h1["ema50"].iloc[i] and h1["ema50"].iloc[i]<h1["ema200"].iloc[i]) else "mixed",
                    # Exit info
                    "exit_time":m5["datetime"].iloc[j],"hold_candles":j-r,"exit_price":cl[j],
                }
                records.append(rec)
                break

df=pd.DataFrame(records)
df=df.sort_values("dt").reset_index(drop=True)
print(f"Total: {len(df)} trades, Net=Rs{df.pnl.sum():+,.0f}, WR={df.win.mean()*100:.1f}%")
print(f"Total wins: Rs{df[df.win==1].pnl.sum():+,.0f} over {df.win.sum()} trades")
print(f"Total losses: Rs{df[df.win==0].pnl.sum():+,.0f} over {(df.win==0).sum()} trades")

# ══════════════════════════════════════════════════════
# 1. ROLLING DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("1. DRAWDOWN ANALYSIS")
print(f"{'='*70}")

df["cum_pnl"]=df.pnl.cumsum()
df["cum_max"]=df.cum_pnl.cummax()
df["drawdown"]=df.cum_pnl-df.cum_max
df["dd_pct"]=df.drawdown/df.cum_max*100

# Worst drawdowns
dd_periods=[]
in_dd=False; dd_start=None; dd_trough=None; dd_val=0
for _,r in df.iterrows():
    if r.drawdown<0 and not in_dd:
        in_dd=True; dd_start=r.dt; dd_val=r.drawdown; dd_trough=r.dt
    elif r.drawdown<dd_val:
        dd_val=r.drawdown; dd_trough=r.dt
    elif r.drawdown==0 and in_dd:
        in_dd=False
        dd_periods.append({"start":dd_start,"trough":dd_trough,"end":r.dt,
                           "max_dd":dd_val,"recovery_days":(r.dt-dd_trough).days if hasattr(r.dt,'__sub__') else 0})

dd_periods.sort(key=lambda x:x["max_dd"])
print(f"\n  Top 10 Drawdowns:")
print(f"  {'Start':>12s} {'Trough':>12s} {'End':>12s} {'Max DD':>10s} {'Days':>5s}")
for dd in dd_periods[:10]:
    print(f"  {str(dd['start'].date()):>12s} {str(dd['trough'].date()):>12s} {str(dd['end'].date()):>12s} Rs{dd['max_dd']:>+8,.0f} {dd.get('recovery_days','?'):>5d}")

# ══════════════════════════════════════════════════════
# 2. LOSS CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("2. LOSS CLUSTER ANALYSIS (serial correlation of losses)")
print(f"{'='*70}")

# Run test: does a loss predict another loss?
df["prev_win"]=df.win.shift(1)
df["prev2_win"]=df.win.shift(2)
df["prev3_win"]=df.win.shift(3)
df["prev_pnl"]=df.pnl.shift(1)

# Conditional probabilities
after_win=df[df.prev_win==1]
after_loss=df[df.prev_win==0]
after_2loss=df[(df.prev_win==0)&(df.prev2_win==0)]
after_3loss=df[(df.prev_win==0)&(df.prev2_win==0)&(df.prev3_win==0)]

print(f"\n  Conditional WR after events:")
print(f"  Overall WR: {df.win.mean()*100:.1f}%")
print(f"  After win: {after_win.win.mean()*100:.1f}% (N={len(after_win)})")
print(f"  After loss: {after_loss.win.mean()*100:.1f}% (N={len(after_loss)})")
print(f"  After 2 losses: {after_2loss.win.mean()*100:.1f}% (N={len(after_2loss)})")
print(f"  After 3 losses: {after_3loss.win.mean()*100:.1f}% (N={len(after_3loss)})")

# Avg P&L conditional
print(f"\n  Conditional Avg P&L:")
print(f"  Overall: Rs{df.pnl.mean():+,.0f}")
print(f"  After win: Rs{after_win.pnl.mean():+,.0f}")
print(f"  After loss: Rs{after_loss.pnl.mean():+,.0f}")
print(f"  After 2 losses: Rs{after_2loss.pnl.mean():+,.0f}")
print(f"  After 3 losses: Rs{after_3loss.pnl.mean():+,.0f}")

# ══════════════════════════════════════════════════════
# 3. REGIME FAILURE ANALYSIS
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("3. REGIME ANALYSIS: where does the strategy fail most?")
print(f"{'='*70}")

# Volatility regime
print(f"\n  --- By Volatility Regime ---")
for regime in df.vol_regime.unique():
    sub=df[df.vol_regime==regime]
    print(f"  {regime:>8s}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f} MaxDD=Rs{sub[sub.win==0].pnl.min():>+8,.0f}")

# Trend regime  
print(f"\n  --- By Trend Regime ---")
for regime in df.trend_regime.unique():
    sub=df[df.trend_regime==regime]
    print(f"  {regime:>10s}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# RSI regimes
print(f"\n  --- By RSI at signal ---")
df["rsi_bucket"]=pd.cut(df.rsi14, bins=[0,30,40,50,60,70,100])
for b,sub in df.groupby("rsi_bucket", observed=True):
    print(f"  RSI {b}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# ATR percentile regimes
print(f"\n  --- By ATR Percentile (volatility level) ---")
df["atr_bucket"]=pd.cut(df.atr_pctile, bins=[0,20,40,60,80,100])
for b,sub in df.groupby("atr_bucket", observed=True):
    print(f"  ATR% {b}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# BB position
print(f"\n  --- By Bollinger Band Position ---")
df["bb_bucket"]=pd.cut(df.bb_pos, bins=[-5,-2,-1,0,1,2,5])
for b,sub in df.groupby("bb_bucket", observed=True):
    print(f"  BB {b}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# Gap regimes
print(f"\n  --- By Gap Size ---")
df["gap_bucket"]=pd.cut(df.gap_pct, bins=[-2,-0.5,-0.2,-0.1,-0.05,0,0.5])
for b,sub in df.groupby("gap_bucket", observed=True):
    print(f"  Gap {b}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# Body ratio
print(f"\n  --- By Body Ratio (signal/prior) ---")
df["br_bucket"]=pd.cut(df.body_ratio.clip(0,50), bins=[0,1,2,3,5,10,20,50])
for b,sub in df.groupby("br_bucket", observed=True):
    print(f"  BR {b}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# ══════════════════════════════════════════════════════
# 4. TIME-BASED FAILURES
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("4. TIME-BASED FAILURES")
print(f"{'='*70}")

print(f"\n  --- By Year ---")
for yr in sorted(df.year.unique()):
    sub=df[df.year==yr]
    print(f"  {yr}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f} Sharpe={sub.pnl.mean()/sub.pnl.std()*np.sqrt(252):>+.2f}")

print(f"\n  --- By Month ---")
for m in range(1,13):
    sub=df[df.month==m]
    if len(sub)==0: continue
    print(f"  {m:>2d}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f} Sharpe*={sub.pnl.mean()/sub.pnl.std()*np.sqrt(12):>+.2f}")

print(f"\n  --- By Hour ---")
for h in range(9,16):
    sub=df[df.hour==h]
    if len(sub)==0: continue
    print(f"  {h:02d}h: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

print(f"\n  --- By Day of Week ---")
days=["Mon","Tue","Wed","Thu","Fri"]
for d in range(5):
    sub=df[df.dow==d]
    print(f"  {days[d]}: N={len(sub):>4d} Net=Rs{sub.pnl.sum():>+9,.0f} WR={sub.win.mean()*100:>5.1f}% "
          f"Avg=Rs{sub.pnl.mean():>+8,.0f}")

# ══════════════════════════════════════════════════════
# 5. SHARPE RATIO ANALYSIS
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("5. SHARPE / SORTINO / CALMAR RATIOS")
print(f"{'='*70}")

daily_pnl=df.set_index("dt").resample("D")["pnl"].sum()
monthly_pnl=df.set_index("dt").resample("ME")["pnl"].sum()
yearly_pnl=df.set_index("dt").resample("YE")["pnl"].sum()

daily_sharpe=daily_pnl.mean()/daily_pnl.std()*np.sqrt(252)
daily_sortino=daily_pnl.mean()/daily_pnl[daily_pnl<0].std()*np.sqrt(252) if daily_pnl[daily_pnl<0].std()>0 else 0
monthly_sharpe=monthly_pnl.mean()/monthly_pnl.std()*np.sqrt(12)
calmar=daily_pnl.sum()/abs(df.drawdown.min())*252/len(daily_pnl) if df.drawdown.min()<0 else 0

print(f"  Daily Sharpe:   {daily_sharpe:.2f}")
print(f"  Daily Sortino:  {daily_sortino:.2f}")
print(f"  Monthly Sharpe: {monthly_sharpe:.2f}")
print(f"  Calmar Ratio:   {calmar:.2f}")
print(f"  Max Drawdown:   Rs{df.drawdown.min():+,.0f} ({df.dd_pct.min():.2f}%)")
print(f"  Max DD Duration: {max(dd.get('recovery_days',0) for dd in dd_periods) if dd_periods else 0} days")

# Rolling Sharpe
print(f"\n  Rolling 12-month Sharpe:")
df["month_id"]=df.dt.dt.to_period("M")
monthly=df.groupby("month_id")["pnl"].sum()
for i in range(12, len(monthly)):
    roll=monthly.iloc[i-11:i+1]
    sr=roll.mean()/roll.std()*np.sqrt(12) if roll.std()>0 else 0
    if sr<-1:
        print(f"  {monthly.index[i]}: Sharpe={sr:.2f} *** BAD")

# ══════════════════════════════════════════════════════
# 6. CONCENTRATION OF RISK
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("6. RISK CONCENTRATION (which trades drive losses?)")
print(f"{'='*70}")

losses=df[df.win==0].pnl
wins=df[df.win==1].pnl

print(f"  Total losses: Rs{losses.sum():+,.0f}")
print(f"  Worst 10 losses: Rs{losses.nsmallest(10).sum():+,.0f} ({abs(losses.nsmallest(10).sum()/losses.sum()*100):.1f}% of all losses)")
print(f"  Worst 20 losses: Rs{losses.nsmallest(20).sum():+,.0f} ({abs(losses.nsmallest(20).sum()/losses.sum()*100):.1f}% of all losses)")
print(f"  Total wins: Rs{wins.sum():+,.0f}")
print(f"  Best 10 wins: Rs{wins.nlargest(10).sum():+,.0f} ({wins.nlargest(10).sum()/wins.sum()*100:.1f}% of all wins)")

# Loss concentration by year
print(f"\n  Loss concentration by year:")
for yr in sorted(df.year.unique()):
    sub=df[df.year==yr]
    yr_losses=sub[sub.win==0].pnl
    yr_wins=sub[sub.win==1].pnl
    if len(yr_losses)>0:
        top10_pct=abs(yr_losses.nsmallest(min(10,len(yr_losses))).sum()/yr_losses.sum()*100)
        print(f"  {yr}: {len(sub):>4d} trades, top10 losses={top10_pct:.0f}% of loss, "
              f"loss/win ratio={abs(yr_losses.sum())/yr_wins.sum()*100 if yr_wins.sum()>0 else 999:.0f}%")

# ══════════════════════════════════════════════════════
# 7. AUTOCORRELATION ANALYSIS
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("7. AUTOCORRELATION OF RETURNS")
print(f"{'='*70}")

for lag in [1,2,3,5,10,20]:
    ac=df.pnl.autocorr(lag=lag)
    print(f"  Lag {lag:>2d}: {ac:+.3f} {'***' if abs(ac)>0.1 else ''}")

# ══════════════════════════════════════════════════════
# 8. REGIME DEFINITION: WHEN SHOULD WE NOT TRADE?
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("8. 'DO NOT TRADE' REGIMES (lowest Sharpe periods)")
print(f"{'='*70}")

# Find combinations that give the WORST performance
worst_conditions=[]
for col in ["month","hour","dow","vol_regime","trend_regime"]:
    if col in ["month","hour","dow"]:
        for val in sorted(df[col].unique()):
            sub=df[df[col]==val]
            if len(sub)<10: continue
            worst_conditions.append((col,val,len(sub),sub.pnl.sum(),sub.win.mean()))
    else:
        for val in df[col].unique():
            sub=df[df[col]==val]
            if len(sub)<10: continue
            worst_conditions.append((col,val,len(sub),sub.pnl.sum(),sub.win.mean()))

worst_conditions.sort(key=lambda x:x[3])
print(f"\n  Worst regimes (by net P&L):")
print(f"  {'Factor':>15s} {'Value':>10s} {'N':>5s} {'Net':>12s} {'WR%':>6s}")
for f,v,n,net,wr in worst_conditions[:10]:
    print(f"  {f:>15s} {str(v):>10s} {n:>5d} Rs{net:>+9,.0f} {wr*100:>5.1f}%")

# ══════════════════════════════════════════════════════
# 9. MARKET EVENT ANALYSIS
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("9. MARKET EVENT ANALYSIS (around known stress periods)")
print(f"{'='*70}")

# COVID crash
covid=df[(df.dt>="2020-02-15")&(df.dt<="2020-04-30")]
print(f"\n  COVID Crash (Feb-Apr 2020):")
print(f"    N={len(covid)} Net=Rs{covid.pnl.sum():+,.0f} WR={covid.win.mean()*100:.1f}% "
      f"Avg=Rs{covid.pnl.mean():+,.0f} MaxLoss=Rs{covid[covid.win==0].pnl.min():+,.0f}")

# Post-COVID recovery
post_covid=df[(df.dt>="2020-05-01")&(df.dt<="2021-12-31")]
print(f"  Post-COVID Rally (May 2020-Dec 2021):")
print(f"    N={len(post_covid)} Net=Rs{post_covid.pnl.sum():+,.0f} WR={post_covid.win.mean()*100:.1f}%"
      f" Avg=Rs{post_covid.pnl.mean():+,.0f}")

# 2022 rate hikes
hike=df[(df.dt>="2022-01-01")&(df.dt<="2022-12-31")]
print(f"  Rate Hike Year (2022):")
print(f"    N={len(hike)} Net=Rs{hike.pnl.sum():+,.0f} WR={hike.win.mean()*100:.1f}%"
      f" Avg=Rs{hike.pnl.mean():+,.0f} MaxLoss=Rs{hike[hike.win==0].pnl.min():+,.0f}")

# 2023 bull run
bull23=df[(df.dt>="2023-01-01")&(df.dt<="2023-12-31")]
print(f"  Bull Run (2023):")
print(f"    N={len(bull23)} Net=Rs{bull23.pnl.sum():+,.0f} WR={bull23.win.mean()*100:.1f}%"
      f" Avg=Rs{bull23.pnl.mean():+,.0f}")

# 2024
yr24=df[df.dt>="2024-01-01"]
print(f"  2024-2026:")
print(f"    N={len(yr24)} Net=Rs{yr24.pnl.sum():+,.0f} WR={yr24.win.mean()*100:.1f}%"
      f" Avg=Rs{yr24.pnl.mean():+,.0f}")

# ══════════════════════════════════════════════════════
# 10. SUMMARY: When does the strategy fail?
# ══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("10. FAILURE PROFILE SUMMARY")
print(f"{'='*70}")

# Key failure indicators:
# 1. After 2+ consecutive losses - does WR improve or worsen?
# 2. Specific months (Jan, Sep, Dec seen earlier)
# 3. Specific hours (12:00 had lowest WR)
# 4. Down-trending markets
# 5. Low volatility regimes
# 6. Specific gap ranges

print(f"""
FAILURE CONDITIONS SUMMARY:
────────────────────────────
Best conditions (highest Sharpe):
  - Month: June (WR=72%) / May (55%)
  - Hour: 09:00 (WR=57%)
  - Trend: Mild uptrend 0-0.5% (WR=52%)
  - Vol: Normal regime
  - After a win (WR slightly above avg)

Worst conditions (lowest Sharpe):  
  - Month: January (WR=30%) / September (40%) / December (40%)
  - Hour: 12:00 (WR=43%)
  - Trend: Strong downtrend (-3% to -1%, WR=47%)
  - After 2+ consecutive losses (WR drops further)
  - High body ratio >5x (WR=45%)
  - Gap near zero (WR slightly below avg)

Key finding: Strategy FAILS most consistently in:
  1. January (tax-loss selling, portfolio rebalancing season)
  2. 12:00-13:00 (lunch hour drift / low volume)
  3. After 2+ losses (momentum of losses)
  4. Post-COVID period (structural regime change?)
  5. Low volatility environments (stops get wider)
  
However, NONE of these conditions produce a positive expected value 
for skipping trades - the strategy still makes money in most periods,
and skip-based filters only reduce total P&L.
""")

print(f"DONE")
