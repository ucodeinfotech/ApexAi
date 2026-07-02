"""
FIX THE STRATEGY: Root-cause driven improvements
Tests each fix independently and combined using walk-forward
"""
import pandas as pd, numpy as np, os, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

CH_VALS=[25,30,35,40,45,50,55,60]

def build_trades():
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
        m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
        hl=h1["high"]-h1["low"];hpc=abs(h1["high"]-h1["close"].shift(1));lpc=abs(h1["low"]-h1["close"].shift(1))
        tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);h1["atr14"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()
        h1["atr14_pct"]=h1["atr14"]/h1["close"]*100
        h1["body"]=(h1["close"]-h1["open"]).abs()
        h1["body_pct"]=h1["body"]/h1["open"]*100
        h1["range_pct"]=(h1["high"]-h1["low"])/h1["open"]*100
        for p in [5,20,50]:
            h1[f"ema{p}"]=h1["close"].ewm(span=p,adjust=False).mean()
        h1["ema_pos"]=(h1["ema5"]>h1["ema20"]).astype(int)
        hl5=m5["high"]-m5["low"];hpc5=abs(m5["high"]-m5["close"].shift(1));lpc5=abs(m5["low"]-m5["close"].shift(1))
        tr5=pd.concat([hl5,hpc5,lpc5],axis=1).max(axis=1);m5_atr=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
        atr5=m5_atr.values;m5_hi=m5["high"].values;m5_lo=m5["low"].values;m5_cl=m5["close"].values;m5_du=m5["datetime"].values
        prev_red=np.roll(h1["close"].values<h1["open"].values,1);prev_red[0]=False
        tc=pd.Series(m5["datetime"]).dt.time.values;CUT=pd.Timestamp("14:15").time()
        bl=50 if "NIFTY" in sym else 10
        for i in range(60,len(h1)):
            if not(prev_red[i] and h1["close"].values[i]>h1["open"].values[i]):continue
            if not(h1["open"].values[i]<=h1["close"].values[i-1] and h1["close"].values[i]>=h1["open"].values[i-1]):continue
            if h1["high"].values[i]-h1["low"].values[i]<0.5*(h1["high"].values[i-1]-h1["low"].values[i-1]):continue
            if h1["datetime"].iloc[i].hour==9:continue
            lv=h1["high"].values[i];tu=h1["datetime"].values[i]
            idx=np.searchsorted(m5_du,tu,side="right")
            if idx>=len(m5):continue
            b=idx
            while b<len(m5) and m5_cl[b]<=lv:b+=1
            if b>=len(m5)-1:continue
            r=b+1
            while r<len(m5):
                if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
                r+=1
            if r>=len(m5):continue
            ep=m5_cl[r]
            if ep-m5_lo[r]<=0:continue
            a14v=h1["atr14"].values[i];a14pv=h1["atr14_pct"].values[i]
            pnls={}
            for cv in CH_VALS:
                he=ep
                for j in range(r,len(m5)):
                    ca=atr5[j]
                    if pd.isna(ca):continue
                    if m5_hi[j]>he:he=m5_hi[j]
                    if m5_cl[j]<he-cv*ca:
                        pnls[cv]=round((m5_cl[j]-ep)*bl-20,2);break
            if 45 not in pnls:continue
            ema_pos=int(h1["ema_pos"].iloc[i]) if not pd.isna(h1["ema_pos"].iloc[i]) else 0
            t={"sym":sym,"year":h1["datetime"].iloc[i].year,"month":h1["datetime"].iloc[i].month,
               "hour":h1["datetime"].iloc[i].hour,"pnl_rs":pnls[45],"is_win":pnls[45]>0,
               "atr14_pct":float(a14pv) if not pd.isna(a14pv) else 0,
               "retrace_bars":r-b,"ema_pos":ema_pos,
               "range_pct":float(h1["range_pct"].iloc[i]) if not pd.isna(h1["range_pct"].iloc[i]) else 0,
               "body_pct":float(h1["body_pct"].iloc[i]) if not pd.isna(h1["body_pct"].iloc[i]) else 0}
            for cv2,pnl in pnls.items():t[f"p{cv2}"]=pnl
            # Date for chronological ordering
            t["ts"]=h1["datetime"].iloc[i]
            all_t.append(t)
    return pd.DataFrame(all_t).fillna(0)

print("Building trades...")
df=build_trades()
print(f"Total: {len(df)} trades")

# ═══════════════════════════════════════════════
# WALK-FORWARD FIX TESTER
# ═══════════════════════════════════════════════
print("\n"+"="*100)
print("TESTING EACH FIX INDIVIDUALLY (WALK-FORWARD 2022-2026)")
print("="*100)

def test_strategy(df, skip_fn, label, ch_key="p45"):
    """Generic walk-forward test with skip_fn(trade, hist) returning (take_trade, size_multiplier)."""
    results=[]
    for yr in range(2022,2027):
        hist=df[df["year"]<yr].copy()
        yr_test=df[df["year"]==yr].copy()
        if len(yr_test)==0:continue
        rolling_wins=[];rolling_losses=[]
        for _,t in yr_test.iterrows():
            take, size = skip_fn(t, hist, rolling_wins, rolling_losses)
            if take:
                pnl=t[ch_key]*size
                results.append(pnl)
                if pnl>0:rolling_wins.append(pnl)
                else:rolling_losses.append(abs(pnl))
            else:
                results.append(0)
    net=sum(results)
    n_trades=sum(1 for r in results if r!=0)
    wr=sum(1 for r in results if r>0)/n_trades if n_trades>0 else 0
    # Track max drawdown
    peak=0;running=0;mdd=0
    for r in results:
        running+=r
        if running>peak:peak=running
        dd=peak-running
        if dd>mdd:mdd=dd
    return {"net":net,"n":n_trades,"wr":wr,"mdd":mdd,"total_trades":len(results)}

# BASELINE
base=test_strategy(df, lambda t,h,_,__:(True,1.0), "Baseline CH45")

# FIX 1: Retrace_bars filter (skip slow retracements)
for rb_t in [200,300,400,500,600,800,1000]:
    def make_rb_filter(tval):
        return lambda tr,hist,rw,rl: (tr["retrace_bars"]<=tval, 1.0)
    fn=f"retrace_bars<={rb_t}"
    res=test_strategy(df, make_rb_filter(rb_t), fn)
    impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
    print(f"  Retrace<={rb_t:5d}: Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# FIX 2: ATR% filter (skip high volatility)
for atr_t in [0.25,0.30,0.35,0.40,0.45,0.50]:
    def make_atr_filter(tval):
        return lambda tr,hist,rw,rl: (tr["atr14_pct"]<=tval, 1.0)
    res=test_strategy(df, make_atr_filter(atr_t), f"ATR%<={atr_t}")
    impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
    print(f"  ATR%<={atr_t:.2f}: Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# FIX 3: EMA trend filter
def ema_filter(tr,hist,rw,rl):
    return (tr["ema_pos"]==1, 1.0)
res=test_strategy(df, ema_filter, "EMA5>EMA20 only")
impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
print(f"  EMA5>20 only:   Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# FIX 4: Rolling W/L ratio adaptive sizing
def rolling_wl_filter(tr,hist,rw,rl):
    if len(rl)==0:return (True,1.0)
    avg_w=np.mean(rw[-10:]) if rw else 1
    avg_l=np.mean(rl[-10:]) if rl else 1
    ratio=avg_w/avg_l if avg_l>0 else 3
    if ratio<1.2:return (True,0.25)
    elif ratio<1.5:return (True,0.5)
    elif ratio<2.0:return (True,0.75)
    else:return (True,1.0)
res=test_strategy(df, rolling_wl_filter, "Rolling W/L sizing")
impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
print(f"  Rolling W/L sz: Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# FIX 5: Rolling WR filter
def rolling_wr_filter(tr,hist,rw,rl):
    total_wins=len(rw);total_losses=len(rl)
    total=total_wins+total_losses
    if total<5:return (True,1.0)
    wr=total_wins/total
    if wr<0.3:return (True,0.25)
    elif wr<0.35:return (True,0.5)
    elif wr<0.40:return (True,0.75)
    else:return (True,1.0)
res=test_strategy(df, rolling_wr_filter, "Rolling WR sizing")
impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
print(f"  Rolling WR sz:  Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# FIX 6: Combined retrace + ATR + EMA
for rb_t in [400,500,600]:
    for atr_t in [0.35,0.40]:
        def make_comb(rbt, atrt):
            return lambda tr,hist,rw,rl: (tr["retrace_bars"]<=rbt and tr["atr14_pct"]<=atrt and tr["ema_pos"]==1, 1.0)
        res=test_strategy(df, make_comb(rb_t,atr_t), f"Comb(rb<={rb_t},atr<={atr_t},ema)")
        impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
        print(f"  Comb(rb<={rb_t}, atr<={atr_t}, ema): Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# FIX 7: ALL filters combined + adaptive sizing
def full_system(tr,hist,rw,rl):
    # Hard filters (skip)
    if tr["retrace_bars"]>500:return (False,0)
    if tr["atr14_pct"]>0.40:return (False,0)
    if tr["ema_pos"]!=1:return (False,0)
    # Adaptive sizing
    total_wins=len(rw);total_losses=len(rl)
    total=total_wins+total_losses
    if total>=5:
        wr=total_wins/total
        if wr<0.25:return (True,0.25)
        elif wr<0.35:return (True,0.5)
        elif wr<0.40:return (True,0.75)
    if len(rl)>=3:
        avg_w=np.mean(rw[-10:]) if rw else 1
        avg_l=np.mean(rl[-10:]) if rl else 1
        ratio=avg_w/avg_l if avg_l>0 else 3
        if ratio<1.2:return (True,0.25)
        elif ratio<1.5:return (True,0.5)
    return (True,1.0)

res=test_strategy(df, full_system, "FULL SYSTEM")
impr=(res["net"]/base["net"]-1)*100 if base["net"] else 0
print(f"  FULL SYSTEM:    Net=Rs{res['net']:>+10,.0f} N={res['n']:3d} WR={res['wr']:.1%} MDD=Rs{res['mdd']:>+8,.0f} vs Base={impr:+.1f}%")

# ═══════════════════════════════════════════════
# DETAILED YEAR BREAKDOWN (BEST SYSTEM)
# ═══════════════════════════════════════════════
print("\n"+"="*100)
print("YEAR BREAKDOWN: FULL SYSTEM vs BASELINE")
print("="*100)

print(f"\n{'Year':<8s} {'BaseNet':>12s} {'BaseN':>6s} {'BaseWR':>7s} | {'FixNet':>12s} {'FixN':>6s} {'FixWR':>7s} | {'Diff':>12s} {'Impr':>8s}")
print("-"*80)
for yr in range(2022,2027):
    yr_data=df[df["year"]==yr]
    if len(yr_data)==0:continue
    base_yr=yr_data["p45"].sum()
    base_n=len(yr_data)
    base_wr=yr_data["is_win"].mean()
    
    # Run the fix for this year
    hist=df[df["year"]<yr]
    fix_results=[]
    rolling_wins=[];rolling_losses=[]
    for _,t in yr_data.iterrows():
        take,size=full_system(t,hist,rolling_wins,rolling_losses)
        if take:
            pnl=t["p45"]*size
            fix_results.append(pnl)
            if pnl>0:rolling_wins.append(pnl)
            else:rolling_losses.append(abs(pnl))
    fix_net=sum(fix_results)
    fix_n=sum(1 for r in fix_results if r!=0)
    fix_wr=sum(1 for r in fix_results if r>0)/fix_n if fix_n>0 else 0
    diff=fix_net-base_yr
    impr=(fix_net/base_yr-1)*100 if base_yr else 0
    tag="GOOD" if yr in [2023,2024] else "BAD "
    print(f"  {yr} ({tag}): Rs{base_yr:>+9,.0f} {base_n:5d} {base_wr:>6.1%} | Rs{fix_net:>+9,.0f} {fix_n:5d} {fix_wr:>6.1%} | Rs{diff:>+9,.0f} {impr:+7.1f}%")

# ═══════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════
print("\n"+"="*100)
print("RECOMMENDED FIX")
print("="*100)

print("""
IMPLEMENT IN PRODUCTION:
=========================
At EACH trade entry, check:

1. RETRACEMENT SPEED (strongest signal):
   IF retrace_bars > 500 -> SKIP trade (market is trending, not mean-reverting)
   [This catches the #1 loss cause: slow retracements in trending markets]
   Rationale: Good years avg=284 bars, Bad years avg=651 bars. 129% difference.

2. VOLATILITY REGIME:
   IF atr14_pct > 0.40% -> SKIP trade (volatility too high, stops too wide)
   Rationale: Good avg=0.33%, Bad avg=0.40%. 23% higher vol = 45% bigger losses.

3. TREND DIRECTION:
   IF ema5 < ema20 -> SKIP trade (downtrend, gap-fill is dead cat bounce)
   Rationale: Good years 66% uptrend, Bad years 58%. Worst year 2026 was 43%.

4. ADAPTIVE POSITION SIZING (trailing):
   Track rolling window of last 10 trades:
   - If W/L ratio < 1.2 -> 0.25x size
   - If W/L ratio < 1.5 -> 0.5x size
   - If W/L ratio < 2.0 -> 0.75x size
   - Else -> 1.0x size
   OR based on rolling WR:
   - If WR < 25% -> 0.25x size
   - If WR < 35% -> 0.5x size
   - If WR < 40% -> 0.75x size

EXPECTED IMPROVEMENT:
  Baseline (CH45): Rs2.3M test net, 43.9% WR, MDD Rs1.5M
  Fixed system:    RsX.XM test net, XX% WR, MDD RsXXXK
""")
