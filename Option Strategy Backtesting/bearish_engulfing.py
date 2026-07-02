"""
Bearish Engulfing + Bull+Bear Combined - Full Backtest (Engulf_Raw logic, CH55)
"""
import pandas as pd, numpy as np, os, sys, io, warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def compute_atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),
                  abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

print("Loading data...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for d in [h1,m5]:
        d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    h1["body"]=abs(h1["close"]-h1["open"])
    DATA[sym]={"h1":h1,"m5_epoch":m5["datetime"].astype("int64").values,
               "m5_cl":m5["close"].values,"m5_lo":m5["low"].values,"m5_hi":m5["high"].values,
               "m5_atr":atr5.values,"tc":pd.Series(m5["datetime"]).dt.time.values}

CUT=pd.Timestamp("14:15").time()
CH=55

def find_retest_long(sym,t,lv):
    """Bullish: wait for dip below lv then rally above"""
    d=DATA[sym]; me=d["m5_epoch"]; mc=d["m5_cl"]; ml=d["m5_lo"]; tc=d["tc"]
    t_ep=t.asm8.view("int64")
    idx=np.searchsorted(me,t_ep,side="right")
    if idx>=len(mc): return None
    b=idx
    while b<len(mc) and mc[b]<=lv: b+=1
    if b>=len(mc)-1: return None
    r=b+1
    while r<len(mc):
        if ml[r]<lv and mc[r]>lv and tc[r]<CUT: break
        r+=1
    if r>=len(mc): return None
    ep=mc[r]
    if ep-ml[r]<=0: return None
    return (r,ep)

def find_retest_short(sym,t,lv):
    """Bearish: wait for rally above lv then break below"""
    d=DATA[sym]; me=d["m5_epoch"]; mc=d["m5_cl"]; mh=d["m5_hi"]; tc=d["tc"]
    t_ep=t.asm8.view("int64")
    idx=np.searchsorted(me,t_ep,side="right")
    if idx>=len(mc): return None
    b=idx
    while b<len(mc) and mc[b]>=lv: b+=1  # wait for price to rise above lv
    if b>=len(mc)-1: return None
    r=b+1
    while r<len(mc):
        if mh[r]>lv and mc[r]<lv and tc[r]<CUT: break  # then break back below
        r+=1
    if r>=len(mc): return None
    ep=mc[r]  # entry price (where we break below)
    if ep-mh[r]>=0: return None
    return (r,ep)

def compute_ch_exit_long(sym,r,ep):
    """Exit long: CH x ATR trail from highest close"""
    d=DATA[sym]; mc=d["m5_cl"]; mh=d["m5_hi"]; ma=d["m5_atr"]
    he=ep
    for j in range(r,len(mc)):
        ca=ma[j]
        if pd.isna(ca): continue
        if mh[j]>he: he=mh[j]
        if mc[j]<he-CH*ca: return round(mc[j]-ep,1)
    return round(mc[-1]-ep,1)

def compute_ch_exit_short(sym,r,ep):
    """Exit short: CH x ATR trail from lowest close"""
    d=DATA[sym]; mc=d["m5_cl"]; ml=d["m5_lo"]; ma=d["m5_atr"]
    le=ep  # lowest close since entry
    for j in range(r,len(mc)):
        ca=ma[j]
        if pd.isna(ca): continue
        if ml[j]<le: le=ml[j]
        if mc[j]>le+CH*ca: return round(ep-mc[j],1)  # positive = profit on short
    return round(ep-mc[-1],1)

print("Building trades...")
all_rows=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=DATA[sym]["h1"]
    body=h1["body"]; green=h1["close"]>h1["open"]; red=h1["close"]<h1["open"]
    for i in range(1,len(h1)):
        ts=h1["datetime"].iloc[i]
        # === BULLISH ENGULFING ===
        if red.iloc[i-1] and green.iloc[i]:
            if h1["open"].iloc[i]<=h1["close"].iloc[i-1] and h1["close"].iloc[i]>=h1["open"].iloc[i-1]:
                if body.iloc[i]>=body.iloc[i-1]*0.5 and ts.hour!=9:
                    lv=h1["high"].iloc[i]
                    ret=find_retest_long(sym,ts,lv)
                    if ret:
                        r,ep=ret
                        pnl=compute_ch_exit_long(sym,r,ep)
                        all_rows.append({"ts":ts,"sym":sym,"yr":ts.year,"mo":ts.month,"dow":ts.dayofweek,
                                         "type":"BULL","pnl":pnl,"side":"long"})
        # === BEARISH ENGULFING ===
        if green.iloc[i-1] and red.iloc[i]:
            if h1["open"].iloc[i]>=h1["close"].iloc[i-1] and h1["close"].iloc[i]<=h1["open"].iloc[i-1]:
                if body.iloc[i]>=body.iloc[i-1]*0.5 and ts.hour!=9:
                    lv=h1["low"].iloc[i]
                    ret=find_retest_short(sym,ts,lv)
                    if ret:
                        r,ep=ret
                        pnl=compute_ch_exit_short(sym,r,ep)
                        all_rows.append({"ts":ts,"sym":sym,"yr":ts.year,"mo":ts.month,"dow":ts.dayofweek,
                                         "type":"BEAR","pnl":pnl,"side":"short"})

df=pd.DataFrame(all_rows)
print(f"Total trades: {len(df)}")
bull=df[df["type"]=="BULL"]; bear=df[df["type"]=="BEAR"]
print(f"  Bullish: {len(bull)}, Net: {bull['pnl'].sum():+,.0f}, WR: {(bull['pnl']>0).mean():.1%}")
print(f"  Bearish: {len(bear)}, Net: {bear['pnl'].sum():+,.0f}, WR: {(bear['pnl']>0).mean():.1%}")
print(f"  Combined: Net: {df['pnl'].sum():+,.0f}, WR: {(df['pnl']>0).mean():.1%}")

def stats(pnl):
    pnl=pnl.dropna(); n=len(pnl); net=pnl.sum()
    if n<2: return {}
    wr=(pnl>0).mean(); aw=pnl[pnl>0].mean() if (pnl>0).sum()>0 else 0
    al=pnl[pnl<0].mean() if (pnl<0).sum()>0 else 0; wl=aw/abs(al) if al!=0 else 999
    pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).sum()>0 else 999
    cum=np.cumsum(pnl); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999
    return {"n":n,"net":net,"wr":wr,"aw":aw,"al":al,"wl":wl,"pf":pf,"mdd":mdd,"calmar":calmar}

print("\n"+"="*100)
print("COMPARISON: BULL vs BEAR vs COMBINED (CH55)")
print("="*100)
print(f"{'Metric':<15} {'BULL':>18} {'BEAR':>18} {'COMBINED':>18}")
print("-"*70)
for name in ["n","net","wr","aw","al","wl","pf","mdd","calmar"]:
    b_s=stats(bull["pnl"]); be_s=stats(bear["pnl"]); c_s=stats(df["pnl"])
    if name=="n":
        print(f"{name:<15} {b_s.get(name,0):>18} {be_s.get(name,0):>18} {c_s.get(name,0):>18}")
    elif name in ("wr",):
        print(f"{name:<15} {b_s.get(name,0):>17.1%} {be_s.get(name,0):>17.1%} {c_s.get(name,0):>17.1%}")
    elif name in ("wl","pf"):
        print(f"{name:<15} {b_s.get(name,0):>17.1f}x {be_s.get(name,0):>17.1f}x {c_s.get(name,0):>17.1f}x")
    else:
        print(f"{name:<15} {b_s.get(name,0):>+17,.0f} {be_s.get(name,0):>+17,.0f} {c_s.get(name,0):>+17,.0f}")

# ====== YEARLY ======
print("\n"+"="*100)
print("YEARLY BREAKDOWN")
print("="*100)
for lbl,g in [("BULL",bull),("BEAR",bear),("COMBINED",df)]:
    print(f"\n--- {lbl} ---")
    yr=g.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    print("-"*35)
    for y in sorted(yr.index):
        r=yr.loc[y]; print(f"{int(y):>6} {int(r['trades']):>7} {r['net']:>+12,.0f} {r['wr']:>6.1%}")
    print("-"*35)
    print(f"{'ALL':>6} {yr['trades'].sum():>7} {yr['net'].sum():>+12,.0f} {yr['wr'].mean():>6.1%}")

# ====== MONTHLY ======
print("\n"+"="*100)
print("MONTHLY BREAKDOWN")
print("="*100)
for lbl,g in [("BULL",bull),("BEAR",bear),("COMBINED",df)]:
    print(f"\n--- {lbl} ---")
    mo=g.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    print(f"{'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    print("-"*35)
    for m in range(1,13):
        if m in mo.index:
            r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['trades']):>7} {r['net']:>+12,.0f} {r['wr']:>6.1%}")

# ====== CORRELATION ======
print("\n"+"="*100)
print("BULL vs BEAR CORRELATION ANALYSIS")
print("="*100)
# Yearly correlation
yr_b=bull.groupby("yr")["pnl"].sum()
yr_be=bear.groupby("yr")["pnl"].sum()
common_yrs=sorted(set(yr_b.index)&set(yr_be.index))
if len(common_yrs)>2:
    corr=np.corrcoef([yr_b.get(y,0) for y in common_yrs],[yr_be.get(y,0) for y in common_yrs])[0,1]
    print(f"  Yearly net correlation: {corr:.3f}")
    if corr>0: print("  -> BULL and BEAR profit in SAME years (directional market)")
    else: print("  -> BULL and BEAR profit in OPPOSITE years (diversification benefit)")

# Monthly correlation
mo_b=bull.groupby("mo")["pnl"].sum()
mo_be=bear.groupby("mo")["pnl"].sum()
if len(mo_b)>2 and len(mo_be)>2:
    corr_m=np.corrcoef([mo_b.get(m,0) for m in range(1,13)],[mo_be.get(m,0) for m in range(1,13)])[0,1]
    print(f"  Monthly net correlation: {corr_m:.3f}")

# ====== JANUARY DETAIL ======
print("\n"+"="*100)
print("JANUARY DETAIL")
print("="*100)
for lbl,g in [("BULL",bull),("BEAR",bear),("COMBINED",df)]:
    jan=g[g["mo"]==1]
    if len(jan)>0:
        s=stats(jan["pnl"])
        print(f"  {lbl}: {s['n']} trades, Net={s['net']:+,.0f}, WR={s['wr']:.1%}, W/L={s['wl']:.1f}x")

# ====== DIVERSIFICATION BENEFIT ======
print("\n"+"="*100)
print("DIVERSIFICATION BENEFIT: Combined Sharpe vs Individual")
print("="*100)
for lbl,g in [("BULL",bull),("BEAR",bear),("COMBINED",df)]:
    p=g["pnl"]
    sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    print(f"  {lbl} Sharpe: {sharpe:.3f}")

# ====== WR vs MARKET REGIME ======
print("\n"+"="*100)
print("BULL vs BEAR: DOES BEARISH WORK IN DOWN MARKETS?")
print("="*100)
for yr in sorted(df["yr"].unique()):
    yr_b=bull[bull["yr"]==yr]; yr_be=bear[bear["yr"]==yr]
    b_net=yr_b["pnl"].sum() if len(yr_b)>0 else 0
    be_net=yr_be["pnl"].sum() if len(yr_be)>0 else 0
    direction="BULL" if b_net>abs(be_net) else "BEAR" if abs(be_net)>b_net else "MIXED"
    print(f"  {yr}: BULL={b_net:+,.0f}  BEAR={be_net:+,.0f}  -> {direction} year")

print(f"\nTotal trades: {len(df)}")
