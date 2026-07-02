"""
MA Directional Filter Re-Backtest - FULL DATA, ALL MONTHS, proper verification
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
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    for p in [5,10,20,30,50,100,120,150,200]:
        h1[f"ma{p}"]=h1["close"].rolling(p,min_periods=p).mean()
    h1["date"]=h1["datetime"].dt.normalize()
    daily=h1.groupby("date").agg({"close":"last"}).rename(columns={"close":"dclose"})
    for p in [5,10,20,30,50,100,150,200]:
        daily[f"dma{p}"]=daily["dclose"].rolling(p,min_periods=p).mean().shift(1)
    h1=h1.merge(daily,on="date",how="left").ffill()
    for p in [20,50,100,200]:
        h1[f"ma{p}_slope"]=h1[f"ma{p}"]-h1[f"ma{p}"].shift(5)
        h1[f"dma{p}_slope"]=h1[f"dma{p}"]-h1[f"dma{p}"].shift(1)
    DATA[sym]={"h1":h1,"m5_epoch":m5["datetime"].astype("int64").values,
               "m5_cl":m5["close"].values,"m5_lo":m5["low"].values,"m5_hi":m5["high"].values,
               "m5_atr":atr5.values,"tc":pd.Series(m5["datetime"]).dt.time.values}

CUT=pd.Timestamp("14:15").time()

print("Building ALL trades with H1 entry context...")
rows=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=DATA[sym]["h1"]; d=DATA[sym]
    b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; r=h1["close"]<h1["open"]
    me=d["m5_epoch"]; mc=d["m5_cl"]; ml=d["m5_lo"]; mh=d["m5_hi"]; ma=d["m5_atr"]; tc=d["tc"]
    for i in range(1,len(h1)):
        if not (r.iloc[i-1] and g.iloc[i]): continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
        if b.iloc[i] < b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour == 9: continue
        lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
        t_ep=ts.asm8.view("int64")
        idx=np.searchsorted(me,t_ep,side="right")
        if idx>=len(mc): continue
        bi=idx
        while bi<len(mc) and mc[bi]<=lv: bi+=1
        if bi>=len(mc)-1: continue
        ri=bi+1
        while ri<len(mc):
            if ml[ri]<lv and mc[ri]>lv and tc[ri]<CUT: break
            ri+=1
        if ri>=len(mc): continue
        ep=mc[ri]
        if ep-ml[ri]<=0: continue
        he=ep; exit_pnl=None
        for j in range(ri,len(mc)):
            ca=ma[j]
            if pd.isna(ca): continue
            if mh[j]>he: he=mh[j]
            if mc[j]<he-55*ca:
                exit_pnl=round(mc[j]-ep,1); break
        if exit_pnl is None: exit_pnl=round(mc[-1]-ep,1)
        row={"ts":ts,"sym":sym,"yr":ts.year,"mo":ts.month,"dow":ts.dayofweek,"pnl":exit_pnl}
        for col in ["close","open","ma5","ma10","ma20","ma30","ma50","ma100","ma120","ma150","ma200",
                     "dma5","dma10","dma20","dma30","dma50","dma100","dma150","dma200",
                     "ma20_slope","ma50_slope","ma100_slope","ma200_slope",
                     "dma20_slope","dma50_slope","dma100_slope","dma200_slope"]:
            try: row[col]=h1[col].iloc[i]
            except: row[col]=np.nan
        rows.append(row)

df=pd.DataFrame(rows)
df=df.dropna(subset=["ma50","dma50"],how="any").reset_index(drop=True)
print(f"Total trades with valid MAs: {len(df)}, Net: {df['pnl'].sum():+,.0f}")

# ====== FILTER BUILDING - using explicit column-based masks (not lambdas) ======
filter_defs=[]

# Price > MA (hourly)
for p in [20,30,50,100,200]:
    col=f"ma{p}"
    filter_defs.append((f"Close>MA{p}",lambda df,c=col: df["close"]>df[c]))

# Price > Daily MA
for p in [20,30,50,100,200]:
    col=f"dma{p}"
    filter_defs.append((f"Close>DMA{p}",lambda df,c=col: df["close"]>df[c]))

# MA crossover (hourly)
for fp,sp in [(10,30),(10,50),(20,50),(20,100),(30,100),(50,100),(50,200),(100,200)]:
    fc=f"ma{fp}"; sc=f"ma{sp}"
    filter_defs.append((f"MA{fp}>MA{sp}",lambda df,a=fc,b=sc: df[a]>df[b]))

# MA crossover (daily)
for fp,sp in [(10,30),(10,50),(20,50),(20,100),(30,100),(50,100),(50,200),(100,200)]:
    fc=f"dma{fp}"; sc=f"dma{sp}"
    filter_defs.append((f"DMA{fp}>DMA{sp}",lambda df,a=fc,b=sc: df[a]>df[b]))

# MA slope > 0 (hourly)
for p in [20,50,100,200]:
    col=f"ma{p}_slope"
    filter_defs.append((f"MA{p}_slope>0",lambda df,c=col: df[c]>0))

# MA slope > 0 (daily)
for p in [20,50,100,200]:
    col=f"dma{p}_slope"
    filter_defs.append((f"DMA{p}_slope>0",lambda df,c=col: df[c]>0))

# Combo: close > MA50 AND MA50 rising
filter_defs.append(("Close>MA50_MArising",lambda df: (df["close"]>df["ma50"])&(df["ma50_slope"]>0)))
filter_defs.append(("Close>DMA50_DMArsing",lambda df: (df["close"]>df["dma50"])&(df["dma50_slope"]>0)))

# Classic trend: MA50>MA200 AND close > MA50
filter_defs.append(("Trend_MA50>200_C>50",lambda df: (df["ma50"]>df["ma200"])&(df["close"]>df["ma50"])))
filter_defs.append(("Trend_DMA50>200_C>50",lambda df: (df["dma50"]>df["dma200"])&(df["close"]>df["dma50"])))

# ====== TEST EACH FILTER ======
def compute_stats(pnl):
    pnl=pnl.dropna(); n=len(pnl); net=pnl.sum()
    if n<2: return {"n":0}
    wr=(pnl>0).mean(); aw=pnl[pnl>0].mean() if (pnl>0).sum()>0 else 0
    al=pnl[pnl<0].mean() if (pnl<0).sum()>0 else 0
    wl=aw/abs(al) if al!=0 else 999; pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).sum()>0 else 999
    cum=np.cumsum(pnl); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999
    return {"n":n,"net":net,"wr":wr,"aw":aw,"al":al,"wl":wl,"pf":pf,"mdd":mdd,"calmar":calmar}

results_full=[]; results_jan=[]
for fname,fn in filter_defs:
    mask=fn(df)
    keep_mask=mask
    if keep_mask.sum()<5: continue
    s=compute_stats(df.loc[keep_mask,"pnl"])
    if s["n"]==0: continue
    results_full.append({"Filter":fname,"Keep":s["n"],"Drop":len(df)-s["n"],
        "Pct":f"{s['n']/len(df):.0%}","Net":f"{s['net']:+,.0f}",
        "WR":f"{s['wr']:.0%}","W/L":f"{s['wl']:.1f}x",
        "PF":f"{s['pf']:.1f}x","MDD":f"{s['mdd']:,.0f}",
        "Calmar":f"{s['calmar']:.1f}x","Change":f"{s['net']-df['pnl'].sum():+,.0f}"})
    # January
    jan_mask=keep_mask & (df["mo"]==1)
    if jan_mask.sum()>2:
        js=compute_stats(df.loc[jan_mask,"pnl"])
        if js["n"]>0:
            results_jan.append({"Filter":fname,"Trades":js["n"],"Net":f"{js['net']:+,.0f}",
                "WR":f"{js['wr']:.0%}","W/L":f"{js['wl']:.1f}x",
                "PF":f"{js['pf']:.1f}x","MDD":f"{js['mdd']:,.0f}"})

# Baseline
s_base=compute_stats(df["pnl"])
results_full.insert(0,{"Filter":"NO FILTER (BASELINE)","Keep":s_base["n"],"Drop":0,"Pct":"100%",
    "Net":f"{s_base['net']:+,.0f}","WR":f"{s_base['wr']:.0%}",
    "W/L":f"{s_base['wl']:.1f}x","PF":f"{s_base['pf']:.1f}x",
    "MDD":f"{s_base['mdd']:,.0f}","Calmar":f"{s_base['calmar']:.1f}x","Change":"--"})

rdf=pd.DataFrame(results_full)
rdf["NetVal"]=rdf["Net"].str.replace(",","").str.replace("+","").astype(float)
rdf=rdf.sort_values("NetVal",ascending=False).reset_index(drop=True)

print("\n"+"="*130)
print("FULL DATA: ALL MA DIRECTIONAL FILTERS (Engulf_Raw CH55, 2015-2026)")
print("="*130)
print(f"{'Rank':>4} {'Filter':<38} {'Keep':>6} {'Drop':>6} {'Pct':>5} {'Net':>12} {'WR':>5} "
      f"{'W/L':>6} {'PF':>6} {'MDD':>10} {'Calmar':>8} {'Change':>12}")
print("-"*130)
for i,(_,r) in enumerate(rdf.iterrows()):
    is_best="*" if i==1 else " "; is_base="<<" if i==0 else ""
    print(f"{i+1:>3}{is_best} {r['Filter']:<38} {r['Keep']:>6} {r['Drop']:>6} {r['Pct']:>5} "
          f"{r['Net']:>12} {r['WR']:>5} {r['W/L']:>6} {r['PF']:>6} "
          f"{r['MDD']:>10} {r['Calmar']:>8} {r['Change']:>12}")

# January
print("\n"+"="*130)
print("JANUARY ONLY: MA FILTER RESULTS")
print("="*130)
if results_jan:
    jdf=pd.DataFrame(results_jan)
    jdf["NetVal"]=jdf["Net"].str.replace(",","").str.replace("+","").astype(float)
    jdf=jdf.sort_values("NetVal",ascending=False).reset_index(drop=True)
    print(f"{'Rank':>4} {'Filter':<38} {'Trades':>6} {'Net':>12} {'WR':>5} {'W/L':>6} {'PF':>6} {'MDD':>10}")
    print("-"*100)
    for i,(_,r) in enumerate(jdf.iterrows()):
        print(f"{i+1:>3}  {r['Filter']:<38} {r['Trades']:>6} {r['Net']:>12} {r['WR']:>5} "
              f"{r['W/L']:>6} {r['PF']:>6} {r['MDD']:>10}")

# ====== TOP 5 MONTHLY BREAKDOWN ======
top5=rdf.iloc[1:6]["Filter"].values
print("\n"+"="*130)
print("TOP 5 FILTERS - MONTHLY BREAKDOWN")
print("="*130)
for fname in top5:
    fn=dict(filter_defs).get(fname)
    if fn is None: continue
    mask=fn(df)
    passed=df[mask]; p=passed["pnl"]
    print(f"\n--- {fname} ({len(passed)} trades, Net: {p.sum():+,.0f}, WR: {(p>0).mean():.0%}) ---")
    mo=passed.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    print(f"  {'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>6}")
    print(f"  {'-'*35}")
    for m in range(1,13):
        if m in mo.index:
            r=mo.loc[m]
            print(f"  {MONTHS[m-1]:<6} {int(r['trades']):>7} {r['net']:>+12,.0f} {r['wr']:>5.0%}")

# ====== SPOT CHECK: WHY Price>MA filters pass all trades ======
print("\n"+"="*130)
print("SPOT CHECK: Price > MA filter verification")
print("="*130)
for p in [20,30,50,100,200]:
    col=f"ma{p}"
    above=(df["close"]>df[col]).sum(); below=len(df)-above
    print(f"  Close > MA{p}: {above}/{len(df)} ({above/len(df):.0%})  |  Below: {below}")

print(f"\n  WHY? Engulfing requires close>open on index in long-term uptrend")
avg_close=df["close"].mean()
for p in [20,50,100,200]:
    print(f"  Avg Close={avg_close:.0f}, Avg MA{p}={df[f'ma{p}'].mean():.0f}")
print(f"  2015-2026 NIFTY50 rose from ~8,000 to ~25,000 (3.1x) - secular bull")
print(f"  -> Bullish engulfing in a bull market = always above MA")

# ====== VERIFICATION TABLE ======
print("\n"+"="*130)
print("FILTER VERIFICATION: Trades rejected per filter")
print("="*130)
for fname,fn in filter_defs:
    mask=fn(df); rej=len(df)-mask.sum()
    if rej>0:
        rej_pnl=df.loc[~mask,"pnl"]
        rej_net=rej_pnl.sum()
        print(f"  {fname:<38} rejected {rej:>5} trades ({rej/len(df):>5.0%}), "
              f"rejected net={rej_net:+,.0f} pts")

# Filter that rejects most
print(f"\nTotal trades: {len(df)}, baseline net: {df['pnl'].sum():+,.0f}")
