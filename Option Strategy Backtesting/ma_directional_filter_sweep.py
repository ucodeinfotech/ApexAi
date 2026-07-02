"""
MA directional filter sweep: ALL combinations tested on Engulf_Raw CH55
"""
import pandas as pd, numpy as np, os, sys, io, warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

def compute_atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
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
    # Compute H1 MAs
    for p in [5,10,15,20,30,40,50,60,100,150,200]:
        h1[f"ma{p}"]=h1["close"].rolling(p,min_periods=p).mean()
    # Daily MAs
    h1["date"]=h1["datetime"].dt.normalize()
    daily_close=h1.groupby("date")["close"].last().to_frame("dclose")
    for p in [5,10,20,30,50,100,200]:
        daily_close[f"dma{p}"]=daily_close["dclose"].rolling(p,min_periods=p).mean()
    h1=h1.merge(daily_close,on="date",how="left").ffill()
    DATA[sym]={"h1":h1,"m5_epoch":m5["datetime"].astype("int64").values,
               "m5_cl":m5["close"].values,"m5_lo":m5["low"].values,"m5_hi":m5["high"].values,
               "m5_atr":atr5.values,"tc":pd.Series(m5["datetime"]).dt.time.values}

CUT=pd.Timestamp("14:15").time()

def get_all_trades():
    """Return df of ALL trades with H1 context at entry"""
    rows=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=DATA[sym]["h1"]; d=DATA[sym]
        b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; r=h1["close"]<h1["open"]
        me=d["m5_epoch"]; mc=d["m5_cl"]; ml=d["m5_lo"]; mh=d["m5_hi"]; ma=d["m5_atr"]; tc=d["tc"]
        for i in range(1,len(h1)):
            if not (r.iloc[i-1] and g.iloc[i]): continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
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
            ep=mc[ri]; sl=ml[ri]
            if ep-sl<=0: continue
            he=ep; exit_pnl=None
            for j in range(ri,len(mc)):
                ca=ma[j]
                if pd.isna(ca): continue
                if mh[j]>he: he=mh[j]
                if mc[j]<he-55*ca:
                    exit_pnl=round(mc[j]-ep,1); break
            if exit_pnl is None: exit_pnl=round(mc[-1]-ep,1)
            # Store H1 context for filter
            row={"ts":ts,"sym":sym,"yr":ts.year,"mo":ts.month,"pnl":exit_pnl}
            for col in h1.columns:
                if col.startswith("ma") or col.startswith("dma") or col in ["close","open","high","low"]:
                    try: row[col]=h1[col].iloc[i]
                    except: pass
            rows.append(row)
    return pd.DataFrame(rows)

print("Building all trades...")
df=get_all_trades()
print(f"Total trades: {len(df)}, Net: {df['pnl'].sum():+,.0f}")

# ====== FILTER DEFINITIONS ======
filters=[]
# 1. Price > MA (H1 hourly)
for p in [5,10,15,20,30,40,50,60,100,150,200]:
    filters.append((f"Close>MA{p}_H1","close",p,"h1",True,False))
    filters.append((f"Close>MA{p}_H1_STRICT","close",p,"h1_strict",True,False))
# 2. Price > Daily DMA
for p in [5,10,20,30,50,100,200]:
    filters.append((f"Close>DMA{p}","close",p,"daily",True,False))
# 3. Fast MA > Slow MA (hourly - non strict + strict)
pairs=[(5,20),(10,30),(10,50),(20,50),(20,100),(30,100),(50,100),(5,100),(10,200),(50,200),(100,200)]
for fp,sp in pairs:
    if fp<10 and fp!=5: continue
    filters.append((f"MA{fp}>MA{sp}_H1",fp,sp,"cross_h1",True,False))
    filters.append((f"MA{fp}>MA{sp}_H1_STRICT",fp,sp,"cross_h1_strict",True,True))
# 4. Fast MA > Slow MA (daily)
for fp,sp in [(5,20),(10,30),(10,50),(20,50),(20,100),(50,100),(50,200),(100,200)]:
    filters.append((f"DMA{fp}>DMA{sp}",fp,sp,"cross_daily",True,False))
# 5. MA slope positive
for p in [10,20,30,50,100]:
    filters.append((f"MA{p}_slope>0_H1",p,None,"slope_h1",True,False))
# 6. Combo filters
filters.append(("Close>MA50_AND_MA50>MA200_H1",None,None,"combo_classic",True,False))
filters.append(("AllTrend_H1",None,None,"combo_alltrend",True,False))
filters.append(("MA50rising_Close>MA50_H1",None,None,"combo_risetrend",True,False))
filters.append(("MA50>MA200_lag5_H1",None,None,"combo_lag",True,False))

# ====== TEST ON FULL DATA ======
def apply_filter(fname,fp,sp,ftype,df):
    """Return boolean mask for trades that pass the filter"""
    mask=pd.Series(True,index=df.index)
    if ftype=="h1":
        col=f"ma{fp}"
        if col in df.columns:
            mask=df["close"]>df[col]
    elif ftype=="h1_strict":
        col=f"ma{fp}"
        if col in df.columns:
            mask=(df["close"]>df[col])&(df["close"].shift(1)>df[col].shift(1))  # 2 bars in a row
    elif ftype=="daily":
        col=f"dma{fp}"
        if col in df.columns:
            mask=df["close"]>df[col]
    elif ftype=="cross_h1":
        fc=f"ma{fp}"; sc=f"ma{sp}"
        if fc in df.columns and sc in df.columns:
            mask=df[fc]>df[sc]
    elif ftype=="cross_h1_strict":
        fc=f"ma{fp}"; sc=f"ma{sp}"
        if fc in df.columns and sc in df.columns:
            mask=(df[fc]>df[sc])&(df[fc].shift(2)>df[sc].shift(2))  # sustained for 2+ bars
    elif ftype=="cross_daily":
        fc=f"dma{fp}"; sc=f"dma{sp}"
        if fc in df.columns and sc in df.columns:
            mask=df[fc]>df[sc]
    elif ftype=="slope_h1":
        col=f"ma{fp}"
        if col in df.columns:
            shift=df[col].shift(5)  # 5-hr slope
            mask=df[col]>shift
    elif ftype=="combo_classic":
        if "ma50" in df.columns and "ma200" in df.columns:
            mask=(df["close"]>df["ma50"])&(df["ma50"]>df["ma200"])
    elif ftype=="combo_alltrend":
        if all(c in df.columns for c in ["ma50","ma100","ma200"]):
            mask=(df["close"]>df["ma50"])&(df["ma50"]>df["ma100"])&(df["ma100"]>df["ma200"])
    elif ftype=="combo_risetrend":
        if "ma50" in df.columns:
            mask=(df["ma50"]>df["ma50"].shift(5))&(df["close"]>df["ma50"])
    elif ftype=="combo_lag":
        if "ma50" in df.columns and "ma200" in df.columns:
            mask=(df["ma50"]>df["ma200"])&(df["ma50"].shift(5)>df["ma200"].shift(5))
    return mask

def test_filter(df,fname,fp,sp,ftype,label):
    mask=apply_filter(fname,fp,sp,ftype,df)
    if mask.sum()<3: return None
    sub=df[mask]
    p=sub["pnl"]; n=len(p); net=p.sum(); wr=(p>0).mean()
    aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0
    wl=aw/abs(al) if al!=0 else 999
    pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999
    # Filter info
    pct_trades=n/len(df)
    return {"Filter":fname,"Trades":n,"Pct":f"{pct_trades:.0%}","Net":f"{net:+,.0f}",
            "WR":f"{wr:.0%}","AvgW":f"{aw:,.0f}","AvgL":f"{al:,.0f}","W/L":f"{wl:.1f}x",
            "PF":f"{pf:.1f}x","MDD":f"{mdd:,.0f}","Calmar":f"{calmar:.1f}x",
            "NetChg":f"{net-df['pnl'].sum():+,.0f}"}

results=[]
for fname,fp,sp,ftype,cond,strict in filters:
    r=test_filter(df,fname,fp,sp,ftype,"")
    if r: results.append(r)

# Baseline
p=df["pnl"]; n=len(p); net=p.sum(); wr=(p>0).mean()
aw=p[p>0].mean() if (p>0).sum()>0 else 0; al=p[p<0].mean() if (p<0).sum()>0 else 0
wl=aw/abs(al) if al!=0 else 999
pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
calmar=net/mdd if mdd>0 else 999
results.insert(0,{"Filter":"NO FILTER (BASELINE)","Trades":n,"Pct":"100%","Net":f"{net:+,.0f}",
                   "WR":f"{wr:.0%}","AvgW":f"{aw:,.0f}","AvgL":f"{al:,.0f}","W/L":f"{wl:.1f}x",
                   "PF":f"{pf:.1f}x","MDD":f"{mdd:,.0f}","Calmar":f"{calmar:.1f}x","NetChg":"--"})

rdf=pd.DataFrame(results).sort_values("Net",key=lambda x: x.str.replace(",","").str.replace("+","").astype(float),ascending=False)

print("\n"+"="*120)
print("ALL MA DIRECTIONAL FILTERS - RANKED BY NET POINTS (Engulf_Raw CH55, Full Data)")
print("="*120)
cols=["Filter","Trades","Pct","Net","WR","W/L","PF","MDD","Calmar","NetChg"]
for c in cols: rdf[c]=rdf[c].astype(str)
print(f"{'Rank':>4} {'Filter':<42} {'Trades':>7} {'Pct':>5} {'Net':>12} {'WR':>5} {'W/L':>6} {'PF':>6} {'MDD':>10} {'Calmar':>8} {'NetChg':>12}")
print("-"*120)
for i,(_,r) in enumerate(rdf.iterrows()):
    rank=i+1
    hl="<<<" if i==0 or r["Filter"]=="NO FILTER (BASELINE)" else ""
    # highlight best
    is_best="*" if i==1 else " "
    print(f"{rank:>3}{is_best} {r['Filter']:<42} {r['Trades']:>7} {r['Pct']:>5} {r['Net']:>12} {r['WR']:>5} {r['W/L']:>6} {r['PF']:>6} {r['MDD']:>10} {r['Calmar']:>8} {r['NetChg']:>12}")

# ====== TEST ON JANUARY SPECIFICALLY ======
print("\n"+"="*120)
print("JANUARY-ONLY: MA DIRECTIONAL FILTER RESULTS (Engulf_Raw CH55)")
print("="*120)
jan=df[df["mo"]==1].copy()
if len(jan)>0:
    jan_results=[]
    for fname,fp,sp,ftype,cond,strict in filters:
        mask=apply_filter(fname,fp,sp,ftype,jan)
        if mask.sum()<2: continue
        sub=jan[mask]
        p=sub["pnl"]; n=len(p); net=p.sum(); wr=(p>0).mean(); aw=p[p>0].mean() if (p>0).sum()>0 else 0
        al=p[p<0].mean() if (p<0).sum()>0 else 0; wl=aw/abs(al) if al!=0 else 999
        cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
        pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
        calmar=net/mdd if mdd>0 else 999
        jan_results.append({"Filter":fname,"Trades":n,"Net":f"{net:+,.0f}","WR":f"{wr:.0%}",
                            "W/L":f"{wl:.1f}x","PF":f"{pf:.1f}x","MDD":f"{mdd:,.0f}","Calmar":f"{calmar:.1f}x"})
    # Jan baseline
    p=jan["pnl"]; n=len(p); net=p.sum(); wr=(p>0).mean(); aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0; wl=aw/abs(al) if al!=0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999; calmar=net/mdd if mdd>0 else 999
    jan_results.insert(0,{"Filter":"NO FILTER (JAN BASELINE)","Trades":n,"Net":f"{net:+,.0f}",
                          "WR":f"{wr:.0%}","W/L":f"{wl:.1f}x","PF":f"{pf:.1f}x","MDD":f"{mdd:,.0f}","Calmar":f"{calmar:.1f}x"})
    jdf=pd.DataFrame(jan_results).sort_values("Net",key=lambda x: x.str.replace(",","").str.replace("+","").astype(float) if x.dtype==object else x,ascending=False)
    print(f"{'Rank':>4} {'Filter':<42} {'Trades':>7} {'Net':>12} {'WR':>5} {'W/L':>6} {'PF':>6} {'MDD':>10} {'Calmar':>8}")
    print("-"*105)
    for i,(_,r) in enumerate(jdf.iterrows()):
        hl="*" if i==0 else " "; b="<<" if r["Filter"]=="NO FILTER (JAN BASELINE)" else ""
        print(f"{i+1:>3}{hl} {r['Filter']:<42} {r['Trades']:>7} {r['Net']:>12} {r['WR']:>5} {r['W/L']:>6} {r['PF']:>6} {r['MDD']:>10} {r['Calmar']:>8}")

# ====== TOP 5 FILTERS DETAILED BREAKDOWN ======
print("\n"+"="*120)
print("TOP 5 FILTERS - DETAILED MONTHLY+ YEARLY BREAKDOWN")
print("="*120)
top5=rdf.iloc[1:6]["Filter"].values if len(rdf)>5 else rdf["Filter"].values
for fname in top5:
    # Find filter definition
    match=None
    for fn,fp,sp,ftype,cond,strict in filters:
        if fn==fname: match=(fn,fp,sp,ftype,cond,strict); break
    if match is None: continue
    _,fp,sp,ftype,_,_=match
    mask=apply_filter(fname,fp,sp,ftype,df)
    filt_df=df[mask].copy()
    if len(filt_df)<5: continue
    print(f"\n--- {fname} ({len(filt_df)} trades, Net: {filt_df['pnl'].sum():+,.0f}) ---")
    # Monthly
    mo=filt_df.groupby("mo").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    MONS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"  Monthly: { {MONS[m-1]:f'{v["net"]:+,.0f}' for m,v in mo.iterrows()} }")
    # Yearly
    yr=filt_df.groupby("yr").agg(trades=("pnl","count"),net=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()))
    yr_prof=(yr["net"]>0).sum()/len(yr)
    print(f"  Years profitable: {yr_prof:.0%}")

print("\n"+"="*120)
print("SUMMARY: BEST FILTERS THAT FIX JANUARY")
print("="*120)
if len(jan_results)>0:
    jdf_sorted=pd.DataFrame(jan_results).sort_values("Net",key=lambda x: x.str.replace(",","").str.replace("+","").astype(float) if x.dtype==object else x,ascending=False)
    for i,(_,r) in enumerate(jdf_sorted.iterrows()):
        if i>6: break
        net_v=float(r["Net"].replace(",",""))
        print(f"  {r['Filter']:<45} JanNet={r['Net']:>12}  WR={r['WR']:>5}  W/L={r['W/L']:>6}  Calmar={r['Calmar']:>8}")
