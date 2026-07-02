"""
New filters test - V2: Ultra-fast, focused on key ideas only
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20; CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# Load data once
h1d={}; m5d={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["ema50"]=h1["close"].ewm(span=50).mean(); h1["ema200"]=h1["close"].ewm(span=200).mean()
    # VWAP
    if "volume" in h1.columns and h1["volume"].sum()>0:
        h1["vwap"]=(h1["volume"]*(h1["high"]+h1["low"]+h1["close"])/3).cumsum()/h1["volume"].cumsum()
    else:
        h1["vwap"]=h1["close"].expanding().mean()
    # MFI
    if "volume" in h1.columns and h1["volume"].sum()>0:
        tp=(h1["high"]+h1["low"]+h1["close"])/3
        rmf=tp*h1["volume"]
        pmf=rmf.where(tp>tp.shift(1),0).rolling(14).sum()
        nmf=rmf.where(tp<tp.shift(1),0).rolling(14).sum()
        h1["mfi14"]=100-(100/(1+pmf/(nmf+1e-10)))
    else:
        h1["mfi14"]=50.0
    h1["vol_ratio"]=h1["volume"]/h1["volume"].rolling(20).mean() if "volume" in h1.columns else 1.0
    h1d[sym]=h1; m5d[sym]=m5

print("Simulating trades...")
rows=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=h1d[sym]; m5=m5d[sym]
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=NLOT if "NIFTY" in sym else SLOT
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; tc=m5["datetime"].dt.time
    
    for i in range(50,len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*0.5: continue
        
        tu=int(pd.to_datetime(h1["datetime"].iloc[i]).timestamp())
        lv=h1["high"].iloc[i]
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
        ch=35 if (not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v) else 55 if (not pd.isna(atr14_v) and not pd.isna(atr_ma_v)) else 45
        
        he=ep
        for j in range(r+1,len(m5)):
            ca=atr5[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch*ca:
                pnl=(cl[j]-ep)*bl*1 - CHG*1
                rows.append({
                    "pnl":pnl,"win":1 if pnl>0 else 0,
                    "vol_ratio":h1["vol_ratio"].iloc[i] if not pd.isna(h1["vol_ratio"].iloc[i]) else 1.0,
                    "mfi14":h1["mfi14"].iloc[i] if not pd.isna(h1["mfi14"].iloc[i]) else 50.0,
                    "vwap_dist":(h1["close"].iloc[i]/h1["vwap"].iloc[i]-1)*100,
                    "below_vwap":1 if h1["close"].iloc[i]<h1["vwap"].iloc[i] else 0,
                    "above_ema50":1 if h1["close"].iloc[i]>h1["ema50"].iloc[i] else 0,
                    "above_ema200":1 if h1["close"].iloc[i]>h1["ema200"].iloc[i] else 0,
                    "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
                    "body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
                    "hour":h1["datetime"].iloc[i].hour,
                    "dow":h1["datetime"].iloc[i].dayofweek,
                    "day":h1["datetime"].iloc[i].day,
                    "month":h1["datetime"].iloc[i].month,
                })
                break

df=pd.DataFrame(rows)
print(f"Trades: {len(df)}, WR: {df['win'].mean()*100:.1f}%, Avg: Rs{df['pnl'].mean():+,.0f}")

BN=df["pnl"].sum(); BW=df["win"].mean()*100

# Test each filter
results=[]
def test(label, fn):
    sub=df[df.apply(fn,axis=1)].reset_index(drop=True)
    n=len(sub)
    if n<5: return
    net=sub["pnl"].sum(); wr=sub["win"].mean()*100
    pf=sub[sub["pnl"]>0]["pnl"].sum()/abs(sub[sub["pnl"]<0]["pnl"].sum()) if (sub["pnl"]<0).sum()!=0 else 99
    skip=(1-n/len(df))*100
    vs=(net/BN-1)*100
    results.append({"f":label,"n":n,"skip":skip,"net":net,"wr":wr,"pf":pf,"vs":vs})

# Volume
for t in [1.5,2.0]:
    test(f"Vol>{t}x", lambda r,t=t: r["vol_ratio"]>=t)
# MFI
test("MFI<50", lambda r: r["mfi14"]<50)
for t in [40,35,30]:
    test(f"MFI<={t}", lambda r,t=t: r["mfi14"]<=t)
# VWAP
test("BelowVWAP", lambda r: r["below_vwap"]==1)
# Trend
test("1H>EMA50", lambda r: r["above_ema50"]==1)
test("1H>EMA200", lambda r: r["above_ema200"]==1)
test("1H>EMA50+200", lambda r: r["above_ema50"]==1 and r["above_ema200"]==1)
# Gap
test("Gap<-0.2%", lambda r: r["gap_pct"]<-0.2)
test("Gap<-0.5%", lambda r: r["gap_pct"]<-0.5)
# Body ratio
test("Body>=1x", lambda r: r["body_ratio"]>=1.0)
test("Body>=1.5x", lambda r: r["body_ratio"]>=1.5)
# Combos
test("Trend+VWAP+MFI", lambda r: r["above_ema50"]==1 and r["below_vwap"]==1 and r["mfi14"]<50)
test("Trend+BelowVWAP", lambda r: r["above_ema50"]==1 and r["below_vwap"]==1)
test("Gap+MFI", lambda r: r["gap_pct"]<-0.2 and r["mfi14"]<50)
test("Gap+Body", lambda r: r["gap_pct"]<-0.2 and r["body_ratio"]>=1.0)
# Time/day filters (seasonality)
test("Hour10-12", lambda r: 10<=r["hour"]<=12)
test("Hour13-14", lambda r: 13<=r["hour"]<=14)
test("Hour9-10", lambda r: 9<=r["hour"]<=10)
test("Mon-Wed", lambda r: r["dow"]<=2)
test("Thu-Fri", lambda r: r["dow"]>=3)
# Month seasonality
for m in range(1,13):
    test(f"Month{m}", lambda r,m=m: r["month"]==m)

# Sorted results
print(f"\n{'='*95}")
print(f"{'Filter':30s}  {'Trades':>5s}  {'Skip%':>6s}  {'Net P&L':>11s}  {'WR%':>5s}  {'PF':>5s}  {'vsBase':>8s}")
print(f"{'-'*95}")
print(f"{'NO FILTER':30s}  {len(df):5d}  {'0%':>6s}  Rs{BN:>+9,.0f}  {BW:4.1f}%  {'-':>5s}  {'0.0%':>8s}")
sorted_r=sorted(results, key=lambda x: x["net"], reverse=True)
for r in sorted_r:
    print(f"{r['f']:30s}  {r['n']:5d}  {r['skip']:5.1f}%  Rs{r['net']:>+9,.0f}  {r['wr']:4.1f}%  {r['pf']:4.2f}  {r['vs']:>+7.1f}%")

# Winners vs Losers feature comparison
print(f"\n{'='*95}")
print("WINNERS vs LOSERS — Feature Means")
print(f"{'='*95}")
for col in ["vol_ratio","mfi14","vwap_dist","gap_pct","body_ratio"]:
    w=df[df["win"]==1][col].mean()
    l=df[df["win"]==0][col].mean()
    print(f"  {col:15s}: Winners={w:.4f}  Losers={l:.4f}  Diff={w-l:+.4f}")

print(f"\nDONE")
