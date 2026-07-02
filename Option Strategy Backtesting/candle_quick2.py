import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()
RS=42; np.random.seed(RS)

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

h1d={}; m5d={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["ema50"]=h1["close"].ewm(span=50).mean(); h1["ema200"]=h1["close"].ewm(span=200).mean()
    h1d[sym]=h1; m5d[sym]=m5

records=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=h1d[sym]; m5=m5d[sym]
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
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
                records.append({"date":h1["datetime"].iloc[i],"sym":sym,"pnl":pnl,"win":1 if pnl>0 else 0,
                    "s_body":body.iloc[i],"p_body":body.iloc[i-1],
                    "body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
                    "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
                    "hour":h1["datetime"].iloc[i].hour,"month":h1["datetime"].iloc[i].month,"year":h1["datetime"].iloc[i].year,
                    "c_ema50":(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100,
                    "c_ema200":(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100,
                    "trend_5":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0})
                break

df=pd.DataFrame(records)
print(f"Total: {len(df)}, WR={df.win.mean()*100:.1f}%, Net=Rs{df.pnl.sum():+,.0f}")

print(f"\n--- WORST 20 LOSSES ---")
for _,r in df.nsmallest(20,"pnl").iterrows():
    d=pd.Timestamp(r.date)
    print(f"  {d.date()} {r.sym:>7s} | PnL=Rs{r.pnl:>+8,.0f} | Body={r.s_body:.0f}/{r.p_body:.0f} | Gap={r.gap_pct:+.2f}% | Trend5={r.trend_5:+.2f}% | {'ABV' if r.c_ema50>0 else 'BLW'}50")

print(f"\n--- WINNER VS LOSER ---")
w=df[df.win==1]; l=df[df.win==0]
for col in ["s_body","p_body","body_ratio","gap_pct","trend_5","c_ema50","c_ema200"]:
    wm=w[col].mean(); lm=l[col].mean()
    print(f"  {col:>15s}: W={wm:>+.2f} L={lm:>+.2f} Diff={wm-lm:>+.2f}")

print(f"\n--- FEATURE SWEEP (WR spread > 10%) ---")
for col in ["s_body","body_ratio","gap_pct","trend_5","c_ema50","c_ema200"]:
    df["_b"]=pd.qcut(df[col].rank(method="first"), q=10, labels=False, duplicates="drop")
    wr=df.groupby("_b")["win"].mean(); span=wr.max()-wr.min()
    if span>0.10:
        print(f"  {col:>15s}: span={span*100:.1f}% best={wr.max()*100:.1f}% worst={wr.min()*100:.1f}%")
    del df["_b"]

print(f"\n--- P&L DISTRIBUTION ---")
p=df.pnl
print(f"  Min={p.min():>+,.0f} Max={p.max():>+,.0f} Mean={p.mean():>+,.0f} Median={p.median():>+,.0f}")
print(f"  Top10={p.nlargest(10).sum()/p.sum()*100:.1f}% Top50={p.nlargest(50).sum()/p.sum()*100:.1f}%")
print(f"  Bottom10={p.nsmallest(10).sum()/p.sum()*100:.1f}% Bottom50={p.nsmallest(50).sum()/p.sum()*100:.1f}%")

print(f"\n--- TOP 10 WINNERS ---")
top10=df.nlargest(10,"pnl")
for _,r in top10.iterrows():
    d=pd.Timestamp(r.date)
    print(f"  {d.date()} {r.sym:>7s} | PnL=Rs{r.pnl:>+8,.0f} | Body={r.s_body:.0f}/{r.p_body:.0f} | Gap={r.gap_pct:+.2f}% | Trend5={r.trend_5:+.2f}% | {'ABV' if r.c_ema50>0 else 'BLW'}50")
print(f"  Avg body_ratio={top10.body_ratio.mean():.1f}x (all={df.body_ratio.mean():.1f}x)")
print(f"  Avg gap={top10.gap_pct.mean():+.2f}% (all={df.gap_pct.mean():+.2f}%)")
print(f"  % above EMA50: {(top10.c_ema50>0).mean()*100:.0f}% (all={(df.c_ema50>0).mean()*100:.0f}%)")
print(f"  % EMA50>2%: {((top10.c_ema50>2).mean()*100):.0f}%")

print(f"\n--- MONTHLY VARIATION ---")
for m,grp in df.groupby("month"):
    print(f"  Month {m:>2d}: n={len(grp):>4d} WR={grp.win.mean()*100:.1f}% Net=Rs{grp.pnl.sum():>+9,.0f}")

print(f"\n--- SIMPLE FILTER SEARCH (top 10) ---")
ffs={"body_ratio_ge":lambda df,v:df[df.body_ratio>=v],"gap_le":lambda df,v:df[df.gap_pct<=v],
     "trend5_ge":lambda df,v:df[df.trend_5>=v],"trend5_le":lambda df,v:df[df.trend_5<=v],
     "above_ema50":lambda df,v:df[df.c_ema50>0] if v else df[df.c_ema50<=0]}
best_r=[]
for fn,ff in ffs.items():
    vals=[True,False] if "above" in fn else ([-5,-2,-1,-0.5,0,0.5,1,2,5] if "trend" in fn else [0.5,1,2,3,5,10])
    for v in vals:
        try: sub=ff(df,v)
        except: continue
        if len(sub)<20 or len(sub)>len(df)-20: continue
        imp=(sub.pnl.sum()/df.pnl.sum()-1)*100
        best_r.append((fn,v,len(sub),sub.win.mean(),sub.pnl.sum(),imp))
best_r.sort(key=lambda x:-x[5])
print(f"  {'Filter':>20s} {'N':>5s} {'WR':>6s} {'Net':>12s} {'Imp':>8s}")
for fn,v,n,wr,net,imp in best_r[:10]:
    print(f"  {fn:>20s} {n:>5d} {wr*100:>5.1f}% Rs{net:>+9,.0f} {imp:>+7.1f}%")

print(f"\n  BEST: {best_r[0][0]}={best_r[0][1]}")
for yr in sorted(df.year.unique()):
    sub=df[df.year==yr]
    if len(sub)<10: continue
    fil=ffs[best_r[0][0]](sub,best_r[0][1])
    if len(fil)<5: continue
    imp2=(fil.pnl.sum()/sub.pnl.sum()-1)*100
    print(f"    {yr}: {len(fil)}/{len(sub)} WR={fil.win.mean()*100:.1f}% vs {sub.win.mean()*100:.1f}% Imp={imp2:+.1f}%")
print(f"\nDONE")
