import pandas as pd, numpy as np, os, warnings, glob
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

files = sorted(glob.glob("*_FIVE_MINUTE.csv"))
print("CSV files found:", files)

for sym in sorted(set(f.replace("_FIVE_MINUTE.csv","").replace("_ONE_HOUR.csv","") for f in files)):
    h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
    m5 = pd.read_csv(f"{sym}_FIVE_MINUTE.csv", parse_dates=["datetime"])
    print(f"\n{sym}:")
    print(f"  1h: {len(h1)} rows, {h1['datetime'].min()} -> {h1['datetime'].max()}")
    print(f"  5m: {len(m5)} rows, {m5['datetime'].min()} -> {m5['datetime'].max()}")
    print(f"  cols: {list(h1.columns)}")

VER = {"DynCH 25+10":(25,10),"DynCH 30+10":(30,10),"DynCH 30+15":(30,15),"DynCH 35+10":(35,10),"DynCH 35+15":(35,15),"DynCH 40+5":(40,5),"DynCH 40+10":(40,10),"DynCH 40+12":(40,12),"DynCH 45+5":(45,5),"DynCH 45+8":(45,8),"DynCH 45+10":(45,10),"DynCH 45+12":(45,12),"DynCH 45+15":(45,15),"DynCH 50+8":(50,8),"DynCH 50+10":(50,10),"DynCH 50+12":(50,12),"DynCH 55+10":(55,10),"DynCH 55+15":(55,15),"DynCH 60+10":(60,10),"DynCH 60+15":(60,15)}
VN=list(VER.keys()); CB=[VER[v][0] for v in VN]; CR=[VER[v][1] for v in VN]
CH_VALS=sorted(set(CB))

all_t = []
for sym in sorted(set(f.replace("_FIVE_MINUTE.csv","").replace("_ONE_HOUR.csv","") for f in glob.glob("*_FIVE_MINUTE.csv"))):
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"]); m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True); df.reset_index(drop=True,inplace=True)
    hl=h1["high"]-h1["low"];hpc=abs(h1["high"]-h1["close"].shift(1));lpc=abs(h1["low"]-h1["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);h1["atr14"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()
    a14=h1["atr14"].values;a20=pd.Series(a14).rolling(20).mean().values
    hl5=m5["high"]-m5["low"];hpc5=abs(m5["high"]-m5["close"].shift(1));lpc5=abs(m5["low"]-m5["close"].shift(1))
    tr5=pd.concat([hl5,hpc5,lpc5],axis=1).max(axis=1);m5_atr=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
    atr5=m5_atr.values;du=m5["datetime"].values;hi=m5["high"].values;lo=m5["low"].values;cl=m5["close"].values
    tc=pd.Series(m5["datetime"]).dt.time.values;bl=50 if "NIFTY" in sym else 10
    CUT=pd.Timestamp("14:15").time();prev_red=np.roll(h1["close"].values<h1["open"].values,1);prev_red[0]=False
    for i in range(60,len(h1)):
        if not (prev_red[i] and h1["close"].values[i]>h1["open"].values[i]): continue
        if not (h1["open"].values[i]<=h1["close"].values[i-1] and h1["close"].values[i]>=h1["open"].values[i-1]): continue
        if h1["high"].values[i]-h1["low"].values[i]<0.5*(h1["high"].values[i-1]-h1["low"].values[i-1]): continue
        lv=h1["high"].values[i];tu=h1["datetime"].values[i]
        idx=np.searchsorted(du,tu,side="right")
        if idx>=len(m5): continue
        b=idx
        while b<len(m5) and cl[b]<=lv: b+=1
        if b>=len(m5)-1: continue
        r=b+1
        while r<len(m5):
            _tc=tc[r] if not isinstance(tc[r],str) else pd.Timestamp(tc[r]).time()
            if lo[r]<lv and cl[r]>lv and _tc<CUT: break
            r+=1
        if r>=len(m5): continue
        ep=cl[r]
        if ep-lo[r]<=0: continue
        if h1["datetime"].iloc[i].hour==9: continue
        a14v=a14[i];a20v=a20[i];reg=0
        if not pd.isna(a14v) and not pd.isna(a20v) and a14v>a20v: reg=1
        elif not pd.isna(a14v): reg=2
        pnls={}
        for cv in CH_VALS:
            he=ep
            for j in range(r,len(m5)):
                ca=atr5[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-cv*ca:
                    pnls[cv]=(cl[j]-ep)*bl-20
                    break
        if 45 not in pnls: continue
        all_t.append({"dt":h1["datetime"].iloc[i],"sym":sym,"year":h1["datetime"].iloc[i].year,"bl":bl,"reg":reg,"pnls":pnls.copy(),
                       "ep":ep,"exit_pnl_45":pnls[45]})

print(f"\nTotal trades built: {len(all_t)}")
print(f"\n--- Per Symbol Per Year ---")
for sym in ["NIFTY50","SENSEX"]:
    ty=[t for t in all_t if t["sym"]==sym]
    print(f"\n{sym} ({len(ty)} total):")
    for y in sorted(set(t["year"] for t in all_t)):
        n=sum(1 for t in all_t if t["sym"]==sym and t["year"]==y)
        if n==0: continue
        pts45=[t["exit_pnl_45"] for t in all_t if t["sym"]==sym and t["year"]==y]
        # raw points = (pnl+20)/bl
        raw=[(p+20)/t["bl"] for t,p in zip([t for t in all_t if t["sym"]==sym and t["year"]==y],pts45)]
        print(f"  {y}: {n} trades | Rs sum={sum(pts45):>+10,.0f} | raw_pts sum={sum(raw):>+10,.1f} | reg1={sum(1 for t in all_t if t['sym']==sym and t['year']==y and t['reg']==1)} reg2={sum(1 for t in all_t if t['sym']==sym and t['year']==y and t['reg']==2)}")

print(f"\n--- Regime distribution ---")
for rv in [0,1,2]:
    print(f"  reg={rv}: {sum(1 for t in all_t if t['reg']==rv)} trades")

print(f"\n--- Per-symbol CH key availability ---")
for sym in ["NIFTY50","SENSEX"]:
    ty=[t for t in all_t if t["sym"]==sym]
    print(f"  {sym}:")
    for cv in CH_VALS:
        c=sum(1 for t in ty if cv in t["pnls"])
        print(f"    CH{cv}: {c}/{len(ty)}")

print(f"\nDone validation")
