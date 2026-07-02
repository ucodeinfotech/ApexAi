"""Debug test trades specifically"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

all_trades=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
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
        all_trades.append({"dt":h1["datetime"].iloc[i],"sym":sym,"r":r,"ep":ep,"bl":bl})

split=int(len(all_trades)*0.7)
test_trades=all_trades[split:]
print(f"Test trades: {len(test_trades)}")

# Simulate each test trade with CH=45
for sym in ["NIFTY50","SENSEX"]:
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    m5["datetime"]=pd.to_datetime(m5["datetime"])
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values
    for ti,t in enumerate([tt for tt in test_trades if tt["sym"]==sym]):
        if ti>=5: break
        r=t["r"]; ep=t["ep"]; bl=t["bl"]
        he=ep; ch=45; found=False
        for j in range(r, min(r+200, len(m5))):
            ca=atr5[j]
            if pd.isna(ca): print(f"  {t['dt']}: NaN ATR at j={j}"); continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch*ca:
                pnl=(cl[j]-ep)*bl*1-20
                print(f"  {t['dt']}: EXIT at+{j-r}, close={cl[j]:.0f} < {he-ch*ca:.0f}, pnl=Rs{pnl:+,.0f}")
                found=True; break
        if not found:
            last_stop=he-ch*atr5[min(r+199,len(m5)-1)]
            print(f"  {t['dt']}: NO EXIT, ep={ep:.0f}, he={he:.0f}, stop={last_stop:.0f}, last_c={cl[min(r+199,len(m5)-1)]:.0f}")
