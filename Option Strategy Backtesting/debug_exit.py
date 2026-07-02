"""Debug: why does simulate_pnl return None for most trades?"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

h1=pd.read_csv(os.path.join(BASE,"NIFTY50_ONE_HOUR.csv"))
m5=pd.read_csv(os.path.join(BASE,"NIFTY50_FIVE_MINUTE.csv"))
h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
body=(h1["close"]-h1["open"]).abs(); is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
atr5_fn=A(m5,14).values; tc=m5["datetime"].dt.time

trade_count=0; exit_count=0
for i in range(60, len(h1)):
    if trade_count>=10: break
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
    
    trade_count+=1
    he=ep; ch=45
    found=False
    for j in range(r, min(r+200, len(m5))):
        ca=atr5_fn[j]
        if pd.isna(ca): continue
        if hi[j]>he: he=hi[j]
        if cl[j]<he-ch*ca:
            pnl=(cl[j]-ep)*50-20
            exit_count+=1
            found=True
            break
    if not found:
        print(f"Trade {trade_count}: {h1['datetime'].iloc[i]} NO EXIT in 200 candles. "
              f"Entry={ep:.0f}, Last he={he:.0f}, ch={ch}, atr at end={atr5_fn[r+199]:.1f}")

print(f"\nTrades found: {trade_count}, Exits found: {exit_count}")
print(f"ATR stats: min={atr5_fn.min():.2f}, mean={atr5_fn.mean():.2f}, max={atr5_fn.max():.2f}")
print(f"NaN ATR: {pd.isna(atr5_fn).sum()}/{len(atr5_fn)}")
