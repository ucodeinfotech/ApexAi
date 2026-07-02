import pandas as pd, numpy as np, os
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

# Load data
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for df in [h1,m5]:
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)

# Pure engulfing signals
body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
sigs=[]
for i in range(1,len(h1)):
    if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1]:continue
    if h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
    if body.iloc[i]<body.iloc[i-1]*0.5:continue
    sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
print(f"Signals: {len(sigs)}")

# Execute first signal step by step
sig=sigs[0]
t,lv=sig["trigger_time"],sig["level"]
print(f"Signal time: {t}  level: {lv}")
print(f"Signal candle idx: {sig['idx']}")
si=sig['idx']
print(f"Signal candle: open={h1['open'].iloc[si]} high={h1['high'].iloc[si]} low={h1['low'].iloc[si]} close={h1['close'].iloc[si]}")

# Find first m5 bar after signal
mask=m5["datetime"]>t
print(f"m5 bars after signal: {mask.sum()}")
idx=mask.idxmax()
print(f"First after signal: idx={idx} time={m5['datetime'].iloc[idx]}")

# Breakout scan
m5_epoch=m5["datetime"].astype('int64').values
t_ep=t.asm8.astype('int64')
print(f"t_ep={t_ep}, m5_epoch[idx]={m5_epoch[idx]}")
print(f"Searchsorted result: {np.searchsorted(m5_epoch,t_ep,side='right')}")

# Check the breakout/retest logic with working code
m5_cl=m5["close"].values;m5_lo=m5["low"].values;m5_hi=m5["high"].values
m5_du=m5["datetime"].values;tc=pd.Series(m5["datetime"]).dt.time.values
CUT=pd.Timestamp("14:15").time()

tu=t;lv_val=lv
idx2=np.searchsorted(m5_du,tu,side="right")
print(f"\nUsing np.searchsorted on m5_du array:")
print(f"  idx={idx2}, time={m5_du[idx2]}")

b=idx2
while b<len(m5) and m5_cl[b]<=lv_val:
    b+=1
print(f"  Breakout at idx={b}, time={m5_du[b] if b<len(m5) else 'N/A'} close={m5_cl[b] if b<len(m5) else 'N/A'}")
if b<len(m5):
    r=b+1
    found=False
    while r<len(m5):
        if m5_lo[r]<lv_val and m5_cl[r]>lv_val and tc[r]<CUT:
            found=True
            break
        r+=1
    print(f"  Retest at idx={r}, time={m5_du[r] if found else 'N/A'}")
    if found:
        ep=m5_cl[r];sl=m5_lo[r]
        print(f"  Entry price={ep}, SL={sl}, risk={ep-sl}")
