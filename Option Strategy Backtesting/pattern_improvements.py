"""
TEST IMPROVEMENTS from discovered patterns:
- Adaptive Skip (Skip2 normally, Skip3 after 3+ consecutive losses)
- Sequence sizing (double after 3 consecutive wins)
- Friday boost (1.5x on Friday)
- Skip Q1 (Jan-Mar entire quarter)
- Pre-budget skip (Feb 1-7)
- Combined: all of the above
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def compute_atr20(h1):
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    return tr.rolling(20,min_periods=20).mean()

def compute_adx14(h1):
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    return 100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()

def compute_daily_ema(h1,period=50):
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

print("Loading data...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    DATA[sym]={"h1":h1,"m5":m5,"atr5v":atr5.values,"m5_hi":m5["high"].values,"m5_lo":m5["low"].values,
               "m5_cl":m5["close"].values,"m5_atr":atr5.values,
               "m5_epoch":m5["datetime"].astype('int64').values,"tc":pd.Series(m5["datetime"]).dt.time.values}
CUT=pd.Timestamp("14:15").time()

def find_retest(sym,t,lv):
    d=DATA[sym];m5_epoch=d["m5_epoch"];m5_cl=d["m5_cl"];m5_lo=d["m5_lo"];tc=d["tc"]
    t_ep=t.asm8.view('int64')
    idx=np.searchsorted(m5_epoch,t_ep,side="right")
    if idx>=len(m5_cl):return None
    b=idx
    while b<len(m5_cl) and m5_cl[b]<=lv:b+=1
    if b>=len(m5_cl)-1:return None
    r=b+1
    while r<len(m5_cl):
        if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
        r+=1
    if r>=len(m5_cl):return None
    ep=m5_cl[r];sl=m5_lo[r]
    if ep-sl<=0:return None
    return (r,ep,sl)

def compute_ch_exits(sym,r,ep):
    d=DATA[sym];m5_cl=d["m5_cl"];m5_hi=d["m5_hi"];m5_atr=d["m5_atr"]
    pnls={}
    for cv in CH_VALS:
        he=ep
        for j in range(r,len(m5_cl)):
            ca=m5_atr[j]
            if pd.isna(ca):continue
            if m5_hi[j]>he:he=m5_hi[j]
            if m5_cl[j]<he-cv*ca:
                pnls[cv]=round(m5_cl[j]-ep,1);break
    return pnls

def sigs_engulf_raw(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],
                    "yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month,
                    "dow":h1["datetime"].iloc[i].dayofweek,"dom":h1["datetime"].iloc[i].day,
                    "idx":i,"sym":sym})
    return out

print("Building trades...")
trades=[]
for sym in ["NIFTY50","SENSEX"]:
    for sig in sigs_engulf_raw(sym):
        ret=find_retest(sym,sig["ts"],sig["lv"])
        if ret is None:continue
        r,ep,sl=ret
        pnls=compute_ch_exits(sym,r,ep)
        if 45 not in pnls:continue
        t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"ts":sig["ts"],
           "dow":sig["dow"],"dom":sig["dom"]}
        for c,p in pnls.items():t[f"p{c}"]=p
        t["pts45"]=pnls.get(45,0)
        trades.append(t)
df=pd.DataFrame(trades)
print(f"Total: {len(df)} trades")

# === BACKTEST ENGINE (extended) ===
def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def run_bt(df, ch_val=55, use_wl=False, skip_n=0, size_fn=None, skip_q1=False, friday_boost=False,
           skip_feb1_7=False, triple_skip_loss=False, triple_win_size=False):
    """Extended backtest with pattern-based rules"""
    rows=[]
    for yr in sorted(df["yr"].unique()):
        yr_d=df[df["yr"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        rw=[];rl=[];skip_c=0;consec_losses=0;consec_wins=0
        for _,t in yr_d.iterrows():
            # Skip Q1
            if skip_q1 and t["mo"] in [1,2,3]:continue
            # Skip Feb 1-7
            if skip_feb1_7 and t["mo"]==2 and t["dom"]<=7:continue
            # Skip counter
            if skip_c>0:skip_c-=1;continue
            # Get PnL
            pts=t.get(f"p{ch_val}",t["pts45"])
            # Base size
            sz=1.0
            # WL sizing
            if use_wl:sz=wl_size(rw,rl,lb=5)
            # 2w1l sizing
            if size_fn=="2w1l":
                nw=sum(1 for x in rw[-2:] if x>0);nl=sum(1 for x in rl[-1:] if x<0)
                sz=1.5 if nw>=2 else (0.5 if nl>=1 else 1.0)
            # Triple win size (from pattern: after WWW -> 88.8% WR)
            if triple_win_size and consec_wins>=3:
                sz*=2.0
            # Friday boost
            if friday_boost and t["dow"]==4:
                sz*=1.5
            pnl=pts*sz
            rows.append({"pts":pnl,"yr":t["yr"],"mo":t["mo"],"sz":sz})
            # Track streaks
            if pnl>0:
                rw.append(pnl);consec_losses=0;consec_wins+=1
                # Adaptive: after win, normal skip (0)
                skip_c=0
            else:
                rl.append(abs(pnl));consec_losses+=1;consec_wins=0
                # Adaptive skip: if triple_skip_loss and >=3 consecutive losses, skip 3
                if triple_skip_loss and consec_losses>=3:
                    skip_c=3
                else:
                    skip_c=skip_n
    return pd.DataFrame(rows)

def compute_metrics(res):
    if len(res)==0:return {"net":0,"n":0,"wr":0,"wl":0,"mdd":0}
    net=res["pts"].sum();n=len(res);wr=(res["pts"]>0).mean()
    w=res[res["pts"]>0]["pts"];l=res[res["pts"]<0]["pts"]
    wl_r=w.mean()/abs(l.mean()) if len(l)>0 and l.mean()!=0 else 999
    cum=res["pts"].cumsum();mx=cum.cummax();mdd=(mx-cum).max()
    return {"net":net,"n":n,"wr":wr,"wl":wl_r,"mdd":mdd}

# === CONFIGURATIONS ===
configs=[
    ("CH55+WL+Skip2 (BASELINE)",  55, True,  2, None,  False, False, False, False, False),
    # From patterns
    ("+TripleSkipLoss",           55, True,  2, None,  False, False, False, True,  False),
    ("+TripleWinSize",            55, True,  2, None,  False, False, False, False, True),
    ("+SkipQ1",                   55, True,  2, None,  True,  False, False, False, False),
    ("+FridayBoost",              55, True,  2, None,  False, True,  False, False, False),
    ("+SkipFeb1_7",               55, True,  2, None,  False, False, True,  False, False),
    # Combined simple
    ("+SkipQ1+FriBoost",          55, True,  2, None,  True,  True,  False, False, False),
    ("+TripleSkip+TripleWin",     55, True,  2, None,  False, False, False, True,  True),
    # ALL combined (no Q1 skip - too aggressive)
    ("+ALL-exceptQ1",             55, True,  2, None,  False, True,  True,  True,  True),
    # With 2w1l sizing instead of WL
    ("2w1l+TripleSkip+TripleWin", 55, False, 2, "2w1l",False, False, False, True,  True),
    # 2w1l + Q1 skip
    ("2w1l+SkipQ1+FriBoost",      55, False, 2, "2w1l",True,  True,  False, False, False),
]

print("\n"+"="*130)
print("PATTERN-BASED IMPROVEMENTS TEST")
print("="*130)
print(f"{'#':>3} {'Config':<32} {'Trades':>7} {'Net Pts':>12} {'WR':>6} {'W/L':>7} {'MDD':>9} {'Net/MDD':>8}")
print("-"*130)

results=[]
for cname, cv, wl, skip, szfn, sq1, fb, sf17, tsl, tws in configs:
    res=run_bt(df, ch_val=cv, use_wl=wl, skip_n=skip, size_fn=szfn,
               skip_q1=sq1, friday_boost=fb, skip_feb1_7=sf17,
               triple_skip_loss=tsl, triple_win_size=tws)
    m=compute_metrics(res)
    nm=m["net"]/m["mdd"] if m["mdd"]>0 else 999
    results.append((cname,m["net"],m["wr"],m["wl"],m["mdd"],m["n"],nm))
    print(f"{len(results):>3} {cname:<32} {m['n']:>7} {m['net']:>+11,.0f}  {m['wr']:>5.1%} {m['wl']:>5.1f}x {m['mdd']:>+7,.0f}  {nm:>6.1f}x")

# Sort by net
results.sort(key=lambda r: -r[1])
print("\n"+"="*130)
print("RANKED BY NET PNL")
print("="*130)
for i,(cname,net,wr,wl,mdd,n,nm) in enumerate(results,1):
    chg=""
    if i==1:chg=" <<< BEST"
    elif i<len(results):
        base_net=results[-1][1] if results[-1][0].startswith("CH55+WL+Skip2 (BASELINE)") else 0
        for rn,rn_net,_,_,_,_,_ in results:
            if "BASELINE" in rn:base_net=rn_net;break
        if base_net and net>base_net:chg=f" +{net-base_net:+,.0f} vs baseline"
    print(f"{i:>2}. {cname:<32s} Net={net:>+11,.0f} WR={wr:>5.1%} W/L={wl:>5.1f}x MDD={mdd:>+7,.0f} Net/MDD={nm:>6.1f}x{chg}")

# Find baseline net
base_net=0
for cname,net,_,_,_,_,_ in results:
    if "BASELINE" in cname:base_net=net;break
if base_net:
    print(f"\n  Baseline net: {base_net:+,.0f}")
    for cname,net,wr,wl,mdd,n,nm in results:
        if "BASELINE" in cname:continue
        chg=net-base_net
        if chg>0:
            print(f"  >>> {cname:<32s} BEATS BASELINE by {chg:+,.0f} pts ({chg/base_net*100:+.1f}%)")

print("\n"+"="*130)
print("SUMMARY: WHAT TO IMPLEMENT")
print("="*130)
print("""
  FROM THIS TEST:

  1. TripleSkipLoss (skip 3 after 3+ consecutive losses)
     - Pros: breaks 23-loss streak pattern
     - Cons: may skip too many trades

  2. TripleWinSize (double after 3 consecutive wins)
     - Pros: 88.8% WR after WWW sequence
     - Cons: higher MDD if streak breaks

  3. SkipQ1 (skip January-March)
     - Pros: Q1 net = -23K (worst quarter)
     - Cons: misses occasional good trades

  4. FridayBoost (1.5x on Fridays)
     - Pros: Friday WR 53.2% (best day)
     - Cons: moderate gain

  5. SkipFeb1_7 (skip budget week)
     - Pros: WR only 23.8% that week
     - Cons: only 42 trades affected
""")

print("="*130)
print("DONE")
print("="*130)
