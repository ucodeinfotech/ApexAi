"""
COMPREHENSIVE: ALL STRATEGIES UPDATED (points)
Tests every entry method with CH45 baseline vs W/L + DynCH fix
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def build_entry_trades(name, detect_fn, load_all_ch=True):
    """Generic trade builder. For each entry signal, computes CH exit for all CH_VALS."""
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
        m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
        atr5=compute_atr(m5);atr5v=atr5.values;m5_hi=m5["high"].values;m5_lo=m5["low"].values;m5_cl=m5["close"].values;m5_du=m5["datetime"].values
        m5_epoch=m5["datetime"].astype('int64').values
        tc=pd.Series(m5["datetime"]).dt.time.values;CUT=pd.Timestamp("14:15").time()
        sigs=detect_fn(sym, h1)
        if sym=="NIFTY50":print(f"    {sname} {sym}: {len(sigs)} signals",end=" ")
        for sig in sigs:
            t,lv=sig["trigger_time"],sig["level"]
            t_ep=t.asm8.view('int64')
            idx=np.searchsorted(m5_epoch,t_ep,side="right")
            if idx>=len(m5):continue
            b=idx
            while b<len(m5) and m5_cl[b]<=lv:b+=1
            if b>=len(m5)-1:continue
            r=b+1
            while r<len(m5):
                if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
                r+=1
            if r>=len(m5):continue
            ep=m5_cl[r]
            if ep-m5_lo[r]<=0:continue
            pnls={}
            for cv2 in CH_VALS:
                he=ep
                for j in range(r,len(m5)):
                    ca=atr5v[j]
                    if pd.isna(ca):continue
                    if m5_hi[j]>he:he=m5_hi[j]
                    if m5_cl[j]<he-cv2*ca:
                        pnls[cv2]=round(m5_cl[j]-ep,1);break
            if 45 not in pnls:continue
            tdict={"sym":sym,"year":t.year,"month":t.month,
                   "pts45":pnls[45],"ts":t}
            for c,p in pnls.items():tdict[f"p{c}"]=p
            all_t.append(tdict)
    return pd.DataFrame(all_t).fillna(0)

# ═══════════════════════════════════════════════
# ENTRY DETECTION FUNCTIONS (each returns list of dicts with trigger_time, level)
# ═══════════════════════════════════════════════

def detect_engulfing(sym, h1):
    """Pure bullish engulfing (no filters)."""
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1]:continue
        if h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    return sigs

def detect_engulfing_filt(sym, h1):
    """Engulfing with filters: EMA50>200, ADX>20, session 9:30-12:30."""
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    # ATR20, ADX14, EMA50/200
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df=h1.copy();df["date"]=h1["datetime"].dt.normalize()
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
    sigs=[]
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1]:continue
        if h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    return sigs

def detect_big_candle(sym, h1):
    """Big Candle Reversal: prev body > 1.5x avg_body(20), BUY only."""
    body=(h1["close"]-h1["open"]).abs();avg_body=body.rolling(20,min_periods=20).mean()
    is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if pd.isna(avg_body.iloc[i]):continue
        if not is_red.iloc[i-1]:continue
        if not is_green.iloc[i]:continue
        if body.iloc[i-1]<=avg_body.iloc[i-1]*1.5:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
        if h1["close"].iloc[i]<mid:continue
        if (h1["open"].iloc[i]-h1["low"].iloc[i])>body.iloc[i]*0.5:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    return sigs

def detect_sir(sym, h1):
    """Sir strategy: ATR(20) body threshold + all filters (EMA50>200, ADX>20, session)."""
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"]
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df=h1.copy();df["date"]=h1["datetime"].dt.normalize()
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
    sigs=[]
    for i in range(1,len(h1)):
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if not is_green.iloc[i]:continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        if not(h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue  # prev bearish
        if body.iloc[i-1]<=1.0*atr20.iloc[i]:continue  # must be big body
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
        if h1["close"].iloc[i]<mid:continue
        if (h1["open"].iloc[i]-h1["low"].iloc[i])>body.iloc[i]*0.5:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    return sigs

def detect_and(sym, h1):
    """Combined AND: both engulfing AND big candle."""
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"]
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df=h1.copy();df["date"]=h1["datetime"].dt.normalize()
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
    sigs=[]
    for i in range(1,len(h1)):
        if not(is_green.iloc[i] and h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue
        # Big Candle checks
        bc_ok=False
        if not pd.isna(atr20.iloc[i]) and body.iloc[i-1]>1.0*atr20.iloc[i]:
            if body.iloc[i]>=body.iloc[i-1]*0.5:
                mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
                if h1["close"].iloc[i]>=mid and (h1["open"].iloc[i]-h1["low"].iloc[i])<=body.iloc[i]*0.5:
                    bc_ok=True
        # Engulfing checks
        eng_ok = (h1["open"].iloc[i]<=h1["close"].iloc[i-1] and h1["close"].iloc[i]>=h1["open"].iloc[i-1]
                  and body.iloc[i]>=body.iloc[i-1]*0.5)
        if not (bc_ok and eng_ok):continue
        # Filters
        if pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    return sigs

def detect_or(sym, h1):
    """Combined OR: either engulfing OR big candle."""
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"]
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df=h1.copy();df["date"]=h1["datetime"].dt.normalize()
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
    sigs=[]
    for i in range(1,len(h1)):
        if not(is_green.iloc[i] and h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue
        # Big Candle
        bc_ok=False
        if not pd.isna(atr20.iloc[i]) and body.iloc[i-1]>1.0*atr20.iloc[i]:
            if body.iloc[i]>=body.iloc[i-1]*0.5:
                mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
                if h1["close"].iloc[i]>=mid and (h1["open"].iloc[i]-h1["low"].iloc[i])<=body.iloc[i]*0.5:
                    bc_ok=True
        # Engulfing
        eng_ok = (h1["open"].iloc[i]<=h1["close"].iloc[i-1] and h1["close"].iloc[i]>=h1["open"].iloc[i-1]
                  and body.iloc[i]>=body.iloc[i-1]*0.5)
        if not (bc_ok or eng_ok):continue
        if pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    return sigs

# ═══════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════
def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def run_bt(df, use_wl, use_dynch):
    rows=[]
    for yr in sorted(df["year"].unique()):
        yr_d=df[df["year"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        month_best={}
        if use_dynch:
            hist=df[df["year"]<yr]
            for m in range(1,13):
                sub=hist[hist["month"]==m]
                if len(sub)<5:month_best[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_best[m]=best
        rw=[];rl=[]
        for _,t in yr_d.iterrows():
            pts=t["pts45"]
            if use_dynch:
                m_ch=month_best.get(t["month"],45)
                pts=t.get(f"p{m_ch}",t["pts45"])
            sz=1.0
            if use_wl:sz=wl_size(rw,rl,lb=5)
            pnl_pts=pts*sz
            rows.append({"pts":pnl_pts,"year":t["year"],"month":t["month"],"size":sz,"pts45":t["pts45"]})
            if pnl_pts>0:rw.append(pnl_pts)
            else:rl.append(abs(pnl_pts))
    return pd.DataFrame(rows)

def calc_stats(res):
    pk=0;rn=0;mdd=0
    for r in res["pts"]:rn+=r;pk=max(pk,rn);mdd=max(mdd,pk-rn)
    w=res[res["pts"]>0]["pts"];l=res[res["pts"]<0]["pts"]
    aw=w.mean() if len(w)>0 else 0;al=abs(l.mean()) if len(l)>0 else 0
    wl=aw/al if al>0 else float('inf')
    wr=(res["pts"]>0).mean()
    ann_ret=res.groupby("year")["pts"].sum().mean()
    ann_vol=res.groupby("year")["pts"].sum().std()
    sharpe=ann_ret/ann_vol if ann_vol>0 else 0
    ms=0;cur=0
    for _,r in res.iterrows():
        if r["pts"]<=0:cur+=1;ms=max(ms,cur)
        else:cur=0
    return {"net":res["pts"].sum(),"wr":wr,"wl":wl,"mdd":mdd,"sharpe":sharpe,"avg_w":aw,"avg_l":al,"max_cons":ms,"n":len(res),"avg_sz":res["size"].mean()}

# ═══════════════════════════════════════════════
# RUN ALL STRATEGIES
# ═══════════════════════════════════════════════
strategies = [
    ("Engulfing", detect_engulfing),
    ("Engulf_Filt", detect_engulfing_filt),
    ("BigCandle", detect_big_candle),
    ("Sir", detect_sir),
    ("Comb_AND", detect_and),
    ("Comb_OR", detect_or),
]

print("="*120)
print("ALL STRATEGIES UPDATED — POINTS RESULTS")
print(f"{'='*120}\n")

all_rows=[]
for sname, sfunc in strategies:
    print(f"  Building {sname}...", flush=True)
    df=build_entry_trades(sname, sfunc)
    print(f"    Total trades: {len(df)}")
    if len(df)<10:
        print(f"SKIP (only {len(df)} trades)")
        continue
    base=run_bt(df, use_wl=False, use_dynch=False)
    fix=run_bt(df, use_wl=True, use_dynch=True)
    bs=calc_stats(base);fs=calc_stats(fix)
    impr=((fs["net"]/bs["net"]-1)*100) if bs["net"]!=0 else 0
    all_rows.append({
        "Strategy":sname,"Trades":len(df),
        "Base_Net":bs["net"],"Fix_Net":fs["net"],"Delta":fs["net"]-bs["net"],"Impr%":impr,
        "Base_WR":bs["wr"],"Fix_WR":fs["wr"],
        "Base_WL":bs["wl"],"Fix_WL":fs["wl"],
        "Base_MDD":bs["mdd"],"Fix_MDD":fs["mdd"],
        "Base_Sharp":bs["sharpe"],"Fix_Sharp":fs["sharpe"],
        "Avg_Size":fs["avg_sz"],"Max_Cons":bs["max_cons"],
        "Base_AvgW":bs["avg_w"],"Fix_AvgW":fs["avg_w"],
        "Base_AvgL":bs["avg_l"],"Fix_AvgL":fs["avg_l"],
    })
    arrow=" -> "
    print(f"{len(df):4d} trades | Base={bs['net']:>+8,.0f}{arrow}Fix={fs['net']:>+8,.0f} ({impr:+.1f}%) WR={bs['wr']:.0%} WRf={fs['wr']:.0%} WLb={bs['wl']:.1f}x WLf={fs['wl']:.1f}x MDDb={bs['mdd']:>+7,.0f} MDDf={fs['mdd']:>+7,.0f}")

    # Year-by-year for each
    print(f"    Year    Base     Fix     Delta")
    for yr in sorted(df["year"].unique()):
        b=base[base["year"]==yr]["pts"].sum()
        f=fix[fix["year"]==yr]["pts"].sum()
        print(f"    {yr:4d}  {b:>+8,.0f} {f:>+8,.0f} {f-b:>+8,.0f}")
    print()

# ═══════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════
print("="*120)
print("SUMMARY: ALL STRATEGIES RANKED BY FIXED NET")
print("="*120)
cols=["Strategy","Trades","Base_Net","Fix_Net","Delta","Impr%","Base_WR","Fix_WR","Base_WL","Fix_WL","Base_MDD","Fix_MDD","Avg_Size"]
print(f"{'Strategy':<15s} {'Trades':>6s} {'BaseNet':>9s} {'FixNet':>9s} {'Delta':>9s} {'Impr%':>7s} {'WRb':>4s} {'WRf':>4s} {'WLb':>5s} {'WLf':>5s} {'MDDb':>8s} {'MDDf':>8s} {'Sz':>4s}")
print("-"*120)
for row in sorted(all_rows, key=lambda r: -r["Fix_Net"]):
    print(f"{row['Strategy']:<15s} {row['Trades']:6d} {row['Base_Net']:>+8,.0f} {row['Fix_Net']:>+8,.0f} {row['Delta']:>+8,.0f} {row['Impr%']:>+6.1f}% {row['Base_WR']:.0%} {row['Fix_WR']:.0%} {row['Base_WL']:>4.1f}x {row['Fix_WL']:>4.1f}x {row['Base_MDD']:>+7,.0f} {row['Fix_MDD']:>+7,.0f} {row['Avg_Size']:.2f}")

print(f"\n{'='*120}")
print(f"BEST STRATEGY: {max(all_rows, key=lambda r: r['Fix_Net'])['Strategy']}")
print(f"BEST IMPROVEMENT: {max(all_rows, key=lambda r: r['Impr%'])['Strategy']}")
print(f"{'='*120}")
