"""
COMPREHENSIVE YEARLY + MONTHLY RETURNS — ALL STRATEGIES (points)
Baseline CH45 vs Fixed (W/L + DynCH) per strategy
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def build_entry_trades(name, detect_fn):
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
        m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
        atr5=compute_atr(m5);atr5v=atr5.values;m5_hi=m5["high"].values;m5_lo=m5["low"].values;m5_cl=m5["close"].values
        m5_epoch=m5["datetime"].astype('int64').values
        tc=pd.Series(m5["datetime"]).dt.time.values;CUT=pd.Timestamp("14:15").time()
        sigs=detect_fn(sym, h1)
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
            tdict={"sym":sym,"year":t.year,"month":t.month,"pts45":pnls[45],"ts":t}
            for c,p in pnls.items():tdict[f"p{c}"]=p
            all_t.append(tdict)
    return pd.DataFrame(all_t).fillna(0)

# ═══ ENTRY DETECTION ═══
def detect_engulfing(sym, h1):
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1]:continue
        if h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
    return sigs

def detect_engulfing_filt(sym, h1):
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df_["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df_["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
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
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
    return sigs

def detect_big_candle(sym, h1):
    body=(h1["close"]-h1["open"]).abs();avg_body=body.rolling(20,min_periods=20).mean()
    is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if pd.isna(avg_body.iloc[i]):continue
        if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
        if body.iloc[i-1]<=avg_body.iloc[i-1]*1.5:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
        if h1["close"].iloc[i]<mid:continue
        if (h1["open"].iloc[i]-h1["low"].iloc[i])>body.iloc[i]*0.5:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
    return sigs

def detect_sir(sym, h1):
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"]
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df_["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df_["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
    sigs=[]
    for i in range(1,len(h1)):
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if not is_green.iloc[i]:continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        if not(h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue
        if body.iloc[i-1]<=1.0*atr20.iloc[i]:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
        if h1["close"].iloc[i]<mid:continue
        if (h1["open"].iloc[i]-h1["low"].iloc[i])>body.iloc[i]*0.5:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
    return sigs

def detect_or(sym, h1):
    body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"]
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr20=tr.rolling(20,min_periods=20).mean()
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    adx14=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    ema50=df_["date"].map(daily["close"].ewm(span=50,adjust=False).mean()).values
    ema200=df_["date"].map(daily["close"].ewm(span=200,adjust=False).mean()).values
    sigs=[]
    for i in range(1,len(h1)):
        if not(is_green.iloc[i] and h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue
        bc_ok=False
        if not pd.isna(atr20.iloc[i]) and body.iloc[i-1]>1.0*atr20.iloc[i]:
            if body.iloc[i]>=body.iloc[i-1]*0.5:
                mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
                if h1["close"].iloc[i]>=mid and (h1["open"].iloc[i]-h1["low"].iloc[i])<=body.iloc[i]*0.5:bc_ok=True
        eng_ok = (h1["open"].iloc[i]<=h1["close"].iloc[i-1] and h1["close"].iloc[i]>=h1["open"].iloc[i-1]
                  and body.iloc[i]>=body.iloc[i-1]*0.5)
        if not (bc_ok or eng_ok):continue
        if pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
    return sigs

# ═══ BACKTEST ═══
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
            pnl=pts*sz
            rows.append({"pts":pnl,"year":t["year"],"month":t["month"],"size":sz})
            if pnl>0:rw.append(pnl)
            else:rl.append(abs(pnl))
    return pd.DataFrame(rows)

# ═══ RUN ═══
strategies = [
    ("Engulfing", detect_engulfing),
    ("Engulf_Filt", detect_engulfing_filt),
    ("BigCandle", detect_big_candle),
    ("Sir", detect_sir),
    ("Comb_OR", detect_or),
]

print("="*130)
print("ALL STRATEGIES — YEARLY + MONTHLY RETURNS (POINTS)")
print("="*130)

all_results={}
for sname, sfunc in strategies:
    print(f"\n{'='*130}")
    print(f"{sname}")
    print(f"{'='*130}")
    df=build_entry_trades(sname, sfunc)
    if len(df)<10:
        print(f"  Only {len(df)} trades — skipping")
        continue
    base=run_bt(df, use_wl=False, use_dynch=False)
    fix=run_bt(df, use_wl=True, use_dynch=True)
    all_results[sname]={"df":df,"base":base,"fix":fix}

    # ── Yearly ──
    print(f"\n  YEARLY RETURNS (points):")
    header=f"  {'Year':<6s}"
    for v in ["Base","Fixed","Delta"]:header+=f" {v:>10s}"
    header+=f"  {'BaseWR':>7s} {'FixWR':>7s} {'BaseN':>5s} {'FixN':>5s}"
    print(header)
    print("  "+"-"*len(header))
    for yr in sorted(df["year"].unique()):
        b=base[base["year"]==yr];f=fix[fix["year"]==yr]
        bn=b["pts"].sum();fn=f["pts"].sum()
        bw=(b["pts"]>0).mean() if len(b)>0 else 0
        fw=(f["pts"]>0).mean() if len(f)>0 else 0
        bc=len(b);fc=len(f)
        print(f"  {yr:<6d} {bn:>+9,.0f}  {fn:>+9,.0f}  {fn-bn:>+9,.0f}  {bw:>6.1%}  {fw:>6.1%}  {bc:4d}  {fc:4d}")
    bt=base["pts"].sum();ft=fix["pts"].sum()
    print(f"  {'Total':<6s} {bt:>+9,.0f}  {ft:>+9,.0f}  {ft-bt:>+9,.0f}")

    # ── Monthly ──
    print(f"\n  MONTHLY RETURNS — BASELINE CH45 (points):")
    mon_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"  {'Year':<6s}",end="")
    for mn in mon_names:print(f" {mn:>8s}",end="")
    print(f"  {'Total':>9s}")
    print("  "+"-"*125)
    for yr in sorted(df["year"].unique()):
        print(f"  {yr:<6d}",end="")
        for m in range(1,13):
            v=base[(base["year"]==yr)&(base["month"]==m)]["pts"].sum()
            print(f" {v:>+8,.0f}",end="")
        print(f"  {base[base['year']==yr]['pts'].sum():>+9,.0f}")
    # Monthly average
    print(f"  {'Avg':<6s}",end="")
    for m in range(1,13):
        avg=base[base["month"]==m]["pts"].mean()
        print(f" {avg:>+8,.0f}",end="")
    print(f"  {base['pts'].sum():>+9,.0f}")

    print(f"\n  MONTHLY RETURNS — FIXED W/L+DynCH (points):")
    print(f"  {'Year':<6s}",end="")
    for mn in mon_names:print(f" {mn:>8s}",end="")
    print(f"  {'Total':>9s}")
    print("  "+"-"*125)
    for yr in sorted(df["year"].unique()):
        print(f"  {yr:<6d}",end="")
        for m in range(1,13):
            v=fix[(fix["year"]==yr)&(fix["month"]==m)]["pts"].sum()
            print(f" {v:>+8,.0f}",end="")
        print(f"  {fix[fix['year']==yr]['pts'].sum():>+9,.0f}")
    print(f"  {'Avg':<6s}",end="")
    for m in range(1,13):
        avg=fix[fix["month"]==m]["pts"].mean()
        print(f" {avg:>+8,.0f}",end="")
    print(f"  {fix['pts'].sum():>+9,.0f}")

    # Monthly W/L ratio table
    print(f"\n  MONTHLY W/L RATIO — FIXED:")
    print(f"  {'Month':<8s}",end="")
    for mn in mon_names:print(f" {mn:>6s}",end="")
    print("  Avg")
    print("  "+"-"*90)
    for yr in sorted(df["year"].unique()):
        print(f"  {yr:<8d}",end="")
        for m in range(1,13):
            sub=fix[(fix["year"]==yr)&(fix["month"]==m)&(fix["pts"]>0)]["pts"]
            sl=fix[(fix["year"]==yr)&(fix["month"]==m)&(fix["pts"]<0)]["pts"].abs()
            wl=sub.mean()/sl.mean() if len(sl)>0 and sl.mean()>0 else (999 if len(sub)>0 else 0)
            print(f" {wl:>5.1f}x",end="")
        print("")

# ═══ FINAL COMPARISON TABLE ═══
print(f"\n\n{'='*130}")
print("FINAL SUMMARY — ALL STRATEGIES COMPARED")
print(f"{'='*130}")
header=f"  {'Strategy':<15s} {'Trades':>6s} {'BaseNet':>10s} {'FixNet':>10s} {'Delta':>10s} {'Impr%':>8s} {'WRb':>5s} {'WRf':>5s} {'WLb':>6s} {'WLf':>6s} {'MDDb':>8s} {'MDDf':>8s} {'AnnRet':>8s}"
print(header)
print("  "+"-"*len(header))
for sname in sorted(all_results.keys()):
    r=all_results[sname]
    b=r["base"];f=r["fix"];d=r["df"]
    bn=b["pts"].sum();fn=f["pts"].sum()
    bw=(b["pts"]>0).mean();fw=(f["pts"]>0).mean()
    wb=b[b["pts"]>0]["pts"].mean() if (b["pts"]>0).any() else 0
    lb=b[b["pts"]<0]["pts"].abs().mean() if (b["pts"]<0).any() else 0
    wl_b=wb/lb if lb>0 else 999
    wf=f[f["pts"]>0]["pts"].mean() if (f["pts"]>0).any() else 0
    lf=f[f["pts"]<0]["pts"].abs().mean() if (f["pts"]<0).any() else 0
    wl_f=wf/lf if lf>0 else 999
    pk=0;rn=0;mdd_b=0;mdd_f=0
    for v in b["pts"]:rn+=v;pk=max(pk,rn);mdd_b=max(mdd_b,pk-rn)
    pk=0;rn=0
    for v in f["pts"]:rn+=v;pk=max(pk,rn);mdd_f=max(mdd_f,pk-rn)
    ny=d["year"].nunique()
    ann=fn/ny
    impr=(fn/bn-1)*100 if bn!=0 else 0
    print(f"  {sname:<15s} {len(d):>6d} {bn:>+9,.0f}  {fn:>+9,.0f}  {fn-bn:>+9,.0f}  {impr:>+6.1f}%  {bw:.0%}  {fw:.0%}  {wl_b:>4.1f}x  {wl_f:>4.1f}x  {mdd_b:>+7,.0f}  {mdd_f:>+7,.0f}  {ann:>+7,.0f}")

print(f"\n{'='*130}")
best_abs=sorted(all_results.items(),key=lambda x:x[1]["fix"]["pts"].sum(),reverse=True)[0]
best_impr=sorted(all_results.items(),key=lambda x:(x[1]["fix"]["pts"].sum()/x[1]["base"]["pts"].sum()-1)*100 if x[1]["base"]["pts"].sum()!=0 else 0,reverse=True)[0]
best_wr=sorted(all_results.items(),key=lambda x:(x[1]["fix"]["pts"]>0).mean(),reverse=True)[0]
best_wl=sorted(all_results.items(),key=lambda x:(x[1]["fix"][x[1]["fix"]["pts"]>0]["pts"].mean()/max(x[1]["fix"][x[1]["fix"]["pts"]<0]["pts"].abs().mean(),1)),reverse=True)[0]
print(f"  BEST ABSOLUTE:      {best_abs[0]} ({best_abs[1]['fix']['pts'].sum():>+,.0f} pts)")
print(f"  BEST IMPROVEMENT:   {best_impr[0]} ({(best_impr[1]['fix']['pts'].sum()/best_impr[1]['base']['pts'].sum()-1)*100 if best_impr[1]['base']['pts'].sum()!=0 else 0:+.1f}%)")
print(f"  BEST WIN RATE:      {best_wr[0]} ({(best_wr[1]['fix']['pts']>0).mean():.0%})")
print(f"{'='*130}")
