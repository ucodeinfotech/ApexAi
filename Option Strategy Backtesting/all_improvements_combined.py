"""
ALL STRATEGIES × ALL IMPROVEMENTS (points)
Tests every combination of: entry × CH × sizing × skip
"""
import pandas as pd, numpy as np, os, warnings, re
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

def compute_daily_ema(h1, period=50):
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

# === LOAD DATA ===
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
               "m5_cl":m5["close"].values,"m5_epoch":m5["datetime"].astype('int64').values,
               "tc":pd.Series(m5["datetime"]).dt.time.values}
CUT=pd.Timestamp("14:15").time()

def find_retest(sym, t, lv):
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
    return (r, ep, sl)

def compute_ch_exits(sym, r, ep):
    d=DATA[sym];atr5v=d["atr5v"];m5_hi=d["m5_hi"];m5_cl=d["m5_cl"]
    pnls={}
    for cv in CH_VALS:
        he=ep
        for j in range(r,len(m5_cl)):
            ca=atr5v[j]
            if pd.isna(ca):continue
            if m5_hi[j]>he:he=m5_hi[j]
            if m5_cl[j]<he-cv*ca:
                pnls[cv]=round(m5_cl[j]-ep,1);break
    return pnls

# === ENTRY SIGNALS ===
def sigs_engulf_raw(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

def sigs_engulf_filt(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    atr20=compute_atr20(h1);adx14=compute_adx14(h1);ema50=compute_daily_ema(h1,50);ema200=compute_daily_ema(h1,200)
    out=[]
    for i in range(1,len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if b.iloc[i]<b.iloc[i-1]*0.5:continue
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i] or adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750 or h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

def sigs_big_candle(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();ab=b.rolling(20,min_periods=20).mean()
    g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if pd.isna(ab.iloc[i]) or not r.iloc[i-1] or not g.iloc[i]:continue
        if b.iloc[i-1]<=ab.iloc[i-1]*1.5 or b.iloc[i]<b.iloc[i-1]*0.5:continue
        mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
        if h1["close"].iloc[i]<mid or (h1["open"].iloc[i]-h1["low"].iloc[i])>b.iloc[i]*0.5:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

def sigs_sir(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"]
    atr20=compute_atr20(h1);adx14=compute_adx14(h1);ema50=compute_daily_ema(h1,50);ema200=compute_daily_ema(h1,200)
    out=[]
    for i in range(1,len(h1)):
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if not g.iloc[i] or ema50[i]<=ema200[i] or adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        if not(h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue
        if b.iloc[i-1]<=1.0*atr20.iloc[i] or b.iloc[i]<b.iloc[i-1]*0.5:continue
        mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
        if h1["close"].iloc[i]<mid or (h1["open"].iloc[i]-h1["low"].iloc[i])>b.iloc[i]*0.5:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

def sigs_comb_or(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"]
    atr20=compute_atr20(h1);adx14=compute_adx14(h1);ema50=compute_daily_ema(h1,50);ema200=compute_daily_ema(h1,200)
    out=[]
    for i in range(1,len(h1)):
        if not(g.iloc[i] and h1["close"].iloc[i-1]<h1["open"].iloc[i-1]):continue
        bc=False
        if not pd.isna(atr20.iloc[i]) and b.iloc[i-1]>1.0*atr20.iloc[i] and b.iloc[i]>=b.iloc[i-1]*0.5:
            mid=(h1["open"].iloc[i-1]+h1["close"].iloc[i-1])/2
            if h1["close"].iloc[i]>=mid and (h1["open"].iloc[i]-h1["low"].iloc[i])<=b.iloc[i]*0.5:bc=True
        en = (h1["open"].iloc[i]<=h1["close"].iloc[i-1] and h1["close"].iloc[i]>=h1["open"].iloc[i-1] and b.iloc[i]>=b.iloc[i-1]*0.5)
        if not (bc or en):continue
        if pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i] or adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

# === BUILD ALL TRADE SETS ===
strategies = [
    ("Engulf_Raw", sigs_engulf_raw),
    ("Engulf_Filt", sigs_engulf_filt),
    ("BigCandle", sigs_big_candle),
    ("Sir", sigs_sir),
    ("Comb_OR", sigs_comb_or),
]
trade_sets={}
for sname, sfunc in strategies:
    print(f"Building {sname}...", end=" ", flush=True)
    rows=[]
    for sym in ["NIFTY50","SENSEX"]:
        for sig in sfunc(sym):
            ret=find_retest(sym, sig["ts"], sig["lv"])
            if ret is None:continue
            r,ep,sl=ret
            pnls=compute_ch_exits(sym, r, ep)
            if 45 not in pnls:continue
            t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"ts":sig["ts"],"pts45":pnls[45]}
            for c,p in pnls.items():t[f"p{c}"]=p
            rows.append(t)
    trade_sets[sname]=pd.DataFrame(rows).fillna(0)
    print(f"{len(trade_sets[sname])} trades")

# === BACKTEST ===
def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def run_bt(df, ch_val=45, use_wl=False, use_dynch=False, skip_n=0, size_fn=None):
    rows=[]
    for yr in sorted(df["yr"].unique()):
        yr_d=df[df["yr"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        month_best={}
        if use_dynch:
            hist=df[df["yr"]<yr]
            for m in range(1,13):
                sub=hist[hist["mo"]==m]
                if len(sub)<5:month_best[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_best[m]=best
        rw=[];rl=[];skip_c=0
        for _,t in yr_d.iterrows():
            if skip_c>0:skip_c-=1;continue
            if use_dynch:
                m_ch=month_best.get(t["mo"],45)
                pts=t.get(f"p{m_ch}",t["pts45"])
            else:
                pts=t.get(f"p{ch_val}",t["pts45"])
            sz=1.0
            if use_wl:sz=wl_size(rw,rl,lb=5)
            if size_fn=="2w1l":
                nw=sum(1 for x in rw[-2:] if x>0);nl=sum(1 for x in rl[-1:] if x<0)
                sz=1.5 if nw>=2 else (0.5 if nl>=1 else 1.0)
            if size_fn=="3w2l":
                nw=sum(1 for x in rw[-3:] if x>0);nl=sum(1 for x in rl[-2:] if x<0)
                sz=1.5 if nw>=3 else (0.5 if nl>=2 else 1.0)
            pnl=pts*sz
            rows.append({"pts":pnl,"yr":t["yr"],"mo":t["mo"],"sz":sz})
            if pnl>0:rw.append(pnl);skip_c=0
            else:rl.append(abs(pnl));skip_c=skip_n
    return pd.DataFrame(rows)

# === CONFIGURATIONS ===
configs=[
    # (name, ch_val, use_wl, use_dynch, skip_n, size_fn)
    ("CH45_base",          45, False, False, 0, None),
    ("CH55",               55, False, False, 0, None),
    ("CH55+WL",            55, True,  False, 0, None),
    ("CH55+Skip2",         55, False, False, 2, None),
    ("CH55+WL+Skip2",      55, True,  False, 2, None),
    ("CH55+2w1l",          55, False, False, 0, "2w1l"),
    ("DynCH",              45, False, True,  0, None),
    ("DynCH+WL",           45, True,  True,  0, None),
    ("DynCH+Skip2",        45, False, True,  2, None),
    ("DynCH+WL+Skip2",     45, True,  True,  2, None),
]

# === RUN ALL ===
print("\n"+"="*140)
print("ALL STRATEGIES × ALL IMPROVEMENTS (points)")
print("="*140)

all_results={}
for sname, sdf in trade_sets.items():
    print(f"\n  -- {sname} ({len(sdf)} trades) --")
    for cname, cv, wl, dyn, skip, szfn in configs:
        res=run_bt(sdf, ch_val=cv, use_wl=wl, use_dynch=dyn, skip_n=skip, size_fn=szfn)
        net=res["pts"].sum();n=len(res);wr=(res["pts"]>0).mean()
        w=res[res["pts"]>0]["pts"];l=res[res["pts"]<0]["pts"]
        wl_r=w.mean()/abs(l.mean()) if len(l)>0 and l.mean()!=0 else 999
        pk=0;rn=0;mdd=0
        for v in res["pts"]:rn+=v;pk=max(pk,rn);mdd=max(mdd,pk-rn)
        ar=net/len(set(sdf["yr"]))
        impr=(net/(sdf["pts45"].sum())-1)*100
        key=f"{sname}|{cname}"
        all_results[key]={"sname":sname,"cname":cname,"net":net,"n":n,"wr":wr,"wl":wl_r,"mdd":mdd,"ar":ar,"impr":impr,"res":res}
        print(f"    {cname:<20s} Net={net:>+9,.0f} WR={wr:.0%} W/L={wl_r:.1f}x MDD={mdd:>+8,.0f} AnnRet={ar:>+8,.0f} Impr={impr:+.1f}%")

# === FINAL SUMMARY ===
print(f"\n\n{'='*140}")
print("FINAL RANKING — ALL COMBINATIONS")
print("="*140)
header=f"  {'Strategy+Config':<35s} {'Trades':>6s} {'Net':>10s} {'WR':>5s} {'W/L':>6s} {'MDD':>9s} {'AnnRet':>8s} {'Impr%':>7s}"
print(header);print("  "+"-"*len(header))
sorted_res=sorted(all_results.items(), key=lambda x:-x[1]["net"])
for key,v in sorted_res:
    print(f"  {key:<35s} {v['n']:>6d} {v['net']:>+9,.0f}  {v['wr']:.0%} {v['wl']:>4.1f}x {v['mdd']:>+8,.0f} {v['ar']:>+7,.0f} {v['impr']:>+6.1f}%")

# === BEST PER STRATEGY ===
print(f"\n{'='*140}")
print("BEST CONFIGURATION PER STRATEGY")
print("="*140)
for sname in [s for s in strategies if s[0] in trade_sets]:
    sn=sname[0]
    matches=[(k,v) for k,v in all_results.items() if v["sname"]==sn]
    if not matches:continue
    best=max(matches, key=lambda x:x[1]["net"])
    print(f"  {sn:<20s} -> {best[1]['cname']:<20s} Net={best[1]['net']:>+9,.0f} WR={best[1]['wr']:.0%} W/L={best[1]['wl']:.1f}x MDD={best[1]['mdd']:>+8,.0f}")

# === OVERALL BEST ===
print(f"\n{'='*140}")
print("OVERALL TOP 10 COMBINATIONS")
print("="*140)
for rank,(key,v) in enumerate(sorted_res[:10],1):
    print(f"  #{rank:<2d} {key:<35s} Net={v['net']:>+9,.0f} WR={v['wr']:.0%} W/L={v['wl']:.1f}x MDD={v['mdd']:>+8,.0f}")

# === YEAR-BY-YEAR FOR TOP 5 ===
top5_keys=[k for k,_ in sorted_res[:5]]
print(f"\n{'='*140}")
print("YEAR-BY-YEAR: TOP 5")
print("="*140)
years=sorted(trade_sets["Engulf_Raw"]["yr"].unique())
header="  Year"
for k in top5_keys:
    parts=k.split("|")
    header+=f"  {parts[0][:10]}+{parts[1][:15]:<18s}"
print(header)
print("  "+"-"*len(header))
for yr in years:
    line=f"  {yr:<4d}"
    for k in top5_keys:
        v=all_results[k]["res"];yr_pts=v[v["yr"]==yr]["pts"].sum()
        line+=f" {yr_pts:>+18,.0f}"
    print(line)
# Totals
line="  Total"
for k in top5_keys:
    line+=f" {all_results[k]['net']:>+18,.0f}"
print(line)

# === BEST COMBINATION: ALL IMPROVEMENTS ON ALL STRATEGIES ===
print(f"\n{'='*140}")
print("BEST OF ALL: CH55 + WL + Skip2 on EVERY strategy")
print("="*140)
header=f"  {'Strategy':<20s} {'BaseCH45':>10s} {'CH55':>10s} {'CH55+WL':>10s} {'CH55+Skip2':>10s} {'CH55+WL+Skip2':>10s} {'DynCH+WL+Skip2':>10s}"
print(header);print("  "+"-"*len(header))
for sname in [s[0] for s in strategies if s[0] in trade_sets]:
    sdf=trade_sets[sname];base=sdf["pts45"].sum()
    r1=run_bt(sdf, ch_val=55, use_wl=False, use_dynch=False, skip_n=0)
    r2=run_bt(sdf, ch_val=55, use_wl=True, use_dynch=False, skip_n=0)
    r3=run_bt(sdf, ch_val=55, use_wl=False, use_dynch=False, skip_n=2)
    r4=run_bt(sdf, ch_val=55, use_wl=True, use_dynch=False, skip_n=2)
    r5=run_bt(sdf, ch_val=45, use_wl=True, use_dynch=True, skip_n=2)
    print(f"  {sname:<20s} {base:>+9,.0f}  {r1['pts'].sum():>+9,.0f}  {r2['pts'].sum():>+9,.0f}  {r3['pts'].sum():>+9,.0f}  {r4['pts'].sum():>+9,.0f}  {r5['pts'].sum():>+9,.0f}")

# === BEST OVERALL YEAR-BY-YEAR ===
print(f"\n{'='*140}")
print("BEST OVERALL: CH55+WL+Skip2 on Engulf_Raw (year-by-year)")
print("="*140)
sdf=trade_sets["Engulf_Raw"]
best=run_bt(sdf, ch_val=55, use_wl=True, use_dynch=False, skip_n=2)
print(f"  Total Net: {best['pts'].sum():>+9,.0f}")
print(f"  WR: {(best['pts']>0).mean():.0%}")
print(f"  W/L: {best[best['pts']>0]['pts'].mean()/abs(best[best['pts']<0]['pts'].mean()):.1f}x")
pk=0;rn=0;mdd=0
for v in best["pts"]:rn+=v;pk=max(pk,rn);mdd=max(mdd,pk-rn)
print(f"  MDD: {mdd:>+9,.0f}")
print(f"\n  Year    Net      WR    W/L   Trades  AvgSz")
for yr in years:
    y=best[best["yr"]==yr]
    if len(y)==0:continue
    w=y[y["pts"]>0]["pts"];l=y[y["pts"]<0]["pts"]
    wr=(y["pts"]>0).mean();wl=w.mean()/abs(l.mean()) if len(l)>0 and l.mean()!=0 else 999
    print(f"  {yr:<4d}  {y['pts'].sum():>+8,.0f}  {wr:.0%}  {wl:>4.1f}x  {len(y):5d}  {y['sz'].mean():.2f}")

print(f"\n{'='*140}")
print("DONE")
print(f"{'='*140}")
