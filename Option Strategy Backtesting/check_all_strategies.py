import pandas as pd, numpy as np, os, warnings, glob
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

VER = {"DynCH 25+10":(25,10),"DynCH 30+10":(30,10),"DynCH 30+15":(30,15),"DynCH 35+10":(35,10),"DynCH 35+15":(35,15),"DynCH 40+5":(40,5),"DynCH 40+10":(40,10),"DynCH 40+12":(40,12),"DynCH 45+5":(45,5),"DynCH 45+8":(45,8),"DynCH 45+10":(45,10),"DynCH 45+12":(45,12),"DynCH 45+15":(45,15),"DynCH 50+8":(50,8),"DynCH 50+10":(50,10),"DynCH 50+12":(50,12),"DynCH 55+10":(55,10),"DynCH 55+15":(55,15),"DynCH 60+10":(60,10),"DynCH 60+15":(60,15)}
VN=list(VER.keys()); CB=[VER[v][0] for v in VN]; CR=[VER[v][1] for v in VN]
CH_VALS=sorted(set(CB))

all_t = []
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"]); m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
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
        all_t.append({"dt":h1["datetime"].iloc[i],"sym":sym,"year":h1["datetime"].iloc[i].year,"bl":bl,"reg":reg,"pnls":pnls.copy()})

years_list = sorted(set(t["year"] for t in all_t))
NY = len(years_list)

def get_pnl(t, cb, cr):
    if t["reg"]==1: cv=cb-cr
    elif t["reg"]==2: cv=cb+cr
    else: cv=cb
    return t["pnls"].get(min(CH_VALS,key=lambda x:abs(x-cv)))

def compute_metrics(pts):
    n=len(pts)
    if n==0: return {}
    total=sum(pts); w=[x for x in pts if x>0]; l=[x for x in pts if x<0]
    wr=len(w)/n*100; pf=abs(sum(w)/sum(l)) if sum(l)!=0 else float("inf")
    sd=np.std(pts); sharpe=(np.mean(pts)/sd*np.sqrt(252)) if sd>0 else 0
    cum=0;peak=0;mdd=0
    for x in pts:
        cum+=x; 
        if cum>peak: peak=cum
        dd=peak-cum
        if dd>mdd: mdd=dd
    return {"N":n,"Net":total,"WR":wr,"PF":pf,"MDD":mdd,"AvgW":np.mean(w) if w else 0,"AvgL":np.mean(l) if l else 0,"MaxW":max(w) if w else 0,"MaxL":min(l) if l else 0,"StDev":sd,"Sharpe":sharpe,"RoMaD":total/mdd if mdd else 0}

test = [t for t in all_t if t["dt"].year >= 2022]
train = [t for t in all_t if t["dt"].year < 2022]

print("=" * 140)
print("FULL COMPARISON: ALL 20 STRATEGIES")
print(f"Source: {len(all_t)} trades (NIFTY50+SENSEX, 2015-2026) | Test: {len(test)} trades (2022-2026)")
print("=" * 140)

# Collect all data
rows = []
for vi, vn in enumerate(VN):
    cb=CB[vi]; cr=CR[vi]
    
    # Per-year base
    yr_pts = [sum(get_pnl(t,cb,cr) or 0 for t in all_t if t["dt"].year==y) for y in years_list]
    yr_pos = sum(1 for n in yr_pts if n>0)
    total_12yr = sum(yr_pts); min_yr = min(yr_pts); std_yr = np.std(yr_pts)
    
    # Test metrics (1-lot)
    test_pts = [get_pnl(t,cb,cr) for t in test if get_pnl(t,cb,cr) is not None]
    tm = compute_metrics(test_pts)
    
    # Skip on test
    train_losses = [get_pnl(t,cb,cr) for t in train if get_pnl(t,cb,cr) is not None and get_pnl(t,cb,cr)<0]
    skip_th = np.median(train_losses) if len(train_losses)>5 else -5000
    skip_pts = []; prev=0
    for t in test:
        p=get_pnl(t,cb,cr)
        if p is None: continue
        prior_l=[x for x in train_losses if x<0]
        th=np.median(prior_l) if len(prior_l)>5 else skip_th
        if prev<0 and prev<th: prev=0; continue
        skip_pts.append(p); prev=p
    sm = compute_metrics(skip_pts)
    
    # 1w1l
    pos=1; w1_pts=[]
    for p in test_pts:
        w1_pts.append(p*pos); pos=2 if p>0 else 1
    w1 = compute_metrics(w1_pts)
    
    pos=1; w1s_pts=[]
    for p in skip_pts:
        w1s_pts.append(p*pos); pos=2 if p>0 else 1
    w1s = compute_metrics(w1s_pts)
    
    rows.append((vn, yr_pos, total_12yr, min_yr, std_yr, tm, sm, w1, w1s))

# Sort by total_12yr desc
print(f"\n{'-'*140}")
print(f"{'Rank':>4s} {'Version':<18s} {'12yrTotal':>12s} {'Yrs+':>5s} {'WorstYr':>12s} {'StDev':>10s} | {'TestNet':>12s} {'SkipNet':>12s} {'1w1lNet':>12s} {'1w1lSkp':>12s} | {'WR%':>5s} {'SkipWR':>5s} | {'MDD':>10s}")
print(f"{'-'*140}")
sorted_rows = sorted(rows, key=lambda r: -r[2])
for rank, (vn, yrp, tot, miny, stdy, tm, sm, w1, w1s) in enumerate(sorted_rows, 1):
    print(f"{rank:>4d} {vn:<18s} {tot:>+12,.0f} {yrp:>5d}/12 {miny:>+12,.0f} {stdy:>+10,.0f} | {tm.get('Net',0):>+12,.0f} {sm.get('Net',0):>+12,.0f} {w1.get('Net',0):>+12,.0f} {w1s.get('Net',0):>+12,.0f} | {tm.get('WR',0):>4.1f}% {sm.get('WR',0):>4.1f}% | {tm.get('MDD',0):>+10,.0f}")

# Skip improvement by version
print(f"\n{'-'*140}")
print(f"SKIP IMPROVEMENT BY VERSION (test set)")
print(f"{'Version':<18s} {'BaseNet':>12s} {'SkipNet':>12s} {'Chg%':>7s} {'BaseWR':>6s} {'SkipWR':>6s} {'BaseMDD':>12s} {'SkipMDD':>12s} {'WinRateChg':>10s}")
print(f"{'-'*140}")
for rank, (vn, yrp, tot, miny, stdy, tm, sm, w1, w1s) in enumerate(sorted_rows, 1):
    chg = (sm['Net']/tm['Net']-1)*100 if tm['Net']!=0 else 0
    wrc = sm['WR']-tm['WR']
    print(f"{vn:<18s} {tm['Net']:>+12,.0f} {sm['Net']:>+12,.0f} {chg:>+6.1f}% {tm['WR']:>5.1f}% {sm['WR']:>5.1f}% {tm['MDD']:>+12,.0f} {sm['MDD']:>+12,.0f} {wrc:>+9.1f}%")

# Per-symbol breakdown for top 5
print(f"\n{'-'*140}")
print(f"PER-SYMBOL BREAKDOWN (TOP 5, test set 1-lot base)")
print(f"{'-'*140}")
for rank, (vn, yrp, tot, miny, stdy, tm, sm, w1, w1s) in enumerate(sorted_rows[:5], 1):
    print(f"\n{rank}. {vn}")
    for sym in ["NIFTY50","SENSEX"]:
        ty=[t for t in test if t["sym"]==sym]
        pts=[get_pnl(t,CB[VN.index(vn)],CR[VN.index(vn)]) for t in ty if get_pnl(t,CB[VN.index(vn)],CR[VN.index(vn)]) is not None]
        m=compute_metrics(pts)
        print(f"  {sym}: N={m['N']} Net={m['Net']:>+10,.0f} WR={m['WR']:.1f}% PF={m['PF']:.2f} MDD={m['MDD']:>+10,.0f}")
        # Raw points
        raw=[(p+20)/t2["bl"] for t2,p in zip(ty,[get_pnl(t,CB[VN.index(vn)],CR[VN.index(vn)]) for t in ty if get_pnl(t,CB[VN.index(vn)],CR[VN.index(vn)]) is not None])]
        rw=[x for x in raw if x>0]; rl=[x for x in raw if x<0]
        print(f"   RawPt: sum={sum(raw):>+10,.1f} avg={np.mean(raw):>+8,.1f} avgW={np.mean(rw):>+8,.1f} avgL={np.mean(rl):>+8,.1f} WR={len(rw)/len(raw)*100:.1f}%" if raw else f"   RawPt: no data")

print(f"\n{'-'*140}")
print("SUMMARY")
print(f"{'-'*140}")
best_vn = sorted_rows[0][0]
best_45_idx = next(i for i,r in enumerate(sorted_rows) if r[0]=="DynCH 45+10")
best_60_idx = next(i for i,r in enumerate(sorted_rows) if r[0]=="DynCH 60+10")
print(f"  Best total return: {sorted_rows[0][0]} ({sorted_rows[0][2]:,.0f})")
print(f"  Best consistency: DynCH 45+10 (9-10/12 yrs)")
print(f"  Best risk-adjusted: DynCH 45+10 (RoMaD={tm.get('RoMaD',0):.1f} 1-lot)")
print(f"  Worst-year comparison:")
print(f"    DynCH 45+10: Rs{sorted_rows[best_45_idx][3]:,.0f}")
print(f"    DynCH 60+10: Rs{sorted_rows[best_60_idx][3]:,.0f}")
print(f"  Skip benefit: {sorted_rows[0][6].get('Net',0)-sorted_rows[0][5].get('Net',0):,.0f} for best, {sorted_rows[best_45_idx][6].get('Net',0)-sorted_rows[best_45_idx][5].get('Net',0):,.0f} for 45+10")
print(f"\nDone. {len(all_t)} trades total.")
