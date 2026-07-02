"""
Deep Research — Optimize & Reduce Losses
Tests: Time-stop, Dynamic CH, Trend Gate, Signal Scoring, Anti-Martingale rules
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20
CUTOFF_TIME=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

def RSI(df,p=14):
    d=df["close"].diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

# ── Engulfing with full config ──
def run_eng(max_lots=3, ch_base=20, dyn_ch=False, init_stop_h=0, init_ch=5,
            trend_gate=False, min_score=0, mom_stop=10, mom_ts=24):
    """
    dyn_ch: if True, use ch_base when ATR>ATR_MA20, else ch_base+8 (wider in low vol)
    init_stop_h: hours to use init_ch multiplier before switching to ch_base
    trend_gate: if True, only trade when 1H close > 1H EMA50
    min_score: minimum signal quality score (0=all)
    """
    et=[]  # engulfing trades
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
        h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
        
        # Precompute indicators for scoring/trend gate
        h1["ema50"]=h1["close"].ewm(span=50).mean()
        h1["atr14"]=A(h1,14)
        h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
        h1["rsi14"]=RSI(h1,14)
        
        body=(h1["close"]-h1["open"]).abs()
        is_red=h1["close"]<h1["open"]; is_green=h1["close"]>h1["open"]
        sigs=[]
        for i in range(1,len(h1)):
            if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.50: continue
            
            # Trend gate
            if trend_gate and h1["close"].iloc[i]<=h1["ema50"].iloc[i]: continue
            
            # Signal quality score (0-100)
            if min_score>0:
                gap_pct=(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100
                br=body.iloc[i]/body.iloc[i-1] if body.iloc[i-1]>0 else 1
                rsi=h1["rsi14"].iloc[i] if not pd.isna(h1["rsi14"].iloc[i]) else 50
                adx=abs(h1["close"].iloc[i]-h1["ema50"].iloc[i])/h1["ema50"].iloc[i]*100 if not pd.isna(h1["ema50"].iloc[i]) else 0
                # Score components (each 0-25)
                gap_score=min(25,max(0,-gap_pct*10))  # bigger gap down = higher score
                body_score=min(25,br*5)  # bigger body ratio = higher score
                rsi_score=min(25,max(0,(50-rsi)))  # lower RSI = higher score
                adx_score=min(25,adx*50)  # further from EMA = higher score
                score=gap_score+body_score+rsi_score+adx_score
                if score<min_score: continue
            
            sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],
                         "idx":i,"h1":h1})
        
        base_lot=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=A(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        
        cur_lots=1; ws=0; ls=0
        for sig in sigs:
            idx_h1=sig["idx"]; h1_ref=sig["h1"]
            tu=int(pd.to_datetime(sig["trigger_time"]).timestamp()); lv=sig["level"]
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5): continue
            broke=idx
            while broke<len(m5) and cl[broke]<=lv: broke+=1
            if broke>=len(m5): continue
            retest=broke+1
            while retest<len(m5):
                if lo[retest]<lv and cl[retest]>lv and tc.iloc[retest]<CUTOFF_TIME: break
                retest+=1
            if retest>=len(m5): continue
            entry=cl[retest]; sl=lo[retest]
            if entry-sl<=0 or m5["datetime"].iloc[retest].hour==9: continue
            
            # Determine chandelier multiplier
            ch_mult=ch_base
            if dyn_ch and not pd.isna(h1_ref["atr14"].iloc[idx_h1]) and not pd.isna(h1_ref["atr_ma20"].iloc[idx_h1]):
                if h1_ref["atr14"].iloc[idx_h1] > h1_ref["atr_ma20"].iloc[idx_h1]:
                    ch_mult=ch_base-5  # tighter in high vol
                else:
                    ch_mult=ch_base+5  # wider in low vol
            
            entry_time=m5["datetime"].iloc[retest]
            highest=entry
            for j in range(retest+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>highest: highest=hi[j]
                
                # Time-based initial stop
                if init_stop_h>0:
                    hours_held=(m5["datetime"].iloc[j]-entry_time).total_seconds()/3600
                    if hours_held<=init_stop_h:
                        trail=highest-init_ch*ca
                    else:
                        trail=highest-ch_mult*ca
                else:
                    trail=highest-ch_mult*ca
                
                if cl[j]<trail:
                    pts=cl[j]-entry
                    pnl=pts*base_lot*cur_lots - CHG*cur_lots
                    et.append({"pnl_rs":pnl,"lot":cur_lots,"exit_time":m5["datetime"].iloc[j],"sym":sym,"strat":"Eng"})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=2 and cur_lots<max_lots: cur_lots+=1; ws=0
                    if ls>=1 and cur_lots>1: cur_lots-=1; ls=0
                    break
    return pd.DataFrame(et).sort_values("exit_time").reset_index(drop=True) if et else pd.DataFrame()

def run_mom(ch=10, ts=24):
    mt=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
        atr=A(h1,14); hi20=h1["high"].rolling(20).max().shift(1)
        lot=NLOT if "NIFTY" in sym else SLOT
        it=False; ep=0; he=0; etm=None
        for i in range(20,len(h1)):
            if not it:
                if h1["close"].iloc[i]>hi20.iloc[i] and h1["close"].iloc[i]>h1["open"].iloc[i] and h1["datetime"].iloc[i].time()<CUTOFF_TIME and h1["datetime"].iloc[i].hour>=9:
                    it=True; ep=h1["close"].iloc[i]; he=ep; etm=h1["datetime"].iloc[i]
            else:
                if h1["high"].iloc[i]>he: he=h1["high"].iloc[i]
                ca=atr.iloc[i]
                ex=False
                if not pd.isna(ca) and h1["close"].iloc[i]<he-ch*ca: ex=True
                if ts>0 and not ex:
                    hr=(h1["datetime"].iloc[i]-etm).total_seconds()/3600
                    if hr>ts and h1["close"].iloc[i]<=ep: ex=True
                if ex:
                    pnl=(h1["close"].iloc[i]-ep)*lot-CHG
                    mt.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl,"strat":"Mom","lot":1,"sym":sym})
                    it=False
    return pd.DataFrame(mt).sort_values("exit_time").reset_index(drop=True) if mt else pd.DataFrame()

def lf(df, sn=2):
    df=df.sort_values("exit_time").reset_index(drop=True)
    lc=0; k=np.ones(len(df),dtype=bool)
    for i in range(len(df)):
        if lc>=sn: k[i]=False; lc=0; continue
        if df["pnl_rs"].iloc[i]<=0: lc+=1
        else: lc=0
    return df[k].reset_index(drop=True)

def test(label, **kw):
    eng=run_eng(**kw)
    mom=run_mom(kw.get("mom_stop",10), kw.get("mom_ts",24))
    cb=pd.concat([eng,mom],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    cb=lf(cb,2)
    n=len(cb); net=cb["pnl_rs"].sum() if n>0 else 0
    wr=(cb["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    aw=cb[cb["pnl_rs"]>0]["pnl_rs"].mean() if (cb["pnl_rs"]>0).sum()>0 else 0
    al=cb[cb["pnl_rs"]<0]["pnl_rs"].mean() if (cb["pnl_rs"]<0).sum()>0 else 0
    pf=(cb[cb["pnl_rs"]>0]["pnl_rs"].sum()/abs(cb[cb["pnl_rs"]<0]["pnl_rs"].sum())) if (cb["pnl_rs"]<0).sum()!=0 else 99
    eq=cb["pnl_rs"].cumsum()+200000; pk=eq.cummax(); mdd=(pk-eq).max()
    mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
    alot=cb[cb["strat"]=="Eng"]["lot"].mean() if "Eng" in cb["strat"].values else 0
    yrs=(pd.to_datetime(cb["exit_time"]).max()-pd.to_datetime(cb["exit_time"]).min()).total_seconds()/31536000 if len(cb)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    return {"label":label,"n":n,"net":net,"wr":wr,"aw":aw,"al":al,"pf":pf,"mdd":mdd,"mdd_p":mdd_p,"alot":alot,"cagr":cagr,"ndd":net/mdd if mdd>0 else 0}

# ══════════════════════════════════════════════════════════════════════
print("="*130)
print("DEEP RESEARCH — OPTIMIZE & REDUCE LOSSES")
print("="*130)

all_results=[]

# ── 1. BASELINE ──
all_results.append(test("1_Baseline_CH20", ch_base=20))
print(f"\n1. BASELINE: {all_results[-1]['net']:>+10,.0f}")

# ── 2. TIME-BASED INITIAL STOP ──
print(f"\n2. TIME-BASED INITIAL STOP (tight stop first N hours, then CH20)")
for init_h,init_ch in [(12,3),(12,5),(24,3),(24,5),(24,8),(48,5),(48,8)]:
    r=test(f"2_Init{init_h}h_{init_ch}x", init_stop_h=init_h, init_ch=init_ch)
    all_results.append(r)
    vs=(r["net"]/all_results[0]["net"]-1)*100
    print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  vsBase={vs:+.1f}%")

# ── 3. DYNAMIC CHANDELIER ──
print(f"\n3. DYNAMIC CHANDELIER (adjust mult based on volatility regime)")
b_net=all_results[0]["net"]
for base,adj in [(15,5),(15,8),(20,5),(20,8),(20,10),(25,8)]:
    r=test(f"3_DynCH{base}±{adj}", ch_base=base, dyn_ch=True)
    all_results.append(r)
    vs=(r["net"]/b_net-1)*100
    print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  vsBase={vs:+.1f}%")

# ── 4. TREND GATE ──
print(f"\n4. TREND GATE (only trade when close > 1H EMA50)")
for tg in [True]:
    r=test(f"4_TrendGate", trend_gate=tg)
    all_results.append(r)
    vs=(r["net"]/b_net-1)*100
    print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  vsBase={vs:+.1f}%")

# ── 5. SIGNAL QUALITY SCORE ──
print(f"\n5. SIGNAL QUALITY SCORE (only trade signals scoring above threshold)")
for ms in [10,20,30,40,50,60]:
    r=test(f"5_Score>{ms}", min_score=ms)
    all_results.append(r)
    vs=(r["net"]/b_net-1)*100
    print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  n={r['n']:3d}")

# ── 6. ANTI-MARTINGALE RULE VARIANTS ──
print(f"\n6. ANTI-MARTINGALE RULE VARIANTS (different win/loss thresholds)")
# Fix: test with different win_add/loss_sub rules via custom run_eng wrapper
def run_eng_custom(wa, la, ml=3, ch=20):
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
        h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
        body=(h1["close"]-h1["open"]).abs()
        is_red=h1["close"]<h1["open"]; is_green=h1["close"]>h1["open"]
        sigs=[]
        for i in range(1,len(h1)):
            if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.50: continue
            sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
        bl=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=A(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        cl2=1; ws=0; ls=0
        for sig in sigs:
            tu=int(pd.to_datetime(sig["trigger_time"]).timestamp()); lv=sig["level"]
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5): continue
            brk=idx
            while brk<len(m5) and cl[brk]<=lv: brk+=1
            if brk>=len(m5): continue
            rt=brk+1
            while rt<len(m5):
                if lo[rt]<lv and cl[rt]>lv and tc.iloc[rt]<CUTOFF_TIME: break
                rt+=1
            if rt>=len(m5): continue
            ep=cl[rt]; sl=lo[rt]
            if ep-sl<=0 or m5["datetime"].iloc[rt].hour==9: continue
            he=ep
            for j in range(rt+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-ch*ca:
                    pnl=(cl[j]-ep)*bl*cl2 - CHG*cl2
                    et.append({"pnl_rs":pnl,"lot":cl2,"exit_time":m5["datetime"].iloc[j],"sym":sym,"strat":"Eng"})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=wa and cl2<ml: cl2+=1; ws=0
                    if ls>=la and cl2>1: cl2-=1; ls=0
                    break
    return pd.DataFrame(et).sort_values("exit_time").reset_index(drop=True) if et else pd.DataFrame()

mom_ts=24; mom_ch=10
for wa,la,lb in [(1,1,"1w1l"),(2,1,"2w1l"),(2,2,"2w2l"),(3,1,"3w1l"),(3,2,"3w2l")]:
    eng=run_eng_custom(wa,la,3,20)
    mom=run_mom(mom_ch,mom_ts)
    cb=pd.concat([eng,mom],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    cb=lf(cb,2)
    n=len(cb); net=cb["pnl_rs"].sum() if n>0 else 0
    wr=(cb["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    pf=(cb[cb["pnl_rs"]>0]["pnl_rs"].sum()/abs(cb[cb["pnl_rs"]<0]["pnl_rs"].sum())) if (cb["pnl_rs"]<0).sum()!=0 else 99
    eq=cb["pnl_rs"].cumsum()+200000; pk=eq.cummax(); mdd=(pk-eq).max()
    mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
    alot=cb[cb["strat"]=="Eng"]["lot"].mean() if "Eng" in cb["strat"].values else 0
    yrs=(pd.to_datetime(cb["exit_time"]).max()-pd.to_datetime(cb["exit_time"]).min()).total_seconds()/31536000 if len(cb)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    ndd=net/mdd if mdd>0 else 0
    r={"label":f"6_AM{lb}","n":n,"net":net,"wr":wr,"aw":0,"al":0,"pf":pf,"mdd":mdd,"mdd_p":mdd_p,"alot":alot,"cagr":cagr,"ndd":ndd}
    all_results.append(r)
    vs=(r["net"]/b_net-1)*100
    print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  AvgLot={r['alot']:.2f}  vsBase={vs:+.1f}%")

# ── 7. COMBINATIONS (best from each category) ──
print(f"\n7. BEST COMBINATIONS")
# Combination: init stop 24h@5x + trend gate + CH20
r=test("7_Init24h5x+Trend", init_stop_h=24, init_ch=5, trend_gate=True)
all_results.append(r)
vs=(r["net"]/b_net-1)*100
print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  vsBase={vs:+.1f}%")
# Init stop + dyn ch
r=test("7_Init24h5x+DynCH", init_stop_h=24, init_ch=5, ch_base=20, dyn_ch=True)
all_results.append(r)
vs=(r["net"]/b_net-1)*100
print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  vsBase={vs:+.1f}%")
# Trend gate + dyn ch
r=test("7_Trend+DynCH20±5", trend_gate=True, ch_base=20, dyn_ch=True)
all_results.append(r)
vs=(r["net"]/b_net-1)*100
print(f"  {r['label']:25s}  Rs{r['net']:>+9,.0f}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  vsBase={vs:+.1f}%")

# ── FINAL RANKING ──
print(f"\n{'='*130}")
print("FINAL RANKING BY NET P&L")
print(f"{'='*130}")
print(f"{'RANK':>4s}  {'Variant':30s}  {'Trades':>5s}  {'Net_RS':>11s}  {'WR%':>5s}  {'PF':>5s}  "
      f"{'MDD%':>6s}  {'CAGR':>6s}  {'AvgLot':>6s}  {'Net/MDD':>7s}  {'vsBASE':>8s}")
print("-"*110)
sorted_r=sorted(all_results, key=lambda x: x["net"], reverse=True)
for i,r in enumerate(sorted_r):
    vs=(r["net"]/b_net-1)*100
    print(f"{i+1:4d}  {r['label']:30s}  {r['n']:4d}  Rs{r['net']:>+9,.0f}  {r['wr']:4.1f}%  "
          f"{r['pf']:5.2f}  {r['mdd_p']:5.2f}%  {r['cagr']:5.1f}%  {r['alot']:5.2f}  "
          f"{r['ndd']:6.1f}x  {vs:>+7.1f}%")

# Save all results
out_dir=os.path.join(BASE,"backtest_results","deep_research")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(sorted_r).to_csv(os.path.join(out_dir,"deep_research_results.csv"),index=False)
print(f"\nSaved to: {out_dir}")
print("DONE")
