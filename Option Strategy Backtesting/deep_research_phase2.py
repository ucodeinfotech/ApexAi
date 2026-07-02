"""
Deep Research Phase 2 — Combine Best Findings + New Dimensions
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20
CUTOFF_TIME=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

def run_eng_advanced(max_lots=3, ch_base=20, dyn_ch=0, am_rule=(2,1),
                     corr_filter=False, gap_min=0, gap_max=999,
                     vol_regime=False):
    """
    dyn_ch: 0=off, 1=±5 (2-regime), 2=3-regime (tight/medium/loose)
    am_rule: (wins_to_add, losses_to_remove)
    corr_filter: skip when both indices have signals same day
    gap_min/gap_max: filter by gap down % 
    vol_regime: if True, use 3-regime dynamic chandelier
    """
    wa,la=am_rule
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
        h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
        
        pre=None
        try:
            pre=pd.read_csv(os.path.join(BASE,f"{sym}_PRE_COVID.csv"),nrows=1)
        except: pass
        if pre is not None: h1_use=pd.read_csv(os.path.join(BASE,f"{sym}_PRE_COVID.csv"))
        else: h1_use=h1
        
        h1["atr14"]=A(h1,14)
        h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
        h1["atr_pctile"]=h1["atr14"].rolling(252).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()+1e-10)*100 if x.max()>x.min() else 50, raw=False)
        h1["rsi14"]=100-(100/(1+h1["close"].diff().clip(lower=0).rolling(14).mean()/(-h1["close"].diff().clip(upper=0).rolling(14).mean()+1e-10)))
        
        body=(h1["close"]-h1["open"]).abs()
        is_red=h1["close"]<h1["open"]; is_green=h1["close"]>h1["open"]
        sigs=[]
        for i in range(1,len(h1)):
            if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.50: continue
            
            gap_pct=(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100
            if -gap_pct<gap_min or -gap_pct>gap_max: continue
            
            sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],
                         "idx":i,"date":h1["datetime"].iloc[i].date()})
        
        base_lot=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=A(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        
        cur_lots=1; ws=0; ls=0
        for sig in sigs:
            idx_h1=sig["idx"]; h1_ref=h1
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
            
            entry_time=m5["datetime"].iloc[retest]
            atr14_val=h1_ref["atr14"].iloc[idx_h1]
            atr_ma20_val=h1_ref["atr_ma20"].iloc[idx_h1]
            atr_pctile=h1_ref["atr_pctile"].iloc[idx_h1]
            
            if dyn_ch==1:
                # 2-regime: high vol→tighter, low vol→wider
                if not pd.isna(atr14_val) and not pd.isna(atr_ma20_val) and atr14_val>atr_ma20_val:
                    ch_mult=ch_base-5
                else:
                    ch_mult=ch_base+5
            elif dyn_ch==2:
                # 3-regime: use ATR percentile
                if not pd.isna(atr_pctile):
                    if atr_pctile>66: ch_mult=ch_base-8      # high vol: tight
                    elif atr_pctile<33: ch_mult=ch_base+8     # low vol: wide
                    else: ch_mult=ch_base                      # normal vol
                else:
                    ch_mult=ch_base
            else:
                ch_mult=ch_base
            
            highest=entry
            for j in range(retest+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>highest: highest=hi[j]
                trail=highest-ch_mult*ca
                if cl[j]<trail:
                    pts=cl[j]-entry
                    pnl=pts*base_lot*cur_lots - CHG*cur_lots
                    et.append({"pnl_rs":pnl,"lot":cur_lots,"exit_time":m5["datetime"].iloc[j],"sym":sym,"strat":"Eng","date":sig["date"]})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=wa and cur_lots<max_lots: cur_lots+=1; ws=0
                    if ls>=la and cur_lots>1: cur_lots-=1; ls=0
                    break
    
    df=pd.DataFrame(et).sort_values("exit_time").reset_index(drop=True) if et else pd.DataFrame()
    
    # Correlation filter: if same date appears for both indices with signals, skip one (second occurrence)
    if corr_filter and len(df)>0:
        df_date_sym=df.copy()
        keep=[]; seen_dates=set()
        for _,r in df_date_sym.iterrows():
            key=(r["date"],r["strat"])
            if key in seen_dates: continue
            seen_dates.add(key)
            keep.append(_)
        df=df.loc[keep].reset_index(drop=True)
    
    return df

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
                    mt.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl,"strat":"Mom","lot":1,"sym":sym,"date":h1["datetime"].iloc[i].date()})
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

def test(label, ch_base=20, dyn_ch=0, am_rule=(2,1), mom_stop=10, mom_ts=24,
          corr_filter=False, gap_min=0, gap_max=999, vol_regime=False):
    if gap_max>900: gap_max=999
    
    # Map am_rule to label descriptor
    wa,la=am_rule
    if wa==1 and la==1: am_desc="1w1l"
    elif wa==2 and la==1: am_desc="2w1l"
    elif wa==2 and la==2: am_desc="2w2l"
    elif wa==3 and la==1: am_desc="3w1l"
    elif wa==3 and la==2: am_desc="3w2l"
    else: am_desc=f"{wa}w{la}l"
    
    eng=run_eng_advanced(ch_base=ch_base, dyn_ch=dyn_ch, am_rule=am_rule,
                         corr_filter=corr_filter, gap_min=gap_min, gap_max=gap_max,
                         vol_regime=vol_regime)
    mom=run_mom(mom_stop, mom_ts)
    c=pd.concat([eng,mom],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    c=lf(c,2)
    
    n=len(c); net=c["pnl_rs"].sum() if n>0 else 0
    wr=(c["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    aw=c[c["pnl_rs"]>0]["pnl_rs"].mean() if (c["pnl_rs"]>0).sum()>0 else 0
    al=c[c["pnl_rs"]<0]["pnl_rs"].mean() if (c["pnl_rs"]<0).sum()>0 else 0
    pf=(c[c["pnl_rs"]>0]["pnl_rs"].sum()/abs(c[c["pnl_rs"]<0]["pnl_rs"].sum())) if (c["pnl_rs"]<0).sum()!=0 else 99
    eq=c["pnl_rs"].cumsum()+200000; pk=eq.cummax(); mdd=(pk-eq).max()
    mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
    alot=c[c["strat"]=="Eng"]["lot"].mean() if "Eng" in c["strat"].values else 0
    yrs=(pd.to_datetime(c["exit_time"]).max()-pd.to_datetime(c["exit_time"]).min()).total_seconds()/31536000 if len(c)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    ndd=net/mdd if mdd>0 else 0
    return {"label":label,"n":n,"net":net,"wr":wr,"aw":aw,"al":al,"pf":pf,"mdd":mdd,"mdd_p":mdd_p,"alot":alot,"cagr":cagr,"ndd":ndd}

print("="*130)
print("DEEP RESEARCH PHASE 2 — COMBINATIONS & NEW DIMENSIONS")
print("="*130)
all_r=[]

# ── Reference baseline ──
all_r.append(test("BL_CH20_2w1l"))
print(f"\nBaseline: Rs{all_r[0]['net']:+,.0f} | MDD={all_r[0]['mdd_p']:.2f}% | PF={all_r[0]['pf']:.2f} | CAGR={all_r[0]['cagr']:.1f}%")
b_net=all_r[0]["net"]

# ── 1. Dyn CH25 best from Phase 1 ──
all_r.append(test("1_DynCH25_2w1l", ch_base=25, dyn_ch=1))
print(f"1_DynCH25_2w1l:    Rs{all_r[-1]['net']:+,.0f}  MDD={all_r[-1]['mdd_p']:.2f}%  +{(all_r[-1]['net']/b_net-1)*100:+.1f}%")

# ── 2. AM 1w1l best from Phase 1 ──
all_r.append(test("2_CH20_1w1l", am_rule=(1,1)))
print(f"2_CH20_1w1l:       Rs{all_r[-1]['net']:+,.0f}  MDD={all_r[-1]['mdd_p']:.2f}%  +{(all_r[-1]['net']/b_net-1)*100:+.1f}%")

# ── 3. Dyn CH25 + AM 1w1l (combine best two) ──
all_r.append(test("3_DynCH25_1w1l", ch_base=25, dyn_ch=1, am_rule=(1,1)))
print(f"3_DynCH25_1w1l:    Rs{all_r[-1]['net']:+,.0f}  MDD={all_r[-1]['mdd_p']:.2f}%  +{(all_r[-1]['net']/b_net-1)*100:+.1f}%")

# ── 4. 3-regime dynamic CH (low/mid/high vol) ──
all_r.append(test("4_DynCH3reg_CH20", ch_base=20, dyn_ch=2))
print(f"4_DynCH3reg_CH20:  Rs{all_r[-1]['net']:+,.0f}  MDD={all_r[-1]['mdd_p']:.2f}%  +{(all_r[-1]['net']/b_net-1)*100:+.1f}%")
all_r.append(test("4b_DynCH3reg_CH25", ch_base=25, dyn_ch=2))
print(f"4b_DynCH3reg_CH25: Rs{all_r[-1]['net']:+,.0f}  MDD={all_r[-1]['mdd_p']:.2f}%  +{(all_r[-1]['net']/b_net-1)*100:+.1f}%")

# ── 5. Dyn CH25 + AM 1w1l + 3-regime ──
all_r.append(test("5_DynCH3reg25_1w1l", ch_base=25, dyn_ch=2, am_rule=(1,1)))
print(f"5_DynCH3reg25_1w1l: Rs{all_r[-1]['net']:+,.0f}  MDD={all_r[-1]['mdd_p']:.2f}%  +{(all_r[-1]['net']/b_net-1)*100:+.1f}%")

# ── 6. Gap size filters ──
print(f"\n--- Gap Size Filters (on DynCH25+1w1l) ---")
for gmin,gmax in [(0.1,3),(0.2,3),(0.3,5),(0.5,5),(0.5,3)]:
    r=test(f"6_Gap{gmin}-{gmax}", ch_base=25, dyn_ch=1, am_rule=(1,1), gap_min=gmin, gap_max=gmax)
    all_r.append(r)
    print(f"  Gap[{gmin}-{gmax}%]: Rs{r['net']:+,.0f}  n={r['n']}  MDD={r['mdd_p']:.2f}%  PF={r['pf']:.2f}  +{(r['net']/b_net-1)*100:+.1f}%")

# ── 7. Correlation filter ──
r=test("7_CorrFilter", ch_base=25, dyn_ch=1, am_rule=(1,1), corr_filter=True)
all_r.append(r)
print(f"7_CorrFilter:      Rs{r['net']:+,.0f}  n={r['n']}  MDD={r['mdd_p']:.2f}%  +{(r['net']/b_net-1)*100:+.1f}%")

# ── 8. DynCH + different base values (asymmetric) ──
print(f"\n--- Asymmetric Dynamic CH (only tighten in hi-vol, never widen) ---")
ML=3
for base,tight in [(20,5),(20,8),(25,5),(25,8),(25,10),(30,8)]:
    # Custom run: asymmetric adjustment
    wa,la=(1,1)
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
        h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
        h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
        body=(h1["close"]-h1["open"]).abs()
        is_red=h1["close"]<h1["open"]; is_green=h1["close"]>h1["open"]
        sigs=[]
        for i in range(1,len(h1)):
            if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.5: continue
            sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
        bl=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=A(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        c2=1; ws=0; ls=0
        for sig in sigs:
            idx_h1=sig["idx"]
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
            # Asymmetric: only tighten in high vol, never widen
            atr14_v=h1["atr14"].iloc[idx_h1]; atr_ma_v=h1["atr_ma20"].iloc[idx_h1]
            ch_m=base
            if not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v:
                ch_m=base-tight
            for j in range(rt+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-ch_m*ca:
                    pnl=(cl[j]-ep)*bl*c2 - CHG*c2
                    et.append({"pnl_rs":pnl,"lot":c2})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=wa and c2<ML: c2+=1; ws=0
                    if ls>=la and c2>1: c2-=1; ls=0
                    break
    eng=pd.DataFrame(et) if et else pd.DataFrame()
    mom=run_mom(10,24)
    c=pd.concat([eng,mom],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    c=lf(c,2)
    n=len(c); net=c["pnl_rs"].sum() if n>0 else 0
    wr=(c["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    pf=(c[c["pnl_rs"]>0]["pnl_rs"].sum()/abs(c[c["pnl_rs"]<0]["pnl_rs"].sum())) if (c["pnl_rs"]<0).sum()!=0 else 99
    eq=c["pnl_rs"].cumsum()+200000; pk=eq.cummax(); mdd=(pk-eq).max()
    mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
    alot=c[c["strat"]=="Eng"]["lot"].mean() if "Eng" in c["strat"].values else 0
    yrs=(pd.to_datetime(c["exit_time"]).max()-pd.to_datetime(c["exit_time"]).min()).total_seconds()/31536000 if len(c)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    ndd=net/mdd if mdd>0 else 0
    r={"label":f"8_AsymCH{base}T{tight}","n":n,"net":net,"wr":wr,"aw":0,"al":0,"pf":pf,"mdd":mdd,"mdd_p":mdd_p,"alot":alot,"cagr":cagr,"ndd":ndd}
    all_r.append(r)
    print(f"  Asym{base}T{tight}: Rs{net:+,.0f}  MDD={mdd_p:.2f}%  PF={pf:.2f}  +{(net/b_net-1)*100:+.1f}%")

# ── FINAL RANKING ──
print(f"\n{'='*130}")
print("PHASE 2 FINAL RANKING")
print(f"{'='*130}")
print(f"{'RANK':>4s}  {'Variant':30s}  {'Trades':>5s}  {'Net_RS':>11s}  {'WR%':>5s}  {'PF':>5s}  "
      f"{'MDD%':>6s}  {'CAGR':>6s}  {'AvgLot':>6s}  {'Net/MDD':>7s}  {'vsBASE':>8s}")
print("-"*110)
sorted_r=sorted(all_r, key=lambda x: x["net"], reverse=True)
for i,r in enumerate(sorted_r):
    vs=(r["net"]/b_net-1)*100
    print(f"{i+1:4d}  {r['label']:30s}  {r['n']:4d}  Rs{r['net']:>+9,.0f}  {r['wr']:4.1f}%  "
          f"{r['pf']:5.2f}  {r['mdd_p']:5.2f}%  {r['cagr']:5.1f}%  {r['alot']:5.2f}  "
          f"{r['ndd']:6.1f}x  {vs:>+7.1f}%")

out_dir=os.path.join(BASE,"backtest_results","deep_research")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(sorted_r).to_csv(os.path.join(out_dir,"deep_research_phase2.csv"),index=False)
print(f"\nSaved to: {out_dir}")
print("DONE")
