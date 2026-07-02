"""
SL Reduction Test — Test tighter stops on the combined strategy.
Tests various Chandelier multipliers and initial stop types.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()

def compute_atr(df, p=14):
    tr = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# ── Engulfing with adjustable stop ──
def run_engulfing(ch_mult=15, max_lots=3):
    all_t=[]
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
        base_lot=NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=compute_atr(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        cur_lots=1; ws=0; ls=0
        for sig in sigs:
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
            highest=entry
            for j in range(retest+1,len(m5)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>highest: highest=hi[j]
                if cl[j]<highest-ch_mult*ca:
                    pts=cl[j]-entry
                    pnl=pts*base_lot*cur_lots - CHG*cur_lots
                    all_t.append({"pnl_rs":pnl,"lot":cur_lots,"exit_time":m5["datetime"].iloc[j],"sym":sym,"strat":"Engulfing"})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=2 and cur_lots<max_lots: cur_lots+=1; ws=0
                    if ls>=1 and cur_lots>1: cur_lots-=1; ls=0
                    break
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True) if all_t else pd.DataFrame()

# ── Momentum with adjustable stop ──
def run_momentum(ch_mult=10, time_stop_h=24):
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
        atr=compute_atr(h1,14); hi20=h1["high"].rolling(20).max().shift(1)
        lot=NLOT if "NIFTY" in sym else SLOT
        intrade=False; ep=0; hi_en=0; entry_time=None
        for i in range(20,len(h1)):
            if not intrade:
                if h1["close"].iloc[i]>hi20.iloc[i] and h1["close"].iloc[i]>h1["open"].iloc[i] and h1["datetime"].iloc[i].time()<CUTOFF_TIME and h1["datetime"].iloc[i].hour>=9:
                    intrade=True; ep=h1["close"].iloc[i]; hi_en=ep; entry_time=h1["datetime"].iloc[i]
            else:
                if h1["high"].iloc[i]>hi_en: hi_en=h1["high"].iloc[i]
                ca=atr.iloc[i]
                exit_here=False
                if not pd.isna(ca) and h1["close"].iloc[i] < hi_en - ch_mult*ca:
                    exit_here=True
                if time_stop_h>0 and not exit_here:
                    hours=(h1["datetime"].iloc[i]-entry_time).total_seconds()/3600
                    if hours>time_stop_h and h1["close"].iloc[i]<=ep:
                        exit_here=True
                if exit_here:
                    pnl=(h1["close"].iloc[i]-ep)*lot-CHG
                    all_t.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl,"strat":"Momentum","lot":1,"sym":sym})
                    intrade=False
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True) if all_t else pd.DataFrame()

def loss_filter(df, skip_n=2):
    df=df.sort_values("exit_time").reset_index(drop=True)
    lc=0; k=np.ones(len(df),dtype=bool)
    for i in range(len(df)):
        if lc>=skip_n: k[i]=False; lc=0; continue
        if df["pnl_rs"].iloc[i]<=0: lc+=1
        else: lc=0
    return df[k].reset_index(drop=True)

def run_test(label, eng_ch, mom_ch, mom_time_stop=24):
    eng=run_engulfing(eng_ch, 3)
    mom=run_momentum(mom_ch, mom_time_stop)
    combo=pd.concat([eng,mom],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    combo=loss_filter(combo,2)
    n=len(combo); net=combo["pnl_rs"].sum() if n>0 else 0
    wr=(combo["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    pf=(combo[combo["pnl_rs"]>0]["pnl_rs"].sum()/abs(combo[combo["pnl_rs"]<0]["pnl_rs"].sum())) if (combo["pnl_rs"]<0).sum()!=0 else 99
    eq=combo["pnl_rs"].cumsum()+200000; peak=eq.cummax(); mdd=(peak-eq).max()
    mdd_pct=mdd/peak.max()*100 if peak.max()>0 else 0
    yrs=(pd.to_datetime(combo["exit_time"]).max()-pd.to_datetime(combo["exit_time"]).min()).total_seconds()/31536000 if len(combo)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    avg_lot=combo[combo["strat"]=="Engulfing"]["lot"].mean() if "Engulfing" in combo["strat"].values else 0
    return {"label":label,"trades":n,"net_rs":net,"wr":wr,"pf":pf,"mdd":mdd,"mdd_pct":mdd_pct,"cagr":cagr,"avg_lot":avg_lot}

tests = [
    # (label, eng_ch, mom_ch, mom_time_stop)
    ("Baseline E15+M10+TS24", 15, 10, 24),
    # Reduce engulfing stop
    ("Eng_CH12+M10+TS24",     12, 10, 24),
    ("Eng_CH10+M10+TS24",     10, 10, 24),
    ("Eng_CH8+M10+TS24",      8,  10, 24),
    ("Eng_CH5+M10+TS24",      5,  10, 24),
    # Reduce momentum stop
    ("Eng_CH15+M7+TS24",      15, 7,  24),
    ("Eng_CH15+M5+TS24",      15, 5,  24),
    ("Eng_CH15+M3+TS24",      15, 3,  24),
    # Reduce both
    ("Eng_CH12+M7+TS24",      12, 7,  24),
    ("Eng_CH10+M7+TS24",      10, 7,  24),
    ("Eng_CH10+M5+TS24",      10, 5,  24),
    ("Eng_CH8+M5+TS24",       8,  5,  24),
    # Tighter with no time stop
    ("Eng_CH15+M7+TS0",       15, 7,  0),
    ("Eng_CH15+M5+TS0",       15, 5,  0),
    # Also test wider for reference
    ("Eng_CH18+M10+TS24",     18, 10, 24),
    ("Eng_CH20+M10+TS24",     20, 10, 24),
]

print("="*120)
print("SL REDUCTION TEST — Combined Strategy Variant E")
print("="*120)
print(f"\n{'Variant':30s}  {'Trades':>5s}  {'Net_RS':>10s}  {'WR%':>5s}  {'PF':>5s}  "
      f"{'MDD_RS':>9s}  {'MDD%':>6s}  {'CAGR':>6s}  {'AvgLot':>6s}")
print("-"*110)

results=[]
for label,eng_ch,mom_ch,ts in tests:
    r=run_test(label,eng_ch,mom_ch,ts)
    results.append(r)
    print(f"{label:30s}  {r['trades']:4d}  Rs{r['net_rs']:>+8,.0f}  {r['wr']:4.1f}%  "
          f"{r['pf']:5.2f}  Rs{r['mdd']:>+7,.0f}  {r['mdd_pct']:5.2f}%  {r['cagr']:5.1f}%  {r['avg_lot']:5.2f}")

print(f"\n{'='*120}")
print("RANKED BY NET P&L")
print(f"{'='*120}")
for i,r in enumerate(sorted(results, key=lambda x: x["net_rs"], reverse=True)):
    print(f"{i+1:2d}. {r['label']:30s}  Rs{r['net_rs']:>+9,.0f}  MDD={r['mdd_pct']:5.2f}%  PF={r['pf']:5.2f}  WR={r['wr']:4.1f}%  T={r['trades']:4d}")

print(f"\n{'='*120}")
print("RANKED BY LOWEST DRAWDOWN %")
print(f"{'='*120}")
for i,r in enumerate(sorted(results, key=lambda x: x["mdd_pct"])):
    print(f"{i+1:2d}. {r['label']:30s}  MDD={r['mdd_pct']:5.2f}%  Rs{r['net_rs']:>+9,.0f}  PF={r['pf']:5.2f}  WR={r['wr']:4.1f}%  T={r['trades']:4d}")

print(f"\n{'='*120}")
print("RANKED BY RETURN/DRAWDOWN RATIO")
print(f"{'='*120}")
for i,r in enumerate(sorted(results, key=lambda x: x["net_rs"]/x["mdd"] if x["mdd"]>0 else 0, reverse=True)):
    ratio=r["net_rs"]/r["mdd"] if r["mdd"]>0 else 0
    print(f"{i+1:2d}. {r['label']:30s}  Rat={ratio:6.1f}x  Rs{r['net_rs']:>+9,.0f}  MDD={r['mdd_pct']:5.2f}%  PF={r['pf']:5.2f}")

out_dir=os.path.join(BASE,"backtest_results","sl_reduction")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(results).sort_values("net_rs",ascending=False).to_csv(os.path.join(out_dir,"sl_reduction_results.csv"),index=False)
print(f"\nSaved to: {out_dir}")
