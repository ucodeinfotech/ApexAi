"""
Risk Optimization — Minimize drawdown and losses
Tests tighter stops, more aggressive loss skipping, lower max lots, time stops.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()

def compute_atr(df, p=14):
    tr = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# ── Engulfing with configurable params ──

def run_engulfing(max_lots=3, skip_n=2, chandelier_mult=15):
    """Run engulfing with configurable risk params."""
    all_t = []
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
                if cl[j]<highest-chandelier_mult*ca:
                    pts=cl[j]-entry
                    pnl=pts*base_lot*cur_lots - CHG*cur_lots
                    all_t.append({"points":pts,"pnl_rs":pnl,"lot":cur_lots,
                        "exit_time":m5["datetime"].iloc[j],
                        "sym":sym})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=2 and cur_lots<max_lots: cur_lots+=1; ws=0
                    if ls>=1 and cur_lots>1: cur_lots-=1; ls=0
                    break
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True) if all_t else pd.DataFrame()

# ── Momentum with configurable params ──

def run_momentum(chandelier_mult=10, time_stop_h=0, skip_n=2, fixed_risk_stop=False):
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
        atr=compute_atr(h1,14); hi20=h1["high"].rolling(20).max().shift(1)
        lot=NLOT if "NIFTY" in sym else SLOT
        intrade=False; ep=0; hi_en=0; entry_idx=0
        for i in range(20,len(h1)):
            if not intrade:
                if h1["close"].iloc[i]>hi20.iloc[i] and h1["close"].iloc[i]>h1["open"].iloc[i] and h1["datetime"].iloc[i].time()<CUTOFF_TIME and h1["datetime"].iloc[i].hour>=9:
                    intrade=True; ep=h1["close"].iloc[i]; hi_en=ep; entry_idx=i
            else:
                if h1["high"].iloc[i]>hi_en: hi_en=h1["high"].iloc[i]
                ca=atr.iloc[i]
                exit_here=False
                # Chandelier exit
                if not pd.isna(ca):
                    if fixed_risk_stop:
                        risk = h1["high"].iloc[entry_idx] - h1["low"].iloc[entry_idx]
                        if h1["close"].iloc[i] < hi_en - chandelier_mult * ca:
                            exit_here=True
                        elif h1["close"].iloc[i] < ep - 2*risk:
                            exit_here=True
                    elif h1["close"].iloc[i] < hi_en - chandelier_mult * ca:
                        exit_here=True
                # Time stop
                if time_stop_h>0 and not exit_here:
                    hours=(h1["datetime"].iloc[i]-h1["datetime"].iloc[entry_idx]).total_seconds()/3600
                    if hours>time_stop_h and h1["close"].iloc[i]<=ep:
                        exit_here=True
                if exit_here:
                    pts=h1["close"].iloc[i]-ep; pnl=pts*lot-CHG
                    all_t.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl})
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

def calc_metrics(df, name):
    if len(df)==0: return {"name":name,"trades":0,"net_rs":0,"wr":0,"pf":0,"cagr":0,"sharpe":0,"mdd":0,"mdd_pct":0,"avg_hold":0}
    n=len(df); net=df["pnl_rs"].sum(); wr=(df["pnl_rs"]>0).sum()/n*100
    pf=(df[df["pnl_rs"]>0]["pnl_rs"].sum()/abs(df[df["pnl_rs"]<0]["pnl_rs"].sum())) if (df["pnl_rs"]<0).sum()!=0 else 99
    cum=df["pnl_rs"].cumsum()+200000; peak=cum.cummax(); mdd=(peak-cum).max()
    mdd_pct=mdd/peak.max()*100 if peak.max()>0 else 0
    yrs=(df["exit_time"].max()-df["exit_time"].min()).total_seconds()/31536000 if len(df)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    sh=df["pnl_rs"].mean()/df["pnl_rs"].std()*np.sqrt(252*6.5) if df["pnl_rs"].std()>0 else 0
    avg_h=df.get("hold_hours", pd.Series([0])).mean()
    return {"name":name,"trades":n,"net_rs":net,"wr":wr,"pf":pf,"cagr":cagr,"sharpe":sh,"mdd":mdd,"mdd_pct":mdd_pct}

print("="*120)
print("RISK OPTIMIZATION — Minimize Drawdown & Losses")
print("="*120)

# Define all variants to test
variants = []

# Baseline: Engulfing dyn 2w1l + Momentum 10x
eng=run_engulfing(max_lots=3, skip_n=2)
mom=run_momentum(10, 0)
eng_f=loss_filter(eng,2); mom_f=loss_filter(mom,2)
combo=pd.concat([eng_f,mom_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo=loss_filter(combo,2)
variants.append(calc_metrics(combo, "A) Baseline (max3, skip2, mom10x)"))

# B) Max lots=2
eng2=run_engulfing(max_lots=2, skip_n=2)
mom2=run_momentum(10, 0)
eng2_f=loss_filter(eng2,2); mom2_f=loss_filter(mom2,2)
combo2=pd.concat([eng2_f,mom2_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo2=loss_filter(combo2,2)
variants.append(calc_metrics(combo2, "B) Max lots=2"))

# C) Skip after 1 loss (more aggressive)
eng3=run_engulfing(max_lots=3, skip_n=1)
mom3=run_momentum(10, 0)
eng3_f=loss_filter(eng3,1); mom3_f=loss_filter(mom3,1)
combo3=pd.concat([eng3_f,mom3_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo3=loss_filter(combo3,1)
variants.append(calc_metrics(combo3, "C) Skip after 1 loss"))

# D) Momentum tighter stop (7x instead of 10x)
eng4=run_engulfing(max_lots=3, skip_n=2)
mom4=run_momentum(7, 0)
eng4_f=loss_filter(eng4,2); mom4_f=loss_filter(mom4,2)
combo4=pd.concat([eng4_f,mom4_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo4=loss_filter(combo4,2)
variants.append(calc_metrics(combo4, "D) Mom stop 7x"))

# E) Momentum time stop (exit after 24h if not profitable)
eng5=run_engulfing(max_lots=3, skip_n=2)
mom5=run_momentum(10, 24)
eng5_f=loss_filter(eng5,2); mom5_f=loss_filter(mom5,2)
combo5=pd.concat([eng5_f,mom5_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo5=loss_filter(combo5,2)
variants.append(calc_metrics(combo5, "E) Mom time stop 24h"))

# F) Max lots=2 + skip after 1 + mom stop 7x
eng6=run_engulfing(max_lots=2, skip_n=1)
mom6=run_momentum(7, 0)
eng6_f=loss_filter(eng6,1); mom6_f=loss_filter(mom6,1)
combo6=pd.concat([eng6_f,mom6_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo6=loss_filter(combo6,1)
variants.append(calc_metrics(combo6, "F) max2+skip1+mom7x"))

# G) Max lots=2 + skip=2 + mom 7x
eng7=run_engulfing(max_lots=2, skip_n=2)
mom7=run_momentum(7, 0)
eng7_f=loss_filter(eng7,2); mom7_f=loss_filter(mom7,2)
combo7=pd.concat([eng7_f,mom7_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo7=loss_filter(combo7,2)
variants.append(calc_metrics(combo7, "G) max2+skip2+mom7x"))

# H) Max lots=2 + skip=2 + mom 5x
eng8=run_engulfing(max_lots=2, skip_n=2)
mom8=run_momentum(5, 0)
eng8_f=loss_filter(eng8,2); mom8_f=loss_filter(mom8,2)
combo8=pd.concat([eng8_f,mom8_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo8=loss_filter(combo8,2)
variants.append(calc_metrics(combo8, "H) max2+skip2+mom5x"))

# I) Chandelier 12x for engulfing (tighter)
eng9=run_engulfing(max_lots=3, skip_n=2, chandelier_mult=12)
mom9=run_momentum(10, 0)
eng9_f=loss_filter(eng9,2); mom9_f=loss_filter(mom9,2)
combo9=pd.concat([eng9_f,mom9_f],ignore_index=True).sort_values("exit_time").reset_index(drop=True)
combo9=loss_filter(combo9,2)
variants.append(calc_metrics(combo9, "I) Eng CH12 + mom10x"))

# J) Just Engulfing with max lots=2 (no momentum) - simplest
eng10=run_engulfing(max_lots=2, skip_n=2)
eng10_f=loss_filter(eng10,2)
variants.append(calc_metrics(eng10_f, "J) Engulfing only, max2"))

# K) Engulfing only with maxlots=2 + skip=1
eng11=run_engulfing(max_lots=2, skip_n=1)
eng11_f=loss_filter(eng11,1)
variants.append(calc_metrics(eng11_f, "K) Eng only, max2+skip1"))

# L) Engulfing only baseline (no momentum, max lots=3)
eng12=run_engulfing(max_lots=3, skip_n=2)
eng12_f=loss_filter(eng12,2)
variants.append(calc_metrics(eng12_f, "L) Eng only, baseline max3"))

# M) Engulfing only with CH12, max2, skip2
eng13=run_engulfing(max_lots=2, skip_n=2, chandelier_mult=12)
eng13_f=loss_filter(eng13,2)
variants.append(calc_metrics(eng13_f, "M) Eng CH12+max2+skip2"))

# Print ranking by P&L (primary) with drawdown as secondary
print(f"\n{'#'*120}")
print(f"{'RANK':>4s}  {'Variant':40s}  {'Trades':>6s}  {'Net_RS':>10s}  {'WR%':>5s}  "
      f"{'PF':>5s}  {'CAGR':>6s}  {'Sharpe':>7s}  {'MaxDD_RS':>9s}  {'MaxDD%':>7s}")
print("-"*120)
for i,v in enumerate(sorted(variants, key=lambda x: x["net_rs"], reverse=True)):
    print(f"{i+1:4d}  {v['name']:40s}  {v['trades']:5d}  Rs{v['net_rs']:>+8,.0f}  "
          f"{v['wr']:4.1f}%  {v['pf']:5.2f}  {v['cagr']:5.1f}%  {v['sharpe']:6.2f}  "
          f"Rs{v['mdd']:>+7,.0f}  {v['mdd_pct']:6.2f}%")

# Sort by drawdown (ascending) for risk-focused view
print(f"\n{'#'*120}")
print(f"RANKED BY LOWEST DRAWDOWN")
print(f"{'#'*120}")
print(f"{'RANK':>4s}  {'Variant':40s}  {'MaxDD_RS':>9s}  {'MaxDD%':>7s}  {'Net_RS':>10s}  "
      f"{'PF':>5s}  {'Sharpe':>7s}  {'Trades':>6s}")
print("-"*120)
for i,v in enumerate(sorted(variants, key=lambda x: x["mdd"])):
    print(f"{i+1:4d}  {v['name']:40s}  Rs{v['mdd']:>+7,.0f}  {v['mdd_pct']:6.2f}%  "
          f"Rs{v['net_rs']:>+8,.0f}  {v['pf']:5.2f}  {v['sharpe']:6.2f}  {v['trades']:4d}")

# Combined metric: return / drawdown ratio
print(f"\n{'#'*120}")
print(f"RANKED BY RETURN/DRAWDOWN RATIO (best = highest return per unit of drawdown)")
print(f"{'#'*120}")
print(f"{'RANK':>4s}  {'Variant':40s}  {'Net/MDD':>8s}  {'Net_RS':>10s}  {'MaxDD_RS':>9s}  "
      f"{'PF':>5s}  {'WR%':>5s}")
print("-"*120)
for i,v in enumerate(sorted(variants, key=lambda x: x["net_rs"]/x["mdd"] if x["mdd"]>0 else 0, reverse=True)):
    ratio=v["net_rs"]/v["mdd"] if v["mdd"]>0 else 0
    print(f"{i+1:4d}  {v['name']:40s}  {ratio:7.2f}x  Rs{v['net_rs']:>+8,.0f}  "
          f"Rs{v['mdd']:>+7,.0f}  {v['pf']:5.2f}  {v['wr']:4.1f}%")

out_dir=os.path.join(BASE,"backtest_results","risk_optimizer")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(variants).sort_values("net_rs",ascending=False).to_csv(os.path.join(out_dir,"risk_optimizer.csv"),index=False)
print(f"\nSaved to: {out_dir}")
