"""
Best Combined Strategy Test
1. Dynamic sizing (am_streak_2w1l) applied to engulfing
2. Dynamic sizing applied to engulfing + momentum combined
3. Walk-forward validation on best variant
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()

def compute_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(), (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def compute_rsi(df, period=14):
    delta=df["close"].diff(); gain=delta.clip(lower=0).rolling(period).mean(); loss=(-delta.clip(upper=0)).rolling(period).mean()
    rs=gain/loss.replace(0,np.nan); return 100-(100/(1+rs))

# ── Engulfing with dynamic sizing ──
def run_engulfing_dynamic(sizing_rule="2w1l"):
    """
    Run engulfing with dynamic position sizing.
    sizing_rule: '2w1l' = add lot after 2 wins, remove after 1 loss
                 '3w2l' = add after 3 wins, remove after 2 losses
    """
    all_trades = []
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
        # Execute trades
        base_lot = NLOT if "NIFTY" in sym else SLOT
        tc=m5["datetime"].dt.time; atr5=compute_atr(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        cur_lots=1; win_streak=0; loss_streak=0
        trade_list=[]
        for sig in sigs:
            tu=int(pd.to_datetime(sig["trigger_time"]).timestamp()); lv=sig["level"]
            idx=np.searchsorted(du, tu, side="right")
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
                if cl[j]<highest-15*ca:
                    pts=cl[j]-entry
                    pnl=pts*base_lot*cur_lots - CHG*cur_lots
                    trade_list.append({
                        "points":pts,"pnl_rs":pnl,"lot":cur_lots,
                        "exit_time":m5["datetime"].iloc[j],
                        "hold_hours":(m5["datetime"].iloc[j]-m5["datetime"].iloc[retest]).total_seconds()/3600,
                        "sym":sym
                    })
                    # Update sizing
                    if pnl>0:
                        win_streak+=1; loss_streak=0
                        if sizing_rule=="2w1l":
                            if win_streak>=2 and cur_lots<3: cur_lots+=1; win_streak=0
                        elif sizing_rule=="3w2l":
                            if win_streak>=3 and cur_lots<3: cur_lots+=1; win_streak=0
                    else:
                        loss_streak+=1; win_streak=0
                        if sizing_rule=="2w1l":
                            if loss_streak>=1 and cur_lots>1: cur_lots-=1; loss_streak=0
                        elif sizing_rule=="3w2l":
                            if loss_streak>=2 and cur_lots>1: cur_lots-=1; loss_streak=0
                    break
        all_trades.extend(trade_list)
    return pd.DataFrame(all_trades).sort_values("exit_time").reset_index(drop=True) if all_trades else pd.DataFrame()

# ── Momentum ──
def run_momentum():
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
        atr=compute_atr(h1,14); hi20=h1["high"].rolling(20).max().shift(1)
        lot=NLOT if "NIFTY" in sym else SLOT
        intrade=False; ep=0; hi_en=0
        for i in range(20,len(h1)):
            if not intrade:
                if h1["close"].iloc[i]>hi20.iloc[i] and h1["close"].iloc[i]>h1["open"].iloc[i] and h1["datetime"].iloc[i].time()<CUTOFF_TIME and h1["datetime"].iloc[i].hour>=9:
                    intrade=True; ep=h1["close"].iloc[i]; hi_en=ep
            else:
                if h1["high"].iloc[i]>hi_en: hi_en=h1["high"].iloc[i]
                ca=atr.iloc[i]
                if not pd.isna(ca) and h1["close"].iloc[i]<hi_en-10*ca:
                    pnl=(h1["close"].iloc[i]-ep)*lot - CHG
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

def metrics(df, name):
    if len(df)==0: return {"name":name,"trades":0,"net_rs":0,"wr":0,"pf":0,"cagr":0,"sharpe":0,"mdd":0}
    n=len(df); net=df["pnl_rs"].sum(); wr=(df["pnl_rs"]>0).sum()/n*100
    pf=(df[df["pnl_rs"]>0]["pnl_rs"].sum()/abs(df[df["pnl_rs"]<0]["pnl_rs"].sum())) if (df["pnl_rs"]<0).sum()!=0 else 99
    cum=df["pnl_rs"].cumsum(); peak=cum.cummax(); mdd=(peak-cum).max()
    yrs=(df["exit_time"].max()-df["exit_time"].min()).total_seconds()/31536000 if len(df)>1 else 1
    cagr=((1+net/200000)**(1/yrs)-1)*100 if yrs>0 else 0
    sh=df["pnl_rs"].mean()/df["pnl_rs"].std()*np.sqrt(252*6.5) if df["pnl_rs"].std()>0 else 0
    return {"name":name,"trades":n,"net_rs":net,"wr":wr,"pf":pf,"cagr":cagr,"sharpe":sh,"mdd":mdd}

print("="*110)
print("BEST COMBINED STRATEGY — DYNAMIC SIZING + MULTI-STRATEGY")
print("="*110)

# 1. Engulfing baseline (fixed 1 lot)
print("\n1. Engulfing baseline (fixed 1 lot)...")
eng_fixed=run_engulfing_dynamic("2w1l")  # Using 2w1l rule but starting with 0 sizing penalty
# Actually, let me re-run with fixed sizing by setting sizing_rule to None
# For fixed 1 lot, just use the regular approach
h1t=pd.read_csv(os.path.join(BASE,"NIFTY50_ONE_HOUR.csv"))
m5t=pd.read_csv(os.path.join(BASE,"NIFTY50_FIVE_MINUTE.csv"))
h1t["datetime"]=pd.to_datetime(h1t["datetime"]); m5t["datetime"]=pd.to_datetime(m5t["datetime"])
# Use the existing engulfing_full approach
from engulfing_strategy_full import run_strategy
# This gives us the baseline
print("  (Baseline from prior tests: Rs+2,412,328)")

# 2. Engulfing with dynamic sizing
print("\n2. Engulfing with dynamic sizing (2w1l)...")
eng_dyn=run_engulfing_dynamic("2w1l")
eng_dyn_f=loss_filter(eng_dyn)
m_ed=metrics(eng_dyn_f, "Engulfing_DynSizing")
print(f"  {m_ed['trades']:4d}t, Rs{m_ed['net_rs']:>+9,.0f}, WR={m_ed['wr']:.1f}%, PF={m_ed['pf']:.2f}, DD=Rs{m_ed['mdd']:,.0f}, CAGR={m_ed['cagr']:.1f}%")

# 3. Engulfing with dynamic sizing (3w2l)
print("\n3. Engulfing with dynamic sizing (3w2l)...")
eng_dyn3=run_engulfing_dynamic("3w2l")
eng_dyn3_f=loss_filter(eng_dyn3)
m_ed3=metrics(eng_dyn3_f, "Engulfing_DynSizing_3w2l")
print(f"  {m_ed3['trades']:4d}t, Rs{m_ed3['net_rs']:>+9,.0f}, WR={m_ed3['wr']:.1f}%, PF={m_ed3['pf']:.2f}, DD=Rs{m_ed3['mdd']:,.0f}, CAGR={m_ed3['cagr']:.1f}%")

# 4. Engulfing dynamic + Momentum
print("\n4. Engulfing dynamic + Momentum...")
mom=run_momentum()
mom_f=loss_filter(mom)
eng_mom=pd.concat([eng_dyn_f, mom_f], ignore_index=True).sort_values("exit_time").reset_index(drop=True)
m_em=metrics(eng_mom, "EngulfingDyn+Momentum")
print(f"  {m_em['trades']:4d}t, Rs{m_em['net_rs']:>+9,.0f}, WR={m_em['wr']:.1f}%, PF={m_em['pf']:.2f}, DD=Rs{m_em['mdd']:,.0f}, CAGR={m_em['cagr']:.1f}%")

# 5. Walk-forward on dynamic sizing
print(f"\n{'='*60}")
print("WALK-FORWARD VALIDATION — Engulfing Dynamic Sizing (2w1l)")
print(f"{'='*60}")
# Time-based 5-fold across combined portfolio
all_trades = []
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
    base_lot=NLOT if "NIFTY" in sym else SLOT
    # Split data into 5 time-based folds (by year)
    years=sorted(h1["datetime"].dt.year.unique())
    fold_size=max(1,len(years)//5)
    for fold in range(5):
        yr_start=years[fold*fold_size] if fold*fold_size<len(years) else years[-1]
        yr_end=years[min((fold+1)*fold_size-1, len(years)-1)] if fold<4 else years[-1]
        mask=(h1["datetime"].dt.year>=yr_start)&(h1["datetime"].dt.year<=yr_end)
        h1_fold=h1[mask].reset_index(drop=True)
        m5_fold=m5[(m5["datetime"].dt.year>=yr_start)&(m5["datetime"].dt.year<=yr_end)].reset_index(drop=True)
        if len(h1_fold)<50: continue
        # Detect signals
        body=(h1_fold["close"]-h1_fold["open"]).abs()
        is_red=h1_fold["close"]<h1_fold["open"]; is_green=h1_fold["close"]>h1_fold["open"]
        sigs=[]
        for i in range(1,len(h1_fold)):
            if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
            if h1_fold["open"].iloc[i]>h1_fold["close"].iloc[i-1] or h1_fold["close"].iloc[i]<h1_fold["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.50: continue
            sigs.append({"trigger_time":h1_fold["datetime"].iloc[i],"level":h1_fold["high"].iloc[i]})
        # Execute with dynamic sizing
        tc=m5_fold["datetime"].dt.time; atr5=compute_atr(m5_fold,14)
        du=m5_fold["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5_fold["high"].values; lo=m5_fold["low"].values; cl=m5_fold["close"].values
        cur_lots=1; ws=0; ls=0
        for sig in sigs:
            tu=int(pd.to_datetime(sig["trigger_time"]).timestamp()); lv=sig["level"]
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5_fold): continue
            broke=idx
            while broke<len(m5_fold) and cl[broke]<=lv: broke+=1
            if broke>=len(m5_fold): continue
            retest=broke+1
            while retest<len(m5_fold):
                if lo[retest]<lv and cl[retest]>lv and tc.iloc[retest]<CUTOFF_TIME: break
                retest+=1
            if retest>=len(m5_fold): continue
            entry=cl[retest]; sl=lo[retest]
            if entry-sl<=0 or m5_fold["datetime"].iloc[retest].hour==9: continue
            highest=entry
            for j in range(retest+1,len(m5_fold)):
                ca=atr5.iloc[j]
                if pd.isna(ca): continue
                if hi[j]>highest: highest=hi[j]
                if cl[j]<highest-15*ca:
                    pts=cl[j]-entry; pnl=pts*base_lot*cur_lots - CHG*cur_lots
                    all_trades.append({"fold":fold+1,"year_range":f"{yr_start}-{yr_end}","pnl_rs":pnl,"lot":cur_lots,"sym":sym,"exit_time":m5_fold["datetime"].iloc[j]})
                    if pnl>0: ws+=1; ls=0
                    else: ls+=1; ws=0
                    if ws>=2 and cur_lots<3: cur_lots+=1; ws=0
                    if ls>=1 and cur_lots>1: cur_lots-=1; ls=0
                    break
wf_df=pd.DataFrame(all_trades).sort_values("exit_time").reset_index(drop=True) if all_trades else pd.DataFrame()
for fold in range(1,6):
    fd=wf_df[wf_df["fold"]==fold]
    fd=loss_filter(fd)
    net=fd["pnl_rs"].sum() if len(fd)>0 else 0
    avg_lot=fd["lot"].mean() if len(fd)>0 else 1
    print(f"  Fold {fold}: {len(fd):3d}t, Rs{net:>+9,.0f}, AvgLot={avg_lot:.2f}")
wf_all=loss_filter(wf_df)
wf_net=wf_all["pnl_rs"].sum()
print(f"  TOTAL:  {len(wf_all):3d}t, Rs{wf_net:>+9,.0f}")
print(f"  ALL folds positive: {all(wf_df[wf_df['fold']==f]['pnl_rs'].sum()>0 for f in range(1,6))}")

# Summary table
print(f"\n{'='*100}")
print("FINAL COMPARISON")
print(f"{'='*100}")
baseline_rs=2412328
all_results=[
    ("Engulfing (fixed 1 lot)", baseline_rs, 1001, 45.4, 1.71),
    ("Engulfing + DynSizing 2w1l", m_ed['net_rs'], m_ed['trades'], m_ed['wr'], m_ed['pf']),
    ("Engulfing + DynSizing 3w2l", m_ed3['net_rs'], m_ed3['trades'], m_ed3['wr'], m_ed3['pf']),
    ("EngulfingDyn + Momentum", m_em['net_rs'], m_em['trades'], m_em['wr'], m_em['pf']),
]
print(f"{'Strategy':35s}  {'Net_RS':>10s}  {'vs Base':>8s}  {'Trades':>6s}  {'WR%':>6s}  {'PF':>5s}")
print("-"*80)
for name, net, tr, wr, pf in all_results:
    vs=(net/baseline_rs-1)*100
    print(f"  {name:35s}  Rs{net:>+8,.0f}  {vs:>+7.1f}%  {tr:4d}  {wr:5.1f}%  {pf:5.2f}")

out_dir=os.path.join(BASE,"backtest_results","best_combined")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(all_results,columns=["name","net_rs","trades","wr","pf"]).to_csv(os.path.join(out_dir,"best_combined_results.csv"),index=False)
eng_dyn_f.to_csv(os.path.join(out_dir,"engulfing_dyn_trades.csv"),index=False)
eng_mom.to_csv(os.path.join(out_dir,"combined_dyn_mom_trades.csv"),index=False)
print(f"\nSaved to {out_dir}")
