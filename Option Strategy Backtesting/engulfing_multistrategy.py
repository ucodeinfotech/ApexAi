"""
Multi-Strategy Overlay Test
Combine Engulfing with Mean-Reversion and Momentum strategies.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()

# ── Shared Helpers ──

def compute_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(), (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def compute_rsi(df, period=14):
    delta=df["close"].diff(); gain=delta.clip(lower=0).rolling(period).mean(); loss=(-delta.clip(upper=0)).rolling(period).mean()
    rs=gain/loss.replace(0,np.nan); return 100-(100/(1+rs))

# ============================================================
# STRATEGY 1: Engulfing (CH15 + skip=2) — our baseline
# ============================================================

def run_engulfing():
    """Return series of P&L values per trade (chronological), skip=2 applied."""
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
            if body.iloc[i]<body.iloc[i-1]*0.5: continue
            sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i]})
        # Execute
        tc=m5["datetime"].dt.time; atr5=compute_atr(m5,14)
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        lot=NLOT if "NIFTY" in sym else SLOT
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
                    pnl=(cl[j]-entry)*lot - CHG
                    all_t.append({"exit_time":m5["datetime"].iloc[j],"pnl_rs":pnl})
                    break
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True)

# ============================================================
# STRATEGY 2: RSI Mean-Reversion
# Buy when RSI(2) < 5 (extreme oversold), exit when RSI > 50
# ============================================================

def run_rsi_reversion():
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
        rsi2=compute_rsi(h1, 2)
        rsi14=compute_rsi(h1, 14)
        in_trade=False; entry_price=0; entry_time=None; idx_entry=0
        lot=NLOT if "NIFTY" in sym else SLOT
        for i in range(1, len(h1)):
            if not in_trade:
                if not pd.isna(rsi2.iloc[i]) and rsi2.iloc[i] < 5:
                    in_trade=True; entry_price=h1["close"].iloc[i]; entry_time=h1["datetime"].iloc[i]; idx_entry=i
            else:
                if not pd.isna(rsi14.iloc[i]) and rsi14.iloc[i] > 50:
                    pnl=(h1["close"].iloc[i]-entry_price)*lot - CHG
                    all_t.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl})
                    in_trade=False
                # Trail stop: exit if close < entry - 3*ATR
                atr_val=compute_atr(h1).iloc[i]
                if not pd.isna(atr_val) and h1["close"].iloc[i] < entry_price - 3*atr_val:
                    pnl=(h1["close"].iloc[i]-entry_price)*lot - CHG
                    all_t.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl})
                    in_trade=False
                # Time stop: exit after 20 bars if not profitable
                if i - idx_entry >= 20:
                    pnl=(h1["close"].iloc[i]-entry_price)*lot - CHG
                    all_t.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl})
                    in_trade=False
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True) if all_t else pd.DataFrame(columns=["exit_time","pnl_rs"])

# ============================================================
# STRATEGY 3: Momentum Breakout
# Buy when close > highest high of last 20 bars (breakout)
# Exit: trailing stop at 10xATR from highest point since entry
# ============================================================

def run_momentum():
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); h1=h1.sort_values("datetime").reset_index(drop=True)
        atr=compute_atr(h1, 14)
        hi20=h1["high"].rolling(20).max().shift(1)
        in_trade=False; entry_price=0; highest=0; entry_time=None
        lot=NLOT if "NIFTY" in sym else SLOT
        for i in range(20, len(h1)):
            if not in_trade:
                if h1["close"].iloc[i] > hi20.iloc[i] and h1["close"].iloc[i] > h1["open"].iloc[i]:
                    # Only take breakouts during market hours
                    if h1["datetime"].iloc[i].time() < CUTOFF_TIME and h1["datetime"].iloc[i].hour >= 9:
                        in_trade=True; entry_price=h1["close"].iloc[i]; highest=entry_price
            else:
                if h1["high"].iloc[i] > highest: highest=h1["high"].iloc[i]
                ca=atr.iloc[i]
                if not pd.isna(ca) and h1["close"].iloc[i] < highest - 10*ca:
                    pnl=(h1["close"].iloc[i]-entry_price)*lot - CHG
                    all_t.append({"exit_time":h1["datetime"].iloc[i],"pnl_rs":pnl})
                    in_trade=False
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True) if all_t else pd.DataFrame(columns=["exit_time","pnl_rs"])

# ============================================================
# COMBINATION
# ============================================================

def combine_strategies(strat_dfs):
    """Merge multiple strategy P&L time series, sum same-day P&L."""
    if not strat_dfs:
        return pd.DataFrame(columns=["exit_time","pnl_rs"])
    comb=pd.concat(strat_dfs, ignore_index=True)
    comb=comb.sort_values("exit_time").reset_index(drop=True)
    return comb

def calc_metrics(df, name, capital=200000):
    if len(df)==0:
        return {"name":name,"trades":0,"net_rs":0,"wr":0,"pf":0,"cagr":0,"sharpe":0,"max_dd":0}
    net=df["pnl_rs"].sum()
    n=len(df)
    wr=(df["pnl_rs"]>0).sum()/n*100
    pf=(df[df["pnl_rs"]>0]["pnl_rs"].sum()/abs(df[df["pnl_rs"]<0]["pnl_rs"].sum())) if (df["pnl_rs"]<0).sum()!=0 else 99
    cum=df["pnl_rs"].cumsum()
    peak=cum.cummax()
    dd=peak-cum
    mdd=dd.max()
    years=(df["exit_time"].max()-df["exit_time"].min()).total_seconds()/31536000 if len(df)>1 else 1
    cagr=((1+net/capital)**(1/years)-1)*100 if years>0 else 0
    sharpe=df["pnl_rs"].mean()/df["pnl_rs"].std()*np.sqrt(252*6.5) if df["pnl_rs"].std()>0 else 0
    return {"name":name,"trades":n,"net_rs":net,"wr":wr,"pf":pf,"cagr":cagr,"sharpe":sharpe,"max_dd":mdd}

def loss_filter(df, skip_n=2):
    df=df.sort_values("exit_time").reset_index(drop=True)
    lc=0; keep=np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if lc>=skip_n: keep[i]=False; lc=0; continue
        if df["pnl_rs"].iloc[i]<=0: lc+=1
        else: lc=0
    return df[keep].reset_index(drop=True)

# ── Correlation between strategies ──

def calc_correlation(df1, df2, name1, name2):
    """Correlation of daily P&L between two strategies."""
    if len(df1)==0 or len(df2)==0: return 0
    df1["date"]=df1["exit_time"].dt.date
    df2["date"]=df2["exit_time"].dt.date
    d1=df1.groupby("date")["pnl_rs"].sum().reset_index().rename(columns={"pnl_rs":"pnl1"})
    d2=df2.groupby("date")["pnl_rs"].sum().reset_index().rename(columns={"pnl_rs":"pnl2"})
    merged=pd.merge(d1, d2, on="date", how="outer").fillna(0)
    if merged["pnl1"].std()==0 or merged["pnl2"].std()==0: return 0
    return merged["pnl1"].corr(merged["pnl2"])

print("="*100)
print("MULTI-STRATEGY OVERLAY TEST")
print("="*100)

# Run all strategies
print("\nRunning Engulfing (CH15 skip=2)...")
eng=run_engulfing()
eng_f=loss_filter(eng)
m_eng=calc_metrics(eng_f, "Engulfing_CH15")
print(f"  {m_eng['trades']:4d} trades, Rs{m_eng['net_rs']:>+9,.0f}, WR={m_eng['wr']:.1f}%, PF={m_eng['pf']:.2f}")

print("\nRunning RSI Mean-Reversion...")
rsi=run_rsi_reversion()
rsi_f=loss_filter(rsi)
m_rsi=calc_metrics(rsi_f, "RSI_Reversion")
print(f"  {m_rsi['trades']:4d} trades, Rs{m_rsi['net_rs']:>+9,.0f}, WR={m_rsi['wr']:.1f}%, PF={m_rsi['pf']:.2f}")

print("\nRunning Momentum Breakout...")
mom=run_momentum()
mom_f=loss_filter(mom)
m_mom=calc_metrics(mom_f, "Momentum")
print(f"  {m_mom['trades']:4d} trades, Rs{m_mom['net_rs']:>+9,.0f}, WR={m_mom['wr']:.1f}%, PF={m_mom['pf']:.2f}")

# Correlations
print(f"\n{'='*60}")
print("CORRELATION (Daily P&L)")
print(f"{'='*60}")
corr_eng_rsi=calc_correlation(eng_f, rsi_f, "Engulfing", "RSI")
corr_eng_mom=calc_correlation(eng_f, mom_f, "Engulfing", "Momentum")
corr_rsi_mom=calc_correlation(rsi_f, mom_f, "RSI", "Momentum")
print(f"  Engulfing vs RSI Reversion:      {corr_eng_rsi:.3f}")
print(f"  Engulfing vs Momentum:           {corr_eng_mom:.3f}")
print(f"  RSI Reversion vs Momentum:       {corr_rsi_mom:.3f}")

# Combined portfolios
print(f"\n{'='*60}")
print("COMBINED PORTFOLIOS")
print(f"{'='*60}")

# Engulfing + RSI
eng_rsi=combine_strategies([eng_f, rsi_f])
m_eng_rsi=calc_metrics(eng_rsi, "Engulfing+RSI")
print(f"  Engulfing + RSI:          {m_eng_rsi['trades']:4d} trades, Rs{m_eng_rsi['net_rs']:>+9,.0f}, WR={m_eng_rsi['wr']:.1f}%, PF={m_eng_rsi['pf']:.2f}")

# Engulfing + Momentum
eng_mom=combine_strategies([eng_f, mom_f])
m_eng_mom=calc_metrics(eng_mom, "Engulfing+Momentum")
print(f"  Engulfing + Momentum:     {m_eng_mom['trades']:4d} trades, Rs{m_eng_mom['net_rs']:>+9,.0f}, WR={m_eng_mom['wr']:.1f}%, PF={m_eng_mom['pf']:.2f}")

# All 3
all3=combine_strategies([eng_f, rsi_f, mom_f])
m_all3=calc_metrics(all3, "All_3")
print(f"  All 3 Combined:           {m_all3['trades']:4d} trades, Rs{m_all3['net_rs']:>+9,.0f}, WR={m_all3['wr']:.1f}%, PF={m_all3['pf']:.2f}")

# Also compute: combined with equal weight rebalancing
# Engulfing + 50% RSI (scale RSI to have similar risk)
eng_rsi_scale=combine_strategies([eng_f, rsi_f])  # Same weight (both 1 lot)
print(f"\n{'='*60}")
print(f"DIVERSIFICATION BENEFIT")
print(f"{'='*60}")
comb_nets=[m_eng["net_rs"], m_rsi["net_rs"], m_mom["net_rs"], m_eng_rsi["net_rs"], m_eng_mom["net_rs"], m_all3["net_rs"]]
print(f"  Best single strategy:      Engulfing at Rs{m_eng['net_rs']:+,.0f}")
print(f"  Best combined:             {m_all3['name']} at Rs{m_all3['net_rs']:+,.0f}")
print(f"  Improvement over single:   {((m_all3['net_rs']/m_eng['net_rs'])-1)*100 if m_eng['net_rs']!=0 else 0:+.1f}%")

out_dir=os.path.join(BASE,"backtest_results","multistrategy")
os.makedirs(out_dir,exist_ok=True)
all_metrics=[m_eng, m_rsi, m_mom, m_eng_rsi, m_eng_mom, m_all3]
pd.DataFrame(all_metrics).to_csv(os.path.join(out_dir,"multistrategy_results.csv"),index=False)
print(f"\nSaved to: {os.path.join(out_dir,'multistrategy_results.csv')}")

# Save individual trade lists
eng_f.to_csv(os.path.join(out_dir,"engulfing_trades.csv"),index=False)
rsi_f.to_csv(os.path.join(out_dir,"rsi_trades.csv"),index=False)
mom_f.to_csv(os.path.join(out_dir,"momentum_trades.csv"),index=False)
all3.to_csv(os.path.join(out_dir,"combined_all3_trades.csv"),index=False)
print(f"Trade files saved.")
