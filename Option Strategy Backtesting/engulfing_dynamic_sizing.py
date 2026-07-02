"""
Dynamic Position Sizing Test (Anti-Martingale)
Scale up after wins, scale down in drawdown periods.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 0.50
CHG = 20

# Lot definitions
NLOT_BASE = 50; SLOT_BASE = 10

def compute_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(), (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def detect_signals(h1):
    body = (h1["close"]-h1["open"]).abs()
    is_red = h1["close"]<h1["open"]; is_green = h1["close"]>h1["open"]
    sigs = []
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
        if body.iloc[i] < body.iloc[i-1]*MIN_BODY_PCT: continue
        sigs.append({"trigger_time": h1["datetime"].iloc[i], "level": h1["high"].iloc[i]})
    return sigs

def get_trades(m5, signals):
    """Execute trades with CH15, return list of trade dicts with entry/exit info."""
    tc = m5["datetime"].dt.time
    atr5 = compute_atr(m5, 14)
    dt_unix = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    trades=[]
    for sig in signals:
        t_unix=int(pd.to_datetime(sig["trigger_time"]).timestamp())
        lv=sig["level"]
        idx=np.searchsorted(dt_unix, t_unix, side="right")
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
        if entry-sl<=0: continue
        if m5["datetime"].iloc[retest].hour==9: continue
        highest=entry
        for j in range(retest+1,len(m5)):
            ca=atr5.iloc[j]
            if pd.isna(ca): continue
            if hi[j]>highest: highest=hi[j]
            if cl[j]<highest-15*ca:
                trades.append({
                    "points": cl[j]-entry,
                    "exit_time": m5["datetime"].iloc[j],
                    "hold_hours": (m5["datetime"].iloc[j]-m5["datetime"].iloc[retest]).total_seconds()/3600,
                })
                break
    return pd.DataFrame(trades)

# ── Sizing Strategies ──

def fixed_sizing(trades_df, sym):
    """Fixed 1 lot baseline."""
    lot = NLOT_BASE if "NIFTY" in sym else SLOT_BASE
    trades_df["lot"] = 1
    trades_df["pnl_rs"] = trades_df["points"] * lot - CHG
    return trades_df

def anti_martingale_streak(trades_df, sym, win_streak_add=3, loss_streak_sub=2, max_lots=3, min_lots=1):
    """
    Anti-Martingale based on win/loss streak:
    - After `win_streak_add` consecutive wins: add 1 lot
    - After `loss_streak_sub` consecutive losses: reduce 1 lot
    - Never go below min_lots or above max_lots
    """
    lot = NLOT_BASE if "NIFTY" in sym else SLOT_BASE
    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
    lots = []; sizes = []
    cur_lots = 1; win_streak = 0; loss_streak = 0
    for i in range(len(trades_df)):
        lots.append(cur_lots)
        sizes.append(cur_lots * lot)
        pnl = trades_df["points"].iloc[i] * lot * cur_lots - CHG * cur_lots
        trades_df.at[i, "pnl_rs"] = pnl
        trades_df.at[i, "lot"] = cur_lots
        if pnl > 0:
            win_streak += 1; loss_streak = 0
            if win_streak >= win_streak_add and cur_lots < max_lots:
                cur_lots += 1; win_streak = 0
        else:
            loss_streak += 1; win_streak = 0
            if loss_streak >= loss_streak_sub and cur_lots > min_lots:
                cur_lots -= 1; loss_streak = 0
    return trades_df

def anti_martingale_equity(trades_df, sym, step=200000, max_lots=3, min_lots=1):
    """
    Scale lot size based on equity growth:
    - Every Rs200K profit: add 1 lot
    - Every Rs200K loss from peak: reduce 1 lot
    """
    lot = NLOT_BASE if "NIFTY" in sym else SLOT_BASE
    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
    cur_lots = 1
    base_capital = 200000
    for i in range(len(trades_df)):
        pnl = trades_df["points"].iloc[i] * lot * cur_lots - CHG * cur_lots
        trades_df.at[i, "pnl_rs"] = pnl
        trades_df.at[i, "lot"] = cur_lots
        # Update equity
        if i == 0:
            equity = base_capital + pnl
            peak = equity
        else:
            equity = trades_df["pnl_rs"].iloc[:i+1].sum() + base_capital
            peak = max(peak, equity)
        dd_from_peak = peak - equity
        # Scale up based on profit
        target_lots = max(min_lots, min(max_lots, 1 + int(max(0, equity - base_capital) / step)))
        # Scale down based on drawdown
        if dd_from_peak > step:
            target_lots = max(min_lots, target_lots - int(dd_from_peak / step))
        cur_lots = target_lots
    return trades_df

def kelly_sizing(trades_df, sym, kelly_pct=0.25, max_lots=3, min_lots=1):
    """
    Kelly Criterion based sizing:
    f* = (win_rate * avg_win / avg_loss - (1 - win_rate)) / (avg_win / avg_loss)
    Use fractional Kelly (25%) for safety.
    """
    lot = NLOT_BASE if "NIFTY" in sym else SLOT_BASE
    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
    cur_lots = 1
    for i in range(len(trades_df)):
        pnl = trades_df["points"].iloc[i] * lot * cur_lots - CHG * cur_lots
        trades_df.at[i, "pnl_rs"] = pnl
        trades_df.at[i, "lot"] = cur_lots
        # Recalculate Kelly after each trade
        hist = trades_df.iloc[:i+1]
        if i >= 10:
            wins = hist[hist["pnl_rs"] > 0]
            losses = hist[hist["pnl_rs"] < 0]
            wr = len(wins) / len(hist) if len(hist) > 0 else 0.5
            avg_w = wins["pnl_rs"].mean() if len(wins) > 0 else 1
            avg_l = abs(losses["pnl_rs"].mean()) if len(losses) > 0 else 1
            if avg_l > 0 and avg_w > 0:
                r = avg_w / avg_l
                kelly = (wr * r - (1 - wr)) / r
                kelly = max(0, min(1, kelly))
                target = max(min_lots, min(max_lots, int(kelly * kelly_pct * max_lots)))
                cur_lots = target
    return trades_df

# ── Test Framework ──

def run_test(sizing_fn, label):
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
        h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
        sigs=detect_signals(h1)
        tr=get_trades(m5, sigs)
        tr["sym"]=sym
        tr=sizing_fn(tr, sym)
        all_t.append(tr)
    comb=pd.concat(all_t,ignore_index=True)
    comb=comb.sort_values("exit_time").reset_index(drop=True)
    # Apply portfolio loss filter
    lc=0; keep=np.ones(len(comb), dtype=bool)
    for i in range(len(comb)):
        if lc>=2: keep[i]=False; lc=0; continue
        if comb["pnl_rs"].iloc[i]<=0: lc+=1
        else: lc=0
    comb=comb[keep].reset_index(drop=True)
    net=comb["pnl_rs"].sum() if len(comb)>0 else 0
    n=len(comb)
    wr=(comb["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    pf=(comb[comb["pnl_rs"]>0]["pnl_rs"].sum()/abs(comb[comb["pnl_rs"]<0]["pnl_rs"].sum())) if (comb["pnl_rs"]<0).sum()!=0 else 99
    peak=(comb["pnl_rs"].cumsum()+200000).cummax()
    dd=peak-(comb["pnl_rs"].cumsum()+200000)
    mdd=dd.max()
    avg_lot=comb["lot"].mean() if "lot" in comb.columns else 1
    return {"name":label,"trades":n,"net_rs":net,"wr":wr,"pf":pf,"max_dd":mdd,"avg_lot":avg_lot}

print("="*110)
print("DYNAMIC POSITION SIZING TEST")
print("="*110)
print(f"{'Sizing':30s}  Trades  Net_RS        WR%   PF    MaxDD      AvgLot")
print("-"*85)

tests = [
    (lambda df,sym: fixed_sizing(df,sym), "fixed_1lot_baseline"),
    (lambda df,sym: anti_martingale_streak(df,sym,3,2,3,1), "am_streak_3w2l"),
    (lambda df,sym: anti_martingale_streak(df,sym,2,1,3,1), "am_streak_2w1l"),
    (lambda df,sym: anti_martingale_streak(df,sym,4,3,3,1), "am_streak_4w3l"),
    (lambda df,sym: anti_martingale_streak(df,sym,3,2,5,1), "am_streak_5max"),
    (lambda df,sym: anti_martingale_equity(df,sym,200000,3,1), "am_equity_2L"),
    (lambda df,sym: anti_martingale_equity(df,sym,100000,3,1), "am_equity_1L"),
    (lambda df,sym: anti_martingale_equity(df,sym,300000,3,1), "am_equity_3L"),
    (lambda df,sym: kelly_sizing(df,sym,0.25,3,1), "kelly_25pct"),
    (lambda df,sym: kelly_sizing(df,sym,0.50,3,1), "kelly_50pct"),
]

results=[]
for fn,name in tests:
    try:
        r=run_test(fn,name)
        results.append(r)
        vs=(r["net_rs"]-results[0]["net_rs"])/abs(results[0]["net_rs"])*100 if len(results)>1 and results[0]["net_rs"]!=0 else 0
        m=" (baseline)" if name=="fixed_1lot_baseline" else ""
        print(f"  {name:30s}  {r['trades']:4d}   Rs{r['net_rs']:>+9,.0f}  {r['wr']:5.1f}% {r['pf']:5.2f}  Rs{r['max_dd']:>7,.0f}  {r['avg_lot']:5.2f}{m}")
    except Exception as e:
        print(f"  {name:30s}  ERROR: {e}")

results.sort(key=lambda x: x["net_rs"], reverse=True)
print(f"\nRANKED:")
for i,r in enumerate(results):
    vs=(r["net_rs"]-results[-1]["net_rs"])/abs(results[-1]["net_rs"])*100 if results[-1]["net_rs"]!=0 else 0
    m=" <<<" if i==0 else ""
    print(f"  {i+1}. {r['name']:30s}  Rs{r['net_rs']:>+9,.0f}  WR={r['wr']:5.1f}%  PF={r['pf']:5.2f}  DD=Rs{r['max_dd']:,.0f}  Lot={r['avg_lot']:.2f}{m}")

out_dir=os.path.join(BASE,"backtest_results","dynamic_sizing")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(results).to_csv(os.path.join(out_dir,"sizing_results.csv"),index=False)
print(f"\nSaved to: {os.path.join(out_dir,'sizing_results.csv')}")
