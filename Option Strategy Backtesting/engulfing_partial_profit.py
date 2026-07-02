"""
Partial Profit Booking Test
Book 25%/33%/50% at 1:1 RR, let rest ride on CH15 trail.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 0.50
SKIP_N = 2

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

def execute_trades_pp(signals, m5, book_pct, target_rr=1.0):
    """
    book_pct: % of position to book at 1:1 RR (0.25, 0.33, 0.50)
    target_rr: risk-reward for partial booking (default 1.0 = 1:1)
    """
    tc = m5["datetime"].dt.time
    atr5 = compute_atr(m5, 14)
    dt_unix = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi = m5["high"].values; lo = m5["low"].values; cl = m5["close"].values
    trades = []
    for sig in signals:
        t_unix = int(pd.to_datetime(sig["trigger_time"]).timestamp())
        lv = sig["level"]
        idx = np.searchsorted(dt_unix, t_unix, side="right")
        if idx >= len(m5): continue
        broke = idx
        while broke < len(m5) and cl[broke] <= lv: broke += 1
        if broke >= len(m5): continue
        retest = broke + 1
        while retest < len(m5):
            if lo[retest] < lv and cl[retest] > lv and tc.iloc[retest] < CUTOFF_TIME: break
            retest += 1
        if retest >= len(m5): continue
        entry = cl[retest]; sl = lo[retest]
        if entry - sl <= 0: continue
        if m5["datetime"].iloc[retest].hour == 9: continue

        risk = entry - sl
        target = entry + risk * target_rr
        highest = entry
        booked = False; booked_pnl = 0.0

        for j in range(retest+1, len(m5)):
            ca = atr5.iloc[j]
            if pd.isna(ca): continue
            if hi[j] > highest: highest = hi[j]
            # Check if target hit (partial book)
            if not booked and hi[j] >= target:
                booked = True
                booked_pnl = book_pct * risk  # points contribution from booked portion
                # Adjust entry for remaining portion to track P&L correctly
                remaining_entry = entry  # same entry, just different P&L attribution
            # Chandelier exit
            if cl[j] < highest - 15 * ca:
                rem_pnl = (1 - book_pct) * (cl[j] - entry)
                total_pts = booked_pnl + rem_pnl
                trades.append({
                    "points": total_pts,
                    "exit_time": m5["datetime"].iloc[j],
                    "hold_hours": (m5["datetime"].iloc[j] - m5["datetime"].iloc[retest]).total_seconds()/3600,
                    "booked": booked
                })
                break
    return pd.DataFrame(trades)

def portfolio_loss_filter(df, skip_n=SKIP_N):
    df = df.sort_values("exit_time").reset_index(drop=True)
    lc=0; keep=np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if lc >= skip_n: keep[i]=False; lc=0; continue
        if df["points"].iloc[i]<=0: lc+=1
        else: lc=0
    return df[keep].reset_index(drop=True)

def run_test(book_pct, label):
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
        m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
        h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
        sigs=detect_signals(h1)
        tr=execute_trades_pp(sigs, m5, book_pct)
        tr["sym"]=sym
        lot=NLOT if "NIFTY" in sym else SLOT
        tr["pnl_rs"]=tr["points"]*lot - CHG
        all_t.append(tr)
    comb=pd.concat(all_t,ignore_index=True)
    comb_unf=comb.copy()
    comb=portfolio_loss_filter(comb)
    net=comb["pnl_rs"].sum() if len(comb)>0 else 0
    n=len(comb)
    wr=(comb["pnl_rs"]>0).sum()/n*100 if n>0 else 0
    pf=(comb[comb["pnl_rs"]>0]["pnl_rs"].sum()/abs(comb[comb["pnl_rs"]<0]["pnl_rs"].sum())) if (comb["pnl_rs"]<0).sum()!=0 else 99
    ah=comb["hold_hours"].mean() if n>0 else 0
    booked_pct_val = (comb["booked"].sum()/n*100) if n>0 else 0
    return {"name": label, "trades": n, "net_rs": net, "wr": wr, "pf": pf, "avg_h": ah, "booked_pct": booked_pct_val}

print("="*100)
print("PARTIAL PROFIT BOOKING TEST")
print("Book X% at 1:1 RR, rest rides on CH15 trail")
print("="*100)

variants = [
    (0.00, "ch15_0pct_baseline"),
    (0.25, "ch15_25pct_book"),
    (0.33, "ch15_33pct_book"),
    (0.50, "ch15_50pct_book"),
    (0.40, "ch15_40pct_book"),
]
for b, l in variants[:1]:
    print(f"\nTesting {l}...")
base = run_test(0.00, "ch15_0pct_baseline")
print(f"{'Variant':25s}  Trades  Net_RS       WR%   PF    AvgH  Booked%")
print("-"*75)
print(f"  {base['name']:25s}  {base['trades']:4d}   Rs{base['net_rs']:>+9,.0f}  {base['wr']:5.1f}% {base['pf']:5.2f}  {base['avg_h']:5.1f}h  -")

results=[base]
for book_pct, label in variants[1:]:
    r=run_test(book_pct, label)
    results.append(r)
    vs = (r["net_rs"]-base["net_rs"])/abs(base["net_rs"])*100 if base["net_rs"]!=0 else 0
    print(f"  {label:25s}  {r['trades']:4d}   Rs{r['net_rs']:>+9,.0f}  {r['wr']:5.1f}% {r['pf']:5.2f}  {r['avg_h']:5.1f}h  {r['booked_pct']:5.1f}%  ({vs:+.1f}%)")

results.sort(key=lambda x: x["net_rs"], reverse=True)
print(f"\nRANKED:")
for i,r in enumerate(results):
    vs=(r["net_rs"]-base["net_rs"])/abs(base["net_rs"])*100
    m=" <<<" if i==0 else ""
    print(f"  {i+1}. {r['name']:25s}  Rs{r['net_rs']:>+9,.0f}  {r['wr']:5.1f}%  PF={r['pf']:5.2f}  {vs:+7.1f}%{m}")

out_dir=os.path.join(BASE,"backtest_results","partial_profit")
os.makedirs(out_dir,exist_ok=True)
pd.DataFrame(results).to_csv(os.path.join(out_dir,"partial_profit_results.csv"),index=False)
print(f"\nSaved to: {os.path.join(out_dir,'partial_profit_results.csv')}")
