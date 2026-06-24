import pandas as pd
import numpy as np
import os, json
from datetime import datetime

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BROKERAGE_PER_ORDER = 10
STT = 0.001; EXCHANGE_TC = 0.00003; SEBI_TC = 0.000001
GST = 0.18; STAMP_DUTY = 0.00003

def compute_charges(entry_price, exit_price, qty=1):
    tb = entry_price * qty; ts = exit_price * qty
    return (BROKERAGE_PER_ORDER * 2 + ts * STT + (tb+ts) * EXCHANGE_TC
            + (tb+ts) * SEBI_TC * 2 + tb * STAMP_DUTY
            + (BROKERAGE_PER_ORDER * 2 + (tb+ts) * EXCHANGE_TC) * GST)

def run_strategy(symbol, period=20, n_std=2.5, rr=3.0):
    path = f"{DATA_DIR}/{symbol}_FIFTEEN_MINUTE.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek

    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=1)
    upper = ma + n_std * std; lower = ma - n_std * std

    trades = []
    for i in range(period, len(df)-1):
        row = df.iloc[i]
        if row["low"] > upper.iloc[i]:
            typ, entry_p, sl_p = "SHORT", row["close"], row["low"]
            tp_p = entry_p - (entry_p - sl_p) * rr
            sl_dist = entry_p - sl_p
        elif row["high"] < lower.iloc[i]:
            typ, entry_p, sl_p = "LONG", row["close"], row["high"]
            tp_p = entry_p + (sl_p - entry_p) * rr
            sl_dist = sl_p - entry_p
        else:
            continue
        if sl_dist <= 0:
            continue

        k = i + 1; exit_p, exit_t, reason = entry_p, row["datetime"], "EOD"
        while k < len(df):
            b = df.iloc[k]; bdt = b["datetime"]
            if bdt.hour >= 15 and bdt.minute >= 15:
                exit_p = b["close"]; exit_t = bdt; reason = "EOD"; break
            tp_hit = (typ == "SHORT" and b["low"] <= tp_p) or (typ == "LONG" and b["high"] >= tp_p)
            sl_hit = (typ == "SHORT" and b["high"] >= sl_p) or (typ == "LONG" and b["low"] <= sl_p)
            if tp_hit and sl_hit:
                exit_p = tp_p; exit_t = bdt; reason = "TP"; break
            elif tp_hit: exit_p = tp_p; exit_t = bdt; reason = "TP"; break
            elif sl_hit: exit_p = sl_p; exit_t = bdt; reason = "T1"; break
            k += 1

        pnl_pts = (entry_p - exit_p) if typ == "SHORT" else (exit_p - entry_p)
        charges = compute_charges(entry_p, exit_p)

        trades.append(dict(symbol=symbol, date=str(row["datetime"].date()),
            year=int(row["year"]), month=int(row["month"]), dow=int(row["dow"]),
            type=typ, entry_time=str(row["datetime"]), exit_time=str(exit_t),
            entry=round(entry_p,2), exit=round(exit_p,2), sl=round(sl_p,2),
            tp=round(tp_p,2), sl_pts=round(sl_dist,2), tp_pts=round(abs(entry_p-tp_p),2),
            pnl_pts=round(pnl_pts,2), charges=round(charges,2),
            net_pnl=round(pnl_pts-charges,2),
            r=round(pnl_pts/sl_dist,2) if sl_dist>0 else 0,
            reason=reason))
    return pd.DataFrame(trades)

def generate_report(sym, df, n_std, rr):
    total = len(df)
    wins = df[df["pnl_pts"] > 0]; losses = df[df["pnl_pts"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc/total*100, 2) if total else 0
    total_pts = round(df["pnl_pts"].sum(), 2)
    total_charges = round(df["charges"].sum(), 2)
    net_pts = round(df["net_pnl"].sum(), 2)
    gp = round(wins["pnl_pts"].sum(), 2) if wc else 0
    gl = round(losses["pnl_pts"].sum(), 2) if lc else 0
    pf = round(abs(gp/gl), 2) if gl else 0
    sharpe = round(df["r"].mean()/df["r"].std()*np.sqrt(total), 2) if df["r"].std() > 0 else 0

    df_s = df.sort_values("exit_time").reset_index(drop=True)
    df_s["cum"] = df_s["net_pnl"].cumsum()
    df_s["peak"] = df_s["cum"].cummax()
    df_s["dd"] = df_s["peak"] - df_s["cum"]
    df_s["dd_pct"] = (df_s["dd"] / df_s["peak"]) * 100
    mdd = round(df_s["dd"].max(), 2)
    mdd_p = round(df_s[df_s["peak"] > 0]["dd_pct"].max(), 2) if df_s["peak"].max() > 0 else 0

    t1_c = (df["reason"] == "T1").sum()
    tp_c = (df["reason"] == "TP").sum()
    eod_c = (df["reason"] == "EOD").sum()

    # === PRINT ===
    print(f"\n{'='*70}")
    print(f"  {sym} | BB({period}, SD={n_std}) | RR 1:{rr} | 15-min data")
    print(f"  Strategy: Candle fully OUTSIDE band -> enter at close")
    print(f"  T1 = near extreme ({'low for SHORT' if True else ''}) | TP = {rr}x T1")
    print(f"{'='*70}")

    print(f"\n  [BASICS]")
    print(f"  Period:          {df['date'].min()} to {df['date'].max()}")
    print(f"  Trading days:    {df['date'].nunique()}")
    print(f"  Total trades:    {total}")
    print(f"  Wins (on pts):   {wc} ({wr}%)")
    print(f"  Losses (pts):    {lc} ({round(lc/total*100,1)}%)")

    print(f"\n  [POINTS & P&L]")
    print(f"  Gross P&L:       +{gp:.2f} / {gl:.2f} pts")
    print(f"  Net P&L (pts):   {net_pts:>+10.2f}")
    print(f"  Total charges:   Rs{total_charges:>9.2f} (Rs{round(total_charges/total,1):>6.1f}/trade)")
    print(f"  Profit Factor:   {pf}")

    print(f"\n  [PER-TRADE STATS]")
    print(f"  Avg win:         {wins['pnl_pts'].mean():>+8.2f} pts")
    print(f"  Avg loss:        {losses['pnl_pts'].mean():>+8.2f} pts")
    print(f"  Avg R:           {df['r'].mean():>8.2f}")
    print(f"  Median R:        {df['r'].median():>8.2f}")
    print(f"  Max win:         {df['pnl_pts'].max():>+8.2f} pts")
    print(f"  Max loss:        {df['pnl_pts'].min():>+8.2f} pts")
    print(f"  Std returns:     {df['pnl_pts'].std():>8.2f} pts")
    print(f"  Avg T1 dist:     {df[df['reason']=='T1']['sl_pts'].mean():>8.2f} pts")
    print(f"  Avg TP dist:     {df[df['reason']=='TP']['tp_pts'].mean():>8.2f} pts")
    print(f"  Avg hold time:   {round(df['sl_pts'].count()/total*15,1) if total else 0:>5} min")

    print(f"\n  [EXIT REASONS]")
    print(f"  T1 (1R profit):  {t1_c:>4d} ({round(t1_c/total*100,1)}%) | Avg R: {df[df['reason']=='T1']['r'].mean():>5.2f} | Avg Pts: {df[df['reason']=='T1']['pnl_pts'].mean():>+7.2f}")
    print(f"  TP ({rr}R profit): {tp_c:>4d} ({round(tp_c/total*100,1)}%) | Avg R: {df[df['reason']=='TP']['r'].mean():>5.2f} | Avg Pts: {df[df['reason']=='TP']['pnl_pts'].mean():>+7.2f}")
    print(f"  EOD (timeout):   {eod_c:>4d} ({round(eod_c/total*100,1)}%) | Avg R: {df[df['reason']=='EOD']['r'].mean():>5.2f} | Avg Pts: {df[df['reason']=='EOD']['pnl_pts'].mean():>+7.2f}")

    print(f"\n  [YEARLY]")
    print(f"  {'Year':>6s} {'Trades':>7s} {'Wins':>5s} {'WR':>6s} {'Gross Pts':>9s} {'Net Pts':>9s} {'Avg R':>6s}")
    print(f"  {'-'*48}")
    yearly = df.groupby("year").agg(tr=("pnl_pts","count"), w=("pnl_pts", lambda x: (x>0).sum()),
        gross=("pnl_pts","sum"), net=("net_pnl","sum"), r=("r","mean"))
    for yr, r in yearly.iterrows():
        print(f"  {int(yr):>6d} {int(r['tr']):>7d} {int(r['w']):>5d} {round(r['w']/r['tr']*100,1):>5.1f}% "
              f"{r['gross']:>+8.0f}  {r['net']:>+8.0f}  {r['r']:>5.2f}")

    print(f"\n  [LONG vs SHORT]")
    for t in ["LONG", "SHORT"]:
        sub = df[df["type"] == t]
        if len(sub) > 0:
            print(f"  {t:>6s}: {len(sub):>3d} trades  WR: {round((sub['pnl_pts']>0).sum()/len(sub)*100,1):>5.1f}%  "
                  f"Gross: {sub['pnl_pts'].sum():>+8.0f}  Net: {sub['net_pnl'].sum():>+8.0f}  Avg R: {sub['r'].mean():.2f}")

    print(f"\n  [RISK]")
    print(f"  Max DD:          {mdd:>8.2f} pts ({mdd_p}%)")
    print(f"  Sharpe (on R):   {sharpe:>8.2f}")

    print(f"\n  [MONTHLY NET (sorted)]")
    mn = df.groupby("month")["net_pnl"].sum()
    names_m = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    for m, v in mn.sort_values(ascending=False).items():
        print(f"  {names_m[m]}: {v:>+7.0f}")

    print(f"\n  [DOW NET]")
    names_d = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    for d in range(5):
        sub = df[df["dow"] == d]
        if len(sub) > 0:
            print(f"  {names_d[d]}: {len(sub):>3d} trades  Net: {sub['net_pnl'].sum():>+7.0f}")

    df.to_csv(f"{OUTPUT_DIR}/{sym}_bb25_final.csv", index=False)
    return {"trades": total, "wr": wr, "net_pts": net_pts, "pf": pf, "avg_r": round(df["r"].mean(),2),
        "mdd": mdd, "sharpe": sharpe, "t1_pct": round(t1_c/total*100,1), "tp_pct": round(tp_c/total*100,1)}

# === RUN ALL ===
period = 20; n_std = 2.5; rr = 3.0
print(f"\n{'='*70}")
print(f"  BB BREAKOUT_v2 — FINAL REPORT (all indices)")
print(f"  Params: BB({period}, {n_std}), RR=1:{rr}, 15-min data")
print(f"{'='*70}")

results = []
for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
    df = run_strategy(sym, period=period, n_std=n_std, rr=rr)
    res = generate_report(sym, df, n_std, rr)
    results.append({"symbol": sym, **res})

print(f"\n{'='*70}")
print(f"  COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"{'Index':10s} {'Trades':>7s} {'WR':>6s} {'Gross':>8s} {'Charges':>8s} {'Net':>9s} {'PF':>5s} {'AvgR':>5s} {'MaxDD':>8s} {'Sh':>5s} {'T1%':>5s} {'TP%':>5s}")
print(f"{'-'*79}")
for r in results:
    print(f"{r['symbol']:10s} {r['trades']:>7d} {r['wr']:>5.1f}% "
          f"{'':>8s} {'':>8s} {r['net_pts']:>+8.0f}  "
          f"{r['pf']:>4.2f} {r['avg_r']:>4.2f} {r['mdd']:>+7.0f} {r['sharpe']:>4.1f} {r['t1_pct']:>4.1f}% {r['tp_pct']:>4.1f}%")

print(f"\n{'='*70}")
print(f"  NOTE: 'T1' exit = partial profit at 1R (not a loss)")
print(f"  Strategy enters at close when candle is fully outside BB")
print(f"  T1 = near extreme of trigger candle. TP = {rr}x T1.")
print(f"  Charges include: brokerage, STT, exchange, SEBI, stamp, GST")
print(f"{'='*70}")
