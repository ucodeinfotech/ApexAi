import pandas as pd
import numpy as np
import os

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

def run(symbol, period=20, n_std=2.5, rr=3.0):
    path = f"{DATA_DIR}/{symbol}_FIFTEEN_MINUTE.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek

    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + n_std * std; lower = ma - n_std * std

    trades = []
    for i in range(period, len(df)-1):
        row = df.iloc[i]
        if row["low"] > upper.iloc[i]:
            typ, entry_p, sl_p, tp_p = "SHORT", row["close"], row["low"], row["close"] - (row["close"] - row["low"]) * rr
        elif row["high"] < lower.iloc[i]:
            typ, entry_p, sl_p, tp_p = "LONG", row["close"], row["high"], row["close"] + (row["high"] - row["close"]) * rr
        else:
            continue

        sl_dist = abs(entry_p - sl_p)
        if sl_dist <= 0: continue

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
            elif sl_hit: exit_p = sl_p; exit_t = bdt; reason = "SL"; break
            k += 1

        pnl_pts = (entry_p - exit_p) if typ == "SHORT" else (exit_p - entry_p)
        charges = compute_charges(entry_p, exit_p)

        trades.append(dict(symbol=symbol, date=str(row["datetime"].date()), year=int(row["year"]),
            month=int(row["month"]), dow=int(row["dow"]), type=typ,
            entry=round(entry_p,2), exit=round(exit_p,2), sl=round(sl_p,2), tp=round(tp_p,2),
            pnl_pts=round(pnl_pts,2), charges=round(charges,2), net_pnl=round(pnl_pts-charges,2),
            reason=reason, r=round(pnl_pts/sl_dist,2) if sl_dist > 0 else 0))

    return pd.DataFrame(trades)

def report(sym, df, n_std=2.5, rr=3.0):
    total = len(df)
    wins = df[df["pnl_pts"] > 0]; losses = df[df["pnl_pts"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc/total*100, 2) if total else 0
    total_pts = round(df["pnl_pts"].sum(), 2)
    charges_total = round(df["charges"].sum(), 2)
    net_pts = round(df["net_pnl"].sum(), 2)
    gp = round(wins["pnl_pts"].sum(), 2) if wc else 0
    gl = round(losses["pnl_pts"].sum(), 2) if lc else 0
    pf = round(abs(gp/gl), 2) if gl else 0
    sharpe = round(df["r"].mean()/df["r"].std()*np.sqrt(total), 2) if df["r"].std() > 0 else 0

    df_s = df.sort_values("date").reset_index(drop=True)
    df_s["cum"] = df_s["net_pnl"].cumsum()
    df_s["peak"] = df_s["cum"].cummax()
    df_s["dd"] = df_s["peak"] - df_s["cum"]
    mdd = round(df_s["dd"].max(), 2)
    mdd_p = round(mdd / df_s["peak"].max() * 100, 2) if df_s["peak"].max() > 0 else 0

    tp_c = (df["reason"] == "TP").sum()
    sl_c = (df["reason"] == "SL").sum()
    eod_c = (df["reason"] == "EOD").sum()

    print(f"\n{'='*65}")
    print(f"  {sym} | BB(20, {n_std}) breakout_v2 | RR 1:{rr}")
    print(f"{'='*65}")
    print(f"  Period:       {df['date'].min()} to {df['date'].max()} ({df['date'].nunique()} days)")
    print(f"  Trades:       {total}  |  Wins: {wc} ({wr}%)  |  Losses: {lc}")
    print(f"  Gross Pts:    +{gp} / {gl}")
    print(f"  Net Pts:      {net_pts:>+10.2f}")
    print(f"  Charges:      Rs{charges_total:>8.2f}  |  Avg/trade: Rs{round(charges_total/total,1) if total else 0}")
    print(f"  Profit Fac:   {pf}")

    print(f"\n  -- POINT ANALYSIS --")
    print(f"  Avg win:      {wins['pnl_pts'].mean():>+8.2f} pts")
    print(f"  Avg loss:     {losses['pnl_pts'].mean():>+8.2f} pts")
    print(f"  Avg R:        {df['r'].mean():>8.2f}")
    print(f"  Med R:        {df['r'].median():>8.2f}")
    print(f"  Max win:      {df['pnl_pts'].max():>+8.2f} pts")
    print(f"  Max loss:     {df['pnl_pts'].min():>+8.2f} pts")
    print(f"  Std returns:  {df['pnl_pts'].std():>8.2f} pts")
    sl_sub = df[df["reason"]=="SL"]
    if len(sl_sub) > 0:
        sl_dist = sl_sub["sl"].sub(sl_sub["entry"]).abs().mean()
        print(f"  Avg SL dist:  {sl_dist:>8.2f} pts")

    print(f"\n\n  -- EXIT REASONS --")
    print(f"  TP (full):    {tp_c:>4d} ({round(tp_c/total*100,1)}%)  | Avg R: {df[df['reason']=='TP']['r'].mean():>5.2f}  | Avg Pts: {df[df['reason']=='TP']['pnl_pts'].mean():>+7.2f}")
    print(f"  SL (partial): {sl_c:>4d} ({round(sl_c/total*100,1)}%)  | Avg R: {df[df['reason']=='SL']['r'].mean():>5.2f}  | Avg Pts: {df[df['reason']=='SL']['pnl_pts'].mean():>+7.2f}")
    print(f"  EOD:          {eod_c:>4d} ({round(eod_c/total*100,1)}%)  | Avg R: {df[df['reason']=='EOD']['r'].mean():>5.2f}  | Avg Pts: {df[df['reason']=='EOD']['pnl_pts'].mean():>+7.2f}")

    print(f"\n  -- YEARLY --")
    yearly = df.groupby("year").agg(tr=("pnl_pts","count"), w=("pnl_pts", lambda x: (x>0).sum()),
        net=("pnl_pts","sum"), net_r=("net_pnl","sum"), r=("r","mean"))
    for yr, r in yearly.iterrows():
        print(f"  {int(yr)}: {int(r['tr']):>3d} trades  W:{int(r['w']):>3d}  WR:{round(r['w']/r['tr']*100,1):>5}%  "
              f"Gross:{r['net']:>+7.0f}  Net:{r['net_r']:>+7.0f}  AvgR:{r['r']:.2f}")

    print(f"\n  -- LONG vs SHORT --")
    for t in ["LONG", "SHORT"]:
        sub = df[df["type"] == t]
        if len(sub) > 0:
            print(f"  {t}: {len(sub):>3d} trades  WR: {round((sub['pnl_pts']>0).sum()/len(sub)*100,1)}%  "
                  f"Gross: {sub['pnl_pts'].sum():>+7.0f}  Net: {sub['net_pnl'].sum():>+7.0f}  AvgR: {sub['r'].mean():.2f}")

    print(f"\n  -- DRAWDOWN --")
    print(f"  Max DD:       {mdd:>8.2f} pts ({mdd_p}%)")
    print(f"  Sharpe:       {sharpe:>8.2f}")

    # Monthly heat
    print(f"\n  -- MONTHLY NET PTS --")
    mn = df.groupby("month")["net_pnl"].sum()
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    best_m = mn.idxmax(); worst_m = mn.idxmin()
    for m in range(1,13):
        if m in mn.index:
            star = " << BEST" if m == best_m else (" << WORST" if m == worst_m else "")
            print(f"  {names[m]}: {mn[m]:>+7.0f}{star}")

    df.to_csv(f"{OUTPUT_DIR}/{sym}_bb25_report.csv", index=False)
    return {"trades": total, "wr": wr, "net_pts": net_pts, "pf": pf, "avg_r": round(df["r"].mean(),2),
        "mdd": mdd, "sharpe": sharpe, "tp_pct": round(tp_c/total*100,1), "sl_pct": round(sl_c/total*100,1)}

# Run for all
print(f"{'='*65}")
print(f"  BB BREAKOUT_V2 COMPREHENSIVE REPORT (SD=2.5, RR=3.0)")
print(f"{'='*65}")
results = []
for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
    df = run(sym)
    res = report(sym, df, n_std=2.5, rr=3.0)
    results.append({"symbol": sym, **res})

print(f"\n{'='*65}")
print(f"  COMPARISON")
print(f"{'='*65}")
print(f"{'Index':10s} {'Trades':>7s} {'WR':>6s} {'Net Pts':>9s} {'PF':>5s} {'AvgR':>5s} {'MaxDD':>8s} {'Sh':>5s} {'TP%':>5s} {'SL%':>5s}")
for r in results:
    print(f"{r['symbol']:10s} {r['trades']:>7d} {r['wr']:>5.1f}% {r['net_pts']:>+8.0f}  {r['pf']:>4.2f} {r['avg_r']:>4.2f} {r['mdd']:>+7.0f} {r['sharpe']:>4.1f} {r['tp_pct']:>4.1f}% {r['sl_pct']:>4.1f}%")
