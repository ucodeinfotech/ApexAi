import pandas as pd
import numpy as np
import os, time, json

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

def run_sweep():
    symbols = ["NIFTY50", "BANKNIFTY", "SENSEX"]
    # Reduce params for speed: gap range combos x SL x hold
    param_list = [(gm, gx, sm, h) for gm in [0.15,0.2] for gx in [0.4,0.5]
                  for sm in [1.5,2.0,2.5,3.0] for h in [120,240]]

    print(f"Params: {len(param_list)}, Symbols: {len(symbols)}", flush=True)
    all_results = []
    start = time.time()

    for sym in symbols:
        path = f"{DATA_DIR}/{sym}_FIVE_MINUTE.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["date"] = df["datetime"].dt.date
        df["dtime"] = df["datetime"].values.astype(np.int64) // 10**9

        # Pre-split data by date: {date: DataFrame slice}
        print(f"\n  Pre-processing {sym}...", end=" ", flush=True)
        day_groups = {d: g for d, g in df.groupby("date")}
        print(f"{len(day_groups)} days", flush=True)

        # First bar of each day with gap info
        daily_close = df.groupby("date")["close"].last().shift(1)
        first_bars = []
        for d, g in day_groups.items():
            row = g.iloc[0]
            pc = daily_close.get(d, np.nan)
            if pd.isna(pc):
                continue
            gap_pct = (row["open"] - pc) / pc * 100
            first_bars.append({
                "date": d, "entry_p": row["open"], "entry_t": row["dtime"],
                "prev_close": pc, "gap_pct": gap_pct,
                "gap_pt": abs(row["open"] - pc),
            })
        print(f"  {len(first_bars)} tradable days", flush=True)

        # For each param combo
        for gm, gx, sm, h in param_list:
            trades = []
            max_sec = h * 60

            for fb in first_bars:
                gap = fb["gap_pct"]
                if not (gm <= abs(gap) <= gx):
                    continue

                entry_p = fb["entry_p"]
                entry_t = fb["entry_t"]
                gap_pt = fb["gap_pt"]
                sl_dist = gap_pt * sm
                if sl_dist == 0:
                    continue

                d = fb["date"]
                day_df = day_groups.get(d)
                if day_df is None:
                    continue

                # Get bars after entry
                after = day_df[day_df["dtime"] > entry_t]
                if len(after) == 0:
                    continue

                if gap > 0:
                    tp_p = fb["prev_close"]
                    sl_p = entry_p + sl_dist
                    tp_check = after["low"].values <= tp_p
                    sl_check = after["high"].values >= sl_p
                    typ = "SHORT"
                else:
                    tp_p = fb["prev_close"]
                    sl_p = entry_p - sl_dist
                    tp_check = after["high"].values >= tp_p
                    sl_check = after["low"].values <= sl_p
                    typ = "LONG"

                elapsed = (after["dtime"].values - entry_t)
                timeout_check = elapsed > max_sec

                # Find first event
                tp_idx = np.where(tp_check)[0]
                sl_idx = np.where(sl_check)[0]
                to_idx = np.where(timeout_check)[0]

                exit_p, reason = entry_p, "TIMEOUT"
                if len(tp_idx) > 0:
                    if len(sl_idx) > 0 and sl_idx[0] < tp_idx[0]:
                        exit_p = sl_p; reason = "SL"
                    else:
                        exit_p = tp_p; reason = "GAP_FILL"
                elif len(sl_idx) > 0:
                    exit_p = sl_p; reason = "SL"

                pnl = (entry_p - exit_p) if typ == "SHORT" else (exit_p - entry_p)
                charges = compute_charges(entry_p, exit_p)
                trades.append(dict(symbol=sym, date=str(d), type=typ,
                    gap_pct=round(gap,2), entry_p=round(entry_p,2),
                    exit_p=round(exit_p,2), sl=round(sl_p,2),
                    target=round(tp_p,2), reason=reason,
                    pnl=round(pnl,2), charges=round(charges,2),
                    net_pnl=round(pnl-charges,2),
                    r=round(pnl/sl_dist,2) if sl_dist>0 else 0))

            if not trades:
                continue

            tdf = pd.DataFrame(trades)
            n = len(tdf)
            wc = (tdf["net_pnl"] > 0).sum(); lc = n - wc
            wr = round(wc/n*100,2) if n else 0
            net = round(tdf["net_pnl"].sum(),2)
            gp = round(tdf[tdf["net_pnl"]>0]["net_pnl"].sum(),2)
            gl = round(tdf[tdf["net_pnl"]<=0]["net_pnl"].sum(),2)
            pf = round(abs(gp/gl),2) if gl else 0
            ar = round(tdf["r"].mean(),3)
            as_ = round(tdf["r"].std(),3)
            sh = round(ar/as_*np.sqrt(n),2) if as_ else 0
            fills = (tdf["reason"]=="GAP_FILL").sum()

            all_results.append(dict(symbol=sym, gap_min=gm, gap_max=gx,
                sl_mult=sm, max_hold_min=h, trades=n, wins=wc, win_rate=wr,
                net_pnl=net, profit_factor=pf, avg_r=ar, sharpe=sh, gap_fills=fills))

        print(f"  {sym} done ({time.time()-start:.0f}s)", flush=True)

    pd.DataFrame(all_results).to_csv(f"{OUTPUT_DIR}/gap_fill_sweep.csv", index=False)
    print(f"\nSaved {len(all_results)} results to gap_fill_sweep.csv", flush=True)

    print(f"\n=== BEST 15 BY SHARPE (min 50 trades) ===", flush=True)
    top = sorted([r for r in all_results if r["trades"]>=50], key=lambda x: x["sharpe"], reverse=True)[:15]
    for r in top:
        print(f"  {r['symbol']:10s} gap=[{r['gap_min']}-{r['gap_max']}] sl={r['sl_mult']}x hold={r['max_hold_min']} "
              f"| n={r['trades']:>4d} WR={r['win_rate']:>5.1f}% PnL={r['net_pnl']:>+8.0f} "
              f"PF={r['profit_factor']:>4.2f} R={r['avg_r']:>5.2f} Sh={r['sharpe']:>5.2f} fills={r['gap_fills']}", flush=True)

    print(f"\n=== BEST 15 BY NET P&L (min 50 trades) ===", flush=True)
    top2 = sorted([r for r in all_results if r["trades"]>=50], key=lambda x: x["net_pnl"], reverse=True)[:15]
    for r in top2:
        print(f"  {r['symbol']:10s} gap=[{r['gap_min']}-{r['gap_max']}] sl={r['sl_mult']}x hold={r['max_hold_min']} "
              f"| n={r['trades']:>4d} WR={r['win_rate']:>5.1f}% PnL={r['net_pnl']:>+8.0f} "
              f"PF={r['profit_factor']:>4.2f} R={r['avg_r']:>5.2f} Sh={r['sharpe']:>5.2f} fills={r['gap_fills']}", flush=True)

    print(f"\nTotal time: {time.time()-start:.0f}s", flush=True)

if __name__ == "__main__":
    run_sweep()
