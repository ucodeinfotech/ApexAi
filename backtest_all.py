"""
Pivot Breakout Backtester - Sequential per stock
"""
import pandas as pd
import numpy as np
import os, time, sys
from datetime import datetime, timedelta

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BROKERAGE_PER_ORDER = 10
STT = 0.001
EXCHANGE_TC = 0.00003
SEBI_TC = 0.000001
GST = 0.18
STAMP_DUTY = 0.00003
SLIPPAGE_POINTS = 1

# Exclude indices from stock list
EXCLUDE = {"NIFTY50", "BANKNIFTY", "SENSEX"}

def calc_pivot_daily(high, low, close):
    pivot = (high + low + close) / 3
    return pivot, 2*pivot - low, 2*pivot - high, pivot + (high - low), pivot - (high - low)

def compute_charges(entry_price, exit_price, qty=1):
    turnover_buy = entry_price * qty
    turnover_sell = exit_price * qty
    turnover_total = turnover_buy + turnover_sell
    brokerage = BROKERAGE_PER_ORDER * 2
    stt_total = turnover_sell * STT
    exchange_total = turnover_total * EXCHANGE_TC
    sebi_total = turnover_total * SEBI_TC * 2
    stamp = turnover_buy * STAMP_DUTY
    gst_total = (brokerage + exchange_total) * GST
    total = brokerage + stt_total + exchange_total + sebi_total + stamp + gst_total
    return total

def backtest_stock(symbol, data_dir):
    df15_path = f"{data_dir}/{symbol}_FIFTEEN_MINUTE.csv"
    df1_path = f"{data_dir}/{symbol}_ONE_MINUTE.csv"
    if not os.path.exists(df15_path) or not os.path.exists(df1_path):
        return None, "Missing data files"

    df15 = pd.read_csv(df15_path)
    df1 = pd.read_csv(df1_path)
    df15["datetime"] = pd.to_datetime(df15["datetime"])
    df1["datetime"] = pd.to_datetime(df1["datetime"])
    df15["date"] = df15["datetime"].dt.date
    df1["date"] = df1["datetime"].dt.date
    df15 = df15.sort_values("datetime").reset_index(drop=True)
    df1 = df1.sort_values("datetime").reset_index(drop=True)

    # Daily pivots
    daily = df15.groupby("date").agg({"high":"max","low":"min","close":"last"}).reset_index()
    pivots = []
    for _, row in daily.iterrows():
        p, r1, s1, r2, s2 = calc_pivot_daily(row["high"], row["low"], row["close"])
        pivots.append({"date": row["date"], "pivot": p, "r1": r1, "s1": s1, "r2": r2, "s2": s2})
    pivots_df = pd.DataFrame(pivots)
    pivots_df["date"] = pivots_df["date"].shift(1)
    pivots_df = pivots_df.dropna().reset_index(drop=True)
    df15 = df15.merge(pivots_df, on="date", how="left")
    df15 = df15.dropna(subset=["r1","s1"]).reset_index(drop=True)

    # Find triggers
    triggers = []
    for _, row in df15.iterrows():
        long_t = row["high"] >= row["r1"]
        short_t = row["low"] <= row["s1"]
        if long_t or short_t:
            tp = "LONG" if long_t else "SHORT"
            entry = row["high"] + SLIPPAGE_POINTS if long_t else row["low"] - SLIPPAGE_POINTS
            sl = row["low"] if long_t else row["high"]
            if tp == "LONG":
                risk = entry - sl
                tp_price = entry + 2 * risk
            else:
                risk = sl - entry
                tp_price = entry - 2 * risk
            triggers.append({
                "datetime": row["datetime"], "date": row["date"],
                "type": tp, "trigger_high": row["high"], "trigger_low": row["low"],
                "entry_level": entry, "sl": sl, "tp": tp_price,
                "r1": row["r1"], "s1": row["s1"]
            })

    if not triggers:
        return None, "No triggers"

    # Match 1-min bars
    trades = []
    for tr in triggers:
        t_dt = tr["datetime"]
        t_date = tr["date"]
        window_end = datetime.combine(t_date, datetime.max.time()).replace(hour=15, minute=30)
        mask = (df1["datetime"] > t_dt) & (df1["datetime"] <= pd.Timestamp(window_end, tz=df1["datetime"].dt.tz))
        scan = df1[mask].copy()
        if scan.empty:
            continue

        filled = False; entry_p = None; entry_t = None
        exit_p = None; exit_t = None; reason = None

        for _, bar in scan.iterrows():
            if not filled:
                if tr["type"] == "LONG" and bar["high"] >= tr["entry_level"]:
                    entry_p = tr["entry_level"]; entry_t = bar["datetime"]; filled = True
                elif tr["type"] == "SHORT" and bar["low"] <= tr["entry_level"]:
                    entry_p = tr["entry_level"]; entry_t = bar["datetime"]; filled = True
            else:
                if tr["type"] == "LONG":
                    if bar["low"] <= tr["sl"]:
                        exit_p = tr["sl"]; exit_t = bar["datetime"]; reason = "SL"; break
                    elif bar["high"] >= tr["tp"]:
                        exit_p = tr["tp"]; exit_t = bar["datetime"]; reason = "TP"; break
                else:
                    if bar["high"] >= tr["sl"]:
                        exit_p = tr["sl"]; exit_t = bar["datetime"]; reason = "SL"; break
                    elif bar["low"] <= tr["tp"]:
                        exit_p = tr["tp"]; exit_t = bar["datetime"]; reason = "TP"; break

        if filled and exit_p is not None:
            pnl = exit_p - entry_p if tr["type"] == "LONG" else entry_p - exit_p
            risk_amt = abs(entry_p - tr["sl"])
            r_m = round(pnl / risk_amt, 2) if risk_amt > 0 else 0
            charges = compute_charges(entry_p, exit_p)
            net = round(pnl - charges, 2)
            trades.append({
                "symbol": symbol, "date": str(t_date), "type": tr["type"],
                "entry_time": str(entry_t), "exit_time": str(exit_t),
                "trigger_time": str(t_dt),
                "entry": round(entry_p,2), "exit": round(exit_p,2),
                "sl": round(tr["sl"],2), "tp": round(tr["tp"],2),
                "reason": reason, "pnl": round(pnl,2), "net_pnl": net,
                "r": r_m, "charges": round(charges,2)
            })

    if not trades:
        return None, "No fills"

    return trades, None

def print_stock_result(symbol, trades):
    if not trades:
        print(f"\n{'='*55}")
        print(f"  {symbol:15s}  NO TRADES")
        print(f"{'='*55}")
        return None

    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc / total * 100, 1) if total else 0
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_ = round(df["net_pnl"].sum(), 2)
    pf = round(abs(gp / gl), 2) if gl != 0 else float('inf')
    avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
    avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
    avg_r = round(df["r"].mean(), 2)
    exp_ = round(avg_w * (wr/100) + avg_l * (1 - wr/100), 2)

    # Max DD
    df_s = df.sort_values("exit_time").reset_index(drop=True)
    df_s["cum"] = df_s["net_pnl"].cumsum()
    df_s["peak"] = df_s["cum"].cummax()
    df_s["dd"] = df_s["peak"] - df_s["cum"]
    mdd = round(df_s["dd"].max(), 2)
    mdd_p = round(mdd / df_s["peak"].max() * 100, 2) if df_s["peak"].max() > 0 else 0

    sharpe = round(df["r"].mean() / df["r"].std() * np.sqrt(total), 2) if df["r"].std() > 0 else 0
    tcharges = round(df["charges"].sum(), 2)

    print(f"\n{'='*55}")
    print(f"  {symbol:15s}  Trades: {total:>5}  Win: {wc}/{total} ({wr}%)")
    print(f"  {'':15s}  Net P&L: Rs{np_:>8,.2f}  PF: {pf:>6.2f}")
    print(f"  {'':15s}  Avg W/L: Rs{avg_w:>6,.2f} / Rs{avg_l:>6,.2f}  Avg R: {avg_r}")
    print(f"  {'':15s}  Expectancy: Rs{exp_:>6,.2f}  Sharpe: {sharpe}")
    print(f"  {'':15s}  Max DD: Rs{mdd:>8,.2f} ({mdd_p}%)")
    print(f"  {'':15s}  Charges: Rs{tcharges:>8,.2f}")
    print(f"{'='*55}")

    return {
        "symbol": symbol, "trades": total, "wins": wc, "losses": lc,
        "win_rate": wr, "net_pnl": np_, "profit_factor": pf,
        "avg_win": avg_w, "avg_loss": avg_l, "avg_r": avg_r,
        "expectancy": exp_, "max_dd": mdd, "max_dd_pct": mdd_p,
        "sharpe": sharpe, "charges": tcharges
    }

def main():
    print("=" * 60)
    print("PIVOT BREAKOUT BACKTESTER")
    print("15-min TRIGGERS (touch R1/S1) -> 1-min ENTRIES")
    print(f"Slippage: {SLIPPAGE_POINTS}pt | Brokerage: Rs{BROKERAGE_PER_ORDER}/order")
    print("=" * 60)

    files = sorted([f.replace("_FIFTEEN_MINUTE.csv","") for f in os.listdir(DATA_DIR) if f.endswith("_FIFTEEN_MINUTE.csv")])
    symbols = [s for s in files if s not in EXCLUDE]
    print(f"\nStocks to backtest: {len(symbols)}")
    print("-" * 55)

    all_results = []
    all_trades = []
    start_time = time.time()

    for idx, sym in enumerate(symbols, 1):
        stock_start = time.time()
        print(f"\n[{idx}/{len(symbols)}] {sym}...", end="")
        sys.stdout.flush()

        trades, err = backtest_stock(sym, DATA_DIR)
        if trades:
            all_trades.extend(trades)
            res = print_stock_result(sym, trades)
            if res:
                all_results.append(res)
            elapsed = time.time() - stock_start
            print(f"  [{elapsed:.1f}s]", end="")
        else:
            print(f"  SKIP ({err}) [{time.time()-stock_start:.1f}s]", end="")
        sys.stdout.flush()

        # Save per-stock trade book
        if trades:
            pd.DataFrame(trades).to_csv(f"{OUTPUT_DIR}/{sym}_trades.csv", index=False)

    # Combined results
    print(f"\n\n{'='*60}")
    print(f"COMBINED RESULTS - ALL {len(symbols)} STOCKS")
    print(f"{'='*60}")

    if not all_trades:
        print("No trades generated across any stock.")
        return

    combined_df = pd.DataFrame(all_trades)
    combined_df.to_csv(f"{OUTPUT_DIR}/all_trades.csv", index=False)

    total_trades = len(combined_df)
    wins = combined_df[combined_df["net_pnl"] > 0]
    losses = combined_df[combined_df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc / total_trades * 100, 2) if total_trades else 0
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_total = round(combined_df["net_pnl"].sum(), 2)
    pf = round(abs(gp / gl), 2) if gl != 0 else float('inf')

    avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
    avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
    avg_r = round(combined_df["r"].mean(), 2)
    exp_ = round(avg_w * (wr/100) + avg_l * (1 - wr/100), 2)

    combined_sorted = combined_df.sort_values("exit_time").reset_index(drop=True)
    combined_sorted["cum"] = combined_sorted["net_pnl"].cumsum()
    combined_sorted["peak"] = combined_sorted["cum"].cummax()
    combined_sorted["dd"] = combined_sorted["peak"] - combined_sorted["cum"]
    mdd = round(combined_sorted["dd"].max(), 2)
    mdd_p = round(mdd / combined_sorted["peak"].max() * 100, 2) if combined_sorted["peak"].max() > 0 else 0

    sharpe = round(combined_df["r"].mean() / combined_df["r"].std() * np.sqrt(total_trades), 2) if combined_df["r"].std() > 0 else 0
    total_charges = round(combined_df["charges"].sum(), 2)

    print(f"\n{'─'*55}")
    print(f"  Total Trades:      {total_trades:>8}")
    print(f"  Wins / Losses:     {wc:>8} / {lc}")
    print(f"  Win Rate:          {wr:>8.2f}%")
    print(f"  Net P&L:           Rs{np_total:>8,.2f}")
    print(f"  Gross Profit:      Rs{gp:>8,.2f}")
    print(f"  Gross Loss:        Rs{gl:>8,.2f}")
    print(f"  Profit Factor:     {pf:>8.2f}")
    print(f"  Avg Win:           Rs{avg_w:>8,.2f}")
    print(f"  Avg Loss:          Rs{avg_l:>8,.2f}")
    print(f"  Avg R Multiple:    {avg_r:>8.2f}")
    print(f"  Expectancy:        Rs{exp_:>8,.2f}")
    print(f"  Max Drawdown:      Rs{mdd:>8,.2f} ({mdd_p}%)")
    print(f"  Sharpe Ratio:      {sharpe:>8.2f}")
    print(f"  Total Charges:     Rs{total_charges:>8,.2f}")
    print(f"{'─'*55}")

    # Stock ranking
    print(f"\n{'='*60}")
    print("STOCK RANKING")
    print(f"{'='*60}")
    sorted_results = sorted(all_results, key=lambda x: x["net_pnl"], reverse=True)
    print(f"\n{'Rank':>4s} {'Symbol':18s} {'Trades':>7s} {'Net P&L':>10s} {'Win%':>7s} {'Avg R':>7s} {'PF':>7s} {'Sharpe':>7s}")
    print(f"{'─'*70}")
    for rank, r in enumerate(sorted_results, 1):
        print(f"{rank:>4d} {r['symbol']:18s} {r['trades']:>7d} Rs{r['net_pnl']:>7,.2f} {r['win_rate']:>6.1f}% {r['avg_r']:>6.2f} {r['profit_factor']:>6.2f} {r['sharpe']:>6.2f}")

    # Suggestions
    print(f"\n{'='*60}")
    print("SUGGESTIONS & OBSERVATIONS")
    print(f"{'='*60}")
    
    best = sorted_results[0] if sorted_results else None
    worst = sorted_results[-1] if sorted_results else None
    profitable = [r for r in sorted_results if r["net_pnl"] > 0]
    losing = [r for r in sorted_results if r["net_pnl"] <= 0]

    print(f"  Profitable stocks: {len(profitable)}/{len(sorted_results)}")
    print(f"  Losing stocks:     {len(losing)}/{len(sorted_results)}")
    if best:
        print(f"  Best stock:        {best['symbol']} (Rs{best['net_pnl']:.2f}, WR: {best['win_rate']}%)")
    if worst:
        print(f"  Worst stock:       {worst['symbol']} (Rs{worst['net_pnl']:.2f}, WR: {worst['win_rate']}%)")
    
    avg_win_rate = np.mean([r["win_rate"] for r in sorted_results])
    print(f"  Avg win rate:      {avg_win_rate:.1f}%")
    
    high_win = [r for r in sorted_results if r["win_rate"] >= 50]
    low_win = [r for r in sorted_results if r["win_rate"] < 30]
    if high_win:
        print(f"  Stocks with >=50% WR: {len(high_win)}")
        print(f"    {', '.join(r['symbol'] for r in high_win[:10])}")
    if low_win:
        print(f"  Stocks with <30% WR: {len(low_win)}")
        print(f"    {', '.join(r['symbol'] for r in low_win[:10])}")

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"\nTrade books saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
