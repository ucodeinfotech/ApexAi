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
SLIPPAGE_PTS = 0

INDICES = {"NIFTY50": "NIFTY50", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}

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

def backtest_gap_fill(symbol, data_dir):
    path = f"{data_dir}/{symbol}_FIVE_MINUTE.csv"
    if not os.path.exists(path):
        return None, "Missing data"

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["is_first"] = df["date"] != df["date"].shift(1)

    daily_close = df.groupby("date")["close"].last().shift(1)
    first_bars = df[df["is_first"]].copy()
    first_bars = first_bars.merge(daily_close.rename("prev_close"), left_on="date", right_index=True, how="left")
    first_bars["gap_pct"] = (first_bars["open"] - first_bars["prev_close"]) / first_bars["prev_close"] * 100

    trades = []
    for _, row in first_bars.iterrows():
        gap = row["gap_pct"]
        d = row["date"]
        entry_time = row["datetime"]
        prev_close = row["prev_close"]
        entry_price = row["open"]

        # Classify gap
        if 0.2 <= gap <= 0.5:
            trade_type = "SHORT"
            sl_pct = 0.3
            max_hold_min = 120
        elif gap > 0.5:
            trade_type = "NO_TRADE_BIG_GAP"
            continue
        elif gap < -0.5:
            trade_type = "NO_TRADE_BIG_GAP_DOWN"
            continue
        elif -0.5 <= gap <= -0.2:
            trade_type = "LONG"
            sl_pct = 0.3
            max_hold_min = 120
        else:
            continue

        sl_price = entry_price * (1 + sl_pct/100) if trade_type == "SHORT" else entry_price * (1 - sl_pct/100)
        tp_price = prev_close  # gap fill = target

        day_bars = df[(df["date"] == d) & (df["datetime"] > entry_time)].copy()
        filled = False
        exit_price = entry_price
        exit_time = entry_time
        reason = "TIMEOUT"

        for _, bar in day_bars.iterrows():
            elapsed = (pd.Timestamp(bar["datetime"]) - pd.Timestamp(entry_time)).total_seconds() / 60
            if elapsed > max_hold_min:
                exit_price = bar["open"]
                exit_time = bar["datetime"]
                reason = "TIMEOUT"
                break

            if trade_type == "SHORT":
                if bar["low"] <= tp_price:
                    exit_price = tp_price
                    exit_time = bar["datetime"]
                    reason = "GAP_FILL"
                    filled = True
                    break
                if bar["high"] >= sl_price:
                    exit_price = sl_price
                    exit_time = bar["datetime"]
                    reason = "SL"
                    break
            else:  # LONG (gap down)
                if bar["high"] >= tp_price:
                    exit_price = tp_price
                    exit_time = bar["datetime"]
                    reason = "GAP_FILL"
                    filled = True
                    break
                if bar["low"] <= sl_price:
                    exit_price = sl_price
                    exit_time = bar["datetime"]
                    reason = "SL"
                    break

        pnl = (entry_price - exit_price) if trade_type == "SHORT" else (exit_price - entry_price)
        charges = compute_charges(entry_price, exit_price)
        net_pnl = pnl - charges
        r_mult = round(pnl / (sl_pct/100 * entry_price), 2) if sl_pct > 0 else 0

        trades.append({
            "symbol": symbol,
            "date": str(d),
            "type": trade_type,
            "gap_pct": round(gap, 2),
            "entry_time": str(entry_time),
            "exit_time": str(exit_time),
            "entry": round(entry_price, 2),
            "exit": round(exit_price, 2),
            "sl": round(sl_price, 2),
            "target": round(tp_price, 2),
            "reason": reason,
            "filled": filled,
            "pnl": round(pnl, 2),
            "charges": round(charges, 2),
            "net_pnl": round(net_pnl, 2),
            "r": r_mult,
        })

    if not trades:
        return None, "No gap trades"

    return trades, None

def print_stock_result(symbol, trades):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc / total * 100, 2) if total else 0
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_ = round(df["net_pnl"].sum(), 2)
    pf = round(abs(gp / gl), 2) if gl != 0 else float('inf')
    avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
    avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
    avg_r = round(df["r"].mean(), 2)
    total_charges = round(df["charges"].sum(), 2)

    df_sorted = df.sort_values("exit_time").reset_index(drop=True)
    df_sorted["cum"] = df_sorted["net_pnl"].cumsum()
    df_sorted["peak"] = df_sorted["cum"].cummax()
    df_sorted["dd"] = df_sorted["peak"] - df_sorted["cum"]
    mdd = round(df_sorted["dd"].max(), 2)
    mdd_p = round(mdd / df_sorted["peak"].max() * 100, 2) if df_sorted["peak"].max() > 0 else 0
    sharpe = round(df["r"].mean() / df["r"].std() * np.sqrt(total), 2) if df["r"].std() > 0 else 0

    gap_fill_trades = df[df["reason"] == "GAP_FILL"]
    sl_trades = df[df["reason"] == "SL"]
    timeout_trades = df[df["reason"] == "TIMEOUT"]

    print(f"\n{'='*55}")
    print(f"  {symbol:15s}  Trades: {total:>5}  Win: {wc}/{total} ({wr}%)")
    print(f"  {'':15s}  Net P&L: Rs{np_:>8,.2f}  PF: {pf:>6.2f}")
    print(f"  {'':15s}  GapFill: {len(gap_fill_trades)}  SL: {len(sl_trades)}  Timeout: {len(timeout_trades)}")
    print(f"  {'':15s}  Avg W/L: Rs{avg_w:>6,.2f} / Rs{avg_l:>6,.2f}  Avg R: {avg_r:>6.2f}")
    print(f"  {'':15s}  Max DD: Rs{mdd:>8,.2f} ({mdd_p}%)  Sharpe: {sharpe:>6.2f}")
    print(f"  {'':15s}  Charges: Rs{total_charges:>8,.2f}")
    print(f"{'='*55}")

    return {
        "symbol": symbol, "trades": total, "wins": wc, "losses": lc,
        "win_rate": wr, "net_pnl": np_, "profit_factor": pf,
        "avg_win": avg_w, "avg_loss": avg_l, "avg_r": avg_r,
        "max_dd": mdd, "max_dd_pct": mdd_p,
        "sharpe": sharpe, "charges": total_charges,
        "gap_fills": len(gap_fill_trades),
    }

def main():
    print("=" * 60)
    print("  GAP FILL STRATEGY BACKTEST (0.2-0.5% gaps)")
    print("  Short gap-ups, Long gap-downs, TP=prev close, SL=0.3%")
    print("=" * 60)

    all_trades = []
    all_results = []
    start_time = time.time()

    for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
        stock_start = time.time()
        print(f"\n[{sym}]...", end="")
        sys.stdout.flush()

        trades, err = backtest_gap_fill(sym, DATA_DIR)
        if trades:
            all_trades.extend(trades)
            res = print_stock_result(sym, trades)
            if res:
                all_results.append(res)
        else:
            print(f"  SKIP ({err})", end="")
        print(f"  [{time.time()-stock_start:.1f}s]", end="")
        sys.stdout.flush()

        if trades:
            pd.DataFrame(trades).to_csv(f"{OUTPUT_DIR}/{sym}_gap_fill_trades.csv", index=False)

    print(f"\n\n{'='*60}")
    print("  COMBINED RESULTS")
    print(f"{'='*60}")

    if not all_trades:
        print("No trades.")
        return

    combined_df = pd.DataFrame(all_trades)
    combined_df.to_csv(f"{OUTPUT_DIR}/all_gap_fill_trades.csv", index=False)

    total = len(combined_df)
    wins = combined_df[combined_df["net_pnl"] > 0]
    losses = combined_df[combined_df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc / total * 100, 2)
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_total = round(combined_df["net_pnl"].sum(), 2)
    pf = round(abs(gp / gl), 2) if gl != 0 else float('inf')
    avg_r = round(combined_df["r"].mean(), 2)
    gap_fills = len(combined_df[combined_df["reason"] == "GAP_FILL"])
    sl_hits = len(combined_df[combined_df["reason"] == "SL"])
    timeouts = len(combined_df[combined_df["reason"] == "TIMEOUT"])

    combined_sorted = combined_df.sort_values("exit_time").reset_index(drop=True)
    combined_sorted["cum"] = combined_sorted["net_pnl"].cumsum()
    combined_sorted["peak"] = combined_sorted["cum"].cummax()
    combined_sorted["dd"] = combined_sorted["peak"] - combined_sorted["cum"]
    mdd = round(combined_sorted["dd"].max(), 2)
    mdd_p = round(mdd / combined_sorted["peak"].max() * 100, 2) if combined_sorted["peak"].max() > 0 else 0
    sharpe = round(combined_df["r"].mean() / combined_df["r"].std() * np.sqrt(total), 2) if combined_df["r"].std() > 0 else 0
    total_charges = round(combined_df["charges"].sum(), 2)

    print(f"\n  Total Trades:      {total:>8}")
    print(f"  Wins / Losses:     {wc:>8} / {lc}")
    print(f"  Win Rate:          {wr:>8.2f}%")
    print(f"  Net P&L:           Rs{np_total:>8,.2f}")
    print(f"  Profit Factor:     {pf:>8.2f}")
    print(f"  Gap Fills:         {gap_fills:>8}  SL: {sl_hits}  Timeout: {timeouts}")
    print(f"  Avg R Multiple:    {avg_r:>8.2f}")
    print(f"  Max Drawdown:      Rs{mdd:>8,.2f} ({mdd_p}%)")
    print(f"  Sharpe:            {sharpe:>8.2f}")
    print(f"  Charges:           Rs{total_charges:>8,.2f}")

    print(f"\n  Time: {time.time()-start_time:.1f}s")
    print(f"  Trades saved to {OUTPUT_DIR}/all_gap_fill_trades.csv")

if __name__ == "__main__":
    main()
