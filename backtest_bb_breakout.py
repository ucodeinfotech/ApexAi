import pandas as pd
import numpy as np
import os, time, sys
from datetime import datetime, timedelta

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BROKERAGE_PER_ORDER = 10
STT = 0.001; EXCHANGE_TC = 0.00003; SEBI_TC = 0.000001
GST = 0.18; STAMP_DUTY = 0.00003
SLIPPAGE_PTS = 0

INDICES = ["NIFTY50", "BANKNIFTY", "SENSEX"]

def compute_charges(entry_price, exit_price, qty=1):
    tb = entry_price * qty; ts = exit_price * qty
    return (BROKERAGE_PER_ORDER * 2 + ts * STT + (tb+ts) * EXCHANGE_TC
            + (tb+ts) * SEBI_TC * 2 + tb * STAMP_DUTY
            + (BROKERAGE_PER_ORDER * 2 + (tb+ts) * EXCHANGE_TC) * GST)

def backtest_bb_breakout(symbol, data_dir, period=20, n_std=1.5):
    path = f"{data_dir}/{symbol}_FIFTEEN_MINUTE.csv"
    if not os.path.exists(path):
        return None, "Missing data"

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date

    # BB calculation
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = ma + n_std * std
    df["bb_lower"] = ma - n_std * std
    df["bb_mid"] = ma
    df = df.dropna().reset_index(drop=True)

    trades = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        # Check if candle is fully outside BB
        if row["low"] > row["bb_upper"]:
            typ = "SHORT"
            trigger_high = row["high"]
            trigger_low = row["low"]
            trigger_idx = i
            trigger_dt = row["datetime"]
        elif row["high"] < row["bb_lower"]:
            typ = "LONG"
            trigger_high = row["high"]
            trigger_low = row["low"]
            trigger_idx = i
            trigger_dt = row["datetime"]
        else:
            i += 1
            continue

        # Slide trigger forward until a break happens
        entry_made = False
        j = trigger_idx + 1
        while j < len(df):
            next_row = df.iloc[j]
            next_date = next_row["date"]

            if typ == "SHORT":
                if next_row["low"] <= trigger_low:
                    # Entry: break of trigger low
                    entry_price = min(next_row["open"], trigger_low)
                    sl_price = trigger_high
                    entry_time = next_row["datetime"]
                    entry_made = True
                    break
            else:
                if next_row["high"] >= trigger_high:
                    entry_price = max(next_row["open"], trigger_high)
                    sl_price = trigger_low
                    entry_time = next_row["datetime"]
                    entry_made = True
                    break

            # Check if this new candle becomes trigger
            if typ == "SHORT" and next_row["low"] > next_row["bb_upper"]:
                # Still above upper BB, slide trigger
                trigger_high = max(trigger_high, next_row["high"])
                trigger_low = min(trigger_low, next_row["low"])
                trigger_idx = j
            elif typ == "LONG" and next_row["high"] < next_row["bb_lower"]:
                trigger_high = max(trigger_high, next_row["high"])
                trigger_low = min(trigger_low, next_row["low"])
                trigger_idx = j
            else:
                # Candle is inside or touching BB → stop sliding, break out
                break
            j += 1

        if not entry_made:
            i = j if j > trigger_idx else trigger_idx + 1
            continue

        # Compute SL distance and TP
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            i = j + 1
            continue

        if typ == "SHORT":
            tp_price = entry_price - sl_distance * 2
        else:
            tp_price = entry_price + sl_distance * 2

        # Simulate exit on 15-min bars
        exit_price = entry_price
        exit_time = entry_time
        reason = "TIMEOUT"

        k = j + 1
        while k < len(df):
            bar = df.iloc[k]
            bar_dt = bar["datetime"]

            # End of day check (close at 15:15)
            if bar_dt.hour >= 15 and bar_dt.minute >= 15:
                exit_price = bar["close"]
                exit_time = bar_dt
                reason = "EOD"
                break

            tp_hit = (typ == "SHORT" and bar["low"] <= tp_price) or (typ == "LONG" and bar["high"] >= tp_price)
            sl_hit = (typ == "SHORT" and bar["high"] >= sl_price) or (typ == "LONG" and bar["low"] <= sl_price)

            if tp_hit and sl_hit:
                # Both hit in same candle → use which is closer to entry
                if typ == "SHORT":
                    if bar["high"] - sl_price < sl_price - bar["low"]:
                        exit_price = sl_price; reason = "SL"
                    else:
                        exit_price = tp_price; reason = "TP"
                else:
                    if sl_price - bar["low"] < bar["high"] - tp_price:
                        exit_price = sl_price; reason = "SL"
                    else:
                        exit_price = tp_price; reason = "TP"
            elif tp_hit:
                exit_price = tp_price; exit_time = bar_dt; reason = "TP"; break
            elif sl_hit:
                exit_price = sl_price; exit_time = bar_dt; reason = "SL"; break

            k += 1
        else:
            # Ran out of bars
            exit_price = df.iloc[-1]["close"]
            exit_time = df.iloc[-1]["datetime"]
            reason = "NO_EXIT"

        pnl = (entry_price - exit_price) if typ == "SHORT" else (exit_price - entry_price)
        charges = compute_charges(entry_price, exit_price)
        trades.append({
            "symbol": symbol, "date": str(trigger_dt.date()), "type": typ,
            "trigger_time": str(trigger_dt), "entry_time": str(entry_time),
            "exit_time": str(exit_time), "entry": round(entry_price, 2),
            "exit": round(exit_price, 2), "sl": round(sl_price, 2),
            "tp": round(tp_price, 2), "trigger_high": round(trigger_high, 2),
            "trigger_low": round(trigger_low, 2),
            "reason": reason, "pnl": round(pnl, 2),
            "charges": round(charges, 2), "net_pnl": round(pnl - charges, 2),
            "r": round(pnl / sl_distance, 2) if sl_distance > 0 else 0,
        })

        i = k + 1 if k < len(df) else len(df)

    if not trades:
        return None, "No trades"
    return trades, None

def print_stock_result(symbol, trades):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["net_pnl"] > 0]; losses = df[df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc / total * 100, 2) if total else 0
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_ = round(df["net_pnl"].sum(), 2)
    pf = round(abs(gp/gl), 2) if gl != 0 else float('inf')
    avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
    avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
    avg_r = round(df["r"].mean(), 2)

    df_sorted = df.sort_values("exit_time").reset_index(drop=True)
    df_sorted["cum"] = df_sorted["net_pnl"].cumsum()
    df_sorted["peak"] = df_sorted["cum"].cummax()
    df_sorted["dd"] = df_sorted["peak"] - df_sorted["cum"]
    mdd = round(df_sorted["dd"].max(), 2)
    mdd_p = round(mdd / df_sorted["peak"].max() * 100, 2) if df_sorted["peak"].max() > 0 else 0
    sharpe = round(df["r"].mean() / df["r"].std() * np.sqrt(total), 2) if df["r"].std() > 0 else 0

    tp_count = (df["reason"] == "TP").sum()
    sl_count = (df["reason"] == "SL").sum()
    eod_count = (df["reason"] == "EOD").sum()

    print(f"\n{'='*55}")
    print(f"  {symbol:15s}  Trades: {total:>5}  Win: {wc}/{total} ({wr}%)")
    print(f"  {'':15s}  Net P&L: Rs{np_:>8,.2f}  PF: {pf:>6.2f}")
    print(f"  {'':15s}  TP: {tp_count}  SL: {sl_count}  EOD: {eod_count}")
    print(f"  {'':15s}  Avg W/L: Rs{avg_w:>6,.2f} / Rs{avg_l:>6,.2f}  Avg R: {avg_r}")
    print(f"  {'':15s}  Max DD: Rs{mdd:>8,.2f} ({mdd_p}%)  Sharpe: {sharpe}")
    print(f"  {'':15s}  Charges: Rs{round(df['charges'].sum(),2):>8,.2f}")
    print(f"{'='*55}")

    return {"symbol": symbol, "trades": total, "wins": wc, "losses": lc,
            "win_rate": wr, "net_pnl": np_, "profit_factor": pf,
            "avg_win": avg_w, "avg_loss": avg_l, "avg_r": avg_r,
            "max_dd": mdd, "max_dd_pct": mdd_p, "sharpe": sharpe}

def main():
    print("="*60)
    print("  BB BREAKOUT STRATEGY (SD=1.5, 15-min)")
    print("  Trigger: candle fully outside BB | Entry: break of trigger H/L")
    print("  Sliding trigger | SL=trigger opposite | TP=2x SL")
    print("="*60)

    all_trades = []
    all_results = []
    start_time = time.time()

    for sym in INDICES:
        stock_start = time.time()
        print(f"\n[{sym}]...", end="")
        sys.stdout.flush()

        trades, err = backtest_bb_breakout(sym, DATA_DIR)
        if trades:
            all_trades.extend(trades)
            res = print_stock_result(sym, trades)
            if res:
                all_results.append(res)
            pd.DataFrame(trades).to_csv(f"{OUTPUT_DIR}/{sym}_bb15_trades.csv", index=False)
        else:
            print(f"  SKIP ({err})", end="")
        print(f"  [{time.time()-stock_start:.1f}s]", end="")
        sys.stdout.flush()

    if not all_trades:
        print("\nNo trades.")
        return

    combined_df = pd.DataFrame(all_trades)
    combined_df.to_csv(f"{OUTPUT_DIR}/all_bb15_trades.csv", index=False)

    total = len(combined_df)
    wins = combined_df[combined_df["net_pnl"] > 0]; losses = combined_df[combined_df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc / total * 100, 2)
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_total = round(combined_df["net_pnl"].sum(), 2)
    pf = round(abs(gp/gl), 2) if gl != 0 else float('inf')
    avg_r = round(combined_df["r"].mean(), 2)

    cs = combined_df.sort_values("exit_time").reset_index(drop=True)
    cs["cum"] = cs["net_pnl"].cumsum()
    cs["peak"] = cs["cum"].cummax()
    cs["dd"] = cs["peak"] - cs["cum"]
    mdd = round(cs["dd"].max(), 2)
    mdd_p = round(mdd / cs["peak"].max() * 100, 2) if cs["peak"].max() > 0 else 0
    sharpe = round(combined_df["r"].mean() / combined_df["r"].std() * np.sqrt(total), 2) if combined_df["r"].std() > 0 else 0

    print(f"\n\n{'='*60}")
    print("  COMBINED RESULTS")
    print(f"{'='*60}")
    print(f"\n  Total Trades:      {total:>8}")
    print(f"  Wins / Losses:     {wc:>8} / {lc}")
    print(f"  Win Rate:          {wr:>8.2f}%")
    print(f"  Net P&L:           Rs{np_total:>8,.2f}")
    print(f"  Profit Factor:     {pf:>8.2f}")
    print(f"  Avg R:             {avg_r:>8.2f}")
    print(f"  Max DD:            Rs{mdd:>8,.2f} ({mdd_p}%)")
    print(f"  Sharpe:            {sharpe:>8.2f}")
    print(f"  Charges:           Rs{round(combined_df['charges'].sum(),2):>8,.2f}")
    print(f"  Time:              {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()
