"""
Big Candle Reversal Pattern Backtester
1-hour signal → 5-min entry/exit (spot index data)
"""
import pandas as pd
import numpy as np
import os, time
from datetime import datetime

DATA_DIR = "."
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy params
LENGTH = 20
MULTIPLIER = 1.5
STRONG_BODY_PCT = 50.0

# Trading params
CHARGES_PER_ORDER = 10
CUTOFF_TIME = "14:15"
ENTRY_CUTOFF = pd.Timestamp(CUTOFF_TIME).time()

INDICES = ["NIFTY50", "BANKNIFTY", "SENSEX"]


def detect_signals(df_1h):
    body = (df_1h["close"] - df_1h["open"]).abs()
    avg_body = body.rolling(LENGTH, min_periods=LENGTH).mean()

    is_green = df_1h["close"] > df_1h["open"]
    is_red = df_1h["close"] < df_1h["open"]

    big_buy = is_green & (body > avg_body * MULTIPLIER)
    big_sell = is_red & (body > avg_body * MULTIPLIER)

    signals = []

    for i in range(1, len(df_1h)):
        if pd.isna(avg_body.iloc[i]):
            continue

        if big_buy.iloc[i - 1]:
            if not is_red.iloc[i]:
                continue
            prev_body = body.iloc[i - 1]
            curr_body = body.iloc[i]
            if curr_body < prev_body * (STRONG_BODY_PCT / 100):
                continue
            big_open = df_1h["open"].iloc[i - 1]
            big_close = df_1h["close"].iloc[i - 1]
            big_high = df_1h["high"].iloc[i - 1]
            big_low = df_1h["low"].iloc[i - 1]
            mid = (big_open + big_close) / 2
            if df_1h["close"].iloc[i] > mid:
                continue
            upper_wick = df_1h["high"].iloc[i] - df_1h["open"].iloc[i]
            if upper_wick > curr_body * 0.5:
                continue
            signals.append({
                "trigger_idx": i,
                "trigger_time": df_1h["datetime"].iloc[i],
                "date": df_1h["datetime"].iloc[i].date(),
                "dir": "SELL",
                "trigger_high": round(df_1h["high"].iloc[i], 2),
                "trigger_low": round(df_1h["low"].iloc[i], 2),
                "trigger_open": round(df_1h["open"].iloc[i], 2),
                "trigger_close": round(df_1h["close"].iloc[i], 2),
                "big_high": round(big_high, 2),
                "big_low": round(big_low, 2),
            })

        elif big_sell.iloc[i - 1]:
            if not is_green.iloc[i]:
                continue
            prev_body = body.iloc[i - 1]
            curr_body = body.iloc[i]
            if curr_body < prev_body * (STRONG_BODY_PCT / 100):
                continue
            big_open = df_1h["open"].iloc[i - 1]
            big_close = df_1h["close"].iloc[i - 1]
            big_high = df_1h["high"].iloc[i - 1]
            big_low = df_1h["low"].iloc[i - 1]
            mid = (big_open + big_close) / 2
            if df_1h["close"].iloc[i] < mid:
                continue
            lower_wick = df_1h["open"].iloc[i] - df_1h["low"].iloc[i]
            if lower_wick > curr_body * 0.5:
                continue
            signals.append({
                "trigger_idx": i,
                "trigger_time": df_1h["datetime"].iloc[i],
                "date": df_1h["datetime"].iloc[i].date(),
                "dir": "BUY",
                "trigger_high": round(df_1h["high"].iloc[i], 2),
                "trigger_low": round(df_1h["low"].iloc[i], 2),
                "trigger_open": round(df_1h["open"].iloc[i], 2),
                "trigger_close": round(df_1h["close"].iloc[i], 2),
                "big_high": round(big_high, 2),
                "big_low": round(big_low, 2),
            })

    return signals


def execute_trades(signals, df_5m, symbol):
    trades = []

    for sig in signals:
        trigger_time = sig["trigger_time"]
        direction = sig["dir"]
        level = sig["trigger_high"] if direction == "BUY" else sig["trigger_low"]

        mask = df_5m["datetime"] > trigger_time
        scan = df_5m[mask].copy()
        if scan.empty:
            continue

        if direction == "BUY":
            breakout_idx = scan[scan["close"] > level].index
            if len(breakout_idx) == 0:
                continue
            first_breakout = breakout_idx[0]
            retest_scan = scan.loc[first_breakout + 1:]
            if retest_scan.empty:
                continue
            for idx, bar in retest_scan.iterrows():
                if bar["low"] < level and bar["close"] > level:
                    if bar["datetime"].time() >= ENTRY_CUTOFF:
                        continue
                    entry_price = bar["close"]
                    sl_price = bar["low"]
                    risk = entry_price - sl_price
                    if risk <= 0:
                        continue
                    tp_price = entry_price + 2 * risk
                    exit_scan = scan.loc[idx + 1:]
                    exit_price = None
                    exit_time = None
                    reason = None
                    for _, bar2 in exit_scan.iterrows():
                        if bar2["low"] <= sl_price:
                            exit_price = sl_price
                            exit_time = bar2["datetime"]
                            reason = "SL"
                            break
                        elif bar2["high"] >= tp_price:
                            exit_price = tp_price
                            exit_time = bar2["datetime"]
                            reason = "TP"
                            break
                    if exit_price is not None:
                        points = exit_price - entry_price
                        charges_rs = CHARGES_PER_ORDER * 2
                        trades.append({
                            "symbol": symbol, "date": str(sig["date"]),
                            "dir": "BUY", "trigger_time": str(trigger_time),
                            "entry_time": str(bar["datetime"]),
                            "entry_price": round(entry_price, 2),
                            "sl": round(sl_price, 2),
                            "tp": round(tp_price, 2),
                            "exit_time": str(exit_time),
                            "exit_price": round(exit_price, 2),
                            "reason": reason,
                            "points": round(points, 2),
                            "charges_rs": charges_rs,
                        })
                    break

        else:
            breakout_idx = scan[scan["close"] < level].index
            if len(breakout_idx) == 0:
                continue
            first_breakout = breakout_idx[0]
            retest_scan = scan.loc[first_breakout + 1:]
            if retest_scan.empty:
                continue
            for idx, bar in retest_scan.iterrows():
                if bar["high"] > level and bar["close"] < level:
                    if bar["datetime"].time() >= ENTRY_CUTOFF:
                        continue
                    entry_price = bar["close"]
                    sl_price = bar["high"]
                    risk = sl_price - entry_price
                    if risk <= 0:
                        continue
                    tp_price = entry_price - 2 * risk
                    exit_scan = scan.loc[idx + 1:]
                    exit_price = None
                    exit_time = None
                    reason = None
                    for _, bar2 in exit_scan.iterrows():
                        if bar2["high"] >= sl_price:
                            exit_price = sl_price
                            exit_time = bar2["datetime"]
                            reason = "SL"
                            break
                        elif bar2["low"] <= tp_price:
                            exit_price = tp_price
                            exit_time = bar2["datetime"]
                            reason = "TP"
                            break
                    if exit_price is not None:
                        points = entry_price - exit_price
                        charges_rs = CHARGES_PER_ORDER * 2
                        trades.append({
                            "symbol": symbol, "date": str(sig["date"]),
                            "dir": "SELL", "trigger_time": str(trigger_time),
                            "entry_time": str(bar["datetime"]),
                            "entry_price": round(entry_price, 2),
                            "sl": round(sl_price, 2),
                            "tp": round(tp_price, 2),
                            "exit_time": str(exit_time),
                            "exit_price": round(exit_price, 2),
                            "reason": reason,
                            "points": round(points, 2),
                            "charges_rs": charges_rs,
                        })
                    break

    return trades


def compute_metrics(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["points"] > 0]
    losses = df[df["points"] <= 0]
    wc = len(wins)
    lc = len(losses)
    wr = round(wc / total * 100, 2) if total else 0
    gp = round(wins["points"].sum(), 2) if wc else 0
    gl = round(losses["points"].sum(), 2) if lc else 0
    net_pts = round(df["points"].sum(), 2)
    pf = round(abs(gp / gl), 2) if gl != 0 else float("inf")
    avg_w = round(wins["points"].mean(), 2) if wc else 0
    avg_l = round(losses["points"].mean(), 2) if lc else 0
    total_charges = int(df["charges_rs"].sum())
    df_sorted = df.sort_values("exit_time").reset_index(drop=True)
    df_sorted["cum"] = df_sorted["points"].cumsum()
    df_sorted["peak"] = df_sorted["cum"].cummax()
    df_sorted["dd"] = df_sorted["peak"] - df_sorted["cum"]
    mdd = round(df_sorted["dd"].max(), 2)
    mdd_pct = round(mdd / df_sorted["peak"].max() * 100, 2) if df_sorted["peak"].max() > 0 else 0
    sharpe = round(df["points"].mean() / df["points"].std() * np.sqrt(total), 2) if df["points"].std() > 0 else 0
    return {
        "total": total, "wins": wc, "losses": lc, "win_rate": wr,
        "gross_profit": gp, "gross_loss": gl, "net_points": net_pts,
        "profit_factor": pf, "avg_win": avg_w, "avg_loss": avg_l,
        "max_dd": mdd, "max_dd_pct": mdd_pct, "sharpe": sharpe,
        "total_charges_rs": total_charges,
    }


def main():
    print("=" * 65)
    print("BIG CANDLE REVERSAL BACKTESTER")
    print("1-hour signal -> 5-min entry/exit")
    print("=" * 65)

    all_trades = []
    combined_metrics = {}

    for sym in INDICES:
        print(f"\n--- {sym} ---")
        h1_path = f"{DATA_DIR}/{sym}_ONE_HOUR.csv"
        m5_path = f"{DATA_DIR}/{sym}_FIVE_MINUTE.csv"

        if not os.path.exists(h1_path) or not os.path.exists(m5_path):
            print(f"  SKIP - data files missing")
            continue

        df_1h = pd.read_csv(h1_path)
        df_5m = pd.read_csv(m5_path)
        df_1h["datetime"] = pd.to_datetime(df_1h["datetime"])
        df_5m["datetime"] = pd.to_datetime(df_5m["datetime"])
        df_1h = df_1h.sort_values("datetime").reset_index(drop=True)
        df_5m = df_5m.sort_values("datetime").reset_index(drop=True)

        start = time.time()
        signals = detect_signals(df_1h)
        print(f"  Signals found: {len(signals)}")
        if not signals:
            print(f"  No trades")
            continue

        trades = execute_trades(signals, df_5m, sym)
        print(f"  Trades executed: {len(trades)}")

        if trades:
            all_trades.extend(trades)
            pd.DataFrame(trades).to_csv(f"{OUTPUT_DIR}/{sym}_trades.csv", index=False)
            metrics = compute_metrics(trades)
            combined_metrics[sym] = metrics
            print(f"  Net Points: {metrics['net_points']:>8.2f}  Win Rate: {metrics['win_rate']:>5.1f}%  "
                  f"Trades: {metrics['total']}  PF: {metrics['profit_factor']}")
        print(f"  Time: {time.time() - start:.1f}s")

    if not all_trades:
        print("\nNo trades across any index.")
        return

    all_df = pd.DataFrame(all_trades)
    all_df.to_csv(f"{OUTPUT_DIR}/all_trades.csv", index=False)

    print(f"\n{'=' * 65}")
    print("COMBINED SUMMARY (ALL INDICES)")
    print(f"{'=' * 65}")

    total_metrics = compute_metrics(all_trades)
    print(f"\n{'Metric':<25s} {'Value':>15s}")
    print(f"{'-' * 42}")
    print(f"{'Total Trades':<25s} {total_metrics['total']:>15}")
    print(f"{'Wins / Losses':<25s} {total_metrics['wins']:>5} / {total_metrics['losses']:<5}")
    print(f"{'Win Rate':<25s} {total_metrics['win_rate']:>14.1f}%")
    print(f"{'Net Points':<25s} {total_metrics['net_points']:>15.2f}")
    print(f"{'Gross Profit':<25s} {total_metrics['gross_profit']:>15.2f}")
    print(f"{'Gross Loss':<25s} {total_metrics['gross_loss']:>15.2f}")
    print(f"{'Profit Factor':<25s} {total_metrics['profit_factor']:>15.2f}")
    print(f"{'Avg Win (pts)':<25s} {total_metrics['avg_win']:>15.2f}")
    print(f"{'Avg Loss (pts)':<25s} {total_metrics['avg_loss']:>15.2f}")
    print(f"{'Max DD (pts)':<25s} {total_metrics['max_dd']:>15.2f}")
    print(f"{'Max DD %':<25s} {total_metrics['max_dd_pct']:>14.1f}%")
    print(f"{'Sharpe Ratio':<25s} {total_metrics['sharpe']:>15.2f}")
    print(f"{'Total Charges (Rs)':<25s} {total_metrics['total_charges_rs']:>15}")

    print(f"\n{'=' * 65}")
    print("PER-INDEX BREAKDOWN")
    print(f"{'=' * 65}")
    print(f"{'Index':<12s} {'Trades':>7s} {'Net Pts':>9s} {'Win%':>7s} {'PF':>7s} {'AvgW':>8s} {'AvgL':>8s} {'DD':>9s} {'Charges':>8s}")
    print(f"{'-' * 80}")
    for sym in INDICES:
        m = combined_metrics.get(sym)
        if m:
            print(f"{sym:<12s} {m['total']:>7d} {m['net_points']:>9.2f} {m['win_rate']:>6.1f}% "
                  f"{m['profit_factor']:>6.2f} {m['avg_win']:>7.2f} {m['avg_loss']:>7.2f} "
                  f"{m['max_dd']:>8.2f} {m['total_charges_rs']:>7d}")

    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"  Trade books:  all_trades.csv, {', '.join(f'{s}_trades.csv' for s in INDICES if s in combined_metrics)}")


if __name__ == "__main__":
    main()
