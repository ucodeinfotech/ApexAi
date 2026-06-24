"""
Opening Range Breakout (ORB) Backtester
Strategy: First 30-min OR → breakout on 15-min close → 1-min exit sim
"""
import pandas as pd
import numpy as np
import os, time, sys
from datetime import datetime, timedelta

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BROKERAGE_PER_ORDER = 5
STT = 0.001
EXCHANGE_TC = 0.00003
SEBI_TC = 0.000001
GST = 0.18
STAMP_DUTY = 0.00003
ATR_MULTIPLIER = 2.5       # wider SL to let trades breathe
ATR_PERIOD = 14
VOLUME_MULTIPLIER = 1.3
TP_RATIO = 1.5             # 1:1.5 R:R instead of 1:2
PARTIAL_PROFIT = True

EXCLUDE = {"NIFTY50", "BANKNIFTY", "SENSEX"}

# OR parameters
OR_BARS = 3          # 45-min opening range (3 x 15min)
NO_ENTRY_AFTER = 14  # 2 PM
EOD_TIME = 15        # 3 PM

def compute_atr(df, period=14):
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df[["high","low","prev_close"]].apply(
        lambda r: max(r["high"]-r["low"], abs(r["high"]-r["prev_close"]), abs(r["low"]-r["prev_close"])), axis=1
    )
    df["atr"] = df["tr"].rolling(window=period, min_periods=period).mean()
    return df["atr"]

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

def build_hourly_trend(df15):
    df = df15.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    hourly = df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    hourly = hourly.dropna()
    hourly["ema20"] = hourly["close"].ewm(span=20, adjust=False).mean()
    hourly["trend"] = np.where(hourly["close"] >= hourly["ema20"], "UP", "DOWN")
    return hourly.reset_index()[["datetime","trend"]]

def get_trend_at(dt, trend_df):
    mask = trend_df["datetime"] <= pd.Timestamp(dt)
    if mask.any():
        return trend_df[mask].iloc[-1]["trend"]
    return "UP"

def get_volume_avg(df15):
    return df15["volume"].rolling(20, min_periods=10).mean().shift(1)

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

    # Precompute ATR on 15-min
    df15["atr"] = compute_atr(df15, ATR_PERIOD)
    # Volume avg
    df15["vol_avg20"] = get_volume_avg(df15)

    # Precompute 1-hr trend
    trend_df = build_hourly_trend(df15)

    # Build per-day OR and find breakouts
    trades = []

    for day, day_group in df15.groupby("date"):
        day_group = day_group.sort_values("datetime").reset_index(drop=True)
        
        # Opening range: first N bars (N=3 = 9:15-10:00 = 45 min)
        or_bars = day_group.iloc[:OR_BARS]
        if len(or_bars) < 1:
            continue

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_end_time = or_bars["datetime"].max()

        # Remaining bars after OR period
        remaining = day_group[day_group["datetime"] > or_end_time]
        if remaining.empty:
            continue

        # Find first breakout (long or short, first trigger wins)
        trigger = None
        for _, row in remaining.iterrows():
            # Time check
            hour = row["datetime"].hour
            if hour >= NO_ENTRY_AFTER:
                break

            long_t = row["close"] > or_high
            short_t = row["close"] < or_low
            if not (long_t or short_t):
                continue

            # Volume filter
            vol_ok = True
            if pd.notna(row["vol_avg20"]) and row["vol_avg20"] > 0:
                vol_ok = row["volume"] >= VOLUME_MULTIPLIER * row["vol_avg20"]
            if not vol_ok:
                continue

            # Trend filter
            trend = get_trend_at(row["datetime"], trend_df)
            if long_t and trend == "DOWN":
                continue
            if short_t and trend == "UP":
                continue

            tp = "LONG" if long_t else "SHORT"
            entry_price = row["close"]
            atr_val = row.get("atr", np.nan)
            if pd.isna(atr_val) or atr_val <= 0:
                atr_val = (row["high"] - row["low"]) * 0.6

            sl_distance = ATR_MULTIPLIER * atr_val
            if tp == "LONG":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + TP_RATIO * sl_distance
                partial_level = entry_price + sl_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - TP_RATIO * sl_distance
                partial_level = entry_price - sl_distance

            trigger = {
                "datetime": row["datetime"], "date": day,
                "type": tp,
                "entry_level": entry_price,
                "sl": sl_price, "tp": tp_price,
                "partial_level": partial_level,
                "or_high": or_high, "or_low": or_low,
                "atr_used": round(atr_val, 2),
                "trend": trend
            }
            break  # first breakout only

        if trigger is None:
            continue

        # Fill on 1-min bars
        t_dt = trigger["datetime"]
        t_date = trigger["date"]
        # Entry on next 1-min bar
        next_bar_mask = df1["datetime"] > t_dt
        if not next_bar_mask.any():
            continue
        next_bar_dt = df1[next_bar_mask].iloc[0]["datetime"]

        window_end = datetime.combine(t_date, datetime.max.time()).replace(hour=EOD_TIME, minute=25)
        mask = (df1["datetime"] >= next_bar_dt) & (df1["datetime"] <= pd.Timestamp(window_end, tz=df1["datetime"].dt.tz))
        scan = df1[mask].copy()
        if scan.empty:
            continue

        entry_p = trigger["entry_level"]
        entry_t = next_bar_dt
        risk_amt = abs(entry_p - trigger["sl"])

        partial_done = False
        exit_p1 = None; exit_t1 = None
        exit_p2 = None; exit_t2 = None; reason2 = None
        sl_for_remaining = trigger["sl"]

        for _, bar in scan.iterrows():
            if not partial_done:
                if trigger["type"] == "LONG":
                    hit_partial = bar["high"] >= trigger["partial_level"]
                else:
                    hit_partial = bar["low"] <= trigger["partial_level"]

                if hit_partial:
                    partial_done = True
                    exit_p1 = trigger["partial_level"]
                    exit_t1 = bar["datetime"]
                    sl_for_remaining = entry_p

            # Check SL/TP for remaining
            if trigger["type"] == "LONG":
                sl_hit = bar["low"] <= sl_for_remaining
                tp_hit = bar["high"] >= trigger["tp"]
            else:
                sl_hit = bar["high"] >= sl_for_remaining
                tp_hit = bar["low"] <= trigger["tp"]

            if sl_hit:
                exit_p2 = sl_for_remaining
                exit_t2 = bar["datetime"]
                reason2 = "SL"
                break
            elif tp_hit:
                exit_p2 = trigger["tp"]
                exit_t2 = bar["datetime"]
                reason2 = "TP"
                break

        # Calculate results
        partial_qty = 0.5
        remaining_qty = 0.5

        if partial_done and exit_p2 is not None:
            pnl1 = (exit_p1 - entry_p) if trigger["type"] == "LONG" else (entry_p - exit_p1)
            pnl2 = (exit_p2 - entry_p) if trigger["type"] == "LONG" else (entry_p - exit_p2)
            total_pnl = pnl1 * partial_qty + pnl2 * remaining_qty
            r_m = round((pnl1/risk_amt*partial_qty + pnl2/risk_amt*remaining_qty), 2) if risk_amt > 0 else 0
            avg_exit = round(exit_p1 * partial_qty + exit_p2 * remaining_qty, 2)
            reason = f"PARTIAL+{reason2}"
            exit_for_charges = avg_exit
        elif partial_done and exit_p2 is None:
            pnl1 = (exit_p1 - entry_p) if trigger["type"] == "LONG" else (entry_p - exit_p1)
            total_pnl = pnl1 * partial_qty
            r_m = round((pnl1/risk_amt)*partial_qty, 2) if risk_amt > 0 else 0
            reason = "PARTIAL_ONLY"
            avg_exit = exit_p1
            exit_for_charges = exit_p1
        elif not partial_done and exit_p2 is not None:
            total_pnl = (exit_p2 - entry_p) if trigger["type"] == "LONG" else (entry_p - exit_p2)
            r_m = round(total_pnl/risk_amt, 2) if risk_amt > 0 else 0
            reason = reason2
            avg_exit = exit_p2
            exit_for_charges = exit_p2
        else:
            continue

        charges = compute_charges(entry_p, exit_for_charges)
        net = round(total_pnl - charges, 2)

        trades.append({
            "symbol": symbol, "date": str(day), "type": trigger["type"],
            "entry_time": str(entry_t), "exit_time": str(exit_t2 or exit_t1),
            "trigger_time": str(t_dt),
            "entry": round(entry_p,2), "exit": round(avg_exit,2),
            "sl": round(trigger["sl"],2), "tp": round(trigger["tp"],2),
            "or_high": round(trigger["or_high"],2), "or_low": round(trigger["or_low"],2),
            "reason": reason, "pnl": round(total_pnl,2), "net_pnl": net,
            "r": r_m, "charges": round(charges,2),
            "partial": partial_done, "atr": trigger["atr_used"],
            "trend": trigger["trend"]
        })

    if not trades:
        return None, "No trades"

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
    print(f"  {'':15s}  Max DD: Rs{mdd:>8,.2f} ({mdd_p}%)  Sharpe: {sharpe}")
    print(f"  {'':15s}  Charges: Rs{tcharges:>8,.2f}")
    print(f"{'='*55}")

    return {
        "symbol": symbol, "trades": total, "wins": wc, "losses": lc,
        "win_rate": wr, "net_pnl": np_, "profit_factor": pf,
        "avg_win": avg_w, "avg_loss": avg_l, "avg_r": avg_r,
        "max_dd": mdd, "max_dd_pct": mdd_p,
        "sharpe": sharpe, "charges": tcharges
    }

def main():
    print("=" * 60)
    print("OPENING RANGE BREAKOUT (ORB) BACKTESTER")
    print(f"OR: {OR_BARS*15}min ({OR_BARS} bars) | SL: {ATR_MULTIPLIER}xATR | TP: {TP_RATIO}:1 | Rs{BROKERAGE_PER_ORDER}")
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

        if trades:
            pd.DataFrame(trades).to_csv(f"{OUTPUT_DIR}/{sym}_orb_trades.csv", index=False)

    print(f"\n\n{'='*60}")
    print(f"COMBINED RESULTS - ALL {len(symbols)} STOCKS")
    print(f"{'='*60}")

    if not all_trades:
        print("No trades generated across any stock.")
        return

    combined_df = pd.DataFrame(all_trades)
    combined_df.to_csv(f"{OUTPUT_DIR}/all_orb_trades.csv", index=False)

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

    combined_sorted = combined_df.sort_values("exit_time").reset_index(drop=True)
    combined_sorted["cum"] = combined_sorted["net_pnl"].cumsum()
    combined_sorted["peak"] = combined_sorted["cum"].cummax()
    combined_sorted["dd"] = combined_sorted["peak"] - combined_sorted["cum"]
    mdd = round(combined_sorted["dd"].max(), 2)
    mdd_p = round(mdd / combined_sorted["peak"].max() * 100, 2) if combined_sorted["peak"].max() > 0 else 0

    sharpe = round(combined_df["r"].mean() / combined_df["r"].std() * np.sqrt(total_trades), 2) if combined_df["r"].std() > 0 else 0
    total_charges = round(combined_df["charges"].sum(), 2)
    partials = combined_df["partial"].sum()

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
    print(f"  Max Drawdown:      Rs{mdd:>8,.2f} ({mdd_p}%)")
    print(f"  Sharpe Ratio:      {sharpe:>8.2f}")
    print(f"  Total Charges:     Rs{total_charges:>8,.2f}")
    print(f"  Partial Booked:    {int(partials):>8} ({partials/total_trades*100:.1f}%)")
    print(f"{'─'*55}")

    sorted_results = sorted(all_results, key=lambda x: x["net_pnl"], reverse=True)
    print(f"\n{'='*60}")
    print("STOCK RANKING")
    print(f"{'='*60}")
    print(f"\n{'Rank':>4s} {'Symbol':18s} {'Trades':>7s} {'Net P&L':>10s} {'Win%':>7s} {'Avg R':>7s} {'PF':>7s} {'Sharpe':>7s}")
    print(f"{'─'*70}")
    for rank, r in enumerate(sorted_results, 1):
        print(f"{rank:>4d} {r['symbol']:18s} {r['trades']:>7d} Rs{r['net_pnl']:>7,.2f} {r['win_rate']:>6.1f}% {r['avg_r']:>6.2f} {r['profit_factor']:>6.2f} {r['sharpe']:>6.2f}")

    profitable = [r for r in sorted_results if r["net_pnl"] > 0]
    losing = [r for r in sorted_results if r["net_pnl"] <= 0]
    print(f"\n  Profitable stocks: {len(profitable)}/{len(sorted_results)}")
    print(f"  Losing stocks:     {len(losing)}/{len(sorted_results)}")
    if sorted_results:
        print(f"  Best: {sorted_results[0]['symbol']} (Rs{sorted_results[0]['net_pnl']:.2f}, WR: {sorted_results[0]['win_rate']}%)")
        print(f"  Worst: {sorted_results[-1]['symbol']} (Rs{sorted_results[-1]['net_pnl']:.2f}, WR: {sorted_results[-1]['win_rate']}%)")

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
