"""ORB parameter sweeper - tests multiple combos on representative stocks"""
import sys, os, time, itertools, json
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
DATA_DIR = "nifty50_full_history"

# Representative stocks covering high/low performers
TEST_SYMBOLS = ["MARUTI", "RELIANCE", "TCS", "SBIN", "BAJAJFINSV"]

# Parameter grid (narrowed to efficient set)
OR_BARS_OPTIONS = [1, 2, 3]      # 15, 30, 45 min
SL_ATR_OPTIONS  = [1.5, 2.0, 2.5]
TP_RATIO_OPTIONS = [1.5, 2.0, 2.5]

# Import modified backtest function
from backtest_orb import (
    compute_atr, compute_charges, build_hourly_trend,
    get_trend_at, get_volume_avg, BROKERAGE_PER_ORDER as BRK
)

def backtest_with_params(symbol, data_dir, or_bars, sl_atr, tp_ratio, partial=True, brokerage=5):
    """Run ORB with custom parameters"""
    df15_path = f"{data_dir}/{symbol}_FIFTEEN_MINUTE.csv"
    df1_path = f"{data_dir}/{symbol}_ONE_MINUTE.csv"
    if not os.path.exists(df15_path) or not os.path.exists(df1_path):
        return None

    df15 = pd.read_csv(df15_path)
    df1 = pd.read_csv(df1_path)
    df15["datetime"] = pd.to_datetime(df15["datetime"])
    df1["datetime"] = pd.to_datetime(df1["datetime"])
    df15["date"] = df15["datetime"].dt.date
    df1["date"] = df1["datetime"].dt.date
    df15 = df15.sort_values("datetime").reset_index(drop=True)
    df1 = df1.sort_values("datetime").reset_index(drop=True)

    df15["atr"] = compute_atr(df15, 14)
    df15["vol_avg20"] = df15["volume"].rolling(20, min_periods=10).mean().shift(1)
    trend_df = build_hourly_trend(df15)

    trades = []
    for day, day_group in df15.groupby("date"):
        day_group = day_group.sort_values("datetime").reset_index(drop=True)
        if len(day_group) < or_bars:
            continue

        # OR from first N bars
        or_bars_df = day_group.iloc[:or_bars]
        or_high = or_bars_df["high"].max()
        or_low = or_bars_df["low"].min()
        or_end_time = or_bars_df["datetime"].max()

        remaining = day_group[day_group["datetime"] > or_end_time]
        if remaining.empty:
            continue

        # Find first breakout
        trigger = None
        for _, row in remaining.iterrows():
            if row["datetime"].hour >= 14:
                break

            long_t = row["close"] > or_high
            short_t = row["close"] < or_low
            if not (long_t or short_t):
                continue

            vol_ok = True
            if pd.notna(row["vol_avg20"]) and row["vol_avg20"] > 0:
                vol_ok = row["volume"] >= 1.3 * row["vol_avg20"]
            if not vol_ok:
                continue

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

            sl_distance = sl_atr * atr_val
            if tp == "LONG":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_ratio * sl_distance
                partial_level = entry_price + sl_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_ratio * sl_distance
                partial_level = entry_price - sl_distance

            trigger = {
                "datetime": row["datetime"], "type": tp,
                "entry_level": entry_price, "sl": sl_price,
                "tp": tp_price, "partial_level": partial_level if partial else None,
                "atr_used": round(atr_val, 2)
            }
            break

        if trigger is None:
            continue

        # 1-min fill
        t_dt = trigger["datetime"]
        next_bar_mask = df1["datetime"] > t_dt
        if not next_bar_mask.any():
            continue
        next_bar_dt = df1[next_bar_mask].iloc[0]["datetime"]

        eod = datetime.combine(day, datetime.max.time()).replace(hour=15, minute=25)
        eod_ts = pd.Timestamp(eod)
        # Match timezone of data
        data_tz = df1["datetime"].dt.tz
        if data_tz is not None:
            eod_ts = eod_ts.tz_localize(str(data_tz))
        mask = (df1["datetime"] >= next_bar_dt) & (df1["datetime"] <= eod_ts)
        scan = df1[mask].copy()
        if scan.empty:
            continue

        entry_p = trigger["entry_level"]
        entry_t = next_bar_dt
        risk_amt = abs(entry_p - trigger["sl"])

        partial_done = False
        exit_p1 = exit_t1 = None
        exit_p2 = exit_t2 = reason2 = None
        sl_for_remaining = trigger["sl"]

        for _, bar in scan.iterrows():
            if partial and not partial_done and trigger["partial_level"] is not None:
                if trigger["type"] == "LONG":
                    hit_partial = bar["high"] >= trigger["partial_level"]
                else:
                    hit_partial = bar["low"] <= trigger["partial_level"]
                if hit_partial:
                    partial_done = True
                    exit_p1 = trigger["partial_level"]
                    exit_t1 = bar["datetime"]
                    sl_for_remaining = entry_p

            if trigger["type"] == "LONG":
                sl_hit = bar["low"] <= sl_for_remaining
                tp_hit = bar["high"] >= trigger["tp"]
            else:
                sl_hit = bar["high"] >= sl_for_remaining
                tp_hit = bar["low"] <= trigger["tp"]

            if sl_hit:
                exit_p2 = sl_for_remaining; exit_t2 = bar["datetime"]; reason2 = "SL"; break
            elif tp_hit:
                exit_p2 = trigger["tp"]; exit_t2 = bar["datetime"]; reason2 = "TP"; break

        if not (partial_done or exit_p2 is not None):
            continue

        pq = rq = 0.5
        if partial_done and exit_p2 is not None:
            p1 = (exit_p1 - entry_p) if trigger["type"]=="LONG" else (entry_p - exit_p1)
            p2 = (exit_p2 - entry_p) if trigger["type"]=="LONG" else (entry_p - exit_p2)
            total_pnl = p1*pq + p2*rq
            r_m = round((p1/risk_amt*pq + p2/risk_amt*rq), 2) if risk_amt>0 else 0
            avg_exit = round(exit_p1*pq + exit_p2*rq, 2)
        elif partial_done and exit_p2 is None:
            p1 = (exit_p1 - entry_p) if trigger["type"]=="LONG" else (entry_p - exit_p1)
            total_pnl = p1*pq
            r_m = round((p1/risk_amt)*pq, 2) if risk_amt>0 else 0
            avg_exit = exit_p1
        elif not partial_done and exit_p2 is not None:
            total_pnl = (exit_p2 - entry_p) if trigger["type"]=="LONG" else (entry_p - exit_p2)
            r_m = round(total_pnl/risk_amt, 2) if risk_amt>0 else 0
            avg_exit = exit_p2
        else:
            continue

        charges = 0
        if brokerage > 0:
            # Recompute charges with custom brokerage
            turnover_buy = entry_p * 1
            turnover_sell = avg_exit * 1
            turnover_total = turnover_buy + turnover_sell
            brk_amt = brokerage * 2
            stt_total = turnover_sell * 0.001
            exchange_total = turnover_total * 0.00003
            sebi_total = turnover_total * 0.000001 * 2
            stamp = turnover_buy * 0.00003
            gst_total = (brk_amt + exchange_total) * 0.18
            charges = brk_amt + stt_total + exchange_total + sebi_total + stamp + gst_total

        net = round(total_pnl - charges, 2)

        trades.append({
            "pnl": round(total_pnl, 2), "net_pnl": net,
            "r": r_m, "charges": round(charges, 2), "partial": partial_done
        })

    return trades if trades else None

def compute_metrics(trades):
    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc/total*100, 2) if total else 0
    np_ = round(df["net_pnl"].sum(), 2)
    avg_r = round(df["r"].mean(), 2)
    partials = int(df["partial"].sum()) if "partial" in df else 0
    charges = round(df["charges"].sum(), 2)
    # Without charges
    raw_pnl = round(df["pnl"].sum(), 2) if "pnl" in df else np_
    return {
        "trades": total, "wins": wc, "losses": lc,
        "win_rate": wr, "net_pnl": np_, "raw_pnl": raw_pnl,
        "avg_r": avg_r, "partials": partials, "charges": charges
    }

# ─── SWEEP ───
print("=" * 100)
print("ORB PARAMETER SWEEP - 5 Representative Stocks")
print(f"Grid: OR={OR_BARS_OPTIONS} bars | SL={SL_ATR_OPTIONS}xATR | TP={TP_RATIO_OPTIONS}xSL")
print(f"Total combos: {len(OR_BARS_OPTIONS) * len(SL_ATR_OPTIONS) * len(TP_RATIO_OPTIONS)} per stock")
print("=" * 100)

results = []
total_runs = len(TEST_SYMBOLS) * len(OR_BARS_OPTIONS) * len(SL_ATR_OPTIONS) * len(TP_RATIO_OPTIONS)
run_count = 0

for sym in TEST_SYMBOLS:
    for or_bars in OR_BARS_OPTIONS:
        for sl_atr in SL_ATR_OPTIONS:
            for tp_ratio in TP_RATIO_OPTIONS:
                run_count += 1
                start_t = time.time()
                trades = backtest_with_params(sym, DATA_DIR, or_bars, sl_atr, tp_ratio, 
                                            partial=True, brokerage=5)
                elapsed = time.time() - start_t

                if trades:
                    m = compute_metrics(trades)
                    results.append({
                        "symbol": sym, "or_bars": or_bars, "sl_atr": sl_atr, "tp_ratio": tp_ratio,
                        **m
                    })
                    print(f"[{run_count}/{total_runs}] {sym}: OR={or_bars} SL={sl_atr}x TP={tp_ratio}x "
                          f"Trades={m['trades']} WR={m['win_rate']:.1f}% "
                          f"Net={m['net_pnl']:,.0f} Raw={m['raw_pnl']:,.0f} "
                          f"AvgR={m['avg_r']:.2f} Charges={m['charges']:,.0f} [{elapsed:.0f}s]")
                else:
                    print(f"[{run_count}/{total_runs}] {sym}: OR={or_bars} SL={sl_atr}x TP={tp_ratio}x -> NO TRADES [{elapsed:.0f}s]")
                sys.stdout.flush()

# ─── REPORT ───
print(f"\n{'='*100}")
print("SWEEP COMPLETE - TOP 20 COMBOS (by Net P&L)")
print(f"{'='*100}")

# Aggregate across stocks for each param combo
from collections import defaultdict
combo_scores = defaultdict(lambda: {"trades": 0, "net_pnl": 0, "raw_pnl": 0, "avg_r": 0, "count": 0})
for r in results:
    key = (r["or_bars"], r["sl_atr"], r["tp_ratio"])
    combo_scores[key]["trades"] += r["trades"]
    combo_scores[key]["net_pnl"] += r["net_pnl"]
    combo_scores[key]["raw_pnl"] += r["raw_pnl"]
    combo_scores[key]["avg_r"] += r["avg_r"]
    combo_scores[key]["count"] += 1

combo_list = []
for key, val in combo_scores.items():
    combo_list.append({
        "or": key[0], "sl": key[1], "tp": key[2],
        "total_net": round(val["net_pnl"], 2),
        "total_raw": round(val["raw_pnl"], 2),
        "total_trades": val["trades"],
        "avg_avg_r": round(val["avg_r"] / val["count"], 2) if val["count"] > 0 else 0
    })

combo_list.sort(key=lambda x: x["total_net"], reverse=True)

print(f"\n{'#':>3} {'OR':>4} {'SLx':>4} {'TPx':>4} {'Net P&L':>10} {'Raw P&L':>10} {'Trades':>7} {'Avg R':>6}")
print(f"{'─'*55}")
for i, c in enumerate(combo_list[:20], 1):
    print(f"{i:>3} {c['or']:>4} {c['sl']:>4.1f} {c['tp']:>4.1f} Rs{c['total_net']:>7,.0f} Rs{c['total_raw']:>7,.0f} {c['total_trades']:>7} {c['avg_avg_r']:>5.2f}")

# Also find best per stock
print(f"\n{'='*100}")
print("BEST PARAM COMBO PER STOCK (by Net P&L)")
print(f"{'='*100}")
for sym in TEST_SYMBOLS:
    sym_results = [r for r in results if r["symbol"] == sym]
    sym_results.sort(key=lambda x: x["net_pnl"], reverse=True)
    if sym_results:
        b = sym_results[0]
        print(f"  {sym:12s}: OR={b['or_bars']}b SL={b['sl_atr']}x TP={b['tp_ratio']}x  "
              f"Trades={b['trades']} Net=Rs{b['net_pnl']:,.0f} Raw=Rs{b['raw_pnl']:,.0f} AvgR={b['avg_r']:.2f}")
    else:
        print(f"  {sym:12s}: NO TRADES")

print(f"\nDone. {len(results)} stock-param combinations tested.")
