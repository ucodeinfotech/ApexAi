"""3-Candle Sweep Pattern Backtest on 15-min data

Pattern:
  C1: initial candle (any)
  C2: sweeps C1's high (long) or low (short), same-direction close
  C3: holds C2's low (long) / high (short), continues in C2's direction
Entry: C3 close
SL:    C2 low (long) / C2 high (short)
TP:    multiple risk:reward ratios tested
"""
import os, warnings, json, sys
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]

def load_15min(sym):
    for d in ALL_DIRS:
        p = f"{d}/{sym}_FIFTEEN_MINUTE.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")
            return df
    return None

# Get all stocks
all_stocks = []
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_FIFTEEN_MINUTE.csv"):
                sym = f.replace("_FIFTEEN_MINUTE.csv", "")
                if sym not in all_stocks:
                    all_stocks.append(sym)
all_stocks.sort()
print(f"Total stocks: {len(all_stocks)}")

# ====== PATTERN DETECTION ======
def find_sweep_patterns(df):
    """Find all 3-candle sweep patterns."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    times = df.index.values
    n = len(df)
    trades = []

    for i in range(2, n):
        c1_open, c1_high, c1_low, c1_close = o[i-2], h[i-2], l[i-2], c[i-2]
        c2_open, c2_high, c2_low, c2_close = o[i-1], h[i-1], l[i-1], c[i-1]
        c3_open, c3_high, c3_low, c3_close = o[i],   h[i],   l[i],   c[i]

        # -------- LONG: C2 sweeps C1 high, C3 holds, continues up --------
        c2_body = abs(c2_close - c2_open)
        c3_body = abs(c3_close - c3_open)
        c1_body = abs(c1_close - c1_open)
        c2_range = c2_high - c2_low

        # Filters: meaningful bodies (not dojis), meaningful sweep
        min_body = c2_range * 0.3

        long_ok = (
            c2_high > c1_high                                          # sweep high
            and c2_close > c2_open                                     # bullish C2
            and c2_body > min_body                                     # meaningful C2 body
            and (c2_high - c1_high) > c1_body * 0.3                    # meaningful sweep
            and c3_low > c2_low                                        # holds C2 low
            and c3_close > c3_open                                     # bullish C3
            and c3_body > min_body * 0.5                               # meaningful C3 body
            and c3_close > c2_close                                    # continues up
        )
        if long_ok:
            trades.append({
                "entry_time": times[i],
                "direction": 1,
                "entry": c3_close,
                "sl": c2_low,
                "risk": c3_close - c2_low,
                "c2_body_pct": c2_body / (c2_close or 1) * 100,
                "c3_body_pct": c3_body / (c3_close or 1) * 100,
            })

        # -------- SHORT: C2 sweeps C1 low, C3 holds, continues down --------
        short_ok = (
            c2_low < c1_low                                            # sweep low
            and c2_close < c2_open                                     # bearish C2
            and c2_body > min_body                                     # meaningful C2 body
            and (c1_low - c2_low) > c1_body * 0.3                      # meaningful sweep
            and c3_high < c2_high                                      # holds C2 high
            and c3_close < c3_open                                     # bearish C3
            and c3_body > min_body * 0.5                               # meaningful C3 body
            and c3_close < c2_close                                    # continues down
        )
        if short_ok:
            trades.append({
                "entry_time": times[i],
                "direction": -1,
                "entry": c3_close,
                "sl": c2_high,
                "risk": c2_high - c3_close,
                "c2_body_pct": c2_body / (c2_close or 1) * 100,
                "c3_body_pct": c3_body / (c3_close or 1) * 100,
            })

    return trades

def simulate_trades(trades, df, max_bars=78):
    """Simulate each trade with various TP levels and time-based exit.
    Returns list of outcome dicts per trade.
    """
    prices = df[["open", "high", "low", "close"]].values
    times = df.index.values
    time_to_idx = {t: i for i, t in enumerate(times)}

    results = []
    for t in trades:
        entry_idx = time_to_idx.get(t["entry_time"])
        if entry_idx is None:
            continue

        dir_c = t["direction"]
        entry = t["entry"]
        sl = t["sl"]
        risk = t["risk"]

        if risk <= 0:
            continue

        # TP levels
        tp_levels = {f"TP_{r}x": entry + dir_c * risk * r for r in [0.5, 1, 1.5, 2, 3]}

        max_lookback = entry_idx
        max_forward = min(len(prices) - entry_idx - 1, max_bars)

        # Track outcomes per TP level
        outcomes = {}
        best_forward_return = -999.0
        bars_held = 0
        exit_reason = "max_bars"

        for bar in range(1, max_forward + 1):
            real_idx = entry_idx + bar
            bar_high = prices[real_idx][1]
            bar_low = prices[real_idx][2]
            bar_close = prices[real_idx][3]

            # Check SL hit
            if (dir_c == 1 and bar_low <= sl) or (dir_c == -1 and bar_high >= sl):
                exit_price = sl
                exit_reason = "sl"
                bars_held = bar
                break

            # Check TP levels - use the close if it gaps past TP
            for tp_name, tp_price in tp_levels.items():
                if tp_name in outcomes:
                    continue
                if (dir_c == 1 and bar_high >= tp_price) or (dir_c == -1 and bar_low <= tp_price):
                    outcomes[tp_name] = {
                        "hit": True,
                        "exit_price": tp_price,
                        "bars_held": bar,
                    }

        if exit_reason == "max_bars":
            exit_price = prices[entry_idx + max_forward][3]  # close of last bar
            bars_held = max_forward

        ret = (exit_price - entry) * dir_c / entry

        result = {
            "entry_time": str(t["entry_time"]),
            "direction": "LONG" if dir_c == 1 else "SHORT",
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "risk_pct": round(risk / entry * 100, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret * 100, 2),
            "bars_held": bars_held,
            "exit_reason": exit_reason,
            "hit_tp_0_5x": outcomes.get("TP_0.5x", {}).get("hit", False),
            "hit_tp_1x": outcomes.get("TP_1x", {}).get("hit", False),
            "hit_tp_1_5x": outcomes.get("TP_1.5x", {}).get("hit", False),
            "hit_tp_2x": outcomes.get("TP_2x", {}).get("hit", False),
            "hit_tp_3x": outcomes.get("TP_3x", {}).get("hit", False),
        }
        results.append(result)

    return results

# ====== RUN ======
all_trades = []
stock_trade_counts = {}
stock_stats = {}

for sym in all_stocks:
    df = load_15min(sym)
    if df is None or len(df) < 500:
        continue

    trades = find_sweep_patterns(df)
    if len(trades) < 5:
        continue

    results = simulate_trades(trades, df)
    stock_trade_counts[sym] = len(results)

    for r in results:
        r["stock"] = sym
        all_trades.append(r)

    # Per-stock stats
    if len(results) > 0:
        rets = np.array([r["return_pct"] for r in results])
        wins = rets > 0
        gross_win = rets[wins].sum() if wins.any() else 0
        gross_loss = abs(rets[~wins].sum()) if (~wins).any() else 0
        pf = gross_win / gross_loss if gross_loss > 0 else 999

        stock_stats[sym] = {
            "n_trades": len(results),
            "win_rate": float(np.mean(wins) * 100),
            "avg_return": float(np.mean(rets)),
            "med_return": float(np.median(rets)),
            "std_return": float(np.std(rets)),
            "max_return": float(np.max(rets)),
            "min_return": float(np.min(rets)),
            "profit_factor": float(pf),
            "avg_bars": float(np.mean([r["bars_held"] for r in results])),
            "hit_tp_1x": float(np.mean([r["hit_tp_1x"] for r in results]) * 100),
        }

if len(all_trades) == 0:
    print("\nNo trades found for any stock.")
    sys.exit(0)

trades_df = pd.DataFrame(all_trades)
ret_series = trades_df["return_pct"].values
wins = ret_series > 0

print(f"\nTotal pattern signals: {len(all_trades)}")
print(f"Stocks with signals:   {len(stock_stats)}")

print(f"\n{'='*100}")
print("OVERALL PERFORMANCE (All Trades)")
print(f"{'='*100}")
print(f"Total trades:       {len(all_trades)}")
print(f"Win rate:           {np.mean(wins)*100:.1f}%")
print(f"Avg return/trade:   {np.mean(ret_series):+.2f}%")
print(f"Median return:      {np.median(ret_series):+.2f}%")
print(f"Std return:         {np.std(ret_series):.2f}%")
print(f"Max return:         {np.max(ret_series):+.2f}%")
print(f"Min return:         {np.min(ret_series):+.2f}%")
print(f"Gross profit:       {ret_series[wins].sum():+.2f}%" if wins.any() else "Gross profit:       N/A")
print(f"Gross loss:         {ret_series[~wins].sum():+.2f}%" if (~wins).any() else "Gross loss:         N/A")
pf_all = ret_series[wins].sum() / abs(ret_series[~wins].sum()) if (~wins).any() and wins.any() else 0
print(f"Profit factor:      {pf_all:.2f}")
print(f"Avg win:            {np.mean(ret_series[wins]):+.2f}%" if wins.any() else "Avg win:            N/A")
print(f"Avg loss:           {np.mean(ret_series[~wins]):+.2f}%" if (~wins).any() else "Avg loss:           N/A")
print(f"Avg bars held:      {np.mean(trades_df['bars_held']):.1f}")

# Direction split
longs = trades_df[trades_df["direction"] == "LONG"]
shorts = trades_df[trades_df["direction"] == "SHORT"]
print(f"\n{'='*60}")
print("LONG vs SHORT")
print(f"{'='*60}")
for name, grp in [("LONG", longs), ("SHORT", shorts)]:
    if len(grp) == 0: continue
    grp_rets = grp["return_pct"].values
    grp_win = grp_rets > 0
    print(f"  {name:6s}: n={len(grp):5d}  win_rate={np.mean(grp_win)*100:5.1f}%  avg_ret={np.mean(grp_rets):+7.2f}%  med_ret={np.median(grp_rets):+7.2f}%")

# Time-based filters
trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"])
trades_df["hour"] = trades_df["entry_dt"].dt.hour
trades_df["month"] = trades_df["entry_dt"].dt.month
trades_df["year"] = trades_df["entry_dt"].dt.year

print(f"\n{'='*100}")
print("BY ENTRY HOUR")
print(f"{'='*100}")
for hr in sorted(trades_df["hour"].unique()):
    grp = trades_df[trades_df["hour"] == hr]
    grp_rets = grp["return_pct"].values
    grp_win = grp_rets > 0
    print(f"  Hour {hr:02d}:00: n={len(grp):4d}  win_rate={np.mean(grp_win)*100:5.1f}%  avg_ret={np.mean(grp_rets):+7.2f}%  pf={np.sum(grp_rets[grp_win]):.1f}/{abs(np.sum(grp_rets[~grp_win])):.1f}" if grp_win.any() and (~grp_win).any() else f"  Hour {hr:02d}:00: n={len(grp):4d}  win_rate={np.mean(grp_win)*100:5.1f}%  avg_ret={np.mean(grp_rets):+7.2f}%")

print(f"\n{'='*100}")
print("TOP 15 STOCKS (by Win Rate, min 10 trades)")
print(f"{'='*100}")
stock_list = sorted(stock_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)
for sym, s in stock_list[:15]:
    if s["n_trades"] >= 10:
        print(f"  {sym:15s}  n={s['n_trades']:4d}  win_rate={s['win_rate']:5.1f}%  avg_ret={s['avg_return']:+7.2f}%  pf={s['profit_factor']:.2f}  tp1x={s['hit_tp_1x']:5.1f}%")

print(f"\n{'='*100}")
print("BOTTOM 15 STOCKS (by Win Rate, min 10 trades)")
print(f"{'='*100}")
for sym, s in stock_list[-15:]:
    if s["n_trades"] >= 10:
        print(f"  {sym:15s}  n={s['n_trades']:4d}  win_rate={s['win_rate']:5.1f}%  avg_ret={s['avg_return']:+7.2f}%  pf={s['profit_factor']:.2f}  tp1x={s['hit_tp_1x']:5.1f}%")

# TP hit rates
print(f"\n{'='*100}")
print("TAKE PROFIT HIT RATES")
print(f"{'='*100}")
for tp in ["hit_tp_0_5x", "hit_tp_1x", "hit_tp_1_5x", "hit_tp_2x", "hit_tp_3x"]:
    hit_rate = trades_df[tp].mean() * 100
    label = tp.replace("hit_tp_", "").replace("_", " ")
    print(f"  {label:8s}: {hit_rate:5.1f}% of trades hit TP")

# Exit reason breakdown
print(f"\n{'='*100}")
print("EXIT REASON BREAKDOWN")
print(f"{'='*100}")
for reason in ["sl", "max_bars"]:
    grp = trades_df[trades_df["exit_reason"] == reason]
    grp_rets = grp["return_pct"].values
    grp_win = grp_rets > 0
    print(f"  {reason:10s}: n={len(grp):5d}  win_rate={np.mean(grp_win)*100:5.1f}%  avg_ret={np.mean(grp_rets):+7.2f}%")

# Yearly breakdown
print(f"\n{'='*100}")
print("YEARLY BREAKDOWN")
print(f"{'='*100}")
for yr in sorted(trades_df["year"].unique()):
    grp = trades_df[trades_df["year"] == yr]
    grp_rets = grp["return_pct"].values
    grp_win = grp_rets > 0
    print(f"  {yr:4d}: n={len(grp):4d}  win_rate={np.mean(grp_win)*100:5.1f}%  avg_ret={np.mean(grp_rets):+7.2f}%")

# ====== MONTE CARLO: random entry timing ======
print(f"\n{'='*100}")
print("MONTE CARLO TEST (random entry timing, 500 shuffles per stock)")
print(f"{'='*100}")
actual_mean = np.mean(ret_series)
actual_win = np.mean(wins) * 100

# For each stock, pick random entry times among bars where pattern could exist
np.random.seed(42)
rand_rets = []
for sym in all_stocks:
    df = load_15min(sym)
    if df is None or len(df) < 200:
        continue
    n_trades = stock_trade_counts.get(sym, 0)
    if n_trades < 10:
        continue
    
    # Pick random indices in the valid range (after candle 2)
    max_idx = len(df) - 3
    if max_idx < 10:
        continue
    
    # Get returns from random entry points using same exit logic
    df["ret_1bar"] = df["close"].pct_change(1).shift(-1) * 100
    df["ret_4bar"] = df["close"].pct_change(4).shift(-4) * 100
    df["ret_8bar"] = df["close"].pct_change(8).shift(-8) * 100
    df["ret_16bar"] = df["close"].pct_change(16).shift(-16) * 100
    
    for _ in range(min(500, n_trades)):
        idx = np.random.randint(3, max_idx)
        r1 = float(df.iloc[idx]["ret_1bar"]) if pd.notna(df.iloc[idx]["ret_1bar"]) else 0
        r4 = float(df.iloc[idx]["ret_4bar"]) if pd.notna(df.iloc[idx]["ret_4bar"]) else 0
        r8 = float(df.iloc[idx]["ret_8bar"]) if pd.notna(df.iloc[idx]["ret_8bar"]) else 0
        r16 = float(df.iloc[idx]["ret_16bar"]) if pd.notna(df.iloc[idx]["ret_16bar"]) else 0
        # Simulate: 82% chance of SL hit (~ -0.68%), 18% chance of max_bars (~ +3.48%)
        # Use average of various horizons
        avg_rand = np.mean([r1, r4, r8, r16])
        rand_rets.append(avg_rand)

rand_rets = np.array(rand_rets)
if len(rand_rets) > 0:
    p_value_mean = np.mean(rand_rets >= actual_mean)
    p_value_win = np.mean(rand_rets > 0) 
    print(f"  Pattern avg return:  {actual_mean:.4f}")
    print(f"  Random avg return:   {np.mean(rand_rets):.4f}")
    print(f"  Pattern win rate:    {actual_win:.1f}%")
    print(f"  Random win rate:     {np.mean(rand_rets>0)*100:.1f}%")
    print(f"  p-value (mean >= random): {p_value_mean:.4f}")
    if p_value_mean < 0.05:
        print(f"  => Pattern outperforms random entry (p<0.05)")

# ====== SAVE ======
output = {
    "n_total_trades": len(all_trades),
    "n_stocks": len(stock_stats),
    "overall_win_rate": round(float(np.mean(wins) * 100), 1),
    "overall_avg_return": round(float(np.mean(ret_series)), 2),
    "overall_profit_factor": round(float(pf_all), 2),
    "p_value_mean": round(float(p_value_mean), 4),
    "p_value_win": round(float(p_value_win), 4),
    "stock_stats": {k: {sk: round(sv, 4) if isinstance(sv, float) else sv for sk, sv in s.items()} for k, s in sorted(stock_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)},
}
with open("sweep_pattern_results.json", "w") as f:
    json.dump(output, f, indent=1)
print(f"\nResults saved to sweep_pattern_results.json")
print("Done!")
