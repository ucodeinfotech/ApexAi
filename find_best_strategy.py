"""Strategy discovery: test 10+ trading strategies across all 175 stocks"""
import os, warnings, numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]

def load_daily(sym):
    for d in ALL_DIRS:
        p = f"{d}/{sym}_ONE_DAY.csv"
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
            if f.endswith("_ONE_DAY.csv"):
                sym = f.replace("_ONE_DAY.csv", "")
                if sym not in all_stocks:
                    all_stocks.append(sym)
all_stocks.sort()
print(f"Total stocks: {len(all_stocks)}")

def backtest(df, strategy_fn, params=None):
    """Generic backtester: strategy_fn(df) -> series of positions (-1, 0, +1)"""
    if df is None or len(df) < 100:
        return None
    positions = strategy_fn(df, params)
    if positions is None or len(positions) < 10:
        return None
    
    # Both use datetime index now - positions from strategy uses df's index
    closes = df["close"].values
    pos = positions.values
    
    # Ensure same length
    n_min = min(len(closes), len(pos))
    closes = closes[:n_min]
    pos = pos[:n_min]
    
    # Simple daily returns: (close_t - close_{t-1}) / close_{t-1}
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]
    daily_ret = (closes - prev_closes) / prev_closes
    daily_ret = daily_ret[1:]  # drop first (no prev)
    pos_aligned = pos[:-1]  # use today's position on tomorrow's return
    
    strategy_ret = daily_ret * pos_aligned
    bh_ret = daily_ret  # always long
    
    n = len(strategy_ret)
    if n < 20 or np.std(strategy_ret) == 0:
        return None
    
    total_s = np.prod(1 + strategy_ret) - 1
    total_bh = np.prod(1 + bh_ret) - 1
    ann_s = (1 + total_s) ** (252 / n) - 1
    ann_bh = (1 + total_bh) ** (252 / n) - 1
    sharpe_s = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) > 0 else 0
    sharpe_bh = np.mean(bh_ret) / np.std(bh_ret) * np.sqrt(252) if np.std(bh_ret) > 0 else 0
    max_dd_s = np.min((1 + np.cumsum(strategy_ret)) / np.maximum.accumulate(1 + np.cumsum(strategy_ret)) - 1)
    win_rate = np.sum(strategy_ret > 0) / n * 100 if n > 0 else 0
    avg_win = np.mean(strategy_ret[strategy_ret > 0]) * 100 if np.sum(strategy_ret > 0) > 0 else 0
    avg_loss = np.mean(strategy_ret[strategy_ret < 0]) * 100 if np.sum(strategy_ret < 0) > 0 else 0
    profit_factor = abs(np.sum(strategy_ret[strategy_ret > 0]) / np.sum(strategy_ret[strategy_ret < 0])) if np.sum(strategy_ret < 0) != 0 else 0
    
    return {
        "total_return_pct": round(total_s * 100, 1),
        "bh_return_pct": round(total_bh * 100, 1),
        "ann_return_pct": round(ann_s * 100, 1),
        "sharpe": round(sharpe_s, 2),
        "bh_sharpe": round(sharpe_bh, 2),
        "max_dd_pct": round(max_dd_s * 100, 1),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "n_trades": n,
    }

# ====== STRATEGIES ======

def strat_ma_cross(df, p):
    """Moving average crossover: fast > slow = long"""
    fast, slow = p["fast"], p["slow"]
    closes = df["close"]
    ma_fast = closes.rolling(fast).mean()
    ma_slow = closes.rolling(slow).mean()
    pos = pd.Series(0, index=closes.index)
    pos[ma_fast > ma_slow] = 1
    pos[ma_fast <= ma_slow] = -1
    return pos.dropna()

def strat_rsi(df, p):
    """RSI oversold/overbought"""
    period = p.get("period", 14)
    oversold = p.get("oversold", 30)
    overbought = p.get("overbought", 70)
    closes = df["close"]
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    pos = pd.Series(0, index=closes.index)
    pos[rsi < oversold] = 1
    pos[rsi > overbought] = -1
    return pos.dropna()

def strat_bb_meanrev(df, p):
    """Bollinger Band mean reversion: touch band = revert"""
    period = p.get("period", 20)
    n_std = p.get("n_std", 2)
    closes = df["close"]
    ma = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    pos = pd.Series(0, index=closes.index)
    pos[closes < lower] = 1   # oversold = long
    pos[closes > upper] = -1  # overbought = short
    return pos.dropna()

def strat_momentum(df, p):
    """N-day momentum: top/bottom decile"""
    lookback = p.get("lookback", 20)
    closes = df["close"]
    mom = closes.pct_change(lookback)
    pos = pd.Series(0, index=closes.index)
    pos[mom > mom.rolling(252).quantile(0.8)] = 1
    pos[mom < mom.rolling(252).quantile(0.2)] = -1
    return pos.dropna()

def strat_volume_breakout(df, p):
    """Volume breakout: price up + volume spike"""
    lookback = p.get("lookback", 20)
    vol_mult = p.get("vol_mult", 2)
    closes = df["close"]
    volume = df["volume"]
    avg_vol = volume.rolling(lookback).mean()
    ret = closes.pct_change(1)
    pos = pd.Series(0, index=closes.index)
    pos[(ret > 0.01) & (volume > avg_vol * vol_mult)] = 1
    pos[(ret < -0.01) & (volume > avg_vol * vol_mult)] = -1
    return pos.dropna()

def strat_day_of_week(df, p):
    """Day-of-week effect: best/worst day"""
    closes = df["close"]
    day = df.index.dayofweek
    day_ret = closes.groupby(day).pct_change(1)
    avg_by_day = closes.groupby(day).apply(lambda x: x.pct_change(1).mean())
    
    best_day = avg_by_day.idxmax()
    worst_day = avg_by_day.idxmin()
    
    pos = pd.Series(0, index=closes.index)
    pos[day == best_day] = 1
    pos[day == worst_day] = -1
    return pos.dropna()

def strat_prev_day_range(df, p):
    """Previous day range expansion: high range -> reversal"""
    lookback = p.get("lookback", 20)
    closes = df["close"]
    highs = df["high"].values
    lows = df["low"].values
    day_range = pd.Series((highs - lows) / closes.values, index=closes.index)
    avg_range = day_range.rolling(lookback).mean()
    
    pos = pd.Series(0, index=closes.index)
    pos[(day_range.shift(1) > avg_range.shift(1) * 1.5) & (closes.pct_change(1) < 0)] = 1
    pos[(day_range.shift(1) > avg_range.shift(1) * 1.5) & (closes.pct_change(1) > 0)] = -1
    return pos.dropna()

def strat_dual_momentum(df, p):
    """Absolute + relative momentum"""
    lookback = p.get("lookback", 126)  # 6 months
    closes = df["close"]
    mom = closes.pct_change(lookback)
    # Long only if positive momentum (absolute)
    pos = pd.Series(0, index=closes.index)
    pos[mom > 0] = 1
    pos[mom <= 0] = -1
    return pos.dropna()

# ====== RUN ALL STRATEGIES ======
strategies = {
    "MA_Cross_20_50": (strat_ma_cross, {"fast": 20, "slow": 50}),
    "MA_Cross_50_200": (strat_ma_cross, {"fast": 50, "slow": 200}),
    "RSI_30_70": (strat_rsi, {"period": 14, "oversold": 30, "overbought": 70}),
    "RSI_25_75": (strat_rsi, {"period": 14, "oversold": 25, "overbought": 75}),
    "BB_MeanRev_2std": (strat_bb_meanrev, {"period": 20, "n_std": 2}),
    "BB_MeanRev_3std": (strat_bb_meanrev, {"period": 20, "n_std": 3}),
    "Momentum_20d": (strat_momentum, {"lookback": 20}),
    "Momentum_60d": (strat_momentum, {"lookback": 60}),
    "Vol_Breakout_2x": (strat_volume_breakout, {"lookback": 20, "vol_mult": 2}),
    "Vol_Breakout_3x": (strat_volume_breakout, {"lookback": 20, "vol_mult": 3}),
    "Dual_Mom_6m": (strat_dual_momentum, {"lookback": 126}),
    "Day_Of_Week": (strat_day_of_week, {}),
    "Prev_Range_Rev": (strat_prev_day_range, {"lookback": 20}),
}

# Summary: strategy -> list of sharpe ratios across stocks
strategy_results = {name: [] for name in strategies}

# Per-stock best strategy
stock_best = {}

tested = 0
for sym in all_stocks:
    df = load_daily(sym)
    if df is None or len(df) < 250:
        continue
    tested += 1
    
    best_name = None
    best_sharpe = -999
    
    for sname, (sfn, sp) in strategies.items():
        result = backtest(df, sfn, sp)
        if result and result["n_trades"] > 50:
            sharpe = result["sharpe"]
            strategy_results[sname].append(sharpe)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = sname
    
    if best_name:
        stock_best[sym] = (best_name, best_sharpe)

print(f"\nTested on {tested} stocks")
print(f"\n{'='*100}")
print("STRATEGY PERFORMANCE SUMMARY (across all stocks)")
print(f"{'='*100}")
print(f"{'Strategy':20s} {'Avg Sharpe':10s} {'Med Sharpe':10s} {'Std Sharpe':10s} {'Win% (>0)':10s} {'% >0.5':10s} {'Best':10s} {'Stocks':8s}")
print("-"*100)

strategy_summary = []
for sname, sharpe_list in strategy_results.items():
    sharpe_arr = np.array(sharpe_list)
    if len(sharpe_arr) == 0:
        continue
    avg_s = np.mean(sharpe_arr)
    med_s = np.median(sharpe_arr)
    std_s = np.std(sharpe_arr)
    win_pct = np.sum(sharpe_arr > 0) / len(sharpe_arr) * 100
    win_05 = np.sum(sharpe_arr > 0.5) / len(sharpe_arr) * 100
    best_s = np.max(sharpe_arr)
    
    strategy_summary.append({
        "name": sname,
        "avg_sharpe": avg_s,
        "med_sharpe": med_s,
        "std_sharpe": std_s,
        "win_pct": win_pct,
        "win_05_pct": win_05,
        "best_sharpe": best_s,
        "n_stocks": len(sharpe_arr),
    })
    
    print(f"{sname:20s} {avg_s:>8.3f}  {med_s:>8.3f}  {std_s:>8.3f}  {win_pct:>8.1f}% {win_05:>8.1f}% {best_s:>8.2f} {len(sharpe_arr):>5d}")

summary_df = pd.DataFrame(strategy_summary).sort_values("avg_sharpe", ascending=False)

print(f"\n{'='*100}")
print("STRATEGY RANKING (by Avg Sharpe)")
print(f"{'='*100}")
for i, (_, r) in enumerate(summary_df.iterrows()):
    print(f"  {i+1:2d}. {r['name']:20s}  avg_sharpe={r['avg_sharpe']:.3f}  med={r['med_sharpe']:.3f}  win%={r['win_pct']:.0f}%  stocks={r['n_stocks']}")

# ====== BEST STRATEGY PER STOCK ======
print(f"\n{'='*100}")
print("STOCKS WHERE BEST STRATEGY WORKS WELL (Sharpe > 1.0)")
print(f"{'='*100}")

good_stocks = [(sym, name, sh) for sym, (name, sh) in stock_best.items() if sh > 1.0]
good_stocks.sort(key=lambda x: x[2], reverse=True)

print(f"Found {len(good_stocks)} stocks with Sharpe > 1.0:")
for sym, name, sh in good_stocks[:30]:
    print(f"  {sym:15s}  best={name:20s}  sharpe={sh:.2f}")

# ====== TOP STRATEGY DEEP DIVE ======
best_overall = summary_df.iloc[0]["name"]
print(f"\n{'='*100}")
print(f"DEEP DIVE: BEST STRATEGY = {best_overall}")
print(f"{'='*100}")

# Show top/bottom stocks for the best strategy
bfn, bp = strategies[best_overall]
stock_perf = []
for sym in all_stocks:
    df = load_daily(sym)
    if df is None or len(df) < 250:
        continue
    result = backtest(df, bfn, bp)
    if result:
        stock_perf.append((result["sharpe"], result["total_return_pct"], result["max_dd_pct"], result["win_rate_pct"], sym))

stock_perf.sort(reverse=True)
print(f"\nTOP 10 STOCKS for {best_overall}:")
for sh, ret, dd, wr, sym in stock_perf[:10]:
    print(f"  {sym:15s} sharpe={sh:.2f}  return={ret:>7.1f}%  max_dd={dd:>5.1f}%  win={wr:.0f}%")

print(f"\nWORST 10 STOCKS for {best_overall}:")
for sh, ret, dd, wr, sym in stock_perf[-10:]:
    print(f"  {sym:15s} sharpe={sh:.2f}  return={ret:>7.1f}%  max_dd={dd:>5.1f}%  win={wr:.0f}%")

# ====== BEST STOCK-SPECIFIC STRATEGY ======
print(f"\n{'='*100}")
print("BEST SINGLE (STOCK, STRATEGY) COMBINATIONS")
print(f"{'='*100}")
all_combos = []
for sym in all_stocks:
    df = load_daily(sym)
    if df is None or len(df) < 250:
        continue
    for sname, (sfn, sp) in strategies.items():
        result = backtest(df, sfn, sp)
        if result and result["sharpe"] > 0.5:
            all_combos.append((result["sharpe"], result["total_return_pct"], sname, sym))

all_combos.sort(reverse=True)
for sh, ret, sname, sym in all_combos[:20]:
    print(f"  {sym:15s} x {sname:20s}  sharpe={sh:.2f}  return={ret:>8.1f}%")

# ====== ENSEMBLE SIGNAL ======
print(f"\n{'='*100}")
print("ENSEMBLE: BEST 3 STRATEGIES COMBINED (on each stock)")
print(f"{'='*100}")

top3_strats = [s["name"] for _, s in summary_df.head(3).iterrows()]
print(f"Top 3 strategies: {top3_strats}")

ensemble_results = []
for sym in all_stocks:
    df = load_daily(sym)
    if df is None or len(df) < 250:
        continue
    
    pos_list = []
    for sname in top3_strats:
        if sname in strategies:
            sfn, sp = strategies[sname]
            pos = sfn(df, sp)
            if pos is not None and len(pos) > 50:
                pos_list.append(pos)
    
    if len(pos_list) < 2:
        continue
    
    # Average positions
    common_idx = pos_list[0].index
    for p in pos_list[1:]:
        common_idx = common_idx.intersection(p.index)
    
    if len(common_idx) < 50:
        continue
    
    ensemble_pos = sum(p.reindex(common_idx) for p in pos_list) / len(pos_list)
    
    # Backtest
    close_series = df["close"].reindex(common_idx)
    closes = close_series.values
    pos_vals = ensemble_pos.values
    daily_ret = np.diff(closes, prepend=closes[0]) / closes[0]
    daily_ret = daily_ret[1:]
    pos_vals = pos_vals[:-1]
    
    strat_ret = daily_ret * np.sign(pos_vals)
    n = len(strat_ret)
    if n < 20:
        continue
    
    total_s = np.prod(1 + strat_ret) - 1
    sharpe_s = np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252) if np.std(strat_ret) > 0 else 0
    ensemble_results.append((sharpe_s, total_s * 100, sym))

ensemble_results.sort(reverse=True)
print(f"\nTOP 10 STOCKS for Ensemble:")
for sh, ret, sym in ensemble_results[:10]:
    print(f"  {sym:15s} sharpe={sh:.2f}  return={ret:>7.1f}%")
print(f"\nENSEMBLE AVERAGE Sharpe: {np.mean([r[0] for r in ensemble_results]):.3f}")
print(f"ENSEMBLE MEDIAN Sharpe:  {np.median([r[0] for r in ensemble_results]):.3f}")

# ====== SAVE RESULTS ======
import json
output = {
    "strategy_summary": {r["name"]: {k: v for k, v in r.items() if k != "name"} for _, r in summary_df.iterrows()},
    "best_combos": [(sym, sname, float(sh)) for sh, ret, sname, sym in all_combos[:50]],
}
with open("strategy_results.json", "w") as f:
    json.dump(output, f, indent=1)

print(f"\nResults saved to strategy_results.json")
print("Done!")
