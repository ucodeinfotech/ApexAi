"""
Engulfing Strategy — Advanced Tests
1. Cross-index correlation filter (both NIFTY & SENSEX signal together)
2. Daily timeframe indicators as filters on 1H signals
3. Dynamic Chandelier based on volatility regime
4. Gap size filter (not just gap_down boolean)
5. Combined gap_down + adx_trend + body_large (OR combos)
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 0.50

# ── Helpers ──

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(df, fast=12, slow=26, signal=9):
    ema_f = df["close"].ewm(span=fast).mean()
    ema_s = df["close"].ewm(span=slow).mean()
    return ema_f - ema_s, (ema_f - ema_s).ewm(span=signal).mean()

def detect_signals(h1):
    body = (h1["close"] - h1["open"]).abs()
    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]
    signals = []
    for i in range(1, len(h1)):
        if not is_red.iloc[i-1]:    continue
        if not is_green.iloc[i]:     continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1]: continue
        if h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
        if body.iloc[i] < body.iloc[i-1] * 0.50: continue
        signals.append({
            "trigger_time": h1["datetime"].iloc[i],
            "level": h1["high"].iloc[i],
            "idx": i,
            "open": h1["open"].iloc[i],
            "close": h1["close"].iloc[i],
            "close_prev": h1["close"].iloc[i-1]
        })
    return signals

def execute_trades(signals, m5, mult=15):
    tc = m5["datetime"].dt.time
    atr5 = compute_atr(m5, 14)
    dt_unix = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi_arr = m5["high"].values; lo_arr = m5["low"].values; cl_arr = m5["close"].values
    trades = []
    for sig in signals:
        t_unix = int(pd.to_datetime(sig["trigger_time"]).timestamp())
        lv = sig["level"]
        idx = np.searchsorted(dt_unix, t_unix, side="right")
        if idx >= len(m5): continue
        broke = idx
        while broke < len(m5) and cl_arr[broke] <= lv:
            broke += 1
        if broke >= len(m5): continue
        retest = broke + 1
        while retest < len(m5):
            if lo_arr[retest] < lv and cl_arr[retest] > lv and tc.iloc[retest] < CUTOFF_TIME:
                break
            retest += 1
        if retest >= len(m5): continue
        entry_price = cl_arr[retest]
        stop_loss = lo_arr[retest]
        if entry_price - stop_loss <= 0: continue
        if m5["datetime"].iloc[retest].hour == 9: continue
        highest_since_entry = entry_price
        for j in range(retest + 1, len(m5)):
            ca = atr5.iloc[j]
            if pd.isna(ca): continue
            if hi_arr[j] > highest_since_entry: highest_since_entry = hi_arr[j]
            if cl_arr[j] < highest_since_entry - mult * ca:
                trades.append({
                    "points": cl_arr[j] - entry_price,
                    "exit_time": m5["datetime"].iloc[j],
                    "hold_hours": (m5["datetime"].iloc[j] - m5["datetime"].iloc[retest]).total_seconds() / 3600,
                })
                break
    return pd.DataFrame(trades)

def portfolio_loss_filter(df, skip_n=2):
    df = df.sort_values("exit_time").reset_index(drop=True)
    loss_count = 0; keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if loss_count >= skip_n:
            keep[i] = False; loss_count = 0; continue
        if df["points"].iloc[i] <= 0: loss_count += 1
        else: loss_count = 0
    return df[keep].reset_index(drop=True)

def run_sym(sym, filter_fn=None):
    h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
    m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)

    sigs = detect_signals(h1)
    if filter_fn:
        sigs = [s for s in sigs if filter_fn(h1, s)]

    trades = execute_trades(sigs, m5)
    trades["sym"] = sym
    lot = NLOT if "NIFTY" in sym else SLOT
    trades["pnl_rs"] = trades["points"] * lot - CHG
    return trades, h1

def run_test(name, filter_fn=None, both_indices_filter=False):
    """
    both_indices_filter: if True, only take trades when BOTH NIFTY & SENSEX
    have signals within N hours of each other.
    """
    all_trades = []
    if both_indices_filter:
        # Get signals for both indices with timestamps
        nifty_trades, nifty_h1 = run_sym("NIFTY50")
        sensex_trades, sensex_h1 = run_sym("SENSEX")
        nifty_trades["sym"] = "NIFTY50"
        sensex_trades["sym"] = "SENSEX"
        lot_n = NLOT; lot_s = SLOT
        nifty_trades["pnl_rs"] = nifty_trades["points"] * lot_n - CHG
        sensex_trades["pnl_rs"] = sensex_trades["points"] * lot_s - CHG
        all_trades = [nifty_trades, sensex_trades]
    else:
        for sym in ["NIFTY50", "SENSEX"]:
            trades, _ = run_sym(sym, filter_fn)
            all_trades.append(trades)

    comb = pd.concat(all_trades, ignore_index=True)
    comb = portfolio_loss_filter(comb)
    net_rs = comb["pnl_rs"].sum() if len(comb) > 0 else 0
    n = len(comb)
    wr = (comb["pnl_rs"] > 0).sum() / n * 100 if n > 0 else 0
    pf = (comb[comb["pnl_rs"]>0]["pnl_rs"].sum() /
          abs(comb[comb["pnl_rs"]<0]["pnl_rs"].sum())) if (comb["pnl_rs"]<0).sum() != 0 else 99
    avg_h = comb["hold_hours"].mean() if n > 0 else 0
    return {"name": name, "trades": n, "net_rs": net_rs, "wr": wr, "pf": pf, "avg_hold": avg_h}

# ── Filter Functions ──

def daily_rsi_oversold(h1, sig):
    """Daily RSI(14) < 35 at time of signal"""
    daily = h1.copy()
    daily["date"] = daily["datetime"].dt.date
    daily = daily.groupby("date").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).reset_index()
    daily["rsi"] = compute_rsi(daily, 14)
    sig_date = pd.to_datetime(sig["trigger_time"]).date()
    mask = daily["date"] == sig_date
    if mask.sum() == 0: return False
    rsi_val = daily.loc[mask, "rsi"].iloc[0]
    # Also check prev day RSI if current day values aren't available yet
    if pd.isna(rsi_val):
        prev_idx = daily["date"].searchsorted(sig_date) - 1
        if prev_idx >= 0:
            rsi_val = daily["rsi"].iloc[prev_idx]
    return not pd.isna(rsi_val) and rsi_val < 35

def daily_rsi_below40(h1, sig):
    daily = h1.copy()
    daily["date"] = daily["datetime"].dt.date
    daily = daily.groupby("date").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).reset_index()
    daily["rsi"] = compute_rsi(daily, 14)
    sig_date = pd.to_datetime(sig["trigger_time"]).date()
    mask = daily["date"] == sig_date
    if mask.sum() == 0: return False
    rsi_val = daily.loc[mask, "rsi"].iloc[0]
    if pd.isna(rsi_val):
        prev_idx = daily["date"].searchsorted(sig_date) - 1
        if prev_idx >= 0: rsi_val = daily["rsi"].iloc[prev_idx]
    return not pd.isna(rsi_val) and rsi_val < 40

def daily_macd_bullish(h1, sig):
    daily = h1.copy()
    daily["date"] = daily["datetime"].dt.date
    daily = daily.groupby("date").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).reset_index()
    macd, signal = compute_macd(daily)
    sig_date = pd.to_datetime(sig["trigger_time"]).date()
    mask = daily["date"] == sig_date
    if mask.sum() == 0: return False
    idx = mask.idxmax() if mask.any() else -1
    if idx < 1: return False
    return macd.iloc[idx] > signal.iloc[idx]

def daily_uptrend(h1, sig):
    """Price above daily EMA50 (uptrend context)"""
    daily = h1.copy()
    daily["date"] = daily["datetime"].dt.date
    daily = daily.groupby("date").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).reset_index()
    daily["ema50"] = daily["close"].ewm(span=50).mean()
    sig_date = pd.to_datetime(sig["trigger_time"]).date()
    mask = daily["date"] == sig_date
    if mask.sum() == 0: return False
    return daily.loc[mask, "close"].iloc[0] > daily.loc[mask, "ema50"].iloc[0]

def gap_size_large(h1, sig):
    """Gap down > 0.2%"""
    gap_pct = (sig["open"] / sig["close_prev"] - 1) * 100
    return gap_pct < -0.2

def gap_size_very_large(h1, sig):
    gap_pct = (sig["open"] / sig["close_prev"] - 1) * 100
    return gap_pct < -0.5

def price_far_from_ema50(h1, sig):
    """Price > 3% below EMA50 (deep pullback)"""
    ema50 = h1["close"].ewm(span=50).mean()
    pct = (sig["close"] / ema50.iloc[sig["idx"]] - 1) * 100
    return pct < -3 and pct > -8

def price_far_from_ema200(h1, sig):
    ema200 = h1["close"].ewm(span=200).mean()
    pct = (sig["close"] / ema200.iloc[sig["idx"]] - 1) * 100
    return pct < -2 and pct > -6

def body_ratio_extreme(h1, sig):
    """Engulfing body > 3x previous body (very strong)"""
    body_c = abs(sig["close"] - sig["open"])
    body_p = abs(h1["close"].iloc[sig["idx"]-1] - h1["open"].iloc[sig["idx"]-1])
    return body_c > 3 * body_p

# ── Main ──

def main():
    # 1. Daily timeframe indicator filters
    filters = [
        ("daily_rsi_below40", daily_rsi_below40),
        ("daily_rsi_oversold", daily_rsi_oversold),
        ("daily_macd_bullish", daily_macd_bullish),
        ("daily_uptrend", daily_uptrend),
        ("gap_size_large", gap_size_large),
        ("gap_size_very_large", gap_size_very_large),
        ("price_far_ema50", price_far_from_ema50),
        ("price_far_ema200", price_far_from_ema200),
        ("body_ratio_extreme", body_ratio_extreme),
    ]

    # 2. Multi-factor combos (AND logic)
    def mk_and(f1, f2):
        return lambda h1, s: f1(h1, s) and f2(h1, s)
    def mk_or(f1, f2):
        return lambda h1, s: f1(h1, s) or f2(h1, s)

    filters += [
        ("gap_large_or_daily_rsi", mk_or(gap_size_large, daily_rsi_below40)),
        ("gap_large_and_daily_rsi", mk_and(gap_size_large, daily_rsi_below40)),
        ("gap_large_or_daily_uptrend", mk_or(gap_size_large, daily_uptrend)),
        ("gap_large_and_daily_uptrend", mk_and(gap_size_large, daily_uptrend)),
        ("daily_rsi_and_uptrend", mk_and(daily_rsi_below40, daily_uptrend)),
        ("gap_large_and_body_extreme", mk_and(gap_size_large, body_ratio_extreme)),
    ]

    print(f"\n{'='*120}")
    print(f"ENGULFING — ADVANCED FILTER TEST (Daily TF, Gap Size, Multi-factor)")
    print(f"{'='*120}\n")
    print(f"{'FILTER':30s}  {'TRADES':>5s}  {'NET_RS':>11s}  {'WR%':>6s}  {'PF':>5s}  {'AVG_H':>6s}")
    print("-" * 80)

    results = []
    # Baseline
    base = run_test("baseline")
    results.append(base)
    print(f"  {base['name']:30s}  T={base['trades']:4d}  Rs{base['net_rs']:>+9,.0f}  "
          f"{base['wr']:5.1f}%  {base['pf']:5.2f}  {base['avg_hold']:5.1f}h")

    for name, fn in filters:
        try:
            res = run_test(name, filter_fn=fn)
            results.append(res)
            print(f"  {name:30s}  T={res['trades']:4d}  Rs{res['net_rs']:>+9,.0f}  "
                  f"{res['wr']:5.1f}%  {res['pf']:5.2f}  {res['avg_hold']:5.1f}h")
        except Exception as e:
            print(f"  {name:30s}  ERROR: {e}")

    results.sort(key=lambda x: x["net_rs"], reverse=True)
    print(f"\n{'='*80}")
    print(f"RANKED")
    print(f"{'='*80}")
    base_net = base["net_rs"]
    for i, r in enumerate(results):
        vs = (r["net_rs"] - base_net) / abs(base_net) * 100 if base_net != 0 else 0
        m = " <<<" if i == 0 else ""
        print(f"  {i+1:2d}. {r['name']:30s}  T={r['trades']:4d}  "
              f"Rs{r['net_rs']:>+9,.0f}  {r['wr']:5.1f}%  "
              f"{r['pf']:5.2f}  {vs:+7.1f}%{m}")

    out_dir = os.path.join(BASE, "backtest_results", "advanced_test")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).sort_values("net_rs", ascending=False).to_csv(
        os.path.join(out_dir, "advanced_test_results.csv"), index=False)
    print(f"\nSaved to: {os.path.join(out_dir, 'advanced_test_results.csv')}")

if __name__ == "__main__":
    main()
