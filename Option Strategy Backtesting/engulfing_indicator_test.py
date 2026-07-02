"""
Engulfing Strategy — New Indicator Filters Test
Tests all untested technical indicators as entry filters on CH15 skip=2 baseline.
"""
import pandas as pd, numpy as np, os, warnings, json
from datetime import datetime
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 0.50
CHANDELIER_MULT = 15
SKIP_AFTER_N_LOSSES = 2

# ── Indicator Functions ──────────────────────────────────────────────────

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
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std  # upper, mid, lower

def compute_stoch(df, k_period=14, d_period=3):
    low14 = df["low"].rolling(k_period).min()
    high14 = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low14) / (high14 - low14).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def compute_adx(df, period=14):
    atr = compute_atr(df, period)
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = df["low"].diff().clip(upper=0).abs()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

# ── Signal Detection (modified to accept indicator data) ──────────────────

def detect_signals(h1, ind=None, filter_name=None):
    """
    Detect bullish engulfing on 1H chart.
    If ind and filter_name provided, apply indicator filter.
    """
    body = (h1["close"] - h1["open"]).abs()
    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]

    signals = []
    for i in range(1, len(h1)):
        if not is_red.iloc[i-1]:    continue
        if not is_green.iloc[i]:     continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1]: continue
        if h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
        if body.iloc[i] < body.iloc[i-1] * MIN_BODY_PCT: continue

        # Apply indicator filter if specified
        if filter_name is not None and ind is not None:
            if not apply_filter(h1, ind, i, filter_name):
                continue

        signals.append({
            "trigger_time": h1["datetime"].iloc[i],
            "level": h1["high"].iloc[i]
        })
    return signals

def apply_filter(h1, ind, i, filter_name):
    """Apply a specific indicator filter at candle index i. Returns True to KEEP signal."""
    if filter_name == "none":
        return True
    elif filter_name == "rsi_oversold":
        return ind["rsi"].iloc[i] < 30
    elif filter_name == "rsi_below40":
        return ind["rsi"].iloc[i] < 40
    elif filter_name == "rsi_below50":
        return ind["rsi"].iloc[i] < 50
    elif filter_name == "rsi_mid":
        return 30 <= ind["rsi"].iloc[i] <= 50
    elif filter_name == "rsi_cross_up":
        return ind["rsi"].iloc[i-1] < 30 <= ind["rsi"].iloc[i]
    elif filter_name == "macd_above_signal":
        return ind["macd_line"].iloc[i] > ind["macd_signal"].iloc[i]
    elif filter_name == "macd_hist_pos":
        return ind["macd_hist"].iloc[i] > 0
    elif filter_name == "macd_hist_inc":
        return ind["macd_hist"].iloc[i] > ind["macd_hist"].iloc[i-1]
    elif filter_name == "macd_below0_cross":
        return (ind["macd_line"].iloc[i] < 0 and
                ind["macd_line"].iloc[i] > ind["macd_signal"].iloc[i])
    elif filter_name == "bb_lower":
        return h1["close"].iloc[i] <= ind["bb_lower"].iloc[i]
    elif filter_name == "bb_mid_below":
        return h1["close"].iloc[i] <= ind["bb_mid"].iloc[i]
    elif filter_name == "bb_oversold":
        bw = (ind["bb_upper"].iloc[i] - ind["bb_lower"].iloc[i])
        return h1["close"].iloc[i] <= ind["bb_lower"].iloc[i] + 0.25 * bw
    elif filter_name == "stoch_oversold":
        return ind["stoch_k"].iloc[i] < 20
    elif filter_name == "stoch_below30":
        return ind["stoch_k"].iloc[i] < 30
    elif filter_name == "stoch_below50":
        return ind["stoch_k"].iloc[i] < 50
    elif filter_name == "stoch_cross_up":
        return (ind["stoch_k"].iloc[i-1] < ind["stoch_d"].iloc[i-1] and
                ind["stoch_k"].iloc[i] > ind["stoch_d"].iloc[i] and
                ind["stoch_k"].iloc[i] < 30)
    elif filter_name == "atr_high_regime":
        atr = ind["atr"]
        return atr.iloc[i] > atr.rolling(20).mean().iloc[i]
    elif filter_name == "atr_low_regime":
        atr = ind["atr"]
        return atr.iloc[i] < atr.rolling(20).mean().iloc[i]
    elif filter_name == "atr_percentile_50":
        atr = ind["atr"]
        p50 = atr.rolling(50).quantile(0.50).iloc[i]
        return atr.iloc[i] >= p50
    elif filter_name == "adx_strong":
        return ind["adx"].iloc[i] > 25
    elif filter_name == "adx_trend":
        return ind["adx"].iloc[i] > 20
    elif filter_name == "ema50_trend":
        return h1["close"].iloc[i] > ind["ema50"].iloc[i]
    elif filter_name == "ema200_trend":
        return h1["close"].iloc[i] > ind["ema200"].iloc[i]
    elif filter_name == "ema50_uptrend":
        return ind["ema50"].iloc[i] > ind["ema200"].iloc[i]
    elif filter_name == "price_vs_ema50":
        pct = (h1["close"].iloc[i] / ind["ema50"].iloc[i] - 1) * 100
        return -5 <= pct <= -1
    elif filter_name == "price_vs_ema200":
        pct = (h1["close"].iloc[i] / ind["ema200"].iloc[i] - 1) * 100
        return -3 <= pct <= 0
    elif filter_name == "consecutive_bearish_2":
        return (i >= 2 and h1["close"].iloc[i-2] < h1["open"].iloc[i-2])
    elif filter_name == "consecutive_bearish_3":
        return (i >= 3 and h1["close"].iloc[i-2] < h1["open"].iloc[i-2] and
                h1["close"].iloc[i-3] < h1["open"].iloc[i-3])
    elif filter_name == "gap_down":
        return h1["open"].iloc[i] < h1["close"].iloc[i-1]
    elif filter_name == "body_ratio_large":
        body_c = abs(h1["close"].iloc[i] - h1["open"].iloc[i])
        body_p = abs(h1["close"].iloc[i-1] - h1["open"].iloc[i-1])
        return body_c > 1.5 * body_p
    elif filter_name == "body_ratio_very_large":
        body_c = abs(h1["close"].iloc[i] - h1["open"].iloc[i])
        body_p = abs(h1["close"].iloc[i-1] - h1["open"].iloc[i-1])
        return body_c > 2.0 * body_p
    elif filter_name == "near_swing_low":
        low20 = h1["low"].rolling(20).min()
        pct_from_low = (h1["close"].iloc[i] / low20.iloc[i] - 1) * 100
        return pct_from_low < 2
    elif filter_name == "di_plus_cross":
        return (ind["di_plus"].iloc[i-1] < ind["di_minus"].iloc[i-1] and
                ind["di_plus"].iloc[i] > ind["di_minus"].iloc[i])
    elif filter_name == "super_filter":
        return (ind["rsi"].iloc[i] < 50 and
                ind["macd_hist"].iloc[i] > ind["macd_hist"].iloc[i-1] and
                h1["close"].iloc[i] > ind["ema50"].iloc[i] and
                ind["adx"].iloc[i] > 20)
    # ── Combination filters ──
    elif filter_name == "gap_adx_trend":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                ind["adx"].iloc[i] > 20)
    elif filter_name == "gap_adx_strong":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                ind["adx"].iloc[i] > 25)
    elif filter_name == "gap_body_large":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                abs(h1["close"].iloc[i] - h1["open"].iloc[i]) > 1.5 * abs(h1["close"].iloc[i-1] - h1["open"].iloc[i-1]))
    elif filter_name == "gap_near_low":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                (h1["close"].iloc[i] / h1["low"].rolling(20).min().iloc[i] - 1) * 100 < 2)
    elif filter_name == "gap_adx_body":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                ind["adx"].iloc[i] > 20 and
                abs(h1["close"].iloc[i] - h1["open"].iloc[i]) > 1.5 * abs(h1["close"].iloc[i-1] - h1["open"].iloc[i-1]))
    elif filter_name == "gap_ema50_uptrend":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                ind["ema50"].iloc[i] > ind["ema200"].iloc[i])
    elif filter_name == "gap_hist_inc":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                ind["macd_hist"].iloc[i] > ind["macd_hist"].iloc[i-1])
    elif filter_name == "adx_body_large":
        return (ind["adx"].iloc[i] > 20 and
                abs(h1["close"].iloc[i] - h1["open"].iloc[i]) > 1.5 * abs(h1["close"].iloc[i-1] - h1["open"].iloc[i-1]))
    elif filter_name == "adx_near_low":
        return (ind["adx"].iloc[i] > 20 and
                (h1["close"].iloc[i] / h1["low"].rolling(20).min().iloc[i] - 1) * 100 < 2)
    elif filter_name == "adx_ema50_uptrend":
        return (ind["adx"].iloc[i] > 20 and
                ind["ema50"].iloc[i] > ind["ema200"].iloc[i])
    elif filter_name == "gap_adx_ema50":
        return (h1["open"].iloc[i] < h1["close"].iloc[i-1] and
                ind["adx"].iloc[i] > 20 and
                ind["ema50"].iloc[i] > ind["ema200"].iloc[i])
    return True

# ── Trade Execution (same as CH15 skip=2 baseline) ────────────────────────

def execute_trades(signals, m5, mult=CHANDELIER_MULT):
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
            if (lo_arr[retest] < lv and cl_arr[retest] > lv
                and tc.iloc[retest] < CUTOFF_TIME):
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
            if hi_arr[j] > highest_since_entry:
                highest_since_entry = hi_arr[j]
            if cl_arr[j] < highest_since_entry - mult * ca:
                trades.append({
                    "points": cl_arr[j] - entry_price,
                    "exit_time": m5["datetime"].iloc[j],
                    "hold_hours": (m5["datetime"].iloc[j]
                                   - m5["datetime"].iloc[retest]).total_seconds() / 3600,
                    "reason": f"CH{mult}"
                })
                break
    return pd.DataFrame(trades)

def portfolio_loss_filter(df, skip_n=SKIP_AFTER_N_LOSSES):
    df = df.sort_values("exit_time").reset_index(drop=True)
    loss_count = 0; keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if loss_count >= skip_n:
            keep[i] = False; loss_count = 0; continue
        if df["points"].iloc[i] <= 0:
            loss_count += 1
        else:
            loss_count = 0
    return df[keep].reset_index(drop=True)

# ── Indicator calculation for 1H data ─────────────────────────────────────

def calc_indicators(h1):
    """Calculate all indicator values on 1H data."""
    ind = pd.DataFrame(index=h1.index)
    ind["rsi"] = compute_rsi(h1, 14)
    ind["macd_line"], ind["macd_signal"], ind["macd_hist"] = compute_macd(h1)
    ind["bb_upper"], ind["bb_mid"], ind["bb_lower"] = compute_bollinger(h1)
    ind["stoch_k"], ind["stoch_d"] = compute_stoch(h1)
    ind["atr"] = compute_atr(h1, 14)
    ind["adx"], ind["di_plus"], ind["di_minus"] = compute_adx(h1)
    ind["ema50"] = h1["close"].ewm(span=50).mean()
    ind["ema200"] = h1["close"].ewm(span=200).mean()
    return ind

# ── Test a single filter ──────────────────────────────────────────────────

def test_filter(filter_name, verbose=True):
    """Run the full strategy with a given indicator filter on 1H signals."""
    all_trades = []
    for sym in ["NIFTY50", "SENSEX"]:
        h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
        m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"] = pd.to_datetime(h1["datetime"])
        m5["datetime"] = pd.to_datetime(m5["datetime"])
        h1 = h1.sort_values("datetime").reset_index(drop=True)
        m5 = m5.sort_values("datetime").reset_index(drop=True)

        ind = calc_indicators(h1)
        sigs = detect_signals(h1, ind, filter_name)
        trades = execute_trades(sigs, m5)
        trades["sym"] = sym
        lot = NLOT if "NIFTY" in sym else SLOT
        trades["pnl_rs"] = trades["points"] * lot - CHG
        all_trades.append(trades)

    comb = pd.concat(all_trades, ignore_index=True)
    comb = portfolio_loss_filter(comb)
    net_rs = comb["pnl_rs"].sum() if len(comb) > 0 else 0
    n_trades = len(comb)
    wr = (comb["pnl_rs"] > 0).sum() / n_trades * 100 if n_trades > 0 else 0
    avg_win = comb[comb["pnl_rs"] > 0]["pnl_rs"].mean() if (comb["pnl_rs"] > 0).sum() > 0 else 0
    avg_loss = comb[comb["pnl_rs"] < 0]["pnl_rs"].mean() if (comb["pnl_rs"] < 0).sum() > 0 else 0
    pf = (comb[comb["pnl_rs"] > 0]["pnl_rs"].sum() /
          abs(comb[comb["pnl_rs"] < 0]["pnl_rs"].sum())) if (comb["pnl_rs"] < 0).sum() != 0 else 99

    if verbose:
        print(f"  {filter_name:30s}  Trades={n_trades:4d}  "
              f"Net=Rs{net_rs:>+9,.0f}  WR={wr:5.1f}%  "
              f"AvW=Rs{avg_win:>+8,.0f}  AvL=Rs{avg_loss:>+8,.0f}  PF={pf:.2f}")

    return {
        "filter": filter_name,
        "trades": n_trades,
        "net_rs": net_rs,
        "wr": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pf": pf
    }

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Define all filters to test
    filters = [
        # === RSI filters ===
        "rsi_oversold",      # RSI < 30
        "rsi_below40",       # RSI < 40
        "rsi_below50",       # RSI < 50
        "rsi_mid",           # RSI 30-50
        "rsi_cross_up",      # RSI crosses above 30

        # === MACD filters ===
        "macd_above_signal",  # MACD > signal
        "macd_hist_pos",      # Histogram > 0
        "macd_hist_inc",      # Histogram increasing
        "macd_below0_cross",  # MACD < 0 but crossing up

        # === Bollinger Bands ===
        "bb_lower",           # Close <= lower band
        "bb_mid_below",       # Close <= mid band
        "bb_oversold",        # Close in lower 25% of band

        # === Stochastic ===
        "stoch_oversold",     # K < 20
        "stoch_below30",      # K < 30
        "stoch_below50",      # K < 50
        "stoch_cross_up",     # K crosses above D below 30

        # === ATR / Volatility Regime ===
        "atr_high_regime",    # ATR > ATR_MA20 (vol expansion)
        "atr_low_regime",     # ATR < ATR_MA20 (low vol)
        "atr_percentile_50",  # ATR > 50th percentile

        # === ADX / Trend Strength ===
        "adx_strong",         # ADX > 25
        "adx_trend",          # ADX > 20

        # === Moving Average / Trend Context ===
        "ema50_trend",        # Price > EMA50
        "ema200_trend",       # Price > EMA200
        "ema50_uptrend",      # EMA50 > EMA200
        "price_vs_ema50",     # Price -1% to -5% below EMA50
        "price_vs_ema200",    # Price -3% to 0% below EMA200

        # === Price Action ===
        "consecutive_bearish_2",  # 2+ bearish before
        "consecutive_bearish_3",  # 3+ bearish before
        "body_ratio_large",       # Body > 1.5x prev
        "body_ratio_very_large",  # Body > 2x prev
        "gap_down",               # Gap down into engulfing
        "near_swing_low",         # Near 20-period low

        # === Directional Movement ===
        "di_plus_cross",      # +DI crosses above -DI

        # === Super Combo ===
        "super_filter",       # RSI<50 + MACD_hist_inc + close>EMA50 + ADX>20

        # === Combination Filters (top individual filters combined) ===
        "gap_adx_trend",      # gap_down AND ADX > 20
        "gap_adx_strong",     # gap_down AND ADX > 25
        "gap_body_large",     # gap_down AND body_ratio > 1.5x
        "gap_near_low",       # gap_down AND near_swing_low
        "gap_ema50_uptrend",  # gap_down AND EMA50 > EMA200
        "gap_hist_inc",       # gap_down AND MACD_hist_inc
        "gap_adx_body",       # gap_down AND ADX>20 AND body_large
        "gap_adx_ema50",      # gap_down AND ADX>20 AND EMA50_uptrend
        "adx_body_large",     # ADX>20 AND body_large
        "adx_near_low",       # ADX>20 AND near_swing_low
        "adx_ema50_uptrend",  # ADX>20 AND EMA50 > EMA200
    ]

    print(f"\n{'='*100}")
    print(f"ENGULFING STRATEGY — NEW INDICATOR FILTER TEST")
    print(f"Baseline: CH15 + skip-after-2-losses (unfiltered = all engulfing signals)")
    print(f"Each filter applied on 1H signal candle BEFORE entry")
    print(f"{'='*100}\n")

    # First run baseline (no filter)
    print(f"{'FILTER':30s}  {'TRADES':>6s}  {'NET_RS':>12s}  {'WR%':>6s}  "
          f"{'AVG_WIN':>10s}  {'AVG_LOSS':>10s}  {'PF':>6s}")
    print("-" * 100)
    baseline = test_filter("none")

    # Run each filter
    results = [baseline]
    max_len = max(len(f) for f in filters)
    for fname in filters:
        try:
            res = test_filter(fname)
            results.append(res)
        except Exception as e:
            print(f"  {fname:30s}  ERROR: {e}")

    # Sort by net_rs descending
    results.sort(key=lambda x: x["net_rs"], reverse=True)

    print(f"\n{'='*100}")
    print(f"RANKED RESULTS (sorted by Net Rs)")
    print(f"{'='*100}")
    print(f"{'RANK':>4s}  {'FILTER':30s}  {'TRADES':>6s}  {'NET_RS':>12s}  "
          f"{'WR%':>6s}  {'AVG_WIN':>10s}  {'AVG_LOSS':>10s}  {'PF':>6s}  "
          f"{'vs BASE':>10s}")
    print("-" * 120)
    base_net = baseline["net_rs"]
    for i, r in enumerate(results):
        vs_base = (r["net_rs"] - base_net) / abs(base_net) * 100 if base_net != 0 else 0
        marker = " <<< BEST" if i == 0 else ""
        print(f"{i+1:4d}  {r['filter']:30s}  {r['trades']:6d}  "
              f"Rs{r['net_rs']:>+8,.0f}  {r['wr']:5.1f}%  "
              f"Rs{r['avg_win']:>+7,.0f}  Rs{r['avg_loss']:>+7,.0f}  "
              f"{r['pf']:5.2f}  {vs_base:>+9.1f}%{marker}")

    # Save results
    out_dir = os.path.join(BASE, "backtest_results", "indicator_test")
    os.makedirs(out_dir, exist_ok=True)
    df_res = pd.DataFrame(results).sort_values("net_rs", ascending=False)
    df_res.to_csv(os.path.join(out_dir, "indicator_test_results.csv"), index=False)

    # Summary: which filters beat baseline
    winners = [r for r in results if r["net_rs"] > baseline["net_rs"] and r["filter"] != "none"]
    print(f"\n{'='*100}")
    print(f"SUMMARY: {len(winners)}/{len(filters)} filters BEAT baseline")
    if winners:
        print(f"Best filter: {winners[0]['filter']} → Rs{winners[0]['net_rs']:+,.0f} "
              f"({(winners[0]['net_rs']-baseline['net_rs'])/abs(baseline['net_rs'])*100:+.1f}% vs baseline)")
    print(f"Baseline (no filter): {baseline['net_rs']:+,.0f}")
    print(f"{'='*100}")
    print(f"\nResults saved to: {os.path.join(out_dir, 'indicator_test_results.csv')}")

if __name__ == "__main__":
    main()
