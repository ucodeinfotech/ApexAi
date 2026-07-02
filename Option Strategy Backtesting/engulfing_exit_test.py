"""
Engulfing Strategy — Exit Modifications Test
Tests improved exit methods to cut losers faster while letting winners run.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 0.50

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

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
        if body.iloc[i] < body.iloc[i-1] * MIN_BODY_PCT: continue
        signals.append({
            "trigger_time": h1["datetime"].iloc[i],
            "level": h1["high"].iloc[i]
        })
    return signals

def execute_trades_exit(signals, m5, exit_name):
    """
    Execute trades with different exit methods.
    
    Exit methods:
    - ch15:          Standard Chandelier 15xATR (baseline)
    - ch15_fixed3:   Fixed 3xATR stop, CH15 trail after profit
    - ch15_fixed2:   Fixed 2xATR stop, CH15 trail after profit
    - ch5_then_15:   5xATR for first 24h, then 15xATR
    - ch8_then_15:   8xATR for first 24h, then 15xATR
    - ch10_then_15:  10xATR for first 24h, then 15xATR
    - time_stop_48:  15xATR trail, but exit if not profitable in 48h
    - time_stop_72:  15xATR trail, exit if not profitable in 72h
    - time_stop_96:  15xATR trail, exit if not profitable in 96h
    - ch15_breakeven: After 3xATR profit, move stop to breakeven + trail at 15xATR
    """
    tc = m5["datetime"].dt.time
    atr5 = compute_atr(m5, 14)
    dt_unix = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi_arr = m5["high"].values; lo_arr = m5["low"].values; cl_arr = m5["close"].values
    dt_series = m5["datetime"]

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

        entry_atr = atr5.iloc[retest]
        if pd.isna(entry_atr): continue

        highest_since_entry = entry_price
        exit_methods = exit_name.split("+")

        for j in range(retest + 1, len(m5)):
            ca = atr5.iloc[j]
            if pd.isna(ca): continue
            if hi_arr[j] > highest_since_entry:
                highest_since_entry = hi_arr[j]

            exit_here = False

            for method in exit_methods:
                if method == "ch15":
                    if cl_arr[j] < highest_since_entry - 15 * ca:
                        exit_here = True; break
                elif method == "ch5_then_15":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    mult = 5 if hours_held <= 24 else 15
                    if cl_arr[j] < highest_since_entry - mult * ca:
                        exit_here = True; break
                elif method == "ch8_then_15":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    mult = 8 if hours_held <= 24 else 15
                    if cl_arr[j] < highest_since_entry - mult * ca:
                        exit_here = True; break
                elif method == "ch10_then_15":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    mult = 10 if hours_held <= 24 else 15
                    if cl_arr[j] < highest_since_entry - mult * ca:
                        exit_here = True; break
                elif method == "ch3_then_15":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    mult = 3 if hours_held <= 24 else 15
                    if cl_arr[j] < highest_since_entry - mult * ca:
                        exit_here = True; break
                elif method == "time_stop_48":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    if cl_arr[j] < highest_since_entry - 15 * ca:
                        exit_here = True; break
                    if hours_held > 48 and cl_arr[j] <= entry_price:
                        exit_here = True; break
                elif method == "time_stop_72":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    if cl_arr[j] < highest_since_entry - 15 * ca:
                        exit_here = True; break
                    if hours_held > 72 and cl_arr[j] <= entry_price:
                        exit_here = True; break
                elif method == "time_stop_96":
                    hours_held = (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600
                    if cl_arr[j] < highest_since_entry - 15 * ca:
                        exit_here = True; break
                    if hours_held > 96 and cl_arr[j] <= entry_price:
                        exit_here = True; break
                elif method == "ch15_breakeven":
                    profit_since_entry = (highest_since_entry - entry_price) / ca
                    if profit_since_entry >= 3:
                        be_stop = entry_price
                        if cl_arr[j] < max(be_stop, highest_since_entry - 15 * ca):
                            exit_here = True; break
                    else:
                        if cl_arr[j] < highest_since_entry - 15 * ca:
                            exit_here = True; break
                elif method == "ch15_pts100":
                    profit_pts = highest_since_entry - entry_price
                    if profit_pts >= 100:
                        if cl_arr[j] < max(entry_price, highest_since_entry - 15 * ca):
                            exit_here = True; break
                    else:
                        if cl_arr[j] < highest_since_entry - 15 * ca:
                            exit_here = True; break
                elif method == "ch15_pts200":
                    profit_pts = highest_since_entry - entry_price
                    if profit_pts >= 200:
                        if cl_arr[j] < max(entry_price, highest_since_entry - 15 * ca):
                            exit_here = True; break
                    else:
                        if cl_arr[j] < highest_since_entry - 15 * ca:
                            exit_here = True; break
                elif method == "ch15_strict":
                    strict_stop = highest_since_entry - 15 * ca
                    if cl_arr[j] < strict_stop:
                        exit_here = True; break
                elif method.startswith("ch"):
                    try:
                        mult_val = float(method[2:])
                        if cl_arr[j] < highest_since_entry - mult_val * ca:
                            exit_here = True; break
                    except:
                        pass

            if exit_here:
                trades.append({
                    "points": cl_arr[j] - entry_price,
                    "exit_time": dt_series.iloc[j],
                    "hold_hours": (dt_series.iloc[j] - dt_series.iloc[retest]).total_seconds() / 3600,
                    "reason": exit_name
                })
                break

    return pd.DataFrame(trades)

def portfolio_loss_filter(df, skip_n=2):
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

def run_test(exit_name, verbose=True):
    all_trades = []
    for sym in ["NIFTY50", "SENSEX"]:
        h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
        m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"] = pd.to_datetime(h1["datetime"])
        m5["datetime"] = pd.to_datetime(m5["datetime"])
        h1 = h1.sort_values("datetime").reset_index(drop=True)
        m5 = m5.sort_values("datetime").reset_index(drop=True)

        sigs = detect_signals(h1)
        trades = execute_trades_exit(sigs, m5, exit_name)
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
    avg_hold = comb["hold_hours"].mean() if n_trades > 0 else 0
    max_dd = (comb["pnl_rs"].cumsum().cummax() - comb["pnl_rs"].cumsum()).max() if n_trades > 0 else 0

    if verbose:
        print(f"  {exit_name:25s}  T={n_trades:4d}  Net=Rs{net_rs:>+9,.0f}  "
              f"WR={wr:5.1f}%  PF={pf:.2f}  AvH={avg_hold:5.1f}h  "
              f"AvW=Rs{avg_win:>+7,.0f}  AvL=Rs{avg_loss:>+7,.0f}  DD=Rs{max_dd:,.0f}")

    return {
        "exit": exit_name, "trades": n_trades, "net_rs": net_rs,
        "wr": wr, "pf": pf, "avg_hold": avg_hold,
        "avg_win": avg_win, "avg_loss": avg_loss, "max_dd": max_dd
    }

def main():
    exits = [
        "ch15",          # Baseline — 15xATR Chandelier
        "ch5_then_15",   # 5xATR for 24h then 15x
        "ch3_then_15",   # 3xATR for 24h then 15x
        "ch8_then_15",   # 8xATR for 24h then 15x
        "ch10_then_15",  # 10xATR for 24h then 15x
        "time_stop_48",  # 15x trail, exit if not profitable in 48h
        "time_stop_72",  # 15x trail, exit if not profitable in 72h
        "time_stop_96",  # 15x trail, exit if not profitable in 96h
        "ch15_breakeven",# Trail + breakeven after 3xATR profit
        "ch15_pts100",   # Trail + breakeven after 100pt profit
        "ch15_pts200",   # Trail + breakeven after 200pt profit
    ]

    print(f"\n{'='*120}")
    print(f"ENGULFING STRATEGY — EXIT MODIFICATION TEST")
    print(f"Baseline: CH15 + skip-after-2-losses")
    print(f"Testing various exit methods to cut losers faster")
    print(f"{'='*120}\n")
    print(f"{'EXIT_METHOD':25s}  {'TRADES':>5s}  {'NET_RS':>11s}  {'WR%':>6s}  "
          f"{'PF':>5s}  {'AVG_HOLD':>7s}  {'AVG_WIN':>9s}  {'AVG_LOSS':>9s}  "
          f"{'MAX_DD':>10s}")
    print("-" * 120)

    results = []
    for ex in exits:
        try:
            res = run_test(ex)
            results.append(res)
        except Exception as e:
            print(f"  {ex:25s}  ERROR: {e}")

    results.sort(key=lambda x: x["net_rs"], reverse=True)

    print(f"\n{'='*120}")
    print(f"RANKED BY NET RS")
    print(f"{'='*120}")
    base_net = [r for r in results if r["exit"] == "ch15"][0]["net_rs"]
    print(f"{'RANK':>4s}  {'EXIT_METHOD':25s}  {'TRADES':>5s}  {'NET_RS':>11s}  "
          f"{'WR%':>6s}  {'PF':>5s}  {'AVG_HOLD':>7s}  {'vs BASE':>9s}")
    print("-" * 90)
    for i, r in enumerate(results):
        vs_base = (r["net_rs"] - base_net) / abs(base_net) * 100 if base_net != 0 else 0
        marker = " <<<" if i == 0 else ""
        print(f"{i+1:4d}  {r['exit']:25s}  {r['trades']:5d}  "
              f"Rs{r['net_rs']:>+8,.0f}  {r['wr']:5.1f}%  "
              f"{r['pf']:5.2f}  {r['avg_hold']:5.1f}h  "
              f"{vs_base:>+8.1f}%{marker}")

    out_dir = os.path.join(BASE, "backtest_results", "exit_test")
    os.makedirs(out_dir, exist_ok=True)
    df_res = pd.DataFrame(results).sort_values("net_rs", ascending=False)
    df_res.to_csv(os.path.join(out_dir, "exit_test_results.csv"), index=False)
    print(f"\nResults saved to: {os.path.join(out_dir, 'exit_test_results.csv')}")

if __name__ == "__main__":
    main()
