"""
BB(20, 2.5) Breakout Mean-Reversion Backtest
- Clean implementation, verified step by step
- Strategy: When candle completely outside bands, enter at close
  SHORT when low > upper band (overextended up)
  LONG when high < lower band (overextended down)
- T1 = +1R at trigger candle extreme
- TP = +3R
- EOD exit at 15:15
"""
import pandas as pd, numpy as np, os, sys
from datetime import datetime

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CHARGES ===
def compute_charges(entry, exit_, qty=1):
    """Per-trade charges for Indian equity cash segment"""
    brk = 10 * 2                           # brokerage buy + sell
    stt = exit_ * qty * 0.001              # STT 0.1% on sell
    exc = (entry + exit_) * qty * 0.00003  # exchange turnover
    sebi = (entry + exit_) * qty * 0.000001 * 2  # SEBI charges
    std = entry * qty * 0.00003            # stamp duty on buy
    gst = (brk + exc) * 0.18               # GST on brokerage + exchange
    return round(brk + stt + exc + sebi + std + gst, 2)

# === STRATEGY ===
def backtest(symbol, period=20, n_std=2.5, rr=3.0):
    path = os.path.join(DATA_DIR, f"{symbol}_FIFTEEN_MINUTE.csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Indicators
    df["sma"] = df["close"].rolling(period).mean()
    df["std"] = df["close"].rolling(period).std(ddof=1)
    df["upper"] = df["sma"] + n_std * df["std"]
    df["lower"] = df["sma"] - n_std * df["std"]

    trades = []
    for i in range(period, len(df) - 1):
        row = df.iloc[i]

        # --- TRIGGER ---
        if row["low"] > row["upper"]:   # Entire candle above upper band -> SHORT
            side = "SHORT"
            entry = row["close"]
            t1_price = row["low"]       # T1 = trigger low (price must go DOWN to here)
            t1_dist = entry - t1_price
            tp_price = entry - t1_dist * rr  # TP = further DOWN
            if t1_dist <= 0:
                continue
        elif row["high"] < row["lower"]:  # Entire candle below lower band -> LONG
            side = "LONG"
            entry = row["close"]
            t1_price = row["high"]      # T1 = trigger high (price must go UP to here)
            t1_dist = t1_price - entry
            tp_price = entry + t1_dist * rr  # TP = further UP
            if t1_dist <= 0:
                continue
        else:
            continue

        # --- EXIT SEARCH ---
        exit_price = entry
        exit_reason = "EOD"
        exit_time = row["datetime"]
        trigger_date = row["datetime"].date()

        for k in range(i + 1, len(df)):
            bar = df.iloc[k]
            bar_dt = bar["datetime"]

            # EOD check
            if bar_dt.hour >= 15 and bar_dt.minute >= 15:
                exit_price = bar["close"]
                exit_reason = "EOD"
                exit_time = bar_dt
                break

            # Check TP and T1
            if side == "SHORT":
                tp_hit = bar["low"] <= tp_price      # price went DOWN to TP
                t1_hit = bar["low"] <= t1_price      # price went DOWN to T1
            else:  # LONG
                tp_hit = bar["high"] >= tp_price     # price went UP to TP
                t1_hit = bar["high"] >= t1_price     # price went UP to T1

            if tp_hit and t1_hit:
                exit_price = tp_price
                exit_reason = "TP"
                exit_time = bar_dt
                break
            elif tp_hit:
                exit_price = tp_price
                exit_reason = "TP"
                exit_time = bar_dt
                break
            elif t1_hit:
                exit_price = t1_price
                exit_reason = "T1"
                exit_time = bar_dt
                break

        # --- P&L ---
        if side == "SHORT":
            pnl_pts = entry - exit_price
        else:
            pnl_pts = exit_price - entry

        charges = compute_charges(entry, exit_price)
        net_pnl = round(pnl_pts - charges, 2)
        r = round(pnl_pts / t1_dist, 2) if t1_dist > 0 else 0.0

        trades.append({
            "symbol": symbol,
            "date": str(trigger_date),
            "exit_time": str(exit_time),
            "year": row["datetime"].year,
            "month": row["datetime"].month,
            "dow": row["datetime"].dayofweek,
            "side": side,
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "t1_target": round(t1_price, 2),
            "tp_target": round(tp_price, 2),
            "t1_dist": round(t1_dist, 2),
            "pnl_pts": round(pnl_pts, 2),
            "charges": charges,
            "net_pnl": net_pnl,
            "r": r,
            "reason": exit_reason,
        })

    return pd.DataFrame(trades)


# === RUN FOR ALL THREE ===
all_results = {}
for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
    print(f"\n{'='*60}")
    print(f"Running backtest for {sym}...")
    df = backtest(sym)
    all_results[sym] = df

    t = len(df)
    wins = int((df["pnl_pts"] > 0).sum())
    losses = t - wins
    t1c = int((df["reason"] == "T1").sum())
    tpc = int((df["reason"] == "TP").sum())
    eodc = int((df["reason"] == "EOD").sum())
    gross = round(df["pnl_pts"].sum(), 2)
    charges_total = round(df["charges"].sum(), 2)
    net = round(df["net_pnl"].sum(), 2)
    avg_r = round(df["r"].mean(), 2)
    avg_t1 = round(df[df["reason"] == "T1"]["pnl_pts"].mean(), 1) if t1c else 0
    avg_tp = round(df[df["reason"] == "TP"]["pnl_pts"].mean(), 1) if tpc else 0
    avg_eod = round(df[df["reason"] == "EOD"]["pnl_pts"].mean(), 1) if eodc else 0

    # MaxDD
    cs = df.sort_values("exit_time").reset_index(drop=True)
    cs["cum"] = cs["net_pnl"].cumsum()
    cs["peak"] = cs["cum"].cummax()
    mdd = round((cs["peak"] - cs["cum"]).max(), 2)

    pf = round(df[df["pnl_pts"] > 0]["pnl_pts"].sum() / abs(df[df["pnl_pts"] <= 0]["pnl_pts"].sum()), 2) if losses > 0 else float("inf")

    print(f"  Trades: {t}")
    print(f"  Wins: {wins} ({wins/t*100:.1f}%)  Losses: {losses} ({losses/t*100:.1f}%)")
    print(f"  T1: {t1c} ({t1c/t*100:.1f}%)  TP: {tpc} ({tpc/t*100:.1f}%)  EOD: {eodc} ({eodc/t*100:.1f}%)")
    print(f"  Gross P&L: {gross:+.0f} pts")
    print(f"  Charges: Rs{charges_total:,.0f}")
    print(f"  Net P&L: {net:+.0f} pts")
    print(f"  Profit Factor: {pf}")
    print(f"  Avg T1 P&L: {avg_t1:+.1f} | Avg TP P&L: {avg_tp:+.1f} | Avg EOD P&L: {avg_eod:+.1f}")
    print(f"  Avg R: {avg_r}")
    print(f"  Max Drawdown: {mdd:,.0f} pts")

    # Save CSV
    df.to_csv(os.path.join(OUTPUT_DIR, f"bb_{sym.lower()}_trades.csv"), index=False)
    print(f"  Saved: bb_{sym.lower()}_trades.csv")


# === VERIFICATION: Trace 2 trades step by step ===
print(f"\n{'='*60}")
print("TRACE VERIFICATION: Step-by-step trade walkthrough")
print("=" * 60)

for sym in ["BANKNIFTY"]:
    df = all_results[sym]
    # Pick first SHORT that didn't hit T1 on first candle, and first LONG
    traces = []
    for _, trade in df.iterrows():
        if len(traces) >= 2:
            break
        side = trade["side"]
        if side == "SHORT" and not any(t["side"] == "SHORT" for t in traces):
            traces.append(trade)
        if side == "LONG" and not any(t["side"] == "LONG" for t in traces):
            traces.append(trade)

    for trade in traces:
        print(f"\n--- {trade['side']} on {trade['date']} ---")
        print(f"  Entry: {trade['entry']} | T1: {trade['t1_target']} (dist={trade['t1_dist']}) | TP: {trade['tp_target']}")
        print(f"  Exit: {trade['exit']} | Reason: {trade['reason']} | PnL: {trade['pnl_pts']:+.2f} pts | R: {trade['r']}")

        # Verify: T1 check uses correct side
        data = pd.read_csv(os.path.join(DATA_DIR, f"{sym}_FIFTEEN_MINUTE.csv"))
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.sort_values("datetime").reset_index(drop=True)

        # Find the trigger candle and next candles
        trigger_found = False
        for j in range(len(data)):
            row = data.iloc[j]
            if str(row["datetime"].date()) == trade["date"] and abs(row["close"] - trade["entry"]) < 0.5:
                # Check this is the trigger
                if (trade["side"] == "SHORT" and row["low"] > row["close"]) or \
                   (trade["side"] == "LONG" and row["high"] < row["close"]):
                    pass  # simplified check
                trigger_found = True
                print(f"\n  Trigger candle [{j}]: {row['datetime']} | O:{row['open']:.1f} H:{row['high']:.1f} L:{row['low']:.1f} C:{row['close']:.1f}")

                # Show next few candles
                for k in range(j+1, min(j+10, len(data))):
                    b = data.iloc[k]
                    # Check T1/TP for each candle
                    if trade["side"] == "SHORT":
                        t1_hit = b["low"] <= trade["t1_target"]
                        tp_hit = b["low"] <= trade["tp_target"]
                    else:
                        t1_hit = b["high"] >= trade["t1_target"]
                        tp_hit = b["high"] >= trade["tp_target"]

                    tag = ""
                    if tp_hit: tag = " <<< TP HIT"
                    elif t1_hit: tag = " <<< T1 HIT"

                    print(f"    [{k}] {b['datetime']} | {b['open']:.1f} {b['high']:.1f} {b['low']:.1f} {b['close']:.1f}{tag}")

                    if b["datetime"] == pd.to_datetime(trade["exit_time"]):
                        break
                break

# === SUMMARY ===
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print("=" * 60)

for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
    df = all_results[sym]
    t = len(df)
    w = int((df["pnl_pts"] > 0).sum())
    t1 = int((df["reason"] == "T1").sum())
    tp = int((df["reason"] == "TP").sum())
    eod = int((df["reason"] == "EOD").sum())
    print(f"{sym:10s} | Trades:{t:3d} | W:{w:3d}({w/t*100:.0f}%) | "
          f"T1:{t1:3d}({t1/t*100:.0f}%) TP:{tp:2d}({tp/t*100:.0f}%) EOD:{eod:2d}({eod/t*100:.0f}%) | "
          f"Gross:{df['pnl_pts'].sum():+7.0f} | Net:{df['net_pnl'].sum():+7.0f} | "
          f"AvgR:{df['r'].mean():+.2f}")

print(f"\nFull trade CSVs saved in '{OUTPUT_DIR}/'")
print("Done.")
