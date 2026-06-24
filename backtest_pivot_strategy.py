"""
Pivot Breakout Strategy Backtester
15-min triggers (touch R1/S1) -> 1-min entry
"""
import pandas as pd
import numpy as np
import os, json
from datetime import datetime, timedelta

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BROKERAGE_PER_ORDER = 10  # Rs per order (buy or sell)
STT = 0.001  # 0.1% STT on sell
EXCHANGE_TC = 0.00003  # 0.003% exchange charges
SEBI_TC = 0.000001  # 0.0001% SEBI turnover fee
GST = 0.18  # 18% GST on brokerage + exchange charges
STAMP_DUTY = 0.00003  # 0.003% on buy

SLIPPAGE_POINTS = 1  # 1-2 points buffer

def calc_pivot_daily(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def compute_charges(entry_price, exit_price, qty=1):
    """Compute all charges for a round-trip trade on 1 share (qty=1 for per-unit calc)"""
    turnover_buy = entry_price * qty
    turnover_sell = exit_price * qty
    turnover_total = turnover_buy + turnover_sell
    
    brokerage = BROKERAGE_PER_ORDER * 2  # buy + sell
    stt_total = turnover_sell * STT
    exchange_total = turnover_total * EXCHANGE_TC
    sebi_total = turnover_total * SEBI_TC * 2  # buy + sell
    stamp = turnover_buy * STAMP_DUTY
    
    gst_total = (brokerage + exchange_total) * GST
    
    total_charges = brokerage + stt_total + exchange_total + sebi_total + stamp + gst_total
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt_total, 2),
        "exchange": round(exchange_total, 2),
        "sebi": round(sebi_total, 4),
        "stamp": round(stamp, 2),
        "gst": round(gst_total, 2),
        "total": round(total_charges, 2)
    }

def backtest_stock(symbol, token, data_dir, output_dir):
    print(f"  Backtesting {symbol}...")
    
    # Load data
    df15_path = f"{data_dir}/{symbol}_FIFTEEN_MINUTE.csv"
    df1_path = f"{data_dir}/{symbol}_ONE_MINUTE.csv"
    
    if not os.path.exists(df15_path) or not os.path.exists(df1_path):
        print(f"    SKIP - missing data files")
        return None
    
    df15 = pd.read_csv(df15_path)
    df1 = pd.read_csv(df1_path)
    
    df15["datetime"] = pd.to_datetime(df15["datetime"])
    df1["datetime"] = pd.to_datetime(df1["datetime"])
    df15["date"] = df15["datetime"].dt.date
    df1["date"] = df1["datetime"].dt.date
    
    # Sort
    df15 = df15.sort_values("datetime").reset_index(drop=True)
    df1 = df1.sort_values("datetime").reset_index(drop=True)
    
    # === COMPUTE DAILY PIVOTS ===
    daily = df15.groupby("date").agg({"high": "max", "low": "min", "close": "last"}).reset_index()
    pivots = []
    for i, row in daily.iterrows():
        p, r1, s1, r2, s2 = calc_pivot_daily(row["high"], row["low"], row["close"])
        pivots.append({"date": row["date"], "pivot": p, "r1": r1, "s1": s1, "r2": r2, "s2": s2})
    pivots_df = pd.DataFrame(pivots)
    # Shift by 1 day - today's pivots use yesterday's data
    pivots_df["date"] = pivots_df["date"].shift(1)
    pivots_df = pivots_df.dropna().reset_index(drop=True)
    pivots_df["date"] = pivots_df["date"].astype(object)
    
    # Merge pivots into 15-min data
    df15 = df15.merge(pivots_df, on="date", how="left")
    df15 = df15.dropna(subset=["r1", "s1"]).reset_index(drop=True)
    
    # === FIND TRIGGER CANDLES (15-min) ===
    triggers = []
    for _, row in df15.iterrows():
        long_trigger = row["high"] >= row["r1"]
        short_trigger = row["low"] <= row["s1"]
        
        if long_trigger or short_trigger:
            triggers.append({
                "datetime": row["datetime"],
                "date": row["date"],
                "type": "LONG" if long_trigger else "SHORT",
                "trigger_high": row["high"],
                "trigger_low": row["low"],
                "trigger_open": row["open"],
                "trigger_close": row["close"],
                "entry_level": row["high"] + SLIPPAGE_POINTS if long_trigger else row["low"] - SLIPPAGE_POINTS,
                "sl": row["low"] if long_trigger else row["high"],
                "tp": None,  # calculated from entry/sl
                "r1": row["r1"],
                "s1": row["s1"]
            })
    
    if not triggers:
        return {"symbol": symbol, "trades": 0, "error": "No triggers"}
    
    # Convert triggers to DataFrame
    triggers_df = pd.DataFrame(triggers)
    # Calculate TP for each trigger
    for i, t in triggers_df.iterrows():
        if t["type"] == "LONG":
            risk = t["entry_level"] - t["sl"]
            triggers_df.at[i, "tp"] = t["entry_level"] + 2 * risk
        else:
            risk = t["sl"] - t["entry_level"]
            triggers_df.at[i, "tp"] = t["entry_level"] - 2 * risk
    
    # === MATCH 1-MIN BARS TO TRIGGERS ===
    trades = []
    
    for _, trigger in triggers_df.iterrows():
        t_dt = trigger["datetime"]
        t_date = trigger["date"]
        
        # Get 1-min bars from trigger to end of day
        window_end = datetime.combine(t_date, datetime.max.time()).replace(hour=15, minute=30)
        mask = (df1["datetime"] > t_dt) & (df1["datetime"] <= pd.Timestamp(window_end, tz=df1["datetime"].dt.tz))
        scan_bars = df1[mask].copy()
        
        if scan_bars.empty:
            continue
        
        entry_filled = False
        entry_price = None
        entry_time = None
        exit_price = None
        exit_time = None
        exit_reason = None
        
        for _, bar in scan_bars.iterrows():
            bar_h = bar["high"]
            bar_l = bar["low"]
            bar_c = bar["close"]
            bar_t = bar["datetime"]
            
            if not entry_filled:
                # Check for entry
                if trigger["type"] == "LONG" and bar_h >= trigger["entry_level"]:
                    entry_price = trigger["entry_level"]
                    entry_time = bar_t
                    entry_filled = True
                elif trigger["type"] == "SHORT" and bar_l <= trigger["entry_level"]:
                    entry_price = trigger["entry_level"]
                    entry_time = bar_t
                    entry_filled = True
            else:
                # Check SL / TP
                if trigger["type"] == "LONG":
                    if bar_l <= trigger["sl"]:
                        exit_price = trigger["sl"]
                        exit_time = bar_t
                        exit_reason = "SL"
                        break
                    elif bar_h >= trigger["tp"]:
                        exit_price = trigger["tp"]
                        exit_time = bar_t
                        exit_reason = "TP"
                        break
                else:  # SHORT
                    if bar_h >= trigger["sl"]:
                        exit_price = trigger["sl"]
                        exit_time = bar_t
                        exit_reason = "SL"
                        break
                    elif bar_l <= trigger["tp"]:
                        exit_price = trigger["tp"]
                        exit_time = bar_t
                        exit_reason = "TP"
                        break
        
        if entry_filled and exit_price is not None:
            if trigger["type"] == "LONG":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            
            risk_amount = abs(entry_price - trigger["sl"])
            r_multiple = round(pnl / risk_amount, 2) if risk_amount > 0 else 0
            
            charges = compute_charges(entry_price, exit_price)
            net_pnl = round(pnl - charges["total"], 2)
            
            trades.append({
                "symbol": symbol,
                "date": str(t_date),
                "type": trigger["type"],
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "trigger_time": str(t_dt),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "sl": round(trigger["sl"], 2),
                "tp": round(trigger["tp"], 2),
                "exit_reason": exit_reason,
                "pnl": round(pnl, 2),
                "net_pnl": net_pnl,
                "r_multiple": r_multiple,
                "charges": charges["total"],
            })
    
    if not trades:
        return {"symbol": symbol, "trades": 0, "error": "No fills"}
    
    return trades

def compute_metrics(all_trades):
    if not all_trades:
        return {}
    
    df = pd.DataFrame(all_trades)
    
    total_trades = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades * 100 if total_trades else 0
    
    gross_profit = wins["net_pnl"].sum() if win_count else 0
    gross_loss = losses["net_pnl"].sum() if loss_count else 0
    net_profit = df["net_pnl"].sum()
    
    avg_win = wins["net_pnl"].mean() if win_count else 0
    avg_loss = losses["net_pnl"].mean() if loss_count else 0
    
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    avg_r = df["r_multiple"].mean()
    avg_r_wins = wins["r_multiple"].mean() if win_count else 0
    avg_r_losses = losses["r_multiple"].mean() if loss_count else 0
    
    expectancy = avg_win * (win_rate/100) + avg_loss * (1 - win_rate/100)
    
    # Max drawdown on cumulative PnL
    df_sorted = df.sort_values("exit_time").reset_index(drop=True)
    df_sorted["cumulative"] = df_sorted["net_pnl"].cumsum()
    df_sorted["peak"] = df_sorted["cumulative"].cummax()
    df_sorted["dd"] = df_sorted["peak"] - df_sorted["cumulative"]
    max_dd = df_sorted["dd"].max()
    max_dd_pct = (max_dd / df_sorted["peak"].max() * 100) if df_sorted["peak"].max() > 0 else 0
    
    # Sharpe (using R multiples)
    sharpe = df["r_multiple"].mean() / df["r_multiple"].std() * np.sqrt(total_trades) if df["r_multiple"].std() > 0 else 0
    
    total_charges = df["charges"].sum()
    
    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_profit": round(net_profit, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_r_multiple": round(avg_r, 2),
        "avg_r_win": round(avg_r_wins, 2),
        "avg_r_loss": round(avg_r_losses, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_charges": round(total_charges, 2),
    }

def main():
    print("=" * 60)
    print("PIVOT BREAKOUT BACKTESTER")
    print("15-min triggers -> 1-min entries")
    print("=" * 60)
    
    # Get all stock files
    files15 = [f for f in os.listdir(DATA_DIR) if f.endswith("_FIFTEEN_MINUTE.csv")]
    symbols = [f.replace("_FIFTEEN_MINUTE.csv", "") for f in files15]
    # Exclude indices
    symbols = [s for s in symbols if s not in ("NIFTY50", "BANKNIFTY", "SENSEX")]
    
    print(f"\nStocks to backtest: {len(symbols)}")
    
    all_trades = []
    stock_results = {}
    
    for sym in sorted(symbols):
        result = backtest_stock(sym, None, DATA_DIR, OUTPUT_DIR)
        if result is None:
            continue
        if isinstance(result, dict) and "error" in result:
            stock_results[sym] = {"trades": 0, "error": result["error"]}
            continue
        all_trades.extend(result)
        stock_results[sym] = {
            "trades": len(result),
            "net_pnl": round(sum(t["net_pnl"] for t in result), 2),
            "win_rate": round(sum(1 for t in result if t["net_pnl"] > 0) / len(result) * 100, 1),
            "avg_r": round(sum(t["r_multiple"] for t in result) / len(result), 2)
        }
    
    # Compute overall metrics
    if all_trades:
        metrics = compute_metrics(all_trades)
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(f"{OUTPUT_DIR}/trade_book.csv", index=False)
        
        # Stock-wise summary
        stock_summary = pd.DataFrame(stock_results).T
        stock_summary.to_csv(f"{OUTPUT_DIR}/stock_summary.csv")
        
        # Print results
        print("\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        print(f"\nTotal Trades:     {metrics['total_trades']}")
        print(f"Wins / Losses:    {metrics['win_count']} / {metrics['loss_count']}")
        print(f"Win Rate:         {metrics['win_rate']}%")
        print(f"Net P&L:          ₹{metrics['net_profit']:,.2f}")
        print(f"Gross Profit:     ₹{metrics['gross_profit']:,.2f}")
        print(f"Gross Loss:       ₹{metrics['gross_loss']:,.2f}")
        print(f"Profit Factor:    {metrics['profit_factor']}")
        print(f"Avg Win:          ₹{metrics['avg_win']:,.2f}")
        print(f"Avg Loss:         ₹{metrics['avg_loss']:,.2f}")
        print(f"Avg R Multiple:   {metrics['avg_r_multiple']}")
        print(f"Avg R (Wins):     {metrics['avg_r_win']}")
        print(f"Avg R (Losses):   {metrics['avg_r_loss']}")
        print(f"Expectancy:       ₹{metrics['expectancy']:,.2f}")
        print(f"Max Drawdown:     ₹{metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']}%)")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']}")
        print(f"Total Charges:    ₹{metrics['total_charges']:,.2f}")
        
        print(f"\n{'='*60}")
        print(f"TOP 10 STOCKS BY P&L")
        print(f"{'='*60}")
        top = sorted(stock_results.items(), key=lambda x: x[1].get("net_pnl", 0), reverse=True)[:10]
        print(f"{'Symbol':15s} {'Trades':>8s} {'Net P&L':>12s} {'Win Rate':>10s} {'Avg R':>8s}")
        print("-" * 55)
        for sym, res in top:
            print(f"{sym:15s} {res['trades']:>8} ₹{res['net_pnl']:>8,.2f} {res['win_rate']:>9.1f}% {res['avg_r']:>7.2f}")
        
        print(f"\nResults saved to: {OUTPUT_DIR}/")
        print(f"  Trade book:    trade_book.csv")
        print(f"  Stock summary: stock_summary.csv")
    else:
        print("\nNo trades generated.")

if __name__ == "__main__":
    main()
