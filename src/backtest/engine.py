"""
Phase 19 — Walk-Forward Backtesting Framework.
Tests the discovery score as a trading signal with strict no-lookahead.
Uses entry/exit prices to compute actual returns.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def walk_forward_score_backtest(timeframe="1day", top_n=5, rebalance_days=5):
    """
    Walk-forward backtest: each day, score all stocks, hold top N for rebalance_days.
    Uses entry/exit close prices for actual return computation.
    """
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            timeframe VARCHAR,
            start_date DATE,
            end_date DATE,
            strategy VARCHAR,
            total_return DOUBLE,
            annual_return DOUBLE,
            sharpe DOUBLE,
            max_drawdown DOUBLE,
            win_rate DOUBLE,
            avg_return DOUBLE,
            num_trades INT
        )
    """)

    # Load all predictions with close prices
    print(f"  Loading data for {timeframe} walk-forward...")
    data = con.execute(f"""
        SELECT p.symbol, p.datetime::DATE as date,
               p.score as ml_score, p.expected_return,
               f.close, f.atr_14
        FROM ml_predictions p
        JOIN feature_store f ON p.symbol=f.symbol AND p.timeframe=f.timeframe AND p.datetime=f.datetime
        WHERE p.timeframe=? AND p.model_name='xgb_classifier'
          AND f.close IS NOT NULL
        ORDER BY p.datetime
    """, [timeframe]).fetchdf()
    if len(data) == 0:
        print("  No data available")
        con.close()
        return
    data["date"] = pd.to_datetime(data["date"])

    # Build price lookup: symbol -> {date -> close}
    price_map = defaultdict(dict)
    for _, r in data.iterrows():
        price_map[r["symbol"]][r["date"]] = r["close"]

    # Score rank per day
    data["score_rank"] = data.groupby("date")["ml_score"].rank(ascending=False, pct=True)

    # Walk-forward simulation
    dates = sorted(data["date"].unique())
    positions = {}
    portfolio = []
    balance = 1.0

    for i, date in enumerate(dates):
        day_data = data[data["date"] == date].sort_values("score_rank")

        # Close positions held for rebalance_days or longer
        close_symbols = [s for s, info in positions.items()
                         if (date - info["entry_date"]).days >= rebalance_days]
        day_pnl = 0.0
        for sym in close_symbols:
            info = positions.pop(sym)
            entry_close = info["entry_price"]
            exit_close = price_map.get(sym, {}).get(date)
            if exit_close is not None and entry_close > 0:
                actual_ret = (exit_close / entry_close) - 1
            else:
                actual_ret = 0
            portfolio.append({
                "entry_date": info["entry_date"], "exit_date": date,
                "symbol": sym, "direction": "long",
                "expected_return": info.get("expected_return", 0),
                "actual_return": actual_ret,
                "ml_score": info["ml_score"]
            })
            # Each position gets equal capital share
            day_pnl += actual_ret / top_n

        # Apply PnL for this day's closes (one portfolio update per day)
        if day_pnl != 0:
            balance *= (1 + day_pnl)

        # Enter top N positions (skip if already held)
        for _, row in day_data.iterrows():
            if row["symbol"] in positions:
                continue
            if len(positions) >= top_n:
                break
            positions[row["symbol"]] = {
                "entry_date": date,
                "entry_price": row["close"],
                "ml_score": row["ml_score"],
                "expected_return": row["expected_return"]
            }

    # Close remaining at last date
    last_date = dates[-1]
    day_pnl = 0.0
    for sym, info in list(positions.items()):
        exit_close = price_map.get(sym, {}).get(last_date)
        if exit_close is not None and info["entry_price"] > 0:
            actual_ret = (exit_close / info["entry_price"]) - 1
        else:
            actual_ret = 0
        portfolio.append({
            "entry_date": info["entry_date"], "exit_date": last_date,
            "symbol": sym, "direction": "long",
            "expected_return": info.get("expected_return", 0),
            "actual_return": actual_ret,
            "ml_score": info["ml_score"]
        })
        day_pnl += actual_ret / top_n
    if day_pnl != 0:
        balance *= (1 + day_pnl)

    if len(portfolio) == 0:
        print("  No trades executed")
        con.close()
        return

    df_port = pd.DataFrame(portfolio)

    # Metrics from balance curve
    total_ret = balance - 1
    avg_ret = df_port["actual_return"].mean()
    win_rate = (df_port["actual_return"] > 0).mean()

    # Sharpe from daily portfolio returns
    df_port["exit_date"] = pd.to_datetime(df_port["exit_date"])
    daily_returns = df_port.groupby("exit_date")["actual_return"].mean()  # avg of positions closing that day
    sharpe = 0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    # Max drawdown from balance
    max_dd = 0
    peak = 1.0
    for _, r in df_port.sort_values("exit_date").iterrows():
        # Simpler: track balance change per close event
        pass
    # Use portfolio-level simulation for max DD
    eq_curve = df_port.sort_values("exit_date").copy()
    eq_curve["cum_pnl"] = eq_curve["actual_return"].cumsum() / top_n
    max_dd = 0
    peak = 0
    for v in eq_curve["cum_pnl"].values:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    max_dd = min(max_dd, 1.0)

    years = max(1, (dates[-1] - dates[0]).days / 365)
    ann_ret = (1 + total_ret) ** (1 / years) - 1

    # Decile analysis
    df_port["decile"] = (df_port["ml_score"].rank(pct=True) * 10).astype(int).clip(0, 9)
    decile_perf = df_port.groupby("decile").agg(
        count=("actual_return", "count"),
        avg_ret=("actual_return", "mean"),
        win_rate=("actual_return", lambda x: (x > 0).mean()),
        total_ret=("actual_return", "sum")
    ).reset_index()

    # Store
    con.execute(
        "INSERT INTO backtest_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [timeframe, dates[0].date(), dates[-1].date(),
         f"top{top_n}_rebal{rebalance_days}d",
         float(total_ret), float(ann_ret),
         float(sharpe), float(max_dd),
         float(win_rate), float(avg_ret),
         int(len(df_port))]
    )
    con.close()

    print(f"  Period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
    print(f"  Strategy: Top {top_n} by ML score, rebalanced every {rebalance_days}d")
    print(f"  Trades: {len(df_port)}")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Annual return: {ann_ret:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Avg actual ret: {avg_ret:.1%}, expected: {df_port['expected_return'].mean():.1%}")
    print(f"  Best: {df_port['actual_return'].max():.1%}, Worst: {df_port['actual_return'].min():.1%}")
    print(f"\n  Decile analysis:")
    for _, r in decile_perf.iterrows():
        print(f"    D{int(r['decile'])}: cnt={int(r['count']):4d} avg_ret={r['avg_ret']:.2%} win={r['win_rate']:.1%} total={r['total_ret']:.1%}")


def run_all(timeframes=["1day"]):
    for tf in timeframes:
        walk_forward_score_backtest(tf, top_n=3, rebalance_days=5)
        walk_forward_score_backtest(tf, top_n=5, rebalance_days=5)
        walk_forward_score_backtest(tf, top_n=5, rebalance_days=10)
        walk_forward_score_backtest(tf, top_n=10, rebalance_days=5)


if __name__ == "__main__":
    run_all()
