"""
Walk-Forward Backtest for range predictions.
Tests probability of >2%/5%/6% next-day range as a trading signal.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"
RANGE_TARGETS = ["hr_2pct", "hr_5pct", "hr_6pct"]


def backtest_oos(timeframe="1day", target="hr_2pct", top_n=5):
    con = duckdb.connect(str(DB_PATH))
    model_name = f"xgb_range_{target}"

    print(f"\n  OOS backtest: {target} top {top_n}...")

    # Load OOS predictions + actual prices
    data = con.execute(f"""
        SELECT p.symbol, p.datetime::DATE as date, p.score as pred_prob,
               p.expected_return as pred_range, f.close, f.high, f.low
        FROM ml_predictions_oos p
        JOIN feature_store f ON p.symbol=f.symbol AND p.timeframe=f.timeframe AND p.datetime=f.datetime
        WHERE p.timeframe=? AND p.model_name=?
          AND f.close IS NOT NULL AND f.high IS NOT NULL AND f.low IS NOT NULL
        ORDER BY p.datetime
    """, [timeframe, model_name]).fetchdf()
    if len(data) == 0:
        print("  No OOS predictions")
        con.close()
        return
    data["date"] = pd.to_datetime(data["date"])

    # Compute actual next-day range %
    data["actual_range_pct"] = (data["high"] - data["low"]) / data["close"] * 100

    # Score ranking per day
    data["score_rank"] = data.groupby("date")["pred_prob"].rank(ascending=False, pct=True)

    # Walk-forward simulation: buy top N each day, sell next day
    dates = sorted(data["date"].unique())
    portfolio = []
    balance = 1.0

    for i, date in enumerate(dates):
        if i == len(dates) - 1:
            break  # can't trade last day
        next_date = dates[i + 1]

        day_data = data[data["date"] == date].sort_values("score_rank")
        next_data = data[data["date"] == next_date]

        day_pnl = 0.0
        trades_today = 0

        for _, row in day_data.iterrows():
            if trades_today >= top_n:
                break
            sym = row["symbol"]
            # Find next day data for this symbol
            nd = next_data[next_data["symbol"] == sym]
            if len(nd) == 0:
                continue
            nd = nd.iloc[0]
            entry_price = row["close"]
            exit_price = nd["close"]
            if entry_price > 0:
                ret = (exit_price / entry_price) - 1
            else:
                ret = 0

            actual_range = nd["actual_range_pct"]

            portfolio.append({
                "entry_date": date, "exit_date": next_date,
                "symbol": sym, "actual_return": ret,
                "pred_prob": row["pred_prob"], "pred_range": row["pred_range"],
                "actual_range_pct": actual_range
            })
            day_pnl += ret / top_n
            trades_today += 1

        if day_pnl != 0:
            balance *= (1 + day_pnl)

    if len(portfolio) == 0:
        print("  No trades")
        con.close()
        return

    df_port = pd.DataFrame(portfolio)
    total_ret = balance - 1
    avg_ret = df_port["actual_return"].mean()
    win_rate = (df_port["actual_return"] > 0).mean()
    range_hit_rate = (df_port["actual_range_pct"] > float(target.split("_")[1].replace("pct", ""))).mean()

    daily_ret = df_port.groupby("exit_date")["actual_return"].mean()
    sharpe = 0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    eq = df_port.copy()
    eq["cum_pnl"] = eq["actual_return"].cumsum() / top_n
    max_dd = 0
    peak = 0
    for v in eq["cum_pnl"]:
        if v > peak: peak = v
        dd = peak - v
        if dd > max_dd: max_dd = dd
    max_dd = min(max_dd, 1.0)

    years = max(1, (dates[-1] - dates[0]).days / 365)
    ann_ret = (1 + total_ret) ** (1 / years) - 1

    # Decile analysis by predicted probability
    df_port["decile"] = (df_port["pred_prob"].rank(pct=True) * 10).astype(int).clip(0, 9)
    decile = df_port.groupby("decile").agg(
        count=("actual_return", "count"),
        avg_ret=("actual_return", "mean"),
        win_rate=("actual_return", lambda x: (x > 0).mean()),
        range_hit=("actual_range_pct", lambda x: (x > float(target.split("_")[1].replace("pct", ""))).mean())
    ).reset_index()

    print(f"  Period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
    print(f"  Trades: {len(df_port)}")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Annual return: {ann_ret:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Avg return: {avg_ret:.1%}")
    print(f"  Range hit rate: {range_hit_rate:.1%}")
    print(f"\n  Decile analysis:")
    for _, r in decile.iterrows():
        print(f"    D{int(r['decile'])}: cnt={int(r['count']):4d} ret={r['avg_ret']:.2%} win={r['win_rate']:.1%} range_hit={r['range_hit']:.1%}")

    # Store
    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results_oos (
            timeframe VARCHAR, start_date DATE, end_date DATE,
            strategy VARCHAR, total_return DOUBLE, annual_return DOUBLE,
            sharpe DOUBLE, max_drawdown DOUBLE, win_rate DOUBLE,
            avg_return DOUBLE, num_trades INT
        )
    """)
    con.execute(
        "INSERT INTO backtest_results_oos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [timeframe, dates[0].date(), dates[-1].date(),
         f"range_{target}_top{top_n}",
         float(total_ret), float(ann_ret), float(sharpe),
         float(max_dd), float(win_rate), float(avg_ret), int(len(df_port))]
    )
    con.close()

    return total_ret, sharpe, max_dd


if __name__ == "__main__":
    for target in RANGE_TARGETS:
        for top_n in [3, 5, 10]:
            backtest_oos("1day", target, top_n)
            print()
