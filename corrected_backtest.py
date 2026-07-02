"""Vol Breakout - Corrected portfolio backtest with proper daily simulation"""
import duckdb, pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")

DB_PATH = "warehouse/market_data.duckdb"
INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.05  # 5% per stock

UNIVERSE = {
    "BAJAJFINSV": (20,2.0),"IRFC": (30,1.5),"HSCL": (20,2.0),"JINDALSTEL": (20,2.0),
    "KPITTECH": (20,2.0),"VOLTAS": (20,2.0),"BLUESTARCO": (15,1.5),"M&M": (20,2.0),
    "ANGELONE": (10,1.5),"PCJEWELLER": (20,2.0),"LUXIND": (30,1.5),"BEL": (20,2.0),
    "HAL": (10,1.5),"ALKEM": (20,2.0),"SUVEN": (15,1.5),"SHREECEM": (20,2.0),
    "ADANIGREEN": (20,2.0),"ASHOKA": (20,2.0),"APLAPOLLO": (15,1.5),"TATACONSUM": (20,2.0),
}

# Step 1: Get all daily prices for universe stocks + generate signals
con = duckdb.connect(DB_PATH)
all_data = {}  # symbol -> DataFrame with date, close, signal
for sym, (lb, mult) in UNIVERSE.items():
    df = con.execute(f"SELECT datetime::DATE as date, close, volume FROM raw_market WHERE symbol='{sym.replace(chr(39),chr(39)+chr(39))}' AND timeframe='1day' ORDER BY datetime").fetchdf()
    if len(df) < lb + 2: continue
    df["avg_vol"] = df["volume"].rolling(lb).mean().shift(1)
    df["ret"] = df["close"].pct_change(1)
    df["signal"] = (df["ret"] > 0.01) & (df["volume"] > df["avg_vol"] * mult)
    # Price lookup by date
    all_data[sym] = df

# Build date range from all data
all_dates = sorted(set().union(*[set(df["date"]) for df in all_data.values()]))
all_dates = [d for d in all_dates if d >= pd.Timestamp("2016-10-01") and d <= pd.Timestamp("2026-06-17")]
date_to_idx = {d: i for i, d in enumerate(all_dates)}
print(f"Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} trading days)")

# Step 2: Portfolio simulation
# Each position: {symbol, entry_date, entry_price, exit_date, shares, capital_allocated}
positions = []
equity = [INITIAL_CAPITAL]
cash = INITIAL_CAPITAL
trades_log = []
daily_positions = []

for day_idx, today in enumerate(all_dates):
    # --- Exits: close positions expiring today ---
    exited = []
    for pos in positions:
        if pos["exit_date"] == today:
            # Close at today's close
            df = all_data.get(pos["symbol"])
            if df is not None:
                row = df[df["date"] == today]
                if len(row) > 0:
                    exit_price = row["close"].values[0]
                    proceeds = pos["shares"] * exit_price
                    ret_pct = (exit_price / pos["entry_price"] - 1) * 100
                    trades_log.append({
                        "symbol": pos["symbol"],
                        "entry_date": str(pos["entry_date"])[:10],
                        "exit_date": str(today)[:10],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "return_pct": round(ret_pct, 2),
                        "hold_days": (today - pos["entry_date"]).days,
                    })
                    cash += proceeds
                    exited.append(pos)
    for e in exited:
        positions.remove(e)

    # --- Entries: new signals today ---
    for sym, df in all_data.items():
        row = df[df["date"] == today]
        if len(row) == 0: continue
        if not row["signal"].values[0]: continue
        entry_price = row["close"].values[0]
        if entry_price <= 0: continue
        # Enter at close, hold 5 trading days
        future_dates = [d for d in all_dates if d > today]
        exit_date = future_dates[4] if len(future_dates) >= 5 else future_dates[-1] if future_dates else today
        if exit_date == today: continue
        # Allocate capital
        alloc = INITIAL_CAPITAL * POSITION_SIZE
        shares = int(alloc / entry_price)
        if shares <= 0: continue
        cost = shares * entry_price
        if cost > cash: continue  # skip if not enough cash (rare with overlapping)
        cash -= cost
        positions.append({
            "symbol": sym,
            "entry_date": today,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "shares": shares,
            "capital_allocated": cost,
        })

    # --- Mark to market ---
    pv = 0
    for pos in positions:
        df = all_data.get(pos["symbol"])
        if df is None: continue
        row = df[df["date"] == today]
        if len(row) == 0:
            pv += pos["shares"] * pos["entry_price"]
        else:
            pv += pos["shares"] * row["close"].values[0]
    total_value = cash + pv
    equity.append(total_value)
    daily_positions.append({
        "date": today, "cash": cash, "positions_value": pv,
        "total": total_value, "n_positions": len(positions),
        "return": (total_value / equity[-2] - 1) if len(equity) > 1 else 0,
    })

con.close()

# Step 3: Calculate metrics
df_eq = pd.DataFrame(daily_positions)
df_eq["daily_ret"] = df_eq["return"]
df_trades = pd.DataFrame(trades_log)

total_trades = len(df_trades)
win_rate = (df_trades["return_pct"] > 0).mean()
avg_ret = df_trades["return_pct"].mean()
med_ret = df_trades["return_pct"].median()
best_trade = df_trades["return_pct"].max()
worst_trade = df_trades["return_pct"].min()
avg_win = df_trades[df_trades["return_pct"] > 0]["return_pct"].mean()
avg_loss = df_trades[df_trades["return_pct"] < 0]["return_pct"].mean()

# Daily returns for Sharpe
daily_rets = df_eq["daily_ret"].values[1:]  # skip first (0)
sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
ann_ret = (equity[-1] / INITIAL_CAPITAL) ** (252 / len(all_dates)) - 1
max_dd = 0
peak = equity[0]
for v in equity:
    if v > peak: peak = v
    dd = (v - peak) / peak
    if dd < max_dd: max_dd = dd

print("=" * 60)
print("CORRECTED PORTFOLIO BACKTEST")
print("=" * 60)
print(f"{'Initial Capital':30s} Rs {INITIAL_CAPITAL:>8,}")
print(f"{'Final Equity':30s} Rs {equity[-1]:>10,.0f}")
print(f"{'Total Return':30s} {(equity[-1]/INITIAL_CAPITAL-1)*100:>7.1f}%")
print(f"{'Annualized Return':30s} {ann_ret*100:>7.1f}%")
print(f"{'Sharpe Ratio (daily)':30s} {sharpe:>8.2f}")
print(f"{'Max Drawdown':30s} {max_dd*100:>7.1f}%")
print(f"{'Total Trades':30s} {total_trades:>8,}")
print(f"{'Win Rate':30s} {win_rate:>7.1%}")
print(f"{'Avg Trade Return':30s} +{avg_ret:>5.2f}%")
print(f"{'Median Trade Return':30s} +{med_ret:>5.2f}%")
print(f"{'Avg Win':30s} +{avg_win:>5.2f}%")
print(f"{'Avg Loss':30s} {avg_loss:>5.2f}%")
print(f"{'Best Trade':30s} +{best_trade:>5.1f}%")
print(f"{'Worst Trade':30s} {worst_trade:>5.1f}%")
print(f"{'Avg Concurrent Positions':30s} {df_eq['n_positions'].mean():>8.1f}")
print(f"{'Max Concurrent Positions':30s} {df_eq['n_positions'].max():>8,d}")
print(f"{'Trading Days':30s} {len(all_dates):>8,}")

# Yearly from equity curve
print(f"\n{'='*60}")
print("YEARLY PERFORMANCE")
print(f"{'='*60}")
df_eq["year"] = pd.to_datetime(df_eq["date"]).dt.year
yearly = df_eq.groupby("year").agg(
    start_eq=("total", "first"),
    end_eq=("total", "last"),
    trades=("n_positions", "sum")
).reset_index()
for _, r in yearly.iterrows():
    yr_ret = (r["end_eq"] / r["start_eq"] - 1) * 100
    print(f"  {int(r['year'])}: {'Rs '}{r['start_eq']:>9,.0f} -> {'Rs '}{r['end_eq']:>9,.0f}  {yr_ret:>+7.2f}%")

# Monthly from equity curve
print(f"\n{'='*60}")
print("MONTHLY PERFORMANCE")
print(f"{'='*60}")
df_eq["ym"] = pd.to_datetime(df_eq["date"]).dt.strftime("%Y-%m")
monthly = df_eq.groupby("ym").agg(
    start_eq=("total", "first"),
    end_eq=("total", "last"),
).reset_index().sort_values("ym")
print(f"{'Month':>8s} {'Start':>10s} {'End':>10s} {'Return':>8s} {'Trades':>7s}")
print("-" * 50)
for _, r in monthly.iterrows():
    mr = (r["end_eq"] / r["start_eq"] - 1) * 100
    nt = len(df_trades[df_trades["entry_date"].str.startswith(r["ym"])])
    print(f"  {r['ym']} {'Rs '}{r['start_eq']:>8,.0f} {'Rs '}{r['end_eq']:>8,.0f} {mr:>+7.2f}% {nt:>5d}")

# Best/worst months
best_m = monthly.loc[monthly["end_eq"].div(monthly["start_eq"]).idxmax()]
worst_m = monthly.loc[monthly["end_eq"].div(monthly["start_eq"]).idxmin()]
print(f"\nBest month:  {best_m['ym']} ({(best_m['end_eq']/best_m['start_eq']-1)*100:+.2f}%)")
print(f"Worst month: {worst_m['ym']} ({(worst_m['end_eq']/worst_m['start_eq']-1)*100:+.2f}%)")

# Save trade log
df_trades.to_csv("vol_breakout_trades_corrected.csv", index=False)
print(f"\nTrade log saved (vol_breakout_trades_corrected.csv)")
