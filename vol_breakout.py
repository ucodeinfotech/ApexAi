"""
Volatility Breakout — Swing Momentum Strategy
Entry: daily return > +1% AND volume > 1.5-2x 20d avg -> LONG at close, hold 5 days
"""
import duckdb, pandas as pd, numpy as np, json, warnings, os
warnings.filterwarnings("ignore")

DB_PATH = "warehouse/market_data.duckdb"
TOP_N = 20

# === Top 20 stocks with optimized params ===
UNIVERSE = {
    "BAJAJFINSV": (20, 2.0), "IRFC": (30, 1.5), "HSCL": (20, 2.0),
    "JINDALSTEL": (20, 2.0), "KPITTECH": (20, 2.0), "VOLTAS": (20, 2.0),
    "BLUESTARCO": (15, 1.5), "M&M": (20, 2.0), "ANGELONE": (10, 1.5),
    "PCJEWELLER": (20, 2.0), "LUXIND": (30, 1.5), "BEL": (20, 2.0),
    "HAL": (10, 1.5), "ALKEM": (20, 2.0), "SUVEN": (15, 1.5),
    "SHREECEM": (20, 2.0), "ADANIGREEN": (20, 2.0), "ASHOKA": (20, 2.0),
    "APLAPOLLO": (15, 1.5), "TATACONSUM": (20, 2.0),
}

def scan_today(con):
    """Scan for signals today. Returns DataFrame of new signals."""
    rows = []
    for sym, (lb, mult) in UNIVERSE.items():
        df = con.execute(f"""
            SELECT datetime, close, volume 
            FROM raw_market 
            WHERE symbol = '{sym.replace(chr(39), chr(39)+chr(39))}' 
              AND timeframe = '1day' 
            ORDER BY datetime DESC LIMIT {lb + 5}
        """).fetchdf()
        if len(df) < lb + 1: continue
        df = df.sort_values("datetime").reset_index(drop=True)
        close = df["close"].values
        volume = df["volume"].values
        avg_vol = pd.Series(volume).rolling(lb).mean().shift(1).values
        ret = pd.Series(close).pct_change(1).values
        i = len(df) - 1  # latest bar
        if np.isnan(avg_vol[i]) or np.isnan(ret[i]): continue
        signal = ret[i] > 0.01 and volume[i] > avg_vol[i] * mult
        if signal:
            rows.append({
                "symbol": sym, "entry_date": str(df["datetime"].iloc[i])[:10],
                "entry_price": round(close[i], 2), "return_pct": round(ret[i] * 100, 2),
                "volume_ratio": round(volume[i] / avg_vol[i], 2),
                "lookback": lb, "vol_mult": mult,
            })
    return pd.DataFrame(rows)

def backtest_portfolio(con):
    """Full backtest of the strategy portfolio (all-time)."""
    trades = []
    for sym, (lb, mult) in UNIVERSE.items():
        df = con.execute(f"""
            SELECT datetime, close, volume 
            FROM raw_market 
            WHERE symbol = '{sym.replace(chr(39), chr(39)+chr(39))}' 
              AND timeframe = '1day' 
            ORDER BY datetime
        """).fetchdf()
        if len(df) < lb + 1: continue
        df["avg_vol"] = df["volume"].rolling(lb).mean().shift(1)
        df["ret"] = df["close"].pct_change(1)
        df["signal"] = (df["ret"] > 0.01) & (df["volume"] > df["avg_vol"] * mult)
        signal_idx = df.index[df["signal"]].tolist()
        for idx in signal_idx:
            exit_idx = min(idx + 5, len(df) - 1)
            if exit_idx <= idx: continue
            trades.append({
                "symbol": sym, "entry_date": str(df["datetime"].iloc[idx])[:10],
                "exit_date": str(df["datetime"].iloc[exit_idx])[:10],
                "entry_price": round(df["close"].iloc[idx], 2),
                "exit_price": round(df["close"].iloc[exit_idx], 2),
                "return_pct": round((df["close"].iloc[exit_idx] / df["close"].iloc[idx] - 1) * 100, 2),
                "params": f"{lb}/{mult}",
            })
    return pd.DataFrame(trades)

def print_summary(trades):
    """Print strategy summary."""
    print(f"Strategy: Vol Breakout Swing Momentum")
    print(f"Universe: {len(UNIVERSE)} stocks (top 20 by Sharpe)")
    print(f"Period:   {trades['entry_date'].min()} to {trades['entry_date'].max()}")
    print(f"Trades:   {len(trades):,}")
    print(f"Win Rate: {(trades['return_pct'] > 0).mean():.1%}")
    print(f"Avg Ret:  +{trades['return_pct'].mean():.2f}% (5-day hold)")
    print(f"Sharpe:   {trades['return_pct'].mean() / trades['return_pct'].std() * np.sqrt(252/5):.2f}")
    print(f"Best:     +{trades['return_pct'].max():.1f}%")
    print(f"Worst:    {trades['return_pct'].min():.1f}%")
    print(f"\nTop 5 stocks by avg return:")
    top = trades.groupby("symbol").agg(
        trades=("return_pct", "count"), avg_ret=("return_pct", "mean"),
        wr=("return_pct", lambda x: (x > 0).mean())
    ).sort_values("avg_ret", ascending=False).head(5)
    for sym, r in top.iterrows():
        print(f"  {sym:15s}  {int(r['trades']):3d} trades  +{r['avg_ret']:.2f}% avg  {r['wr']:.0%} WR")
    print(f"\nMost recent 5 trades:")
    print(trades.tail(5).to_string(index=False))

if __name__ == "__main__":
    con = duckdb.connect("warehouse/market_data.duckdb")
    
    # Backtest
    print("=" * 60)
    trades = backtest_portfolio(con)
    print_summary(trades)
    
    # Today's signals
    print("\n" + "=" * 60)
    print("TODAY'S SIGNALS")
    signals = scan_today(con)
    if len(signals) > 0:
        print(signals.to_string(index=False))
    else:
        print("No signals today. Conditions not met for any stock in universe.")
    
    con.close()
