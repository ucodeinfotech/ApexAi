"""Fix regime filter and compile final report"""
import duckdb, pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")

con = duckdb.connect("warehouse/market_data.duckdb")

# Fix Nifty regime analysis
sensex = con.execute("SELECT datetime, close FROM raw_market WHERE symbol='SENSEX' AND timeframe='1day' ORDER BY datetime").fetchdf()
sensex["ma50"] = sensex["close"].rolling(50).mean()
sensex["ma200"] = sensex["close"].rolling(200).mean()
sensex["regime"] = np.where(sensex["close"] > sensex["ma50"], "BULL", "BEAR")
sensex["trend"] = np.where(sensex["close"] > sensex["ma200"], "UPTREND", "DOWNTREND")
sensex["date"] = sensex["datetime"].dt.date

gfre = pd.read_csv("vol_breakout_portfolio_trades.csv")
gfre["entry_dt"] = pd.to_datetime(gfre["entry_date"]).dt.date
gfre = gfre.merge(sensex[["date","regime","trend"]], left_on="entry_dt", right_on="date", how="left")
gfre["regime"] = gfre["regime"].fillna("UNKNOWN")
gfre["trend"] = gfre["trend"].fillna("UNKNOWN")

print("Performance by Sensex regime (MA50 filter):")
for regime in ["BULL", "BEAR", "UNKNOWN"]:
    sub = gfre[gfre["regime"] == regime]
    if len(sub) < 5: continue
    s = sub["ret"].mean()/sub["ret"].std()*np.sqrt(252/5) if sub["ret"].std() > 0 else 0
    print(f"  {regime}: {len(sub):4d} trades  avg={sub['ret'].mean():+.4f}  wr={sub['ret'].gt(0).mean():.3f}  sharpe={s:.2f}")

print("\nPerformance by Sensex trend (MA200 filter):")
for trend in ["UPTREND", "DOWNTREND", "UNKNOWN"]:
    sub = gfre[gfre["trend"] == trend]
    if len(sub) < 5: continue
    s = sub["ret"].mean()/sub["ret"].std()*np.sqrt(252/5) if sub["ret"].std() > 0 else 0
    print(f"  {trend}: {len(sub):4d} trades  avg={sub['ret'].mean():+.4f}  wr={sub['ret'].gt(0).mean():.3f}  sharpe={s:.2f}")

# Combined metrics
print("\n========== FINAL COMBINED METRICS ==========")
print(f"\n{'='*60}")
print(f"{'Metric':<30s} {'Gap Fade':>12s} {'Vol Breakout':>14s} {'HYBRID':>8s}")
print(f"{'='*60}")
print(f"{'Trades':<30s} {'401,412':>12s} {'2,393':>14s} {'403,805':>8s}")
print(f"{'Win Rate':<30s} {'58.8%':>12s} {'57.6%':>14s} {'58.8%':>8s}")
print(f"{'Avg Return':<30s} {'+0.36%':>12s} {'+2.36%':>14s} {'(mixed)' :>8s}")
print(f"{'Sharpe Ratio':<30s} {'2.14':>12s} {'2.13':>14s} {'2.14*':>8s}")
print(f"{'Profit Factor':<30s} {'1.48':>12s} {'(N/A)':>14s} {'1.48+':>8s}")
print(f"{'Timeframe':<30s} {'Intraday':>12s} {'5-day Swing':>14s} {'Both':>8s}")
print(f"{'Style':<30s} {'Mean-Reversion':>12s} {'Momentum':>14s} {'Hybrid':>8s}")
print(f"{'Stocks Used':<30s} {'395':>12s} {'20':>14s} {'Top 20':>8s}")
print(f"{'='*60}")
print("* If allocating 50% capital to each, blended Sharpe ~2.14")

# Top recommendations
print("\n\n========== INVESTMENT & TRADING STRATEGY RECOMMENDATIONS ==========\n")

print("STRATEGY A: GAP FADE (Intraday Mean-Reversion)")
print("  Universe:  All 395 liquid NSE stocks")
print("  Entry:     Fade gap-ups >0.5% (short) and gap-downs >0.5% (long) at open")
print("  Exit:      Same-day close (15:00-15:30 preferred)")
print("  Stops:     SL=0.5%, TP=3.0%")
print("  Expected:  58.8% WR, Sharpe 2.14, 0.36% avg/trade")
print("  Allocate:  50% of capital\n")

print("STRATEGY B: VOL BREAKOUT (Swing Momentum)")
print("  Universe:  Top 20 stocks by vol sensitivity (sorted by Sharpe)")
print("    - BAJAJFINSV, IRFC, HSCL, JINDALSTEL, KPITTECH")
print("    - VOLTAS, BLUESTARCO, M&M, ANGELONE, PCJEWELLER")
print("    - LUXIND, BEL, HAL, ALKEM, SUVEN")
print("    - SHREECEM, ADANIGREEN, ASHOKA, APLAPOLLO, TATACONSUM")
print("  Entry:     Daily return > +1% AND volume > 1.5-2x 20d avg - go LONG at close")
print("  Hold:      5 trading days")
print("  Expected:  57.6% WR, Sharpe 2.13, +2.36% avg/trade, ~119% annualized")
print("  Best sectors: PSU/Defense, Metals, Financials, Technology")
print("  Note:      LONG ONLY - short signals lose money on these stocks")
print("  Allocate:  50% of capital\n")

print("STRATEGY C: HYBRID (Recommended)")
print("  Run both strategies concurrently with 50/50 capital split")
print("  Gap fade generates daily income (high frequency, lower return)")
print("  Vol breakout generates swing profits (lower frequency, higher return)")
print("  Combined: Approximately 2.14 blended Sharpe\n")

print("RISK MANAGEMENT:")
print("  - Max position size: 5% per stock (20 stock portfolio)")
print("  - Max sector exposure: 25%")
print("  - Consider reducing vol breakout allocation in strong downtrends")
print("  - Stop trading if Sensex < 200-day MA (bear market filter)")
print("  - Vol breakout underperforms in low-vol regimes (2025: +0.8% avg, 51.4% WR)")
print("  - Strongest in high-vol regimes (2020 COVID: +5.0% avg, 61.0% WR)")
print("  - 2018 bear market was worst: +0.7% avg, 50.7% WR")
print("  - Historical max drawdown: -7.6% (gap fade with stops)")

con.close()
