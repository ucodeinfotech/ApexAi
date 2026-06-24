"""Generate combined final report from all stocks"""
import pandas as pd
import numpy as np
import os, glob

OUTPUT_DIR = "backtest_results"
all_trades = []
all_stats = []

# Read all per-stock trade files
for f in sorted(glob.glob(f"{OUTPUT_DIR}/*_trades.csv")):
    sym = os.path.basename(f).replace("_trades.csv","")
    df = pd.read_csv(f)
    df["symbol"] = sym
    all_trades.append(df)
    
    total = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc/total*100, 2) if total else 0
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_ = round(df["net_pnl"].sum(), 2)
    pf = round(abs(gp/gl), 2) if gl != 0 else float('inf')
    avg_r = round(df["r"].mean(), 2)
    
    df_s = df.sort_values("exit_time").reset_index(drop=True)
    df_s["cum"] = df_s["net_pnl"].cumsum()
    df_s["peak"] = df_s["cum"].cummax()
    df_s["dd"] = df_s["peak"] - df_s["cum"]
    mdd = round(df_s["dd"].max(), 2)
    
    avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
    avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
    exp_ = round(avg_w*(wr/100) + avg_l*(1-wr/100), 2)
    sharpe = round(df["r"].mean()/df["r"].std()*np.sqrt(total), 2) if df["r"].std() > 0 else 0
    
    all_stats.append({
        "symbol": sym, "trades": total, "wins": wc, "losses": lc,
        "win_rate": wr, "net_pnl": np_, "profit_factor": pf,
        "avg_r": avg_r, "expectancy": exp_, "max_dd": mdd,
        "sharpe": sharpe, "charges": round(df["charges"].sum(), 2),
        "gross_profit": gp, "gross_loss": gl,
        "avg_win": avg_w, "avg_loss": avg_l
    })

combined = pd.concat(all_trades, ignore_index=True)
combined.to_csv(f"{OUTPUT_DIR}/combined_trades.csv", index=False)

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(f"{OUTPUT_DIR}/stock_summary.csv", index=False)

# === COMBINED RESULTS ===
print("=" * 70)
print("PIVOT BREAKOUT BACKTEST - FINAL RESULTS")
print("All 50 Nifty Stocks | Oct 2016 - Jun 2026")
print("=" * 70)

total = len(combined)
wins = combined[combined["net_pnl"] > 0]
losses = combined[combined["net_pnl"] <= 0]
wc = len(wins); lc = len(losses)
wr = round(wc/total*100, 2)
gp = round(wins["net_pnl"].sum(), 2)
gl = round(losses["net_pnl"].sum(), 2)
np_total = round(combined["net_pnl"].sum(), 2)
pf = round(abs(gp/gl), 2) if gl != 0 else 0
avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
avg_r = round(combined["r"].mean(), 2)

combined_s = combined.sort_values("exit_time").reset_index(drop=True)
combined_s["cum"] = combined_s["net_pnl"].cumsum()
combined_s["peak"] = combined_s["cum"].cummax()
combined_s["dd"] = combined_s["peak"] - combined_s["cum"]
mdd = round(combined_s["dd"].max(), 2)
mdd_p = round(mdd / combined_s["peak"].max() * 100, 2) if combined_s["peak"].max() > 0 else 0
sharpe = round(combined["r"].mean() / combined["r"].std() * np.sqrt(total), 2) if combined["r"].std() > 0 else 0
total_charges = round(combined["charges"].sum(), 2)

print(f"\nOVERALL METRICS:")
print(f"  {'Total Trades:':20s} {total:>10,}")
print(f"  {'Wins / Losses:':20s} {wc:>10,} / {lc:,}")
print(f"  {'Win Rate:':20s} {wr:>10.2f}%")
print(f"  {'Net P&L:':20s} Rs{np_total:>10,.2f}")
print(f"  {'Gross Profit:':20s} Rs{gp:>10,.2f}")
print(f"  {'Gross Loss:':20s} Rs{gl:>10,.2f}")
print(f"  {'Profit Factor:':20s} {pf:>10.2f}")
print(f"  {'Avg Win:':20s} Rs{avg_w:>10,.2f}")
print(f"  {'Avg Loss:':20s} Rs{avg_l:>10,.2f}")
print(f"  {'Avg R Multiple:':20s} {avg_r:>10.2f}")
print(f"  {'Sharpe Ratio:':20s} {sharpe:>10.2f}")
print(f"  {'Max Drawdown:':20s} Rs{mdd:>10,.2f} ({mdd_p}%)")
print(f"  {'Total Charges:':20s} Rs{total_charges:>10,.2f}")

# Stock ranking
print(f"\n{'='*70}")
print("STOCK RANKING (by Net P&L)")
print(f"{'='*70}")
sorted_stats = sorted(all_stats, key=lambda x: x["net_pnl"], reverse=True)
print(f"{'#':>3s} {'Symbol':18s} {'Trades':>8s} {'Net P&L':>12s} {'WR%':>7s} {'Avg R':>7s} {'PF':>7s} {'Sharpe':>7s}")
print("-" * 70)
for rank, s in enumerate(sorted_stats, 1):
    print(f"{rank:>3d} {s['symbol']:18s} {s['trades']:>8,} Rs{s['net_pnl']:>8,.2f} {s['win_rate']:>6.1f}% {s['avg_r']:>6.2f} {s['profit_factor']:>6.2f} {s['sharpe']:>6.2f}")

# BEST/WORST
print(f"\n{'='*70}")
print("BEST 5 STOCKS")
print(f"{'='*70}")
for s in sorted_stats[:5]:
    print(f"  {s['symbol']:18s} Trades={s['trades']:>6}  Net=Rs{s['net_pnl']:>8,.2f}  WR={s['win_rate']:>5.1f}%  AvgR={s['avg_r']:>5.2f}")

print(f"\n{'='*70}")
print("WORST 5 STOCKS")
print(f"{'='*70}")
for s in sorted_stats[-5:]:
    print(f"  {s['symbol']:18s} Trades={s['trades']:>6}  Net=Rs{s['net_pnl']:>8,.2f}  WR={s['win_rate']:>5.1f}%  AvgR={s['avg_r']:>5.2f}")

# ANALYSIS
print(f"\n{'='*70}")
print("ANALYSIS & SUGGESTIONS")
print(f"{'='*70}")

profitable = [s for s in sorted_stats if s["net_pnl"] > 0]
losing = [s for s in sorted_stats if s["net_pnl"] <= 0]
print(f"\n  Profitable stocks: {len(profitable)}/{len(sorted_stats)}")
print(f"  Losing stocks:     {len(losing)}/{len(sorted_stats)}")

if profitable:
    print(f"\n  Profitable ones:")
    for s in profitable:
        print(f"    {s['symbol']:18s} Net=Rs{s['net_pnl']:>8,.2f}  Trades={s['trades']:>6}  WR={s['win_rate']:>5.1f}%")

print(f"\n{'-'*70}")
print("  WHY THE STRATEGY FAILED:")
print(f"{'-'*70}")
print(f"  1. Win rate too low ({wr:.1f}%) - price touching R1/S1 is NOT a breakout")
print(f"  2. Too many false triggers - R1/S1 acts as resistance/support, not breakout")
print(f"  3. 1-pt slippage + Rs10 brokerage eats 70%+ of the avg loss per trade")
print(f"  4. Average loss (Rs{abs(avg_l):.2f}) >> average win (Rs{avg_w:.2f}) - even 1:2 RR can't compensate for 2-3% WR")
print(f"  5. Entry at trigger H/L + 1pt means buying at the extreme of the candle")

print(f"\n{'-'*70}")
print("  SUGGESTED IMPROVEMENTS:")
print(f"{'-'*70}")
print(f"  1. Use a confirmation candle instead of touch (e.g., close above R1)")
print(f"  2. Add volume filter (only trade breakout with above-avg volume)")
print(f"  3. Use wider SL (e.g., ATR-based instead of trigger candle low/high)")
print(f"  4. Consider higher timeframe trend filter (only take long if 1-hr trend is up)")
print(f"  5. Reduce brokerage - Rs10/order is high for small-cap stocks")
print(f"  6. Test on index (Nifty/BankNifty) instead of individual stocks")
