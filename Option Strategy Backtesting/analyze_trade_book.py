"""Analyze OR strategy trade books: overlapping days, position sizing, P&L"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "combined_or")
OR_DIR = os.path.join(BASE, "backtest_results", "combined_or")

# ── Load OR Raw+FixTP trade books ──
nifty = pd.read_csv(os.path.join(OR_DIR, "NIFTY50_OR_Raw_FixTP.csv"))
sensex = pd.read_csv(os.path.join(OR_DIR, "SENSEX_OR_Raw_FixTP.csv"))

nifty["exit_time"] = pd.to_datetime(nifty["exit_time"])
sensex["exit_time"] = pd.to_datetime(sensex["exit_time"])
nifty["date"] = nifty["exit_time"].dt.date
sensex["date"] = sensex["exit_time"].dt.date

# ── Parameters ──
CAPITAL_PER_INDEX = 100000  # Rs1L each
NIFTY_LOT = 50   # Nifty futures: 1 lot = 50 units, Rs50/pt
SENSEX_LOT = 10  # Sensex futures: 1 lot = 10 units, Rs10/pt
CHARGES = 20     # Round-trip brokerage Rs10 entry + Rs10 exit

# ── Compute P&L ──
nifty["pnl"] = nifty["points"] * NIFTY_LOT - CHARGES
sensex["pnl"] = sensex["points"] * SENSEX_LOT - CHARGES
nifty["cum_pnl"] = nifty["pnl"].cumsum()
sensex["cum_pnl"] = sensex["pnl"].cumsum()

# ── Overlapping days ──
nifty_dates = set(nifty["date"])
sensex_dates = set(sensex["date"])
overlap_dates = nifty_dates & sensex_dates

nifty_overlap = nifty[nifty["date"].isin(overlap_dates)]
sensex_overlap = sensex[sensex["date"].isin(overlap_dates)]

# Daily combined P&L
nifty_daily = nifty.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl":"nifty_pnl"})
sensex_daily = sensex.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl":"sensex_pnl"})
combined_daily = pd.merge(nifty_daily, sensex_daily, on="date", how="outer").fillna(0)
combined_daily["total_pnl"] = combined_daily["nifty_pnl"] + combined_daily["sensex_pnl"]
combined_daily = combined_daily.sort_values("date").reset_index(drop=True)
combined_daily["cum_pnl"] = combined_daily["total_pnl"].cumsum()

# ── Print analysis ──
print("=" * 75)
print("OR STRATEGY - TRADE BOOK ANALYSIS (OR Raw+FixTP)")
print("=" * 75)

print(f"\nTRADE SUMMARY:")
print(f"  {'':20s} {'NIFTY50':>12s} {'SENSEX':>12s} {'COMBINED':>12s}")
print(f"  {'-'*56}")
print(f"  {'Total Trades':20s} {len(nifty):>8d}      {len(sensex):>8d}      {len(nifty)+len(sensex):>8d}")
print(f"  {'Trading Days':20s} {len(nifty_dates):>8d}      {len(sensex_dates):>8d}      {len(nifty_dates | sensex_dates):>8d}")

n_wins = (nifty["points"]>0).sum()
s_wins = (sensex["points"]>0).sum()
print(f"  {'Wins':20s} {n_wins:>8d} ({n_wins/len(nifty)*100:.0f}%)   {s_wins:>8d} ({s_wins/len(sensex)*100:.0f}%)")
n_loss = (nifty["points"]<=0).sum()
s_loss = (sensex["points"]<=0).sum()
print(f"  {'Losses':20s} {n_loss:>8d} ({n_loss/len(nifty)*100:.0f}%)   {s_loss:>8d} ({s_loss/len(sensex)*100:.0f}%)")

print(f"\nOVERLAPPING DAYS: {len(overlap_dates)}")
print(f"  NIFTY50 trades on overlap days: {len(nifty_overlap)}")
print(f"  SENSEX trades on overlap days: {len(sensex_overlap)}")
if len(overlap_dates) > 0:
    print(f"  NIFTY50 P&L on overlap: Rs {nifty_overlap['pnl'].sum():+,.0f}")
    print(f"  SENSEX P&L on overlap: Rs {sensex_overlap['pnl'].sum():+,.0f}")
    print(f"  Combined on overlap: Rs {nifty_overlap['pnl'].sum() + sensex_overlap['pnl'].sum():+,.0f}")

print(f"\nP&L ANALYSIS (Rs):")
n_net = nifty["pnl"].sum()
s_net = sensex["pnl"].sum()
print(f"  {'Net P&L':20s} Rs{n_net:>+10,.0f}   Rs{s_net:>+10,.0f}   Rs{n_net+s_net:>+10,.0f}")

# Peak & drawdown
peak = combined_daily["cum_pnl"].cummax()
dd = peak - combined_daily["cum_pnl"]
mdd = dd.max()
mdd_date = combined_daily.loc[dd.idxmax(), "date"]

print(f"  {'Max Drawdown':20s} {'':>11s} {'':>11s} Rs{mdd:>+10,.0f} (on {mdd_date})")

# Sharpe-like (daily)
daily_returns = combined_daily["total_pnl"]
mean_daily = daily_returns.mean()
std_daily = daily_returns.std()
trading_days = len(combined_daily)
sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
print(f"  {'Daily Sharpe':20s} {'':>11s} {'':>11s} {sharpe:>10.2f}")

# Win rate on trades
n_wr = n_wins / len(nifty) * 100
s_wr = s_wins / len(sensex) * 100
trades_total = len(nifty) + len(sensex)
wins_total = n_wins + s_wins
print(f"  {'Win Rate (trades)':20s} {n_wr:>7.1f}%     {s_wr:>7.1f}%     {wins_total/trades_total*100:>7.1f}%")

# Profit Factor
n_gp = nifty[nifty["pnl"]>0]["pnl"].sum()
n_gl = abs(nifty[nifty["pnl"]<=0]["pnl"].sum())
s_gp = sensex[sensex["pnl"]>0]["pnl"].sum()
s_gl = abs(sensex[sensex["pnl"]<=0]["pnl"].sum())
print(f"  {'Profit Factor':20s} {n_gp/n_gl:>7.2f}     {s_gp/s_gl:>7.2f}     {(n_gp+s_gp)/(n_gl+s_gl):>7.2f}")

# Return on Capital
total_pnl = n_net + s_net
total_capital = CAPITAL_PER_INDEX * 2
print(f"\nRETURN ANALYSIS:")
print(f"  Capital: Rs{CAPITAL_PER_INDEX:,}/index = Rs{total_capital:,} total")
print(f"  Total P&L: Rs{total_pnl:+,.0f}")
print(f"  Return: {total_pnl/total_capital*100:+.1f}%")
print(f"  Max Drawdown: Rs{mdd:+,.0f} ({mdd/total_capital*100:.1f}% of capital)")

# Monthly breakdown
print(f"\nTOP 10 TRADING DAYS (by combined P&L):")
top10 = combined_daily.nlargest(10, "total_pnl")
for _, r in top10.iterrows():
    print(f"  {r['date']}: NIFTY Rs{r['nifty_pnl']:>+8,.0f} + SENSEX Rs{r['sensex_pnl']:>+8,.0f} = Rs{r['total_pnl']:>+8,.0f}")

print(f"\nBOTTOM 10 TRADING DAYS (by combined P&L):")
bot10 = combined_daily.nsmallest(10, "total_pnl")
for _, r in bot10.iterrows():
    print(f"  {r['date']}: NIFTY Rs{r['nifty_pnl']:>+8,.0f} + SENSEX Rs{r['sensex_pnl']:>+8,.0f} = Rs{r['total_pnl']:>+8,.0f}")

# Pattern analysis
print(f"\nPATTERN BREAKDOWN:")
if "pattern" in nifty.columns:
    n_pat = nifty["pattern"].value_counts()
    s_pat = sensex["pattern"].value_counts()
    print(f"  NIFTY50: {n_pat.to_dict()}")
    print(f"  SENSEX:  {s_pat.to_dict()}")
    # Avg points by pattern
    print(f"\n  Avg Points by Pattern:")
    for pat in ["BC", "ENG", "BC+ENG"]:
        if pat in nifty["pattern"].values:
            n_avg = nifty[nifty["pattern"]==pat]["points"].mean()
            print(f"    NIFTY {pat}: {n_avg:+.1f} pts/trade ({len(nifty[nifty['pattern']==pat])} trades)")
        if pat in sensex["pattern"].values:
            s_avg = sensex[sensex["pattern"]==pat]["points"].mean()
            print(f"    SENSEX {pat}: {s_avg:+.1f} pts/trade ({len(sensex[sensex['pattern']==pat])} trades)")

# Exit reason analysis
print(f"\nEXIT REASON BREAKDOWN:")
print(f"  NIFTY50: {nifty['reason'].value_counts().to_dict()}")
print(f"  SENSEX:  {sensex['reason'].value_counts().to_dict()}")

print(f"\n{'='*75}")
print("RECOMMENDED POSITION SIZING (1 lot each, Rs1L capital each):")
print(f"  NIFTY: 1 lot (50 units) | 1 pt = Rs50 | Margin req: ~Rs60,000")
print(f"  SENSEX: 1 lot (10 units) | 1 pt = Rs10 | Margin req: ~Rs60,000")
print(f"  Total P&L: Rs{total_pnl:+,.0f} on Rs{total_capital:,} capital ({total_pnl/total_capital*100:+.1f}%)")
