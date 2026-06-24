"""Test backtester on 3 stocks"""
from backtest_pivot_strategy import *

data_dir = "nifty50_full_history"
all_trades = []

for sym in ["SBIN", "RELIANCE", "TCS"]:
    result = backtest_stock(sym, None, data_dir, "backtest_results")
    if result and not (isinstance(result, dict) and "error" in result):
        all_trades.extend(result)
        print(f"{sym}: {len(result)} trades")

if all_trades:
    m = compute_metrics(all_trades)
    print(f"\nTest Results ({len(all_trades)} trades):")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print("\nSample trades:")
    for t in all_trades[:5]:
        print(f"  {t['symbol']:10s} {t['type']:5s} Entry={t['entry_price']:>8.2f} Exit={t['exit_price']:>8.2f} NetPnL={t['net_pnl']:>8.2f} R={t['r_multiple']:>5.2f} ({t['exit_reason']})")
else:
    print("No trades")
