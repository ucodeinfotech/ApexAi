import pandas as pd
df = pd.read_csv('backtest_results/bb_strategy_sweep.csv')
pos = df[(df['net_pnl'] > 0) & (df['trades'] >= 30)].sort_values('net_pnl', ascending=False)
cols = ['strategy','symbol','n_std','rr','slide','trades','win_rate','net_pnl','profit_factor','sharpe']
print(pos[cols].to_string())
print(f'\nProfitable strategies: {len(pos)}')
if len(pos) > 0:
    r = pos.iloc[0]
    print(f'Best: {r["strategy"]} {r["symbol"]} SD={r["n_std"]} RR={r["rr"]} PnL={r["net_pnl"]} WR={r["win_rate"]}% Trades={r["trades"]}')
