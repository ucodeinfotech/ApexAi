import sys, pandas as pd
sys.path.insert(0, r'C:\Users\pc\Downloads\stock hist data')
from backtest_orb import backtest_stock
trades, err = backtest_stock('RELIANCE', 'nifty50_full_history')
if trades:
    df = pd.DataFrame(trades)
    wins = df[df['net_pnl'] > 0]
    print(f'RELIANCE: {len(trades)} trades')
    print(f'Win rate: {len(wins)}/{len(df)} = {len(wins)/len(df)*100:.1f}%')
    print(f'Net P&L: Rs{df["net_pnl"].sum():.2f}')
    print(f'Avg R: {df["r"].mean():.2f}')
    print(f'Charges: Rs{df["charges"].sum():.2f}')
    print(f'Partial: {df["partial"].sum()}/{len(df)}')
    print(f'Reasons: {df["reason"].value_counts().to_dict()}')
    print(df.head(10).to_string())
else:
    print(f'Error: {err}')
