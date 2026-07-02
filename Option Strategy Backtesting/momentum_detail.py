"""Detailed Momentum Breakout Strategy Analysis"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()

def compute_atr(df, p=14):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(), (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()

def run_momentum():
    all_t = []
    for sym in ["NIFTY50", "SENSEX"]:
        h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
        h1["datetime"] = pd.to_datetime(h1["datetime"])
        h1 = h1.sort_values("datetime").reset_index(drop=True)
        atr = compute_atr(h1, 14)
        hi20 = h1["high"].rolling(20).max().shift(1)
        lot = NLOT if "NIFTY" in sym else SLOT
        intrade = False; ep = 0; hi_en = 0
        for i in range(20, len(h1)):
            if not intrade:
                if (h1["close"].iloc[i] > hi20.iloc[i] and h1["close"].iloc[i] > h1["open"].iloc[i]
                    and h1["datetime"].iloc[i].time() < CUTOFF_TIME and h1["datetime"].iloc[i].hour >= 9):
                    intrade = True; ep = h1["close"].iloc[i]; hi_en = ep
            else:
                if h1["high"].iloc[i] > hi_en: hi_en = h1["high"].iloc[i]
                ca = atr.iloc[i]
                if not pd.isna(ca) and h1["close"].iloc[i] < hi_en - 10 * ca:
                    pts = h1["close"].iloc[i] - ep
                    pnl = pts * lot - CHG
                    all_t.append({
                        "sym": sym, "entry_time": h1["datetime"].iloc[i-1],
                        "exit_time": h1["datetime"].iloc[i], "points": pts,
                        "pnl_rs": pnl,
                        "hold_hours": (h1["datetime"].iloc[i] - h1["datetime"].iloc[i-1]).total_seconds() / 3600
                    })
                    intrade = False
    return pd.DataFrame(all_t).sort_values("exit_time").reset_index(drop=True)

df = run_momentum()
print("=" * 60)
print("MOMENTUM BREAKOUT STRATEGY — FULL DETAILS")
print("=" * 60)
print(f"\nEntry: Close > highest high of last 20 1H candles (breakout)")
print(f"Entry filter: Bullish candle, before 2:15 PM, after 9 AM")
print(f"Exit: Chandelier 10xATR trailing stop")
print(f"Capital: 1 lot each NIFTY (50pts) + SENSEX (10pts), Rs20/trade charge")
print(f"Data: 1-hour candles, ~10 years")
print()
print(f"Total trades:     {len(df)}")
print(f"Net P&L:          Rs{df['pnl_rs'].sum():+,.0f}")
print(f"Win rate:         {(df['pnl_rs']>0).sum()/len(df)*100:.1f}%")
print(f"Profit Factor:    {df[df['pnl_rs']>0]['pnl_rs'].sum()/abs(df[df['pnl_rs']<0]['pnl_rs'].sum()):.2f}")
print(f"Avg Win:          Rs{df[df['pnl_rs']>0]['pnl_rs'].mean():+,.0f}")
print(f"Avg Loss:         Rs{df[df['pnl_rs']<0]['pnl_rs'].mean():+,.0f}")
print(f"Max Win:          Rs{df['pnl_rs'].max():+,.0f}")
print(f"Max Loss:         Rs{df['pnl_rs'].min():+,.0f}")
print(f"Median Win:       Rs{df[df['pnl_rs']>0]['pnl_rs'].median():+,.0f}")
print(f"Median Loss:      Rs{df[df['pnl_rs']<0]['pnl_rs'].median():+,.0f}")
print(f"Avg Hold time:    {df['hold_hours'].mean():.1f}h")
print(f"Median Hold:      {df['hold_hours'].median():.1f}h")
print(f"Max Drawdown:     Rs{(df['pnl_rs'].cumsum().cummax()-df['pnl_rs'].cumsum()).max():,.0f}")
print(f"Sharpe (annual):  {df['pnl_rs'].mean()/df['pnl_rs'].std()*np.sqrt(252*6.5):.2f}")

print(f"\n--- Per Index ---")
for sym in ["NIFTY50", "SENSEX"]:
    sd = df[df["sym"] == sym]
    print(f"  {sym}: {len(sd):3d}t, Rs{sd['pnl_rs'].sum():>+9,.0f}, WR={(sd['pnl_rs']>0).sum()/len(sd)*100:.1f}%")

print(f"\n--- Year-by-Year (combined) ---")
df["year"] = pd.to_datetime(df["exit_time"]).dt.year
for y in sorted(df["year"].unique()):
    yd = df[df["year"] == y]
    print(f"  {y}: {len(yd):3d}t  Rs{yd['pnl_rs'].sum():>+9,.0f}  WR={(yd['pnl_rs']>0).sum()/len(yd)*100:.1f}%  "
          f"AvgWin=Rs{yd[yd['pnl_rs']>0]['pnl_rs'].mean():>+7,.0f}  AvgLoss=Rs{yd[yd['pnl_rs']<0]['pnl_rs'].mean():>+7,.0f}")

# Distribution
wins = df[df["pnl_rs"] > 0]["pnl_rs"]
losses = df[df["pnl_rs"] < 0]["pnl_rs"]
print(f"\n--- Distribution ---")
print(f"  Win/Loss ratio: {wins.mean()/abs(losses.mean()):.2f}")
print(f"  % profitable trades >Rs5000:  {(df['pnl_rs']>5000).sum()/len(df)*100:.1f}%")
print(f"  % profitable trades >Rs10000: {(df['pnl_rs']>10000).sum()/len(df)*100:.1f}%")
print(f"  % profitable trades >Rs25000: {(df['pnl_rs']>25000).sum()/len(df)*100:.1f}%")
print(f"  Best 5 trades: Rs{df.nlargest(5,'pnl_rs')['pnl_rs'].sum():,.0f} ({(df.nlargest(5,'pnl_rs')['pnl_rs'].sum()/df['pnl_rs'].sum())*100:.1f}% of total)")
print(f"  Worst 5 trades: Rs{df.nsmallest(5,'pnl_rs')['pnl_rs'].sum():,.0f}")
