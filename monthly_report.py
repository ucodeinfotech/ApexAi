"""Vol Breakout - monthly breakdown"""
import duckdb, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

DB_PATH = "warehouse/market_data.duckdb"
UNIVERSE = {
    "BAJAJFINSV": (20,2.0),"IRFC": (30,1.5),"HSCL": (20,2.0),"JINDALSTEL": (20,2.0),
    "KPITTECH": (20,2.0),"VOLTAS": (20,2.0),"BLUESTARCO": (15,1.5),"M&M": (20,2.0),
    "ANGELONE": (10,1.5),"PCJEWELLER": (20,2.0),"LUXIND": (30,1.5),"BEL": (20,2.0),
    "HAL": (10,1.5),"ALKEM": (20,2.0),"SUVEN": (15,1.5),"SHREECEM": (20,2.0),
    "ADANIGREEN": (20,2.0),"ASHOKA": (20,2.0),"APLAPOLLO": (15,1.5),"TATACONSUM": (20,2.0),
}

con = duckdb.connect(DB_PATH)
trades = []
for sym, (lb, mult) in UNIVERSE.items():
    df = con.execute(f"SELECT datetime, close, volume FROM raw_market WHERE symbol='{sym.replace(chr(39),chr(39)+chr(39))}' AND timeframe='1day' ORDER BY datetime").fetchdf()
    if len(df) < lb + 1: continue
    df["avg_vol"] = df["volume"].rolling(lb).mean().shift(1)
    df["ret"] = df["close"].pct_change(1)
    df["signal"] = (df["ret"] > 0.01) & (df["volume"] > df["avg_vol"] * mult)
    for idx in df.index[df["signal"]].tolist():
        ex = min(idx + 5, len(df) - 1)
        if ex <= idx: continue
        trades.append({
            "symbol": sym, "entry_date": str(df["datetime"].iloc[idx])[:10],
            "exit_date": str(df["datetime"].iloc[ex])[:10],
            "ret_pct": float((df["close"].iloc[ex] / df["close"].iloc[idx] - 1) * 100),
            "entry_px": float(df["close"].iloc[idx]),
            "exit_px": float(df["close"].iloc[ex]),
        })
tdf = pd.DataFrame(trades)
tdf["entry_dt"] = pd.to_datetime(tdf["entry_date"])
tdf["year"] = tdf["entry_dt"].dt.year
tdf["month"] = tdf["entry_dt"].dt.month
tdf["ym"] = tdf["entry_dt"].dt.strftime("%Y-%m")
tdf = tdf.sort_values("entry_dt").reset_index(drop=True)
tdf["trade_no"] = range(1, len(tdf) + 1)

total_trades = len(tdf)
win_rate = (tdf["ret_pct"] > 0).mean()
avg_ret = tdf["ret_pct"].mean()
sharpe = avg_ret / tdf["ret_pct"].std() * np.sqrt(252/5) if tdf["ret_pct"].std() > 0 else 0
ann_ret = avg_ret / 100 * 252 / 5

print("=" * 70)
print("VOL BREAKOUT - CORRECTED BACKTEST SUMMARY")
print("=" * 70)
print(f"{'Total Trades':30s} {total_trades:>8,}")
print(f"{'Win Rate':30s} {win_rate:>7.1%}")
print(f"{'Avg Return (5-day)':30s} +{avg_ret:>5.2f}%")
print(f"{'Median Return':30s} +{tdf['ret_pct'].median():>5.2f}%")
print(f"{'Sharpe Ratio':30s} {sharpe:>8.2f}")
print(f"{'Annualized Return':30s} {ann_ret:>7.1%}")
print(f"{'Best Trade':30s} +{tdf['ret_pct'].max():>5.1f}%")
print(f"{'Worst Trade':30s} {tdf['ret_pct'].min():>5.1f}%")
print(f"{'Avg Win':30s} +{tdf[tdf['ret_pct']>0]['ret_pct'].mean():.2f}%")
print(f"{'Avg Loss':30s} {tdf[tdf['ret_pct']<0]['ret_pct'].mean():.2f}%")
print(f"{'Avg Trades/Month':30s} {total_trades/len(tdf['ym'].unique()):>8.1f}")

print("\n" + "=" * 70)
print("MONTHLY BREAKDOWN")
print("=" * 70)
print(f"{'Year-Month':>8s} {'Trades':>7s} {'Avg Ret':>9s} {'WR':>7s} {'Cumul':>9s} {'Sharpe':>8s}")
print("-" * 70)

monthly = tdf.groupby("ym").agg(trades=("ret_pct","count"), avg_ret=("ret_pct","mean"), wr=("ret_pct",lambda x:(x>0).mean())).reset_index().sort_values("ym")
monthly["cumul"] = (1 + monthly["avg_ret"] / 100).cumprod()
monthly["sharpe_m"] = monthly.apply(lambda r: r["avg_ret"] / tdf[tdf["ym"]==r["ym"]]["ret_pct"].std() * np.sqrt(252/5) if len(tdf[tdf["ym"]==r["ym"]]) >= 2 and tdf[tdf["ym"]==r["ym"]]["ret_pct"].std() > 0 else 0, axis=1)

for _, r in monthly.iterrows():
    print(f"{r['ym']:>8s} {int(r['trades']):>7d} +{r['avg_ret']:>6.2f}% {r['wr']:>5.1%} {r['cumul']:>8.2f}x {r['sharpe_m']:>7.2f}")

print("-" * 70)
tot_cumul = (1 + tdf["ret_pct"] / 100).prod()
print(f"{'TOTAL / AVG':>8s} {total_trades:>7d} +{avg_ret:>6.2f}% {win_rate:>5.1%} {tot_cumul:>8.2f}x {sharpe:>7.2f}")

# Yearly summary
print("\n" + "=" * 70)
print("YEARLY BREAKDOWN")
print("=" * 70)
print(f"{'Year':>5s} {'Trades':>7s} {'Avg Ret':>9s} {'WR':>7s} {'Cumul':>9s} {'Sharpe':>8s}")
print("-" * 70)
yearly = tdf.groupby("year").agg(trades=("ret_pct","count"), avg_ret=("ret_pct","mean"), wr=("ret_pct",lambda x:(x>0).mean())).reset_index()
for _, r in yearly.iterrows():
    yr_cumul = (1 + tdf[tdf["year"]==r["year"]]["ret_pct"]/100).prod()
    yr_sharpe = r["avg_ret"]/tdf[tdf["year"]==r["year"]]["ret_pct"].std()*np.sqrt(252/5) if len(tdf[tdf["year"]==r["year"]])>=2 and tdf[tdf["year"]==r["year"]]["ret_pct"].std()>0 else 0
    print(f"{int(r['year']):>5d} {int(r['trades']):>7d} +{r['avg_ret']:>6.2f}% {r['wr']:>5.1%} {yr_cumul:>8.2f}x {yr_sharpe:>7.2f}")

# Best/worst months
print("\n" + "=" * 70)
print("BEST 5 MONTHS")
print("=" * 70)
best5 = monthly.nlargest(5, "avg_ret")
for _, r in best5.iterrows():
    print(f"  {r['ym']}: {int(r['trades']):3d} trades, +{r['avg_ret']:.2f}% avg, {r['wr']:.0%} WR")

print("\nWORST 5 MONTHS")
print("=" * 70)
worst5 = monthly.nsmallest(5, "avg_ret")
for _, r in worst5.iterrows():
    print(f"  {r['ym']}: {int(r['trades']):3d} trades, {r['avg_ret']:.2f}% avg, {r['wr']:.0%} WR")

# Regime analysis
print("\n" + "=" * 70)
print("REGIME ANALYSIS (Sensex MA50)")
print("=" * 70)
con2 = duckdb.connect(DB_PATH)
sensex = con2.execute("SELECT datetime, close FROM raw_market WHERE symbol='SENSEX' AND timeframe='1day' ORDER BY datetime").fetchdf()
con2.close()
sensex["ma50"] = sensex["close"].rolling(50).mean().shift(1)
sensex["regime"] = np.where(sensex["close"] > sensex["ma50"], "BULL", "BEAR")
sensex["date"] = sensex["datetime"].dt.date

tdf2 = tdf.copy()
tdf2["entry_date_obj"] = tdf2["entry_dt"].dt.date
tdf2 = tdf2.merge(sensex[["date","regime"]], left_on="entry_date_obj", right_on="date", how="left")
tdf2["regime"] = tdf2["regime"].fillna("UNKNOWN")

for regime in ["BULL","BEAR"]:
    sub = tdf2[tdf2["regime"]==regime]
    if len(sub) < 3: continue
    s = sub["ret_pct"].mean()/sub["ret_pct"].std()*np.sqrt(252/5)
    print(f"  {regime}: {len(sub):4d} trades  avg={sub['ret_pct'].mean():+.2f}%  wr={(sub['ret_pct']>0).mean():.0%}  sharpe={s:.2f}")

con.close()
