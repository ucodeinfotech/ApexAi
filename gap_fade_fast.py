"""Fast gap-fade backtest: daily data only, all 395 stocks"""
import os, json, numpy as np, pandas as pd
from datetime import datetime

DATA_DIR = "comprehensive_data"
SLIPPAGE = 0.05
RESULTS_FILE = "gap_fade_results.json"
LOG_FILE = "gap_fade_trade_log.csv"
EQ_FILE = "gap_fade_equity.csv"

one_day_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ONE_DAY.csv")])
print(f"Files: {len(one_day_files)}")

all_trades = []
for f in one_day_files:
    sym = f.replace("_ONE_DAY.csv","")
    df = pd.read_csv(f"{DATA_DIR}/{f}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    df["gap_pct"] = (df["open"] / df["prev_close"] - 1) * 100
    for _, row in df.iterrows():
        gap = row["gap_pct"]
        if pd.isna(gap) or abs(gap) < 0.5: continue
        side = "short" if gap > 0 else "long"
        entry, close_px = row["open"], row["close"]
        ret = (entry/close_px - 1)*100 - 2*SLIPPAGE if side=="short" else (close_px/entry - 1)*100 - 2*SLIPPAGE
        all_trades.append({
            "date": str(row["datetime"].date()), "sym": sym, "side": side,
            "gap_pct": round(gap,2), "entry": round(entry,2), "close": round(close_px,2),
            "ret": round(ret,2)
        })

trades = pd.DataFrame(all_trades)
trades["year"] = pd.to_datetime(trades["date"]).dt.year
trades["gap_abs"] = trades["gap_pct"].abs()
trades["bucket"] = pd.cut(trades["gap_abs"],
    bins=[0,0.5,0.75,1.0,1.25,1.5,2.0,100],
    labels=["0-0.5","0.5-0.75","0.75-1.0","1.0-1.25","1.25-1.5","1.5-2.0","2.0+"])
print(f"Total trades: {len(trades):,}")

def stats(df, label=""):
    r = df["ret"]; w = r[r>0]; l = r[r<0]
    c = (1+r/100).cumprod()
    dd = (c/c.expanding().max()-1)*100
    s = {"label": label, "trades": len(df), "stocks": int(df["sym"].nunique()),
         "win_rate": round((r>0).mean()*100,1), "avg_ret": round(r.mean(),2),
         "median_ret": round(r.median(),2), "std_ret": round(r.std(),2),
         "total_ret": round((c.iloc[-1]-1)*100,1), "max_dd": round(dd.min(),1),
         "sharpe": round(r.mean()/r.std()*np.sqrt(252),2) if r.std()>0 else 0,
         "avg_win": round(w.mean(),2) if len(w)>0 else 0,
         "avg_loss": round(l.mean(),2) if len(l)>0 else 0,
         "profit_factor": round(abs(w.sum()/l.sum()),2) if len(l)>0 and l.sum()!=0 else "inf"}
    return s

# Overall
overall = stats(trades, "All")

# By side + bucket
by_bucket = {}
for side in ["short","long"]:
    for bucket in trades[trades["side"]==side]["bucket"].dropna().unique():
        sub = trades[(trades["side"]==side) & (trades["bucket"]==bucket)]
        by_bucket[f"{side}_{bucket}"] = stats(sub, f"{side}_{bucket}")

# By year
yearly = trades.groupby("year").apply(lambda g: stats(g, str(g.name))).tolist()

# By stock
stock = trades.groupby("sym").agg(trades=("ret","count"), avg_ret=("ret","mean"),
    win_rate=("ret", lambda x: round((x>0).mean()*100,1)))
stock = stock[stock["trades"]>=20].sort_values("avg_ret", ascending=False)
best_w10 = stock.head(10).reset_index().to_dict("records")
worst_w10 = stock.tail(10).reset_index().to_dict("records")

# Apply SL/TP approximation (cap trades)
for sl in [0.5,1.0,1.5,2.0,3.0]:
    for tp in [0.5,1.0,1.5,2.0,3.0]:
        trades[f"sl{sl}_tp{tp}"] = trades["ret"].clip(-sl, tp)

# Find best SL/TP combo
best_sharpe, best_sl, best_tp = -999, 0, 0
for sl in [0.5,1.0,1.5,2.0,3.0]:
    for tp in [0.5,1.0,1.5,2.0,3.0]:
        r = trades[f"sl{sl}_tp{tp}"]
        s = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
        if s > best_sharpe: best_sharpe, best_sl, best_tp = s, sl, tp

final_col = f"sl{best_sl}_tp{best_tp}"
final = stats(trades.assign(ret=trades[final_col]), f"SL={best_sl}% TP={best_tp}%")

# Save
results = {
    "timestamp": str(datetime.now()),
    "slippage_pct": SLIPPAGE, "stocks": len(one_day_files),
    "overall": overall, "by_bucket": by_bucket,
    "yearly": yearly, "best_stocks": best_w10, "worst_stocks": worst_w10,
    "best_params": {"stop_loss": best_sl, "take_profit": best_tp, "sharpe": round(best_sharpe,2)},
    "final_with_stops": final,
}
with open(RESULTS_FILE, "w") as f: json.dump(results, f, indent=2, default=str)

# Save trade log
trades["ret_final"] = trades[final_col]
trades[["date","sym","side","gap_pct","entry","close","ret","ret_final"]].to_csv(LOG_FILE, index=False)
# Save equity curve
rets = trades["ret_final"].values
eq = (1+rets/100).cumprod()
eq_df = pd.DataFrame({"trade": range(1,len(eq)+1), "equity": eq})
eq_df.to_csv(EQ_FILE, index=False)

print(f"\nSaved: {RESULTS_FILE}, {LOG_FILE}, {EQ_FILE}")

# Summary
print(f"\n{'='*60}")
print("GAP FADE BACKTEST — ALL 395 STOCKS")
print(f"{'='*60}")
print(f"Total: {len(trades):,} trades | Slippage: {SLIPPAGE}%")
print(f"\nNo stops:")
print(f"  WR: {overall['win_rate']}% | Avg: {overall['avg_ret']}% | Sharpe: {overall['sharpe']} | PF: {overall['profit_factor']}")
print(f"  MaxDD: {overall['max_dd']}% | MaxRet: {overall['avg_win']}% | MaxLoss: {overall['avg_loss']}%")
print(f"\nWith SL={best_sl}% / TP={best_tp}%:")
print(f"  WR: {final['win_rate']}% | Avg: {final['avg_ret']}% | Sharpe: {final['sharpe']} | PF: {final['profit_factor']}")
print(f"  MaxDD: {final['max_dd']}% | TotalRet: {final['total_ret']}%")
print(f"\nBy gap bucket (short):")
for k,v in sorted(by_bucket.items()):
    if "short" in k: print(f"  {k:25s}: n={v['trades']:>6d}  WR={v['win_rate']:>5.1f}%  avg={v['avg_ret']:>5.2f}%  PF={v['profit_factor']:>5.2f}")
print(f"\nBy gap bucket (long):")
for k,v in sorted(by_bucket.items()):
    if "long" in k: print(f"  {k:25s}: n={v['trades']:>6d}  WR={v['win_rate']:>5.1f}%  avg={v['avg_ret']:>5.2f}%  PF={v['profit_factor']:>5.2f}")
