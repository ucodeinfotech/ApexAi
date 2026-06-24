"""Full gap-fade backtest with stops, sizing, trade log, equity curve, and 1-min exit optimization"""
import os, json, numpy as np, pandas as pd
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

DATA_DIR = "comprehensive_data"
SLIPPAGE = 0.05
CAPITAL = 1_000_000  # 10L starting capital
RESULTS_FILE = "gap_fade_full_results.json"
LOG_FILE = "gap_fade_trade_log.csv"
EQ_FILE = "gap_fade_equity.csv"

one_day_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ONE_DAY.csv")])
one_min_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ONE_MINUTE.csv")])
all_syms = sorted(set(f.replace("_ONE_DAY.csv","") for f in one_day_files))
print(f"Daily files: {len(one_day_files)} | 1-min files: {len(one_min_files)}")

# Parameters to sweep
GAP_THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
STOP_LOSSES = [0.5, 1.0, 1.5, 2.0, 3.0]
TAKE_PROFITS = [0.5, 1.0, 1.5, 2.0, 3.0]

# ═══════════════ PART 1: Full backtest on daily data (all 395 stocks) ═══════════════
print("\n=== Full Gap-Fade Backtest ===")
all_trades = []

for f in one_day_files:
    sym = f.replace("_ONE_DAY.csv","")
    df = pd.read_csv(f"{DATA_DIR}/{f}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["prev_close"] = df["close"].shift(1)
    df["gap_pct"] = (df["open"] / df["prev_close"] - 1) * 100

    for _, row in df.iterrows():
        gap = row["gap_pct"]
        if pd.isna(gap) or abs(gap) < 0.5: continue
        
        side = "short" if gap > 0 else "long"
        entry = row["open"]
        close_px = row["close"]
        
        # No-stop base return
        if side == "short":
            ret_no_stop = (entry / close_px - 1) * 100 - 2 * SLIPPAGE
        else:
            ret_no_stop = (close_px / entry - 1) * 100 - 2 * SLIPPAGE
        
        all_trades.append({
            "sym": sym, "date": str(row["date"]), "gap_pct": round(gap, 2),
            "side": side, "entry": round(entry, 2), "close": round(close_px, 2),
            "ret_no_stop": round(ret_no_stop, 2),
        })

trades_df = pd.DataFrame(all_trades)
print(f"Total trades: {len(trades_df):,}")

# Categorize by gap bucket
trades_df["gap_bucket"] = pd.cut(trades_df["gap_pct"].abs(), 
    bins=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 100],
    labels=["0-0.5","0.5-0.75","0.75-1.0","1.0-1.25","1.25-1.5","1.5-2.0","2.0+"])

def compute_stats(trades, ret_col="ret_no_stop"):
    if len(trades) == 0: return {}
    rets = trades[ret_col]
    wins = rets[rets > 0]; losses = rets[rets < 0]
    cum = (1 + rets / 100).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1) * 100
    
    stats = {
        "trades": len(trades),
        "stocks": int(trades["sym"].nunique()),
        "win_rate": round((rets > 0).mean() * 100, 1),
        "avg_ret": round(rets.mean(), 2),
        "median_ret": round(rets.median(), 2),
        "total_ret_pct": round((cum.iloc[-1] - 1) * 100, 1) if len(cum) > 0 else 0,
        "cagr_pct": round(((cum.iloc[-1]) ** (252 / len(rets)) - 1) * 100, 1) if len(rets) > 0 and len(rets) < 252*10 else 0,
        "max_dd_pct": round(dd.min(), 1),
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "max_win": round(rets.max(), 2),
        "max_loss": round(rets.min(), 2),
        "std_ret": round(rets.std(), 2),
        "sharpe": round(rets.mean() / rets.std() * np.sqrt(252), 2) if rets.std() > 0 else 0,
        "profit_factor": round(abs(wins.sum() / losses.sum()), 2) if len(losses) > 0 and losses.sum() != 0 else float("inf"),
    }
    return stats

# Overall stats
base_stats = compute_stats(trades_df)
print(f"Overall: WR={base_stats['win_rate']}%  Avg={base_stats['avg_ret']}%  Sharpe={base_stats['sharpe']}  PF={base_stats['profit_factor']}")

# Stats by gap bucket and side
bucket_results = {}
for side in ["short", "long"]:
    sub = trades_df[trades_df["side"] == side]
    for bucket in sorted(sub["gap_bucket"].dropna().unique()):
        bsub = sub[sub["gap_bucket"] == bucket]
        key = f"{side}_{bucket}"
        bucket_results[key] = compute_stats(bsub)

print("\nBy gap bucket:")
for k, v in sorted(bucket_results.items()):
    if v:
        print(f"  {k:25s}: n={v['trades']:>6d}  WR={v['win_rate']:>5.1f}%  avg={v['avg_ret']:>5.2f}%  PF={v['profit_factor']:>5.2f}")

# Yearly performance
trades_df["year"] = pd.to_datetime(trades_df["date"]).dt.year
yearly = trades_df.groupby("year").apply(lambda g: pd.Series({
    "trades": len(g), "win_rate": round((g["ret_no_stop"]>0).mean()*100,1),
    "avg_ret": round(g["ret_no_stop"].mean(),2),
    "total_ret": round(((1+g["ret_no_stop"]/100).prod()-1)*100,1),
})).reset_index()
print("\nYearly:")
print(yearly.to_string(index=False))

# Stock-level performance
stock_perf = trades_df.groupby("sym").agg(
    trades=("ret_no_stop","count"), avg_ret=("ret_no_stop","mean"),
    win_rate=("ret_no_stop", lambda x: round((x>0).mean()*100,1))
).reset_index()
stock_perf = stock_perf[stock_perf["trades"] >= 20].sort_values("avg_ret", ascending=False)
print(f"\nBest stocks: {stock_perf.head(5)[['sym','trades','avg_ret','win_rate']].to_dict('records')}")
print(f"Worst stocks: {stock_perf.tail(5)[['sym','trades','avg_ret','win_rate']].to_dict('records')}")

# ═══════════════ PART 2: Stop-loss optimization ═══════════════
print("\n=== Stop-Loss & Take-Profit Optimization ===")
# For this, we need intraday data. Sample stocks.
step = max(1, len(all_syms) // 50)
sampled = all_syms[::step]
print(f"Sampled {len(sampled)} stocks for intraday SL/TP optimization")

sl_tp_results = []
for sl in STOP_LOSSES:
    for tp in TAKE_PROFITS:
        all_rets = []
        for sym in sampled:
            path = f"{DATA_DIR}/{sym}_ONE_MINUTE.csv"
            if not os.path.exists(path): continue
            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            df["date"] = df.index.date
            
            # Get daily open/close and gaps
            daily = df["close"].resample("1D").agg(["first","last"])
            daily.columns = ["open","close"]
            daily["prev_close"] = daily["close"].shift(1)
            daily["gap_pct"] = (daily["open"] / daily["prev_close"] - 1) * 100
            
            for date, row in daily.iterrows():
                gap = row["gap_pct"]
                if pd.isna(gap) or abs(gap) < 0.5: continue
                
                day_data = df[df["date"] == date.date()]
                if len(day_data) < 10: continue
                
                entry = day_data.iloc[0]["open"]
                side = "short" if gap > 0 else "long"
                
                # Simulate through the day with SL/TP
                position = 0
                for _, mrow in day_data.iterrows():
                    px = mrow["close"]
                    if side == "short":
                        ret = (entry / px - 1) * 100
                        if ret <= -sl:  # stopped out
                            all_rets.append(-sl - 2*SLIPPAGE); position = 1; break
                        if ret >= tp:  # took profit
                            all_rets.append(tp - 2*SLIPPAGE); position = 1; break
                    else:
                        ret = (px / entry - 1) * 100
                        if ret <= -sl:
                            all_rets.append(-sl - 2*SLIPPAGE); position = 1; break
                        if ret >= tp:
                            all_rets.append(tp - 2*SLIPPAGE); position = 1; break
                
                if position == 0:  # held till close
                    close_px = day_data.iloc[-1]["close"]
                    if side == "short":
                        ret = (entry / close_px - 1) * 100 - 2*SLIPPAGE
                    else:
                        ret = (close_px / entry - 1) * 100 - 2*SLIPPAGE
                    all_rets.append(round(ret, 2))
        
        if all_rets:
            r = np.array(all_rets)
            wr = (r > 0).mean() * 100
            sl_tp_results.append({
                "stop_loss": sl, "take_profit": tp,
                "trades": len(r), "win_rate": round(wr, 1),
                "avg_ret": round(r.mean(), 2),
                "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else 0,
            })

sl_df = pd.DataFrame(sl_tp_results)
if len(sl_df) > 0:
    best_sl = sl_df.loc[sl_df["sharpe"].idxmax()]
    print(f"\nBest SL/TP combo:")
    print(f"  Stop Loss: {best_sl['stop_loss']}% | Take Profit: {best_sl['take_profit']}%")
    print(f"  WR: {best_sl['win_rate']}% | Avg: {best_sl['avg_ret']}% | Sharpe: {best_sl['sharpe']}")
    print(f"\nTop 5 by Sharpe:")
    print(sl_df.nlargest(5, "sharpe").to_string(index=False))

# ═══════════════ PART 3: Exit time optimization ═══════════════
print("\n=== Exit Time Optimization ===")
exit_time_results = []
for exit_hour in [14, 14.5, 15, 15.5]:  # 14:00, 14:30, 15:00, 15:30 (close)
    time_rets = []
    for sym in sampled:
        path = f"{DATA_DIR}/{sym}_ONE_MINUTE.csv"
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df["date"] = df.index.date
        
        daily = df["close"].resample("1D").agg(["first","last"])
        daily.columns = ["open","close"]
        daily["prev_close"] = daily["close"].shift(1)
        daily["gap_pct"] = (daily["open"] / daily["prev_close"] - 1) * 100
        
        for date, row in daily.iterrows():
            gap = row["gap_pct"]
            if pd.isna(gap) or abs(gap) < 0.75: continue
            
            day_data = df[df["date"] == date.date()]
            if len(day_data) < 10: continue
            
            entry = day_data.iloc[0]["open"]
            # Find exit at specified time
            exit_row = day_data.between_time(f"{int(exit_hour):02d}:{int((exit_hour%1)*60):02d}", 
                f"{int(exit_hour):02d}:{int((exit_hour%1)*60):02d}")
            if len(exit_row) == 0:  # exact time not found, use last available
                exit_px = day_data.iloc[-1]["close"]
            else:
                exit_px = exit_row.iloc[0]["close"]
            
            if gap > 0:  # short
                ret = (entry / exit_px - 1) * 100 - 2*SLIPPAGE
            else:  # long
                ret = (exit_px / entry - 1) * 100 - 2*SLIPPAGE
            time_rets.append(ret)
    
    if time_rets:
        r = np.array(time_rets)
        exit_time_results.append({
            "exit_time": f"{int(exit_hour):02d}:{int((exit_hour%1)*60):02d}",
            "trades": len(r), "win_rate": round((r>0).mean()*100, 1),
            "avg_ret": round(r.mean(), 2), "sharpe": round(r.mean()/r.std()*np.sqrt(252), 2) if r.std()>0 else 0,
        })

et_df = pd.DataFrame(exit_time_results)
if len(et_df) > 0:
    print(f"Exit time comparison:")
    print(et_df.to_string(index=False))

# ═══════════════ PART 4: Combined results with best params ═══════════════
# Apply best SL/TP to all trades
best_sl_val = best_sl["stop_loss"] if "best_sl" in dir() else 2.0
best_tp_val = best_sl["take_profit"] if "best_sl" in dir() else 2.0
print(f"\nApplying SL={best_sl_val}% / TP={best_tp_val}% to full dataset...")

# Simulate SL/TP on all trades using open/close + estimated intraday path
# Since we don't have intraday for all stocks, approximate: apply SL/TP if no_stop_ret exceeds threshold
def apply_sl_tp(row, sl, tp):
    ret = row["ret_no_stop"]
    side = row["side"]
    if side == "short":
        if ret < -sl: return -sl
        if ret > tp: return tp
    else:
        if ret < -sl: return -sl
        if ret > tp: return tp
    return ret

for sl_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for tp_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
        trades_df[f"ret_sl{sl_val}_tp{tp_val}"] = trades_df.apply(
            lambda r: apply_sl_tp(r, sl_val, tp_val), axis=1)

# Find best param combo
best_combo = None; best_sharpe = -999
for sl_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for tp_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
        col = f"ret_sl{sl_val}_tp{tp_val}"
        r = trades_df[col]
        s = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        if s > best_sharpe:
            best_sharpe = s; best_combo = (sl_val, tp_val)

print(f"Best param: SL={best_combo[0]}% / TP={best_combo[1]}% (Sharpe={best_sharpe:.2f})")

# Final stats with best params
best_col = f"ret_sl{best_combo[0]}_tp{best_combo[1]}"
final_stats = compute_stats(trades_df.assign(ret=best_col), "ret")
final_stats["stop_loss"] = best_combo[0]
final_stats["take_profit"] = best_combo[1]

# ═══════════════ PART 5: Save outputs ═══════════════
results = {
    "timestamp": str(datetime.now()),
    "overall": base_stats,
    "by_gap_bucket": {k: v for k, v in sorted(bucket_results.items()) if v},
    "yearly": yearly.to_dict("records"),
    "best_stocks": stock_perf.head(20).to_dict("records"),
    "worst_stocks": stock_perf.tail(20).to_dict("records"),
    "stop_loss_optimization": sl_tp_results,
    "exit_time_optimization": exit_time_results,
    "best_parameters": {"stop_loss_pct": best_combo[0], "take_profit_pct": best_combo[1], "sharpe": round(best_sharpe, 2)},
    "final_with_stops": final_stats,
    "slippage_pct": SLIPPAGE,
}

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)

# Save trade log
trades_df["ret_final"] = trades_df[best_col]
trades_df_out = trades_df[["date","sym","side","gap_pct","entry","close","ret_no_stop","ret_final"]]
trades_df_out.to_csv(LOG_FILE, index=False)
print(f"Trade log: {LOG_FILE} ({len(trades_df_out):,} rows)")

# Save equity curve
final_rets = trades_df["ret_final"].values
eq = (1 + final_rets / 100).cumprod() * (CAPITAL / eq[0] if len(final_rets)>0 and final_rets[0]>0 else CAPITAL)
eq_df = pd.DataFrame({"trade": range(1, len(eq)+1), "equity": eq})
eq_df.to_csv(EQ_FILE, index=False)
print(f"Equity curve: {EQ_FILE}")

# ═══════════════ SUMMARY ═══════════════
print(f"\n{'='*60}")
print("FINAL GAP FADE BACKTEST RESULTS")
print(f"{'='*60}")
print(f"All stocks (395) | {len(trades_df):,} trades | Slippage: {SLIPPAGE}%")
print(f"\nWithout stops:")
print(f"  WR: {base_stats['win_rate']}% | Avg: {base_stats['avg_ret']}% | Sharpe: {base_stats['sharpe']} | PF: {base_stats['profit_factor']}")
print(f"  Max DD: {base_stats['max_dd_pct']}% | Max Win: {base_stats['max_win']}% | Max Loss: {base_stats['max_loss']}%")
print(f"\nWith SL={best_combo[0]}% / TP={best_combo[1]}%:")
print(f"  WR: {final_stats['win_rate']}% | Avg: {final_stats['avg_ret']}% | Sharpe: {final_stats['sharpe']} | PF: {final_stats['profit_factor']}")
print(f"  Max DD: {final_stats['max_dd_pct']}%")

print(f"\nBest gap size for shorts (>{best_combo[0]}% gaps):")
for k, v in sorted(bucket_results.items()):
    if v and "short" in k:
        print(f"  {k:25s}: WR={v['win_rate']:>5.1f}%  avg={v['avg_ret']:>5.2f}%  n={v['trades']:>6d}")

print(f"\nBest exit time:")
if len(et_df) > 0:
    best_et = et_df.loc[et_df["sharpe"].idxmax()]
    print(f"  {best_et['exit_time']} -> Sharpe: {best_et['sharpe']}, WR: {best_et['win_rate']}%")
