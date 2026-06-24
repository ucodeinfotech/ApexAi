import pandas as pd
import numpy as np
import os, json

DATA_DIR = "nifty50_full_history"
INDICES = {"NIFTY50": "NIFTY50", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}
TIMEFRAMES = {"15min": "FIFTEEN_MINUTE", "30min": "THIRTY_MINUTE", "daily": "ONE_DAY"}

def load_tf(sym, tf_name):
    df = pd.read_csv(f"{DATA_DIR}/{sym}_{tf_name}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def bb_analysis(df, name, tf_label):
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    # Test BB mean reversion at various parameter combinations
    results = []
    for period in [15, 20, 30, 50]:
        for n_std in [1.5, 2.0, 2.5, 3.0]:
            ma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            upper = ma + n_std * std
            lower = ma - n_std * std

            # Long signals: close < lower band
            long_sig = df["close"] < lower
            # Short signals: close > upper band
            short_sig = df["close"] > upper

            # Next bar return after signal
            next_ret = df["ret"].shift(-1)
            long_ret = next_ret[long_sig].dropna()
            short_ret = -next_ret[short_sig].dropna()  # short: profit when price falls
            all_ret = pd.concat([long_ret, short_ret])

            if len(all_ret) < 10:
                continue

            win_rate = (all_ret > 0).sum() / len(all_ret) * 100
            avg_ret = all_ret.mean() * 100
            sharpe = all_ret.mean() / all_ret.std() * np.sqrt(252*26) if all_ret.std() > 0 else 0
            total_trades = len(all_ret)
            long_count = len(long_ret)
            short_count = len(short_ret)

            results.append({
                "index": name,
                "timeframe": tf_label,
                "period": period,
                "n_std": n_std,
                "total_trades": total_trades,
                "long_trades": long_count,
                "short_trades": short_count,
                "win_rate_pct": round(win_rate, 2),
                "avg_next_ret_pct": round(avg_ret, 4),
                "sharpe": round(sharpe, 3),
            })

    return results

def volatility_regime(df, name, tf_label):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["vol_20"] = df["ret"].rolling(20).std() * 100

    vol = df["vol_20"].dropna()
    thresholds = {
        "low_vol": vol.quantile(0.33),
        "med_vol": vol.quantile(0.67),
    }

    regimes = []
    for label, thresh in [("low", vol.quantile(0.33)), ("high", vol.quantile(0.67))]:
        mask = df["vol_20"] > thresh if label == "high" else df["vol_20"] <= thresh
        regime_ret = df.loc[mask, "ret"].shift(-1).dropna()
        if len(regime_ret) > 0:
            regimes.append({
                "index": name,
                "timeframe": tf_label,
                "regime": f"{label}_vol",
                "thresh": round(thresh, 4),
                "n_bars": len(regime_ret),
                "win_rate": round((regime_ret > 0).sum() / len(regime_ret) * 100, 2),
                "avg_ret_pct": round(regime_ret.mean() * 100, 4),
                "std_pct": round(regime_ret.std() * 100, 4),
            })
    return regimes

def main():
    all_bb = []
    all_regimes = []

    for idx_name, sym in INDICES.items():
        for tf_label, tf_file in TIMEFRAMES.items():
            print(f"\n=== {sym} {tf_label} ===")
            df = load_tf(sym, tf_file)
            print(f"  Rows: {len(df)}, from {df['datetime'].min()} to {df['datetime'].max()}")

            # BB param sweep
            bb_res = bb_analysis(df, sym, tf_label)
            all_bb.extend(bb_res)

            # Vol regime
            all_regimes.extend(volatility_regime(df, sym, tf_label))

    # Find best performing params overall
    print("\n\n=== TOP 15 BB PARAM COMBOS (by Sharpe) ===")
    best = sorted(all_bb, key=lambda x: x["sharpe"], reverse=True)[:15]
    for r in best:
        print(f"  {r['index']} {r['timeframe']} | period={r['period']} std={r['n_std']} | "
              f"trades={r['total_trades']} WR={r['win_rate_pct']}% avg_ret={r['avg_next_ret_pct']}% sharpe={r['sharpe']}")

    print("\n\n=== WORST 10 BB PARAM COMBOS (by Sharpe) ===")
    worst = sorted(all_bb, key=lambda x: x["sharpe"])[:10]
    for r in worst:
        print(f"  {r['index']} {r['timeframe']} | period={r['period']} std={r['n_std']} | "
              f"trades={r['total_trades']} WR={r['win_rate_pct']}% avg_ret={r['avg_next_ret_pct']}% sharpe={r['sharpe']}")

    print("\n\n=== VOLATILITY REGIME ANALYSIS ===")
    for r in all_regimes:
        print(f"  {r['index']} {r['timeframe']} {r['regime']}: thresh={r['thresh']} "
              f"n={r['n_bars']} WR={r['win_rate']}% avg={r['avg_ret_pct']}% std={r['std_pct']}%")

    # Save
    output = {
        "bb_sweep": all_bb,
        "volatility_regimes": all_regimes,
    }
    with open("bb_param_sweep.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n Saved to bb_param_sweep.json")

if __name__ == "__main__":
    main()
