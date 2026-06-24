import pandas as pd
import numpy as np
import json

DATA_DIR = "nifty50_full_history"
INDICES = {"NIFTY50": "NIFTY50", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}
TF = "FIFTEEN_MINUTE"

def load(sym):
    df = pd.read_csv(f"{DATA_DIR}/{sym}_{TF}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    return df

def bb_forward_test(df, name, period=20, n_std=2.0, forward_bars=12):
    """Check next N bars after BB touch"""
    df = df.copy()
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std

    results = []
    for bars_ahead in [1, 2, 4, 6, 8, 12]:
        long_sig = df["close"] < lower
        short_sig = df["close"] > upper

        fwd_ret = df["close"].pct_change(bars_ahead).shift(-bars_ahead)
        long_ret = fwd_ret[long_sig].dropna()
        short_ret = -fwd_ret[short_sig].dropna()
        all_ret = pd.concat([long_ret, short_ret])

        if len(all_ret) < 5:
            continue
        results.append({
            "index": name,
            "period": period,
            "n_std": n_std,
            "forward_bars": bars_ahead,
            "total_trades": len(all_ret),
            "win_rate": round((all_ret > 0).sum() / len(all_ret) * 100, 2),
            "avg_ret_pct": round(all_ret.mean() * 100, 4),
            "sharpe": round(all_ret.mean() / all_ret.std() * np.sqrt(252*26 / bars_ahead) if all_ret.std() > 0 else 0, 3),
        })
    return results

def pattern_analysis(df, name):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    insights = {}

    # Time-of-day directional bias
    df["hour_min"] = df["hour"] * 100 + df["minute"]
    time_buckets = {
        "9:15-9:30": (915, 930), "9:30-10:00": (930, 1000),
        "10:00-11:00": (1000, 1100), "11:00-12:00": (1100, 1200),
        "12:00-13:00": (1200, 1300), "13:00-14:00": (1300, 1400),
        "14:00-15:00": (1400, 1500), "15:00-15:15": (1500, 1515),
    }
    time_stats = {}
    for label, (s, e) in time_buckets.items():
        mask = (df["hour_min"] >= s) & (df["hour_min"] < e)
        chunk = df.loc[mask, "ret"].dropna()
        if len(chunk) > 0:
            time_stats[label] = {
                "n": len(chunk),
                "win_rate": round((chunk > 0).sum() / len(chunk) * 100, 1),
                "avg_ret_pct": round(chunk.mean() * 100, 4),
                "vol_pct": round(chunk.std() * 100, 4),
            }
    insights["time_of_day"] = time_stats

    # Gap up/down next bar behavior
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    gap_buckets = {
        "big_gap_up": df["gap"] > 0.3,
        "small_gap_up": (df["gap"] > 0.1) & (df["gap"] <= 0.3),
        "no_gap": abs(df["gap"]) <= 0.1,
        "small_gap_down": (df["gap"] < -0.1) & (df["gap"] >= -0.3),
        "big_gap_down": df["gap"] < -0.3,
    }
    gap_stats = {}
    for label, mask in gap_buckets.items():
        nxt = df["ret"].shift(-1)[mask].dropna()
        if len(nxt) > 5:
            gap_stats[label] = {
                "n": len(nxt),
                "win_rate": round((nxt > 0).sum() / len(nxt) * 100, 1),
                "avg_ret_pct": round(nxt.mean() * 100, 4),
                "vol_pct": round(nxt.std() * 100, 4),
            }
    insights["gap_stats"] = gap_stats

    # Day of week full day bias
    dow_stats = {}
    names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    for d in range(5):
        mask = df["dow"] == d
        day_ret = df.loc[mask, "ret"]
        if len(day_ret) > 0:
            dow_stats[names[d]] = {
                "n_15min_bars": len(day_ret),
                "win_rate": round((day_ret > 0).sum() / len(day_ret) * 100, 1),
                "avg_ret_pct": round(day_ret.mean() * 100, 4),
                "vol_pct": round(day_ret.std() * 100, 4),
            }
    insights["day_of_week"] = dow_stats

    # Monthly patterns
    month_stats = {}
    names_m = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    for m in range(1, 13):
        mask = df["month"] == m
        mret = df.loc[mask, "ret"]
        if len(mret) > 0:
            month_stats[names_m[m]] = {
                "n_15min_bars": len(mret),
                "win_rate": round((mret > 0).sum() / len(mret) * 100, 1),
                "avg_ret_pct": round(mret.mean() * 100, 4),
            }
    insights["monthly"] = month_stats

    return insights

def main():
    report = {}

    for idx_name, sym in INDICES.items():
        print(f"\n{'='*60}")
        print(f"  {sym}")
        print(f"{'='*60}")
        df = load(sym)
        df["ret"] = df["close"].pct_change()

        # Pattern analysis
        pat = pattern_analysis(df, sym)
        report[sym] = pat

        # BB forward test
        for period in [20, 30]:
            for n_std in [2.0, 2.5]:
                fwd = bb_forward_test(df, sym, period=period, n_std=n_std)
                key = f"bb_fwd_p{period}_s{n_std}"
                report[sym][key] = fwd

        # Print key patterns
        print("\n--- Time of Day ---")
        for t, v in sorted(pat["time_of_day"].items()):
            print(f"  {t}: WR={v['win_rate']}% avg={v['avg_ret_pct']}% vol={v['vol_pct']}%")

        print("\n--- Gap Analysis ---")
        for g, v in pat.get("gap_stats", {}).items():
            print(f"  {g}: WR={v['win_rate']}% avg={v['avg_ret_pct']}% vol={v['vol_pct']}%")

        print("\n--- Day of Week ---")
        for d, v in pat["day_of_week"].items():
            print(f"  {d}: WR={v['win_rate']}% avg={v['avg_ret_pct']}% vol={v['vol_pct']}%")

        print("\n--- Monthly ---")
        for m, v in sorted(pat["monthly"].items()):
            print(f"  {m}: WR={v['win_rate']}% avg={v['avg_ret_pct']}%")

        for period in [20, 30]:
            for n_std in [2.0, 2.5]:
                key = f"bb_fwd_p{period}_s{n_std}"
                print(f"\n--- BB (p={period}, s={n_std}) forward returns ---")
                for r in report[sym][key]:
                    print(f"  {r['forward_bars']} bars ahead: WR={r['win_rate']}% avg={r['avg_ret_pct']}% sharpe={r['sharpe']} trades={r['total_trades']}")

    with open("index_patterns.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n Saved to index_patterns.json")

if __name__ == "__main__":
    main()
