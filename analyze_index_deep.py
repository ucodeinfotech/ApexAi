import pandas as pd
import numpy as np
import os, json
from datetime import datetime

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

def returns_analysis(df, name):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
    df["intra_candle_ret"] = (df["close"] - df["open"]) / df["open"] * 100
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["body"] = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_shadow"] = df[["open","close"]].min(axis=1) - df["low"]
    df["real_body"] = df["close"] - df["open"]
    bullish = df["real_body"] > 0
    bearish = df["real_body"] < 0
    doji = abs(df["real_body"]) <= (df["high"] - df["low"]) * 0.1

    r = df["ret"].dropna()
    lr = df["log_ret"].dropna()

    results = {
        "name": name,
        "rows": len(df),
        "date_from": str(df["datetime"].min()),
        "date_to": str(df["datetime"].max()),
        "trading_days": df["date"].nunique(),
        "price_now": round(df["close"].iloc[-1], 2),
        "price_min": round(df["close"].min(), 2),
        "price_max": round(df["close"].max(), 2),
        "return_mean_pct": round(r.mean() * 100, 6),
        "return_std_pct": round(r.std() * 100, 6),
        "return_skew": round(r.skew(), 4),
        "return_kurtosis": round(r.kurtosis(), 4),
        "return_min_pct": round(r.min() * 100, 4),
        "return_max_pct": round(r.max() * 100, 4),
        "logret_std_pct": round(lr.std() * 100, 6),
        "avg_range_pct": round(df["range_pct"].mean(), 4),
        "avg_body_pct": round((df["body"] / df["close"].shift(1) * 100).mean(), 4),
        "avg_gap_pct": round(df["gap"].mean(), 4),
        "gap_std_pct": round(df["gap"].std(), 4),
        "bullish_pct": round(bullish.sum() / len(df) * 100, 2),
        "bearish_pct": round(bearish.sum() / len(df) * 100, 2),
        "doji_pct": round(doji.sum() / len(df) * 100, 2),
        "pos_ret_pct": round((r > 0).sum() / len(r) * 100, 2),
    }

    # Volatility by year
    yearly_vol = df.groupby("year")["ret"].std() * 100 * np.sqrt(252*26)
    results["yearly_vol_mean"] = round(yearly_vol.mean(), 2)
    results["yearly_vol_min"] = round(yearly_vol.min(), 2)
    results["yearly_vol_max"] = round(yearly_vol.max(), 2)

    # Month seasonality
    monthly_ret = df.groupby("month")["ret"].mean() * 100
    results["best_month"] = int(monthly_ret.idxmax())
    results["best_month_ret"] = round(monthly_ret.max(), 4)
    results["worst_month"] = int(monthly_ret.idxmin())
    results["worst_month_ret"] = round(monthly_ret.min(), 4)

    # DOW seasonality
    dow_ret = df.groupby("dow")["ret"].mean() * 100
    results["best_dow"] = int(dow_ret.idxmax())
    results["best_dow_ret"] = round(dow_ret.max(), 4)
    results["worst_dow"] = int(dow_ret.idxmin())
    results["worst_dow_ret"] = round(dow_ret.min(), 4)

    # Intraday hour patterns
    df["period"] = df["hour"] * 100 + df["minute"]
    def label_session(h, m):
        if h == 9 and m >= 15: return "open"
        elif h <= 10: return "morning"
        elif h <= 12: return "midday"
        elif h <= 14: return "afternoon"
        else: return "close"
    df["session"] = df.apply(lambda r: label_session(r["hour"], r["minute"]), axis=1)
    session_vol = df.groupby("session")["range_pct"].mean()
    results["session_vol_open"] = round(session_vol.get("open", 0), 4)
    results["session_vol_morning"] = round(session_vol.get("morning", 0), 4)
    results["session_vol_midday"] = round(session_vol.get("midday", 0), 4)
    results["session_vol_afternoon"] = round(session_vol.get("afternoon", 0), 4)
    results["session_vol_close"] = round(session_vol.get("close", 0), 4)

    # Consecutive moves
    df["direction"] = np.sign(df["real_body"])
    col = (df["direction"] != 0).astype(int)
    df["streak"] = col.groupby((col != col.shift()).cumsum()).cumcount() + 1
    df["streak_dir"] = df["streak"] * df["direction"]

    long_streaks = df[df["direction"] == 1]["streak"]
    short_streaks = df[df["direction"] == -1]["streak"]
    results["avg_bull_streak"] = round(long_streaks.mean(), 2) if len(long_streaks) > 0 else 0
    results["max_bull_streak"] = int(long_streaks.max()) if len(long_streaks) > 0 else 0
    results["avg_bear_streak"] = round(short_streaks.mean(), 2) if len(short_streaks) > 0 else 0
    results["max_bear_streak"] = int(short_streaks.max()) if len(short_streaks) > 0 else 0

    # Volatility clustering (serial correlation of absolute returns)
    abs_ret = np.abs(r)
    results["vol_clustering_corr"] = round(pd.Series(abs_ret).autocorr(lag=1), 4)
    results["ret_autocorr"] = round(r.autocorr(lag=1), 4)

    # Tail risk
    var_95 = r.quantile(0.05)
    var_99 = r.quantile(0.01)
    cvar_95 = r[r <= var_95].mean()
    results["var_95_pct"] = round(var_95 * 100, 4)
    results["var_99_pct"] = round(var_99 * 100, 4)
    results["cvar_95_pct"] = round(cvar_95 * 100, 4)

    # Gap analysis
    gaps = df["gap"].dropna()
    big_gaps = gaps[abs(gaps) > 0.5]
    results["gap_freq_pct"] = round(len(big_gaps) / len(gaps) * 100, 2)
    results["gap_fill_freq"] = "N/A (intraday fill not computed)"
    results["avg_gap_up"] = round(gaps[gaps > 0].mean(), 4) if (gaps > 0).sum() > 0 else 0
    results["avg_gap_down"] = round(gaps[gaps < 0].mean(), 4) if (gaps < 0).sum() > 0 else 0

    # Range contraction / expansion (for BB squeeze)
    df["range_20ma"] = df["range_pct"].rolling(20).mean()
    df["range_ratio"] = df["range_pct"] / df["range_20ma"]
    results["range_shrink_pct"] = round((df["range_ratio"] < 0.7).sum() / len(df) * 100, 2)
    results["range_expand_pct"] = round((df["range_ratio"] > 1.3).sum() / len(df) * 100, 2)

    return results, df

def main():
    report = {}
    all_dfs = {}
    for key, sym in INDICES.items():
        print(f"\n Analyzing {sym}...")
        res, df = returns_analysis(load(sym), sym)
        report[key] = res
        all_dfs[key] = df
        for k, v in res.items():
            if k != "name":
                print(f"  {k}: {v}")

    # Cross-correlation
    print("\n\n=== CROSS-CORRELATION OF 15-MIN RETURNS ===")
    for k, df in all_dfs.items():
        df["ret_close"] = df["close"].pct_change()
    corr_data = pd.DataFrame({k: df["ret_close"] for k, df in all_dfs.items()})
    print(corr_data.corr().to_string())

    # Save report
    report["correlation"] = corr_data.corr().to_dict()
    with open("index_deep_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n Report saved to index_deep_analysis.json")

if __name__ == "__main__":
    main()
