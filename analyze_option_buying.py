import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

DATA_DIR = "nifty50_full_history"
INDICES = {"NIFTY50": "NIFTY50", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}

def load_5min(sym):
    df = pd.read_csv(f"{DATA_DIR}/{sym}_FIVE_MINUTE.csv")
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

def option_buying_insights(df, name):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
    df["intra_ret"] = (df["close"] - df["open"]) / df["open"] * 100
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100
    df["upper_w"] = (df["high"] - df[["open","close"]].max(axis=1)) / df["close"].shift(1) * 100
    df["lower_w"] = (df[["open","close"]].min(axis=1) - df["low"]) / df["close"].shift(1) * 100

    # Group by date for daily stats
    daily = df.groupby("date").agg(
        day_open=("open","first"), day_high=("high","max"),
        day_low=("low","min"), day_close=("close","last"),
        day_range=("range_pct","sum"), n_5min=("ret","count"),
        day_ret=("close", lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100),
    ).reset_index()
    daily["day_range_pct"] = (daily["day_high"] - daily["day_low"]) / daily["day_open"] * 100

    insights = {}

    # === 1. BIG MOVE DAYS ===
    big_up = daily[daily["day_ret"] > 1.0]
    big_down = daily[daily["day_ret"] < -1.0]
    insights["big_move_days"] = {
        "total_days": len(daily),
        "days_up_1pct": len(big_up),
        "pct_up_1pct": round(len(big_up)/len(daily)*100, 2),
        "days_down_1pct": len(big_down),
        "pct_down_1pct": round(len(big_down)/len(daily)*100, 2),
        "days_over_2pct": len(daily[abs(daily["day_ret"]) > 2]),
        "days_over_3pct": len(daily[abs(daily["day_ret"]) > 3]),
    }

    # === 2. GAP DAYS ===
    daily["gap"] = (daily["day_open"] - daily["day_close"].shift(1)) / daily["day_close"].shift(1) * 100
    gap_up_days = daily[daily["gap"] > 0.5]
    gap_down_days = daily[daily["gap"] < -0.5]
    big_gap_days = daily[abs(daily["gap"]) > 1.0]
    insights["gap_days"] = {
        "gap_up_0.5pct": len(gap_up_days),
        "gap_down_0.5pct": len(gap_down_days),
        "gap_over_1pct": len(big_gap_days),
        "gap_up_avg_move": round(gap_up_days["day_range_pct"].mean(), 2) if len(gap_up_days) > 0 else 0,
        "gap_down_avg_move": round(gap_down_days["day_range_pct"].mean(), 2) if len(gap_down_days) > 0 else 0,
        "no_gap_avg_move": round(daily[abs(daily["gap"]) < 0.3]["day_range_pct"].mean(), 2),
    }

    # === 3. EXPIRY DAY PATTERNS (Thu for weekly) ===
    daily["dow"] = pd.to_datetime(daily["date"].astype(str)).dt.dayofweek
    thu = daily[daily["dow"] == 3]
    fri = daily[daily["dow"] == 4]
    insights["expiry_patterns"] = {
        "thu_days": len(thu),
        "thu_avg_range": round(thu["day_range_pct"].mean(), 2) if len(thu) > 0 else 0,
        "thu_avg_ret": round(thu["day_ret"].mean(), 2) if len(thu) > 0 else 0,
        "thu_win_rate": round((thu["day_ret"] > 0).sum()/len(thu)*100, 1) if len(thu) > 0 else 0,
        "fri_days": len(fri),
        "fri_avg_range": round(fri["day_range_pct"].mean(), 2) if len(fri) > 0 else 0,
        "fri_avg_ret": round(fri["day_ret"].mean(), 2) if len(fri) > 0 else 0,
        "fri_win_rate": round((fri["day_ret"] > 0).sum()/len(fri)*100, 1) if len(fri) > 0 else 0,
    }

    # === 4. CONSECUTIVE RANGE EXPANSION ===
    df["range_10ma"] = df["range_pct"].rolling(10).mean()
    df["range_expand"] = df["range_pct"] > df["range_10ma"] * 1.5

    # After a quiet period (squeeze), what happens next?
    df["squeeze"] = df["range_pct"] < df["range_10ma"].shift(1) * 0.6
    squeeze_next_range = df["range_pct"].shift(-1)[df["squeeze"]]
    insights["squeeze_stats"] = {
        "squeeze_bars": df["squeeze"].sum(),
        "squeeze_pct": round(df["squeeze"].sum()/len(df)*100, 1),
        "next_bar_avg_range": round(squeeze_next_range.mean(), 3) if len(squeeze_next_range) > 0 else 0,
        "normal_avg_range": round(df.loc[~df["squeeze"], "range_pct"].mean(), 3),
        "expansion_ratio": round(squeeze_next_range.mean() / df.loc[~df["squeeze"], "range_pct"].mean(), 2) if len(squeeze_next_range) > 0 else 0,
    }

    # === 5. DIRECTIONAL SUSTAINED MOVES (for Gamma) ===
    df["direction"] = np.sign(df["intra_ret"])
    df["streak_id"] = (df["direction"] != df["direction"].shift(1)).cumsum()
    streaks = df.groupby("streak_id").agg(
        bars=("direction","count"), direction=("direction","first"),
        total_ret=("ret","sum"), start_dt=("datetime","first"), end_dt=("datetime","last"),
    ).reset_index()
    streaks["duration_min"] = (streaks["end_dt"] - streaks["start_dt"]).dt.total_seconds() / 60

    buy_streaks = streaks[(streaks["direction"] == 1) & (streaks["bars"] >= 6)]
    sell_streaks = streaks[(streaks["direction"] == -1) & (streaks["bars"] >= 6)]
    insights["sustained_moves"] = {
        "buy_moves_30min_plus": len(buy_streaks),
        "buy_avg_bars": round(buy_streaks["bars"].mean(), 1) if len(buy_streaks) > 0 else 0,
        "buy_avg_return": round(buy_streaks["total_ret"].mean()*100, 2) if len(buy_streaks) > 0 else 0,
        "sell_moves_30min_plus": len(sell_streaks),
        "sell_avg_bars": round(sell_streaks["bars"].mean(), 1) if len(sell_streaks) > 0 else 0,
        "sell_avg_return": round(abs(sell_streaks["total_ret"]).mean()*100, 2) if len(sell_streaks) > 0 else 0,
        "pct_days_with_sustained_move": round((len(buy_streaks)+len(sell_streaks)) / len(daily) * 100, 1),
    }

    # === 6. TIME-BASED PREMIUM DECAY vs MOVEMENT (theta analysis) ===
    # Which time slots give best movement per minute?
    df["period"] = df["hour"] * 100 + df["minute"]
    sessions = {
        "open_surge (9:15-9:30)": (915, 930),
        "morning (9:30-11:00)": (930, 1100),
        "midday_lull (11:00-13:00)": (1100, 1300),
        "afternoon_recovery (13:00-14:30)": (1300, 1430),
        "close_action (14:30-15:15)": (1430, 1515),
    }
    session_stats = {}
    for label, (s, e) in sessions.items():
        mask = (df["period"] >= s) & (df["period"] < e)
        chunk = df[mask]
        mins = (e - s) // 100 * 60 + (e - s) % 100
        session_stats[label] = {
            "n_5min_bars": len(chunk),
            "avg_abs_range_pct": round(chunk["range_pct"].mean(), 3),
            "avg_ret_pct": round(chunk["intra_ret"].mean(), 4),
            "movement_per_hour": round(chunk["range_pct"].mean() * (60/5), 2),
            "win_rate": round((chunk["intra_ret"] > 0).sum()/len(chunk)*100, 1),
            "big_bar_freq_pct": round((chunk["range_pct"] > 0.5).sum()/len(chunk)*100, 1),
        }
    insights["session_premium_value"] = session_stats

    # === 7. MONTHLY EXPIRY (last Thu) ANALYSIS ===
    daily_df = daily.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"].astype(str))
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["year"] = daily_df["date"].dt.year
    # Find last Thursday of each month
    monthly_expiry = daily_df[
        (daily_df["dow"] == 3) &  # Thursday
        (daily_df["date"].dt.day >= 22)  # Last week
    ]
    insights["monthly_expiry"] = {
        "n_expiries": len(monthly_expiry),
        "avg_range": round(monthly_expiry["day_range_pct"].mean(), 2),
        "avg_ret": round(monthly_expiry["day_ret"].mean(), 2),
        "win_rate": round((monthly_expiry["day_ret"] > 0).sum()/len(monthly_expiry)*100, 1),
        "big_range_days_pct": round((monthly_expiry["day_range_pct"] > 2).sum()/len(monthly_expiry)*100, 1),
    }

    # === 8. VOLATILITY REGIME DURATION (how long do quiet/loud periods last?) ===
    vq = df["range_10ma"].quantile(0.33)
    vh = df["range_10ma"].quantile(0.67)
    vreg = pd.Series("normal", index=df.index)
    vreg[df["range_10ma"] <= vq] = "quiet"
    vreg[df["range_10ma"] >= vh] = "active"
    reg_sw = (vreg != vreg.shift(1)).cumsum()
    reg_gb = df.groupby([vreg.rename("vreg"), reg_sw.rename("sid")], observed=True).agg(n_bars=("range_pct","count")).reset_index()
    rmean = reg_gb.groupby("vreg", observed=True)["n_bars"].mean()
    rmax = reg_gb.groupby("vreg", observed=True)["n_bars"].max()
    rcnt = reg_gb.groupby("vreg", observed=True)["n_bars"].count()
    insights["regime_duration"] = {}
    for regime in rmean.index:
        insights["regime_duration"][str(regime)] = {
            "avg_bars": round(rmean[regime], 0),
            "max_bars": round(rmax[regime], 0),
            "n_occurrences": int(rcnt[regime]),
        }

    return insights

def main():
    report = {}
    for name, sym in INDICES.items():
        print(f"\n{'='*60}")
        print(f"  {sym} — OPTION BUYING INSIGHTS")
        print(f"{'='*60}")
        df = load_5min(sym)
        ins = option_buying_insights(df, sym)
        report[sym] = ins

        for section, data in ins.items():
            print(f"\n  >> {section.replace('_',' ').upper()}")
            for k, v in data.items():
                if isinstance(v, dict):
                    print(f"     {k}:")
                    for sk, sv in v.items():
                        print(f"       {sk}: {sv}")
                else:
                    print(f"     {k}: {v}")

    with open("option_buying_insights.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n Saved to option_buying_insights.json")

if __name__ == "__main__":
    main()
