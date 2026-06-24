import pandas as pd
import numpy as np
import json

DATA_DIR = "nifty50_full_history"
INDICES = {"NIFTY50": "NIFTY50", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}
TF = "FIVE_MINUTE"

def load(sym):
    df = pd.read_csv(f"{DATA_DIR}/{sym}_{TF}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df["prev_close"] = df["close"].shift(1)
    return df

def gap_analysis(df, name):
    # First bar of each day
    df["is_first"] = df["date"] != df["date"].shift(1)
    first_bars = df[df["is_first"]].copy()
    # Get previous day's last close
    daily_close = df.groupby("date")["close"].last().shift(1)
    first_bars = first_bars.merge(daily_close.rename("prev_day_close"), left_on="date", right_index=True, how="left")
    
    first_bars["gap_pct"] = (first_bars["open"] - first_bars["prev_day_close"]) / first_bars["prev_day_close"] * 100
    first_bars["gap_pts"] = first_bars["open"] - first_bars["prev_day_close"]
    
    # Filter gap-up days (>0.2% gap)
    gap_up = first_bars[first_bars["gap_pct"] > 0.2].copy()
    gap_up_minor = first_bars[(first_bars["gap_pct"] > 0.2) & (first_bars["gap_pct"] <= 0.5)].copy()
    gap_up_moderate = first_bars[(first_bars["gap_pct"] > 0.5) & (first_bars["gap_pct"] <= 1.0)].copy()
    gap_up_big = first_bars[first_bars["gap_pct"] > 1.0].copy()
    gap_up_all = first_bars[first_bars["gap_pct"] > 0.0].copy()
    
    gap_down = first_bars[first_bars["gap_pct"] < -0.2].copy()
    no_gap = first_bars[abs(first_bars["gap_pct"]) <= 0.2].copy()
    
    results = {
        "total_trading_days": len(first_bars),
        "gap_up_days_gt_0pct": len(first_bars[first_bars["gap_pct"] > 0]),
        "gap_down_days_lt_0pct": len(first_bars[first_bars["gap_pct"] < 0]),
        "gap_up_0.2_to_0.5pct": len(gap_up_minor),
        "gap_up_0.5_to_1.0pct": len(gap_up_moderate),
        "gap_up_gt_1.0pct": len(gap_up_big),
    }
    
    # === GAP FILL ANALYSIS ===
    def analyze_gap_fill(gap_df, label):
        if len(gap_df) == 0:
            return {f"{label}_count": 0}
        
        filled_same_day = 0
        filled_next_day = 0
        never_filled = 0
        fill_times = []
        fill_pcts = []
        
        for _, row in gap_df.iterrows():
            d = row["date"]
            gap_open = row["open"]
            prev_close = row["prev_day_close"]
            gap_size = abs(gap_open - prev_close)
            
            # Same day bars
            day_bars = df[(df["date"] == d) & (df["datetime"] > row["datetime"])].copy()
            
            filled = False
            fill_dt = None
            for _, bar in day_bars.iterrows():
                # Gap up: check if low touches or goes below prev_close
                if bar["low"] <= prev_close:
                    filled = True
                    fill_dt = bar["datetime"]
                    break
            
            if filled:
                filled_same_day += 1
                time_min = (pd.Timestamp(fill_dt) - pd.Timestamp(row["datetime"])).total_seconds() / 60
                fill_times.append(time_min)
                fill_pcts.append(gap_size)
            else:
                # Check next day
                from datetime import timedelta as _td
                next_day = d + _td(days=1)
                next_bars = df[(df["date"] == next_day)].copy()
                for _, bar in next_bars.iterrows():
                    if bar["low"] <= prev_close:
                        filled = True
                        break
                
                if filled:
                    filled_next_day += 1
                else:
                    never_filled += 1
        
        total = len(gap_df)
        return {
            f"{label}_count": total,
            f"{label}_filled_same_day_pct": round(filled_same_day / total * 100, 1) if total else 0,
            f"{label}_filled_same_day_n": filled_same_day,
            f"{label}_filled_next_day_pct": round(filled_next_day / total * 100, 1) if total else 0,
            f"{label}_filled_next_day_n": filled_next_day,
            f"{label}_never_filled_pct": round(never_filled / total * 100, 1) if total else 0,
            f"{label}_never_filled_n": never_filled,
            f"{label}_avg_fill_time_min": round(np.mean(fill_times), 1) if fill_times else None,
            f"{label}_median_fill_time_min": round(np.median(fill_times), 1) if fill_times else None,
            f"{label}_fill_time_std_min": round(np.std(fill_times), 1) if fill_times else None,
            f"{label}_fastest_fill_min": round(min(fill_times), 1) if fill_times else None,
            f"{label}_slowest_fill_min": round(max(fill_times), 1) if fill_times else None,
        }
    
    results.update(analyze_gap_fill(gap_up_minor, "minor_gap_up"))
    results.update(analyze_gap_fill(gap_up_moderate, "moderate_gap_up"))
    results.update(analyze_gap_fill(gap_up_big, "big_gap_up"))
    results.update(analyze_gap_fill(gap_up, "all_gap_up"))
    
    # === GAP DOWN FILL ANALYSIS ===
    def analyze_gap_down_fill(gap_df, label):
        if len(gap_df) == 0:
            return {f"{label}_count": 0}
        
        filled_same_day = 0
        filled_next_day = 0
        never_filled = 0
        fill_times = []
        
        for _, row in gap_df.iterrows():
            d = row["date"]
            gap_open = row["open"]
            prev_close = row["prev_day_close"]
            
            day_bars = df[(df["date"] == d) & (df["datetime"] > row["datetime"])].copy()
            
            filled = False
            fill_dt = None
            for _, bar in day_bars.iterrows():
                if bar["high"] >= prev_close:
                    filled = True
                    fill_dt = bar["datetime"]
                    break
            
            if filled:
                filled_same_day += 1
                time_min = (pd.Timestamp(fill_dt) - pd.Timestamp(row["datetime"])).total_seconds() / 60
                fill_times.append(time_min)
            else:
                from datetime import timedelta as _td
                next_day = d + _td(days=1)
                next_bars = df[(df["date"] == next_day)].copy()
                for _, bar in next_bars.iterrows():
                    if bar["high"] >= prev_close:
                        filled = True
                        break
                if filled:
                    filled_next_day += 1
                else:
                    never_filled += 1
        
        total = len(gap_df)
        return {
            f"{label}_count": total,
            f"{label}_filled_same_day_pct": round(filled_same_day / total * 100, 1) if total else 0,
            f"{label}_filled_same_day_n": filled_same_day,
            f"{label}_filled_next_day_pct": round(filled_next_day / total * 100, 1) if total else 0,
            f"{label}_never_filled_pct": round(never_filled / total * 100, 1) if total else 0,
            f"{label}_avg_fill_time_min": round(np.mean(fill_times), 1) if fill_times else None,
            f"{label}_median_fill_time_min": round(np.median(fill_times), 1) if fill_times else None,
        }
    
    gap_down_minor = first_bars[(first_bars["gap_pct"] < -0.2) & (first_bars["gap_pct"] >= -0.5)].copy()
    gap_down_moderate = first_bars[(first_bars["gap_pct"] < -0.5) & (first_bars["gap_pct"] >= -1.0)].copy()
    gap_down_big = first_bars[first_bars["gap_pct"] < -1.0].copy()
    gap_down_all = first_bars[first_bars["gap_pct"] < -0.2].copy()
    
    results.update(analyze_gap_down_fill(gap_down_minor, "minor_gap_down"))
    results.update(analyze_gap_down_fill(gap_down_moderate, "moderate_gap_down"))
    results.update(analyze_gap_down_fill(gap_down_big, "big_gap_down"))
    results.update(analyze_gap_down_fill(gap_down_all, "all_gap_down"))
    
    # === HOW GAP-UP DAYS END (close relative to prev close) ===
    gap_up_days = first_bars[first_bars["gap_pct"] > 0.2].copy()
    day_closes = df.groupby("date").last().reset_index()
    gap_up_days = gap_up_days.merge(day_closes[["date","close"]], on="date", suffixes=("","_dayclose"))
    gap_up_days["close_vs_prevclose"] = (gap_up_days["close_dayclose"] - gap_up_days["prev_day_close"]) / gap_up_days["prev_day_close"] * 100
    gap_up_days["gap_filled"] = gap_up_days["close_dayclose"] <= gap_up_days["prev_day_close"]
    
    results["gap_up_day_end_stats"] = {
        "avg_close_vs_prevclose_pct": round(gap_up_days["close_vs_prevclose"].mean(), 2),
        "median_close_vs_prevclose_pct": round(gap_up_days["close_vs_prevclose"].median(), 2),
        "pct_days_end_above_open": round((gap_up_days["close_dayclose"] > gap_up_days["prev_day_close"]).sum() / len(gap_up_days) * 100, 1),
        "pct_days_close_negative": round(gap_up_days["gap_filled"].sum() / len(gap_up_days) * 100, 1),
        "avg_direction": round(gap_up_days["close_vs_prevclose"].mean(), 2),
        "max_up_close": round(gap_up_days["close_vs_prevclose"].max(), 2),
        "max_down_close": round(gap_up_days["close_vs_prevclose"].min(), 2),
    }
    
    # === GAP FILL TIME DISTRIBUTION ===
    results["fill_time_distribution"] = {}
    for label, gdf in [("minor", gap_up_minor), ("moderate", gap_up_moderate), ("big", gap_up_big)]:
        if len(gdf) == 0:
            continue
        times = []
        for _, row in gdf.iterrows():
            d = row["date"]
            prev_close = row["prev_day_close"]
            day_bars = df[(df["date"] == d) & (df["datetime"] > row["datetime"])].copy()
            for _, bar in day_bars.iterrows():
                if bar["low"] <= prev_close:
                    t = (pd.Timestamp(bar["datetime"]) - pd.Timestamp(row["datetime"])).total_seconds() / 60
                    times.append(t)
                    break
        if times:
            results["fill_time_distribution"][f"{label}_gap_up"] = {
                "lt_15min": round(sum(1 for t in times if t <= 15) / len(times) * 100, 1),
                "15_to_30min": round(sum(1 for t in times if 15 < t <= 30) / len(times) * 100, 1),
                "30_to_60min": round(sum(1 for t in times if 30 < t <= 60) / len(times) * 100, 1),
                "60_to_120min": round(sum(1 for t in times if 60 < t <= 120) / len(times) * 100, 1),
                "gt_120min": round(sum(1 for t in times if t > 120) / len(times) * 100, 1),
            }
    
    return results, gap_up

def print_results(name, res):
    print(f"\n{'='*60}")
    print(f"  {name} — COMPLETE GAP ANALYSIS")
    print(f"{'='*60}")
    print(f"\n  Total trading days: {res['total_trading_days']}")
    print(f"  Gap-up days (>0%): {res['gap_up_days_gt_0pct']} ({round(res['gap_up_days_gt_0pct']/res['total_trading_days']*100,1)}%)")
    print(f"  Gap-down days (<0%): {res['gap_down_days_lt_0pct']} ({round(res['gap_down_days_lt_0pct']/res['total_trading_days']*100,1)}%)")

    print(f"\n  --- GAP-UP FREQUENCY ---")
    print(f"  Minor gap-up (0.2-0.5%): {res['minor_gap_up_count']} ({round(res['minor_gap_up_count']/res['total_trading_days']*100,1)}% of days)")
    print(f"  Moderate gap-up (0.5-1.0%): {res['moderate_gap_up_count']} ({round(res['moderate_gap_up_count']/res['total_trading_days']*100,1)}%)")
    print(f"  Big gap-up (>1.0%): {res['big_gap_up_count']} ({round(res['big_gap_up_count']/res['total_trading_days']*100,1)}%)")

    print(f"\n  --- GAP-UP FILL RATES ---")
    for size in ["minor", "moderate", "big"]:
        cnt = res.get(f"{size}_gap_up_count", 0)
        if cnt == 0: continue
        print(f"\n  [{size.upper()} gap-up: {cnt} days]")
        print(f"    Filled same day: {res.get(f'{size}_gap_up_filled_same_day_pct',0)}% ({res.get(f'{size}_gap_up_filled_same_day_n',0)} days)")
        print(f"    Filled next day: {res.get(f'{size}_gap_up_filled_next_day_pct',0)}% ({res.get(f'{size}_gap_up_filled_next_day_n',0)} days)")
        print(f"    Never filled: {res.get(f'{size}_gap_up_never_filled_pct',0)}% ({res.get(f'{size}_gap_up_never_filled_n',0)} days)")
        ft = res.get(f"{size}_gap_up_avg_fill_time_min", None)
        if ft:
            print(f"    Avg fill time: {ft} min (median: {res.get(f'{size}_gap_up_median_fill_time_min')} min, std: {res.get(f'{size}_gap_up_fill_time_std_min')} min)")
            print(f"    Fastest fill: {res.get(f'{size}_gap_up_fastest_fill_min')} min, Slowest: {res.get(f'{size}_gap_up_slowest_fill_min')} min")
    
    print(f"\n  --- GAP-DOWN FILL RATES ---")
    for size in ["minor", "moderate", "big"]:
        cnt = res.get(f"{size}_gap_down_count", 0)
        if cnt == 0: continue
        print(f"\n  [{size.upper()} gap-down: {cnt} days]")
        print(f"    Filled same day: {res.get(f'{size}_gap_down_filled_same_day_pct',0)}% ({res.get(f'{size}_gap_down_filled_same_day_n',0)} days)")
        print(f"    Filled next day: {res.get(f'{size}_gap_down_filled_next_day_pct',0)}% ")
        print(f"    Never filled: {res.get(f'{size}_gap_down_never_filled_pct',0)}%")
        ft = res.get(f"{size}_gap_down_avg_fill_time_min", None)
        if ft:
            print(f"    Avg fill time: {ft} min (median: {res.get(f'{size}_gap_down_median_fill_time_min')} min)")

    print(f"\n  --- HOW GAP-UP DAYS END ---")
    end = res["gap_up_day_end_stats"]
    print(f"    Avg close vs prev close: {end['avg_close_vs_prevclose_pct']}%")
    print(f"    Median close vs prev close: {end['median_close_vs_prevclose_pct']}%")
    print(f"    Days ending above prev close: {end['pct_days_end_above_open']}%")
    print(f"    Days closing negative (gap filled): {end['pct_days_close_negative']}%")
    print(f"    Range of close outcomes: {end['max_down_close']}% to {end['max_up_close']}% (max down to max up)")

    print(f"\n  --- FILL TIME DISTRIBUTION ---")
    for size_key, dist in res.get("fill_time_distribution", {}).items():
        print(f"  [{size_key}]")
        for k, v in dist.items():
            print(f"    {k}: {v}%")

def main():
    for name, sym in INDICES.items():
        df = load(sym)
        res, gap_days = gap_analysis(df, sym)
        print_results(name, res)

if __name__ == "__main__":
    main()
