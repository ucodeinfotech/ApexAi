import pandas as pd
import numpy as np
import os, time, sys, json

DATA_DIR = "nifty50_full_history"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BROKERAGE_PER_ORDER = 10
STT = 0.001; EXCHANGE_TC = 0.00003; SEBI_TC = 0.000001
GST = 0.18; STAMP_DUTY = 0.00003

def compute_charges(entry_price, exit_price, qty=1):
    tb = entry_price * qty; ts = exit_price * qty
    return (BROKERAGE_PER_ORDER * 2 + ts * STT + (tb+ts) * EXCHANGE_TC
            + (tb+ts) * SEBI_TC * 2 + tb * STAMP_DUTY
            + (BROKERAGE_PER_ORDER * 2 + (tb+ts) * EXCHANGE_TC) * GST)

INDICES = ["NIFTY50", "BANKNIFTY", "SENSEX"]

# === STRATEGY VARIANTS ===

def breakout_v1(df, params):
    """Original: SD=?, sliding trigger, enter on H/L break of trigger"""
    p = params["period"]; ns = params["n_std"]; slide = params.get("slide", True)
    rr = params.get("rr", 2.0); min_gap_bars = params.get("min_gap_bars", 1)
    ma = df["close"].rolling(p).mean(); std = df["close"].rolling(p).std()
    upper = ma + ns * std; lower = ma - ns * std
    trades = []
    i = 0
    while i < len(df):
        if df.iloc[i]["low"] > upper.iloc[i]:
            typ, trigger_high, trigger_low, trig_i = "SHORT", df.iloc[i]["high"], df.iloc[i]["low"], i
        elif df.iloc[i]["high"] < lower.iloc[i]:
            typ, trigger_high, trigger_low, trig_i = "LONG", df.iloc[i]["high"], df.iloc[i]["low"], i
        else:
            i += 1; continue
        entry_made = False; j = trig_i + 1
        while j < len(df):
            nr = df.iloc[j]
            if (typ == "SHORT" and nr["low"] <= trigger_low) or (typ == "LONG" and nr["high"] >= trigger_high):
                entry_price = min(nr["open"], trigger_low) if typ == "SHORT" else max(nr["open"], trigger_high)
                entry_price = round(entry_price, 2)
                sl_price = trigger_high if typ == "SHORT" else trigger_low
                entry_time = nr["datetime"]; entry_made = True; break
            if slide:
                if (typ == "SHORT" and nr["low"] > upper.iloc[j]) or (typ == "LONG" and nr["high"] < lower.iloc[j]):
                    trigger_high = max(trigger_high, nr["high"])
                    trigger_low = min(trigger_low, nr["low"]); j += 1; continue
            break
        if not entry_made: i = j + 1; continue
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0: i = j + 1; continue
        tp_price = entry_price - sl_dist * rr if typ == "SHORT" else entry_price + sl_dist * rr
        k = j + 1; exit_p, exit_t, reason = entry_price, entry_time, "EOD"
        while k < len(df):
            b = df.iloc[k]; bdt = b["datetime"]
            if bdt.hour >= 15 and bdt.minute >= 15:
                exit_p = b["close"]; exit_t = bdt; reason = "EOD"; break
            tp_hit = (typ == "SHORT" and b["low"] <= tp_price) or (typ == "LONG" and b["high"] >= tp_price)
            sl_hit = (typ == "SHORT" and b["high"] >= sl_price) or (typ == "LONG" and b["low"] <= sl_price)
            if tp_hit and sl_hit:
                if typ == "SHORT":
                    exit_p = tp_price if b["low"] <= tp_price else sl_price
                else:
                    exit_p = tp_price if b["high"] >= tp_price else sl_price
                exit_t = bdt; reason = "TP"; break
            elif tp_hit: exit_p = tp_price; exit_t = bdt; reason = "TP"; break
            elif sl_hit: exit_p = sl_price; exit_t = bdt; reason = "SL"; break
            k += 1
        pnl = (entry_price - exit_p) if typ == "SHORT" else (exit_p - entry_price)
        charges = compute_charges(entry_price, exit_p)
        trades.append({"symbol": params["_sym"], "date": str(df.iloc[trig_i]["datetime"].date()),
            "type": typ, "trigger_time": str(df.iloc[trig_i]["datetime"]),
            "entry_time": str(entry_time), "exit_time": str(exit_t),
            "entry": round(entry_price,2), "exit": round(exit_p,2),
            "sl": round(sl_price,2), "tp": round(tp_price,2),
            "reason": reason, "pnl": round(pnl,2), "charges": round(charges,2),
            "net_pnl": round(pnl-charges,2), "r": round(pnl/sl_dist,2) if sl_dist>0 else 0})
        i = k + 1 if k < len(df) else len(df)
    return trades

def meanrev_v1(df, params):
    """Mean reversion: touch band = fade. SD=?, Entry on next candle open"""
    p = params["period"]; ns = params["n_std"]; rr = params.get("rr", 2.0)
    ma = df["close"].rolling(p).mean(); std = df["close"].rolling(p).std()
    upper = ma + ns * std; lower = ma - ns * std
    trades = []
    for i in range(p, len(df)-1):
        row = df.iloc[i]
        # Touch: high breaks upper OR low breaks lower (any part of candle crosses)
        if row["low"] < upper.iloc[i] and row["high"] > upper.iloc[i] and row["close"] < row["open"]:
            # Bearish candle touching upper = SHORT
            typ = "SHORT"
            entry_price = row["close"]
            sl_price = row["high"]
        elif row["low"] < lower.iloc[i] and row["high"] > lower.iloc[i] and row["close"] > row["open"]:
            # Bullish candle touching lower = LONG
            typ = "LONG"
            entry_price = row["close"]
            sl_price = row["low"]
        else:
            continue
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0: continue
        tp_price = entry_price - sl_dist * rr if typ == "SHORT" else entry_price + sl_dist * rr
        k = i + 1; exit_p, exit_t, reason = entry_price, row["datetime"], "EOD"
        while k < len(df):
            b = df.iloc[k]; bdt = b["datetime"]
            if bdt.hour >= 15 and bdt.minute >= 15:
                exit_p = b["close"]; exit_t = bdt; reason = "EOD"; break
            tp_hit = (typ == "SHORT" and b["low"] <= tp_price) or (typ == "LONG" and b["high"] >= tp_price)
            sl_hit = (typ == "SHORT" and b["high"] >= sl_price) or (typ == "LONG" and b["low"] <= sl_price)
            if tp_hit and sl_hit:
                exit_p = tp_price; exit_t = bdt; reason = "TP"; break
            elif tp_hit: exit_p = tp_price; exit_t = bdt; reason = "TP"; break
            elif sl_hit: exit_p = sl_price; exit_t = bdt; reason = "SL"; break
            k += 1
        pnl = (entry_price - exit_p) if typ == "SHORT" else (exit_p - entry_price)
        charges = compute_charges(entry_price, exit_p)
        trades.append({"symbol": params["_sym"], "date": str(row["datetime"].date()),
            "type": typ, "trigger_time": str(row["datetime"]),
            "entry_time": str(row["datetime"]), "exit_time": str(exit_t),
            "entry": round(entry_price,2), "exit": round(exit_p,2),
            "sl": round(sl_price,2), "tp": round(tp_price,2),
            "reason": reason, "pnl": round(pnl,2), "charges": round(charges,2),
            "net_pnl": round(pnl-charges,2), "r": round(pnl/sl_dist,2) if sl_dist>0 else 0})
    return trades

def breakout_v2(df, params):
    """Breakout but entry at trigger candle CLOSE (market order), no sliding"""
    p = params["period"]; ns = params["n_std"]; rr = params.get("rr", 2.0)
    ma = df["close"].rolling(p).mean(); std = df["close"].rolling(p).std()
    upper = ma + ns * std; lower = ma - ns * std
    trades = []
    for i in range(p, len(df)-2):
        row = df.iloc[i]
        if row["low"] > upper.iloc[i]:
            entry_price = row["close"]
            sl_price = row["low"]  # SL at trigger low
            tp_price = entry_price - (entry_price - sl_price) * rr
            typ = "SHORT"
        elif row["high"] < lower.iloc[i]:
            entry_price = row["close"]
            sl_price = row["high"]  # SL at trigger high
            tp_price = entry_price + (sl_price - entry_price) * rr
            typ = "LONG"
        else:
            continue
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0: continue
        k = i + 1; exit_p, exit_t, reason = entry_price, row["datetime"], "EOD"
        while k < len(df):
            b = df.iloc[k]; bdt = b["datetime"]
            if bdt.hour >= 15 and bdt.minute >= 15:
                exit_p = b["close"]; exit_t = bdt; reason = "EOD"; break
            tp_hit = (typ == "SHORT" and b["low"] <= tp_price) or (typ == "LONG" and b["high"] >= tp_price)
            sl_hit = (typ == "SHORT" and b["high"] >= sl_price) or (typ == "LONG" and b["low"] <= sl_price)
            if tp_hit and sl_hit:
                exit_p = tp_price; exit_t = bdt; reason = "TP"; break
            elif tp_hit: exit_p = tp_price; exit_t = bdt; reason = "TP"; break
            elif sl_hit: exit_p = sl_price; exit_t = bdt; reason = "SL"; break
            k += 1
        pnl = (entry_price - exit_p) if typ == "SHORT" else (exit_p - entry_price)
        charges = compute_charges(entry_price, exit_p)
        trades.append({"symbol": params["_sym"], "date": str(row["datetime"].date()),
            "type": typ, "entry": round(entry_price,2), "exit": round(exit_p,2),
            "sl": round(sl_price,2), "tp": round(tp_price,2),
            "reason": reason, "pnl": round(pnl,2), "charges": round(charges,2),
            "net_pnl": round(pnl-charges,2), "r": round(pnl/sl_dist,2) if sl_dist>0 else 0})
    return trades

def squeeze_breakout(df, params):
    """BB squeeze: bands contract (width low), then first breakout = trade"""
    p = params["period"]; ns = params["n_std"]; rr = params.get("rr", 2.0)
    lookback = params.get("lookback", 10)
    ma = df["close"].rolling(p).mean(); std = df["close"].rolling(p).std()
    upper = ma + ns * std; lower = ma - ns * std
    df["bbw"] = (upper - lower) / ma * 100
    df["bbw_avg"] = df["bbw"].rolling(lookback).mean()
    trades = []
    in_squeeze = False
    for i in range(max(p, lookback), len(df)-1):
        row = df.iloc[i]
        is_squeeze = row["bbw"] < row["bbw_avg"] * 0.8
        if is_squeeze and not in_squeeze:
            in_squeeze = True
            squeeze_start = i
        elif not is_squeeze and in_squeeze:
            in_squeeze = False
            # breakout from squeeze
            break_i = i
            break_row = df.iloc[break_i]
            if break_row["close"] > ma.iloc[break_i] and break_row["close"] > break_row["open"]:
                typ = "LONG"; entry_price = break_row["close"]
                sl_price = min(df.iloc[squeeze_start:break_i+1]["low"]) if lookback > 100 else break_row["low"]
            elif break_row["close"] < ma.iloc[break_i] and break_row["close"] < break_row["open"]:
                typ = "SHORT"; entry_price = break_row["close"]
                sl_price = max(df.iloc[squeeze_start:break_i+1]["high"]) if lookback > 100 else break_row["high"]
            else:
                continue
            sl_dist = abs(entry_price - sl_price); tp_price = 0
            if sl_dist <= 0: continue
            tp_price = entry_price + sl_dist * rr if typ == "LONG" else entry_price - sl_dist * rr
            k = break_i + 1; exit_p, exit_t, reason = entry_price, break_row["datetime"], "EOD"
            while k < len(df):
                b = df.iloc[k]; bdt = b["datetime"]
                if bdt.hour >= 15 and bdt.minute >= 15: exit_p = b["close"]; exit_t = bdt; reason = "EOD"; break
                tp_hit = (typ == "LONG" and b["high"] >= tp_price) or (typ == "SHORT" and b["low"] <= tp_price)
                sl_hit = (typ == "LONG" and b["low"] <= sl_price) or (typ == "SHORT" and b["high"] >= sl_price)
                if tp_hit and sl_hit: exit_p = tp_price; exit_t = bdt; reason = "TP"; break
                elif tp_hit: exit_p = tp_price; exit_t = bdt; reason = "TP"; break
                elif sl_hit: exit_p = sl_price; exit_t = bdt; reason = "SL"; break
                k += 1
            pnl = (entry_price - exit_p) if typ == "SHORT" else (exit_p - entry_price)
            charges = compute_charges(entry_price, exit_p)
            trades.append({"symbol": params["_sym"], "date": str(break_row["datetime"].date()),
                "type": typ, "entry": round(entry_price,2), "exit": round(exit_p,2),
                "sl": round(sl_price,2), "tp": round(tp_price,2),
                "reason": reason, "pnl": round(pnl,2), "charges": round(charges,2),
                "net_pnl": round(pnl-charges,2), "r": round(pnl/sl_dist,2) if sl_dist>0 else 0})
    return trades

# Strategy registry
STRATEGIES = {
    "breakout_v1": breakout_v1,   # Original: sliding trigger, break H/L entry
    "breakout_v2": breakout_v2,   # Entry at trigger close, no slide
    "meanrev_v1": meanrev_v1,     # Fade the band touch
    "squeeze": squeeze_breakout,  # Squeeze then breakout
}

def main():
    # Parameter grid
    params_grid = []
    for strat_name in ["breakout_v1", "breakout_v2", "meanrev_v1"]:
        for sd in [1.5, 2.0, 2.5, 3.0]:
            for rr in [1.5, 2.0, 3.0]:
                p = dict(strategy=strat_name, period=20, n_std=sd, rr=rr)
                if strat_name == "breakout_v1":
                    for slide in [True, False]:
                        params_grid.append({**p, "slide": slide})
                else:
                    params_grid.append(p)

    # Squeeze has different params
    for sd in [2.0, 2.5]:
        for rr in [2.0, 3.0]:
            params_grid.append(dict(strategy="squeeze", period=20, n_std=sd, rr=rr, lookback=20))

    print(f"Total param combos: {len(params_grid)} x {len(INDICES)} indices", flush=True)
    all_results = []
    start = time.time()

    for sym in INDICES:
        path = f"{DATA_DIR}/{sym}_FIFTEEN_MINUTE.csv"
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        print(f"\n{sym} ({len(df)} bars)...", flush=True)
        for p in params_grid:
            p["_sym"] = sym
            fn = STRATEGIES[p["strategy"]]
            trades = fn(df, p)
            if not trades:
                continue
            tdf = pd.DataFrame(trades)
            n = len(tdf)
            wc = (tdf["net_pnl"] > 0).sum(); lc = n - wc
            wr = round(wc/n*100, 2) if n else 0
            net = round(tdf["net_pnl"].sum(), 2)
            gp = round(tdf[tdf["net_pnl"]>0]["net_pnl"].sum(), 2)
            gl = round(tdf[tdf["net_pnl"]<=0]["net_pnl"].sum(), 2)
            pf = round(abs(gp/gl), 2) if gl else 0
            ar = round(tdf["r"].mean(), 3)
            as_ = round(tdf["r"].std(), 3)
            sh = round(ar/as_*np.sqrt(n), 2) if as_ else 0
            tps = (tdf["reason"]=="TP").sum(); sls = (tdf["reason"]=="SL").sum()
            eods = (tdf["reason"]=="EOD").sum()

            all_results.append({**p, "symbol": sym, "trades": n, "wins": wc,
                "win_rate": wr, "net_pnl": net, "profit_factor": pf, "avg_r": ar,
                "sharpe": sh, "tp": tps, "sl": sls, "eod": eods})

        print(f"  done ({time.time()-start:.0f}s)", flush=True)

    # Save all
    pd.DataFrame(all_results).to_csv(f"{OUTPUT_DIR}/bb_strategy_sweep.csv", index=False)
    print(f"\nSaved {len(all_results)} results", flush=True)

    # Print rankings
    for metric, ascending in [("sharpe", False), ("win_rate", False), ("net_pnl", False)]:
        top = sorted([r for r in all_results if r["trades"]>=30], key=lambda x: x.get(metric,0), reverse=not ascending)[:10]
        print(f"\n=== TOP 10 by {metric.upper()} (min 30 trades) ===", flush=True)
        print(f"{'Strat':12s} {'Sym':10s} {'SD':4s} {'RR':4s} {'Slide':6s} {'Trades':>7s} {'WR':>6s} {'NetP&L':>10s} {'PF':>5s} {'Sh':>5s}", flush=True)
        for r in top:
            slide_str = str(r.get("slide","-"))
            print(f"{r['strategy']:12s} {r['symbol']:10s} {r['n_std']:4.1f} {r['rr']:4.1f} {slide_str:6s} "
                  f"{r['trades']:>7d} {r['win_rate']:>5.1f}% {r['net_pnl']:>+9.0f} {r['profit_factor']:>4.2f} {r['sharpe']:>5.2f}", flush=True)

    print(f"\nTotal time: {time.time()-start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
