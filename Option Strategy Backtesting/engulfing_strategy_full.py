"""Engulfing Pattern Strategy — Full Implementation
   Signal:  Bullish engulfing on 1H chart
   Entry:   5M breakout above signal high → retest close
   Exit:    Chandelier 15×ATR trailing stop
   Filter:  Portfolio-level skip after 2 consecutive losses
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; SLOT = 10; CHG = 20  # NIFTY 50 units/point, SENSEX 10 units/point
CUTOFF_TIME = pd.Timestamp("14:15").time()  # no entry after 2:15 PM
MIN_BODY_PCT = 0.50  # engulfing body must be >= 50% of previous body
CHANDELIER_MULT = 15  # ATR multiplier for trailing stop
SKIP_AFTER_N_LOSSES = 2  # skip trading after N consecutive losses

def compute_atr(df, period=14):
    """Average True Range"""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def detect_signals(h1):
    """
    Step 1: Signal Detection (1-hour chart)
    -----------------------------------------
    Conditions for a bullish engulfing candle:
      a) Previous candle is BEARISH  (close < open)
      b) Current  candle is BULLISH  (close > open)
      c) Current open  ≤ previous close  (wraps below)
      d) Current close ≥ previous open   (wraps above)
      e) Current body ≥ 50% of previous body  (meaningful reversal)
    """
    body = (h1["close"] - h1["open"]).abs()
    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]
    
    signals = []
    for i in range(1, len(h1)):
        if not is_red.iloc[i-1]:    continue  # prev must be bearish
        if not is_green.iloc[i]:     continue  # current must be bullish
        if h1["open"].iloc[i] > h1["close"].iloc[i-1]: continue
        if h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
        if body.iloc[i] < body.iloc[i-1] * MIN_BODY_PCT: continue
        signals.append({
            "trigger_time": h1["datetime"].iloc[i],
            "level": h1["high"].iloc[i]  # signal candle high = breakout level
        })
    return signals

def execute_trades(signals, m5, mult=CHANDELIER_MULT):
    """
    Step 2: Entry Execution (5-minute chart)
    ------------------------------------------
    For each signal:
      a) Wait for price to BREAK OUT above the signal candle's high
      b) Wait for a RETEST (price dips below breakout level, then closes above)
      c) ENTER at the retest close
      d) Initial stop-loss at the retest candle's low
    
    Step 3: Exit (Chandelier 15×ATR trailing stop)
    -------------------------------------------------
      a) Track the highest high since entry
      b) Trail stop = highest_high - 15 × ATR(14)
      c) EXIT when close < trail stop
      d) This wide band (15×ATR) lets winners run for days/weeks
    """
    tc = m5["datetime"].dt.time
    atr5 = compute_atr(m5, 14)
    
    # Convert to Unix timestamps for fast search
    dt_unix = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi_arr = m5["high"].values
    lo_arr = m5["low"].values
    cl_arr = m5["close"].values
    
    trades = []
    for sig in signals:
        t_unix = int(pd.to_datetime(sig["trigger_time"]).timestamp())
        lv = sig["level"]
        
        # Find the first 5M bar after the signal
        idx = np.searchsorted(dt_unix, t_unix, side="right")
        if idx >= len(m5): continue
        
        # 2a) Breakout: close > signal high
        broke = idx
        while broke < len(m5) and cl_arr[broke] <= lv:
            broke += 1
        if broke >= len(m5): continue
        
        # 2b) Retest: price dips below level, then closes above, before 2:15 PM
        retest = broke + 1
        while retest < len(m5):
            if (lo_arr[retest] < lv and cl_arr[retest] > lv
                and tc.iloc[retest] < CUTOFF_TIME):
                break
            retest += 1
        if retest >= len(m5): continue
        
        # 2c) Entry price = retest close, stop = retest low
        entry_price = cl_arr[retest]
        stop_loss = lo_arr[retest]
        if entry_price - stop_loss <= 0: continue
        if m5["datetime"].iloc[retest].hour == 9: continue  # skip 9 AM entries
        
        # 3) Chandelier trailing stop
        highest_since_entry = entry_price
        for j in range(retest + 1, len(m5)):
            ca = atr5.iloc[j]
            if pd.isna(ca): continue
            if hi_arr[j] > highest_since_entry:
                highest_since_entry = hi_arr[j]
            
            trail_stop = highest_since_entry - mult * ca
            
            # Exit when close breaches the trail stop
            if cl_arr[j] < trail_stop:
                trades.append({
                    "points": cl_arr[j] - entry_price,
                    "exit_time": m5["datetime"].iloc[j],
                    "hold_hours": (m5["datetime"].iloc[j]
                                   - m5["datetime"].iloc[retest]).total_seconds() / 3600,
                    "reason": f"CH{mult}"
                })
                break
    
    return pd.DataFrame(trades)

def portfolio_loss_filter(df, skip_n=SKIP_AFTER_N_LOSSES):
    """
    Step 4: Portfolio-Level Loss Streak Filter
    --------------------------------------------
    The strategy's Achilles heel is loss clustering — choppy markets
    produce multiple whipsaws in a row.
    
    After `skip_n` consecutive losing trades (across the combined
    NIFTY50 + SENSEX portfolio), the next trade is SKIPPED.
    The counter resets after any winning trade.
    
    This filter removes ~16% of trades during unfavorable regimes,
    improving net P&L by ~93% vs the unfiltered strategy.
    """
    df = df.sort_values("exit_time").reset_index(drop=True)
    loss_count = 0
    keep = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        if loss_count >= skip_n:
            keep[i] = False        # SKIP this trade
            loss_count = 0          # reset after skip
            continue
        if df["points"].iloc[i] <= 0:
            loss_count += 1
        else:
            loss_count = 0
    
    return df[keep].reset_index(drop=True)


def run_strategy():
    """Run on NIFTY50 and SENSEX, combine, apply filter, report."""
    all_trades = []
    
    for sym in ["NIFTY50", "SENSEX"]:
        print(f"\n=== {sym} ===")
        h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
        m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
        h1["datetime"] = pd.to_datetime(h1["datetime"])
        m5["datetime"] = pd.to_datetime(m5["datetime"])
        h1 = h1.sort_values("datetime").reset_index(drop=True)
        m5 = m5.sort_values("datetime").reset_index(drop=True)
        
        # Step 1: Detect signals on 1H chart
        sigs = detect_signals(h1)
        print(f"  Signals: {len(sigs)}")
        
        # Steps 2-3: Execute on 5M chart
        trades = execute_trades(sigs, m5)
        trades["sym"] = sym
        lot = NLOT if "NIFTY" in sym else SLOT
        trades["pnl_rs"] = trades["points"] * lot - CHG
        print(f"  Trades:  {len(trades)}")
        print(f"  Net Rs:  Rs{trades['pnl_rs'].sum():+,.0f}")
        all_trades.append(trades)
    
    # Step 4: Combine and apply portfolio loss filter
    comb = pd.concat(all_trades, ignore_index=True)
    comb_unfilt = comb.copy()
    comb = portfolio_loss_filter(comb)
    
    print(f"\n{'='*50}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*50}")
    print(f"  Unfiltered: {len(comb_unfilt)} trades, Rs{comb_unfilt['pnl_rs'].sum():+,.0f}")
    print(f"  Filtered:   {len(comb)} trades, Rs{comb['pnl_rs'].sum():+,.0f}")
    print(f"  Skipped:    {len(comb_unfilt)-len(comb)} trades "
          f"(Rs{comb_unfilt['pnl_rs'].sum()-comb['pnl_rs'].sum():+,.0f} removed)")
    
    # Final metrics
    comb = comb.sort_values("exit_time").reset_index(drop=True)
    comb["cum"] = comb["pnl_rs"].cumsum()
    comb["peak"] = comb["cum"].cummax()
    comb["dd_rs"] = comb["peak"] - comb["cum"]
    max_dd = comb["dd_rs"].max()
    max_dd_pct = max_dd / (comb["peak"].max()) * 100 if comb["peak"].max() > 0 else 0
    cagr = ((1 + comb["pnl_rs"].sum() / 200000) ** (1/10) - 1) * 100
    sharpe = comb["pnl_rs"].mean() / comb["pnl_rs"].std() * np.sqrt(len(comb))
    wr = (comb["pnl_rs"] > 0).sum() / len(comb) * 100
    
    print(f"\n{'='*50}")
    print("FINAL METRICS (Rs2L capital, 10 years)")
    print(f"{'='*50}")
    print(f"  Net P&L:       Rs{comb['pnl_rs'].sum():+,.0f}")
    print(f"  Return:        {comb['pnl_rs'].sum()/200000*100:.1f}%")
    print(f"  CAGR:          {cagr:.1f}%")
    print(f"  Trades:        {len(comb)}")
    print(f"  Win Rate:      {wr:.1f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Max DD:        Rs{max_dd:,.0f} ({max_dd_pct:.1f}%)")
    print(f"  Avg Win:       Rs{comb[comb['pnl_rs']>0]['pnl_rs'].mean():,.0f}")
    print(f"  Avg Loss:      Rs{comb[comb['pnl_rs']<0]['pnl_rs'].mean():,.0f}")
    print(f"  Profit Factor: {comb[comb['pnl_rs']>0]['pnl_rs'].sum()/abs(comb[comb['pnl_rs']<0]['pnl_rs'].sum()):.2f}")
    
    return comb

if __name__ == "__main__":
    comb = run_strategy()
