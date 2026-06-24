"""
Market Structure Analysis — Smart Money, Wyckoff, Market Profile, Volume Profile, Relative Strength.
All functions return a DataFrame with additional columns added.
"""
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — SMART MONEY CONCEPTS
# ═══════════════════════════════════════════════════════════════════════

def add_fair_value_gaps(df):
    """Fair Value Gap: gap between consecutive candles where the market didn't trade."""
    # FVG occurs when: prev high < current low (bullish) or prev low > current high (bearish)
    df["fvg_bullish"] = (df["high"].shift(1) < df["low"]) & (df["volume"] > 0)
    df["fvg_bearish"] = (df["low"].shift(1) > df["high"]) & (df["volume"] > 0)
    # Gap size
    df["fvg_size"] = np.where(df["fvg_bullish"], df["low"] - df["high"].shift(1),
                               np.where(df["fvg_bearish"], df["low"].shift(1) - df["high"], 0))
    return df


def add_order_blocks(df, lookback=5):
    """
    Order Block: last bearish candle before a bullish breakout (bullish OB)
    or last bullish candle before a bearish breakout (bearish OB).
    """
    ob_bullish = (
        (df["close"].shift(lookback) < df["open"].shift(lookback)) &  # bearish candle (potential OB)
        (df["close"] > df["high"].shift(lookback))  # breakout above its high
    )
    ob_bearish = (
        (df["close"].shift(lookback) > df["open"].shift(lookback)) &  # bullish candle (potential OB)
        (df["close"] < df["low"].shift(lookback))  # breakout below its low
    )
    df["ob_bullish"] = ob_bullish
    df["ob_bearish"] = ob_bearish
    return df


def add_liquidity_sweeps(df, lookback=10):
    """
    Liquidity Sweep: price briefly moves beyond a recent swing high/low
    then reverses back inside.
    """
    recent_high = df["high"].rolling(lookback, min_periods=lookback).max().shift()
    recent_low = df["low"].rolling(lookback, min_periods=lookback).min().shift()

    # Sweep high and close back below it
    sweep_high = (df["high"] > recent_high) & (df["close"] < recent_high)
    sweep_low = (df["low"] < recent_low) & (df["close"] > recent_low)

    df["liq_sweep_high"] = sweep_high & (recent_high > 0)
    df["liq_sweep_low"] = sweep_low & (recent_low > 0)
    return df


def add_break_of_structure(df, lookback=10):
    """
    Break of Structure: price breaks above a prior swing high (uptrend BOS)
    or below a prior swing low (downtrend BOS).
    """
    prev_high = df["high"].rolling(lookback, min_periods=lookback).max().shift()
    prev_low = df["low"].rolling(lookback, min_periods=lookback).min().shift()

    bos_up = df["close"] > prev_high
    bos_down = df["close"] < prev_low

    df["bos_up"] = bos_up & (prev_high > 0)
    df["bos_down"] = bos_down & (prev_low > 0)
    return df


def add_change_of_character(df, lookback=5):
    """
    Change of Character: after a sustained move, price breaks the last
    swing point in the opposite direction.
    """
    # Look for series of higher highs/higher lows, then a lower low
    higher_low = df["low"] > df["low"].shift(1)
    lower_high = df["high"] < df["high"].shift(1)

    # After 5+ higher lows, a break of the most recent low
    streak_hl = higher_low.rolling(lookback).sum() >= lookback - 1
    choch_sell = streak_hl.shift() & (df["low"] < df["low"].shift(1))

    # After 5+ lower highs, a break of the most recent high
    streak_lh = lower_high.rolling(lookback).sum() >= lookback - 1
    choch_buy = streak_lh.shift() & (df["high"] > df["high"].shift(1))

    df["choch_sell"] = choch_sell
    df["choch_buy"] = choch_buy
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — WYCKOFF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def add_wyckoff_phases(df, lookback=30):
    """
    Wyckoff phase identification:
    - Accumulation: tight range, high volume at lows
    - Markup: rising prices, increasing volume
    - Distribution: tight range, high volume at highs
    - Markdown: falling prices, increasing volume

    Returns phase labels: -1=markdown, 0=accumulation, 1=markup, 2=distribution
    """
    close = df["close"]
    volume = df["volume"]
    avg_vol = volume.rolling(lookback, min_periods=lookback).mean()

    # Accumulation: price in lower 30% of range, volume above average
    low_pct = (close - close.rolling(lookback).min()) / (
        close.rolling(lookback).max() - close.rolling(lookback).min() + 1e-10
    )
    accumulation = (low_pct < 0.3) & (volume > avg_vol * 1.2)

    # Distribution: price in upper 30% of range, volume above average
    distribution = (low_pct > 0.7) & (volume > avg_vol * 1.2)

    # Markup: price above 50-day MA, rising
    ma50 = close.rolling(50, min_periods=20).mean()
    markup = (close > ma50) & (close > close.shift(5)) & (volume > avg_vol)

    # Markdown: price below 50-day MA, falling
    markdown = (close < ma50) & (close < close.shift(5)) & (volume > avg_vol)

    # Combine into single label
    phase = pd.Series(np.where(accumulation, 0, np.where(markup, 1,
                           np.where(distribution, 2, np.where(markdown, -1, np.nan)))),
                      index=df.index)
    df["wyckoff_phase"] = phase
    return df


def add_wyckoff_spring(df, lookback=10):
    """
    Spring (Wycokoff): price briefly breaks below support (accumulation range),
    then quickly reverses back inside. Often signals the start of a markup.
    """
    low_min = df["low"].rolling(lookback, min_periods=lookback).min().shift()
    spring = (df["low"] < low_min) & (df["close"] > low_min)
    df["wyckoff_spring"] = spring & (low_min > 0)
    return df


def add_wyckoff_upthrust(df, lookback=10):
    """
    Upthrust: price briefly breaks above resistance (distribution range),
    then quickly reverses back inside. Often signals the start of a markdown.
    """
    high_max = df["high"].rolling(lookback, min_periods=lookback).max().shift()
    upthrust = (df["high"] > high_max) & (df["close"] < high_max)
    df["wyckoff_upthrust"] = upthrust & (high_max > 0)
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 — MARKET PROFILE
# ═══════════════════════════════════════════════════════════════════════

def add_market_profile(df, value_area_pct=0.7):
    """
    Market Profile analysis for intraday (15min) data.
    Calculates Point of Control, Value Area High/Low, and Initial Balance.

    Note: For daily data, these become rolling estimates.
    """
    # Point of Control: price level with the highest volume (rolling estimate)
    # Simplified: use VWAP-like calculation
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(20, min_periods=5).sum() / df["volume"].rolling(20, min_periods=5).sum()
    df["mkt_point_of_control"] = vwap

    # Value Area: rolling standard deviation bands around VWAP
    std = tp.rolling(20, min_periods=5).std()
    df["mkt_value_area_high"] = vwap + std
    df["mkt_value_area_low"] = vwap - std

    # Initial Balance: first hour range (first 4 bars of 15min data)
    # For daily data, use previous day's range
    df["mkt_initial_balance_high"] = df["high"].rolling(4, min_periods=4).max()
    df["mkt_initial_balance_low"] = df["low"].rolling(4, min_periods=4).min()

    # Where price is relative to value area
    df["mkt_in_value_area"] = (df["close"] >= df["mkt_value_area_low"]) & (df["close"] <= df["mkt_value_area_high"])
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8 — VOLUME PROFILE
# ═══════════════════════════════════════════════════════════════════════

def add_volume_profile(df, lookback=20, num_zones=10):
    """
    Volume Profile analysis — identifies high and low volume nodes.
    Uses rolling window to estimate price levels with heavy/light volume.
    """
    # High Volume Node: prices where volume was significantly above average
    avg_vol = df["volume"].rolling(lookback, min_periods=lookback).mean()
    high_vol = df["volume"] > avg_vol * 1.5
    df["vol_profile_high_vol_node"] = high_vol

    # Low Volume Node: prices where volume was significantly below average
    low_vol = df["volume"] < avg_vol * 0.5
    df["vol_profile_low_vol_node"] = low_vol

    # Volume-weighted average price (VWAP)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(lookback, min_periods=5).sum() / df["volume"].rolling(lookback, min_periods=5).sum()
    df["vol_profile_vwap"] = vwap

    # Volume Point of Control (VPOC): price level with maximum volume
    # Simplified: VWAP is used as approximation for VPOC
    df["vol_profile_vpoc"] = vwap

    # Standard deviation bands for high/low vol zones
    vol_std = df["volume"].rolling(lookback, min_periods=lookback).std()
    df["vol_profile_high_vol_zone"] = df["volume"] > (avg_vol + vol_std)
    df["vol_profile_low_vol_zone"] = df["volume"] < (avg_vol - vol_std * 0.5)
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 9 — RELATIVE STRENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def add_relative_strength(df, market_close=None, sector_close=None, peers=None):
    close = df["close"]
    daily_ret = close.pct_change()
    dt_series = df["datetime"] if "datetime" in df.columns else df.index

    def _align(series):
        idx = pd.to_datetime(dt_series)
        # Only reindex if index type is different
        if not isinstance(series.index, type(idx)):
            return series.reindex(idx, method="ffill")
        return series

    if market_close is not None and len(market_close) > 0:
        aligned = _align(market_close)
        market_ret = aligned.pct_change()
        df["rs_vs_market"] = daily_ret - market_ret
        df["rs_ratio_market"] = close / aligned
    else:
        df["rs_vs_market"] = np.nan
        df["rs_ratio_market"] = np.nan

    if sector_close is not None and len(sector_close) > 0:
        aligned = _align(sector_close)
        sector_ret = aligned.pct_change()
        df["rs_vs_sector"] = daily_ret - sector_ret
        df["rs_ratio_sector"] = close / aligned
    else:
        df["rs_vs_sector"] = np.nan
        df["rs_ratio_sector"] = np.nan

    df["rs_momentum_10"] = df["rs_ratio_market"].pct_change(10) * 100
    df["rs_momentum_20"] = df["rs_ratio_market"].pct_change(20) * 100
    df["rs_peer_rank"] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

ALL_STRUCTURE_FUNCTIONS = [
    add_fair_value_gaps,
    add_order_blocks,
    add_liquidity_sweeps,
    add_break_of_structure,
    add_change_of_character,
    add_wyckoff_phases,
    add_wyckoff_spring,
    add_wyckoff_upthrust,
    add_market_profile,
    add_volume_profile,
    add_relative_strength,
]

STRUCTURE_COLUMNS = [
    "fvg_bullish", "fvg_bearish", "fvg_size",
    "ob_bullish", "ob_bearish",
    "liq_sweep_high", "liq_sweep_low",
    "bos_up", "bos_down",
    "choch_sell", "choch_buy",
    "wyckoff_phase", "wyckoff_spring", "wyckoff_upthrust",
    "mkt_point_of_control", "mkt_value_area_high", "mkt_value_area_low",
    "mkt_initial_balance_high", "mkt_initial_balance_low", "mkt_in_value_area",
    "vol_profile_high_vol_node", "vol_profile_low_vol_node",
    "vol_profile_vwap", "vol_profile_vpoc",
    "vol_profile_high_vol_zone", "vol_profile_low_vol_zone",
    "rs_vs_market", "rs_vs_sector", "rs_ratio_market", "rs_ratio_sector",
    "rs_momentum_10", "rs_momentum_20", "rs_peer_rank",
]


def compute_all_structure(df, market_close=None, sector_close=None):
    """Apply all market structure functions."""
    df = df.copy()
    for func in ALL_STRUCTURE_FUNCTIONS:
        try:
            if func == add_relative_strength:
                df = func(df, market_close, sector_close)
            else:
                df = func(df)
        except Exception:
            pass
    return df
