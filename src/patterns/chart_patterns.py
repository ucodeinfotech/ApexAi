"""
Chart Pattern Detection Engine — vectorized for performance.
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# ─── COILING / NARROWING PATTERNS ────────────────────────────────────

def inside_bar(df):
    return (df["high"] <= df["high"].shift()) & (df["low"] >= df["low"].shift())


def nr4(df):
    rng = df["high"] - df["low"]
    return rng == rng.rolling(4, min_periods=4).min()


def nr7(df):
    rng = df["high"] - df["low"]
    return rng == rng.rolling(7, min_periods=7).min()


def volatility_contraction(df, lookback=10, threshold=0.5):
    rng = df["high"] - df["low"]
    avg_rng = rng.rolling(lookback, min_periods=lookback).mean()
    return (rng <= avg_rng * threshold) & (avg_rng > 0)


def darvas_box(df, box_period=10, vol_lookback=5):
    high_max = df["high"].rolling(box_period, min_periods=box_period).max()
    low_min = df["low"].rolling(box_period, min_periods=box_period).min()
    box_width = (high_max - low_min) / low_min * 100
    in_box = box_width < 5
    avg_vol = df["volume"].rolling(vol_lookback, min_periods=vol_lookback).mean()
    vol_surge = df["volume"] > avg_vol * 1.5
    return (
        (df["close"] > high_max.shift()) & in_box.shift() & vol_surge,
        (df["close"] < low_min.shift()) & in_box.shift() & vol_surge,
    )


# ─── SWING-BASED REVERSAL PATTERNS ───────────────────────────────────

def _swings(high, low, window=5):
    """Return arrays of swing high/low indices and values."""
    high_arr = high.values
    low_arr = low.values
    hi = argrelextrema(high_arr, np.greater, order=window)[0]
    lo = argrelextrema(low_arr, np.less, order=window)[0]
    return hi, high_arr[hi], lo, low_arr[lo]


def double_top(df, window=5, tolerance_pct=2):
    result = pd.Series(False, index=df.index)
    hi_idx, hi_vals, _, _ = _swings(df["high"], df["low"], window)
    if len(hi_idx) < 2:
        return result
    for i in range(1, len(hi_idx)):
        diff = abs(hi_vals[i] - hi_vals[i - 1]) / hi_vals[i - 1] * 100
        if diff <= tolerance_pct:
            valley = df["low"].iloc[hi_idx[i - 1]:hi_idx[i]].min()
            drop = ((hi_vals[i - 1] + hi_vals[i]) / 2 - valley) / valley * 100
            if drop >= 3:
                result.iloc[hi_idx[i]] = True
    return result


def double_bottom(df, window=5, tolerance_pct=2):
    result = pd.Series(False, index=df.index)
    _, _, lo_idx, lo_vals = _swings(df["high"], df["low"], window)
    if len(lo_idx) < 2:
        return result
    for i in range(1, len(lo_idx)):
        diff = abs(lo_vals[i] - lo_vals[i - 1]) / lo_vals[i - 1] * 100
        if diff <= tolerance_pct:
            peak = df["high"].iloc[lo_idx[i - 1]:lo_idx[i]].max()
            rise = (peak - (lo_vals[i - 1] + lo_vals[i]) / 2) / (lo_vals[i - 1] + lo_vals[i]) * 200
            if rise >= 3:
                result.iloc[lo_idx[i]] = True
    return result


def head_and_shoulders(df, window=5):
    result = pd.Series(False, index=df.index)
    hi_idx, hi_vals, _, _ = _swings(df["high"], df["low"], window)
    if len(hi_idx) < 3:
        return result
    for i in range(1, len(hi_idx) - 1):
        if hi_vals[i] > hi_vals[i - 1] and hi_vals[i] > hi_vals[i + 1]:
            shoulder_diff = abs(hi_vals[i - 1] - hi_vals[i + 1]) / max(hi_vals[i - 1], hi_vals[i + 1]) * 100
            if shoulder_diff <= 10:
                valley1 = df["low"].iloc[hi_idx[i - 1]:hi_idx[i]].min()
                valley2 = df["low"].iloc[hi_idx[i]:hi_idx[i + 1]].min()
                head_to_neck = (hi_vals[i] - (valley1 + valley2) / 2) / ((valley1 + valley2) / 2) * 100
                if head_to_neck >= 5:
                    result.iloc[hi_idx[i + 1]] = True
    return result


# ─── CONTINUATION PATTERNS ───────────────────────────────────────────

def flag_pattern(df, lookback=20, pullback_pct=15):
    close = df["close"]
    move = close.pct_change(periods=lookback) * 100
    recent_move = close.pct_change(periods=lookback // 2) * 100
    return (move.abs() >= 10) & (recent_move.abs() <= pullback_pct)


def pennant(df, lookback=20):
    half = lookback // 2
    rng_first = df["high"].rolling(half).max() - df["low"].rolling(half).min()
    rng_last = df["high"].rolling(half).max() - df["low"].rolling(half).min()
    cond = (rng_last < rng_first.shift(half) * 0.7) & (rng_first.shift(half) > 0)
    return cond.fillna(False)


def channel(df, lookback=30, tolerance_pct=30):
    high_range = df["high"].rolling(lookback).max() - df["high"].rolling(lookback).min()
    low_range = df["low"].rolling(lookback).max() - df["low"].rolling(lookback).min()
    max_range = np.maximum(high_range, low_range)
    ratio = np.where(max_range > 0, abs(high_range - low_range) / max_range * 100, 0)
    return pd.Series(ratio <= tolerance_pct, index=df.index)


# ─── REGISTRY ────────────────────────────────────────────────────────

CHART_PATTERNS = {
    "inside_bar": inside_bar,
    "nr4": nr4,
    "nr7": nr7,
    "volatility_contraction": lambda df: volatility_contraction(df),
    "darvas_box_up": lambda df: darvas_box(df)[0],
    "darvas_box_down": lambda df: darvas_box(df)[1],
    "double_top": double_top,
    "double_bottom": double_bottom,
    "head_and_shoulders": head_and_shoulders,
    "flag": flag_pattern,
    "pennant": pennant,
    "channel": channel,
}


def detect_chart_patterns(df):
    results = {}
    for name, func in CHART_PATTERNS.items():
        try:
            mask = func(df)
            results[name] = mask.astype(int)
        except Exception:
            pass
    return pd.DataFrame(results, index=df.index)
