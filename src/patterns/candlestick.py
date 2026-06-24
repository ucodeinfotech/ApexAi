"""
Candlestick Pattern Engine — detects single, two, and three-candle patterns
and computes post-pattern statistics.
"""
import pandas as pd
import numpy as np


def _body(open_, close):
    return (close - open_).abs()


def _upper_wick(high, open_, close):
    return high - np.maximum(open_, close)


def _lower_wick(low, open_, close):
    return np.minimum(open_, close) - low


def _body_top(open_, close):
    return np.maximum(open_, close)


def _body_bot(open_, close):
    return np.minimum(open_, close)


def _is_bull(open_, close):
    return close >= open_


# ─── SINGLE-CANDLE PATTERNS ──────────────────────────────────────────

def hammer(df):
    """Small body at upper end, long lower wick (>= 2x body)."""
    body = _body(df["open"], df["close"])
    lw = _lower_wick(df["low"], df["open"], df["close"])
    uw = _upper_wick(df["high"], df["open"], df["close"])
    return (lw >= body * 2) & (uw <= body * 0.3)


def doji(df):
    """Very small body — open and close are almost equal."""
    body = _body(df["open"], df["close"])
    full_range = df["high"] - df["low"]
    return (body <= full_range * 0.1) & (full_range > 0)


def marubozu(df):
    """Long body with little to no wicks."""
    body = _body(df["open"], df["close"])
    uw = _upper_wick(df["high"], df["open"], df["close"])
    lw = _lower_wick(df["low"], df["open"], df["close"])
    return (uw <= body * 0.05) & (lw <= body * 0.05) & (body > 0)


def spinning_top(df):
    """Small body with upper and lower wicks of similar length."""
    body = _body(df["open"], df["close"])
    uw = _upper_wick(df["high"], df["open"], df["close"])
    lw = _lower_wick(df["low"], df["open"], df["close"])
    full_range = df["high"] - df["low"]
    return (body <= full_range * 0.3) & (uw >= body * 0.5) & (lw >= body * 0.5) & (full_range > 0)


def shooting_star(df):
    """Small body at lower end, long upper wick — appears after uptrend."""
    body = _body(df["open"], df["close"])
    uw = _upper_wick(df["high"], df["open"], df["close"])
    lw = _lower_wick(df["low"], df["open"], df["close"])
    return (uw >= body * 2) & (lw <= body * 0.3) & (body > 0)


def hanging_man(df):
    """Same shape as hammer but appears after uptrend (context-dependent)."""
    body = _body(df["open"], df["close"])
    lw = _lower_wick(df["low"], df["open"], df["close"])
    uw = _upper_wick(df["high"], df["open"], df["close"])
    return (lw >= body * 2) & (uw <= body * 0.3) & (body > 0)


# ─── TWO-CANDLE PATTERNS ─────────────────────────────────────────────

def bullish_engulfing(df):
    """Bull candle fully engulfs prior bear candle."""
    prev_bull = ~_is_bull(df["open"].shift(), df["close"].shift())
    curr_bull = _is_bull(df["open"], df["close"])
    prev_open, prev_close = df["open"].shift(), df["close"].shift()
    return prev_bull & curr_bull & (df["close"] >= prev_open) & (df["open"] <= prev_close)


def bearish_engulfing(df):
    """Bear candle fully engulfs prior bull candle."""
    prev_bull = _is_bull(df["open"].shift(), df["close"].shift())
    curr_bear = ~_is_bull(df["open"], df["close"])
    prev_open, prev_close = df["open"].shift(), df["close"].shift()
    return prev_bull & curr_bear & (df["open"] >= prev_close) & (df["close"] <= prev_open)


def bullish_harami(df):
    """Small bull candle forms inside prior tall bear candle."""
    prev_bear = ~_is_bull(df["open"].shift(), df["close"].shift())
    curr_bull = _is_bull(df["open"], df["close"])
    prev_body_top = _body_top(df["open"].shift(), df["close"].shift())
    prev_body_bot = _body_bot(df["open"].shift(), df["close"].shift())
    return prev_bear & curr_bull & (df["close"] <= prev_body_top) & (df["open"] >= prev_body_bot)


def bearish_harami(df):
    """Small bear candle forms inside prior tall bull candle."""
    prev_bull = _is_bull(df["open"].shift(), df["close"].shift())
    curr_bear = ~_is_bull(df["open"], df["close"])
    prev_body_top = _body_top(df["open"].shift(), df["close"].shift())
    prev_body_bot = _body_bot(df["open"].shift(), df["close"].shift())
    return prev_bull & curr_bear & (df["open"] <= prev_body_top) & (df["close"] >= prev_body_bot)


def piercing_line(df):
    """Bear candle followed by bull that closes above mid-point of prior candle."""
    prev_bear = ~_is_bull(df["open"].shift(), df["close"].shift())
    curr_bull = _is_bull(df["open"], df["close"])
    mid_prev = (df["open"].shift() + df["close"].shift()) / 2
    return prev_bear & curr_bull & (df["close"] > mid_prev) & (df["open"].shift() > df["close"])


def dark_cloud_cover(df):
    """Bull candle followed by bear that closes below mid-point of prior candle."""
    prev_bull = _is_bull(df["open"].shift(), df["close"].shift())
    curr_bear = ~_is_bull(df["open"], df["close"])
    mid_prev = (df["open"].shift() + df["close"].shift()) / 2
    return prev_bull & curr_bear & (df["close"] < mid_prev) & (df["open"].shift() < df["close"])


# ─── THREE-CANDLE PATTERNS ──────────────────────────────────────────

def morning_star(df):
    """Bear, doji/spinning-top, bull — with gaps on both sides."""
    prev_bear = ~_is_bull(df["open"].shift(2), df["close"].shift(2))
    mid_small = (_body(df["open"].shift(), df["close"].shift()) <=
                 (df["high"].shift() - df["low"].shift()) * 0.3)
    curr_bull = _is_bull(df["open"], df["close"])
    gap_down = df["close"].shift(2) < df["open"].shift()
    gap_up = df["close"] > df["open"].shift()
    return prev_bear & mid_small & curr_bull & gap_down & gap_up


def evening_star(df):
    """Bull, doji/spinning-top, bear — with gaps on both sides."""
    prev_bull = _is_bull(df["open"].shift(2), df["close"].shift(2))
    mid_small = (_body(df["open"].shift(), df["close"].shift()) <=
                 (df["high"].shift() - df["low"].shift()) * 0.3)
    curr_bear = ~_is_bull(df["open"], df["close"])
    gap_up = df["close"].shift(2) < df["open"].shift()
    gap_down = df["close"] < df["open"].shift()
    return prev_bull & mid_small & curr_bear & gap_up & gap_down


def three_white_soldiers(df):
    """Three consecutive long-bodied bullish candles, each closing higher."""
    for i in range(3):
        if i > 0:
            cond = _is_bull(df["open"].shift(i), df["close"].shift(i))
            if not cond.any():
                return cond
    c1_bull = _is_bull(df["open"].shift(2), df["close"].shift(2))
    c2_bull = _is_bull(df["open"].shift(), df["close"].shift())
    c3_bull = _is_bull(df["open"], df["close"])
    c2_higher = df["close"].shift() > df["close"].shift(2)
    c3_higher = df["close"] > df["close"].shift()
    bodies_1 = _body(df["open"].shift(2), df["close"].shift(2)) > 0
    bodies_2 = _body(df["open"].shift(), df["close"].shift()) > 0
    bodies_3 = _body(df["open"], df["close"]) > 0
    return c1_bull & c2_bull & c3_bull & c2_higher & c3_higher & bodies_1 & bodies_2 & bodies_3


def three_black_crows(df):
    """Three consecutive long-bodied bearish candles, each closing lower."""
    c1_bear = ~_is_bull(df["open"].shift(2), df["close"].shift(2))
    c2_bear = ~_is_bull(df["open"].shift(), df["close"].shift())
    c3_bear = ~_is_bull(df["open"], df["close"])
    c2_lower = df["close"].shift() < df["close"].shift(2)
    c3_lower = df["close"] < df["close"].shift()
    bodies_1 = _body(df["open"].shift(2), df["close"].shift(2)) > 0
    bodies_2 = _body(df["open"].shift(), df["close"].shift()) > 0
    bodies_3 = _body(df["open"], df["close"]) > 0
    return c1_bear & c2_bear & c3_bear & c2_lower & c3_lower & bodies_1 & bodies_2 & bodies_3


# ─── PATTERN REGISTRY ────────────────────────────────────────────────

SINGLE_CANDLE_PATTERNS = {
    "hammer": hammer,
    "doji": doji,
    "marubozu": marubozu,
    "spinning_top": spinning_top,
    "shooting_star": shooting_star,
    "hanging_man": hanging_man,
}

TWO_CANDLE_PATTERNS = {
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "bullish_harami": bullish_harami,
    "bearish_harami": bearish_harami,
    "piercing_line": piercing_line,
    "dark_cloud_cover": dark_cloud_cover,
}

THREE_CANDLE_PATTERNS = {
    "morning_star": morning_star,
    "evening_star": evening_star,
    "three_white_soldiers": three_white_soldiers,
    "three_black_crows": three_black_crows,
}

ALL_PATTERNS = {}
ALL_PATTERNS.update(SINGLE_CANDLE_PATTERNS)
ALL_PATTERNS.update(TWO_CANDLE_PATTERNS)
ALL_PATTERNS.update(THREE_CANDLE_PATTERNS)


# ─── DETECTION ENGINE ────────────────────────────────────────────────

def detect_patterns(df):
    """Detect all candlestick patterns in a DataFrame. Returns mask DataFrame."""
    results = {}
    for name, func in ALL_PATTERNS.items():
        try:
            mask = func(df)
            results[name] = mask.astype(int)
        except Exception:
            results[name] = 0
    return pd.DataFrame(results, index=df.index)


def compute_pattern_stats(df, pattern_masks, forward_periods=[3, 5, 10]):
    """For each pattern, compute occurrence stats and forward returns."""
    stats = {}
    for name in pattern_masks.columns:
        mask = pattern_masks[name].astype(bool)
        count = mask.sum()
        if count == 0:
            continue

        forward_returns = {}
        for fp in forward_periods:
            fwd_ret = df["close"].pct_change(periods=fp).shift(-fp) * 100
            hit_returns = fwd_ret[mask].dropna()
            if len(hit_returns) == 0:
                continue
            forward_returns[fp] = {
                "count": len(hit_returns),
                "avg_return": hit_returns.mean(),
                "median_return": hit_returns.median(),
                "win_rate": (hit_returns > 0).mean() * 100,
                "std_return": hit_returns.std(),
                "best": hit_returns.max(),
                "worst": hit_returns.min(),
            }

        stats[name] = {
            "total_occurrences": count,
            "frequency_pct": count / len(df) * 100,
            "forward_returns": forward_returns,
        }

    return stats


def run_candlestick_analysis(con, symbol, timeframe, forward_periods=[3, 5, 10]):
    """Full analysis: detect patterns + compute stats for one symbol/timeframe."""
    df = con.execute(
        "SELECT datetime, open, high, low, close, volume FROM raw_market "
        "WHERE symbol=? AND timeframe=? ORDER BY datetime",
        [symbol, timeframe]
    ).fetchdf()

    if len(df) < 100:
        return None

    patterns = detect_patterns(df)
    stats = compute_pattern_stats(df, patterns, forward_periods)
    return {"symbol": symbol, "timeframe": timeframe, "stats": stats, "patterns": patterns}
