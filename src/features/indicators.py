"""
Technical indicator functions — all operate on pandas DataFrames with OHLCV columns.
Every function receives and returns a DataFrame for composability.
"""
import pandas as pd
import numpy as np


def _wma(series, length):
    """Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _true_range(high, low, close):
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift()).abs(),
        "lc": (low - close.shift()).abs(),
    })
    return tr.max(axis=1)


# ─── TREND ───────────────────────────────────────────────────────────

def add_sma(df, periods=[5, 10, 20, 50, 200]):
    for p in periods:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()
    return df


def add_ema(df, periods=[5, 10, 20, 50, 200]):
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def add_wma(df, periods=[10, 20]):
    for p in periods:
        df[f"wma_{p}"] = _wma(df["close"], p)
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    return df


def add_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = _true_range(high, low, close)

    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(span=period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_aroon(df, period=25):
    high_idx = df["high"].rolling(period).apply(np.argmax) / period * 100
    low_idx = df["low"].rolling(period).apply(np.argmin) / period * 100
    df["aroon_up"] = high_idx
    df["aroon_down"] = low_idx
    df["aroon_osc"] = high_idx - low_idx
    return df


def add_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = (tp - sma) / (0.015 * mad + 1e-10)
    return df


def add_parabolic_sar(df, acceleration=0.02, max_accel=0.2):
    high, low = df["high"].values, df["low"].values
    length = len(df)
    sar = np.empty(length)
    trend = np.empty(length, dtype=int)
    ep = np.empty(length)
    af = np.empty(length)

    sar[0] = low[0]
    trend[0] = 1
    ep[0] = high[0]
    af[0] = acceleration

    for i in range(1, length):
        af[i] = af[i - 1]
        if trend[i - 1] == 1:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            if sar[i] > low[i]:
                sar[i] = ep[i - 1]
                trend[i] = -1
                ep[i] = low[i]
                af[i] = acceleration
            else:
                trend[i] = 1
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + acceleration, max_accel)
                else:
                    ep[i] = ep[i - 1]
        else:
            sar[i] = sar[i - 1] - af[i - 1] * (sar[i - 1] - ep[i - 1])
            if sar[i] < high[i]:
                sar[i] = ep[i - 1]
                trend[i] = 1
                ep[i] = high[i]
                af[i] = acceleration
            else:
                trend[i] = -1
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + acceleration, max_accel)
                else:
                    ep[i] = ep[i - 1]
    df["psar"] = sar
    return df


# ─── MOMENTUM ────────────────────────────────────────────────────────

def add_rsi(df, periods=[7, 14, 21]):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for p in periods:
        avg_gain = gain.ewm(span=p, adjust=False).mean()
        avg_loss = loss.ewm(span=p, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    return df


def add_stoch(df, k_period=5, d_period=3):
    low_k = df["low"].rolling(k_period).min()
    high_k = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_k) / (high_k - low_k + 1e-10)
    df["stoch_k"] = k
    df["stoch_d"] = k.rolling(d_period).mean()
    return df


def add_williams_r(df, period=14):
    high_p = df["high"].rolling(period).max()
    low_p = df["low"].rolling(period).min()
    df["williams_r"] = -100 * (high_p - df["close"]) / (high_p - low_p + 1e-10)
    return df


def add_roc(df, periods=[5, 10, 20]):
    for p in periods:
        df[f"roc_{p}"] = df["close"].pct_change(periods=p) * 100
    return df


def add_trix(df, period=15):
    ema1 = df["close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    df["trix"] = ema3.pct_change() * 100
    return df


def add_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    flow = rmf.where(tp > tp.shift(), -rmf)
    pos = flow.where(flow > 0, 0).rolling(period).sum()
    neg = (-flow.where(flow < 0, 0)).rolling(period).sum()
    mfr = pos / (neg + 1e-10)
    df["mfi"] = 100 - (100 / (1 + mfr))
    return df


def add_ultimate_osc(df, p1=7, p2=14, p3=28):
    high, low, close = df["high"], df["low"], df["close"]
    bp = close - np.minimum(low, close.shift())
    tr = _true_range(high, low, close)
    avg1 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + 1e-10)
    avg2 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + 1e-10)
    avg3 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + 1e-10)
    df["uo"] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return df


# ─── VOLATILITY ──────────────────────────────────────────────────────

def add_atr(df, periods=[7, 14, 21]):
    tr = _true_range(df["high"], df["low"], df["close"])
    for p in periods:
        df[f"atr_{p}"] = tr.ewm(span=p, adjust=False).mean()
    return df


def add_bollinger(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + std_dev * std
    df["bb_middle"] = sma
    df["bb_lower"] = sma - std_dev * std
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma * 100
    return df


def add_keltner(df, period=20, atr_mult=1.5):
    ema = df["close"].ewm(span=period, adjust=False).mean()
    atr = _true_range(df["high"], df["low"], df["close"]).ewm(span=period, adjust=False).mean()
    df["kc_upper"] = ema + atr_mult * atr
    df["kc_lower"] = ema - atr_mult * atr
    df["kc_width"] = (df["kc_upper"] - df["kc_lower"]) / ema * 100
    return df


def add_donchian(df, period=20):
    df["dc_upper"] = df["high"].rolling(period).max()
    df["dc_lower"] = df["low"].rolling(period).min()
    df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2
    df["dc_width"] = (df["dc_upper"] - df["dc_lower"]) / df["dc_mid"] * 100
    return df


def add_historical_vol(df, periods=[10, 20, 30]):
    log_ret = np.log(df["close"] / df["close"].shift())
    for p in periods:
        df[f"hv_{p}"] = log_ret.rolling(p).std() * np.sqrt(p) * 100
    return df


# ─── VOLUME ──────────────────────────────────────────────────────────

def add_obv(df):
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


def add_volume_ratio(df, periods=[5, 10, 20]):
    for p in periods:
        df[f"vol_ratio_{p}"] = df["volume"] / (df["volume"].rolling(p).mean() + 1e-10)
    return df


def add_cmf(df, period=20):
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10) * df["volume"]
    df["cmf"] = mfv.rolling(period).sum() / (df["volume"].rolling(period).sum() + 1e-10)
    return df


def add_vpt(df):
    df["vpt"] = (df["volume"] * df["close"].pct_change()).fillna(0).cumsum()
    return df


def add_ad_line(df):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10)
    ad = (mfm * df["volume"]).cumsum()
    df["ad_line"] = ad
    return df


def add_eom(df, period=14):
    distance = (df["high"] + df["low"]) / 2 - (df["high"].shift() + df["low"].shift()) / 2
    box_ratio = (df["volume"] / 1e6) / (df["high"] - df["low"] + 1e-10)
    df["eom"] = (distance / box_ratio).rolling(period).mean()
    return df


def add_force_index(df, period=13):
    fi = df["close"].diff() * df["volume"]
    df["fi"] = fi.ewm(span=period, adjust=False).mean()
    return df


# ─── MARKET STRUCTURE ────────────────────────────────────────────────

def add_returns(df, periods=[1, 5, 10, 20]):
    for p in periods:
        df[f"ret_{p}d"] = df["close"].pct_change(periods=p) * 100
    return df


def add_log_returns(df, periods=[1, 5, 10, 20]):
    for p in periods:
        df[f"log_ret_{p}d"] = np.log(df["close"] / df["close"].shift(p)) * 100
    return df


def add_price_vs_ma(df, periods=[10, 20, 50, 200]):
    for p in periods:
        ma = df["close"].rolling(p).mean()
        df[f"close_vs_sma_{p}"] = (df["close"] / ma - 1) * 100
    return df


def add_pivot_points(df):
    df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    df["r1"] = 2 * df["pivot"] - df["low"]
    df["s1"] = 2 * df["pivot"] - df["high"]
    df["r2"] = df["pivot"] + (df["high"] - df["low"])
    df["s2"] = df["pivot"] - (df["high"] - df["low"])
    return df


def add_swing_detection(df, window=5):
    high, low = df["high"], df["low"]
    df["swing_high"] = high == high.rolling(window * 2 + 1, center=True).max()
    df["swing_low"] = low == low.rolling(window * 2 + 1, center=True).min()
    return df


def add_range_stats(df, periods=[5, 10, 20]):
    for p in periods:
        df[f"range_{p}"] = (df["high"].rolling(p).max() - df["low"].rolling(p).min()) / df["close"] * 100
        df[f"body_ratio_{p}"] = (df["close"] - df["open"]).abs().rolling(p).mean() / (df["high"] - df["low"]).rolling(p).mean()
    return df


# ─── STATISTICAL ─────────────────────────────────────────────────────

def add_zscore(df, period=20):
    df[f"zscore_{period}"] = (df["close"] - df["close"].rolling(period).mean()) / (df["close"].rolling(period).std() + 1e-10)
    return df


def add_rolling_skew(df, period=20):
    df[f"skew_{period}"] = df["close"].rolling(period).skew()
    return df


def add_rolling_kurtosis(df, period=20):
    df[f"kurt_{period}"] = df["close"].rolling(period).kurt()
    return df


def add_serial_corr(df, period=20):
    log_ret = np.log(df["close"] / df["close"].shift())
    df[f"serial_corr_{period}"] = log_ret.rolling(period).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
    )
    return df


# ─── CROSS-TIMEFRAME (placeholder — requires multi-TF data) ──────────

def add_price_vs_higher_tf_ma(df, higher_tf_close=None, periods=[20, 50]):
    if higher_tf_close is None:
        return df
    for p in periods:
        ma = higher_tf_close.rolling(p).mean()
        # Align and forward-fill
        ma_aligned = ma.reindex(df.index, method="ffill")
        df[f"close_vs_higher_sma_{p}"] = (df["close"] / ma_aligned - 1) * 100
    return df


# ─── MASTER ──────────────────────────────────────────────────────────

ALL_FUNCTIONS = [
    add_sma, add_ema, add_wma, add_macd, add_adx, add_aroon, add_cci, add_parabolic_sar,
    add_rsi, add_stoch, add_williams_r, add_roc, add_trix, add_mfi, add_ultimate_osc,
    add_atr, add_bollinger, add_keltner, add_donchian, add_historical_vol,
    add_obv, add_volume_ratio, add_cmf, add_vpt, add_ad_line, add_eom, add_force_index,
    add_returns, add_log_returns, add_price_vs_ma, add_pivot_points, add_swing_detection, add_range_stats,
    add_zscore, add_rolling_skew, add_rolling_kurtosis, add_serial_corr,
]


def compute_all_features(df):
    """Apply all indicator functions in sequence."""
    for func in ALL_FUNCTIONS:
        try:
            df = func(df)
        except Exception as e:
            print(f"  Warning: {func.__name__} failed: {e}")
    return df
