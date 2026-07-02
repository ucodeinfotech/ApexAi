"""
Fixed backtest engine: same-strike tracking for all option backtests
"""
import duckdb, pandas as pd, numpy as np, os, warnings

DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"

def load_fixed_data(trades_pre):
    """Load atm_distance=0 (for entry) + per-strike data (for exit tracking).
    Returns: (atm_ts_arr, atm_cl_arr, atm_st_arr, strike_cache, lookup_fn)
    """
    con = duckdb.connect(DB_PATH)
    df_atm = con.execute("""SELECT timestamp, close, strike FROM options_data_dedup
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
    ORDER BY timestamp""").fetchdf()
    con.close()
    df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
    atm_ts_arr = df_atm["timestamp"].values.astype("datetime64[us]")
    atm_cl_arr = df_atm["close"].values.astype(float)
    atm_st_arr = df_atm["strike"].values.astype(float)

    # Find entry info + collect unique strikes
    def lookup(ts):
        i = np.searchsorted(atm_ts_arr, np.datetime64(ts, "us"))
        if i >= len(atm_ts_arr): return len(atm_ts_arr)-1, atm_cl_arr[-1], atm_st_arr[-1]
        if i == 0: return 0, atm_cl_arr[0], atm_st_arr[0]
        # Use exact match bar if timestamp matches, else previous bar
        return (i, atm_cl_arr[i], atm_st_arr[i]) if atm_ts_arr[i] == np.datetime64(ts, "us") else (i-1, atm_cl_arr[i-1], atm_st_arr[i-1])

    strike_set = set()
    for ed in trades_pre["ed_naive"]:
        _, _, st = lookup(ed)
        strike_set.add(int(st))

    # Load per-strike data (single query for all strikes; use clean table if available)
    TABLE = "options_data_clean"  # use this if exists, else options_data_dedup
    try:
        con2 = duckdb.connect(DB_PATH)
        con2.execute("SELECT COUNT(*) FROM options_data_clean").fetchone()
    except:
        TABLE = "options_data_dedup"
    stk_list = sorted(strike_set)
    stk_where = ",".join(str(s) for s in stk_list)
    df_all = con2.execute(f"""SELECT timestamp, close, open, high, low, strike FROM {TABLE}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({stk_where})
    ORDER BY strike, timestamp""").fetchdf()
    con2.close()
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)

    strike_cache = {}
    for stk, grp in df_all.groupby("strike"):
        ts = grp["timestamp"].values.astype("datetime64[us]")
        cl = grp["close"].values.astype(float)
        hi = grp["high"].values.astype(float)
        lo = grp["low"].values.astype(float)
        op = grp["open"].values.astype(float)
        # Pre-compute ATR for trail functions
        tr_vals = pd.DataFrame({
            "hl": pd.Series(hi) - pd.Series(lo),
            "hc": abs(pd.Series(hi) - pd.Series(cl).shift(1)),
            "lc": abs(pd.Series(lo) - pd.Series(cl).shift(1)),
        }).max(axis=1).fillna(0)
        atr = tr_vals.ewm(span=14, min_periods=3, adjust=False).mean().values.astype(float)
        strike_cache[int(stk)] = {"ts": ts, "cl": cl, "op": op, "hi": hi, "lo": lo, "atr": atr}
    return atm_ts_arr, atm_cl_arr, atm_st_arr, strike_cache, lookup

def get_entry_strike_idx(ed, lookup, atm_ts_arr, strike_cache):
    """Get (entry_si_in_atm, entry_strike, entry_price, s_idx_in_strike_data)"""
    si, ep, st = lookup(ed)
    st = int(st)
    stk_data = strike_cache.get(st)
    if stk_data is None: return None
    entry_ns = atm_ts_arr[si]
    s_idx = np.searchsorted(stk_data["ts"], entry_ns)
    if s_idx >= len(stk_data["cl"]): return None
    return {"si": si, "strike": st, "ep": float(stk_data["cl"][s_idx]), "s_idx": s_idx, "stk_data": stk_data}

def exit_tp_maxd(stk_data, s_idx, tp, max_days):
    """Same-strike: take profit at tp pts or time stop at max_days"""
    end_ns = stk_data["ts"][s_idx] + np.timedelta64(int(max_days * 86400 * 1e6), "us")
    ep = stk_data["cl"][s_idx]
    for i in range(s_idx + 1, min(s_idx + 3000, len(stk_data["cl"]))):
        if stk_data["ts"][i] > end_ns:
            return stk_data["cl"][i], stk_data["ts"][i]
        if stk_data["cl"][i] - ep >= tp:
            return stk_data["cl"][i], stk_data["ts"][i]
    return None, None

def exit_maxd(stk_data, s_idx, max_days):
    """Same-strike: time stop exit"""
    end_ns = stk_data["ts"][s_idx] + np.timedelta64(int(max_days * 86400 * 1e6), "us")
    for i in range(s_idx + 1, min(s_idx + 3000, len(stk_data["cl"]))):
        if stk_data["ts"][i] > end_ns:
            return stk_data["cl"][i], stk_data["ts"][i]
    return None, None

def exit_tp_sl(stk_data, s_idx, tp, sl, max_days):
    """Same-strike: TP + SL + time stop"""
    end_ns = stk_data["ts"][s_idx] + np.timedelta64(int(max_days * 86400 * 1e6), "us")
    ep = stk_data["cl"][s_idx]
    for i in range(s_idx + 1, min(s_idx + 3000, len(stk_data["cl"]))):
        if stk_data["ts"][i] > end_ns:
            return stk_data["cl"][i], stk_data["ts"][i]
        pnl = stk_data["cl"][i] - ep
        if pnl >= tp:
            return stk_data["cl"][i], stk_data["ts"][i]
        if sl is not None and pnl <= -abs(sl):
            return stk_data["cl"][i], stk_data["ts"][i]
    return None, None

def exit_trail(stk_data, s_idx, ch_val, max_days=None):
    """Same-strike: trailing stop on option price with CH*ATR"""
    atr_arr = stk_data["atr"]

    end_ns = None
    if max_days is not None:
        end_ns = stk_data["ts"][s_idx] + np.timedelta64(int(max_days * 86400 * 1e6), "us")

    he = stk_data["hi"][s_idx]
    for i in range(s_idx + 1, min(s_idx + 3000, len(stk_data["cl"]))):
        if end_ns is not None and stk_data["ts"][i] > end_ns:
            return stk_data["cl"][i], stk_data["ts"][i]
        ca = atr_arr[i]
        if np.isnan(ca): continue
        if stk_data["hi"][i] > he: he = stk_data["hi"][i]
        if stk_data["cl"][i] < he - ch_val * ca:
            return stk_data["cl"][i], stk_data["ts"][i]
    return None, None

def exit_tp_eod(stk_data, s_idx, tp):
    """Same-strike day trade: TP exit if hit, else exit at last bar of same trading day.
    The '3:20-3:30' exit is approximated by the last available bar of the entry day."""
    ep = stk_data["cl"][s_idx]
    entry_ns = stk_data["ts"][s_idx]
    entry_date = entry_ns.astype("datetime64[D]")
    next_day = entry_date + np.timedelta64(24*60, "m")
    last_idx = np.searchsorted(stk_data["ts"], next_day) - 1
    if last_idx < 0 or last_idx <= s_idx:
        return None, None
    for i in range(s_idx + 1, last_idx + 1):
        if stk_data["cl"][i] - ep >= tp:
            return stk_data["cl"][i], stk_data["ts"][i]
    return stk_data["cl"][last_idx], stk_data["ts"][last_idx]

def get_entry_info(ed, atm_ts, atm_cl, atm_st, strike_cache):
    """Simplified entry: handles exact match at index 0.
    Returns dict with strike, ep, s_idx, stk_data or None."""
    ts64 = np.datetime64(ed, "us")
    i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts):
        si = len(atm_ts) - 1
    elif i == 0:
        si = 0
    else:
        si = i if atm_ts[i] == ts64 else i - 1
    st = int(atm_st[si])
    sd = strike_cache.get(st)
    if sd is None: return None
    s_idx = np.searchsorted(sd["ts"], atm_ts[si])
    if s_idx >= len(sd["cl"]): return None
    return {"strike": st, "ep": float(sd["cl"][s_idx]), "s_idx": s_idx, "stk_data": sd}
