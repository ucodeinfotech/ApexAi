"""Cache manager for Dhan API daily candle data.
Dhan allows ~1 req/sec, reliable API."""

import json
import os
import time
import sys
import threading
from datetime import datetime, timedelta, timezone
from dhanhq import dhanhq, DhanContext
from dhanhq.marketfeed import MarketFeed
import pandas as pd

DHAN_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzgzMjMwNjY0LCJpYXQiOjE3ODMxNDQyNjQsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNDYxNzQxIn0.AMq29v_vykMNE5iHYHy1qnjnqgH5TCRLemugcVHDFkQ8wCdZX-v0kpxMuAWD0v9pMPt1LFmyCiVr94cFM53Bxw"
DHAN_CLIENT_ID = "1102461741"

CACHE_DIR = r"C:\Users\pc\Downloads\stock hist data\ai-pattern-screener\daily_cache"
CACHE_TTL_HOURS = 24
MAX_CANDLES = 100
REQUEST_DELAY = 1.0  # Dhan allows ~1 req/sec

NSE_TOKENS_FILE = r"C:\Users\pc\Downloads\stock hist data\nse_tokens.json"


def get_cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.json")


def is_cache_valid(symbol, max_age_hours=CACHE_TTL_HOURS):
    path = get_cache_path(symbol)
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < max_age_hours * 3600


def save_to_cache(symbol, candles):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = get_cache_path(symbol)
    with open(path, "w") as f:
        json.dump({
            "symbol": symbol,
            "candles": candles,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "count": len(candles),
        }, f)


def load_from_cache(symbol):
    path = get_cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def build_dhan_security_map():
    """Build symbol->security_id map for NSE EQ stocks. Uses cached CSV if fresh."""
    csv_path = os.path.join(os.path.dirname(__file__), "security_id_list.csv")
    try:
        from datetime import datetime as dt
        use_cache = os.path.exists(csv_path) and (time.time() - os.path.getmtime(csv_path)) < 3600
        if use_cache:
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            df = dhanhq.fetch_security_list("compact", filename=csv_path)
        if df is None:
            raise ValueError("Security list returned None")
        nse_eq = df[df["SEM_EXM_EXCH_ID"] == "NSE"]
        nse_eq = nse_eq[nse_eq["SEM_INSTRUMENT_NAME"] == "EQUITY"]
        result = {}
        for _, row in nse_eq.iterrows():
            sym = str(row["SEM_TRADING_SYMBOL"]).strip().upper()
            sec_id = str(row["SEM_SMST_SECURITY_ID"]).strip()
            result[sym] = sec_id
        return result
    except Exception as e:
        print(f"Failed to fetch security master: {e}", file=sys.stderr)
        return {}


def get_stock_list():
    """Returns list of (security_id, symbol) pairs for stocks with CSV data."""
    csv_dir = r"C:\Users\pc\Downloads\stock hist data\comprehensive_data"
    dhan_map = build_dhan_security_map()
    if not dhan_map:
        print("ERROR: Dhan security map empty — cannot proceed", file=sys.stderr)
        return []

    stock_list = []
    if os.path.exists(csv_dir):
        for fname in os.listdir(csv_dir):
            if fname.endswith("_ONE_DAY.csv"):
                sym = fname.replace("_ONE_DAY.csv", "")
                if sym in dhan_map:
                    stock_list.append((dhan_map[sym], sym))

    if not stock_list:
        for sym, sec_id in dhan_map.items():
            stock_list.append((sec_id, sym))
    return stock_list


def fetch_stock_candles(api, security_id, symbol):
    """Fetch daily candles for a single stock using Dhan API."""
    time.sleep(REQUEST_DELAY)
    try:
        to_date = datetime.now() + timedelta(days=1)
        from_date = to_date - timedelta(days=100)
        result = api.historical_daily_data(
            security_id=security_id,
            exchange_segment="NSE_EQ",
            instrument_type="EQUITY",
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )
        if result.get("status") != "success":
            return symbol, None, result.get("remarks", "API returned failure")
        data = result.get("data", {})
        timestamps = data.get("timestamp", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        if not timestamps:
            return symbol, None, "No data returned"
    except Exception as e:
        return symbol, None, str(e)

    candles = []
    for i in range(min(len(timestamps), MAX_CANDLES)):
        try:
            ts = timestamps[-(i + 1)] if i < MAX_CANDLES else timestamps[i]
            candles.append({
                "time": int(float(ts)),
                "date": datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d"),
                "open": float(opens[-(i + 1)]),
                "high": float(highs[-(i + 1)]),
                "low": float(lows[-(i + 1)]),
                "close": float(closes[-(i + 1)]),
                "volume": int(float(volumes[-(i + 1)])),
            })
        except (IndexError, ValueError):
            continue

    candles.reverse()

    if len(candles) < 10:
        return symbol, None, f"Too few candles: {len(candles)}"

    save_to_cache(symbol, candles)
    return symbol, candles, None


def refresh_all(progress_callback=None):
    """Fetch and cache daily candles for all stocks."""
    stock_list = get_stock_list()
    if not stock_list:
        return {"error": "No stocks to process — security master may have failed"}
    total = len(stock_list)
    to_fetch = [(sid, sym) for sid, sym in stock_list if not is_cache_valid(sym)]

    if not to_fetch:
        return {"total": total, "success": 0, "failed": 0, "message": "All stocks already cached"}

    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    api = dhanhq(ctx)

    success = 0
    failed = 0
    errors = []
    done = 0

    for i, (sec_id, sym) in enumerate(to_fetch):
        _, candles, err = fetch_stock_candles(api, sec_id, sym)
        done += 1
        if candles:
            success += 1
        else:
            failed += 1
            if err:
                errors.append({"symbol": sym, "error": err})
        if progress_callback:
            progress_callback({"current": done, "total": len(to_fetch), "success": success, "failed": failed})

    return {
        "total": total,
        "cached_before": total - len(to_fetch),
        "attempted": len(to_fetch),
        "success": success,
        "failed": failed,
        "errors": errors[:50],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def cache_status():
    """Return cache status for all stocks."""
    csv_dir = r"C:\Users\pc\Downloads\stock hist data\comprehensive_data"
    symbols = set()
    if os.path.exists(csv_dir):
        for fname in os.listdir(csv_dir):
            if fname.endswith("_ONE_DAY.csv"):
                symbols.add(fname.replace("_ONE_DAY.csv", ""))
    total = len(symbols)
    cached = 0
    fresh = 0
    stale = 0
    total_candles = 0
    oldest_cache = time.time()
    newest_cache = 0

    for sym in symbols:
        path = get_cache_path(sym)
        if os.path.exists(path):
            cached += 1
            mtime = os.path.getmtime(path)
            age_hours = (time.time() - mtime) / 3600
            if age_hours < CACHE_TTL_HOURS:
                fresh += 1
            else:
                stale += 1
            oldest_cache = min(oldest_cache, mtime)
            newest_cache = max(newest_cache, mtime)
            try:
                data = json.load(open(path))
                total_candles += data.get("count", 0)
            except:
                pass

    return {
        "total": total,
        "cached": cached,
        "fresh": fresh,
        "stale": stale,
        "uncached": total - cached,
        "total_candles": total_candles,
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "oldest_cache": datetime.fromtimestamp(oldest_cache).isoformat() if cached else None,
        "newest_cache": datetime.fromtimestamp(newest_cache).isoformat() if cached else None,
        "cache_dir": CACHE_DIR,
    }


def get_cached_candles(symbol):
    data = load_from_cache(symbol)
    if data and data.get("candles"):
        return data["candles"]
    return None


def compute_indicators(candles):
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    rsis = [50] * len(candles)
    for i in range(14, len(candles)):
        gains = losses = 0
        for j in range(i - 13, i + 1):
            diff = closes[j] - closes[j - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_g = gains / 14
        avg_l = losses / 14 or 1
        rsis[i] = 100 - 100 / (1 + avg_g / avg_l)

    for i, c in enumerate(candles):
        c["body"] = abs(c["close"] - c["open"])
        c["range"] = max(c["high"] - c["low"], 1)
        c["rsi14"] = rsis[i]
        if i >= 20:
            bodies = [candles[j]["body"] for j in range(i - 20, i)]
            vols = [candles[j]["volume"] for j in range(i - 20, i)]
            c["avgBody20"] = sum(bodies) / 20
            c["avgVol20"] = sum(vols) / 20


def fetch_single(symbol):
    """Fetch and cache a single stock from Dhan API."""
    dhan_map = build_dhan_security_map()
    sec_id = dhan_map.get(symbol)
    if not sec_id:
        return {"error": f"Security ID not found for {symbol}"}
    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    api = dhanhq(ctx)
    _, candles, err = fetch_stock_candles(api, sec_id, symbol)
    if candles:
        return {"symbol": symbol, "success": True, "count": len(candles)}
    else:
        return {"symbol": symbol, "success": False, "error": err}


def fetch_stock_candles_30(api, security_id, symbol):
    """Fetch last ~30 daily candles for a single stock (skips sleep)."""
    try:
        to_date = datetime.now() + timedelta(days=1)
        from_date = to_date - timedelta(days=50)
        result = api.historical_daily_data(
            security_id=security_id,
            exchange_segment="NSE_EQ",
            instrument_type="EQUITY",
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )
        if result.get("status") != "success":
            return symbol, None, result.get("remarks", "API returned failure")
        data = result.get("data", {})
        timestamps = data.get("timestamp", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        if not timestamps:
            return symbol, None, "No data returned"
    except Exception as e:
        return symbol, None, str(e)

    candles = []
    for i in range(len(timestamps)):
        try:
            candles.append({
                "time": int(float(timestamps[i])),
                "date": datetime.fromtimestamp(float(timestamps[i])).strftime("%Y-%m-%d"),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": int(float(volumes[i])),
            })
        except (IndexError, ValueError):
            continue

    if len(candles) < 5:
        return symbol, None, f"Too few candles: {len(candles)}"

    save_to_cache(symbol, candles)
    return symbol, candles, None


def force_refresh_all(progress_callback=None):
    """Fetch last 30 days for ALL stocks regardless of cache."""
    stock_list = get_stock_list()
    if not stock_list:
        return {"error": "No stocks to process — security master may have failed"}
    total = len(stock_list)
    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    api = dhanhq(ctx)
    success = 0
    failed = 0
    errors = []
    for i, (sec_id, sym) in enumerate(stock_list):
        _, candles, err = fetch_stock_candles_30(api, sec_id, sym)
        if candles:
            success += 1
        else:
            failed += 1
            if err:
                errors.append({"symbol": sym, "error": err})
        if progress_callback:
            progress_callback({"current": i + 1, "total": total, "success": success, "failed": failed})
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "errors": errors[:50],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def login():
    """Return Dhan API instance (compatibility with old cache_manager interface)."""
    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    return dhanhq(ctx)


def _fetch_candles_range(api, sec_id, from_date, to_date):
    """Helper: fetch candles for a date range, return list or None."""
    try:
        result = api.historical_daily_data(
            security_id=sec_id,
            exchange_segment="NSE_EQ",
            instrument_type="EQUITY",
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )
        if result.get("status") != "success":
            return None
        data = result.get("data", {})
        if isinstance(data, str):
            return None
        timestamps = data.get("timestamp", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        if not timestamps:
            return None
        candles = []
        for i in range(len(timestamps)):
            candles.append({
                "time": int(float(timestamps[i])),
                "date": datetime.fromtimestamp(float(timestamps[i])).strftime("%Y-%m-%d"),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": int(float(volumes[i])),
            })
        return candles
    except:
        return None


def fetch_today_candle(symbol):
    """Fetch today's candles for a symbol. Tries 5-day then 50-day range. Returns candle list or None."""
    dhan_map = build_dhan_security_map()
    sec_id = dhan_map.get(symbol)
    if not sec_id:
        return None
    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    api = dhanhq(ctx)
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Try 5-day range first (fast)
    to_date = datetime.now() + timedelta(days=1)
    from_date = to_date - timedelta(days=5)
    candles = _fetch_candles_range(api, sec_id, from_date, to_date)
    if candles and any(c["date"] == today_str for c in candles):
        return candles

    # Fallback: try 50-day range (some stocks need wider window)
    from_date = to_date - timedelta(days=50)
    candles = _fetch_candles_range(api, sec_id, from_date, to_date)
    if candles:
        return candles
    return None


def batch_fetch_live_quotes(progress_callback=None):
    """Connect once via WebSocket and fetch live quotes for ALL stocks.
    Updates each stock's cache with today's candle (if not already present).
    Returns counts."""
    dhan_map = build_dhan_security_map()
    stock_list = get_stock_list()
    if not stock_list:
        return {"error": "No stock list"}

    # Map security_id -> symbol
    sec_to_sym = {sid: sym for sid, sym in stock_list}
    total = len(stock_list)

    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    received = {}
    done_count = [0]
    conn_error = [None]
    live_candles_added = [0]
    today_str = datetime.now().strftime("%Y-%m-%d")

    def on_ticks(mf, data):
        if not isinstance(data, dict) or data.get("type") != "Quote Data":
            return
        sid = str(data.get("security_id", ""))
        sym = sec_to_sym.get(sid)
        if not sym:
            return
        if sid in received:
            return
        received[sid] = True
        done_count[0] += 1

        # Build today's candle from live quote
        live = {
            "time": today_start,
            "date": today_str,
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "close": float(data.get("LTP", 0)),
            "volume": int(data.get("volume", 0)),
        }

        # Save to cache if today not already present
        cache_path = get_cache_path(sym)
        if os.path.exists(cache_path):
            try:
                cached = json.load(open(cache_path))
                if cached.get("candles"):
                    last_candle = cached["candles"][-1]
                    if last_candle.get("time") == today_start:
                        # Already has today — skip
                        return
                    # Append live candle
                    cached["candles"].append(live)
                    cached["count"] = len(cached["candles"])
                    cached["cached_at"] = datetime.now(timezone.utc).isoformat()
                    json.dump(cached, open(cache_path, "w"), default=str)
                    live_candles_added[0] += 1
            except:
                pass
        else:
            # No cache at all — save single candle
            save_to_cache(sym, [live])

        if progress_callback:
            progress_callback({"current": done_count[0], "total": total, "symbol": sym})

    def on_error(mf, err):
        conn_error[0] = str(err)

    instruments = [(MarketFeed.NSE, sid, MarketFeed.Quote) for sid, sym in stock_list]

    try:
        mf = MarketFeed(
            dhan_context=ctx,
            instruments=instruments,
            version="v2",
            on_connect=lambda mf: None,
            on_ticks=on_ticks,
            on_error=on_error,
        )
        thread = mf.start()

        # Wait until we've heard from all stocks or timeout
        waited = 0
        while done_count[0] < total and waited < 30:
            time.sleep(0.5)
            waited += 0.5

        mf._running = False
        try:
            if mf.ws:
                asyncio.run_coroutine_threadsafe(mf.disconnect(), mf.loop)
        except:
            pass
        thread.join(timeout=3)
    except Exception as e:
        pass

    return {
        "total": total,
        "received": done_count[0],
        "live_candles_added": live_candles_added[0],
    }


def fetch_live_quote(symbol):
    """Fetch today's live quote (open/high/low/ltp/volume) via Dhan WebSocket.
    Returns dict or None."""
    dhan_map = build_dhan_security_map()
    sec_id = dhan_map.get(symbol)
    if not sec_id:
        return None

    ctx = DhanContext(client_id=DHAN_CLIENT_ID, access_token=DHAN_TOKEN)
    result = {}
    done = threading.Event()

    def on_ticks(mf, data):
        if not isinstance(data, dict) or data.get("type") != "Quote Data":
            return
        if str(data.get("security_id", "")) != str(sec_id):
            return
        result.update({
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "ltp": float(data.get("LTP", 0)),
            "volume": int(data.get("volume", 0)),
            "avg_price": float(data.get("avg_price", 0)),
        })
        done.set()

    def on_error(mf, err):
        err_msg[0] = str(err)
        done.set()

    try:
        mf = MarketFeed(
            dhan_context=ctx,
            instruments=[(MarketFeed.NSE, sec_id, MarketFeed.Quote)],
            version="v2",
            on_connect=lambda mf: None,
            on_ticks=on_ticks,
            on_error=lambda mf, e: None,
        )
        thread = mf.start()
        got = done.wait(timeout=8)
        mf._running = False
        try:
            if mf.ws:
                asyncio.run_coroutine_threadsafe(mf.disconnect(), mf.loop)
        except:
            pass
        thread.join(timeout=3)
        if got and result:
            return result
        return None
    except Exception as e:
        return None


def refresh_all_with_live(progress_callback=None):
    """1) Refresh stale historical caches (skips fresh ones), 2) batch-fetch live quotes via WebSocket for ALL stocks.
    After refresh, the live scanner auto-runs from the frontend.
    Returns combined results."""
    hist_result = refresh_all(progress_callback=progress_callback)
    live_result = batch_fetch_live_quotes(progress_callback=progress_callback)
    return {
        "historical": hist_result,
        "live": live_result,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dhan API daily candle cache manager")
    parser.add_argument("action", choices=["refresh", "force_refresh", "status", "fetch", "today", "live", "refresh_live_all"], nargs="?", default="status")
    parser.add_argument("symbol", nargs="?", help="Stock symbol for fetch action")
    args = parser.parse_args()

    if args.action == "status":
        status = cache_status()
        print(json.dumps(status, indent=2))
    elif args.action == "refresh":
        def progress(p):
            print(json.dumps({"event": "progress", **p}), file=sys.stderr)

        result = refresh_all(progress_callback=progress)
        print(json.dumps(result, indent=2))
    elif args.action == "fetch":
        if not args.symbol:
            print(json.dumps({"error": "symbol required"}))
        else:
            result = fetch_single(args.symbol.upper())
            print(json.dumps(result, indent=2))
    elif args.action == "force_refresh":
        def progress(p):
            print(json.dumps({"event": "progress", **p}), file=sys.stderr)
        result = force_refresh_all(progress_callback=progress)
        print(json.dumps(result, indent=2))
    elif args.action == "today":
        if not args.symbol:
            print(json.dumps({"error": "symbol required"}))
            sys.exit(1)
        candles = fetch_today_candle(args.symbol.upper())
        if candles:
            print(json.dumps({"symbol": args.symbol.upper(), "candles": candles}))
        else:
            print(json.dumps({"error": "no data", "symbol": args.symbol.upper()}))
    elif args.action == "live":
        if not args.symbol:
            print(json.dumps({"error": "symbol required"}))
            sys.exit(1)
        quote = fetch_live_quote(args.symbol.upper())
        if quote:
            print(json.dumps({"symbol": args.symbol.upper(), "quote": quote}))
        else:
            print(json.dumps({"error": "no live data", "symbol": args.symbol.upper()}))
    elif args.action == "refresh_live_all":
        def progress(p):
            print(json.dumps({"event": "progress", **p}), file=sys.stderr)
        result = refresh_all_with_live(progress_callback=progress)
        print(json.dumps(result, indent=2))
