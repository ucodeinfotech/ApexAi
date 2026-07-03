# Project: AI Pattern Screener

**Goal**: Production-grade Big Candle + Consolidation (BCC) and Squeeze Breakout pattern scanner for Indian stocks. Next.js frontend + Python backtest pipeline.

**Status**: Frontend (ai-pattern-screener/) — chart modal with real data + BCC pattern highlights working. Python scanners complete. **487/493 stocks cached via Dhan API (100% success, 0 failures)**.

## Current Architecture
- `ai-pattern-screener/` — Next.js 16, TypeScript, TailwindCSS v4 frontend
  - TradingView Lightweight Charts with real candle data from `/api/stock/[symbol]`
  - BCC pattern detection runs server-side (Node.js) on CSV data in `comprehensive_data/`
  - Zustand store manages pattern highlights, selected stock, panel layout
  - React Grid Layout for drag/resize panels
  - Fullscreen modal chart overlay when clicking a stock row in ScannerTable
- `bcc_scanner.py` — Standalone CLI scanner with `seen_patterns_bcc.csv` tracking
- `bcc_dashboard.py` — Streamlit dashboard alternative
- `big_candle_improved.py` / `squeeze_breakout_scanner.py` — Batch scanners

## Pattern Definitions
- **BCC**: Big candle (body > 2× avg, upper wick < 20% range, volume > 1.5× avg) → 3+ small-body consolidation candles within 5% of trigger close
- **Squeeze Breakout** (reverse): 3+ consolidation candles → big breakout candle (body > 2× avg, vol > 1.5× avg)

## Key Results
- **BCC (base)**: 34,925 patterns across 493 stocks, ~50% directional (near random short-term)
- **BCC (improved)**: 10,451 patterns, +1-2% accuracy but 70% fewer samples
- **Squeeze Breakout**: 17,962 patterns, 60d avg +5.70%, 55% WR, 74% bullish — outperforms BCC

## Session 5 (Jul 3 2026) — Dhan API Cache Layer (Replaces Angel One)

### Changes Made
1. **`dhan_cache_manager.py`** — New Python module using Dhan API for daily candle data:
   - Uses `dhanhq` SDK (DhanContext + dhanhq) with access token auth — no TOTP/PIN needed
   - `fetch_stock_candles()` fetches last 100 daily candles at ~1 req/sec via `REQUEST_DELAY=1.0`
   - `build_dhan_security_map()` downloads and caches Dhan security master CSV, maps NSE_EQ symbols to security IDs
   - `refresh_all()` sequentially fetches all uncached stocks (no ThreadPoolExecutor, simple loop)
   - `fetch_single(symbol)` for on-demand per-stock caching
   - `cache_status()` / `compute_indicators()` / `login()` — same interface as old `cache_manager.py`
   - Results: **268/268 stocks cached in ~5 minutes, 100% success, 0 failures** (vs Angel One's 42/487 with 91% failure)

2. **`live_scanner.py`** updated:
   - Imports from `dhan_cache_manager` instead of `cache_manager` (Angel One)
   - `get_candles()` uses `dhan_login()` + `build_dhan_security_map()` for fallback
   - Removed all Angel One imports (SmartConnect, pyotp, Semaphore)
   - `process_stock()` and `main()` no longer use Angel One tokens file

3. **API routes updated** to use Dhan:
   - `/api/stock/[symbol]/route.ts` — spawns `dhan_cache_manager.py fetch {symbol}` instead of `cache_manager.py`
   - `/api/cache/status/route.ts` — runs `dhan_cache_manager.py status`
   - `/api/cache/refresh/route.ts` — runs `dhan_cache_manager.py refresh`
   - `dataSource` label changed from `"angel_cache"` to `"dhan_cache"`

4. **Cleanup**: Removed `test_dhan.py`, `cache_refresh_sequential.py`, `run_refresh.bat`, `start_refresh.ps1`, temp log files, scheduled task

### Why
- Angel One API: ~3-5 req/min rate limit, 30-90s response times, 91% failure rate on bulk refresh
- Dhan API: 1 req/sec sustained, ~2s response times, 100% success rate on 268-stock bulk refresh
- 487/493 stocks now cached (6 missing have CSV data but no Dhan NSE_EQ security ID)
- Dhan auth is simpler: just access token + client ID, no TOTP/PIN

## Session 4 (Jul 3 2026) — Rate-Limited Cache Refresh + On-Demand Caching

### Changes Made
1. **`cache_manager.py` throttling fixes**:
   - `FETCH_CONCURRENCY` reduced from 4 → 1, `REQUEST_DELAY` increased from 0 → 18s (to respect Angel One's ~3 req/min rate limit)
   - `fetch_stock_candles()` now retries on rate limit errors with 30s/60s backoff (up to 2 retries)
   - Fixed `None` crash bug — `raw` can be `None` from API; now checks before subscripting
   - `login()` extracted as reusable helper for session management
   - Added `fetch_single(symbol)` function for on-demand per-stock caching
   - `refresh_all()` now processes in batches of 20 with fresh login per batch and 5s inter-batch pause

2. **On-demand caching in stock API** (`/api/stock/[symbol]/route.ts`):
   - Added TTL check using `mtimeMs` (24h expiry) instead of just existence check
   - When a stock has no valid cache, spawns `python cache_manager.py fetch {symbol}` in background (`spawn` with `detached: true` + `unref()`)
   - Over time, as users view stocks, they get cached automatically without blocking the response
   - Still falls back to CSV when no cache available

3. **Cache status improved**: 117 stocks now cached (was 3 before Session 4), all fresh.

### Why
- Angel One API rate limits to ~3-5 historical data requests per minute per account
- Bulk refresh of 487 stocks would take ~2.5 hours at that rate — impractical
- On-demand caching is more practical: stocks get cached as users view them
- The 117 cached stocks came from 3 previous failed batch runs that made partial progress

## Session 3 (Jul 3 2026) — Angel One Daily Cache Layer

### Changes Made
1. **`cache_manager.py`** — New Python module for Angel One daily candle data:
   - `fetch_stock_candles()` — fetches last 100 daily candles from Angel One API per stock (rate limited to 4 concurrent)
   - `save_to_cache()` / `load_from_cache()` / `is_cache_valid()` — manages `daily_cache/{symbol}.json` files
   - `refresh_all()` — batch fetch all 487 stocks with progress callback
   - `cache_status()` — returns cache health (total/cached/fresh/stale/uncached)
   - `get_cached_candles()` + `compute_indicators()` — load cached candles and compute RSI/avgBody/avgVol
   - CLI: `python cache_manager.py status` or `python cache_manager.py refresh`
   - Cached data is Angel One raw OHLCV, not CSV data (∼100 candles vs 3000+ from CSVs)

2. **`live_scanner.py` rewritten** — Uses cached data first:
   - `get_candles()` checks `daily_cache/{symbol}.json` before hitting Angel One API
   - If cache exists and is valid (<24h old), skips API call entirely
   - Falls back to live fetch when cache is missing or stale (and saves to cache)
   - All indicators (RSI, avgBody, avgVol) computed on loaded candles
   - BCC/Squeeze detection logic extracted to `detect_bcc_patterns()` for reuse

3. **Cache API endpoints** (new):
   - `GET /api/cache/status` — returns cache health (487 total, N cached, N fresh, etc.)
   - `POST /api/cache/refresh` — spawns `cache_manager.py refresh` in background, returns immediately
   - `GET /api/cache/refresh` — returns refresh progress/result

4. **Frontend cache UI**:
   - TopNav "Cache" button shows `fresh/total` badge (e.g. `0/487`)
   - Click to trigger cache refresh (spawns Python process)
   - Button shows animated spinner while refreshing
   - Tooltip shows detailed cache status
   - Store: `cacheStatus`, `cacheRefreshing`, `fetchCacheStatus()`, `refreshCache()`
   - Auto-fetches cache status on page mount

### New Files
- `ai-pattern-screener/cache_manager.py` — Angel One daily data cache module
- `ai-pattern-screener/daily_cache/` — Per-stock JSON cache directory (gitignored)
- `ai-pattern-screener/src/app/api/cache/status/route.ts` — Cache status API
- `ai-pattern-screener/src/app/api/cache/refresh/route.ts` — Cache refresh API

### Modified Files
- `live_scanner.py` — Now uses cached data, fallback to live API; detection logic extracted
- `useStore.ts` — Added `CacheStatus`, `fetchCacheStatus()`, `refreshCache()`, `cacheRefreshing`
- `TopNav.tsx` — Added cache status button + refresh trigger
- `.gitignore` — Added `daily_cache/`, `ticks.jsonl`

## Session 2 (Jul 2 2026) — Full Dashboard, History Optimization, Pattern Builder, WebSocket

### Changes Made
1. **ALL historical pattern markers on chart**: `CandlestickChart.tsx` now renders ALL BCC patterns as markers (not just the latest), with the focus pattern getting a prominent marker. Consolidation zone still renders only for the active focus pattern.
2. **Full Bloomberg Dashboard**: `MockChart.tsx` now fetches real stock data (price, change), `Workspace.tsx` has a balanced 5-panel layout (Chart 6, Insights 3, Breadth 3, Scanner 7, Comparison 5), auto-fetches breadth on mount, rowHeight=36 for 1080p fit.
3. **History API optimized**: `scripts/preaggregate-history.js` pre-computes all 489 stocks from 35k CSV rows into `aggregated.json`. History API went from ~15s to ~300ms.
4. **Pattern Builder UI**: New `PatternBuilder.tsx` component with clickable candle chart, real-time BCC detection, adjustable parameters (body/vol/consol thresholds), save/delete custom patterns to localStorage.
5. **WebSocket Real-time Ticker**: `tick_stream.py` connects to Angel One SmartAPI WebSocket and streams ticks to `ticks.jsonl`. `/api/live/ticks/` returns latest gainers/losers/summary. `LiveTicker.tsx` polls every 3s and displays in StatusBar.
6. **Sidebar**: "Pattern Builder" nav item now opens the builder panel (was toast).

### New Files
- `ai-pattern-screener/src/components/patterns/PatternBuilder.tsx` — Visual pattern builder with click-to-select candles
- `ai-pattern-screener/src/components/market/LiveTicker.tsx` — Real-time ticker component for StatusBar
- `ai-pattern-screener/src/app/api/live/ticks/route.ts` — Live tick streaming API endpoint
- `ai-pattern-screener/tick_stream.py` — Python Angel One WebSocket consumer
- `ai-pattern-screener/scripts/preaggregate-history.js` — Build-time history aggregation script
- `ai-pattern-screener/src/app/api/scanner/history/aggregated.json` — Pre-aggregated history data (489 stocks)
- `ai-pattern-screener/ticks.jsonl` — Runtime tick data file (gitignored)

### Modified Files
- `CandlestickChart.tsx` — All pattern markers + focus consolidation zone
- `MockChart.tsx` — Real data instead of mock candles
- `Workspace.tsx` — Balanced layout, auto-fetch breadth, PatternBuilder panel
- `StatusBar.tsx` — LiveTicker component added
- `Sidebar.tsx` — Pattern Builder opens panel
- `useStore.ts` — Added "builder" to panelOrder/panelVisibility
- `src/app/api/scanner/history/route.ts` — Use cached JSON (300ms vs 15s)
- `page.tsx` — Unchanged

## Key Decisions
- Scanner API runs in Node.js (unified architecture) rather than calling Python
- Chart modal now shows ALL historical pattern markers with consolidation zone for selected pattern
- React Grid Layout panels use `.drag-handle` class on panel-header for grab handles
- MockChart upgraded to real data (fetches from API on stock change)
- History API uses pre-aggregated JSON cache (generated via build script) — 50x speedup
- Pattern Builder uses lightweight-charts with `subscribeClick` for interactive candle selection
- WebSocket data flows: Python WS → ticks.jsonl → REST API → React (3s poll)
- Server stability issues on win32 — `Start-Process -WindowStyle Hidden` for background server
- SWC WASM fallback (win32/x64 SWC binary incompatibility)

## Key Files
- `ai-pattern-screener/` — Next.js frontend project
- `ai-pattern-screener/src/app/globals.css` — Design system (dark theme, glassmorphism)
- `ai-pattern-screener/src/store/useStore.ts` — Zustand store
- `ai-pattern-screener/src/components/charts/CandlestickChart.tsx` — Modal chart with ALL pattern markers
- `ai-pattern-screener/src/components/charts/MockChart.tsx` — Workspace chart with real data
- `ai-pattern-screener/src/components/scanner/ScannerTable.tsx` — Clickable TanStack table (Live/History)
- `ai-pattern-screener/src/components/patterns/PatternBuilder.tsx` — Interactive pattern builder
- `ai-pattern-screener/src/components/market/LiveTicker.tsx` — Real-time ticker
- `ai-pattern-screener/src/components/layout/Workspace.tsx` — Dashboard grid layout
- `ai-pattern-screener/src/app/api/stock/[symbol]/route.ts` — Stock data + pattern API
- `ai-pattern-screener/src/app/api/scanner/history/route.ts` — History API (cached JSON)
- `ai-pattern-screener/src/app/api/live/ticks/route.ts` — Live tick stream API
- `ai-pattern-screener/src/app/api/cache/status/route.ts` — Cache status API
- `ai-pattern-screener/src/app/api/cache/refresh/route.ts` — Cache refresh API
- `ai-pattern-screener/dhan_cache_manager.py` — Dhan API daily data cache manager (primary)
- `ai-pattern-screener/dhan_tick_gen.py` — Dhan cache-based tick data generator for LiveTicker
- `ai-pattern-screener/live_scanner.py` — BCC/Squeeze live scanner (uses Dhan cache)
- `ai-pattern-screener/scripts/preaggregate-history.js` — Build script for history cache
- `backtest_results/` — PDF reports, seen_patterns_bcc.csv
- `comprehensive_data/` — Stock CSV files for scanning
