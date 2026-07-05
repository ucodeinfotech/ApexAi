# Project: Gap-Cross + 5min Confirmation Strategy (prev: AI Pattern Screener)

**Goal**: Predict/trade next-day >2% gainers and intraday direction for Indian stocks. Key finding: **naive gap-cross fails, but 5min intraday confirmation transforms it to Sharpe 4.28+**.

**Status**: Gap-cross backtest complete (107K+ trades, 475 stocks, 10 years). **Breakthrough**: 5min candle confirmation is the entire edge (Sharpe 0.92 → 4.28). **Data issue**: `cleaned_features.parquet` has critical data gaps — **Dhan cache (209 stocks, 56K rows) is verified clean** and used for `top40_gainers_verified.csv/pdf`.

## Key Results
- **Naive gap-cross**: 40.9% WR, -0.92 Sharpe — loses money
- **5min confirmation**: **57.1% WR, Sharpe 4.28, +0.79% avg** with limit-at-prev_high entry. Triple confirmation best at 59.9%/Sharpe 4.93
- **ML Direction (close>open)**: AUC=0.708, 82.6% WR at 0.80 threshold (saved: `direction_model.txt`)
- **ML Big Day (>2%)**: AUC=0.809, 72.3% WR at 0.80 threshold (saved: `big_day_model.txt`)
- **Top 40 gainers (verified)**: 1,600 rows, 209 stocks, 40 trading days (May 6→Jul 2) — Dhan cache, no gaps

## Repository Layout
- `comprehensive_data/` — 493 stocks × 5 timeframes CSV (1min/5min/15min/1hr/daily)
- `cleaned_features.parquet` — Daily OHLCV + features, 475 symbols, **has data gaps in Jun 2026**
- `ai-pattern-screener/` — Next.js 16 frontend (BCC scanner), Dhan cache manager, live scanner
- `C:\Users\pc\AppData\Local\Temp\opencode\daily_cache\` — 209 Dhan API parquet files (verified, up to Jul 2)
- `gap_cross_full_report.pdf` — 28-page full strategy report
- `top40_gainers_verified.csv/pdf` — Verified top-40 gainers (Dhan cache, no gaps)
- `direction_model.txt` / `big_day_model.txt` — LightGBM models
- `gap_cross_strategy_logic.md` — Complete deployment spec

## Session 10 (Jul 5 2026) — Data Quality Fix via Dhan Cache + Verified Top-40 Gainers

### Changes Made
1. **Data quality audit**: Found `cleaned_features.parquet` has fatal data gaps:
   - CARBORUNIV shows +33.63% on May 25 — fake (stale prev_close from Mar 25 vs actual ~1056)
   - Stock coverage drops from 472 to 15 stocks between Jun 15→Jun 24
   - Nifty 50 stocks (INDIGO, TRENT, ICICIBANK) missing entirely after Jun 15

2. **Verified Dhan cache data**: 209 stocks × 56,624 rows, all 40 days have **complete 209 stocks** (no gaps):
   - CARBORUNIV May 25: **+0.05%** ✅ (matches online)
   - INDIGO Jun 24: **+4.95%** ✅ (matches TOI +4.72%)
   - TRENT Jun 24: **+3.21%** ✅ (matches LiveMint +3.32%)

3. **Generated verified top-40 gainers**:
   - `top40_gainers_verified.csv` — 1,600 rows, 40 days × 40 stocks (Dhan cache)
   - `top40_gainers_verified.pdf` — 254 KB, cover + daily tables + frequency + summary
   - Date range: May 6→Jul 2 (was Apr 28→Jun 24 from broken parquet)
   - 208 unique symbols with complete data (vs 438 with gaps before)

## Session 9 (Jul 5 2026) — Deep Backtest, ML Models, 5min Breakthrough

### Findings
1. **Full gap-cross backtest**: 107,639 trades, 475 stocks, 10 years
   - Naive: 40.9% WR, -0.92 Sharpe, -0.14% avg ret — **loses money**
   - Overnight hold only: Sharpe 0.21 (barely positive)
   - Short-no-cross: Sharpe 1.27 (best of naive variants)

2. **5min intraday confirmation (breakthrough)**: 57 stocks × 10 yrs (59,869 events)
   - "Both candles above prev_high": **57.1% WR, Sharpe 4.28, +0.79% avg**
   - With volume filter: 59.3% WR, Sharpe 4.47
   - Triple confirmation best: 59.9% WR, Sharpe 4.93
   - **Entry must be limit at prev_high** — market entry even 0.1% above destroys edge (drops to 42.8%)
   - 91% of crosses happen before 11 AM

3. **Stock selection filter**: Walk-forward top-10 pick → 44.6% WR, Sharpe 0.40 — marginal help
   - Only 2 stocks (TATAMOTORS, AXISBANK) consistently good, 64% of years

4. **ML models**:
   - Direction (close>open): AUC=0.708, 82.6% WR at 0.80 threshold. Top feature: volume
   - Big Day (>2%): AUC=0.809, 72.3% WR at 0.80 threshold. Top feature: volume > gap_pct (AUC=0.66 alone)
   - Ceiling at 0.71/0.81 AUC — no fundamental/sentiment/options data available

5. **Key insight**: 5min confirmation > ML > stock selection. Simple rule beats complex models.

## Session 5 (Jul 3 2026) — Dhan API Cache Layer (Replaces Angel One)

### Changes Made
1. **`dhan_cache_manager.py`** — Dhan API daily data cache module (primary, replaces Angel One)
   - Uses `dhanhq` SDK with access token auth — no TOTP/PIN
   - `fetch_stock_candles()` fetches last 100 daily candles at ~1 req/sec
   - Results: **268/268 stocks cached in ~5 minutes, 100% success, 0 failures**

2. **`live_scanner.py`** updated to import from `dhan_cache_manager` instead of `cache_manager`

3. **API routes** updated: `/api/stock/[symbol]`, `/api/cache/status`, `/api/cache/refresh` now use Dhan

### Why
- Angel One: ~3-5 req/min rate limit, 91% failure rate, TOTP needed
- Dhan: 1 req/sec, 100% success, simpler auth

## Session 4 (Jul 3 2026) — Rate-Limited Cache Refresh + On-Demand Caching
## Session 3 (Jul 3 2026) — Angel One Daily Cache Layer
## Session 2 (Jul 2 2026) — Full Dashboard, History Optimization, Pattern Builder, WebSocket

## Key Decisions
- **Drop ML for rule-based deployment**: 5min candle confirmation > ML models (Sharpe 4.28 vs 0.71-0.81 AUC)
- **Fix data gaps using Dhan cache**: `cleaned_features.parquet` unreliable for recent dates
- **Same-day exit mandatory**: Overnight hold Sharpe drops 4.28 → 0.21
- **Limit entry at prev_high only**: Market entry even 0.1% above destroys edge
- **Cancel unfilled orders by 11 AM**: 91% of crosses happen before then
- Dhan cache primary data source; Angel One deprecated
- Scanner API in Node.js (unified architecture)
- History API uses pre-aggregated JSON cache (50x speedup)

## Relevant Files
- `ai-pattern-screener/src/app/api/stock/[symbol]/route.ts` — Stock data + pattern API
- `ai-pattern-screener/dhan_cache_manager.py` — Dhan API daily data cache manager
- `ai-pattern-screener/dhan_tick_gen.py` — Dhan cache-based tick generator
- `ai-pattern-screener/live_scanner.py` — BCC/Squeeze live scanner
- `gap_cross_full_report.pdf` — Full strategy report (28p)
- `gap_cross_strategy_logic.md` — Deployment spec
- `top40_gainers_verified.csv` — Verified top 40 gainers (1,600 rows)
- `top40_gainers_verified.pdf` — Verified PDF report (254 KB, 40 days)
- `direction_model.txt` / `big_day_model.txt` — LightGBM models
