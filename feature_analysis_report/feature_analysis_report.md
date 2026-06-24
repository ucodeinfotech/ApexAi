# Feature Engineering & Analysis ReportGenerated: 2026-06-23 12:14Universe: 90 NSE large-capsPeriod: 2016–2026 (226,231 rows)Features: 74 total
---## 1. Feature Engineering — Complete Catalog

### Base Technical (45)
- **sma_5**: Simple Moving Average 5d: avg close over 5 days, captures short-term trend- **sma_10**: SMA 10d: medium-short trend baseline- **sma_20**: SMA 20d: ~1 month trend (trading month)- **sma_50**: SMA 50d: ~2.5 month intermediate trend- **ema_5**: Exponential MA 5d: weighted recent close, reacts faster than SMA- **ema_10**: EMA 10d: short-term momentum with decay weighting- **ema_20**: EMA 20d: 1-month exponential trend- **ema_50**: EMA 50d: intermediate exponential trend- **rsi_7**: RSI 7d: short-term momentum oscillator (0-100), overbought>70 oversold<30- **rsi_14**: RSI 14d: classic momentum oscillator, measures speed/change of price moves- **rsi_21**: RSI 21d: longer momentum view, smoother than 14- **macd_line**: MACD Line: 12d EMA - 26d EMA, measures trend momentum direction- **macd_signal**: MACD Signal: 9d EMA of MACD line, trigger line for crossovers- **macd_hist**: MACD Histogram: MACD line - signal line, shows momentum acceleration- **adx**: ADX 14d: trend strength (0-100), >25=trending, <20=range-bound- **plus_di**: +DI 14d: positive directional indicator, measures upward pressure- **minus_di**: -DI 14d: negative directional indicator, measures downward pressure- **atr_7**: ATR 7d: short-term volatility (avg true range over 7 days)- **atr_14**: ATR 14d: classic volatility measure (Wilder 14-period)- **atr_21**: ATR 21d: longer volatility baseline- **bb_pct_b**: Bollinger %B: where close sits in bands (0-1), >1=above upper band- **bb_width**: Bollinger Width: band width as % of middle, expands/contracts with volatility- **kc_width**: Keltner Width: channel width as % of EMA, volatility with ATR-based bands- **dc_width**: Donchian Width: 20d high-low range as % of midpoint, breakout volatility- **obv**: On-Balance Volume: cumulative volume adjusted by price direction- **cmf**: Chaikin Money Flow 20d: volume-weighted accumulation/distribution- **stoch_k**: Stochastic %K 5d: current close relative to 5d range (fast)- **stoch_d**: Stochastic %D 3d: moving avg of %K (slow signal line)- **williams_r**: Williams %R 14d: inverted stochastic, -100 to 0, overshot below -80- **mfi**: Money Flow Index 14d: volume-weighted RSI, 0-100- **uo**: Ultimate Oscillator: multi-timeframe (7/14/28) momentum composite- **cci**: CCI 20d: Commodity Channel Index, cyclical deviation from statistical mean- **trix**: TRIX 15d: triple-smoothed EMA rate of change, filters noise- **roc_5**: ROC 5d: raw price rate of change over 5 days (%)- **roc_10**: ROC 10d: price momentum over 10 days- **roc_20**: ROC 20d: price momentum over 20 days (~monthly)- **zscore_20**: Z-Score 20d: how many std devs close is from 20d mean- **skew_20**: Skew 20d: asymmetry of return distribution over 20 days- **kurt_20**: Kurtosis 20d: tailedness of return distribution- **hv_10**: HV 10d: annualized historical volatility over 10 days- **hv_20**: HV 20d: annualized historical volatility over 20 days- **hv_30**: HV 30d: annualized historical volatility over 30 days- **eom**: Ease of Movement 14d: price-volume efficiency ratio- **fi**: Force Index 13d: price change * volume, momentum with volume weighting- **vpt**: Volume Price Trend: cumulative volume-weighted price trend
### Range (1)
- **range_pct**: Current day high-low range as % of close, raw volatility
### Calendar (5)
- **dow**: Day of week (0=Mon..4=Fri), captures day-of-week effects- **month**: Calendar month (1-12), captures monthly/seasonal patterns- **is_month_end**: Last trading day of month (0/1), month-end effects- **is_quarter_end**: Last trading day of quarter (0/1), quarter-end rebalancing- **is_thursday**: Thursday flag (0/1), captures weekly expiry effects
### VIX / Fear (9)
- **vix_close**: India VIX closing level, market fear/greed gauge- **vix_change**: VIX daily % change, fear momentum- **vix_range**: VIX daily high-low range, uncertainty magnitude- **vix_ma_5**: VIX 5d MA, short-term fear trend- **vix_ma_20**: VIX 20d MA, medium-term fear trend- **vix_zscore_20**: VIX z-score vs 20d: how extreme current fear is- **vix_ma_5_r**: VIX / VIX_ma5 - 1: short-term fear deviation- **vix_ma_20_r**: VIX / VIX_ma20 - 1: medium-term fear deviation- **vix_high_r**: VIX high ratio (reserved)
### Delivery / Institutional (4)
- **delivery_pct**: % of traded quantity delivered (NSE), institutional participation signal- **delivery_pct_ma5**: 5d MA of delivery %, smoothed institutional flow- **delivery_pct_ma20**: 20d MA of delivery %, long-term delivery baseline- **delivery_delta**: delivery_pct - delivery_pct_ma5, short-term deviation in delivery
### Intraday Multi-timeframe (10)
- **intra_rsi_mean**: Avg RSI 14 across 60min bars today, intraday momentum- **intra_rsi_std**: Std of RSI 14 across 60min bars, intraday momentum dispersion- **intra_vol_std**: Std of close changes across 60min bars, intraday noise- **intra_range_sum**: Sum of 60min bar ranges as % of close, total intraday movement- **intra_range_max**: Max single 60min bar range, intraday volatility spike- **intra_bb_width_mean**: Avg Bollinger Width across 60min bars, intraday volatility regime- **intra_macd_std**: Std of MACD hist across 60min bars, intraday momentum consistency- **intra_rsi_mean_ma5**: 5d MA of intra_rsi_mean, persistent intraday momentum- **intra_range_sum_ma5**: 5d MA of intra_range_sum, persistent intraday volatility- **intra_vol_std_ma5**: 5d MA of intra_vol_std, persistent intraday noise---
## 2. Exploratory Data Analysis
### 2.1 Data Shape
- Rows: 226,231
- Symbols: 90
- Features: 74
- Date range: 2016-12-13 18:30:00 to 2026-06-17 18:30:00
### 2.2 Missing Values
Features with any missing: 0/74

Features with zero missing: 74/74
### 2.3 Feature Summary Statistics
| Feature | Mean | Std | Min | P25 | P50 | P75 | Max |
|---------|------|-----|-----|-----|-----|-----|----|
| sma_5 | 1596.613 | 4752.554 | 3.168 | 90.470 | 302.590 | 1017.930 | 53051.700 |
| sma_10 | 1594.568 | 4748.239 | 3.228 | 90.325 | 302.295 | 1016.890 | 52111.330 |
| sma_20 | 1590.460 | 4739.683 | 3.467 | 90.303 | 301.058 | 1013.361 | 51714.495 |
| sma_50 | 1578.521 | 4715.272 | 3.819 | 90.306 | 298.050 | 1000.618 | 50640.962 |
| ema_5 | 1596.625 | 4752.272 | 3.186 | 90.527 | 302.716 | 1018.368 | 52902.556 |
| ema_10 | 1594.571 | 4747.611 | 3.292 | 90.495 | 302.135 | 1016.549 | 52435.231 |
| ema_20 | 1590.518 | 4738.575 | 3.456 | 90.665 | 301.586 | 1013.071 | 51824.119 |
| ema_50 | 1579.031 | 4713.661 | 3.927 | 90.791 | 298.343 | 1000.321 | 50349.923 |
| rsi_7 | 51.593 | 22.661 | 0.106 | 34.302 | 51.714 | 69.262 | 99.997 |
| rsi_14 | 51.760 | 17.002 | 0.482 | 39.573 | 51.677 | 64.045 | 99.874 |
| rsi_21 | 51.797 | 14.311 | 0.934 | 41.711 | 51.665 | 61.873 | 99.429 |
| macd_line | 5.607 | 99.066 | -1661.143 | -2.398 | 0.465 | 6.348 | 1849.671 |
| macd_signal | 5.549 | 93.336 | -1538.332 | -2.237 | 0.466 | 6.162 | 1632.269 |
| macd_hist | 0.059 | 29.596 | -881.346 | -1.136 | 0.010 | 1.184 | 977.655 |
| adx | 34.308 | 13.387 | 8.338 | 24.185 | 31.746 | 42.168 | 93.182 |
| plus_di | 22.490 | 10.451 | 0.486 | 14.782 | 21.271 | 28.838 | 86.722 |
| minus_di | 20.591 | 9.932 | 0.297 | 13.236 | 19.539 | 26.770 | 92.428 |
| atr_7 | 45.928 | 133.362 | 0.098 | 2.974 | 9.336 | 30.502 | 2388.790 |
| atr_14 | 45.855 | 131.267 | 0.122 | 3.020 | 9.444 | 30.777 | 1992.652 |
| atr_21 | 45.785 | 130.265 | 0.131 | 3.045 | 9.490 | 30.975 | 1739.991 |
---
## 3. Label / Target Analysis
### 3.1 Target: `label_2pct`
- Positive: 69,081 (30.54%)
- Negative: 157,150 (69.46%)
- Ratio: 1:2.3
### 3.1 Target: `label_3pct`
- Positive: 40,241 (17.79%)
- Negative: 185,990 (82.21%)
- Ratio: 1:4.6
### 3.1 Target: `label_5pct`
- Positive: 13,028 (5.76%)
- Negative: 213,203 (94.24%)
- Ratio: 1:16.4
### 3.1 Target: `label_5pct_any`
- Positive: 28,971 (12.81%)
- Negative: 197,260 (87.19%)
- Ratio: 1:6.8
### 3.1 Target: `label_bullish`
- Positive: 28,241 (12.48%)
- Negative: 197,990 (87.52%)
- Ratio: 1:7.0
### 3.2 Target: `label_5pct` (Primary — Bullish Range >5%)
#### By Year
| Year | Total | Events | Base Rate |
|------|-------|--------|-----------|
| 2016 | 1,033 | 43 | 4.16% |
| 2017 | 22,441 | 962 | 4.29% |
| 2018 | 23,700 | 1,424 | 6.01% |
| 2019 | 23,330 | 1,371 | 5.88% |
| 2020 | 24,328 | 2,689 | 11.05% |
| 2021 | 25,493 | 1,814 | 7.12% |
| 2022 | 25,802 | 1,319 | 5.11% |
| 2023 | 25,702 | 916 | 3.56% |
| 2024 | 22,957 | 1,359 | 5.92% |
| 2025 | 21,846 | 746 | 3.41% |
| 2026 | 9,599 | 385 | 4.01% |

#### By Month
| Month | Total | Events | Base Rate |
|-------|-------|--------|-----------|
| 1 | 20,834 | 1,089 | 5.23% |
| 2 | 18,751 | 1,174 | 6.26% |
| 3 | 19,297 | 1,458 | 7.56% |
| 4 | 17,550 | 1,117 | 6.36% |
| 5 | 20,617 | 1,406 | 6.82% |
| 6 | 19,091 | 997 | 5.22% |
| 7 | 18,809 | 954 | 5.07% |
| 8 | 17,925 | 970 | 5.41% |
| 9 | 18,194 | 1,029 | 5.66% |
| 10 | 18,082 | 1,099 | 6.08% |
| 11 | 17,478 | 895 | 5.12% |
| 12 | 19,603 | 840 | 4.29% |

#### By Day of Week
| Day | Total | Events | Base Rate |
|-----|-------|--------|-----------|
| Mon | 47,010 | 2,576 | 5.48% |
| Tue | 47,127 | 2,560 | 5.43% |
| Wed | 46,984 | 2,672 | 5.69% |
| Thu | 46,314 | 2,531 | 5.46% |
| Fri | 590 | 58 | 9.83% |

### 3.3 `range_next` Distribution (Next-Day High-Low Range)
- P10: 1.47%
- P25: 1.95%
- P50: 2.72%
- P75: 3.87%
- P90: 5.44%
- P95: 6.84%
- P99: 10.85%

Threshold exceedance:
- > 1%: 222,313 (98.27%)
- > 2%: 165,837 (73.30%)
- > 3%: 96,025 (42.45%)
- > 4%: 52,330 (23.13%)
- > 5%: 28,971 (12.81%)
- > 6%: 16,901 (7.47%)
- > 8%: 6,678 (2.95%)
- >10%: 2,985 (1.32%)

### 3.4 `fwd_return_1d` Distribution (Next-Day Return)
- P 1: -5.51%
- P 5: -3.07%
- P10: -2.10%
- P25: -0.84%
- P50: 0.00%
- P75: 0.90%
- P90: 2.35%
- P95: 3.52%
- P99: 6.36%

Positive days: 42.3%
Avg return: 0.080%

### 3.5 Joint Distribution (range_next × fwd_return_1d)
| Condition | Count | % of Total | Avg Return |
|-----------|-------|-----------|------------|
| range>5%                            |  28,971 |  12.81% |  +0.520% |
| bullish                             |  95,767 |  42.33% |  +1.711% |
| range>5% + bullish (label_5pct)     |  13,028 |   5.76% |  +4.405% |
| range>5% + bearish                  |  11,058 |   4.89% |  -3.827% |
| range<2% + bullish                  |  26,684 |  11.80% |  +0.670% |
| range<2% + bearish                  |  23,489 |  10.38% |  -0.663% |
---
## 4. Feature Analysis
### 4.1 Feature Importance Ranking
| Rank | Feature | Importance | Weight | Cumulative | Category |
|------|---------|------------|--------|------------|----------|
Feature importance table not available: single positional indexer is out-of-bounds

### 4.3 Feature Correlation with Target
| Rank | Feature | Correlation | | Correlation | Feature | Rank |
|------|---------|------------|-|------------|---------|------|
| 1 | kc_width             | +0.2219 | | -0.0009 | intra_macd_std       | 74 |
| 2 | range_pct            | +0.2088 | | +0.0011 | vix_change           | 73 |
| 3 | dc_width             | +0.1718 | | -0.0035 | delivery_delta       | 72 |
| 4 | bb_width             | +0.1354 | | -0.0042 | eom                  | 71 |
| 5 | vix_ma_5             | +0.1328 | | -0.0064 | is_thursday          | 70 |
| 6 | vix_close            | +0.1319 | | +0.0070 | skew_20              | 69 |
| 7 | hv_10                | +0.1287 | | +0.0073 | is_quarter_end       | 68 |
| 8 | hv_20                | +0.1230 | | +0.0119 | vix_ma_5_r           | 67 |
| 9 | hv_30                | +0.1174 | | +0.0122 | is_month_end         | 66 |
| 10 | vix_ma_20            | +0.1145 | | +0.0132 | fi                   | 65 |

### 4.4 Feature Group Statistics
| Group | Count | Avg Importance | Top Feature |
|-------|-------|---------------|-------------|
| Base Technical (45)                 | 45 | — | — |
| Range (1)                           |  1 | — | — |
| Calendar (5)                        |  5 | — | — |
| VIX / Fear (9)                      |  9 | — | — |
| Delivery / Institutional (4)        |  4 | — | — |
| Intraday Multi-timeframe (10)       | 10 | — | — |
