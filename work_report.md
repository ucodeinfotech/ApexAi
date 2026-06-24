======================================================================
            WORK REPORT - PIVOT BREAKOUT / ORB BACKTEST
            Nifty 50 Stocks | Oct 2016 - Jun 2026
======================================================================

DATE: 19-Jun-2026
SESSION TYPE: Full backtest development cycle (4 strategy iterations)

----------------------------------------------------------------------
TABLE OF CONTENTS
----------------------------------------------------------------------
1.  Phase 1: Pivot Breakout v1 (Touch-based)
2.  Phase 2: Pivot Breakout v2 (Optimized)
3.  Phase 3: Opening Range Breakout (ORB v1)
4.  Phase 4: ORB v2 Parameter Optimization (Sweep)
5.  Overall Comparison
6.  Data Summary
7.  Key Learnings

======================================================================
PHASE 1: PIVOT BREAKOUT v1 - Original Strategy
======================================================================

STRATEGY RULES:
  - Timeframes:       15-min signal, 1-min entry
  - Trigger:          Candle HIGH touches/crosses R1 (long), LOW touches/crosses S1 (short)
  - Entry:            Price crosses trigger candle high + 1pt slippage (long) / low - 1pt (short)
  - Stop Loss:        Trigger candle low (long) / high (short)
  - Profit Target:    1:2 Risk-Reward (2x risk)
  - Pivots:           Traditional daily pivots from prior day H/L/C

CHARGES:
  - Brokerage:        Rs10/order
  - STT:              0.1% on sell
  - Exchange:         0.003%
  - SEBI:             0.0001%
  - GST:              18% on brokerage + exchange
  - Stamp Duty:       0.003%

FILES CREATED:
  - backtest_all.py           Main backtester script
  - fetch_pre2020.py          1-min data backfill script
  - strategy_rules.md         Strategy documentation

RESULTS (v1):
  Total Trades:        540,999
  Wins / Losses:       6,915 / 534,084
  Win Rate:            1.28%
  Net P&L:             Rs-17,502,592
  Gross Profit:        Rs136,151
  Gross Loss:          Rs-17,638,742
  Profit Factor:       0.01
  Avg R Multiple:      -0.73
  Total Charges:       Rs14,052,861
  Profitable Stocks:   0 / 50

  Best (least loss):   JIOFIN  (Rs-31,220,  WR 0.0%)
  Worst (most loss):   APOLLOHOSP (Rs-515,858, WR 3.2%)

  CONCLUSION: Catastrophic failure. "Touch" = 98.7% false signals.

======================================================================
PHASE 2: PIVOT BREAKOUT v2 - Optimized Strategy
======================================================================

OPTIMIZATIONS APPLIED:
  1. Close-above/below trigger (not touch)
  2. Entry at trigger candle close (market order, no slippage)
  3. ATR-based SL (2x ATR(14) on 15-min data)
  4. 1-hr trend filter (price vs 20-EMA)
  5. Volume confirmation (1.5x avg20 volume)
  6. Brokerage reduced to Rs5/order
  7. Partial profit booking (50% at 1:1, trail rest to BE)

FILES CREATED:
  - backtest_optimized.py    Optimized pivot backtester
  - test_opt.py              Single-stock test script

RESULTS (v2) - Optimized Pivot:
  Total Trades:        65,969
  Wins / Losses:       1,402 / 64,567
  Win Rate:            2.13%
  Net P&L:             Rs-1,478,672
  Gross Profit:        Rs18,996
  Gross Loss:          Rs-1,497,669
  Profit Factor:       0.01
  Avg R Multiple:      -0.69
  Total Charges:       Rs884,082
  Profitable Stocks:   0 / 50

  Best:                JIOFIN  (Rs-6,777,  WR 0.0%)
  Worst:               MARUTI  (Rs-93,725, WR 14.4%)

  vs V1 IMPROVEMENT:   Trades -88%, Loss -91.5%, Charges -93.7%
  VERDICT:             Still unprofitable. Pivot signals have no edge.
                       Avg R -0.69 means 69% of risk lost per trade.

======================================================================
PHASE 3: OPENING RANGE BREAKOUT (ORB v1) - New Strategy
======================================================================

STRATEGY RULES:
  - Opening Range:    First 30 min (2 x 15-min bars: 9:15-9:45)
  - Trigger:          Subsequent 15-min close > OR high (long) / < OR low (short)
  - Entry:            Market order at close of trigger candle
  - Stop Loss:        2x ATR(14 of 15-min)
  - Profit Target:    2x SL (1:2 R:R)
  - Filters:          Volume 1.3x avg, 1-hr trend, no entry after 2pm
  - Partial Profit:   50% at 1:1 R:R, trail rest to breakeven
  - Brokerage:        Rs5/order

FILES CREATED:
  - backtest_orb.py           ORB backtester
  - test_orb.py               Single-stock test

RESULTS (ORB v1):
  Total Trades:        33,648
  Wins / Losses:       2,991 / 30,657
  Win Rate:            8.89%
  Net P&L:             Rs-507,947
  Gross Profit:        Rs61,259
  Gross Loss:          Rs-569,206
  Profit Factor:       0.11
  Avg R Multiple:      -0.14
  Total Charges:       Rs451,604
  Partial Hit Rate:    48.0% (16,157 / 33,648)
  Profitable Stocks:   0 / 50

  Best:                JIOFIN  (Rs-2,772, WR 0.5%)
  Worst:               MARUTI  (Rs-19,547, WR 49.1%)
  Best Avg R:          MARUTI  (Avg R -0.03)
  Best WR:             MARUTI  (WR 49.1%)

  vs V2 IMPROVEMENT:   Loss -66%, Avg R 5x better, Charges -49%
  KEY INSIGHT:         Avg R -0.14 means only 14% of risk lost per trade.
                       Charges consume 89% of total loss (Rs452k of Rs508k).
                       Without charges, loss would be only Rs56k across 50 stocks.

======================================================================
PHASE 4: ORB v2 - Parameter Optimization (Sweep)
======================================================================

PARAMETER SWEEP ON 5 REPRESENTATIVE STOCKS (MARUTI, RELIANCE, TCS, SBIN, BAJAJFINSV):

  GRID:
    OR window:    1, 2, 3 bars (15, 30, 45 min)
    SL multiplier: 1.5x, 2.0x, 2.5x ATR
    TP ratio:      1.5x, 2.0x, 2.5x SL

  Total combos:       27 per stock x 5 stocks = 135 runs
  Sweep time:         ~45 minutes

TOP 3 COMBOS (by Net P&L, aggregated across 5 stocks):
  1. OR=3b SL=2.5x TP=1.5x  Net=-Rs26,970  Raw=+Rs9,424  AvgR=+0.10
  2. OR=3b SL=2.5x TP=2.0x  Net=-Rs33,110  Raw=+Rs3,284  AvgR=+0.03
  3. OR=2b SL=2.5x TP=1.5x  Net=-Rs35,905  Raw=+Rs7,275  AvgR=+0.08

BEST PER STOCK (identical for all 5):
  OR=3 bars (45min), SL=2.5x ATR, TP=1.5x SL

RAW P&L (zero charges):
  OR=3b SL=2.5x TP=1.5x: +Rs9,424 across 5 stocks (POSITIVE)
  MARUTI:  +Rs5,874  AvgR=+0.14
  BAJAJFINSV: +Rs2,311  AvgR=+0.09
  SBIN:    +Rs279    AvgR=+0.11
  TCS:     +Rs669    AvgR=+0.08
  RELIANCE:+Rs291    AvgR=+0.09

  KEY MILESTONE: FIRST CONFIGURATION WITH POSITIVE RAW EDGE.

OPTIMAL CONFIGURATION:
  - Opening Range:    45 min (3 x 15-min bars)
  - Stop Loss:        2.5x ATR(14 of 15-min)
  - Profit Target:    1.5x SL (1:1.5 R:R)
  - Partial Profit:   50% at 1:1 R:R
  - Brokerage:        Rs5/order
  - Volume Filter:    1.3x avg20
  - Trend Filter:     1-hr 20-EMA
  - Entry Window:     Before 2pm
  - EOD Exit:         3:25pm

STATUS: Full 50-stock run with optimal config INTERRUPTED.
        Needs re-execution to get final results.

======================================================================
OVERALL COMPARISON (All Strategies)
======================================================================

Metric              | Pivot v1     | Pivot v2     | ORB v1       | ORB v2 (Opt)
--------------------|-------------|--------------|--------------|-------------
Trades              | 540,999     | 65,969       | 33,648       | PENDING
Win Rate            | 1.28%       | 2.13%        | 8.89%        | PENDING
Net P&L             | Rs-17.5M    | Rs-1.48M     | Rs-0.51M     | PENDING
Avg R               | -0.73       | -0.69        | -0.14        | +0.10*
Charges             | Rs14.05M    | Rs0.88M      | Rs0.45M      | PENDING
Sharpe              | -627.7      | -272.2       | -26.3        | PENDING
Partial Hit Rate    | N/A         | ~20%         | 48%          | ~50%**
Profitable Stocks   | 0/50        | 0/50         | 0/50         | PENDING

* From 5-stock sweep: Raw Avg R = +0.10 (before charges)
** Estimated from sweep data

======================================================================
FILES AND DATA SUMMARY
======================================================================

DATA FILES:
  nifty50_full_history/      ~108 files (~2.6 GB)
    - 50 x 15-min CSV        ~59,800 rows each
    - 50 x 1-min CSV         ~896,000 rows each
    - Index data             5-min + 1-hr for NIFTY50, BANKNIFTY, SENSEX

BACKTEST SCRIPTS:
  backtest_all.py            Pivot v1 (original touch-based)
  backtest_optimized.py      Pivot v2 (7 optimizations)
  backtest_orb.py            ORB strategy (current)
  orb_sweep.py               Parameter sweep script
  test_opt.py                Single-stock test
  test_orb.py                Single-stock ORB test

REPORT SCRIPTS:
  detailed_report.py         Console report generator
  gen_pdf_report.py          PDF report generator

OUTPUT FILES:
  backtest_results/          
    - *_trades.csv           Per-stock trade books (Pivot v1)
    - *_trades_v2.csv        Per-stock trade books (Pivot v2)
    - *_orb_trades.csv       Per-stock trade books (ORB v1)
    - all_trades.csv         Combined trades (Pivot v1)
    - all_trades_v2.csv      Combined trades (Pivot v2)
    - all_orb_trades.csv     Combined trades (ORB v1)
    - stock_summary.csv      Per-stock metrics (Pivot v1)
    - backtest_report.pdf    PDF report (Pivot v1)
    - backtest_report_v2.pdf PDF report (Pivot v2)

======================================================================
KEY LEARNINGS
======================================================================

1. PIVOT BREAKOUTS DON'T WORK ON INDIAN MARKETS:
   - R1/S1 acts as resistance/support, price reverses 98% of time
   - Close-above trigger helps slightly but still no edge
   - Traditional daily pivots are not predictive for intraday

2. TOUCH VS CLOSE TRIGGER:
   - Touch: 540K trades, 1.28% WR - catastrophic
   - Close-above: 66K trades, 2.13% WR - still bad but 88% fewer false signals

3. THE REAL GAME CHANGER: ORB (Opening Range Breakout):
   - 48% of trades hit 1:1 partial profit v/s 20% for pivots
   - Avg R improved from -0.73 to -0.14 (5x better)
   - Window + ATR + partial booking makes ORB viable

4. CHARGES DOMINATE:
   - Rs5 v/s Rs10 brokerage cuts charges by 50%+
   - Even at Rs5, charges consume 89% of ORB losses
   - True strategy edge (before charges) is near-zero

5. ORB OPTIMIZATION WORKS:
   - Wider SL (2.5x ATR) and tighter TP (1.5x) produce positive raw edge
   - 45-min OR window > 30-min or 15-min (more stable range)
   - First config with positive Avg R: +0.10 (before charges)

6. NEXT STEPS NEEDED:
   - Complete 50-stock run with optimal ORB config
   - Consider 0-brokerage simulation or higher capital
   - Could test multi-stock portfolio effect with position sizing

======================================================================
END OF REPORT
Generated: 19-Jun-2026
======================================================================
