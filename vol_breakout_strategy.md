VOLATILITY BREAKOUT — Swing Momentum Strategy
==============================================

1. STRATEGY RULES
-----------------
Universe:   Top 20 NSE stocks by volume-sensitivity Sharpe
Entry:      When daily return > +1% AND volume > 1.5-2x 20-day average
            -> Go LONG at CLOSE of the signal day
Hold:       5 trading days (exit at close on day 5)
Direction:  LONG ONLY (short signals lose money — avg -2.48% on ATGL)
Sectors:    PSU/Defense, Metals, Financials, Technology

2. STOCK UNIVERSE (Top 20 by Sharpe)
-------------------------------------
Rank  Stock          Params   Trades  5d Ret   WR     Sharpe
 1    BAJAJFINSV     20/2.0     69    +2.49%  66.7%   3.25
 2    IRFC           30/1.5    112    +4.43%  58.9%   3.14
 3    HSCL           20/2.0     82    +3.15%  59.8%   3.10
 4    JINDALSTEL     20/2.0     66    +2.35%  57.6%   2.56
 5    KPITTECH       20/2.0     96    +2.82%  56.3%   2.55
 6    VOLTAS         20/2.0     91    +1.47%  60.4%   2.40
 7    BLUESTARCO     15/1.5    158    +1.79%  61.4%   2.33
 8    M&M            20/2.0     71    +1.09%  60.6%   2.28
 9    ANGELONE       10/1.5    143    +2.80%  58.7%   2.23
10    PCJEWELLER     20/2.0     50    +3.64%  50.0%   2.21
11    LUXIND         30/1.5    141    +2.65%  53.2%   2.16
12    BEL            20/2.0     95    +1.79%  62.1%   2.09
13    HAL            10/1.5    186    +1.59%  59.7%   2.09
14    ALKEM          20/2.0     94    +1.33%  60.6%   2.09
15    SUVEN          15/1.5    259    +3.18%  51.7%   2.07
16    SHREECEM       20/2.0     63    +1.18%  55.6%   2.04
17    ADANIGREEN     20/2.0    159    +3.43%  54.7%   2.01
18    ASHOKA         20/2.0    167    +2.41%  55.7%   2.01
19    APLAPOLLO      15/1.5    186    +1.49%  59.7%   2.00
20    TATACONSUM     20/2.0    103    +1.30%  57.3%   1.98

3. PERFORMANCE (Portfolio, 2016-2026)
--------------------------------------
Total trades:           2,393
Win rate:               57.6%
Avg return (5-day):     +2.36%
Median return:          +0.86%
Sharpe ratio:           2.13
Annualized return:      119%
Max drawdown:           -31% (single trade)
Best trade:             +55.2%
Worst trade:            -31.0%

By market regime:
  Bull (SENSEX > MA50):  Sharpe 2.17, WR 58.2%
  Bear (SENSEX < MA50):  Sharpe 1.85, WR 54.9%

By year:
  2020 (COVID):  +5.0% avg, 61.0% WR  <- BEST
  2017:          +2.4% avg, 64.1% WR
  2021:          +2.6% avg, 62.0% WR
  2024:          +2.8% avg, 60.3% WR
  2019:          +1.9% avg, 55.4% WR
  2022:          +1.8% avg, 54.9% WR
  2023:          +2.3% avg, 54.8% WR
  2026:          +1.8% avg, 51.4% WR
  2025:          +0.8% avg, 54.1% WR  <- WEAKEST (low vol)
  2018:          +0.7% avg, 50.7% WR

4. POSITION SIZING & RISK
--------------------------
- 5% per stock (equal weight, 20 positions max)
- 5.9 concurrent positions on average
- Rebalance: when a position exits (5 days), enter the next signal
- Max sector exposure: 25%
- Reduce allocation if SENSEX < MA200 (bear filter)

5. MONITORING
--------------
Track daily for each stock in universe:
- Daily return > +1%?
- Volume > (lookback avg * vol_mult)?
If both true -> enter at close, hold 5 days

6. COMPARISON: GAP FADE (Existing)
-----------------------------------
Gap fade generates INTRADAY income (same-day, mean-reversion).
Vol breakout generates SWING profits (5-day, momentum).
They are COMPLEMENTARY — can run both with 50/50 capital split.
