# PIVOT BREAKOUT STRATEGY — RULES DOCUMENT

## 1. TIMEFRAMES
- **Signal Timeframe**: 15-minute candles
- **Entry Timeframe**: 1-minute candles
- **Pivot Calculation**: Daily (Traditional)

## 2. PIVOT LEVELS (Daily Traditional)
Calculated **once per day** from previous day's High, Low, Close:

```
P  = (H + L + C) / 3
R1 = 2P - L
S1 = 2P - H
R2 = P + (H - L)
S2 = P - (H - L)
```

## 3. TRIGGER CANDLE (15-minute)
A 15-minute candle becomes a **trigger** when:

| Trigger Type | Condition |
|---|---|
| **Bullish Trigger** (Long) | 15-min candle **High touches or crosses above R1** |
| **Bearish Trigger** (Short) | 15-min candle **Low touches or crosses below S1** |

Once triggered, store these reference levels from that candle:

| Level | Long Trigger | Short Trigger |
|---|---|---|
| **Breakout Level** | High of trigger candle | Low of trigger candle |
| **Stop Loss** | Low of trigger candle | High of trigger candle |
| **Entry Price** | High + 1 to 2 points | Low - 1 to 2 points |
| **Take Profit** | Entry + 2 × (Entry - SL) | Entry - 2 × (SL - Entry) |

## 4. ENTRY (1-minute)

| Side | Entry Condition |
|---|---|
| **Long** | A 1-minute candle's price **moves above** the trigger candle's **High** → Enter at **High + 1-2 pts** |
| **Short** | A 1-minute candle's price **moves below** the trigger candle's **Low** → Enter at **Low - 1-2 pts** |

- Only **one trade per trigger**
- Entry happens on the **same day** as the trigger
- No re-entry if SL hits

## 5. EXIT

| Exit Type | Long | Short |
|---|---|---|
| **Stop Loss** | Low of trigger candle | High of trigger candle |
| **Take Profit** | Entry + 2 × (Entry - SL) | Entry - 2 × (SL - Entry) |
| **Time Exit** | End of trading day (3:30 PM) if neither hit | |

- **Risk:Reward = 1:2** (fixed)

## 6. CHARGES (Per Trade Round-Trip)

| Charge | Rate |
|---|---|
| **Brokerage** | ₹10 per order (₹20 round-trip) |
| **STT** | 0.1% on sell side |
| **Exchange Charges** | 0.003% of turnover |
| **SEBI Charges** | 0.0001% of turnover |
| **Stamp Duty** | 0.003% on buy side |
| **GST** | 18% on (brokerage + exchange charges) |

## 7. BACKTESTING METHOD
- **Tool**: Python (pandas + numpy) — vectorized, not Backtrader
- **Data**: Nifty 50 stocks, 15-min + 1-min, Oct 2016 to Jun 2026
- **Per stock**: Daily pivots → 15-min triggers → 1-min fills → SL/TP tracking
- **Output**: Trade book + metrics (win rate, Sharpe, drawdown, profit factor, avg R)
