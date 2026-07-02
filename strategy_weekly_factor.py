"""Approach 1: Weekly multi-factor rank — long top decile, rebalance Friday close
Combines RS, delivery, momentum, low-vol into a composite weekly z-score."""
import duckdb, pandas as pd, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005; TOTAL_POS=110000

def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

def calc_metrics(s, n=52):
    if len(s)<5 or s.std()==0: return 0,0,0,0,0
    cagr=(1+s/100).prod()**(n/len(s))-1 if len(s)>0 else 0
    sh=s.mean()/s.std()*np.sqrt(n) if s.std()>0 else 0
    wr=(s>0).mean()
    dd=((1+s/100).cumprod()/(1+s/100).cumprod().cummax()-1).min()*100
    return cagr*100, sh, wr*100, dd, s.mean()

print('Loading data...')
con = duckdb.connect(str(DB), read_only=True)

# Load weekly data: every Friday's close + features
fs = con.execute("""
    SELECT symbol, datetime, close, ret_5d, ret_10d, ret_20d, hv_20, rsi_14, bb_pct_b,
           atr_14, volume, sma_20, ema_20, adx
    FROM feature_store WHERE timeframe='1day'
    ORDER BY datetime
""").fetchdf()
fs_dt = pd.to_datetime(fs['datetime'])
fs['date'] = fs_dt.dt.tz_localize(None).dt.normalize()
fs['dow'] = fs_dt.dt.tz_localize(None).dt.dayofweek

# RS features
ms = con.execute("""
    SELECT symbol, datetime, rs_vs_market, rs_vs_sector, rs_ratio_market, rs_ratio_sector,
           rs_momentum_10, rs_momentum_20
    FROM market_structure WHERE timeframe='1day'
    ORDER BY datetime
""").fetchdf()
ms_dt = pd.to_datetime(ms['datetime'])
ms['date'] = ms_dt.dt.tz_localize(None).dt.normalize()

# Delivery
dv = con.execute("""
    SELECT symbol, date, delivery_pct FROM delivery_data ORDER BY symbol, date
""").fetchdf()
dv['date'] = pd.to_datetime(dv['date'])

con.close()

# Merge
df = fs.merge(ms, on=['symbol','date'], how='left')
df = df.merge(dv, on=['symbol','date'], how='left')
df = df.sort_values(['symbol','date']).reset_index(drop=True)

# Fill forward delivery
df['delivery_pct'] = df.groupby('symbol')['delivery_pct'].ffill().fillna(0)
for c in ['rs_vs_market','rs_vs_sector','rs_ratio_market','rs_ratio_sector','rs_momentum_10','rs_momentum_20']:
    df[c] = df[c].fillna(0)

# Filter penny stocks (avg close < 50 over last 2 yrs)
recent = df[df['date'] >= '2024-06-24'].groupby('symbol')['close'].mean()
penny_syms = set(recent[recent < 50].index)
df = df[~df['symbol'].isin(penny_syms)].copy()
print(f'Penny stocks removed: {len(penny_syms)} ({sorted(penny_syms)})')
print(f'Remaining symbols: {df["symbol"].nunique()}')

# Weekly rebalance: pick at Friday close
df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week.astype(int)
df['year'] = pd.to_datetime(df['date']).dt.year
df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)

# Compute weekly factors
print('Computing factors...')
df['rs_rank'] = df.groupby('date')['rs_vs_market'].rank(pct=True)
df['rs_mom_rank'] = df.groupby('date')['rs_momentum_20'].rank(pct=True)
df['delivery_rank'] = df.groupby('date')['delivery_pct'].rank(pct=True)
df['hv_rank'] = df.groupby('date')['hv_20'].rank(pct=True)  # lower = better (low vol)
df['mom_rank'] = df.groupby('date')['ret_20d'].rank(pct=True)
df['adx_rank'] = df.groupby('date')['adx'].rank(pct=True)

# Composite score: equal-weight z-scores
for f in ['rs_rank','rs_mom_rank','delivery_rank','mom_rank']:
    df[f'{f}_z'] = df.groupby('date')[f].transform(lambda x: (x - x.mean()) / x.std())

# Low vol: invert hv_rank (lower vol = higher score)
df['hv_rank_z'] = -df.groupby('date')['hv_rank'].transform(lambda x: (x - x.mean()) / x.std())

df['composite_z'] = (df['rs_rank_z'] + df['rs_mom_rank_z'] + df['delivery_rank_z'] +
                     df['mom_rank_z'] + df['hv_rank_z']) / 5

# Weekly backtest: rebalance every Friday
# Pick Friday of each week, rank by composite_z, long top decile
print('Running weekly backtest...')
weeks = sorted(df['year_week'].unique())
bt = []; prev_hold = None; prev_date = None

for wk in weeks:
    wk_data = df[df['year_week'] == wk].copy()
    if len(wk_data) < 20: continue

    # Get Friday (or last available day of the week)
    wk_data = wk_data.sort_values('date')
    friday = wk_data['date'].iloc[-1]

    # Rebalance on Friday
    reb_day = wk_data[wk_data['date'] == friday]
    if len(reb_day) < 10: continue

    # Rank by composite score
    reb_day = reb_day.sort_values('composite_z', ascending=False)
    reb_day['rank'] = range(len(reb_day))
    N = max(len(reb_day) // 10, 1)
    buys = set(reb_day.head(N)['symbol'].tolist())

    # Forward returns: merge this week's picks with next week's close
    wk_idx = weeks.index(wk)
    if wk_idx >= len(weeks) - 1:
        continue  # skip last week, no forward data

    next_wk = weeks[wk_idx + 1]
    next_data = df[df['year_week'] == next_wk].sort_values('date')
    if len(next_data) == 0: continue
    next_friday = next_data['date'].iloc[-1]
    next_close = next_data[next_data['date'] == next_friday][['symbol','close']].drop_duplicates('symbol')
    next_close = next_close.set_index('symbol')['close']

    this_close = reb_day[['symbol','close']].drop_duplicates('symbol').set_index('symbol')['close']
    merged = this_close.to_frame('c_this').join(next_close.to_frame('c_next'), how='inner')
    merged = merged[merged.index.isin(buys)]
    if len(merged) == 0: continue
    merged['ret'] = (merged['c_next'] / merged['c_this'] - 1) * 100
    avg_ret = merged['ret'].mean()

    # Turnover
    if prev_hold is not None and prev_date is not None:
        weeks_held = (pd.Timestamp(friday) - pd.Timestamp(prev_date)).days / 7
        if weeks_held >= 1:
            # Full rebalance
            ch = len(buys - prev_hold) + len(prev_hold - buys)
            to = ch / max(len(buys | prev_hold), 1)
        else:
            to = 0.0
    else:
        to = 1.0
    to = min(to, 1.0)

    n_hold = len(merged)
    cost = cost_rt(TOTAL_POS / n_hold) * to * 100
    net = avg_ret - cost

    bt.append({'date':friday, 'n_hold':n_hold, 'gross':avg_ret, 'to':to, 'cost':cost, 'net':net,
               'top_decile_N':N, 'total_symbols':len(reb_day)})
    prev_hold = buys
    prev_date = friday

bt = pd.DataFrame(bt)
print(f'Trades: {len(bt)}, Avg holding: {bt["n_hold"].mean():.0f} symbols')

# Metrics
g = bt['gross'].dropna(); n = bt['net'].dropna()
gc, gs, gw, gdd, gm = calc_metrics(g)
nc, ns, nw, ndd, nm = calc_metrics(n)
print(f'\n{"="*55}')
print(f'WEEKLY MULTI-FACTOR (top decile, rebalance Fri)')
print(f'{"="*55}')
print(f'{"":20s} {"Gross":>12s} {"Net":>12s}')
print(f'{"CAGR":20s} {gc:>+11.1f}% {nc:>+11.1f}%')
print(f'{"Sharpe":20s} {gs:>11.2f} {ns:>11.2f}')
print(f'{"WinRate":20s} {gw:>10.1f}% {nw:>10.1f}%')
print(f'{"MaxDD":20s} {gdd:>10.1f}% {ndd:>10.1f}%')
print(f'{"Mean weekly ret":20s} {gm:>+10.3f}% {nm:>+10.3f}%')
print(f'{"Avg turnover":20s} {bt["to"].mean():>10.1%} {"":>12s}')
print(f'Avg cost/trade: {cost_rt(TOTAL_POS/bt["n_hold"].mean())*100:.3f}%')

# Also test: only top 5 symbols (concentrated)
print(f'\n{"="*55}')
print(f'WEEKLY TOP-5 (concentrated)')
print(f'{"="*55}')
bt5 = []; prev5 = None; prev_d5 = None
for wk in weeks:
    wk_data = df[df['year_week'] == wk].copy()
    if len(wk_data) < 20: continue
    wk_data = wk_data.sort_values('date')
    friday = wk_data['date'].iloc[-1]
    reb_day = wk_data[wk_data['date'] == friday]
    if len(reb_day) < 10: continue
    reb_day = reb_day.sort_values('composite_z', ascending=False)
    buys5 = set(reb_day.head(5)['symbol'].tolist())

    wk_idx = weeks.index(wk)
    if wk_idx >= len(weeks) - 1:
        continue
    else:
        next_wk = weeks[wk_idx + 1]
        next_data = df[df['year_week'] == next_wk].sort_values('date')
        if len(next_data) == 0: continue
        nf = next_data['date'].iloc[-1]
        next_close = next_data[next_data['date'] == nf][['symbol','close']].drop_duplicates('symbol').set_index('symbol')['close']
    this_close = reb_day[['symbol','close']].drop_duplicates('symbol').set_index('symbol')['close']
    merged5 = this_close.to_frame('c_this').join(next_close.to_frame('c_next'), how='inner')
    merged5 = merged5[merged5.index.isin(buys5)]
    if len(merged5) == 0: continue
    merged5['ret'] = (merged5['c_next'] / merged5['c_this'] - 1) * 100
    r5 = merged5['ret'].mean()
    to5 = 1.0 if prev5 is None else (len(buys5-prev5)+len(prev5-buys5))/max(len(buys5|prev5),1)
    cost5 = cost_rt(TOTAL_POS/len(merged5)) * to5 * 100
    bt5.append({'date':friday,'ret':r5,'net':r5-cost5,'to':to5,'n':len(merged5)})
    prev5 = buys5; prev_d5 = friday

bt5f = pd.DataFrame(bt5)
g5 = bt5f['ret'].dropna(); n5 = bt5f['net'].dropna()
gc5, gs5, gw5, gdd5, gm5 = calc_metrics(g5)
nc5, ns5, nw5, ndd5, nm5 = calc_metrics(n5)
print(f'{"CAGR":20s} {gc5:>+11.1f}% {nc5:>+11.1f}%')
print(f'{"Sharpe":20s} {gs5:>11.2f} {ns5:>11.2f}')
print(f'{"WinRate":20s} {gw5:>10.1f}% {nw5:>10.1f}%')
print(f'{"MaxDD":20s} {gdd5:>10.1f}% {ndd5:>10.1f}%')
print(f'{"Mean weekly ret":20s} {gm5:>+10.3f}% {nm5:>+10.3f}%')

# Save
bt.to_csv(BASE/'strategy_weekly_factor.csv', index=False)
if len(bt5) > 0:
    bt5f.to_csv(BASE/'strategy_weekly_top5.csv', index=False)
print(f'\nSaved to {BASE}')
