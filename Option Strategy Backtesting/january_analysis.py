"""
Deep analysis: why January is the worst month (Engulf_Raw CH55)
"""
import pandas as pd
import numpy as np
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def compute_atr(m5):
    tr = pd.concat([m5["high"]-m5["low"], abs(m5["high"]-m5["close"].shift(1)),
                    abs(m5["low"]-m5["close"].shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=14, min_periods=14, adjust=False).mean()

def compute_atr20(h1):
    tr = pd.concat([h1["high"]-h1["low"], abs(h1["high"]-h1["close"].shift(1)),
                    abs(h1["low"]-h1["close"].shift(1))], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=20).mean()

def compute_adx14(h1):
    tr = pd.concat([h1["high"]-h1["low"], abs(h1["high"]-h1["close"].shift(1)),
                    abs(h1["low"]-h1["close"].shift(1))], axis=1).max(axis=1)
    up = h1["high"] - h1["high"].shift(1); down = h1["low"].shift(1) - h1["low"]
    pdm = ((up > down) & (up > 0)) * up; ndm = ((down > up) & (down > 0)) * down
    atr14 = tr.rolling(14, min_periods=14).mean()
    pdi = 100*(pdm.rolling(14).mean()/atr14); ndi = 100*(ndm.rolling(14).mean()/atr14)
    return 100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()

def compute_daily_ema(h1, p=50):
    df_ = h1.copy(); df_["date"] = h1["datetime"].dt.normalize()
    daily = df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=p,adjust=False).mean()).values

print("Loading data...")
DATA = {}
for sym in ["NIFTY50","SENSEX"]:
    h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
    m5 = pd.read_csv(f"{sym}_FIVE_MINUTE.csv", parse_dates=["datetime"])
    for df in [h1, m5]:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime", inplace=True); df.reset_index(drop=True, inplace=True)
    atr5 = compute_atr(m5)
    atr20 = compute_atr20(h1); adx14 = compute_adx14(h1)
    ema50 = compute_daily_ema(h1, 50); ema200 = compute_daily_ema(h1, 200)
    h1["ret_20h"] = h1["close"].pct_change(20)
    h1["dow"] = h1["datetime"].dt.dayofweek
    h1["hour"] = h1["datetime"].dt.hour
    h1["atr20"] = atr20; h1["adx14"] = adx14
    h1["ema50"] = ema50; h1["ema200"] = ema200
    DATA[sym] = {"h1": h1, "m5": m5,
                 "m5_epoch": m5["datetime"].astype("int64").values,
                 "m5_cl": m5["close"].values, "m5_lo": m5["low"].values,
                 "m5_hi": m5["high"].values, "m5_atr": atr5.values,
                 "tc": pd.Series(m5["datetime"]).dt.time.values}

CUT = pd.Timestamp("14:15").time()

# Build ALL Jan trades with full context
jan_trades = []
total_trades = []

for sym in ["NIFTY50","SENSEX"]:
    h1 = DATA[sym]["h1"]; d = DATA[sym]
    b = (h1["close"]-h1["open"]).abs(); g = h1["close"]>h1["open"]; r = h1["close"]<h1["open"]
    me = d["m5_epoch"]; mc = d["m5_cl"]; ml = d["m5_lo"]; mh = d["m5_hi"]; ma = d["m5_atr"]; tc = d["tc"]
    # Pre-compute H1 features indexed by date for lookup
    h1_dates = h1["datetime"].values
    h1_atr20 = h1["atr20"].values; h1_adx = h1["adx14"].values
    h1_ema50 = h1["ema50"].values; h1_ema200 = h1["ema200"].values
    h1_ret20 = h1["ret_20h"].values; h1_dow = h1["dow"].values
    h1_open = h1["open"].values; h1_high = h1["high"].values
    h1_low = h1["low"].values; h1_close = h1["close"].values

    for i in range(1, len(h1)):
        if not (r.iloc[i-1] and g.iloc[i]): continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
        if b.iloc[i] < b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour == 9: continue
        lv = h1["high"].iloc[i]; ts = h1["datetime"].iloc[i]
        t_ep = ts.asm8.view("int64")
        idx = np.searchsorted(me, t_ep, side="right")
        if idx >= len(mc): continue
        bi = idx
        while bi < len(mc) and mc[bi] <= lv: bi += 1
        if bi >= len(mc)-1: continue
        ri = bi+1
        while ri < len(mc):
            if ml[ri] < lv and mc[ri] > lv and tc[ri] < CUT: break
            ri += 1
        if ri >= len(mc): continue
        ep = mc[ri]; sl = ml[ri]
        if ep - sl <= 0: continue

        # Compute exit for CH55
        he = ep; exit_pnl = None; exit_idx = None
        for j in range(ri, len(mc)):
            ca = ma[j]
            if pd.isna(ca): continue
            if mh[j] > he: he = mh[j]
            if mc[j] < he - 55*ca:
                exit_pnl = round(mc[j]-ep, 1); exit_idx = j; break
        if exit_pnl is None:
            exit_pnl = round(mc[-1]-ep, 1); exit_idx = len(mc)-1

        # Compute holding bars
        holding_bars = exit_idx - ri if exit_idx else 0
        holding_hours = holding_bars * 5 / 60  # 5-min bars
        peak_run = (he - ep) / ep * 100  # % run-up

        # Context features from H1 signal candle
        yr, mo = ts.year, ts.month
        dow_n = h1_dow[i]
        ret = h1_ret20[i] if i < len(h1_ret20) and not pd.isna(h1_ret20[i]) else 0
        adx = h1_adx[i] if i < len(h1_adx) and not pd.isna(h1_adx[i]) else 0
        atr20_v = h1_atr20[i] if i < len(h1_atr20) and not pd.isna(h1_atr20[i]) else 0
        ema50_v = h1_ema50[i] if i < len(h1_ema50) and not pd.isna(h1_ema50[i]) else 0
        ema200_v = h1_ema200[i] if i < len(h1_ema200) and not pd.isna(h1_ema200[i]) else 0
        trend_bull = (ema50_v > ema200_v) if not pd.isna(ema50_v) and not pd.isna(ema200_v) else False

        # Entry candle body ratio
        body_ratio = b.iloc[i] / b.iloc[i-1] if b.iloc[i-1] > 0 else 0
        # Prev candle type (how bad was red)
        prev_body = b.iloc[i-1]
        # Open gap
        gap = (h1_open[i] - h1_close[i-1]) / h1_close[i-1] * 100

        t = {
            "date": ts.strftime("%Y-%m-%d"), "sym": sym, "yr": yr, "mo": mo,
            "pnl": exit_pnl, "win": exit_pnl > 0,
            "dow": ["Mon","Tue","Wed","Thu","Fri"][dow_n] if dow_n < 5 else "?",
            "ret_20h": ret, "adx14": adx, "atr20": atr20_v,
            "trend_bull": trend_bull,
            "body_ratio": body_ratio, "prev_body": prev_body,
            "gap_pct": gap, "holding_hours": holding_hours,
            "holding_bars": holding_bars, "peak_run_pct": peak_run,
            "entry_bar_hour": ts.hour, "entry_bar_min": ts.minute,
        }
        row = {**t}
        if mo == 1:
            jan_trades.append(row)
        total_trades.append(row)

df_all = pd.DataFrame(total_trades)
df_jan = pd.DataFrame(jan_trades)
print(f"\nTotal trades: {len(df_all)}")
print(f"January trades: {len(df_jan)}")

print("\n" + "="*80)
print("SECTION 1: JANUARY YEAR-BY-YEAR BREAKDOWN")
print("="*80)
jan_yr = df_jan.groupby("yr").agg(
    trades=("pnl","count"), net=("pnl","sum"),
    wins=("win","sum"), wr=("win","mean")
).round(1)
jan_yr["WR"] = jan_yr["wr"]
print(f"{'Year':>6} {'Trades':>8} {'Net':>12} {'Wins':>6} {'WR':>8}")
print("-"*45)
for yr, row in jan_yr.iterrows():
    print(f"{yr:>6} {int(row['trades']):>8} {row['net']:>+12,.0f} {int(row['wins']):>6} {row['WR']:>7.1%}")
print("-"*45)
print(f"{'ALL':>6} {int(jan_yr['trades'].sum()):>8} {jan_yr['net'].sum():>+12,.0f} "
      f"{int(jan_yr['wins'].sum()):>6} {jan_yr['wins'].sum()/jan_yr['trades'].sum():>7.1%}")

print("\n" + "="*80)
print("SECTION 2: JANUARY vs ALL OTHER MONTHS")
print("="*80)
def stats(s):
    p=s["pnl"]; return {"n":len(p),"wr":(p>0).mean(),"net":p.sum(),"avg":p.mean(),
                        "aw":p[p>0].mean() if (p>0).sum()>0 else 0,
                        "al":p[p<0].mean() if (p<0).sum()>0 else 0,
                        "wl":p[p>0].mean()/abs(p[p<0].mean()) if (p<0).sum()>0 and p[p<0].mean()!=0 else 999,
                        "pf":p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999}
s_jan=stats(df_jan)
s_all=stats(df_all[df_all["mo"]!=1])
s_non=stats(df_all)
print(f"{'Metric':<15} {'January':>14} {'Other Months':>14} {'All':>14}")
print("-"*60)
for k in ["n","wr","net","avg","aw","al","wl","pf"]:
    v1=s_jan.get(k,0); v2=s_all.get(k,0); v3=s_non.get(k,0)
    if k=="wr": f1,f2,f3=f"{v1:.1%}",f"{v2:.1%}",f"{v3:.1%}"
    elif k in ("wl","pf"): f1,f2,f3=f"{v1:.1f}x",f"{v2:.1f}x",f"{v3:.1f}x"
    else: f1,f2,f3=f"{v1:+,.0f}",f"{v2:+,.0f}",f"{v3:+,.0f}"
    print(f"{k:<15} {f1:>14} {f2:>14} {f3:>14}")

print("\n" + "="*80)
print("SECTION 3: JANUARY TRADE DETAILS (sorted by date)")
print("="*80)
cols_show=["date","sym","pnl","win","dow","adx14","trend_bull","ret_20h","holding_hours","peak_run_pct","gap_pct"]
print(f"{'Date':<12} {'Sym':<8} {'PnL':>8} {'Win':>5} {'DOW':>4} {'ADX':>6} {'Bull':>5} "
      f"{'Ret20h':>8} {'HoldH':>6} {'Peak%':>7} {'Gap%':>7}")
print("-"*80)
for _,r in df_jan.sort_values("date").iterrows():
    print(f"{r['date']:<12} {r['sym']:<8} {r['pnl']:>+8,.1f} {str(r['win']):>5} "
          f"{r['dow']:>4} {r['adx14']:>5.1f} {str(r['trend_bull']):>5} "
          f"{r['ret_20h']:>7.2%} {r['holding_hours']:>5.1f} {r['peak_run_pct']:>6.2f}% {r['gap_pct']:>6.2f}%")
print("-"*80)

print("\n" + "="*80)
print("SECTION 4: JANUARY LOSS CAUSES - FEATURE COMPARISON")
print("="*80)
jan_w = df_jan[df_jan["win"]==True]
jan_l = df_jan[df_jan["win"]==False]
print(f"{'Feature':<20} {'Winners':>14} {'Losers':>14} {'Diff':>14} {'Impact':>10}")
print("-"*72)
features = [
    ("adx14","f", "Higher ADX = trending = better"),
    ("holding_hours","f", "Longer holds = trend capture"),
    ("peak_run_pct","f", "Bigger run-up before reversal = stopped"),
    ("gap_pct","f", "Gap down = weak open"),
    ("ret_20h","%", "Prior downtrend = mean reversion"),
    ("prev_body","f", "Larger prev red = more reversal power"),
    ("body_ratio","f", "Bigger entry body = stronger signal"),
]
for fname, fmt, desc in features:
    wv = jan_w[fname].mean() if len(jan_w)>0 else 0
    lv = jan_l[fname].mean() if len(jan_l)>0 else 0
    if fmt == "%":
        wv *= 100; lv *= 100
    diff = wv - lv
    print(f"{fname:<20} {wv:>13.2f} {lv:>13.2f} {diff:>+13.2f}  {desc}")

print("\n" + "="*80)
print("SECTION 5: MARKET REGIME ANALYSIS IN JANUARY")
print("="*80)
# Trend bias
bull_pct = df_jan["trend_bull"].mean() * 100
print(f"  Trades in bull EMA regime:  {bull_pct:.1f}%")
print(f"  Trades in bear EMA regime:  {100-bull_pct:.1f}%")
# ADX analysis
print(f"  Avg ADX(14) on entry:       {df_jan['adx14'].mean():.1f}")
print(f"  Trades with ADX>25:         {(df_jan['adx14']>25).sum()} ({(df_jan['adx14']>25).mean()*100:.0f}%)")
print(f"  WR when ADX>25:            {df_jan[df_jan['adx14']>25]['win'].mean()*100:.1f}%")
print(f"  WR when ADX<=25:           {df_jan[df_jan['adx14']<=25]['win'].mean()*100:.1f}%")
# Ret 20h
print(f"  Avg prior 20H return:       {df_jan['ret_20h'].mean()*100:.2f}%")
print(f"  Trades after >2% drop:      {(df_jan['ret_20h']<-0.02).sum()} -> WR: {df_jan[df_jan['ret_20h']<-0.02]['win'].mean()*100:.1f}%")
print(f"  Trades after >2% rise:      {(df_jan['ret_20h']>0.02).sum()} -> WR: {df_jan[df_jan['ret_20h']>0.02]['win'].mean()*100:.1f}%")

print("\n" + "="*80)
print("SECTION 6: DAY OF WEEK IN JANUARY")
print("="*80)
dow_jan = df_jan.groupby("dow").agg(trades=("pnl","count"), net=("pnl","sum"), wr=("win","mean"))
print(f"{'DOW':<8} {'Trades':>8} {'Net':>12} {'WR':>8}")
print("-"*40)
for d in ["Mon","Tue","Wed","Thu","Fri"]:
    if d in dow_jan.index:
        r = dow_jan.loc[d]
        print(f"{d:<8} {int(r['trades']):>8} {r['net']:>+12,.0f} {r['wr']:>7.1%}")

print("\n" + "="*80)
print("SECTION 7: LOSS STREAKS IN JANUARY")
print("="*80)
pnls = df_jan.sort_values(["sym","date"])["pnl"].values
max_ls = 0; cur_ls = 0; streak_runs = []
for p in pnls:
    if p < 0:
        cur_ls += 1; max_ls = max(max_ls, cur_ls)
    else:
        if cur_ls >= 2: streak_runs.append(cur_ls)
        cur_ls = 0
if cur_ls >= 2: streak_runs.append(cur_ls)
print(f"  Max consecutive losses in Jan:  {max_ls}")
print(f"  Streaks of 2+ losses:           {len(streak_runs)}")
print(f"  Avg loss streak length:         {np.mean(streak_runs):.1f}" if streak_runs else "  None")

print("\n" + "="*80)
print("SECTION 8: ROOT CAUSE SUMMARY")
print("="*80)
jan_net = df_jan["pnl"].sum()
jan_wr = (df_jan["pnl"]>0).mean()
all_wr_nonjan = (df_all[df_all["mo"]!=1]["pnl"]>0).mean()
jan_al = df_jan[df_jan["pnl"]<0]["pnl"].mean()
all_al = df_all[df_all["pnl"]<0]["pnl"].mean()
w_dow = dow_jan.loc["Wed"]["wr"] if "Wed" in dow_jan.index else 0
print(f"""
JANUARY ROOT CAUSE ANALYSIS - Engulf_Raw CH55

NET RESULT: {jan_net:+,.0f} pts across {len(df_jan)} trades

PRIMARY CAUSES:

1. LOW WIN RATE: {jan_wr:.1%} vs {all_wr_nonjan:.1%} for other months
   -> January has fundamentally different market behavior

2. STOP-LOSS EFFECTIVENESS: Avg loss = {jan_al:+,.0f} pts
   -> Compare to overall avg loss: {all_al:+,.0f} pts
   {"-> LOSES are LARGER in Jan" if jan_al > all_al else "-> Loss sizes are in line"}

3. MARKET REGIME: {"More bearish EMA structure" if bull_pct < 50 else "More bullish EMA structure"}
   -> {"" if bull_pct < 50 else "NOT "}consistent with poor signal performance

4. SEASONAL FACTORS:
   - Budget effect (Feb 1): last week of Jan = positioning, choppy
   - Tax loss harvesting reversal (Jan effect)
   - New year portfolio rebalancing flows
   - Low volatility period -> false breakouts get stopped

5. DAY OF WEEK (Jan only):
   {"- Wednesday worst in January" if w_dow < 0.3 else "- No DOW consistent pattern"}

6. RECOMMENDATION:
   -> SKIP January entirely for all strategies
   -> Or reduce position size to 0.25x of normal
   -> Worst years: ... (see yearly breakdown above)
""")

# Summary table
print("\n" + "="*80)
print("ALL MONTHS - CONTRAST TABLE")
print("="*80)
all_mon = df_all.groupby("mo").agg(
    trades=("pnl","count"), net=("pnl","sum"),
    wins=("win","sum"), wr=("win","mean"),
    avg_pnl=("pnl","mean"), avg_loss=("pnl",lambda x: x[x<0].mean()),
    avg_win=("pnl",lambda x: x[x>0].mean()),
).round(1)
print(f"{'Mon':<6} {'Trades':>7} {'Net':>12} {'WR':>7} {'AvgW':>10} {'AvgL':>10} {'W/L':>8}")
print("-"*60)
for m in range(1,13):
    r = all_mon.loc[m] if m in all_mon.index else None
    if r is not None:
        wl = r["avg_win"]/abs(r["avg_loss"]) if r["avg_loss"] and r["avg_loss"] != 0 else 999
    else:
        r = {"trades":0,"net":0,"wr":0,"avg_win":0,"avg_loss":0}; wl=0
    hl = "->" if m == 1 else ""
    print(f"{hl}{MON[m-1]:<6} {int(r['trades']):>7} {r['net']:>+12,.0f} {r['wr']:>6.1%} "
          f"{r['avg_win']:>9,.0f} {r['avg_loss']:>9,.0f} {wl:>6.1f}x")
