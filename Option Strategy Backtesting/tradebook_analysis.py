"""
Trade book deep analysis: duplicates, gaps, anomalies, distribution
"""
import duckdb, pandas as pd, numpy as np, os, time, warnings, calendar
from datetime import timedelta, datetime
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

# ====================================================================
# 1. NIFTY OPTION TRADE BOOK (TP30_Max7d + Prem<130)
# ====================================================================
print("="*70)
print("NIFTY OPTION TRADE BOOK ANALYSIS (TP30_Max7d + Prem<130)")
print("="*70)

# Build trades
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
tr5=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
atr5=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
me=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()

trades=[]; b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; rr=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
    if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
    lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
    idx=np.searchsorted(me,ts.asm8.view("int64"),side="right")
    if idx>=len(m5["close"]): continue
    bi=idx
    while bi<len(m5["close"]) and m5["close"].iloc[bi]<=lv: bi+=1
    if bi>=len(m5["close"])-1: continue
    ri=bi+1
    while ri<len(m5["close"]):
        if m5["low"].iloc[ri]<lv and m5["close"].iloc[ri]>lv and pd.Series(m5["datetime"]).dt.time.iloc[ri]<CUT: break
        ri+=1
    if ri>=len(m5["close"]): continue
    ed=m5["datetime"].iloc[ri]; ep_=m5["close"].iloc[ri]
    if ep_-m5["low"].iloc[ri]<=0: continue
    he=ep_
    for j in range(ri,len(m5["close"])):
        ca=atr5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"entry_px":ep_,"exit_px":m5["close"].iloc[j],
                           "yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); break
trades_all=pd.DataFrame(trades)
trades_all["ed_naive"]=trades_all["entry_dt"].dt.tz_localize(None)
trades_all["entry_hour"]=trades_all["entry_dt"].dt.hour
trades_all["entry_min"]=trades_all["entry_dt"].dt.minute

# Filter option-era
trades_pre=trades_all[trades_all["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Total NIFTY engulfing signals: {len(trades_all)}")
print(f"Option-era (2021+): {len(trades_pre)}")

# Check for duplicate signal timestamps
dup_entries = trades_pre[trades_pre.duplicated(subset=["ed_naive"], keep=False)]
print(f"\nDuplicate entry timestamps: {len(dup_entries)}")
if len(dup_entries)>0:
    print(dup_entries[["ed_naive","entry_px","yr"]].to_string())

# Check for same-entry-different-price
dup_same_ts = trades_pre.groupby("ed_naive").filter(lambda x: x["entry_px"].nunique()>1)
if len(dup_same_ts)>0:
    print(f"\nSame timestamp, DIFFERENT entry price: {len(dup_same_ts)}")
else:
    print("All trades at same timestamp have same entry price - OK")

# Check entry price monotonic (should increase with time for same index)
print(f"\nEntry price range: {trades_pre['entry_px'].min():.1f} to {trades_pre['entry_px'].max():.1f}")
print(f"Exit price range: {trades_pre['exit_px'].min():.1f} to {trades_pre['exit_px'].max():.1f}")

# Check for negative PnL that's impossibly large
trades_pre["pnl"]=trades_pre["exit_px"]-trades_pre["entry_px"]
print(f"\nPnL range: {trades_pre['pnl'].min():+.1f} to {trades_pre['pnl'].max():+.1f}")
print(f"Trades with PnL=0 (entry==exit): {(trades_pre['pnl']==0).sum()}")

# Check CH55 exit timing validity
trades_pre["exit_hour"]=trades_pre["exit_dt"].dt.hour
trades_pre["exit_min"]=trades_pre["exit_dt"].dt.minute
trades_pre["trade_duration_min"]=(trades_pre["exit_dt"]-trades_pre["entry_dt"]).dt.total_seconds()/60

print(f"\nTrade duration: min={trades_pre['trade_duration_min'].min():.0f}m, "
      f"max={trades_pre['trade_duration_min'].max():.0f}m, "
      f"avg={trades_pre['trade_duration_min'].mean():.0f}m")

# Check for trades that exit in < 5 minutes (immediate trigger)
quick_exits = trades_pre[trades_pre["trade_duration_min"]<5]
print(f"Trades exiting in < 5 min: {len(quick_exits)}")
if len(quick_exits)>0:
    for _,r in quick_exits.iterrows():
        print(f"  Entry: {r['entry_dt']} Exit: {r['exit_dt']} PnL: {r['pnl']:+.1f}")

# Check Friday entries (known bad)
print(f"\nEntry day distribution:")
day_names=["Mon","Tue","Wed","Thu","Fri"]
for d in range(5):
    sub=trades_pre[trades_pre["weekday"]==d]
    print(f"  {day_names[d]}: {len(sub)} trades, avg PnL={sub['pnl'].mean():+.1f}")

# Check entry hour distribution
print(f"\nEntry hour distribution:")
for h in sorted(trades_pre["entry_hour"].unique()):
    sub=trades_pre[trades_pre["entry_hour"]==h]
    print(f"  {h}:00: {len(sub)} trades, avg PnL={sub['pnl'].mean():+.1f}")

# ====================================================================
# 2. OPTION TRADE BOOK (the actual executed trades with option prices)
# ====================================================================
print("\n" + "="*70)
print("OPTION EXECUTED TRADE ANALYSIS (Prem<130 filter + TP30_Max7d)")
print("="*70)

# Load option data
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
df_opt=con.execute("SELECT timestamp, close FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp").fetchdf()
con.close()
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
cl_arr=df_opt["close"].values.astype(float)

def lookup(ts):
    i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
    if i<=0: return 0,cl_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1]
    return i-1,cl_arr[i-1]

ep_vals=np.array([lookup(ed)[1] for ed in trades_pre["ed_naive"]])
si_vals=np.array([lookup(ed)[0] for ed in trades_pre["ed_naive"]])

def exit_tp_maxd(s_idx,tp,max_d):
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_d*86400*1e6),"us")
    ep=cl_arr[s_idx]
    for i in range(s_idx+1,min(s_idx+3000,len(cl_arr))):
        if ts_arr[i]>end_ns: return cl_arr[i], ts_arr[i]
        if cl_arr[i]-ep>=tp: return cl_arr[i], ts_arr[i]
    return None, None

# Build executed trade book
tbook=[]
for i in range(len(ep_vals)):
    if ep_vals[i]>130: continue
    xp,xts=exit_tp_maxd(si_vals[i],30,7)
    if xp is None: continue
    tbook.append({"trade":len(tbook)+1,
        "entry_dt":trades_pre.iloc[i]["ed_naive"],
        "exit_dt":pd.Timestamp(xts).tz_localize(None) if isinstance(xts,np.datetime64) else xts,
        "entry_px":ep_vals[i],"exit_px":xp,
        "pnl":round(xp-ep_vals[i],1),
        "spot_entry":trades_pre.iloc[i]["entry_px"],
        "spot_exit":trades_pre.iloc[i]["exit_px"],
        "spot_pnl":round(trades_pre.iloc[i]["exit_px"]-trades_pre.iloc[i]["entry_px"],1),
        "yr":trades_pre.iloc[i]["yr"],"mo":trades_pre.iloc[i]["mo"],
        "weekday":trades_pre.iloc[i]["weekday"]})

otb=pd.DataFrame(tbook)
otb["opt_return_pct"]=round(otb["pnl"]/otb["entry_px"]*100,1)
otb["spot_return_pct"]=round(otb["spot_pnl"]/otb["spot_entry"]*100,2)
otb["dur_hours"]=round((otb["exit_dt"]-otb["entry_dt"]).dt.total_seconds()/3600,1)

print(f"Total option trades executed: {len(otb)}")
print(f"Option PnL: {otb['pnl'].sum():+,.0f} pts (Rs {otb['pnl'].sum()*50:+,.0f})")
print(f"Spot PnL (same period): {otb['spot_pnl'].sum():+,.0f} pts (Rs {otb['spot_pnl'].sum()*50:+,.0f})")

# Check duplicates in trade book
dup_entry=otb[otb.duplicated(subset=["entry_dt"],keep=False)]
print(f"\nDuplicate entry timestamps in executed trades: {len(dup_entry)}")

dup_full=otb[otb.duplicated(keep=False)]
print(f"Fully duplicate rows: {len(dup_full)}")

# Check for zero option PnL (entry==exit)
zero_pnl=otb[otb["pnl"]==0]
print(f"Option trades with PnL=0: {len(zero_pnl)}")
if len(zero_pnl)>0:
    for _,r in zero_pnl.head(5).iterrows():
        print(f"  Trade {r['trade']}: entry={r['entry_dt']}, exit={r['exit_dt']}")

# Check trades where exit <= entry (reversed)
reversed_trades=otb[otb["exit_dt"]<=otb["entry_dt"]]
print(f"Trades with exit <= entry time: {len(reversed_trades)}")

# Option PnL vs Spot PnL comparison
print(f"\nOption vs Spot correlation:")
print(f"  Trades where option wins & spot wins: {((otb['pnl']>0)&(otb['spot_pnl']>0)).sum()}")
print(f"  Trades where option loses & spot loses: {((otb['pnl']<=0)&(otb['spot_pnl']<=0)).sum()}")
print(f"  Trades where option wins but spot loses: {((otb['pnl']>0)&(otb['spot_pnl']<=0)).sum()}")
print(f"  Trades where option loses but spot wins: {((otb['pnl']<=0)&(otb['spot_pnl']>0)).sum()}")

# If option PnL > spot PnL, leverage working
better_than_spot=(otb["pnl"]*50) > (otb["spot_pnl"]*50)
print(f"  Option better than spot (Rs): {better_than_spot.sum()} / {len(otb)}")

# Return analysis
print(f"\nOption return stats (%):")
print(f"  Avg return: {otb['opt_return_pct'].mean():+.1f}%")
print(f"  Best: {otb['opt_return_pct'].max():+.1f}%")
print(f"  Worst: {otb['opt_return_pct'].min():+.1f}%")
print(f"  > 50% return: {(otb['opt_return_pct']>50).sum()} trades")
print(f"  > 100% return: {(otb['opt_return_pct']>100).sum()} trades")

# Premium analysis
print(f"\nPremium analysis:")
print(f"  Avg entry premium: {otb['entry_px'].mean():.1f} pts (Rs {otb['entry_px'].mean()*50:,.0f})")
print(f"  Premium range: {otb['entry_px'].min():.1f} to {otb['entry_px'].max():.1f} pts")
print(f"  Exit price range: {otb['exit_px'].min():.1f} to {otb['exit_px'].max():.1f} pts")

# Time to exit analysis
print(f"\nDuration to exit:")
print(f"  Avg: {otb['dur_hours'].mean():.1f}h ({otb['dur_hours'].mean()/24:.2f} days)")
print(f"  Min: {otb['dur_hours'].min():.1f}h")
print(f"  Max: {otb['dur_hours'].max():.1f}h ({otb['dur_hours'].max()/24:.1f} days)")
print(f"  Exits on same day: {(otb['dur_hours']<=6).sum()} trades")
print(f"  Exits within 1 day: {((otb['dur_hours']>6)&(otb['dur_hours']<=24)).sum()} trades")
print(f"  Exits after 7 days: {(otb['dur_hours']>168).sum()} trades")

# Check if any trades hit the max 7-day time stop vs TP
# TP30 exit: hit profit target of +30 pts
tp_exits=otb[otb["pnl"]>=30]
time_exits=otb[otb["pnl"]<30]
print(f"\nExit reason estimate:")
print(f"  Likely TP hit (pnl >= 30): {len(tp_exits)} ({len(tp_exits)/len(otb)*100:.0f}%)")
print(f"  Likely time stop (pnl < 30): {len(time_exits)} ({len(time_exits)/len(otb)*100:.0f}%)")
print(f"  Of time stops, winners: {(time_exits['pnl']>0).sum()} losers: {(time_exits['pnl']<=0).sum()}")

# Time stop analysis
print(f"\nTime stop analysis (trades with pnl < 30):")
ts=time_exits
print(f"  Avg pnl: {ts['pnl'].mean():+.1f}")
print(f"  Avg duration: {ts['dur_hours'].mean():.1f}h ({ts['dur_hours'].mean()/24:.1f}d)")
print(f"  Max duration: {ts['dur_hours'].max()/24:.1f}d")
for _,r in ts.nsmallest(5,"pnl").iterrows():
    print(f"  Worst: #{r['trade']} pnl={r['pnl']:+.1f} dur={r['dur_hours']:.0f}h prem={r['entry_px']:.1f}")

# ====================================================================
# 3. SENSEX & BANKNIFTY TRADE BOOK ANALYSIS
# ====================================================================
print("\n"+"="*70)
print("SENSEX & BANKNIFTY SPOT TRADE BOOK ANALYSIS")
print("="*70)

for name, f1, f5 in [("SENSEX","SENSEX_ONE_HOUR.csv","SENSEX_FIVE_MINUTE.csv"),
                      ("BANKNIFTY","BANKNIFTY_ONE_HOUR.csv","BANKNIFTY_FIVE_MINUTE.csv")]:
    print(f"\n--- {name} ---")
    h1=pd.read_csv(f1,parse_dates=["datetime"])
    m5=pd.read_csv(f5,parse_dates=["datetime"])
    for d in [h1,m5]:
        d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
    
    tr5=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    atr5=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
    me=m5["datetime"].astype("int64").values
    
    tb=[]
    b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; rr=h1["close"]<h1["open"]
    for i in range(1,len(h1)):
        if not (rr.iloc[i-1] and g.iloc[i]): continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
        lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
        idx=np.searchsorted(me,ts.asm8.view("int64"),side="right")
        if idx>=len(m5["close"]): continue
        bi=idx
        while bi<len(m5["close"]) and m5["close"].iloc[bi]<=lv: bi+=1
        if bi>=len(m5["close"])-1: continue
        ri=bi+1
        while ri<len(m5["close"]):
            if m5["low"].iloc[ri]<lv and m5["close"].iloc[ri]>lv and pd.Series(m5["datetime"]).dt.time.iloc[ri]<CUT: break
            ri+=1
        if ri>=len(m5["close"]): continue
        ed=m5["datetime"].iloc[ri]; ep_=m5["close"].iloc[ri]
        if ep_-m5["low"].iloc[ri]<=0: continue
        he=ep_
        for j in range(ri,len(m5["close"])):
            ca=atr5.iloc[j]
            if pd.isna(ca): continue
            if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
            if m5["close"].iloc[j]<he-55*ca:
                tb.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],
                    "entry":ep_,"exit":m5["close"].iloc[j],"pnl":round(m5["close"].iloc[j]-ep_,1),
                    "yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); break
    
    tb=pd.DataFrame(tb)
    tb["ed_naive"]=tb["entry_dt"].dt.tz_localize(None)
    tb["dur_min"]=(tb["exit_dt"]-tb["entry_dt"]).dt.total_seconds()/60
    
    dup_ts=tb[tb.duplicated(subset=["ed_naive"],keep=False)]
    zero_pnl=(tb["pnl"]==0).sum()
    neg_dur=(tb["dur_min"]<=0).sum()
    
    print(f"Trades: {len(tb)}")
    print(f"Duplicate entry timestamps: {len(dup_ts)}")
    print(f"Zero PnL trades: {zero_pnl}")
    print(f"Non-positive duration: {neg_dur}")
    print(f"PnL range: {tb['pnl'].min():+.1f} to {tb['pnl'].max():+.1f}")
    print(f"Duration range: {tb['dur_min'].min():.0f} to {tb['dur_min'].max()/60:.1f}h")
    print(f"Entry price range: {tb['entry'].min():.1f} to {tb['entry'].max():.1f}")
    
    # Check if PnLs are reasonable (not exceeding index range)
    if len(tb)>0:
        max_reasonable = (tb["entry"].max() - tb["entry"].min()) * 0.5
        outliers = tb[tb["pnl"].abs() > max_reasonable]
        print(f"Outliers (|pnl| > 50% of index range): {len(outliers)}")
        if len(outliers)>0:
            for _,r in outliers.head(3).iterrows():
                print(f"  PnL={r['pnl']:+.1f} entry={r['entry']:.1f} exit={r['exit']:.1f}")

print("\nDone!")
