"""
Option Backtest v3: Enter ATM CALL on bullish engulfing, hold to fixed expiry
Uses calendar-based expiry dates (next Thu for weekly, last Thu for monthly)
Resamples pre-2021 trades to match option timestamps
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, bisect
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

CH=55; MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def compute_atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),
                  abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def next_thursday(dt):
    """Return next Thursday at 15:30 IST"""
    d=dt.date()
    days_ahead=(3-d.weekday())%7  # Thu=3
    if days_ahead==0: days_ahead=7  # If Thursday, go to next Thursday
    from datetime import time,timedelta
    nd=d+timedelta(days=days_ahead)
    return pd.Timestamp(nd, tz="Asia/Kolkata").replace(hour=15,minute=30,second=0)

def last_thursday(dt):
    """Return last Thursday of the month at 15:30 IST"""
    import calendar
    d=dt.date()
    # Last day of month
    _,last_day=calendar.monthrange(d.year,d.month)
    last_date=dt.replace(day=last_day)
    # Walk backwards to find Thursday
    while last_date.weekday()!=3: last_date-=pd.Timedelta(days=1)
    return pd.Timestamp(last_date.date(),tz="Asia/Kolkata").replace(hour=15,minute=30,second=0)

print("Connecting to DuckDB...")
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')

# Build sorted list of option timestamps for nearest-match lookup
res=con.execute("SELECT DISTINCT timestamp FROM options_data WHERE option_type='CALL'").fetchdf()
opt_ts_list=sorted(res["timestamp"].values)

def find_nearest_ts(target):
    """Find nearest option timestamp <= target (returns pandas Timestamp naive)"""
    import datetime as dtlib
    target=target.to_pydatetime().replace(tzinfo=None)
    i=bisect.bisect_right(opt_ts_list,target)
    if i==0: ts=opt_ts_list[0]
    else: ts=opt_ts_list[i-1]
    # Convert numpy datetime64 to pandas Timestamp
    if hasattr(ts,'__str__'):
        return pd.Timestamp(ts)
    return ts

def get_atm_call_price(ts, expiry_flag, expiry_code=None):
    """Get ATM CALL close at timestamp. If expiry_code None, try code 1."""
    ts_str=ts.strftime("%Y-%m-%d %H:%M:%S")
    ec_clause=f"AND expiry_code={expiry_code}" if expiry_code else "AND expiry_code=1"
    r=con.execute(f"""
        SELECT close, strike, expiry_code FROM options_data 
        WHERE timestamp = '{ts_str}'::TIMESTAMP
          AND option_type='CALL' AND expiry_flag='{expiry_flag}'
          {ec_clause} AND atm_distance=0
        LIMIT 1
    """).fetchone()
    if r: return r
    # Try nearby - use timestamp subtraction instead of EPOCH
    r=con.execute(f"""
        SELECT close, strike, expiry_code FROM options_data 
        WHERE option_type='CALL' AND expiry_flag='{expiry_flag}'
          {ec_clause} AND atm_distance=0
        ORDER BY ABS(EXTRACT(EPOCH FROM timestamp)-EXTRACT(EPOCH FROM TIMESTAMP '{ts_str}'))
        LIMIT 1
    """).fetchone()
    return r

def get_price_near(timestamp, expiry_flag, expiry_code):
    """Get price near a specific timestamp, with tolerance"""
    ts_str=timestamp.strftime("%Y-%m-%d %H:%M:%S")
    r=con.execute(f"""
        SELECT close FROM options_data 
        WHERE option_type='CALL' AND expiry_flag='{expiry_flag}'
          AND expiry_code={expiry_code} AND atm_distance=0
          AND ABS(EXTRACT(EPOCH FROM timestamp)-EXTRACT(EPOCH FROM TIMESTAMP '{ts_str}')) < 600
        ORDER BY ABS(EXTRACT(EPOCH FROM timestamp)-EXTRACT(EPOCH FROM TIMESTAMP '{ts_str}'))
        LIMIT 1
    """).fetchone()
    return r[0] if r else None

# Test
test_ts=pd.Timestamp("2024-01-15 09:30:00+05:30")
print(f"Test: {test_ts}")
r=get_atm_call_price(test_ts,"WEEK",1)
print(f"  WEEKLY CALL: close={r[0]}, strike={r[1]}, code={r[2]}")
r=get_atm_call_price(test_ts,"MONTH",1)
print(f"  MONTHLY CALL: close={r[0]}, strike={r[1]}, code={r[2]}")
print(f"  Next Thu: {next_thursday(test_ts)}")
print(f"  Last Thu this month: {last_thursday(test_ts)}")

# ====== BUILD TRADES ON SPOT ======
print("\nLoading NIFTY50 data...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
atr5=compute_atr(m5)
m5_epoch=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()

print("Building trades...")
rows=[]
body=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; r_close=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (r_close.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]: continue
    if body.iloc[i] < body.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
    lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
    t_ep=ts.asm8.view("int64"); idx=np.searchsorted(m5_epoch,t_ep,side="right")
    if idx>=len(m5["close"]): continue
    bi=idx
    while bi<len(m5["close"]) and m5["close"].iloc[bi]<=lv: bi+=1
    if bi>=len(m5["close"])-1: continue
    ri=bi+1
    while ri<len(m5["close"]):
        if m5["low"].iloc[ri]<lv and m5["close"].iloc[ri]>lv and pd.Series(m5["datetime"]).dt.time.iloc[ri]<CUT: break
        ri+=1
    if ri>=len(m5["close"]): continue
    entry_dt=m5["datetime"].iloc[ri]
    rows.append({"entry_dt":entry_dt,"yr":ts.year,"mo":ts.month})

trades=pd.DataFrame(rows)
print(f"Total trades: {len(trades)} (pre-2021 trades will be skipped if no option data)")

# ====== OPTION BACKTEST: HOLD TO CALENDAR EXPIRY ======
print("\n=== OPTION BACKTEST: HOLD TO CALENDAR EXPIRY ===")
opt_results=[]
skipped=0; no_entry=0; no_exit=0; no_code=0

for idx,row in trades.iterrows():
    if (idx+1)%200==0: print(f"  {idx+1}/{len(trades)}...",flush=True)
    ed=row["entry_dt"]  # tz-aware
    # Skip if before options data (2021-06-14)
    if ed.to_pydatetime().replace(tzinfo=None)<pd.Timestamp("2021-06-14 09:15:00"): skipped+=1; continue
    
    # Find nearest option timestamp <= entry
    ts_entry=find_nearest_ts(ed)
    
    # === WEEKLY ===
    expiry_dt_w=next_thursday(ed)
    # Need >2 DTE
    if (expiry_dt_w-ed).total_seconds()<172800: continue  # skip if <2 days
    
    # Enter: find ATM CALL for WEEKLY at entry
    ep_r=get_atm_call_price(ts_entry,"WEEK",1)
    if ep_r is None:
        # Try any available code
        ep_r=get_atm_call_price(ts_entry,"WEEK",None)
        if ep_r is None: no_entry+=1; continue
    ep_w=float(ep_r[0]); stk_w=float(ep_r[1]); code_w=int(ep_r[2])
    
    # Exit: price near expiry
    xp_w=get_price_near(expiry_dt_w,"WEEK",code_w)
    if xp_w is None:
        # Try +1 day tolerance
        xp_w=get_price_near(expiry_dt_w+pd.Timedelta(days=1),"WEEK",code_w)
    if xp_w is None: no_exit+=1; continue
    
    pnl_w=round(xp_w-ep_w,1)
    
    # === MONTHLY ===
    expiry_dt_m=last_thursday(ed)
    if (expiry_dt_m-ed).total_seconds()<604800:  # <7 DTE
        # Use next month's last Thursday
        from calendar import monthrange
        m=ed.month+1 if ed.month<12 else 1; y=ed.year+1 if m==1 else ed.year
        _,ld=monthrange(y,m); ld_dt=ed.replace(year=y,month=m,day=ld)
        while ld_dt.weekday()!=3: ld_dt-=pd.Timedelta(days=1)
        expiry_dt_m=pd.Timestamp(ld_dt.date(),tz="Asia/Kolkata").replace(hour=15,minute=30,second=0)
    
    ep_r_m=get_atm_call_price(ts_entry,"MONTH",1)
    if ep_r_m is None:
        ep_r_m=get_atm_call_price(ts_entry,"MONTH",None)
        if ep_r_m is None: pnl_m=None; month_data=None
        else:
            ep_m=float(ep_r_m[0]); code_m=int(ep_r_m[2])
            xp_m=get_price_near(expiry_dt_m,"MONTH",code_m)
            if xp_m is None: xp_m=get_price_near(expiry_dt_m+pd.Timedelta(days=1),"MONTH",code_m)
            pnl_m=round(xp_m-ep_m,1) if xp_m else None; month_data=(ep_m,xp_m,code_m)
    else:
        ep_m=float(ep_r_m[0]); code_m=int(ep_r_m[2])
        xp_m=get_price_near(expiry_dt_m,"MONTH",code_m)
        if xp_m is None: xp_m=get_price_near(expiry_dt_m+pd.Timedelta(days=1),"MONTH",code_m)
        pnl_m=round(xp_m-ep_m,1) if xp_m else None; month_data=(ep_m,xp_m,code_m)
    
    opt_results.append({
        "yr":row["yr"],"mo":row["mo"],"entry_ts":ts_entry,
        "week_code":code_w,"week_entry":ep_w,"week_exit":xp_w,"week_pnl":pnl_w,
        "week_expiry":expiry_dt_w,"week_strike":stk_w,
        "month_pnl":pnl_m,"month_entry":ep_m if month_data else None,
        "month_exit":xp_m if month_data else None,
        "month_expiry":expiry_dt_m,
    })

df_opt=pd.DataFrame(opt_results)
print(f"\nTrades: {len(df_opt)} (skipped={skipped}, no_entry={no_entry}, no_exit={no_exit})")

for col,label in [("week_pnl","WEEKLY CALL"),("month_pnl","MONTHLY CALL")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    p=sub[col]; wr=(p>0).mean(); net=p.sum()
    aw=p[p>0].mean() if (p>0).sum()>0 else 0; al=p[p<0].mean() if (p<0).sum()>0 else 0
    wl=aw/abs(al) if al!=0 else 999; pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0; calmar=net/mdd if mdd>0 else 999
    print(f"\n{label}: {len(p)} trades, Net={net:+,.0f}, WR={wr:.1%}, W/L={wl:.1f}x, "
          f"PF={pf:.1f}x, MDD={mdd:,.0f}, Sharpe={sharpe:.3f}, Calmar={calmar:.1f}x")
    print(f"  Avg entry={sub[col.replace('pnl','entry')].mean():.1f}, "
          f"Avg exit={sub[col.replace('pnl','exit')].mean():.1f}" if col.replace('pnl','entry') in sub.columns else "")

# ====== YEARLY ======
print("\n"+"="*100)
for col,label in [("week_pnl","WEEKLY")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    print(f"\nYEARLY - {label}")
    yr=sub.groupby("yr")[col].agg(["count","sum",lambda x:(x>0).mean()])
    yr.columns=["Trades","Net","WR"]
    print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    for y in sorted(yr.index):
        r=yr.loc[y]; print(f"{int(y):>6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

print("\n"+"="*100)
for col,label in [("month_pnl","MONTHLY")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    print(f"\nYEARLY - {label}")
    yr=sub.groupby("yr")[col].agg(["count","sum",lambda x:(x>0).mean()])
    yr.columns=["Trades","Net","WR"]
    print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    for y in sorted(yr.index):
        r=yr.loc[y]; print(f"{int(y):>6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

# ====== MONTHLY ======
print("\n"+"="*100)
print("MONTHLY BREAKDOWN")
print("="*100)
for col,label in [("week_pnl","WEEKLY"),("month_pnl","MONTHLY")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    print(f"\n--- {label} ---")
    mo=sub.groupby("mo")[col].agg(["count","sum",lambda x:(x>0).mean()])
    mo.columns=["Trades","Net","WR"]
    print(f"{'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    for m in range(1,13):
        if m in mo.index:
            r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

con.close()
print("\nDone!")
