"""
Option Backtest v2: Enter ATM CALL on bullish engulfing, hold to fixed expiry
Resamples pre-2021 trades to match option timestamps
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

CH=55; MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def compute_atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),
                  abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

print("Connecting to DuckDB...")
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')

print("Building expiry date map...")
# For each expiry_code + expiry_flag, find the last timestamp = expiry date
expiry_map=con.execute("""
    SELECT expiry_flag, expiry_code, 
           MAX(timestamp) as expiry_dt,
           MIN(timestamp) as first_seen
    FROM options_data 
    GROUP BY expiry_flag, expiry_code
    ORDER BY expiry_flag, expiry_dt
""").fetchdf()
print(expiry_map.to_string())

# Build lookup dict: (flag, code) -> expiry_dt
expiry_lookup={}
for _,r in expiry_map.iterrows():
    expiry_lookup[(r["expiry_flag"],r["expiry_code"])]=r["expiry_dt"]

# Build forward map: for each timestamp, which expiry_codes are available
# Pre-compute for fast lookup
print("\nPre-computing expiry availability...")
# Create a mapping table
con.execute("""
    CREATE OR REPLACE TABLE expiry_availability AS
    SELECT DISTINCT timestamp, expiry_flag, expiry_code
    FROM options_data 
    WHERE option_type='CALL'
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_ea_ts ON expiry_availability(timestamp)")

def get_next_expiry(ts, expiry_flag, min_days=3):
    """Find the nearest expiry after ts with at least min_days to expiry"""
    ts_r=ts.floor("5min")
    # Get available expiry_codes at this timestamp
    codes=con.execute(f"""
        SELECT DISTINCT expiry_code FROM expiry_availability
        WHERE timestamp = '{ts_r}' AND expiry_flag = '{expiry_flag}'
        ORDER BY expiry_code
    """).fetchdf()
    if len(codes)==0:
        # Try finding nearest timestamp
        codes=con.execute(f"""
            SELECT DISTINCT expiry_code FROM expiry_availability
            WHERE timestamp >= '{ts_r}' AND expiry_flag = '{expiry_flag}'
            ORDER BY expiry_code LIMIT 5
        """).fetchdf()
        if len(codes)==0: return None
    # For each code, check if expiry_dt > ts + min_days
    for _,r in codes.iterrows():
        ec=int(r["expiry_code"])
        ed=expiry_lookup.get((expiry_flag,ec))
        if ed is None: continue
        days_to_exp=(ed-ts).total_seconds()/86400
        if days_to_exp>=min_days:
            return ec,ed
    # Fallback: return first available
    ec=int(codes.iloc[0]["expiry_code"])
    return ec,expiry_lookup.get((expiry_flag,ec))

def get_price(ts, expiry_flag, expiry_code):
    """Get ATM CALL close price at timestamp"""
    ts_r=ts.floor("5min")
    r=con.execute(f"""
        SELECT close, strike FROM options_data 
        WHERE timestamp = '{ts_r}'
          AND option_type='CALL' AND expiry_flag='{expiry_flag}'
          AND expiry_code={expiry_code} AND atm_distance=0
        LIMIT 1
    """).fetchone()
    if r: return r[0],r[1]
    # Try finding closest timestamp
    r=con.execute(f"""
        SELECT close, strike, timestamp FROM options_data 
        WHERE option_type='CALL' AND expiry_flag='{expiry_flag}'
          AND expiry_code={expiry_code} AND atm_distance=0
          AND timestamp >= '{ts_r}'
        ORDER BY timestamp LIMIT 1
    """).fetchone()
    if r: return r[0],r[1]
    return None,None

def get_expiry_price(expiry_dt, expiry_flag, expiry_code):
    """Get ATM CALL price at/near expiry (last few bars)"""
    # Try the expiry date itself
    r=con.execute(f"""
        SELECT close FROM options_data 
        WHERE timestamp >= '{expiry_dt - pd.Timedelta(hours=2)}'::TIMESTAMP
          AND timestamp <= '{expiry_dt}'
          AND option_type='CALL' AND expiry_flag='{expiry_flag}'
          AND expiry_code={expiry_code} AND atm_distance=0
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    if r: return r[0]
    return None

print(f"\nTest: next WEEKLY expiry from 2024-01-15 09:30:00")
wec,wed=get_next_expiry(pd.Timestamp("2024-01-15 09:30:00"),"WEEK")
print(f"  WEEK: code={wec}, expiry={wed}, DTE={(wed-pd.Timestamp('2024-01-15 09:30:00')).days}")
ep,_=get_price(pd.Timestamp("2024-01-15 09:30:00"),"WEEK",wec)
print(f"  Entry price: {ep}")
xp=get_expiry_price(wed,"WEEK",wec)
print(f"  Exit price (near expiry): {xp}")

mec,med=get_next_expiry(pd.Timestamp("2024-01-15 09:30:00"),"MONTH")
print(f"  MONTH: code={mec}, expiry={med}, DTE={(med-pd.Timestamp('2024-01-15 09:30:00')).days}")

# ====== BUILD TRADES ON SPOT (same as before) ======
print("\nLoading NIFTY50 data...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
atr5=compute_atr(m5)
m5_epoch=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()

print("Building trades (including pre-2021 for option resampling)...")
rows=[]
body=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; r=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (r.iloc[i-1] and g.iloc[i]): continue
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
    ep=m5["close"].iloc[ri]; sl=m5["low"].iloc[ri]
    if ep-sl<=0: continue
    entry_dt=m5["datetime"].iloc[ri]
    rows.append({"entry_dt":entry_dt,"yr":ts.year,"mo":ts.month})

trades=pd.DataFrame(rows)
print(f"Total trades: {len(trades)}")

# ====== BACKTEST ON OPTIONS (HOLD TO EXPIRY) ======
print("\n=== OPTION BACKTEST: HOLD TO FIXED EXPIRY ===")
opt_results=[]
missing_entry=0; missing_exit=0; skipped=0

res_map=con.execute("SELECT DISTINCT timestamp FROM options_data WHERE option_type='CALL'").fetchdf()
opt_timestamps=set(res_map["timestamp"].values)
spot_map=con.execute("SELECT DISTINCT timestamp FROM spot_data").fetchdf()
spot_timestamps=set(spot_map["timestamp"].values)
all_opt_ts=sorted(opt_timestamps)

import bisect
for idx,row in trades.iterrows():
    if (idx+1)%200==0: print(f"  {idx+1}/{len(trades)}...",flush=True)
    ed=row["entry_dt"]
    ed_floor=ed.floor("5min")
    
    # Skip if before options data exists
    if ed<pd.Timestamp("2021-06-14",tz="Asia/Kolkata"): 
        skipped+=1; continue
    
    # Find nearest option timestamp
    if ed_floor in opt_timestamps:
        ts_entry=ed_floor
    else:
        i=bisect.bisect_left(all_opt_ts,ed_floor)
        if i>=len(all_opt_ts): skipped+=1; continue
        ts_entry=all_opt_ts[i]
    
    # === WEEKLY ===
    wec_w,wed_w=get_next_expiry(ts_entry,"WEEK",min_days=3)
    if wec_w is None or wed_w is None: missing_entry+=1; continue
    ep_w,stk_w=get_price(ts_entry,"WEEK",wec_w)
    if ep_w is None: missing_entry+=1; continue
    
    # Exit: last price before/at expiry
    xp_w=get_expiry_price(wed_w,"WEEK",wec_w)
    if xp_w is None: missing_exit+=1; continue
    pnl_w=round(xp_w-ep_w,1)
    
    # === MONTHLY === (same logic but only if entry > 7 DTE)
    wec_m,wed_m=get_next_expiry(ts_entry,"MONTH",min_days=7)
    if wec_m is None or wed_m is None:
        pnl_m=None
    else:
        ep_m,stk_m=get_price(ts_entry,"MONTH",wec_m)
        if ep_m is None:
            pnl_m=None
        else:
            xp_m=get_expiry_price(wed_m,"MONTH",wec_m)
            pnl_m=round(xp_m-ep_m,1) if xp_m else None
    
    opt_results.append({
        "yr":row["yr"],"mo":row["mo"],
        "entry_ts":ts_entry,
        "week_code":wec_w,"week_expiry":wed_w,"week_entry":ep_w,"week_exit":xp_w,"week_pnl":pnl_w,
        "month_code":wec_m,"month_expiry":wed_m,
        "month_entry":ep_m if 'ep_m' in dir() else None,
        "month_exit":xp_m if 'xp_m' in dir() else None,"month_pnl":pnl_m,
    })

df_opt=pd.DataFrame(opt_results)
print(f"\nOption trades: {len(df_opt)} (skipped={skipped}, missing_entry={missing_entry}, missing_exit={missing_exit})")

# Stats
for col,label in [("week_pnl","WEEKLY CALL"),("month_pnl","MONTHLY CALL")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    p=sub[col]
    wr=(p>0).mean(); net=p.sum(); aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0; wl=aw/abs(al) if al!=0 else 999
    pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    calmar=net/mdd if mdd>0 else 999
    # Holding days stats
    if col=="week_pnl":
        sub2=sub.copy()
        sub2["hold_days"]=(sub2["week_expiry"]-sub2["entry_ts"]).dt.total_seconds()/86400
        print(f"\n{label}: {len(p)} trades, Net={net:+,.0f}, WR={wr:.1%}, W/L={wl:.1f}x, "
              f"PF={pf:.1f}x, MDD={mdd:,.0f}, Sharpe={sharpe:.3f}, Calmar={calmar:.1f}x")
        print(f"  Avg hold: {sub2['hold_days'].mean():.1f} days")
    else:
        print(f"\n{label}: {len(p)} trades, Net={net:+,.0f}, WR={wr:.1%}, W/L={wl:.1f}x, "
              f"PF={pf:.1f}x, MDD={mdd:,.0f}, Sharpe={sharpe:.3f}, Calmar={calmar:.1f}x")

# ====== YEARLY ======
print("\n"+"="*100)
print("YEARLY BREAKDOWN")
print("="*100)
for col,label in [("week_pnl","WEEKLY"),("month_pnl","MONTHLY")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    print(f"\n--- {label} ---")
    yr=sub.groupby("yr")[col].agg(["count","sum",lambda x:(x>0).mean()])
    yr.columns=["Trades","Net","WR"]
    print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    print("-"*35)
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
    print("-"*35)
    for m in range(1,13):
        if m in mo.index:
            r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

# ====== COMPARISON: same trades on spot vs option ======
print("\n"+"="*100)
print("DISTRIBUTION: PnL per trade")
print("="*100)
for col,label in [("week_pnl","WEEKLY")]:
    sub=df_opt.dropna(subset=[col])
    if len(sub)==0: continue
    p=sub[col]
    print(f"\n  {label}: Min={p.min():,.1f}, Max={p.max():,.1f}, Median={p.median():,.1f}")
    ptiles=[10,25,50,75,90]
    print(f"  Percentiles:")
    for pt in ptiles:
        print(f"    P{pt}: {np.percentile(p,pt):,.1f}")
    print(f"  Winning trades: {(p>0).sum()}, avg={p[p>0].mean():,.1f}")
    print(f"  Losing trades: {(p<0).sum()}, avg={p[p<0].mean():,.1f}")
    print(f"  Breakeven: {(p==0).sum()}")

con.close()
print("\nDone!")
