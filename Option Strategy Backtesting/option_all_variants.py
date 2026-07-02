"""
ALL option strategy variants compared: CH55/100/150/200 on option price,
hold-to-expiry, fixed-day holds, ATM vs ATM+1, weekly vs monthly
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, bisect
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ============ 1. BUILD ALL SPOT TRADES (same engine) ============
def compute_atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),
                  abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

print("Building spot trades...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
atr5=compute_atr(m5); m5_epoch=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()

trades=[]
body=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; rr=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
    if body.iloc[i]<body.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
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
    entry_dt=m5["datetime"].iloc[ri]; ep=m5["close"].iloc[ri]
    if ep-m5["low"].iloc[ri]<=0: continue
    he=ep
    for j in range(ri,len(m5["close"])):
        ca=atr5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":entry_dt,"exit_dt":m5["datetime"].iloc[j],
                           "yr":ts.year,"mo":ts.month}); break

trades=pd.DataFrame(trades)
print(f"Spot trades: {len(trades)}")

# ============ 2. CONNECT DUCKDB + PREPARE ============
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
opt_ts=con.execute("SELECT DISTINCT timestamp FROM options_data WHERE option_type='CALL'").fetchdf()
opt_ts_list=sorted(opt_ts["timestamp"].values)

def find_ts(target):
    target=target.to_pydatetime().replace(tzinfo=None)
    i=bisect.bisect_right(opt_ts_list,target)
    if i==0: return pd.Timestamp(opt_ts_list[0])
    return pd.Timestamp(opt_ts_list[i-1])

def get_price_at_exact(ts, flag, code, atm=0):
    """Get CALL close at exact timestamp"""
    code_clause=f"AND expiry_code={code}" if code is not None else ""
    r=con.execute(f"""
        SELECT close, strike FROM options_data 
        WHERE timestamp='{ts}'::TIMESTAMP AND option_type='CALL'
          AND expiry_flag='{flag}' {code_clause} AND atm_distance={atm}
        LIMIT 1
    """).fetchone()
    return r

def get_opt_ohlc(ts_from, ts_to, flag, code, atm=0):
    """Get all OHLC bars for an option between timestamps"""
    return con.execute(f"""
        SELECT timestamp, open, high, low, close FROM options_data 
        WHERE timestamp >= '{ts_from}'::TIMESTAMP 
          AND timestamp <= '{ts_to}'::TIMESTAMP
          AND option_type='CALL' AND expiry_flag='{flag}'
          AND expiry_code={code} AND atm_distance={atm}
        ORDER BY timestamp
    """).fetchdf()

# ============ 3. VARIANT TESTING ============
def test_variant(trades, flag, name, entry_atm=0, exit_mode="ch_spot", ch_val=55, hold_days=None):
    """
    Test one variant
    exit_mode: 'ch_spot'=exit on spot CH, 'ch_opt'=trailing stop on option price,
               'expiry'=hold to expiry, 'fix_days'=hold N days
    """
    results=[]
    matched=0; no_data=0
    for _,row in trades.iterrows():
        ed=row["entry_dt"]
        if ed.to_pydatetime().replace(tzinfo=None)<pd.Timestamp("2021-06-14"): continue
        ts_entry=find_ts(ed)
        # Get option at entry
        er=get_price_at_exact(ts_entry, flag, 1, entry_atm)
        if er is None:
            er=get_price_at_exact(ts_entry, flag, None, entry_atm)
            if er is None: no_data+=1; continue
        ep=float(er[0]); strike=float(er[1]); ec=1
        
        # === DETERMINE EXIT ===
        xp=None
        if exit_mode=="ch_spot":
            # Exit at spot CH trigger time
            xd=row["exit_dt"]
            if xd<=ed: xd=ed+pd.Timedelta(minutes=5)
            ts_exit=find_ts(xd)
            xr=get_price_at_exact(ts_exit, flag, ec, entry_atm)
            if xr is None:
                # Try finding nearest
                xr=con.execute(f"""
                    SELECT close FROM options_data WHERE option_type='CALL'
                    AND expiry_flag='{flag}' AND expiry_code={ec} AND atm_distance={entry_atm}
                    ORDER BY ABS(EXTRACT(EPOCH FROM timestamp)-EXTRACT(EPOCH FROM TIMESTAMP '{ts_exit}'))
                    LIMIT 1
                """).fetchone()
            xp=float(xr[0]) if xr else ep
        
        elif exit_mode=="ch_opt":
            # Trailing stop on option price
            he=ep
            # Get option bars from entry to entry+30 days
            ohlc=get_opt_ohlc(ts_entry, ts_entry+pd.Timedelta(days=30), flag, ec, entry_atm)
            if len(ohlc)<5: continue
            # Compute ATR(14) on option close
            opt_atr=ohlc["close"].ewm(span=14,min_periods=14,adjust=False).mean().values
            for j in range(1,len(ohlc)):
                ca=opt_atr[j] if j<len(opt_atr) and not pd.isna(opt_atr[j]) else opt_atr[-1]
                h=ohlc["high"].iloc[j]; c=ohlc["close"].iloc[j]
                if h>he: he=h
                if c<he-ch_val*ca:
                    xp=c; break
            if xp is None: xp=ohlc["close"].iloc[-1]
        
        elif exit_mode=="expiry":
            # Calendar-based expiry
            from datetime import timedelta
            d=ed.to_pydatetime().replace(tzinfo=None).date()
            if flag=="WEEK":
                da=(3-d.weekday())%7; da=da if da>0 else 7
            else:  # MONTH - last Thu
                import calendar
                _,ld=calendar.monthrange(d.year,d.month); dd=d.replace(day=ld)
                while dd.weekday()!=3: dd-=timedelta(days=1)
                da=(dd-d).days; da=da if da>0 else 28
            exp_dt=ed+pd.Timedelta(days=da)
            exp_dt=exp_dt.replace(hour=15,minute=25)
            # Get last price before expiry
            xr=con.execute(f"""
                SELECT close FROM options_data WHERE option_type='CALL'
                AND expiry_flag='{flag}' AND expiry_code={ec} AND atm_distance={entry_atm}
                AND timestamp >= '{exp_dt-pd.Timedelta(hours=2)}'::TIMESTAMP
                AND timestamp <= '{exp_dt}'::TIMESTAMP
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            xp=float(xr[0]) if xr else 0.0
        
        elif exit_mode=="fix_days":
            xd=ed+pd.Timedelta(days=hold_days)
            ts_exit=find_ts(xd)
            xr=get_price_at_exact(ts_exit, flag, ec, entry_atm)
            if xr is None:
                xr=con.execute(f"""
                    SELECT close FROM options_data WHERE option_type='CALL'
                    AND expiry_flag='{flag}' AND expiry_code={ec} AND atm_distance={entry_atm}
                    ORDER BY ABS(EXTRACT(EPOCH FROM timestamp)-EXTRACT(EPOCH FROM TIMESTAMP '{ts_exit}'))
                    LIMIT 1
                """).fetchone()
            xp=float(xr[0]) if xr else ep
        
        if xp is None: no_data+=1; continue
        pnl=round(xp-ep,1); matched+=1
        results.append({"yr":row["yr"],"mo":row["mo"],"pnl":pnl})
    
    df=pd.DataFrame(results)
    return df

def compute_stats(pnl):
    pnl=pnl.dropna(); n=len(pnl); net=pnl.sum()
    if n<2: return {"n":0}
    wr=(pnl>0).mean(); aw=pnl[pnl>0].mean() if (pnl>0).sum()>0 else 0
    al=pnl[pnl<0].mean() if (pnl<0).sum()>0 else 0; wl=aw/abs(al) if al!=0 else 999
    pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).sum()>0 else 999
    cum=np.cumsum(pnl); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    sharpe=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else 0
    calmar=net/mdd if mdd>0 else 999
    return {"n":n,"net":net,"wr":wr,"aw":aw,"al":al,"wl":wl,"pf":pf,"mdd":mdd,"sharpe":sharpe,"calmar":calmar}

# ============ 4. DEFINE ALL VARIANTS ============
variants=[
    # (label, flag, exit_mode, ch_val, atm, hold_days)
    ("CH55_SpotExit_WEEK","WEEK","ch_spot",55,0,None),
    ("CH55_SpotExit_MONTH","MONTH","ch_spot",55,0,None),
    ("CH55_OptExit_WEEK","WEEK","ch_opt",55,0,None),
    ("CH55_OptExit_MONTH","MONTH","ch_opt",55,0,None),
    ("CH100_OptExit_WEEK","WEEK","ch_opt",100,0,None),
    ("CH100_OptExit_MONTH","MONTH","ch_opt",100,0,None),
    ("CH150_OptExit_WEEK","WEEK","ch_opt",150,0,None),
    ("CH150_OptExit_MONTH","MONTH","ch_opt",150,0,None),
    ("CH200_OptExit_WEEK","WEEK","ch_opt",200,0,None),
    ("CH200_OptExit_MONTH","MONTH","ch_opt",200,0,None),
    ("HoldExpiry_WEEK","WEEK","expiry",55,0,None),
    ("HoldExpiry_MONTH","MONTH","expiry",55,0,None),
    ("Fix3Day_WEEK","WEEK","fix_days",55,0,3),
    ("Fix5Day_WEEK","WEEK","fix_days",55,0,5),
    ("Fix7Day_WEEK","WEEK","fix_days",55,0,7),
    ("Fix3Day_MONTH","MONTH","fix_days",55,0,3),
    ("Fix5Day_MONTH","MONTH","fix_days",55,0,5),
    ("Fix7Day_MONTH","MONTH","fix_days",55,0,7),
    # ATM+1 (OTM) variants
    ("CH55_SpotExit_OTM_WEEK","WEEK","ch_spot",55,1,None),
    ("CH55_OptExit_OTM_WEEK","WEEK","ch_opt",55,1,None),
    ("CH100_OptExit_OTM_WEEK","WEEK","ch_opt",100,1,None),
    ("HoldExpiry_OTM_WEEK","WEEK","expiry",55,1,None),
]

print(f"\nTesting {len(variants)} variants...")
all_results=[]
for label,flag,emode,chv,atm,hd in variants:
    print(f"  {label}...",end=" ",flush=True)
    df=test_variant(trades,flag,label,entry_atm=atm,exit_mode=emode,ch_val=chv,hold_days=hd)
    s=compute_stats(df["pnl"]) if len(df)>0 else {"n":0}
    if s["n"]>0:
        all_results.append({"Variant":label,"Trades":s["n"],"Net":f"{s['net']:+,.0f}",
            "WR":f"{s['wr']:.0%}","W/L":f"{s['wl']:.1f}x","PF":f"{s['pf']:.1f}x",
            "MDD":f"{s['mdd']:,.0f}","Sharpe":f"{s['sharpe']:.2f}","Calmar":f"{s['calmar']:.1f}x",
            "AvgPnl":f"{s['net']/s['n']:+,.1f}","ExitMode":emode})
        print(f"{s['n']}t, Net={s['net']:+,.0f}, WR={s['wr']:.0%}")
    else:
        print("NO DATA")

# ============ 5. RANKED RESULTS TABLE ============
rr=pd.DataFrame(all_results)
rr["NetVal"]=rr["Net"].str.replace(",","").str.replace("+","").astype(float)
rr=rr.sort_values("NetVal",ascending=False).reset_index(drop=True)

print("\n"+"="*120)
print("ALL OPTION VARIANTS - RANKED BY NET PnL")
print("="*120)
print(f"{'Rank':>4} {'Variant':<36} {'Trades':>7} {'Net':>12} {'WR':>5} {'W/L':>6} "
      f"{'PF':>6} {'MDD':>10} {'Sharpe':>7} {'Calmar':>8} {'AvgPnl':>8}")
print("-"*120)
for i,(_,r) in enumerate(rr.iterrows()):
    is_best="*" if i==0 else " "
    print(f"{i+1:>3}{is_best} {r['Variant']:<36} {r['Trades']:>7} {r['Net']:>12} {r['WR']:>5} "
          f"{r['W/L']:>6} {r['PF']:>6} {r['MDD']:>10} {r['Sharpe']:>7} {r['Calmar']:>8} {r['AvgPnl']:>8}")

# ============ 6. BEST VARIANT: DETAILED BREAKDOWN ============
if len(rr)>0:
    best_name=rr.iloc[0]["Variant"]
    print(f"\n{'='*100}")
    print(f"BEST VARIANT: {best_name} - Full Breakdown")
    print(f"{'='*100}")
    
    # Re-run to get full df
    for label,flag,emode,chv,atm,hd in variants:
        if label==best_name:
            best_df=test_variant(trades,flag,label,entry_atm=atm,exit_mode=emode,ch_val=chv,hold_days=hd)
            break
    else:
        best_df=pd.DataFrame()
    
    if len(best_df)>0:
        s=compute_stats(best_df["pnl"])
        print(f"\nOverall: {s['n']} trades, Net={s['net']:+,.0f}, WR={s['wr']:.0%}, "
              f"W/L={s['wl']:.1f}x, MDD={s['mdd']:,.0f}, Sharpe={s['sharpe']:.2f}")
        
        # Yearly
        print(f"\n--- Yearly ---")
        yr=best_df.groupby("yr")["pnl"].agg(["count","sum",lambda x:(x>0).mean()])
        yr.columns=["Trades","Net","WR"]
        print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
        for y in sorted(yr.index):
            r=yr.loc[y]; print(f"{int(y):>6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")
        print(f"{'ALL':>6} {yr['Trades'].sum():>7} {yr['Net'].sum():>+12,.0f} {yr['WR'].mean():>6.1%}")
        
        # Monthly
        print(f"\n--- Monthly ---")
        mo=best_df.groupby("mo")["pnl"].agg(["count","sum",lambda x:(x>0).mean()])
        mo.columns=["Trades","Net","WR"]
        print(f"{'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>7}")
        for m in range(1,13):
            if m in mo.index:
                r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

# ============ 7. GROUP COMPARISON BY EXIT MODE ============
print("\n"+"="*100)
print("COMPARISON BY EXIT MODE (avg across variants)")
print("="*100)
for emode in rr["ExitMode"].unique():
    sub=rr[rr["ExitMode"]==emode]
    avg_net=sub["NetVal"].mean()
    print(f"  {emode:<15} avg Net={avg_net:+,.0f}  (variants: {len(sub)})")

con.close()
print("\nDone!")
