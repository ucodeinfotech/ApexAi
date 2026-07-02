"""
Option Backtest: Engulf_Raw CH55 on spot, PnL measured in ATM CALL option premium
Trades both WEEKLY and MONTHLY expiry for comparison
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
# Pre-index options for fast lookup
con.execute("CREATE INDEX IF NOT EXISTS idx_opt_ts ON options_data(timestamp)")
con.execute("CREATE INDEX IF NOT EXISTS idx_spot_ts ON spot_data(timestamp)")
print(f"Options rows: {con.execute('SELECT COUNT(*) FROM options_data').fetchone()[0]:,}")
print(f"Spot rows: {con.execute('SELECT COUNT(*) FROM spot_data').fetchone()[0]:,}")

# ====== BUILD TRADES ON SPOT (same as before, but only on NIFTY50 for option mapping) ======
print("\nLoading H1 + M5 for NIFTY50...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
atr5=compute_atr(m5)
m5_epoch=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()

print("Finding trades...")
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
    he=ep; exit_dt=None; exit_pnl=None
    for j in range(ri,len(m5["close"])):
        ca=atr5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-CH*ca:
            exit_dt=m5["datetime"].iloc[j]; exit_pnl=round(m5["close"].iloc[j]-ep,1); break
    if exit_dt is None:
        exit_dt=m5["datetime"].iloc[-1]; exit_pnl=round(m5["close"].iloc[-1]-ep,1)
    rows.append({"entry_dt":entry_dt,"exit_dt":exit_dt,"yr":ts.year,"mo":ts.month,"spot_pnl":exit_pnl})

trades=pd.DataFrame(rows)
print(f"Total spot trades: {len(trades)}, Spot PnL: {trades['spot_pnl'].sum():+,.0f}")

# ====== OPTION LOOKUP FUNCTIONS ======
def get_option_price(ts, expiry_flag, expiry_code):
    """Get ATM CALL close price at given timestamp for given expiry"""
    # Round to nearest 5-min
    ts_r=ts.floor("5min")
    r=con.execute(f"""
        SELECT close, strike FROM options_data 
        WHERE timestamp = '{ts_r}'
          AND option_type = 'CALL'
          AND expiry_flag = '{expiry_flag}'
          AND expiry_code = {expiry_code}
          AND atm_distance = 0
        LIMIT 1
    """).fetchone()
    return r  # (close, strike) or None

def find_front_expiry(ts, expiry_flag):
    """Find the front expiry code at given timestamp"""
    ts_r=ts.floor("5min")
    codes=con.execute(f"""
        SELECT DISTINCT expiry_code FROM options_data 
        WHERE timestamp = '{ts_r}'
          AND option_type = 'CALL'
          AND expiry_flag = '{expiry_flag}'
        ORDER BY expiry_code
        LIMIT 1
    """).fetchone()
    return codes[0] if codes else None

# Test lookup
test_ts=pd.Timestamp("2024-01-15 09:30:00")
print(f"\nTest lookup at {test_ts}:")
for ef in ["WEEK","MONTH"]:
    ec=find_front_expiry(test_ts,ef)
    p=get_option_price(test_ts,ef,ec)
    print(f"  {ef}: expiry_code={ec}, ATM CALL price={p[0] if p else 'N/A'}, strike={p[1] if p else 'N/A'}")

# ====== BACKTEST ON OPTIONS ======
print("\n=== OPTION BACKTEST ===")
opt_results=[]
missing=0
total=len(trades)
for idx,row in trades.iterrows():
    if (idx+1)%200==0: print(f"  {idx+1}/{total}...",flush=True)
    ed=row["entry_dt"]
    # Find front expiry codes
    wec=find_front_expiry(ed,"WEEK")
    mec=find_front_expiry(ed,"MONTH")
    if wec is None or mec is None:
        missing+=1; continue
    # Entry prices
    ep_w=get_option_price(ed,"WEEK",wec)
    ep_m=get_option_price(ed,"MONTH",mec)
    if ep_w is None or ep_m is None:
        missing+=1; continue
    entry_w=ep_w[0]; entry_m=ep_m[0]
    strike_w=ep_w[1]; strike_m=ep_m[1]
    # Exit prices (at exit_dt or entry+1 if same timestamp)
    xd=row["exit_dt"]
    # Try exit_dt, if same as entry, advance 1 bar
    if xd<=ed:
        xd=ed+pd.Timedelta(minutes=5)
    # Round exit to 5-min
    xd_r=xd.floor("5min")
    xp_w=con.execute(f"""
        SELECT close FROM options_data 
        WHERE timestamp = '{xd_r}'
          AND option_type='CALL' AND expiry_flag='WEEK'
          AND expiry_code={wec} AND atm_distance=0
        LIMIT 1
    """).fetchone()
    xp_m=con.execute(f"""
        SELECT close FROM options_data 
        WHERE timestamp = '{xd_r}'
          AND option_type='CALL' AND expiry_flag='MONTH'
          AND expiry_code={mec} AND atm_distance=0
        LIMIT 1
    """).fetchone()
    exit_w=xp_w[0] if xp_w else entry_w  # hold if no exit found
    exit_m=xp_m[0] if xp_m else entry_m
    
    opt_results.append({
        "yr":row["yr"],"mo":row["mo"],
        "spot_pnl":row["spot_pnl"],
        "opt_week":round(exit_w-entry_w,1),
        "opt_month":round(exit_m-entry_m,1),
        "entry_ts":ed,"exit_ts":xd,
    })

df_opt=pd.DataFrame(opt_results)
print(f"\nOption trades: {len(df_opt)} (missing={missing})")

# Stats
for label,col in [("SPOT","spot_pnl"),("WEEKLY CALL","opt_week"),("MONTHLY CALL","opt_month")]:
    p=df_opt[col]
    if len(p)==0: continue
    wr=(p>0).mean(); net=p.sum(); aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0; wl=aw/abs(al) if al!=0 else 999
    pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    calmar=net/mdd if mdd>0 else 999
    print(f"\n{label}: {len(p)} trades, Net={net:+,.0f}, WR={wr:.1%}, W/L={wl:.1f}x, PF={pf:.1f}x, MDD={mdd:,.0f}, Sharpe={sharpe:.3f}, Calmar={calmar:.1f}x")

# ====== YEARLY ======
print("\n"+"="*100)
print("YEARLY BREAKDOWN")
print("="*100)
for lbl,col in [("SPOT","spot_pnl"),("WEEKLY","opt_week"),("MONTHLY","opt_month")]:
    print(f"\n--- {lbl} ---")
    yr=df_opt.groupby("yr")[col].agg(["count","sum",lambda x:(x>0).mean()])
    yr.columns=["Trades","Net","WR"]
    print(f"{'Year':>6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    print("-"*35)
    for y in sorted(yr.index):
        r=yr.loc[y]; print(f"{int(y):>6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

# ====== MONTHLY ======
print("\n"+"="*100)
print("MONTHLY BREAKDOWN")
print("="*100)
for lbl,col in [("SPOT","spot_pnl"),("WEEKLY","opt_week"),("MONTHLY","opt_month")]:
    print(f"\n--- {lbl} ---")
    mo=df_opt.groupby("mo")[col].agg(["count","sum",lambda x:(x>0).mean()])
    mo.columns=["Trades","Net","WR"]
    print(f"{'Month':<6} {'Trades':>7} {'Net':>12} {'WR':>7}")
    print("-"*35)
    for m in range(1,13):
        if m in mo.index:
            r=mo.loc[m]; print(f"{MONTHS[m-1]:<6} {int(r['Trades']):>7} {r['Net']:>+12,.0f} {r['WR']:>6.1%}")

# ====== CORRELATIONS ======
print("\n"+"="*100)
print("CORRELATION: Spot PnL vs Option PnL")
print("="*100)
for col in ["opt_week","opt_month"]:
    c=df_opt["spot_pnl"].corr(df_opt[col])
    print(f"  Spot vs {col}: {c:.3f}")

# ====== COMPARISON: Option vs Spot per trade ======
print("\n"+"="*100)
print("OPTION PREMIUM EFFICIENCY (Option PnL / Spot PnL per trade)")
print("="*100)
for col in ["opt_week","opt_month"]:
    ratio=(df_opt[col]/df_opt["spot_pnl"].replace(0,np.nan)).dropna()
    ratio=ratio[np.isfinite(ratio)]
    print(f"  {col}: avg ratio={ratio.mean():.3f}, median={ratio.median():.3f}, "
          f"min={ratio.min():.3f}, max={ratio.max():.3f}")
    # Only on winning trades
    w=df_opt[df_opt["spot_pnl"]>0]
    wr=(w[col]/w["spot_pnl"]).dropna()
    wr=wr[np.isfinite(wr)]
    print(f"    On spot wins: avg={wr.mean():.3f}, median={wr.median():.3f}")
    # Only on losing trades
    l=df_opt[df_opt["spot_pnl"]<0]
    lr=(l[col]/l["spot_pnl"].abs()).dropna()
    lr=lr[np.isfinite(lr)]
    print(f"    On spot losses: avg={lr.mean():.3f}, median={lr.median():.3f}")

con.close()
print("\nDone!")
