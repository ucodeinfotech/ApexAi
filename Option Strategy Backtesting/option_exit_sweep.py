"""
Comprehensive option exit & improvement sweep for ATM0_WEEK
Loads once, tests 40+ variants
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, bisect
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# === Build trades ===
def atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()
print("Building trades...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
a5=atr(m5); me=m5["datetime"].astype("int64").values
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
        ca=a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],
                           "yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"]=trades["exit_dt"].dt.tz_localize(None)
from datetime import timedelta
import calendar
exp_w_l,exp_m_l=[],[]
for _,r in trades.iterrows():
    d=r["ed_naive"].date()
    da=(3-d.weekday())%7; da=da if da>0 else 7
    exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
    _,ld=calendar.monthrange(d.year,d.month); dd=d.replace(day=ld)
    while dd.weekday()!=3: dd-=timedelta(days=1)
    da=(dd-d).days; da=da if da>0 else 28
    exp_m_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
trades["exp_w"]=exp_w_l; trades["exp_m"]=exp_m_l
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Option-era trades: {len(trades_pre)}")

# === Load option data ===
print("Loading option data...")
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
df_opt=con.execute("SELECT timestamp,close,open,high,low FROM options_data WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp").fetchdf()
con.close()
print(f"Rows: {len(df_opt):,}")
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
cl_arr=df_opt["close"].values.astype(float)
op_arr=df_opt["open"].values.astype(float)
hi_arr=df_opt["high"].values.astype(float)
lo_arr=df_opt["low"].values.astype(float)

# Pre-compute option ATR (14-period EWMA)
def opt_atr(opn,hi,lo,cl):
    tr=pd.concat([hi-lo,abs(hi-cl),abs(lo-cl)],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=3,adjust=False).mean().values
opt_atr_arr=opt_atr(pd.Series(op_arr),pd.Series(hi_arr),pd.Series(lo_arr),pd.Series(cl_arr))

def lookup(ts):
    """Get option bar at timestamp"""
    ts_ns=np.datetime64(ts,"us")
    i=np.searchsorted(ts_arr,ts_ns)
    if i==0: return 0,cl_arr[0],opt_atr_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1],opt_atr_arr[-1]
    return i-1,cl_arr[i-1],opt_atr_arr[i-1]

def lookup_price(ts):
    _,c,_=lookup(ts); return c

def get_option_bars(from_ts, to_ts):
    """Get option bars in [from_ts, to_ts) range"""
    f_ns=np.datetime64(from_ts,"us")
    t_ns=np.datetime64(to_ts,"us")
    i=np.searchsorted(ts_arr,f_ns); j=np.searchsorted(ts_arr,t_ns)
    return i,j

# === Sweep: for each trade & variant, compute exit price ===
# We'll build a matrix: trades × variants

# Entry prices for all trades
entry_prices=[]
for _,r in trades_pre.iterrows():
    _,c,_=lookup(r["ed_naive"]); entry_prices.append(c)
entry_prices=np.array(entry_prices)

# Pre-compute all needed timestamps
xd_list=trades_pre["xd_naive"].tolist()  # spot exits
ed_list=trades_pre["ed_naive"].tolist()

# Build forward-looking bar indices for each trade
# For each trade, find the index range [start_idx, end_idx) where start_idx = entry bar
bar_starts=[]; bar_ends=[]
for ed in ed_list:
    s_idx,_,_=lookup(ed)
    e_idx=lookup_price(ed+pd.Timedelta(days=30))  # far enough
    bar_starts.append(s_idx); bar_ends.append(s_idx)  # will extend

# For hold days: get exit price at entry+N days
def price_at_days(ed, days):
    return lookup_price(ed+pd.Timedelta(days=days))

# For CH trail on option: walk bars and trail
def trail_exit_price(s_idx, ch_val):
    """Walk forward from s_idx, trail with CH*ATR on option price, return (exit_idx, exit_price)"""
    he=cl_arr[s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(cl_arr))):
        ca=opt_atr_arr[i]
        if np.isnan(ca): continue
        if hi_arr[i]>he: he=hi_arr[i]
        if cl_arr[i]<he-ch_val*ca:
            return i,cl_arr[i]
    return None,None

# Combined: trail + max days
def trail_combined(s_idx, ch_val, max_days):
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
    he=cl_arr[s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(cl_arr))):
        if ts_arr[i]>end_ns:
            return i,cl_arr[i]
        ca=opt_atr_arr[i]
        if np.isnan(ca): continue
        if hi_arr[i]>he: he=hi_arr[i]
        if cl_arr[i]<he-ch_val*ca:
            return i,cl_arr[i]
    return None,None

# Profit target + time stop
def profit_target_exit(s_idx, target_pts, max_days, stop_loss=None):
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
    ep=cl_arr[s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(cl_arr))):
        if ts_arr[i]>end_ns:
            return i,cl_arr[i]
        pnl=cl_arr[i]-ep
        if pnl>=target_pts:
            return i,cl_arr[i]
        if stop_loss and pnl<= -abs(stop_loss):
            return i,cl_arr[i]
    return None,None

# === Run all variants ===
all_rows=[]

def add_result(label, pnl_list, n_trades, extra=""):
    p=pd.Series(pnl_list); n=len(p)
    if n<3: return
    net=p.sum(); wr=(p>0).mean()
    aw=p[p>0].mean() if (p>0).sum()>0 else 0
    al=p[p<0].mean() if (p<0).sum()>0 else 0
    wl=aw/abs(al) if al!=0 else 999; pf=p[p>0].sum()/abs(p[p<0].sum()) if (p<0).sum()>0 else 999
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999; sharpe=p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    all_rows.append({"Variant":label,"Net":net,"WR":wr,"W/L":wl,"PF":pf,"MDD":mdd,
        "Calmar":calmar,"Sharpe":sharpe,"Trades":n,
        "NetStr":f"{net:+,.0f}","WRStr":f"{wr:.0%}","Extra":extra})

# 1) HOLD DAYS SWEEP
print("\n=== Hold Days Sweep ===")
for days in [2,3,4,5,6,7,8,9,10,14,21]:
    pnls=[]
    for i,ed in enumerate(ed_list):
        xp=price_at_days(ed, days)
        if xp is not None: pnls.append(round(xp-entry_prices[i],1))
    add_result(f"Hold{days}D", pnls, len(pnls), f"hold {days}d")
    print(f"  Hold {days:2d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 2) OPTION TRAILING STOP SWEEP
print("\n=== Option Trail Stop Sweep ===")
for ch in [20,30,40,50,75,100,150,200,300,500]:
    pnls=[]
    for i,ed in enumerate(ed_list):
        s_idx,_,_=lookup(ed)
        _,xp=trail_exit_price(s_idx, ch)
        if xp is not None: pnls.append(round(xp-entry_prices[i],1))
    add_result(f"TrailCH{ch}", pnls, len(pnls), f"trail {ch}*ATR on opt")
    print(f"  TrailCH{ch:3d}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 3) SPOT TRAILING STOP SWEEP (spot exit, already have exit timestamps)
print("\n=== Spot Exit (trail on spot) ===")
for ch in [30,45,55,65,80,100,150,200]:
    # Need to recompute spot exits with different CH values
    pnls=[]
    # Re-run spot backtest with different CH
    for _,row in trades_pre.iterrows():
        tr=a5  # 5-min ATR
        ep_idx=np.searchsorted(me,row["entry_dt"].asm8.view("int64"),side="right")-1
        if ep_idx<0: continue
        he=m5["close"].iloc[ep_idx]
        for j in range(ep_idx,len(m5["close"])):
            ca=a5.iloc[j]
            if pd.isna(ca): continue
            if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
            if m5["close"].iloc[j]<he-ch*ca:
                xp_t=lookup_price(m5["datetime"].iloc[j].tz_localize(None))
                if xp_t is not None: pnls.append(round(xp_t-entry_prices[len(pnls) if len(pnls)<len(ed_list) else 0],1))
                break
    add_result(f"SpotTrail{ch}", pnls, len(pnls), f"trail {ch}*ATR on spot")
    print(f"  SpotCH{ch:3d}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 4) PROFIT TARGET + TIME STOP
print("\n=== Profit Target Sweep ===")
for tp in [30,50,75,100,150,200]:
    for max_d in [5,7,10]:
        pnls=[]
        for i,ed in enumerate(ed_list):
            s_idx,_,_=lookup(ed)
            _,xp=profit_target_exit(s_idx,tp,max_d)
            if xp is not None: pnls.append(round(xp-entry_prices[i],1))
        add_result(f"TP{tp}_Max{max_d}d", pnls, len(pnls), f"take profit {tp}pts, max {max_d}d")
        print(f"  TP{tp:3d}_Max{max_d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 5) PROFIT TARGET + STOP LOSS
print("\n=== TP + SL Sweep ===")
for tp in [50,75,100]:
    for sl in [50,75,100,150]:
        for max_d in [7,10]:
            pnls=[]
            for i,ed in enumerate(ed_list):
                s_idx,_,_=lookup(ed)
                _,xp=profit_target_exit(s_idx,tp,max_d,sl)
                if xp is not None: pnls.append(round(xp-entry_prices[i],1))
            add_result(f"TP{tp}_SL{sl}_Max{max_d}d", pnls, len(pnls), f"TP={tp} SL={sl} max={max_d}d")
            print(f"  TP{tp:3d}_SL{sl:3d}_{max_d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 6) COMBINED: Trail + max days
print("\n=== Trail CH + Max Days ===")
for ch in [50,75,100,200]:
    for max_d in [5,7,10,14]:
        pnls=[]
        for i,ed in enumerate(ed_list):
            s_idx,_,_=lookup(ed)
            _,xp=trail_combined(s_idx,ch,max_d)
            if xp is not None: pnls.append(round(xp-entry_prices[i],1))
        add_result(f"Trail{ch}_Max{max_d}d", pnls, len(pnls), f"trail {ch}*ATR + max {max_d}d")
        print(f"  Trail{ch:3d}_Max{max_d:2d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 7) DTE-BASED EXITS (days to expiry)
print("\n=== DTE-Based Exit Sweep ===")
for dte in [0,1,2,3,5]:
    pnls=[]
    for i,ed in enumerate(ed_list):
        exp=trades_pre["exp_w"].iloc[i]  # weekly expiry
        exit_ts=exp-pd.Timedelta(days=dte)
        if exit_ts<=ed: exit_ts=ed+pd.Timedelta(hours=1)
        xp=lookup_price(exit_ts)
        if xp is not None: pnls.append(round(xp-entry_prices[i],1))
    add_result(f"DTE{dte}", pnls, len(pnls), f"exit {dte}d before expiry")
    print(f"  DTE{dte}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# === NON-EXIT IMPROVEMENTS ===
# 8) ENTRY DAY FILTER
print("\n=== Entry Day Filter ===")
base_pnls=[]
for i,ed in enumerate(ed_list):
    xp=price_at_days(ed,5)
    if xp is not None: base_pnls.append((i,round(xp-entry_prices[i],1)))

for dayset_name,day_filter in [("MonOnly",[0]),("TueOnly",[1]),("WedOnly",[2]),
    ("ThuOnly",[3]),("FriOnly",[4]),("MonTueWed",[0,1,2]),
    ("TueWedThu",[1,2,3]),("MonTue",[0,1]),("WedThuFri",[2,3,4]),
    ("MonWedFri",[0,2,4])]:
    pnls=[p for (i,p) in base_pnls if trades_pre.iloc[i]["weekday"] in day_filter]
    if len(pnls)>=3:
        add_result(f"Day{dayset_name}_Hold5D", pnls, len(pnls), f"entry on {dayset_name}")
        print(f"  {dayset_name:>12s}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 9) EXPIRY WEEK FILTER
print("\n=== Expiry Week Filter ===")
# Week of expiry: 1 = entry in expiry week, 2 = 1 week before, etc.
for exp_week in [1,2,3,4]:
    pnls=[]
    for bp_idx,(trade_idx,pnl) in enumerate(base_pnls):
        exp=trades_pre["exp_w"].iloc[trade_idx]
        ed=trades_pre["ed_naive"].iloc[trade_idx]
        days_to_exp=(exp-ed).days
        wk=((days_to_exp-1)//7)+1
        if wk==exp_week:
            pnls.append(pnl)
    if len(pnls)>=3:
        add_result(f"ExpWk{exp_week}_Hold5D", pnls, len(pnls), f"enter {exp_week}wk before expiry")
        print(f"  ExpWk{exp_week}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 10) PREMIUM FILTER
print("\n=== Premium Filter ===")
for max_prem in [50,80,100,120,150,200]:
    pnls=[]
    for bp_idx,(trade_idx,pnl) in enumerate(base_pnls):
        if entry_prices[trade_idx]<=max_prem:
            pnls.append(pnl)
    if len(pnls)>=3:
        add_result(f"Prem<{max_prem}_Hold5D", pnls, len(pnls), f"premium < {max_prem}")
        print(f"  Prem<{max_prem:3d}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 11) SKIP AFTER LOSS
print("\n=== Skip After Loss ===")
for skip in [0,1,2,3]:
    for min_wr in [0]:
        pnls=[]
        n_skip=0
        for bp_idx,(trade_idx,pnl) in enumerate(base_pnls):
            if n_skip>0: n_skip-=1; continue
            pnls.append(pnl)
            if pnl<0: n_skip=skip
        if len(pnls)>=3:
            add_result(f"Skip{skip}_Hold5D", pnls, len(pnls), f"skip {skip} after loss")
            print(f"  Skip{skip}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# === RANKING ===
rr=pd.DataFrame(all_rows).sort_values("Net",ascending=False).reset_index(drop=True)
print("\n"+"="*140)
print("OPTION EXIT SWEEP - FULL RANKING")
print("="*140)
print(f"{'Rank':>4} {'Variant':<25} {'Trades':>6} {'Net':>10} {'WR':>5} {'W/L':>6} {'PF':>6} {'MDD':>8} {'Calmar':>6} {'Sharpe':>6}  {'Desc':<35}")
print("-"*120)
for i,(_,r) in enumerate(rr.iterrows()):
    hl="*" if i==0 else " "
    if r["Net"]>0 or i<10:
        print(f"{i+1:>3}{hl} {r['Variant']:<25} {r['Trades']:>6} {r['NetStr']:>10} {r['WRStr']:>5} {r['W/L']:>5.1f}x {r['PF']:>5.1f}x {r['MDD']:>8,.0f} {r['Calmar']:>5.1f}x {r['Sharpe']:>5.2f}  {r['Extra']:<35}")

pos=rr[rr["Net"]>0]
print(f"\n=== POSITIVE VARIANTS: {len(pos)} of {len(rr)} ===")
for i,(_,r) in enumerate(pos.iterrows()):
    print(f"  {i+1:>3}. {r['Variant']:<25} Net={r['NetStr']:>10} WR={r['WRStr']:>5} W/L={r['W/L']:>4.1f}x PF={r['PF']:>4.1f}x Calmar={r['Calmar']:>5.1f}x  | {r['Extra']}")

# Best detail
if len(pos)>0:
    best=pos.iloc[0]
    print(f"\n{'='*100}")
    print(f"BEST: {best['Variant']} Net={best['NetStr']} WR={best['WRStr']} W/L={best['W/L']:5.1f}x")
    print(f"{'='*100}")

print("\nDone!")
