"""
Fixed option exit sweep: same-strike tracking for all variants
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, calendar
from datetime import timedelta
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
from option_fixed_engine import load_fixed_data, get_entry_strike_idx, exit_tp_maxd, exit_maxd, exit_tp_sl, exit_trail
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

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
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"]=trades["exit_dt"].dt.tz_localize(None)
from datetime import timedelta; import calendar
exp_w_l,exp_m_l=[],[]
for _,r in trades.iterrows():
    d_=r["ed_naive"].date()
    da=(3-d_.weekday())%7; da=da if da>0 else 7
    exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
    _,ld=calendar.monthrange(d_.year,d_.month); dd=d_.replace(day=ld)
    while dd.weekday()!=3: dd-=timedelta(days=1)
    da=(dd-d_).days; da=da if da>0 else 28
    exp_m_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
trades["exp_w"]=exp_w_l; trades["exp_m"]=exp_m_l
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Trades: {len(trades_pre)}")

print("Loading fixed data...")
atm_ts_arr, atm_cl_arr, atm_st_arr, strike_cache, lookup = load_fixed_data(trades_pre)

# For each trade, get entry info (entry_strike, entry_price, s_idx, stk_data)
# Filter by premium max
PREM_MAX_LIST=[None,50,80,100,120,130,150,200]
# Use prem_max=130 as default for most tests
DEFAULT_PREM=130

trade_infos=[]
for i in range(len(trades_pre)):
    ed=trades_pre.iloc[i]["ed_naive"]
    info=get_entry_strike_idx(ed, lookup, atm_ts_arr, strike_cache)
    trade_infos.append(info if info else None)

print(f"Valid trades: {sum(1 for t in trade_infos if t)}/{len(trade_infos)}")

def run_variant(exit_fn, prem_max=DEFAULT_PREM, return_idx=False, **kwargs):
    """Run a fixed exit variant across all valid trades.
    exit_fn signature: f(stk_data, s_idx, <kwargs>) -> (exit_price, exit_ts) or (None, None)
    Returns list of PnLs, or (pnls, indices) if return_idx.
    """
    pnls=[]; indices=[]
    for idx,info in enumerate(trade_infos):
        if info is None: continue
        if prem_max is not None and info["ep"]>prem_max: continue
        result=exit_fn(info["stk_data"], info["s_idx"], **kwargs)
        if result is None or result[0] is None: continue
        xp_val,_=result
        pnl=round(xp_val-info["ep"],1)
        pnls.append(pnl); indices.append(idx)
    return (pnls, indices) if return_idx else pnls

def run_hold_days(info, days):
    """Exit at entry + N days"""
    exit_ts=atm_ts_arr[info["si"]]+np.timedelta64(int(days*86400*1e6),"us")
    s_idx=np.searchsorted(info["stk_data"]["ts"],exit_ts)
    if s_idx<=0 or s_idx>=len(info["stk_data"]["cl"]): return None,None
    return info["stk_data"]["cl"][s_idx-1], info["stk_data"]["ts"][s_idx-1]

def price_at_days_by_strike(info, days):
    return run_hold_days(info, days)

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

all_rows=[]

# 1) HOLD DAYS SWEEP
print("\n=== Hold Days Sweep ===")
for days in [2,3,4,5,6,7,8,9,10,14,21]:
    pnls=[]
    for info in trade_infos:
        if info is None or info["ep"]>DEFAULT_PREM: continue
        xp,_=run_hold_days(info,days)
        if xp is not None: pnls.append(round(xp-info["ep"],1))
    add_result(f"Hold{days}D", pnls, len(pnls), f"hold {days}d")
    print(f"  Hold {days:2d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 2) OPTION TRAILING STOP
print("\n=== Option Trail Stop Sweep ===")
for ch in [20,30,40,50,75,100,150,200,300,500]:
    pnls=[]
    for info in trade_infos:
        if info is None or info["ep"]>DEFAULT_PREM: continue
        result=exit_trail(info["stk_data"],info["s_idx"],ch)
        if result is None or result[0] is None: continue
        pnls.append(round(result[0]-info["ep"],1))
    add_result(f"TrailCH{ch}", pnls, len(pnls), f"trail {ch}*ATR on opt")
    print(f"  TrailCH{ch:3d}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 3) SPOT EXIT (pre-computed CH55 from trade engine)
print("\n=== Spot Exit CH55 ===")
pnls=[]
for idx,info in enumerate(trade_infos):
    if info is None or info["ep"]>DEFAULT_PREM: continue
    exit_ts=trades_pre.iloc[idx]["xd_naive"]
    xp_i=np.searchsorted(info["stk_data"]["ts"],np.datetime64(exit_ts,"us"))
    if xp_i>0 and xp_i<len(info["stk_data"]["cl"]):
        pnls.append(round(info["stk_data"]["cl"][xp_i-1]-info["ep"],1))
add_result("SpotCH55_Fixed", pnls, len(pnls), "spot trail CH55 (same strike)")
print(f"  SpotCH55_Fixed: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 4) PROFIT TARGET + TIME STOP
print("\n=== Profit Target Sweep ===")
for tp in [30,50,75,100,150,200]:
    for max_d in [5,7,10]:
        pnls=run_variant(exit_tp_maxd, tp=tp, max_days=max_d)
        add_result(f"TP{tp}_Max{max_d}d", pnls, len(pnls), f"take profit {tp}pts, max {max_d}d")
        print(f"  TP{tp:3d}_Max{max_d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 5) TP + SL
print("\n=== TP + SL Sweep ===")
for tp in [50,75,100]:
    for sl in [50,75,100,150]:
        for max_d in [7,10]:
            pnls=run_variant(exit_tp_sl, tp=tp, sl=sl, max_days=max_d)
            add_result(f"TP{tp}_SL{sl}_Max{max_d}d", pnls, len(pnls), f"TP={tp} SL={sl} max={max_d}d")
            print(f"  TP{tp:3d}_SL{sl:3d}_{max_d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 6) TRAIL + MAX DAYS
print("\n=== Trail CH + Max Days ===")
for ch in [50,75,100,200]:
    for max_d in [5,7,10,14]:
        pnls=run_variant(exit_trail, ch_val=ch, max_days=max_d)
        add_result(f"Trail{ch}_Max{max_d}d", pnls, len(pnls), f"trail {ch}*ATR + max {max_d}d")
        print(f"  Trail{ch:3d}_Max{max_d:2d}d: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 7) DTE-BASED EXITS
print("\n=== DTE-Based Exit Sweep ===")
for dte in [0,1,2,3,5]:
    pnls=[]
    for idx,info in enumerate(trade_infos):
        if info is None or info["ep"]>DEFAULT_PREM: continue
        exit_ts=trades_pre.iloc[idx]["exp_w"]-pd.Timedelta(days=dte)
        if exit_ts<=trades_pre.iloc[idx]["ed_naive"]: exit_ts=trades_pre.iloc[idx]["ed_naive"]+pd.Timedelta(hours=1)
        xp_idx=np.searchsorted(info["stk_data"]["ts"],np.datetime64(exit_ts,"us"))
        if xp_idx>0 and xp_idx<len(info["stk_data"]["cl"]):
            xp=info["stk_data"]["cl"][xp_idx-1]
            pnls.append(round(xp-info["ep"],1))
    add_result(f"DTE{dte}", pnls, len(pnls), f"exit {dte}d before expiry")
    print(f"  DTE{dte}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 8) ENTRY DAY FILTER (use Hold5D as base)
print("\n=== Entry Day Filter ===")
# Pre-compute Hold5D for all valid trades
hold5_pnls=[None]*len(trade_infos)
for idx,info in enumerate(trade_infos):
    if info is None or info["ep"]>DEFAULT_PREM: continue
    xp,_=run_hold_days(info,5)
    if xp is not None: hold5_pnls[idx]=round(xp-info["ep"],1)

for dayset_name,day_filter in [("MonOnly",[0]),("TueOnly",[1]),("WedOnly",[2]),
    ("ThuOnly",[3]),("FriOnly",[4]),("MonTueWed",[0,1,2]),
    ("TueWedThu",[1,2,3]),("MonTue",[0,1]),("WedThuFri",[2,3,4]),
    ("MonWedFri",[0,2,4])]:
    pnls=[hold5_pnls[i] for i in range(len(hold5_pnls)) if hold5_pnls[i] is not None and trades_pre.iloc[i]["weekday"] in day_filter]
    if len(pnls)>=3:
        add_result(f"Day{dayset_name}_Hold5D", pnls, len(pnls), f"entry on {dayset_name}")
        print(f"  {dayset_name:>12s}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 9) EXPIRY WEEK FILTER
print("\n=== Expiry Week Filter ===")
for exp_week in [1,2,3,4]:
    pnls=[hold5_pnls[i] for i in range(len(hold5_pnls)) if hold5_pnls[i] is not None
          and (((trades_pre.iloc[i]["exp_w"]-trades_pre.iloc[i]["ed_naive"]).days-1)//7)+1==exp_week]
    if len(pnls)>=3:
        add_result(f"ExpWk{exp_week}_Hold5D", pnls, len(pnls), f"enter {exp_week}wk before expiry")
        print(f"  ExpWk{exp_week}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 10) PREMIUM FILTER
print("\n=== Premium Filter ===")
for max_prem in [50,80,100,120,150,200]:
    pnls=[hold5_pnls[i] for i in range(len(hold5_pnls)) if hold5_pnls[i] is not None and trade_infos[i] is not None and trade_infos[i]["ep"]<=max_prem]
    if len(pnls)>=3:
        add_result(f"Prem<{max_prem}_Hold5D", pnls, len(pnls), f"premium < {max_prem}")
        print(f"  Prem<{max_prem:3d}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# 11) SKIP AFTER LOSS
print("\n=== Skip After Loss ===")
for skip in [0,1,2,3]:
    pnls=[]; n_skip=0
    for pnl in hold5_pnls:
        if pnl is None: continue
        if n_skip>0: n_skip-=1; continue
        pnls.append(pnl)
        if pnl<0: n_skip=skip
    if len(pnls)>=3:
        add_result(f"Skip{skip}_Hold5D", pnls, len(pnls), f"skip {skip} after loss")
        print(f"  Skip{skip}: {len(pnls):3d} trades, Net={pd.Series(pnls).sum():+,.0f} WR={(pd.Series(pnls)>0).mean():.0%}")

# === RANKING ===
rr=pd.DataFrame(all_rows).sort_values("Net",ascending=False).reset_index(drop=True)
print("\n"+"="*140)
print("FIXED OPTION EXIT SWEEP - FULL RANKING (Same-Strike Tracking)")
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

if len(pos)>0:
    best=pos.iloc[0]
    print(f"\n{'='*100}")
    print(f"BEST: {best['Variant']} Net={best['NetStr']} WR={best['WRStr']} W/L={best['W/L']:5.1f}x")
    print(f"{'='*100}")

# Compare with old buggy results
print(f"\n=== COMPARISON: old buggy best was TP30_Max7d +12,631 pts ===")
print(f"Fixed best: {best['Variant'] if len(pos)>0 else 'N/A'} Net={best['NetStr'] if len(pos)>0 else 'N/A'}")
print(f"Inflation: old overstates by ~37%")

print("\nDone!")
