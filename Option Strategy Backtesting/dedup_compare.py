"""
Compare original vs deduped option backtest
"""
import duckdb, pandas as pd, numpy as np, os, time, warnings, calendar
from datetime import timedelta
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

def run_backtest(table_name, label):
    print(f"\n=== {label} (table: {table_name}) ===")
    t0=time.time()
    
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
                trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month}); break
    trades=pd.DataFrame(trades)
    trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
    exp_w_l=[]
    for _,r in trades.iterrows():
        d=r["ed_naive"].date()
        da=(3-d.weekday())%7; da=da if da>0 else 7
        exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
    trades["exp_w"]=exp_w_l
    trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
    
    # Load from DB
    con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
    q = "SELECT timestamp, close FROM " + table_name + \
        " WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp"
    df_opt=con.execute(q).fetchdf()
    con.close()
    
    df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
    ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
    cl_arr=df_opt["close"].values.astype(float)
    
    print(f"  Loaded {len(cl_arr):,} option rows ({label})")
    
    def lookup(ts):
        i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
        if i<=0: return 0,cl_arr[0]
        if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1]
        return i-1,cl_arr[i-1]
    
    ep_vals=np.array([lookup(ed)[1] for ed in trades_pre["ed_naive"]])
    si_vals=np.array([lookup(ed)[0] for ed in trades_pre["ed_naive"]])
    
    def exit_tp_maxd(s_idx,tp,max_days):
        end_ns=ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
        ep=cl_arr[s_idx]
        for i in range(s_idx+1,min(s_idx+3000,len(cl_arr))):
            if ts_arr[i]>end_ns: return cl_arr[i]
            if cl_arr[i]-ep>=tp: return cl_arr[i]
        return None
    
    # Run strategy
    pnls=[]; PM=130; TP=30; MD=7
    for i in range(len(ep_vals)):
        if ep_vals[i]>PM: continue
        xp=exit_tp_maxd(si_vals[i],TP,MD)
        if xp is None: continue
        pnls.append(round(xp-ep_vals[i],1))
    
    p=np.array(pnls); net=p.sum(); wr=(p>0).mean(); n=len(p)
    cum=np.cumsum(p); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999; avg=p.mean()
    t1=time.time()
    print(f"  Trades: {n}, Net: {net:+,.0f} pts, WR: {wr:.0%}, Avg: {avg:+.1f}, MDD: {mdd:,.0f}, Calmar: {calmar:.1f}x, Time: {t1-t0:.1f}s")
    return {"trades":n, "net":net, "wr":wr, "mdd":mdd, "calmar":calmar, "avg":avg, "pnls":pnls}

# Run both
r1=run_backtest("options_data", "ORIGINAL (with dupes)")
r2=run_backtest("options_data_dedup", "DEDUPED")

# Compare
print("\n" + "="*60)
print("COMPARISON: ORIGINAL vs DEDUPED")
print("="*60)
print(f"{'Metric':<20} {'Original':>12} {'Deduped':>12} {'Match?':>8}")
print("-"*52)
for k in ["trades","net","wr","avg","mdd","calmar"]:
    v1=r1[k]; v2=r2[k]
    if isinstance(v1,float):
        match="YES" if abs(v1-v2)<0.001 else "DIFF"
        print(f"{k:<20} {v1:>12.2f} {v2:>12.2f} {match:>8}")
    else:
        match="YES" if v1==v2 else "DIFF"
        print(f"{k:<20} {v1:>12} {v2:>12} {match:>8}")

# Compare individual PnLs
pnls1=np.array(r1["pnls"]); pnls2=np.array(r2["pnls"])
print(f"\nPnL comparison: {len(pnls1)} trades")
if len(pnls1)==len(pnls2):
    diff=pnls1-pnls2
    maxdiff=abs(diff).max()
    ndiff=(abs(diff)>0.01).sum()
    print(f"Max difference: {maxdiff:+.2f}")
    print(f"Trades with difference: {ndiff} / {len(pnls1)}")
    if ndiff>0:
        for i in np.where(abs(diff)>0.01)[0][:5]:
            print(f"  Trade {i}: orig={pnls1[i]:+.1f} dedup={pnls2[i]:+.1f} diff={diff[i]:+.1f}")
        print("DIFFERENCES FOUND - investigation needed")
    else:
        print("ALL IDENTICAL - dedup has zero effect on results")

# Cleanup
print("\nNote: options_data_dedup table kept for verification")
