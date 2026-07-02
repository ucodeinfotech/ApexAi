"""
Deep debug: trades where option wins but spot already losing at exit
"""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

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
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],
                "spot_entry":ep_,"spot_exit":m5["close"].iloc[j],
                "spot_pnl":round(m5["close"].iloc[j]-ep_,1),
                "ed_naive":ed.tz_localize(None),
                "xd_naive":m5["datetime"].iloc[j].tz_localize(None)}); break
trades=pd.DataFrame(trades)

con=duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")
q = "SELECT timestamp, close, atm_distance, strike, spot_price FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' ORDER BY timestamp"
df_opt_all=con.execute(q).fetchdf()
con.close()
df_opt_all["timestamp"]=pd.to_datetime(df_opt_all["timestamp"],utc=False)

# Filter ATM only
df_atm=df_opt_all[df_opt_all["atm_distance"]==0].copy()
ts_arr=df_atm["timestamp"].values.astype("datetime64[us]")
cl_arr=df_atm["close"].values.astype(float)
sp_arr=df_atm["spot_price"].values.astype(float)
st_arr=df_atm["strike"].values.astype(float)

def lookup(ts):
    i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
    if i<=0: return 0,cl_arr[0],sp_arr[0],st_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1],sp_arr[-1],st_arr[-1]
    return i-1,cl_arr[i-1],sp_arr[i-1],st_arr[i-1]

# Find suspicious trades: option wins but spot losing at option exit
suspicious=[]
for i in range(len(trades)):
    ed=trades.iloc[i]["ed_naive"]
    if ed < pd.Timestamp("2021-06-14"): continue
    si,opt_ep_,spot_at_entry_,strike_entry=lookup(ed)
    opt_ep=opt_ep_
    if opt_ep>130: continue
    
    # Option exit
    end_ns=ts_arr[si]+np.timedelta64(int(7*86400*1e6),"us")
    opt_xp=None; opt_xts=None; opt_xsi=None
    ep2=cl_arr[si]
    for j in range(si+1,min(si+3000,len(ts_arr))):
        if ts_arr[j]>end_ns:
            opt_xp=cl_arr[j]; opt_xts=ts_arr[j]; opt_xsi=j; break
        if cl_arr[j]-ep2>=30:
            opt_xp=cl_arr[j]; opt_xts=ts_arr[j]; opt_xsi=j; break
    if opt_xp is None: continue
    
    opt_pnl=round(opt_xp-opt_ep,1)
    spot_pnl=trades.iloc[i]["spot_pnl"]
    
    if opt_pnl>0 and spot_pnl<=0:
        opt_exit_ts=pd.Timestamp(opt_xts).tz_localize(None)
        spot_exit_ts=trades.iloc[i]["xd_naive"]
        
        # Check what spot was at option exit time
        oei=np.searchsorted(me,opt_exit_ts.asm8.view("int64"))
        spot_at_opt_exit=m5["close"].iloc[oei-1] if oei>0 and oei<len(m5["close"]) else None
        spot_pnl_at_opt_exit=round(spot_at_opt_exit-trades.iloc[i]["spot_entry"],1) if spot_at_opt_exit else None
        
        # Data at entry and exit bars
        entry_bar=df_atm.iloc[si]
        exit_bar=df_atm.iloc[opt_xsi]
        
        suspicious.append({
            "i":i,"entry":ed,"exit":opt_exit_ts,
            "opt_ep":opt_ep,"opt_xp":opt_xp,"opt_pnl":opt_pnl,
            "strike_entry":strike_entry,
            "spot_entry":spot_at_entry_,"spot_at_opt_exit":spot_at_opt_exit,
            "spot_pnl_at_opt_exit":spot_pnl_at_opt_exit,
            "spot_pnl_final":spot_pnl,
            "entry_close":entry_bar["close"],
            "entry_spot":entry_bar["spot_price"],
            "entry_ts":entry_bar["timestamp"],
            "exit_close":exit_bar["close"],
            "exit_spot":exit_bar["spot_price"],
            "exit_ts":exit_bar["timestamp"],
            "x_spot_exit":spot_exit_ts
        })

print(f"Suspicious trades (opt wins but spot was also losing at opt exit): {len([s for s in suspicious if s['spot_pnl_at_opt_exit'] is not None and s['spot_pnl_at_opt_exit']<0])}")
print(f"Legitimate trades (opt wins early, spot later reverses): {len([s for s in suspicious if s['spot_pnl_at_opt_exit'] is not None and s['spot_pnl_at_opt_exit']>=0])}")

print("\n=== DETAILED SUSPICIOUS TRADES ===")
for s in suspicious:
    if s["spot_pnl_at_opt_exit"] is None or s["spot_pnl_at_opt_exit"]>=0: continue
    print(f"\nTrade #{s['i']}:")
    print(f"  Entry: {s['entry']}  |  Opt exit: {s['exit']}  |  Spot final exit: {s['x_spot_exit']}")
    print(f"  Option: {s['opt_ep']:.1f} -> {s['opt_xp']:.1f} = {s['opt_pnl']:+.1f} pts")
    print(f"  Spot: entry={s['spot_entry']:.1f} @opt_exit={s['spot_at_opt_exit']:.1f} ({s['spot_pnl_at_opt_exit']:+.1f}) final={s['spot_pnl_final']:+.1f}")
    print(f"  Option DB entry bar: ts={s['entry_ts']} close={s['entry_close']:.1f} spot={s['entry_spot']:.1f} strike={s['strike_entry']:.0f}")
    print(f"  Option DB exit bar:  ts={s['exit_ts']} close={s['exit_close']:.1f} spot={s['exit_spot']:.1f}")
    
    # Verify: is option close price at exit reasonable given spot?
    spot_move=s["exit_spot"]-s["entry_spot"]
    opt_move=s["opt_xp"]-s["opt_ep"]
    print(f"  Spot move: {spot_move:+.1f} | Opt move: {opt_move:+.1f} | Delta: {opt_move/spot_move:.2f}" if spot_move!=0 else f"  Spot move: {spot_move:+.1f} | Opt move: {opt_move:+.1f}")

# Also check the option premium move during entry-exit period
print("\n\n=== VERIFICATION: Trades where option wins during big spot drop ===")
for s in suspicious[:5]:
    if s["entry_spot"]-s["exit_spot"]>0 and s["opt_pnl"]>0:
        print(f"\nTrade #{s['i']}: Spot DROPPED {s['entry_spot']-s['exit_spot']:.0f} pts, but option GAINED {s['opt_pnl']:.1f} pts")
        print("  This is IMPOSSIBLE for ATM calls. Data error or lookup bug!")
        # Let's check if strike changed (maybe we're comparing different strikes)
        # Get all option data around the exit time for this trade
        entry_bar_idx=lookup(s["entry"])[0]
        exit_bar_idx=None
        end_ns2=ts_arr[entry_bar_idx]+np.timedelta64(int(7*86400*1e6),"us")
        for j in range(entry_bar_idx+1,min(entry_bar_idx+3000,len(ts_arr))):
            if ts_arr[j]>end_ns2:
                exit_bar_idx=j; break
            if cl_arr[j]-cl_arr[entry_bar_idx]>=30:
                exit_bar_idx=j; break
        
        if exit_bar_idx:
            # Check if the data has weird jumps
            print(f"  Entry bar: idx={entry_bar_idx} ts={df_atm.iloc[entry_bar_idx]['timestamp']} close={df_atm.iloc[entry_bar_idx]['close']:.1f} spot={df_atm.iloc[entry_bar_idx]['spot_price']:.1f} strike={df_atm.iloc[entry_bar_idx]['strike']:.0f}")
            print(f"  Exit bar:  idx={exit_bar_idx} ts={df_atm.iloc[exit_bar_idx]['timestamp']} close={df_atm.iloc[exit_bar_idx]['close']:.1f} spot={df_atm.iloc[exit_bar_idx]['spot_price']:.1f} strike={df_atm.iloc[exit_bar_idx]['strike']:.0f}")
            
            # Check a few bars around exit to see if there's a discontinuity
            for k in range(max(0,exit_bar_idx-3), min(len(df_atm), exit_bar_idx+3)):
                r=df_atm.iloc[k]
                print(f"    [{k}] {r['timestamp']} close={r['close']:.1f} spot={r['spot_price']:.1f} strike={r['strike']:.0f}")
