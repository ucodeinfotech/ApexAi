"""
Check option-win vs spot-loss trades in detail
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
                "yr":ts.year,"mo":ts.month}); break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"]=trades["exit_dt"].dt.tz_localize(None)

con=duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")
q = "SELECT timestamp, close FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp"
df_opt=con.execute(q).fetchdf()
con.close()
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
cl_arr=df_opt["close"].values.astype(float)

def lookup(ts):
    i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
    if i<=0: return 0,cl_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1]
    return i-1,cl_arr[i-1]

print("Checking option-win vs spot-loss trades...")
count_opt_win_spot_loss=0
count_opt_loss_spot_win=0
count_both_win=0
count_both_loss=0

examples=[]
for i in range(len(trades)):
    ed=trades.iloc[i]["ed_naive"]
    if ed < pd.Timestamp("2021-06-14"): continue
    opt_ep=lookup(ed)[1]
    if opt_ep>130: continue
    
    si=lookup(ed)[0]
    end_ns=ts_arr[si]+np.timedelta64(int(7*86400*1e6),"us")
    ep2=cl_arr[si]
    opt_xp=None; opt_xts=None
    for j in range(si+1,min(si+3000,len(ts_arr))):
        if ts_arr[j]>end_ns:
            opt_xp=cl_arr[j]; opt_xts=ts_arr[j]; break
        if cl_arr[j]-ep2>=30:
            opt_xp=cl_arr[j]; opt_xts=ts_arr[j]; break
    if opt_xp is None: continue
    
    opt_pnl=round(opt_xp-ep2,1)
    spot_pnl=trades.iloc[i]["spot_pnl"]
    
    if opt_pnl>0 and spot_pnl>0: count_both_win+=1
    elif opt_pnl>0 and spot_pnl<=0: count_opt_win_spot_loss+=1
    elif opt_pnl<=0 and spot_pnl>0: count_opt_loss_spot_win+=1
    else: count_both_loss+=1
    
    if opt_pnl>0 and spot_pnl<=0 and len(examples)<4:
        opt_exit_ts=pd.Timestamp(opt_xts).tz_localize(None)
        spot_exit_ts=trades.iloc[i]["xd_naive"]
        # Spot price at option exit time
        spot_at_opt_exit=None
        if spot_exit_ts>opt_exit_ts:
            oei=np.searchsorted(me,opt_exit_ts.asm8.view("int64"))
            if oei>0 and oei<len(m5["close"]):
                spot_at_opt_exit=m5["close"].iloc[oei-1]
        spot_at_opt_pnl=round(spot_at_opt_exit-trades.iloc[i]["spot_entry"],1) if spot_at_opt_exit else None
        examples.append({
            "i":i,"entry":ed,"opt_ep":ep2,"opt_pnl":opt_pnl,
            "opt_exit":opt_exit_ts,"spot_exit":spot_exit_ts,
            "spot_pnl":spot_pnl,"spot_pnl_at_opt_exit":spot_at_opt_pnl
        })

print(f"Both win:   {count_both_win}")
print(f"Opt win, spot loss: {count_opt_win_spot_loss}")
print(f"Opt loss, spot win: {count_opt_loss_spot_win}")
print(f"Both loss:  {count_both_loss}")
total=count_both_win+count_opt_win_spot_loss+count_opt_loss_spot_win+count_both_loss
print(f"Total: {total}")

print("\nSample trades where option wins but spot loses:")
for ex in examples:
    print(f"  Trade #{ex['i']}: entry={ex['entry']}")
    print(f"    Opt prem={ex['opt_ep']:.1f} exit={ex['opt_exit']} pnl={ex['opt_pnl']:+.1f}")
    print(f"    Spot exit={ex['spot_exit']} pnl={ex['spot_pnl']:+.1f}")
    if ex["spot_pnl_at_opt_exit"] is not None:
        if ex["spot_pnl_at_opt_exit"]>0:
            print(f"    ** Spot was WINNING ({ex['spot_pnl_at_opt_exit']:+.1f}) at option exit, but later turned to loss!")
        else:
            print(f"    ** Spot was also LOSING at option exit ({ex['spot_pnl_at_opt_exit']:+.1f})")
    print()

# Summary: is this a bug or legitimate?
print("=== ANALYSIS ===")
print(f"Of {count_opt_win_spot_loss} option-win/spot-loss trades:")
print(f"- If spot was winning at option exit time, it means spot later reversed: LEGITIMATE")
print(f"- If spot was already losing at option exit time, option TP triggered on option-specific move: SUSPICIOUS")
