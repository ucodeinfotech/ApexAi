"""
Investigate trade book anomalies
"""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

# Load option data
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
df_opt=con.execute("SELECT timestamp, close FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp").fetchdf()
con.close()
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
cl_arr=df_opt["close"].values.astype(float)

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
                           "yr":ts.year,"mo":ts.month}); break

trades_all=pd.DataFrame(trades)
trades_all["ed_naive"]=trades_all["entry_dt"].dt.tz_localize(None)
trades_pre=trades_all[trades_all["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)

def lookup(ts):
    i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
    if i<=0: return 0,cl_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1]
    return i-1,cl_arr[i-1]

def exit_tp_maxd(s_idx,tp,max_d):
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_d*86400*1e6),"us")
    ep=cl_arr[s_idx]
    for i in range(s_idx+1,min(s_idx+3000,len(cl_arr))):
        if ts_arr[i]>end_ns: return cl_arr[i], ts_arr[i]
        if cl_arr[i]-ep>=tp: return cl_arr[i], ts_arr[i]
    return None, None

# INVESTIGATION 1: exit <= entry time
print("=== INVESTIGATION: exit <= entry time ===")
issues=0
for i in range(len(trades_pre)):
    ep=lookup(trades_pre.iloc[i]["ed_naive"])[1]
    if ep>130: continue
    si=lookup(trades_pre.iloc[i]["ed_naive"])[0]
    xp,xts=exit_tp_maxd(si,30,7)
    if xp is None: continue
    ed=trades_pre.iloc[i]["ed_naive"]
    xd=pd.Timestamp(xts).tz_localize(None) if isinstance(xts,np.datetime64) else xts
    if xd<=ed:
        issues+=1
        print(f"  Trade {i}: entry={ed} exit={xd}")
        print(f"    s_idx={si} entry_bar_ts={pd.Timestamp(ts_arr[si])}")
        print(f"    ep={ep} xp={xp}")
        if si+1<len(ts_arr):
            print(f"    next_bar_ts={pd.Timestamp(ts_arr[si+1])}")
if issues==0: print("  None found - no issues")

# INVESTIGATION 2: near-zero premium entries
print("\n=== INVESTIGATION: near-zero premium entries ===")
low_ct=0
for i in range(len(trades_pre)):
    ep=lookup(trades_pre.iloc[i]["ed_naive"])[1]
    if ep is not None and ep<5 and ep>=0.01:
        low_ct+=1
        if low_ct<=5:
            print(f"  Trade signal {i}: {trades_pre.iloc[i]['ed_naive']} premium={ep:.2f}")
print(f"  Total low-premium signals (<5 pts): {low_ct}")

# INVESTIGATION 3: duplicate signal timestamps
print("\n=== INVESTIGATION: duplicate signal timestamps ===")
dup_mask=trades_all.duplicated(subset=["ed_naive"],keep=False)
dups=trades_all[dup_mask].sort_values(["ed_naive"])
print(f"  Total duplicate timestamps: {len(dups)}")
for ts_key,grp in dups.groupby("ed_naive"):
    print(f"  Timestamp: {ts_key}")
    for _,r in grp.iterrows():
        print(f"    entry={r['entry_dt']} entry_px={r['entry_px']:.1f} exit_px={r['exit_px']:.1f}")

# INVESTIGATION 4: SENSEX long duration trades
print("\n=== INVESTIGATION: SENSEX long-duration trades ===")
h1s=pd.read_csv("SENSEX_ONE_HOUR.csv",parse_dates=["datetime"])
m5s=pd.read_csv("SENSEX_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1s,m5s]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
tr5s=pd.concat([m5s["high"]-m5s["low"],abs(m5s["high"]-m5s["close"].shift(1)),abs(m5s["low"]-m5s["close"].shift(1))],axis=1).max(axis=1)
atr5s=tr5s.ewm(span=14,min_periods=14,adjust=False).mean()
mes=m5s["datetime"].astype("int64").values
bs=(h1s["close"]-h1s["open"]).abs(); gs=h1s["close"]>h1s["open"]; rrs=h1s["close"]<h1s["open"]
long_trades=0
for i in range(1,len(h1s)):
    if not (rrs.iloc[i-1] and gs.iloc[i]): continue
    if h1s["open"].iloc[i]>h1s["close"].iloc[i-1] or h1s["close"].iloc[i]<h1s["open"].iloc[i-1]: continue
    if bs.iloc[i]<bs.iloc[i-1]*0.5 or h1s["datetime"].iloc[i].hour==9: continue
    lv=h1s["high"].iloc[i]; ts=h1s["datetime"].iloc[i]
    idx=np.searchsorted(mes,ts.asm8.view("int64"),side="right")
    if idx>=len(m5s["close"]): continue
    bi=idx
    while bi<len(m5s["close"]) and m5s["close"].iloc[bi]<=lv: bi+=1
    if bi>=len(m5s["close"])-1: continue
    ri=bi+1
    while ri<len(m5s["close"]):
        if m5s["low"].iloc[ri]<lv and m5s["close"].iloc[ri]>lv and pd.Series(m5s["datetime"]).dt.time.iloc[ri]<CUT: break
        ri+=1
    if ri>=len(m5s["close"]): continue
    ed=m5s["datetime"].iloc[ri]; he=m5s["close"].iloc[ri]
    for j in range(ri,len(m5s["close"])):
        ca=atr5s.iloc[j]
        if pd.isna(ca): continue
        if m5s["high"].iloc[j]>he: he=m5s["high"].iloc[j]
        if m5s["close"].iloc[j]<he-55*ca:
            dur=(m5s["datetime"].iloc[j]-ed).total_seconds()/3600
            if dur>240:  # >10 days
                long_trades+=1
                if long_trades<=3:
                    pt=m5s["close"].iloc[j]-m5s["close"].iloc[ri]
                    print(f"  Entry={ed} dur={dur:.0f}h ({dur/24:.1f}d) pnl={pt:+.1f} span={j-ri} bars")
            break

atr_stats=(atr5s.min(),atr5s.mean(),atr5s.max())
print(f"  Long trades (>10d): {long_trades}")
print(f"  SENSEX 5M ATR: min={atr_stats[0]:.1f} avg={atr_stats[1]:.1f} max={atr_stats[2]:.1f}")
print(f"  55*ATR range: {55*atr_stats[0]:.0f} to {55*atr_stats[2]:.0f}")

# Compare NIFTY vs SENSEX ATR
nifty_atr=atr5
sensex_atr=atr5s
print(f"\n  NIFTY 5M ATR: min={nifty_atr.min():.1f} avg={nifty_atr.mean():.1f} max={nifty_atr.max():.1f}")
print(f"  SENSEX is {sensex_atr.mean()/nifty_atr.mean():.1f}x NIFTY volatility (ATR ratio)")
print(f"  CH{55} on SENSEX = CH{round(55/(sensex_atr.mean()/nifty_atr.mean()))} on NIFTY (equivalent)")

print("\nDone!")
