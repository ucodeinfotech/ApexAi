"""
Fixed option backtest: use SAME strike throughout each trade
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, calendar
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
LOT=50

plt.rcParams.update({'font.size':8,'axes.titlesize':11,'axes.labelsize':9,'figure.dpi':120,'savefig.dpi':150,'font.family':'sans-serif'})

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
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Trades: {len(trades_pre)}")

print("Loading option data...")
con=duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")

# Step 1: Load atm_distance=0 data for entry strike+price
print("Loading atm_distance=0 data...")
df_atm=con.execute("""SELECT timestamp, close, strike FROM options_data_dedup 
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
ORDER BY timestamp""").fetchdf()
df_atm["timestamp"]=pd.to_datetime(df_atm["timestamp"],utc=False)
atm_ts_arr=df_atm["timestamp"].values.astype("datetime64[us]")
atm_cl_arr=df_atm["close"].values.astype(float)
atm_st_arr=df_atm["strike"].values.astype(float)

def lookup(ts):
    i=np.searchsorted(atm_ts_arr,np.datetime64(ts,"us"))
    if i<=0: return 0,atm_cl_arr[0],atm_st_arr[0]
    if i>=len(atm_ts_arr): return len(atm_ts_arr)-1,atm_cl_arr[-1],atm_st_arr[-1]
    return i-1,atm_cl_arr[i-1],atm_st_arr[i-1]

# Step 2: For each trade, find entry info; collect unique strikes
print("Collecting entry strikes...")
trade_info=[]  # (trade_idx, entry_si, entry_strike, entry_price, entry_ed)
for i in range(len(trades_pre)):
    ed=trades_pre.iloc[i]["ed_naive"]
    si,ep,entry_strike=lookup(ed)
    if ep>130: continue
    trade_info.append((i,si,int(entry_strike),ep,ed))

unique_strikes=sorted(set(t[2] for t in trade_info))
print(f"Unique strikes across {len(trade_info)} trades: {len(unique_strikes)}")

# Step 3: Load per-strike data (one query per unique strike)
print("Loading per-strike data...")
strike_cache={}
for stk in unique_strikes:
    df_s=con.execute(f"""SELECT timestamp, close FROM options_data_dedup 
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' 
    AND strike={stk} ORDER BY timestamp""").fetchdf()
    df_s["timestamp"]=pd.to_datetime(df_s["timestamp"],utc=False)
    strike_cache[stk]={
        "ts":df_s["timestamp"].values.astype("datetime64[us]"),
        "cl":df_s["close"].values.astype(float)
    }
con.close()

# Step 4: For each trade, find entry in strike data, run exit logic
print("\nRunning fixed backtest...")
TP_VAL=30; MAX_D=7
pnls=[]; trade_book=[]

def exit_tp_maxd_fixed(stk_data, s_idx, tp, max_days, entry_ts_ns):
    end_ns=stk_data["ts"][s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
    ep=stk_data["cl"][s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(stk_data["cl"]))):
        if stk_data["ts"][i]>end_ns: return stk_data["cl"][i], stk_data["ts"][i]
        if stk_data["cl"][i]-ep>=tp: return stk_data["cl"][i], stk_data["ts"][i]
    return None, None

for i,si,stk,ep,ed in trade_info:
    stk_data=strike_cache[stk]
    # Find where this entry bar falls in the strike data
    entry_ns=atm_ts_arr[si]
    s_idx=np.searchsorted(stk_data["ts"],entry_ns)
    if s_idx>=len(stk_data["cl"]): continue
    # Verify we got the right bar (close should match)
    actual_ep=stk_data["cl"][s_idx]
    if abs(actual_ep-ep)>1:
        # Try the previous bar
        if s_idx>0: s_idx-=1
        actual_ep=stk_data["cl"][s_idx]
    
    xp,xts=exit_tp_maxd_fixed(stk_data,s_idx,TP_VAL,MAX_D,entry_ns)
    if xp is None: continue
    
    pnl=round(xp-actual_ep,1)
    pnls.append(pnl)
    trade_book.append({
        "entry":ed,"exit":pd.Timestamp(xts).tz_localize(None).replace(tzinfo=None) if isinstance(xts,np.datetime64) else xts,
        "strike":stk,"entry_prem":round(actual_ep,1),"exit_prem":round(xp,1),"pnl":pnl,
        "yr":ed.year,"mo":ed.month
    })

print(f"Trades: {len(pnls)}")
print(f"Total PnL: {sum(pnls):+,.0f} pts (Rs {sum(pnls)*50:+,.0f})")
wins=sum(1 for p in pnls if p>0)
print(f"Wins: {wins}/{len(pnls)} ({wins/len(pnls)*100:.1f}%)")
avg_pnl=np.mean(pnls)
print(f"Avg: {avg_pnl:+.1f} | Max: {max(pnls):+.1f} | Min: {min(pnls):+.1f}")
if np.std(pnls)>0:
    sharpe=np.mean(pnls)/np.std(pnls)*np.sqrt(250)
    print(f"Sharpe: {sharpe:.2f}")

# Also compute BUGGY results
print("\n=== Buggy (atm_distance=0 switching strikes) ===")
def exit_tp_maxd_old(s_idx,tp,max_days):
    end_ns=atm_ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us")
    ep=atm_cl_arr[s_idx]
    for i in range(s_idx+1,min(s_idx+3000,len(atm_cl_arr))):
        if atm_ts_arr[i]>end_ns: return atm_cl_arr[i], atm_ts_arr[i]
        if atm_cl_arr[i]-ep>=tp: return atm_cl_arr[i], atm_ts_arr[i]
    return None, None

buggy_pnls=[]
for i,si,stk,ep,ed in trade_info:
    xp,xts=exit_tp_maxd_old(si,TP_VAL,MAX_D)
    if xp is None: continue
    buggy_pnls.append(round(xp-ep,1))

print(f"Buggy trades: {len(buggy_pnls)} Total: {sum(buggy_pnls):+,.0f} WR: {sum(1 for p in buggy_pnls if p>0)/len(buggy_pnls)*100:.1f}%")

# Compare
min_len=min(len(pnls),len(buggy_pnls))
print(f"\n=== COMPARISON (first {min(len(pnls),10)} trades) ===")
for j in range(min(10,min_len)):
    print(f"  {j}: BUGGY={buggy_pnls[j]:+.1f} FIXED={pnls[j]:+.1f} Diff={pnls[j]-buggy_pnls[j]:+.1f}")
print(f"\nBuggy > Fixed: {sum(1 for j in range(min_len) if buggy_pnls[j]>pnls[j])}")
print(f"Fixed > Buggy: {sum(1 for j in range(min_len) if pnls[j]>buggy_pnls[j])}")
same=sum(1 for j in range(min_len) if abs(buggy_pnls[j]-pnls[j])<0.5)
print(f"Same (diff<0.5): {same}")

print(f"\n=== FIXED BACKTEST RESULTS ===")
print(f"Trades: {len(pnls)}")
# === VERIFY SPECIFIC TRADES ===
print("\n=== SPOT CHECK: Trade 1 (buggy show win, fixed shows loss) ===")
i,si,stk,ep,ed = trade_info[1]
print(f"Entry: {ed} strike={stk} prem={ep:.1f}")
stk_data=strike_cache[stk]
s_idx = np.searchsorted(stk_data["ts"], atm_ts_arr[si])
if s_idx > 0: s_idx -= 1
print(f"Same-strike tracking:")
for j in range(s_idx, min(s_idx+20, len(stk_data["ts"]))):
    ts = pd.Timestamp(stk_data["ts"][j]).tz_localize(None)
    cl = stk_data["cl"][j]
    diff = cl - stk_data["cl"][s_idx]
    flag = "TP!" if diff >= 30 else ""
    print(f"  {ts} close={cl:.1f} diff={diff:+.1f} {flag}")

print(f"\nBuggy (atm_distance=0) tracking:")
si_buggy = np.searchsorted(atm_ts_arr, np.datetime64(ed, "us")) - 1
for j in range(si_buggy, min(si_buggy+20, len(atm_ts_arr))):
    ts = pd.Timestamp(atm_ts_arr[j]).tz_localize(None)
    cl = atm_cl_arr[j]
    st = atm_st_arr[j]
    diff = cl - atm_cl_arr[si_buggy]
    flag = "TP!" if diff >= 30 else ""
    print(f"  {ts} strike={st:.0f} close={cl:.1f} diff={diff:+.1f} {flag}")

print(f"\nFIXED final: PnL={pnls[1]:+.1f}")
print(f"BUGGY final: PnL={buggy_pnls[1]:+.1f}")

# === COMPUTE DELTA BETWEEN FIXED AND BUGGY ===
print(f"\n=== FIXED RESULTS ===")
print(f"Trades: {len(pnls)}")
print(f"Total: {sum(pnls):+,.0f} pts (Rs {sum(pnls)*50:+,.0f})")
wins = sum(1 for p in pnls if p > 0)
print(f"Wins: {wins}/{len(pnls)} ({wins/len(pnls)*100:.1f}%)")
print(f"Avg: {np.mean(pnls):+.1f} | Max: {max(pnls):+.1f} | Min: {min(pnls):+.1f}")
print(f"Sharpe: {np.mean(pnls)/np.std(pnls)*np.sqrt(250):.2f}" if np.std(pnls)>0 else "")
print(f"Sum wins: {sum(p for p in pnls if p>0):+,.0f} | Sum losses: {sum(p for p in pnls if p<0):+,.0f}")

print(f"\n=== BUGGY ===")
print(f"Total: {sum(buggy_pnls):+,.0f} pts (Rs {sum(buggy_pnls)*50:+,.0f})")
bwins = sum(1 for p in buggy_pnls if p > 0)
print(f"Wins: {bwins}/{len(buggy_pnls)} ({bwins/len(buggy_pnls)*100:.1f}%)")
print(f"Avg: {np.mean(buggy_pnls):+.1f}")

print(f"\n=== OVERSTATED BY {max(sum(buggy_pnls)-sum(pnls),0):+,.0f} pts ({(sum(buggy_pnls)/sum(pnls)-1)*100:.1f}% inflation) ===")

# Verify the max win
max_trade = trade_book[pnls.index(max(pnls))]
print(f"\n=== MAX WIN TRADE ===")
for k,v in max_trade.items():
    print(f"  {k}: {v}")

# Also check for any remaining issues: which trades have PnL > 50 (beyond TP30)
print(f"\n=== TRADES WITH PNL > 50 (exceeds TP30) ===")
big_trades = [(j, p) for j, p in enumerate(pnls) if p > 50]
print(f"Count: {len(big_trades)}")
for j, p in big_trades[:5]:
    tb = trade_book[j]
    print(f"  Trade {j}: PnL={p:+.1f} entry={tb['entry']} exit={tb['exit']} strike={tb['strike']} "
          f"prem={tb['entry_prem']}->{tb['exit_prem']}")

# Check if these big wins are from time stop or TP
print(f"\nBig wins by exit type:")
tp_wins = sum(1 for p in pnls if 30 <= p < 50)  # TP hit (close to +30)
timeout_wins = sum(1 for p in pnls if p >= 50)  # Beyond TP - likely time stop
print(f"  TP range (30-50): {tp_wins}")
print(f"  Beyond TP (>=50): {timeout_wins}")

# Check: do these >50 trades occur because TP is checked per bar and there's a gap?
print(f"\n=== CHECK: PnL > 50 by time stop vs TP gap ===")
# If TP triggers at exact bar where diff >= 30, max should be ~30 + one bar move
# If time stop triggers, PnL could be anything
# Let me check which exit type each >50 trade is
for j, p in big_trades[:3]:
    tb = trade_book[j]
    t = trade_info[j]
    stk_data = strike_cache[t[2]]
    s_idx = np.searchsorted(stk_data["ts"], atm_ts_arr[t[1]])
    if s_idx > 0: s_idx -= 1
    ep_check = stk_data["cl"][s_idx]
    
    # Check when TP would have triggered
    max_diff = 0
    for k in range(s_idx+1, min(s_idx+3000, len(stk_data["cl"]))):
        diff = stk_data["cl"][k] - ep_check
        max_diff = max(max_diff, diff)
        if diff >= 30:
            break
    else:
        print(f"  Trade {j}: TIME STOP ({p:+.1f} pts, entry prem={tb['entry_prem']:.1f}, exit prem={tb['exit_prem']:.1f}, max diff during hold={max_diff:.1f})")
        continue
    
    # TP hit - check the actual diff at exit
    exit_idx = t[1]  # This won't match. Let me compute differently
    # Find where TP exited in strike data
    for k in range(s_idx+1, min(s_idx+3000, len(stk_data["cl"]))):
        if stk_data["cl"][k] - ep_check >= 30:
            actual_diff = stk_data["cl"][k] - ep_check
            print(f"  Trade {j}: TP EXIT at diff={actual_diff:+.1f} (bar close jumped from previous)")
            break
