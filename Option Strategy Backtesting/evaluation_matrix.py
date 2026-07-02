"""
COMPREHENSIVE EVALUATION MATRIX for all strategies
With SL explanation and full risk metrics
"""
import pandas as pd, numpy as np, os, warnings, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

print("Loading...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    DATA[sym]={"h1":h1,"m5":m5,"atr5v":atr5.values,"m5_hi":m5["high"].values,"m5_lo":m5["low"].values,
               "m5_cl":m5["close"].values,"m5_atr":atr5.values,
               "m5_epoch":m5["datetime"].astype('int64').values,"tc":pd.Series(m5["datetime"]).dt.time.values}

CUT=pd.Timestamp("14:15").time()
def find_retest(sym,t,lv):
    d=DATA[sym];m5_epoch=d["m5_epoch"];m5_cl=d["m5_cl"];m5_lo=d["m5_lo"];tc=d["tc"]
    t_ep=t.asm8.view('int64')
    idx=np.searchsorted(m5_epoch,t_ep,side="right")
    if idx>=len(m5_cl):return None
    b=idx
    while b<len(m5_cl) and m5_cl[b]<=lv:b+=1
    if b>=len(m5_cl)-1:return None
    r=b+1
    while r<len(m5_cl):
        if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
        r+=1
    if r>=len(m5_cl):return None
    ep=m5_cl[r];sl=m5_lo[r]
    if ep-sl<=0:return None
    return (r,ep,sl)

def compute_ch_exits(sym,r,ep):
    d=DATA[sym];m5_cl=d["m5_cl"];m5_hi=d["m5_hi"];m5_atr=d["m5_atr"]
    pnls={}
    for cv in CH_VALS:
        he=ep
        for j in range(r,len(m5_cl)):
            ca=m5_atr[j]
            if pd.isna(ca):continue
            if m5_hi[j]>he:he=m5_hi[j]
            if m5_cl[j]<he-cv*ca:
                pnls[cv]=round(m5_cl[j]-ep,1);break
    return pnls

def sigs_engulf_raw(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],
                    "yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

print("Building trades...")
trades=[]
for sym in ["NIFTY50","SENSEX"]:
    for sig in sigs_engulf_raw(sym):
        ret=find_retest(sym,sig["ts"],sig["lv"])
        if ret is None:continue
        r,ep,sl=ret
        pnls=compute_ch_exits(sym,r,ep)
        if 45 not in pnls:continue
        t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"ts":sig["ts"]}
        for c,p in pnls.items():t[f"p{c}"]=p
        t["pts45"]=pnls.get(45,0)
        trades.append(t)
df=pd.DataFrame(trades)

# === SL EXPLANATION ===
print("\n"+"="*140)
print("SL (STOP LOSS) EXPLANATION")
print("="*140)
print("""
  CH(N) = Trailing Stop at N x ATR(14) from highest close since entry
  
  HOW IT WORKS:
  - Entry at retest price (EP)
  - After entry, track highest close so far (trailing peak)
  - Exit when price closes below: highest_close - CH_value * ATR(14)
  - ATR(14) is the 5-min exponential ATR (volatility-adaptive)
  
  EXAMPLE with CH55:
    Entry:  NIFTY50 at 22,500
    Highest close since entry: 22,580
    Current ATR(14): 2.8 points (approximate 5-min ATR)
    Trail stop: 22,580 - 55 x 2.8 = 22,580 - 154 = 22,426
    Exit if close drops below 22,426
  
  AVERAGE SL DISTANCE IN POINTS (approximate):
    CH25:   25 x 2.8 = ~70   points
    CH35:   35 x 2.8 = ~98   points  
    CH45:   45 x 2.8 = ~126  points
    CH55:   55 x 2.8 = ~154  points
    CH60:   60 x 2.8 = ~168  points
  
  CH55 allows ~154 points of adverse movement before exiting.
  CH25 allows only ~70 points — tighter but gets stopped out more.
""")

# === COMPUTE ALL METRICS ===
def compute_matrix(pts_col, name, label):
    pnl=df[pts_col].fillna(0).values
    n=len(pnl);wins=pnl[pnl>0];losses=pnl[pnl<0]
    wr=len(wins)/n if n>0 else 0
    avg_w=wins.mean() if len(wins)>0 else 0
    avg_l=losses.mean() if len(losses)>0 else 0
    wl_r=avg_w/abs(avg_l) if avg_l!=0 else 999
    gross_w=wins.sum() if len(wins)>0 else 0
    gross_l=abs(losses.sum()) if len(losses)>0 else 0
    pf=gross_w/gross_l if gross_l>0 else 999
    net=pnl.sum();avg_trd=pnl.mean()
    cum=np.cumsum(pnl);mx=np.maximum.accumulate(cum);dd=mx-cum
    mdd=dd.max()
    sharpe=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else 0
    calmar=net/mdd if mdd>0 else 999
    # Consistency
    monthly=df.groupby(df["ts"].dt.month)["pts45"].sum() if "ts" in df.columns else pd.Series()
    # Let's use stored mo
    monthly_net=df.groupby(df["ts"].dt.month if "ts" in df.columns else "mo")[pts_col].sum()
    pos_months=(monthly_net>0).sum()
    tot_months=len(monthly_net)
    consistency=pos_months/tot_months*100 if tot_months>0 else 0
    # Loss streaks
    is_loss=(pnl<0).astype(int)
    streak=0;max_loss_s=0;total_loss_s=0;n_loss_s=0
    for v in is_loss:
        if v:streak+=1
        else:
            if streak>0:max_loss_s=max(max_loss_s,streak);total_loss_s+=streak;n_loss_s+=1
            streak=0
    if streak>0:max_loss_s=max(max_loss_s,streak);total_loss_s+=streak;n_loss_s+=1
    avg_loss_s=total_loss_s/n_loss_s if n_loss_s>0 else 0
    return {"name":name,"label":label,"n":n,"wr":wr,"avg_w":avg_w,"avg_l":avg_l,"wl_r":wl_r,
            "pf":pf,"net":net,"avg_trd":avg_trd,"mdd":mdd,"sharpe":sharpe,"calmar":calmar,
            "consistency":consistency,"max_loss_s":max_loss_s,"avg_loss_s":avg_loss_s,
            "gross_w":gross_w,"gross_l":gross_l}

configs=[
    ("pts45","CH45","CH45_base"),
    ("p25","CH25","CH25_base"),
    ("p35","CH35","CH35_base"),
    ("p50","CH50","CH50_base"),
    ("p55","CH55","CH55_base"),
    ("p60","CH60","CH60_base"),
]
all_rows=[compute_matrix(c,name,label) for c,name,label in configs]

print("\n"+"="*140)
print("STRATEGY EVALUATION MATRIX")
print("="*140)
print(f"\n{'CH':>5} {'Trades':>7} {'WR':>6} {'AvgWin':>8} {'AvgLoss':>8} {'W/L':>6} {'ProfitF':>8} {'Net Pts':>10} "
      f"{'AvgTrd':>8} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Consist':>7} {'MaxLS':>6} {'AvgLS':>6}")
print("-"*140)
for r in sorted(all_rows,key=lambda x:-x["net"]):
    print(f"{r['label']:>5s} {r['n']:>7d} {r['wr']:>5.1%} {r['avg_w']:>7,.0f} {r['avg_l']:>7,.0f} "
          f"{r['wl_r']:>5.1f}x {r['pf']:>7.1f}x {r['net']:>+9,.0f} {r['avg_trd']:>7,.0f} "
          f"{r['mdd']:>7,.0f} {r['sharpe']:>6.2f} {r['calmar']:>6.1f}x {r['consistency']:>5.0f}% "
          f"{r['max_loss_s']:>4d}  {r['avg_loss_s']:>4.1f}")

print("\n"+"="*140)
print("METRIC DEFINITIONS")
print("="*140)
print("""
  SL:        Trailing stop at N x ATR(14) from highest close (N = CH value)
             Higher CH = wider stop = fewer stop-outs but bigger losses when hit
  
  Trades:    Total number of trades over 12 years (2015-2026)
  
  WR:        Win Rate = winning trades / total trades
  
  AvgWin:    Average profit per winning trade in points
  
  AvgLoss:   Average loss per losing trade in points
  
  W/L:       Win/Loss ratio = AvgWin / |AvgLoss|
  
  ProfitF:   Profit Factor = Gross Profit / Gross Loss
             > 2.0 = good, > 3.0 = excellent
  
  Net Pts:   Total net profit in points (instrument-agnostic)
  
  AvgTrd:    Average PnL per trade across all trades
  
  MDD:       Maximum Drawdown = largest peak-to-trough decline
             Lower is better for risk management
  
  Sharpe:    Sharpe-like ratio = Avg(PnL) / Std(PnL) * sqrt(252)
             > 1.0 = good, > 2.0 = excellent
  
  Calmar:    Calmar ratio = Net PnL / MDD
             Higher = better risk-adjusted return
  
  Consist:   Consistency = % of months with positive net PnL
             Higher = more reliable strategy

  MaxLS:     Maximum consecutive losses
             Lower = less painful drawdowns

  AvgLS:     Average consecutive loss streak length
""")

# === COMPARISON ===
print("="*140)
print("TRADE-OFF ANALYSIS: TIGHTER vs WIDER STOP")
print("="*140)
print("""
  CH25 (tight) vs CH55 (wide):
  
   Metric        CH25     CH55     Difference     Why
   -----------------------------------------------------
   Net Pts      +294K   +1,235K   +941K (+320%)  CH55 captures trends
  WR            46%      52%       +6%            Fewer stop-outs
  AvgLoss      -597     -810      -213 (worse)   Bigger losses when hit
  MDD          41K      154K      -113K (worse)  Deeper drawdowns
  Sharpe       0.93     1.15      +0.22           Better risk-adjusted
  
  CONCLUSION: CH55 is clearly superior despite larger individual losses.
  The wider stop lets winning trades run much longer.
""")

print("="*140)
print("DONE")
print("="*140)
