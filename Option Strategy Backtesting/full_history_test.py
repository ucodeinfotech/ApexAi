"""
FULL HISTORY BACKTEST: W/L Sizing + DynCH(Month)
Tests from 2015-2026 (NIFTY50) and 2016-2026 (SENSEX)
Walk-forward: each year uses optimal CH from prior years
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]; CH_BASE=45; CH_RANGE=10

def build_trades():
    all_t=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
        m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
        hl=h1["high"]-h1["low"];hpc=abs(h1["high"]-h1["close"].shift(1));lpc=abs(h1["low"]-h1["close"].shift(1))
        tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);h1["atr14"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()
        a14=h1["atr14"].values;a20=pd.Series(a14).rolling(20).mean().values
        h1["atr14_pct"]=h1["atr14"]/h1["close"]*100;h1["range_pct"]=(h1["high"]-h1["low"])/h1["open"]*100
        for p in [5,20,50]:h1[f"ema{p}"]=h1["close"].ewm(span=p,adjust=False).mean()
        h1["ema_pos"]=(h1["ema5"]>h1["ema20"]).astype(int)
        hl5=m5["high"]-m5["low"];hpc5=abs(m5["high"]-m5["close"].shift(1));lpc5=abs(m5["low"]-m5["close"].shift(1))
        tr5=pd.concat([hl5,hpc5,lpc5],axis=1).max(axis=1);m5_atr=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
        atr5=m5_atr.values;m5_hi=m5["high"].values;m5_lo=m5["low"].values;m5_cl=m5["close"].values;m5_du=m5["datetime"].values
        prev_red=np.roll(h1["close"].values<h1["open"].values,1);prev_red[0]=False
        tc=pd.Series(m5["datetime"]).dt.time.values;CUT=pd.Timestamp("14:15").time()
        bl=50 if "NIFTY" in sym else 10
        for i in range(60,len(h1)):
            if not(prev_red[i] and h1["close"].values[i]>h1["open"].values[i]):continue
            if not(h1["open"].values[i]<=h1["close"].values[i-1] and h1["close"].values[i]>=h1["open"].values[i-1]):continue
            if h1["high"].values[i]-h1["low"].values[i]<0.5*(h1["high"].values[i-1]-h1["low"].values[i-1]):continue
            if h1["datetime"].iloc[i].hour==9:continue
            lv=h1["high"].values[i];tu=h1["datetime"].values[i]
            idx=np.searchsorted(m5_du,tu,side="right")
            if idx>=len(m5):continue
            b=idx
            while b<len(m5) and m5_cl[b]<=lv:b+=1
            if b>=len(m5)-1:continue
            r=b+1
            while r<len(m5):
                if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT:break
                r+=1
            if r>=len(m5):continue
            ep=m5_cl[r]
            if ep-m5_lo[r]<=0:continue
            a14v=a14[i];a20v=a20[i]
            reg=0
            if not pd.isna(a14v) and not pd.isna(a20v):
                if a14v>a20v*1.1:reg=1
                elif a14v<a20v*0.9:reg=2
            pnls={}
            for cv2 in CH_VALS:
                he=ep
                for j in range(r,len(m5)):
                    ca=atr5[j]
                    if pd.isna(ca):continue
                    if m5_hi[j]>he:he=m5_hi[j]
                    if m5_cl[j]<he-cv2*ca:
                        pnls[cv2]=round((m5_cl[j]-ep)*bl-20,2);break
            if 45 not in pnls:continue
            t={"sym":sym,"year":h1["datetime"].iloc[i].year,"month":h1["datetime"].iloc[i].month,
               "pnl_45":pnls[45],"is_win":pnls[45]>0,"ts":h1["datetime"].iloc[i]}
            for c,p in pnls.items():t[f"p{c}"]=p
            all_t.append(t)
    df=pd.DataFrame(all_t)
    df["pts_45"]=(df["pnl_45"]+20)/(50 if df["sym"].iloc[0]=="NIFTY50" else 10)  # rough
    return df

print("Building trades...")
df=build_trades()
print(f"Total: {len(df)} trades")
for yr in sorted(df["year"].unique()):
    sub=df[df["year"]==yr]
    print(f"  {yr}: {len(sub):3d} trades (Nifty={len(sub[sub['sym']=='NIFTY50']):3d} Sensex={len(sub[sub['sym']=='SENSEX']):3d})")

# ═══════════════════════════════════════════════
# FULL HISTORY BACKTEST
# ═══════════════════════════════════════════════
print("\n"+"="*100)
print("FULL HISTORY BACKTEST (2015-2026)")
print("Walk-forward: CH per month from prior years + Rolling W/L sizing")
print("="*100)

def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def run_backtest(df, use_wl=False, use_dynch=False):
    """Walk-forward backtest. Returns list of per-trade results."""
    results=[]
    years=sorted(df["year"].unique())
    for yr in years:
        yr_d=df[df["year"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        # Compute optimal CH per month from HISTORY (all years < yr)
        month_ch={}
        if use_dynch:
            hist=df[df["year"]<yr]
            for m in range(1,13):
                sub=hist[hist["month"]==m]
                if len(sub)<5:month_ch[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    if f"p{cv}" not in sub.columns:continue
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_ch[m]=best
        rw=[];rl=[]
        for _,t in yr_d.iterrows():
            sz=1.0
            ch=t["pnl_45"]
            if use_dynch:
                m_ch=month_ch.get(t["month"],45)
                ch=t.get(f"p{m_ch}",t["pnl_45"])
            if use_wl:sz=wl_size(rw,rl,lb=5)
            pnl=ch*sz
            results.append({"year":t["year"],"sym":t["sym"],"month":t["month"],
                           "pnl":pnl,"is_win":pnl>0,"size":sz,"ch_base":t["pnl_45"],
                           "ch":ch})
            if pnl>0:rw.append(pnl)
            else:rl.append(abs(pnl))
    return pd.DataFrame(results)

# Run both
base=run_backtest(df, use_wl=False, use_dynch=False)
fix=run_backtest(df, use_wl=True, use_dynch=True)

# ═══════════════════════════════════════════════
# YEAR-BY-YEAR TABLE
# ═══════════════════════════════════════════════
print(f"\n{'Year':<6s} {'Base Net':>12s} {'Base N':>5s} {'Base WR':>7s} | {'Fix Net':>12s} {'Fix N':>5s} {'Fix WR':>7s} {'Size':>6s} | {'Diff':>12s} {'Impr':>8s}")
print("-"*85)
base_total=0;fix_total=0
for yr in sorted(df["year"].unique()):
    b=base[base["year"]==yr];f=fix[fix["year"]==yr]
    if len(b)==0:continue
    bn=b["pnl"].sum();bw=b["is_win"].mean();bc=len(b)
    fn=f["pnl"].sum();fw=f["is_win"].mean();fc=len(f);fs=f["size"].mean()
    d=fn-bn;impr=(fn/bn-1)*100 if bn!=0 else (0 if fn==0 else float('inf'))
    base_total+=bn;fix_total+=fn
    print(f"  {yr:<6d} Rs{bn:>+9,.0f} {bc:4d} {bw:>6.1%} | Rs{fn:>+9,.0f} {fc:4d} {fw:>6.1%} {fs:.2f}x | Rs{d:>+9,.0f} {impr:+7.1f}%")
print(f"  {'TOTAL':<6s} Rs{base_total:>+9,.0f} {len(base):4d} {base['is_win'].mean():>6.1%} | Rs{fix_total:>+9,.0f} {len(fix):4d} {fix['is_win'].mean():>6.1%} {fix['size'].mean():.2f}x | Rs{fix_total-base_total:>+9,.0f}")

# ═══════════════════════════════════════════════
# PER-SYMBOL
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("PER SYMBOL (all years)")
print(f"{'='*100}")
for sym in ["NIFTY50","SENSEX"]:
    print(f"\n--- {sym} ---")
    sym_d=df[df["sym"]==sym]
    base_sym=base[base["sym"]==sym];fix_sym=fix[fix["sym"]==sym]
    bl=50 if sym=="NIFTY50" else 10
    pts_fn=lambda df:(df["pnl"]+20)/bl
    base_net=sym_d["pnl_45"].sum()
    fix_net=fix_sym["pnl"].sum()
    base_pts=(sym_d["pnl_45"]+20).sum()/bl  # rough
    fix_pts=(fix_sym["pnl"]+20).sum()/bl
    print(f"  Net:         Base=Rs{base_net:>+10,.0f}  Fix=Rs{fix_net:>+10,.0f}  Delta=Rs{fix_net-base_net:>+9,.0f}")
    print(f"  Points:      Base={base_pts:>+10,.0f}  Fix={fix_pts:>+10,.0f}  Delta={fix_pts-base_pts:>+9,.0f}")
    print(f"  Avg Pts/Trd: Base={base_pts/len(sym_d):>+7.1f}  Fix={fix_pts/len(fix_sym):>+7.1f}")
    base_wr=sym_d["is_win"].mean()
    fix_wr=fix_sym["is_win"].mean()
    print(f"  WR:          Base={base_wr:.1%}  Fix={fix_wr:.1%}")
    avg_sz=fix_sym["size"].mean()
    print(f"  Avg Size:    {avg_sz:.2f}x")
    
    # Year breakdown per symbol
    print(f"  {'Year':<6s} {'Base Net':>10s} {'Fix Net':>10s} {'Delta':>10s}")
    for yr in sorted(sym_d["year"].unique()):
        b_s=base_sym[(base_sym["sym"]==sym)&(base_sym["year"]==yr)]["pnl"].sum()
        f_s=fix_sym[(fix_sym["sym"]==sym)&(fix_sym["year"]==yr)]["pnl"].sum()
        print(f"    {yr:<4d}  Rs{b_s:>+8,.0f}  Rs{f_s:>+8,.0f}  Rs{f_s-b_s:>+8,.0f}")

# ═══════════════════════════════════════════════
# KEY METRICS
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("KEY METRICS")
print(f"{'='*100}")
# MDD
for label,res in [("Baseline",base),("Fixed",fix)]:
    peak=0;running=0;mdd=0
    for r in res["pnl"]:running+=r;peak=max(peak,running);mdd=max(mdd,peak-running)
    print(f"  {label} MDD: Rs{mdd:>+10,.0f}")

# Sharpe-like (annualized)
for label,res in [("Baseline",base),("Fixed",fix)]:
    ann_ret=res["pnl"].sum()/len(res["year"].unique())
    ann_vol=res.groupby("year")["pnl"].sum().std()*np.sqrt(1)
    sharpe=ann_ret/ann_vol if ann_vol>0 else 0
    print(f"  {label} Sharpe: {sharpe:.2f}")

# Max consecutive loss
for label,res in [("Baseline",base),("Fixed",fix)]:
    max_s=0;cur=0
    for _,r in res.iterrows():
        if r["pnl"]<=0:cur+=1;max_s=max(max_s,cur)
        else:cur=0
    print(f"  {label} Max Cons Loss: {max_s}")

# Win/Loss ratio
for label,res in [("Baseline",base),("Fixed",fix)]:
    wins=res[res["pnl"]>0]["pnl"];losses=res[res["pnl"]<0]["pnl"]
    aw=wins.mean() if len(wins)>0 else 0
    al=abs(losses.mean()) if len(losses)>0 else 0
    wl=aw/al if al>0 else float('inf')
    print(f"  {label} Avg Win: Rs{aw:>+9,.0f} Avg Loss: Rs{-al:>+9,.0f} W/L: {wl:.2f}x")

# ═══════════════════════════════════════════════
# MONTHLY PNL
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("MONTHLY TOTAL (all years)")
print(f"{'='*100}")
print(f"{'Month':<8s} {'Base Net':>12s} {'Fix Net':>12s} {'Delta':>12s} {'Fix Size':>8s}")
print("-"*52)
for m in range(1,13):
    b=base[base["month"]==m]["pnl"].sum()
    f=fix[fix["month"]==m]["pnl"].sum()
    fs=fix[fix["month"]==m]["size"].mean()
    print(f"  Month {m:<3d}  Rs{b:>+9,.0f}  Rs{f:>+9,.0f}  Rs{f-b:>+9,.0f}  {fs:.2f}x")

# ═══════════════════════════════════════════════
# EQUITY CURVE
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("EQUITY CURVE (cumulative)")
print(f"{'='*100}")
eq_b=base["pnl"].cumsum()
eq_f=fix["pnl"].cumsum()
n=len(eq_b);step=max(1,n//8)
print(f"{'Trade':>8s} {'Base':>12s} {'Fix':>12s} {'Diff':>12s}")
for i in range(0,n,step):
    print(f"  {i+1:4d}/{n:<3d}  Rs{eq_b.iloc[i]:>+9,.0f} Rs{eq_f.iloc[i]:>+9,.0f} Rs{eq_f.iloc[i]-eq_b.iloc[i]:>+9,.0f}")
print(f"  {n:4d}/{n:<3d}  Rs{eq_b.iloc[-1]:>+9,.0f} Rs{eq_f.iloc[-1]:>+9,.0f} Rs{eq_f.iloc[-1]-eq_b.iloc[-1]:>+9,.0f}")

# ═══════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("FINAL RESULT")
print(f"{'='*100}")
print(f"""
W/L Sizing + Dynamic CH by Month - Walk-Forward on FULL HISTORY
================================================================

BASELINE CH45:
  12-Year Total: Rs{base_total:>+10,.0f} ({len(base)} trades)
  WR: {base['is_win'].mean():.1%}
  MDD: Rs{mdd if 'mdd' in dir() else 'see above'}

FIXED (W/L + DynCH):
  12-Year Total: Rs{fix_total:>+10,.0f} ({len(fix)} trades)
  WR: {fix['is_win'].mean():.1%}
  MDD: Rs{mdd if 'mdd' in dir() else 'see above'}
  Avg Size: {fix['size'].mean():.2f}x

IMPROVEMENT:
  Return: {((fix_total/base_total)-1)*100 if base_total else 0:+.1f}%
""")
