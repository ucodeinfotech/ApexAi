"""
POINTS-BACKTEST: Baseline CH45 vs W/L + DynCH(Month)
All figures in POINTS (lot-size neutral)
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

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
        hl5=m5["high"]-m5["low"];hpc5=abs(m5["high"]-m5["close"].shift(1));lpc5=abs(m5["low"]-m5["close"].shift(1))
        tr5=pd.concat([hl5,hpc5,lpc5],axis=1).max(axis=1);m5_atr=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
        atr5=m5_atr.values;m5_hi=m5["high"].values;m5_lo=m5["low"].values;m5_cl=m5["close"].values;m5_du=m5["datetime"].values
        prev_red=np.roll(h1["close"].values<h1["open"].values,1);prev_red[0]=False
        tc=pd.Series(m5["datetime"]).dt.time.values;CUT=pd.Timestamp("14:15").time()
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
            pnls={}
            for cv2 in CH_VALS:
                he=ep
                for j in range(r,len(m5)):
                    ca=atr5[j]
                    if pd.isna(ca):continue
                    if m5_hi[j]>he:he=m5_hi[j]
                    if m5_cl[j]<he-cv2*ca:
                        pnls[cv2]=round(m5_cl[j]-ep,1);break
            if 45 not in pnls:continue
            t={"sym":sym,"year":h1["datetime"].iloc[i].year,"month":h1["datetime"].iloc[i].month,
               "pts45":pnls[45],"ts":h1["datetime"].iloc[i]}
            for c,p in pnls.items():t[f"p{c}"]=p
            all_t.append(t)
    return pd.DataFrame(all_t).fillna(0)

print("Building trades...")
df=build_trades()
print(f"Total: {len(df)} trades")
for yr in sorted(df["year"].unique()):
    sub=df[df["year"]==yr]
    print(f"  {yr}: {len(sub):3d} trades (Nifty={len(sub[sub['sym']=='NIFTY50']):3d} Sensex={len(sub[sub['sym']=='SENSEX']):3d})")

# ═══════════════════════════════════════════════
# BACKTEST (all in POINTS)
# ═══════════════════════════════════════════════
print("\n"+"="*100)
print("POINTS BACKTEST: Baseline CH45 vs W/L + DynCH(Month)")
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

def run_bt(df, use_wl, use_dynch):
    rows=[]
    for yr in sorted(df["year"].unique()):
        yr_d=df[df["year"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        month_best={}
        if use_dynch:
            hist=df[df["year"]<yr]
            for m in range(1,13):
                sub=hist[hist["month"]==m]
                if len(sub)<5:month_best[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_best[m]=best
        rw=[];rl=[]
        for _,t in yr_d.iterrows():
            pts=t["pts45"]
            if use_dynch:
                m_ch=month_best.get(t["month"],45)
                pts=t.get(f"p{m_ch}",t["pts45"])
            sz=1.0
            if use_wl:sz=wl_size(rw,rl,lb=5)
            pnl_pts=pts*sz
            rows.append({"pts":pnl_pts,"sym":t["sym"],"year":t["year"],"month":t["month"],
                        "size":sz,"ch_used":pts,"ch45":t["pts45"]})
            if pnl_pts>0:rw.append(pnl_pts)
            else:rl.append(abs(pnl_pts))
    return pd.DataFrame(rows)

base=run_bt(df, use_wl=False, use_dynch=False)
fix=run_bt(df, use_wl=True, use_dynch=True)

# ═══════════════════════════════════════════════
# SELF-CHECK
# ═══════════════════════════════════════════════
print("\n--- Self-Check (points) ---")
raw_pts=df["pts45"].sum()
base_pts=base["pts"].sum()
print(f"  Raw df pts45 total: {raw_pts:>+10,.1f}")
print(f"  Base backtest total: {base_pts:>+10,.1f}")
print(f"  Match: {'YES' if abs(raw_pts-base_pts)<1 else 'NO - BUG!'}")

# ═══════════════════════════════════════════════
# YEAR BY YEAR (points)
# ═══════════════════════════════════════════════
print(f"\n{'Year':<6s} {'Base Pts':>12s} {'Base WR':>7s} {'Base N':>5s} | {'Fix Pts':>12s} {'Fix WR':>7s} {'Fix N':>5s} {'Size':>6s} | {'Diff':>12s}")
print("-"*85)
for yr in sorted(df["year"].unique()):
    b=base[base["year"]==yr];f=fix[fix["year"]==yr]
    if len(b)==0:continue
    bn=b["pts"].sum();bw=(b["pts"]>0).mean();bc=len(b)
    fn=f["pts"].sum();fw=(f["pts"]>0).mean();fc=len(f);fs=f["size"].mean()
    d=fn-bn
    print(f"  {yr:<6d} {bn:>+10,.0f} {bw:>6.1%} {bc:4d} | {fn:>+10,.0f} {fw:>6.1%} {fc:4d} {fs:.2f}x | {d:>+10,.0f}")

bt=base["pts"].sum();ft=fix["pts"].sum()
print(f"  {'TOTAL':<6s} {bt:>+10,.0f} {(base['pts']>0).mean():>6.1%} {len(base):4d} | {ft:>+10,.0f} {(fix['pts']>0).mean():>6.1%} {len(fix):4d} {fix['size'].mean():.2f}x | {ft-bt:>+10,.0f}")

# ═══════════════════════════════════════════════
# PER SYMBOL (points)
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("PER SYMBOL (points)")
print(f"{'='*100}")
for sym in ["NIFTY50","SENSEX"]:
    sym_d=df[df["sym"]==sym]
    base_sym=base[base["sym"]==sym];fix_sym=fix[fix["sym"]==sym]
    bpts=sym_d["pts45"].sum();fpts=fix_sym["pts"].sum()
    print(f"\n  {sym} ({len(sym_d)} trades):")
    print(f"    Pts:   Base={bpts:>+10,.0f} Fix={fpts:>+10,.0f} Delta={fpts-bpts:>+10,.0f} ({(fpts/bpts-1)*100 if bpts else 0:+.1f}%)")
    print(f"    WR:    Base={(sym_d['pts45']>0).mean():.1%} Fix={(fix_sym['pts']>0).mean():.1%}")
    print(f"    Size:  {fix_sym['size'].mean():.2f}x")

# ═══════════════════════════════════════════════
# KEY METRICS (points)
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("KEY METRICS (points)")
print(f"{'='*100}")
for label,res in [("Baseline CH45 ",base),("Fixed (W/L+CH)",fix)]:
    pk=0;rn=0;mdd=0
    for r in res["pts"]:rn+=r;pk=max(pk,rn);mdd=max(mdd,pk-rn)
    ms=0;cur=0
    for _,r in res.iterrows():
        if r["pts"]<=0:cur+=1;ms=max(ms,cur)
        else:cur=0
    w=res[res["pts"]>0]["pts"];l=res[res["pts"]<0]["pts"]
    aw=w.mean() if len(w)>0 else 0;al=abs(l.mean()) if len(l)>0 else 0
    wl=aw/al if al>0 else float('inf')
    wr=(res["pts"]>0).mean()
    yrs=res["year"].nunique()
    ann_ret=res["pts"].sum()/yrs
    ann_vol=res.groupby("year")["pts"].sum().std()
    sharpe=ann_ret/ann_vol if ann_vol>0 else 0
    print(f"  {label}:")
    print(f"    Total Pts:   {res['pts'].sum():>+10,.0f}")
    print(f"    Max DD:      {mdd:>+10,.0f}")
    print(f"    Sharpe:      {sharpe:.2f}")
    print(f"    Win Rate:    {wr:.1%}")
    print(f"    W/L Ratio:   {wl:.2f}x")
    print(f"    Avg Win:     {aw:>+9,.0f}")
    print(f"    Avg Loss:    {-al:>+9,.0f}")
    print(f"    Max ConsL:   {ms}")

# ═══════════════════════════════════════════════
# EQUITY CURVE (points)
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("EQUITY CURVE (points)")
print(f"{'='*100}")
eq_b=base["pts"].cumsum();eq_f=fix["pts"].cumsum()
n=len(eq_b);step=max(1,n//10)
print(f"{'Point':>8s} {'Base':>12s} {'Fixed':>12s} {'Diff':>12s}")
for i in range(0,n,step):
    print(f"  {i+1:5d} {eq_b.iloc[i]:>+9,.0f} {eq_f.iloc[i]:>+9,.0f} {eq_f.iloc[i]-eq_b.iloc[i]:>+9,.0f}")
print(f"  {n:5d} {eq_b.iloc[-1]:>+9,.0f} {eq_f.iloc[-1]:>+9,.0f} {eq_f.iloc[-1]-eq_b.iloc[-1]:>+9,.0f}")

# ═══════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════
print(f"\n{'='*100}")
print("FINAL VERIFIED RESULT (POINTS)")
print(f"{'='*100}")
impr=(ft/bt-1)*100 if bt else 0
print(f"""
BASELINE CH45 (1-lot, no sizing): {bt:>+10,.0f} pts
FIXED: W/L + DynCH:               {ft:>+10,.0f} pts
IMPROVEMENT:                       {ft-bt:>+10,.0f} pts ({impr:+.1f}%)

MDD: Base={base['pts'].cumsum().cummax().sub(base['pts'].cumsum()).max():>+8,.0f} Fixed={fix['pts'].cumsum().cummax().sub(fix['pts'].cumsum()).max():>+8,.0f}
Sharpe: Base={base.groupby('year')['pts'].sum().mean()/max(base.groupby('year')['pts'].sum().std(),0.01):.2f} Fixed={fix.groupby('year')['pts'].sum().mean()/max(fix.groupby('year')['pts'].sum().std(),0.01):.2f}
""")
