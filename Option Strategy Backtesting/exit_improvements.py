"""
Exit-side improvements & adaptive risk management
Tests: profit target exit, breakeven after X profit, tighten after losses
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20; CUTOFF=pd.Timestamp("14:15").time()

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# Pre-load
h1d={}; m5d={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["atr_pctile"]=h1["atr14"].rolling(252).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()+1e-10)*100 if x.max()>x.min() else 50, raw=False)
    h1["ema50"]=h1["close"].ewm(span=50).mean()
    h1d[sym]=h1; m5d[sym]=m5

def run_exit_test(label, ch_base=45, ch_adj=10, profit_target_atr=0, be_after_atr=0,
                  tighten_after_loss=False, tighten_mult=0.5, max_hold_hours=0,
                  ch_high_adj=0, skip=2):
    """Run engulfing strategy with custom exit rules"""
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=h1d[sym]; m5=m5d[sym]
        body=(h1["close"]-h1["open"]).abs()
        is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
        bl=NLOT if "NIFTY" in sym else SLOT
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        atr5=A(m5,14).values; tc=m5["datetime"].dt.time
        
        for i in range(50,len(h1)):
            if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.5: continue
            tu=int(pd.to_datetime(h1["datetime"].iloc[i]).timestamp()); lv=h1["high"].iloc[i]
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5): continue
            b=idx
            while b<len(m5) and cl[b]<=lv: b+=1
            if b>=len(m5): continue
            r=b+1
            while r<len(m5):
                if lo[r]<lv and cl[r]>lv and tc.iloc[r]<CUTOFF: break
                r+=1
            if r>=len(m5): continue
            ep=cl[r]
            if ep-lo[r]<=0 or m5["datetime"].iloc[r].hour==9: continue
            atr14_v=h1["atr14"].iloc[i]; atr_ma_v=h1["atr_ma20"].iloc[i]
            ch=ch_base
            if not pd.isna(atr14_v) and not pd.isna(atr_ma_v):
                ch=ch_base-ch_adj if atr14_v>atr_ma_v else ch_base+ch_adj
            he=ep; entry_time=m5["datetime"].iloc[r]; be_hit=False
            consec_loss=0  # Track consecutive losses for adaptive tightening
            
            for j in range(r+1,len(m5)):
                ca=atr5[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                cur_ch=ch
                # Adaptive: tighten after consecutive losses
                if tighten_after_loss and consec_loss>=2:
                    cur_ch=max(15, int(ch*tighten_mult))
                # Profit target exit
                if profit_target_atr>0 and not be_hit:
                    if (cl[j]-ep)>=profit_target_atr*ca:
                        pnl=(cl[j]-ep)*bl*1-CHG*1
                        et.append({"pnl":pnl,"et":m5["datetime"].iloc[j]})
                        if pnl>0: consec_loss=0
                        else: consec_loss+=1
                        break
                # Breakeven stop
                if be_after_atr>0 and not be_hit and (he-ep)>=be_after_atr*ca:
                    be_hit=True
                    ch=min(cur_ch, 5)  # Tight stop after breakeven
                # Chandelier exit
                if cl[j]<he-cur_ch*ca:
                    pnl=(cl[j]-ep)*bl*1-CHG*1
                    et.append({"pnl":pnl,"et":m5["datetime"].iloc[j]})
                    if pnl>0: consec_loss=0
                    else: consec_loss+=1
                    break
                # Max hold time exit
                if max_hold_hours>0:
                    hrs=(m5["datetime"].iloc[j]-entry_time).total_seconds()/3600
                    if hrs>max_hold_hours and cl[j]<=ep:
                        pnl=(cl[j]-ep)*bl*1-CHG*1
                        et.append({"pnl":pnl,"et":m5["datetime"].iloc[j]})
                        break
    return pd.DataFrame(et).sort_values("et").reset_index(drop=True) if et else pd.DataFrame()

def eval_exit(c,skip=2):
    if len(c)==0: return None
    c=c.sort_values("et").reset_index(drop=True)
    if skip>0:
        lc=0; k=np.ones(len(c),dtype=bool)
        for i in range(len(c)):
            if lc>=skip: k[i]=False; lc=0; continue
            if c["pnl"].iloc[i]<=0: lc+=1
            else: lc=0
        c=c[k].reset_index(drop=True)
    n=len(c); net=c["pnl"].sum()
    wr=(c["pnl"]>0).sum()/n*100 if n>0 else 0
    aw=c[c["pnl"]>0]["pnl"].mean() if (c["pnl"]>0).sum()>0 else 0
    al=c[c["pnl"]<0]["pnl"].mean() if (c["pnl"]<0).sum()>0 else 0
    pf=(c[c["pnl"]>0]["pnl"].sum()/abs(c[c["pnl"]<0]["pnl"].sum())) if (c["pnl"]<0).sum()!=0 else 99
    eq=c["pnl"].cumsum()+200000; pk=np.maximum.accumulate(eq); mdd=(pk-eq).max()
    mdd_p=mdd/pk.max()*100 if pk.max()>0 else 0
    yrs=10
    cagr=((1+net/200000)**(1/yrs)-1)*100
    return {"n":n,"net":net,"wr":wr,"aw":aw,"al":al,"pf":pf,"mdd":mdd,"mdd_p":mdd_p,"cagr":cagr}

# Baseline
base=run_exit_test("Base"); BR=eval_exit(base,2)
print(f"Baseline DynCH45+/-10: Rs{BR['net']:+,.0f}  WR={BR['wr']:.1f}%  PF={BR['pf']:.2f}  MDD={BR['mdd_p']:.2f}%  CAGR={BR['cagr']:.1f}%")
BN=BR["net"]

print(f"\n{'='*95}")
print("EXIT-SIDE OPTIMIZATIONS")
print(f"{'='*95}")

all_r=[]
for lbl,pt,be,tloss,tmult,mhold in [
    # Profit targets (exit at N×ATR profit)
    ("PT1xATR",1,0,False,0,0),
    ("PT2xATR",2,0,False,0,0),
    ("PT3xATR",3,0,False,0,0),
    ("PT5xATR",5,0,False,0,0),
    # Breakeven after N×ATR
    ("BE1xATR",0,1,False,0,0),
    ("BE2xATR",0,2,False,0,0),
    ("BE1.5xATR",0,1.5,False,0,0),
    # Tighten after 2 consecutive losses
    ("Tight50%_Loss",0,0,True,0.5,0),
    ("Tight70%_Loss",0,0,True,0.7,0),
    ("Tight30%_Loss",0,0,True,0.3,0),
    # Max hold time  
    ("MaxHold24h",0,0,False,0,24),
    ("MaxHold48h",0,0,False,0,48),
    ("MaxHold72h",0,0,False,0,72),
    # Combinations
    ("PT2x+BE1x",2,1,False,0,0),
    ("PT3x+Tight50",3,0,True,0.5,0),
    ("BE1.5x+Tight50",0,1.5,True,0.5,0),
    ("PT3x+MaxHold48h",3,0,False,0,48),
]:
    c=run_exit_test(lbl, ch_base=45, ch_adj=10, profit_target_atr=pt, be_after_atr=be,
                    tighten_after_loss=tloss, tighten_mult=tmult, max_hold_hours=mhold)
    r=eval_exit(c,2)
    if r:
        vs=(r["net"]/BN-1)*100
        all_r.append({"lbl":lbl,**r,"vs":vs})
        print(f"  {lbl:20s}: Rs{r['net']:>+9,.0f}  n={r['n']:4d}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  vsB={vs:+7.1f}%")

# Sort and show best
sorted_r=sorted(all_r, key=lambda x: x["net"], reverse=True)
print(f"\n{'='*95}")
print("RANKED BY NET P&L")
print(f"{'='*95}")
print(f"{'Exit Variant':25s}  {'Net P&L':>11s}  {'Trades':>5s}  {'WR%':>5s}  {'PF':>5s}  {'MDD%':>6s}  {'CAGR':>6s}  {'vsBase':>8s}")
print(f"{'-'*95}")
print(f"{'BASELINE':25s}  Rs{BN:>+9,.0f}  {BR['n']:5d}  {BR['wr']:4.1f}%  {BR['pf']:4.2f}  {BR['mdd_p']:5.2f}%  {BR['cagr']:5.1f}%  {'0.0%':>8s}")
for r in sorted_r:
    print(f"{r['lbl']:25s}  Rs{r['net']:>+9,.0f}  {r['n']:5d}  {r['wr']:4.1f}%  {r['pf']:4.2f}  {r['mdd_p']:5.2f}%  {r['cagr']:5.1f}%  {r['vs']:>+7.1f}%")

# ── ADAPTIVE SIZING BASED ON MARKET REGIME ──
print(f"\n{'='*95}")
print("ADAPTIVE: Only trade in favorable ATR percentile regimes")
print(f"{'='*95}")
for pctl in [30,40,50,60,70]:
    # Only take signals when ATR percentile is above threshold (not dead market)
    et=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=h1d[sym]; m5=m5d[sym]
        body=(h1["close"]-h1["open"]).abs()
        is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
        bl=NLOT if "NIFTY" in sym else SLOT
        du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
        hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
        atr5=A(m5,14).values; tc=m5["datetime"].dt.time
        for i in range(50,len(h1)):
            if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if body.iloc[i]<body.iloc[i-1]*0.5: continue
            # ATR percentile filter
            atp=h1["atr_pctile"].iloc[i]
            if not pd.isna(atp) and atp<pctl: continue  # Skip when vol too low
            tu=int(pd.to_datetime(h1["datetime"].iloc[i]).timestamp()); lv=h1["high"].iloc[i]
            idx=np.searchsorted(du,tu,side="right")
            if idx>=len(m5): continue
            b=idx
            while b<len(m5) and cl[b]<=lv: b+=1
            if b>=len(m5): continue
            r=b+1
            while r<len(m5):
                if lo[r]<lv and cl[r]>lv and tc.iloc[r]<CUTOFF: break
                r+=1
            if r>=len(m5): continue
            ep=cl[r]
            if ep-lo[r]<=0 or m5["datetime"].iloc[r].hour==9: continue
            atr14_v=h1["atr14"].iloc[i]; atr_ma_v=h1["atr_ma20"].iloc[i]
            ch=ch_base=45
            if not pd.isna(atr14_v) and not pd.isna(atr_ma_v):
                ch=ch_base-10 if atr14_v>atr_ma_v else ch_base+10
            he=ep
            for j in range(r+1,len(m5)):
                ca=atr5[j]
                if pd.isna(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-ch*ca:
                    pnl=(cl[j]-ep)*bl*1-CHG*1
                    et.append({"pnl":pnl,"et":m5["datetime"].iloc[j]})
                    break
    c=pd.DataFrame(et).sort_values("et").reset_index(drop=True) if et else pd.DataFrame()
    r=eval_exit(c,2)
    if r:
        vs=(r["net"]/BN-1)*100
        print(f"  ATR%>={pctl:>2d}: Rs{r['net']:>+9,.0f}  n={r['n']:4d}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  MDD={r['mdd_p']:.2f}%  vsB={vs:+7.1f}%")

print(f"\nDONE")
