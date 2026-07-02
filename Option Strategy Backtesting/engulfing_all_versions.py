"""
ENGULFING — ALL VERSIONS COMPARISON (points)
Tests every engulfing variant: entry × exit × sizing × skip
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def compute_atr20(h1):
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    return tr.rolling(20,min_periods=20).mean()

def compute_adx14(h1):
    tr=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    up=h1["high"]-h1["high"].shift(1);down=h1["low"].shift(1)-h1["low"]
    pdm=((up>down)&(up>0))*up;ndm=((down>up)&(down>0))*down
    atr14=tr.rolling(14,min_periods=14).mean();pdi=100*(pdm.rolling(14).mean()/atr14);ndi=100*(ndm.rolling(14).mean()/atr14)
    return 100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()

def compute_daily_ema(h1, period=50):
    df=h1.copy();df["date"]=h1["datetime"].dt.normalize()
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

# ═══════════════════════════════════════════════════════════
# LOAD RAW DATA + COMPUTE ALL EXITS IN ONE PASS
# ═══════════════════════════════════════════════════════════
print("Loading data & building all trade exits...")

def load_data():
    """Returns dict of sym -> {h1, m5, atr5v, m5_hi, m5_lo, m5_cl, m5_epoch, tc}"""
    data={}
    for sym in ["NIFTY50","SENSEX"]:
        h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
        m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
        atr5=compute_atr(m5)
        data[sym]={
            "h1":h1,"m5":m5,
            "atr5v":atr5.values,"m5_hi":m5["high"].values,"m5_lo":m5["low"].values,"m5_cl":m5["close"].values,
            "m5_epoch":m5["datetime"].astype('int64').values,
            "tc":pd.Series(m5["datetime"]).dt.time.values,
            "m5_du":m5["datetime"].values
        }
    return data

DATA=load_data()
CUT=pd.Timestamp("14:15").time()

def find_retest(sym, t, lv):
    """Find entry via breakout+retest. Returns (r_idx, ep, sl) or None."""
    d=DATA[sym];m5_epoch=d["m5_epoch"];m5_cl=d["m5_cl"];m5_lo=d["m5_lo"];m5_hi=d["m5_hi"];tc=d["tc"]
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
    return (r, ep, sl)

def compute_ch_exits(sym, r, ep):
    """Compute exit prices for all CH values. Returns dict {ch: pts}."""
    d=DATA[sym];atr5v=d["atr5v"];m5_hi=d["m5_hi"];m5_cl=d["m5_cl"]
    pnls={}
    for cv in CH_VALS:
        he=ep
        for j in range(r,len(m5_cl)):
            ca=atr5v[j]
            if pd.isna(ca):continue
            if m5_hi[j]>he:he=m5_hi[j]
            if m5_cl[j]<he-cv*ca:
                pnls[cv]=round(m5_cl[j]-ep,1);break
    return pnls

def compute_fixed_tp_exit(sym, r, ep, sl):
    """Fixed 1:2 TP exit. Returns pts or None."""
    d=DATA[sym];m5_cl=d["m5_cl"];m5_lo=d["m5_lo"];m5_hi=d["m5_hi"];m5_du=d["m5_du"]
    tp=ep+2*(ep-sl)
    xs=range(r+1,len(m5_cl))
    for j in xs:
        if m5_lo[j]<=sl:return round(sl-ep,1)
        if m5_hi[j]>=tp:return round(tp-ep,1)
    return None

# ═══════════════════════════════════════════════════════════
# ENTRY SIGNAL FUNCTIONS
# ═══════════════════════════════════════════════════════════

def sigs_engulf_raw(sym):
    """Pure bullish engulfing (no filters)."""
    h1=DATA[sym]["h1"];body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1]:continue
        if h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        if h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

def sigs_engulf_filt(sym):
    """Engulfing + EMA50>200, ADX>20, session 9:30-12:30."""
    h1=DATA[sym]["h1"];body=(h1["close"]-h1["open"]).abs();is_green=h1["close"]>h1["open"];is_red=h1["close"]<h1["open"]
    atr20=compute_atr20(h1);adx14=compute_adx14(h1)
    ema50=compute_daily_ema(h1,50);ema200=compute_daily_ema(h1,200)
    out=[]
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1]:continue
        if h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if body.iloc[i]<body.iloc[i-1]*0.5:continue
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):continue
        if ema50[i]<=ema200[i]:continue
        if adx14.iloc[i]<=20:continue
        t_min=h1["datetime"].iloc[i].hour*60+h1["datetime"].iloc[i].minute
        if t_min<570 or t_min>750:continue
        if h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month})
    return out

# ═══════════════════════════════════════════════════════════
# BUILD TRADE DATABASE
# ═══════════════════════════════════════════════════════════
def build_trades(sig_fn):
    """Returns DataFrame with pts for all CH values + fixTP."""
    rows=[]
    for sym in ["NIFTY50","SENSEX"]:
        for sig in sig_fn(sym):
            ret=find_retest(sym, sig["ts"], sig["lv"])
            if ret is None:continue
            r,ep,sl=ret
            pnls=compute_ch_exits(sym, r, ep)
            if 45 not in pnls:continue
            fix_tp=compute_fixed_tp_exit(sym, r, ep, sl)
            t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"ts":sig["ts"],"pts45":pnls[45],"fix_tp":fix_tp if fix_tp is not None else 0}
            for c,p in pnls.items():t[f"p{c}"]=p
            rows.append(t)
    return pd.DataFrame(rows).fillna(0)

print("\nBuilding Engulf_Raw trades...")
df_raw=build_trades(sigs_engulf_raw)
print(f"  {len(df_raw)} trades (Nifty={len(df_raw[df_raw['sym']=='NIFTY50'])} Sensex={len(df_raw[df_raw['sym']=='SENSEX'])})")

print("Building Engulf_Filt trades...")
df_filt=build_trades(sigs_engulf_filt)
print(f"  {len(df_filt)} trades (Nifty={len(df_filt[df_filt['sym']=='NIFTY50'])} Sensex={len(df_filt[df_filt['sym']=='SENSEX'])})")

# ═══════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════
def wl_size(rw,rl,lb=5):
    if not rw or not rl:return 1.0
    aw=sum(rw[-lb:])/max(len(rw[-lb:]),1);al=sum(rl[-lb:])/max(len(rl[-lb:]),1)
    r=aw/al if al>0 else 10
    if r<0.8:return 0.1
    if r<1.2:return 0.3
    if r<1.5:return 0.5
    if r<2.0:return 0.75
    return 1.0

def run_bt(df, mode, ch_val=None, use_wl=False, use_dynch=False, size_fn=None, skip_n=0):
    """mode: 'ch_fixed' (use ch_val), 'ch_dynch' (walk-forward DynCH), 'fix_tp' (use pts from fix_tp column).
       size_fn: 'wl' -> rolling W/L, '2w1l' -> anti-martingale, None -> 1.0
       skip_n: skip N trades after a loss
    """
    rows=[]
    for yr in sorted(df["yr"].unique()):
        yr_d=df[df["yr"]==yr].sort_values("ts")
        if len(yr_d)==0:continue
        month_best={}
        if use_dynch:
            hist=df[df["yr"]<yr]
            for m in range(1,13):
                sub=hist[hist["mo"]==m]
                if len(sub)<5:month_best[m]=45;continue
                best=45;best_net=-1e9
                for cv in CH_VALS:
                    net=sub[f"p{cv}"].sum()
                    if net>best_net:best_net=net;best=cv
                month_best[m]=best
        rw=[];rl=[];skip_counter=0
        for _,t in yr_d.iterrows():
            if skip_counter>0:skip_counter-=1;continue
            # Determine pts
            if mode=="fix_tp":pts=t["fix_tp"]
            elif mode=="ch_fixed":pts=t.get(f"p{ch_val}",0)
            elif mode=="ch_dynch":
                m_ch=month_best.get(t["mo"],45)
                pts=t.get(f"p{m_ch}",t["pts45"])
            else:pts=t["pts45"]
            # Determine size
            sz=1.0
            if use_wl:sz=wl_size(rw,rl,lb=5)
            if size_fn=="2w1l":
                aw=sum(x>0 for x in rw[-2:]);al=sum(x<0 for x in rl[-1:])
                sz=1.5 if aw>=2 else (0.5 if al>=1 else 1.0)
            if size_fn=="3w2l":
                aw=sum(x>0 for x in rw[-3:]);al=sum(x<0 for x in rl[-2:])
                sz=1.5 if aw>=3 else (0.5 if al>=2 else 1.0)
            pnl=pts*sz
            rows.append({"pts":pnl,"yr":t["yr"],"mo":t["mo"],"sz":sz})
            if pnl>0:rw.append(pnl);skip_counter=0
            else:
                rl.append(abs(pnl))
                if skip_n>0:skip_counter=skip_n
    return pd.DataFrame(rows)

def stats(res):
    if len(res)==0:return {}
    pk=0;rn=0;mdd=0
    for v in res["pts"]:rn+=v;pk=max(pk,rn);mdd=max(mdd,pk-rn)
    w=res[res["pts"]>0]["pts"];l=res[res["pts"]<0]["pts"]
    aw=w.mean() if len(w)>0 else 0;al=l.abs().mean() if len(l)>0 else 1
    wl=aw/al if al>0 else 999
    return {"net":res["pts"].sum(),"n":len(res),"wr":(res["pts"]>0).mean(),"wl":wl,
            "mdd":mdd,"avg_w":aw,"avg_l":al,"avg_sz":res["sz"].mean()}

# ═══════════════════════════════════════════════════════════
# TEST ALL VERSIONS
# ═══════════════════════════════════════════════════════════
versions=[]

def add_version(name, df, mode, ch_val=None, use_wl=False, use_dynch=False, size_fn=None, skip_n=0):
    res=run_bt(df, mode, ch_val, use_wl, use_dynch, size_fn, skip_n)
    s=stats(res)
    versions.append({"name":name,"df":df,**s})
    return s

print("\n"+"="*130)
print("TESTING ALL ENGULFING VERSIONS")
print("="*130)

# ── RAW ENTRY VARIANTS ──
add_version("Raw_FixTP", df_raw, "fix_tp")
add_version("Raw_CH7", df_raw, "ch_fixed", ch_val=7)
add_version("Raw_CH15", df_raw, "ch_fixed", ch_val=15)
add_version("Raw_CH25", df_raw, "ch_fixed", ch_val=25)
add_version("Raw_CH35", df_raw, "ch_fixed", ch_val=35)
add_version("Raw_CH45", df_raw, "ch_fixed", ch_val=45)
add_version("Raw_CH55", df_raw, "ch_fixed", ch_val=55)

# Raw + DynCH
add_version("Raw_DynCH", df_raw, "ch_dynch", use_dynch=True)

# Raw + W/L sizing
add_version("Raw_CH7+WL", df_raw, "ch_fixed", ch_val=7, use_wl=True)
add_version("Raw_CH15+WL", df_raw, "ch_fixed", ch_val=15, use_wl=True)
add_version("Raw_CH45+WL", df_raw, "ch_fixed", ch_val=45, use_wl=True)
add_version("Raw_DynCH+WL", df_raw, "ch_dynch", use_dynch=True, use_wl=True)

# Raw + skip after loss
for sn in [1,2,3]:
    add_version(f"Raw_CH45+Skip{sn}", df_raw, "ch_fixed", ch_val=45, skip_n=sn)

# Raw + anti-martingale sizing
add_version("Raw_CH45+2w1l", df_raw, "ch_fixed", ch_val=45, size_fn="2w1l")
add_version("Raw_CH45+3w2l", df_raw, "ch_fixed", ch_val=45, size_fn="3w2l")

# ── FILTERED ENTRY VARIANTS ──
add_version("Filt_FixTP", df_filt, "fix_tp")
add_version("Filt_CH7", df_filt, "ch_fixed", ch_val=7)
add_version("Filt_CH15", df_filt, "ch_fixed", ch_val=15)
add_version("Filt_CH45", df_filt, "ch_fixed", ch_val=45)
add_version("Filt_DynCH", df_filt, "ch_dynch", use_dynch=True)
add_version("Filt_CH45+WL", df_filt, "ch_fixed", ch_val=45, use_wl=True)
add_version("Filt_DynCH+WL", df_filt, "ch_dynch", use_dynch=True, use_wl=True)

# ═══════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*130}")
print("ALL ENGULFING VERSIONS — RESULTS (points)")
print(f"{'='*130}")

cols=["Version","Trades","Net","WR","W/L","MDD","AvgW","AvgL","AvgSz"]
header=f"  {'Version':<22s} {'Trades':>6s} {'Net':>10s} {'WR':>6s} {'W/L':>6s} {'MDD':>9s} {'AvgW':>9s} {'AvgL':>9s} {'AvgSz':>5s}"
print(header)
print("  "+"-"*len(header))

# Sort by net descending
versions.sort(key=lambda v:-v["net"])
for v in versions:
    print(f"  {v['name']:<22s} {v['n']:>6d} {v['net']:>+9,.0f}  {v['wr']:.0%}  {v['wl']:>4.1f}x {v['mdd']:>+8,.0f} {v['avg_w']:>+8,.0f} {v['avg_l']:>+8,.1f} {v['avg_sz']:.2f}")

# ═══════════════════════════════════════════════════════════
# YEAR-BY-YEAR FOR TOP 8 VERSIONS
# ═══════════════════════════════════════════════════════════
top8=[v["name"] for v in versions[:8]]
print(f"\n{'='*130}")
print("YEAR-BY-YEAR: TOP 8 VERSIONS")
print(f"{'='*130}")

# Re-run top versions with year breakdown
top_results={}
for vname in top8:
    v=[x for x in versions if x["name"]==vname][0]
    # Identify params from name (simplified)
    use_filt="Filt" in vname
    df=df_filt if use_filt else df_raw
    if "FixTP" in vname:mode="fix_tp";kw={}
    elif "DynCH" in vname:
        mode="ch_dynch";kw={"use_dynch":True}
        kw["use_wl"]="WL" in vname or "+WL" in vname
    else:
        # Extract CH value
        import re
        m=re.search(r'CH(\d+)',vname)
        ch_val=int(m.group(1)) if m else 45
        mode="ch_fixed";kw={"ch_val":ch_val}
        kw["use_wl"]="WL" in vname or "+WL" in vname
        kw["skip_n"]=int(re.search(r'Skip(\d)',vname).group(1)) if re.search(r'Skip(\d)',vname) else 0
        kw["size_fn"]="2w1l" if "2w1l" in vname else ("3w2l" if "3w2l" in vname else None)
    res=run_bt(df, mode, **kw)
    top_results[vname]=res

# Yearly table
header=f"  {'Year':<6s}"
for vname in top8:header+=f" {vname:>20s}"
print(header)
print("  "+"-"*len(header))
for yr in sorted(df_raw["yr"].unique()):
    line=f"  {yr:<6d}"
    for vname in top8:
        v=top_results[vname];yr_pts=v[v["yr"]==yr]["pts"].sum()
        line+=f" {yr_pts:>+19,.0f}"
    print(line)
# Totals
line="  Total "
for vname in top8:
    line+=f" {top_results[vname]['pts'].sum():>+19,.0f}"
print(line)

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*130}")
print("BEST PER CATEGORY")
print(f"{'='*130}")

cats={
    "Fixed TP":lambda v:"FixTP" in v["name"],
    "CH7":lambda v:"CH7" in v["name"] and "CH7"==v["name"].split("_")[-1].split("+")[0] if len(v["name"].split("_"))>1 else False,
    "CH15":lambda v:"CH15" in v["name"],
    "CH45":lambda v:"CH45" in v["name"] and "WL" not in v["name"] and "Skip" not in v["name"] and "2w1l" not in v["name"] and "3w2l" not in v["name"] and "Filt" not in v["name"],
    "CH45+WL":lambda v:"CH45+WL" in v["name"],
    "DynCH":lambda v:"DynCH" in v["name"] and "WL" not in v["name"],
    "DynCH+WL":lambda v:"DynCH+WL" in v["name"],
    "Skip":lambda v:"Skip" in v["name"],
    "Filt_CH45":lambda v:"Filt_CH45" in v["name"] and "WL" not in v["name"],
    "Filt_DynCH":lambda v:"Filt_DynCH" in v["name"],
    "Filt_DynCH+WL":lambda v:"Filt_DynCH+WL" in v["name"],
}

for cat,fn in cats.items():
    matches=[v for v in versions if fn(v)]
    if matches:
        best=max(matches,key=lambda v:v["net"])
        print(f"  {cat:20s} -> {best['name']:<22s} Net={best['net']:>+9,.0f} WR={best['wr']:.0%} W/L={best['wl']:.1f}x MDD={best['mdd']:>+8,.0f}")

print(f"\n{'='*130}")
print("DONE")
print(f"{'='*130}")
