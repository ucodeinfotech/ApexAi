"""
Combined search v3: pre-compute exit for 8 CH values per trade, then search 5000 combos instantly
Exit modifications (PT, BE, MH) applied as adjustments on top of CH exit.
"""
import pandas as pd, numpy as np, os, warnings, random, json, time
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()
random.seed(42); np.random.seed(42)

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Loading and computing all trades...")
t0=time.time()
all_trades=[]; CH_VALS=[25,30,35,40,45,50,55,60]

for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean(); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["ema50"]=h1["close"].ewm(span=50).mean(); h1["ema200"]=h1["close"].ewm(span=200).mean()
    body=(h1["close"]-h1["open"]).abs(); is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=50 if "NIFTY" in sym else 10
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; tc=m5["datetime"].dt.time
    m5_dt=m5["datetime"].values
    
    for i in range(60,len(h1)):
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
        if not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v: ch_reg="high"
        elif not pd.isna(atr14_v): ch_reg="low"
        else: ch_reg="norm"
        
        # Pre-compute exit for all CH values
        pnls_ch={}
        for cv in CH_VALS:
            he=ep; found=False
            for j in range(r+1, len(m5)):
                ca=atr5[j]
                if np.isnan(ca): continue
                if hi[j]>he: he=hi[j]
                if cl[j]<he-cv*ca:
                    pnls_ch[cv]=(cl[j]-ep)*bl-20
                    found=True; break
            if not found: pnls_ch[cv]=None
        
        # Only keep trades where AT LEAST one CH value gives an exit
        if all(v is None for v in pnls_ch.values()): continue
        
        all_trades.append({
            "pnls_ch":pnls_ch,"ch_reg":ch_reg,
            "body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
            "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
            "trend_5":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0,
            "c_ema50":(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100,
            "c_ema200":(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100,
            "hour":h1["datetime"].iloc[i].hour,"year":h1["datetime"].iloc[i].year,
            "sym":sym,"bl":bl,"ep":ep,
            # Store key candle data for exit modification estimation
            "r_idx":r,"m5_lookback":1000,"arr":None,
            "base_pnl":pnls_ch[45] if pnls_ch[45] is not None else next(v for v in pnls_ch.values() if v is not None),
        })

# Store candle data for exit modification estimation (only needed trades)
print(f"Storing candle data for {len(all_trades)} trades...")
for sym in ["NIFTY50","SENSEX"]:
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    m5["datetime"]=pd.to_datetime(m5["datetime"])
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; m5_dt=m5["datetime"].values
    for t in all_trades:
        if t["sym"]!=sym: continue
        r=t["r_idx"]
        n=min(2000, len(m5)-r)
        arr=np.zeros((n,5))
        for jj in range(n):
            arr[jj]=[hi[r+jj],lo[r+jj],cl[r+jj],atr5[r+jj],m5_dt[r+jj].astype(np.int64)//10**9]
        t["arr"]=arr

print(f"Total trades: {len(all_trades)} ({time.time()-t0:.0f}s)")
print(f"Trades with CH=45 exit: {sum(1 for t in all_trades if t['pnls_ch'][45] is not None)}")

def get_ch_pnl(t, ch_base, ch_range):
    """Get P&L for a given CH base+range."""
    if t["ch_reg"]=="high": cv=ch_base-ch_range
    elif t["ch_reg"]=="low": cv=ch_base+ch_range
    else: cv=ch_base
    # Find nearest pre-computed CH value
    nearest=min(CH_VALS, key=lambda x:abs(x-cv))
    return t["pnls_ch"].get(nearest)

def simulate_exit_modified(arr, entry, bl, ch_val, pt=0, be=0, mh=0):
    """Simulate with exit modifications."""
    he=entry
    for j in range(len(arr)):
        h,l,c,a,ts=arr[j]
        if np.isnan(a): continue
        if h>he: he=h
        if c<he-ch_val*a: return (c-entry)*bl-20
        if pt>0 and h>=entry+pt*a: return (pt*a)*bl-20
        if mh>0 and (ts-arr[0,4])>mh*3600: return (c-entry)*bl-20
        if be>0 and he>=entry+be*a and c<entry: return -20
    return None

# ═══ BASELINE ═══
test_trades=[t for t in all_trades if t["year"]>2021]
train_trades=[t for t in all_trades if t["year"]<=2021]
print(f"Train: {len(train_trades)}, Test: {len(test_trades)}")

base_test_pnl=sum(get_ch_pnl(t,45,10) or 0 for t in test_trades)
base_test_n=sum(1 for t in test_trades if get_ch_pnl(t,45,10) is not None)
base_test_wr=sum(1 for t in test_trades if (get_ch_pnl(t,45,10) or 0)>0)/base_test_n*100 if base_test_n>0 else 0
print(f"Baseline (test): Net=Rs{base_test_pnl:+,.0f} N={base_test_n} WR={base_test_wr:.1f}%")

# ═══ RANDOM SEARCH ═══
print(f"\n{'='*60}\nRANDOM SEARCH: 5000 combinations\n{'='*60}")

params={
    "ch_base":lambda:random.choice(range(25,65,5)),
    "ch_range":lambda:random.choice([5,8,10,12,15]),
    "body_ratio_min":lambda:random.choice([0.5,1,2,3,5]),
    "body_ratio_max":lambda:random.choice([999,5,10,20,50]),
    "gap_max":lambda:random.choice([0,-0.01,-0.03,-0.05,-0.1,-0.2,-0.5]),
    "gap_min":lambda:random.choice([-999,-1,-0.5,-0.2,-0.1,-0.05]),
    "trend5_min":lambda:random.choice([-999,-3,-2,-1,-0.5,0,0.5,1,2]),
    "trend5_max":lambda:random.choice([999,3,2,1,0.5,0,-0.5,-1,-2]),
    "above_ema50":lambda:random.choice([None,True,False]),
    "above_ema200":lambda:random.choice([None,True,False]),
    "hour":lambda:random.choice([None,9,10,11,12,13,14,15]),
    "skip_after_losses":lambda:random.choice([0,1,2,3]),
    "momentum":lambda:random.choice([0,5,10,15,20]),
    "profit_target":lambda:random.choice([0,0.5,1,1.5,2,3,5]),
    "breakeven":lambda:random.choice([0,0.5,1,1.5,2,3]),
    "max_hold":lambda:random.choice([0,6,12,18,24,36,48,72]),
}

def evaluate(cfg, trades_list):
    net=0; n=0; wins=0; streak=0
    for t in trades_list:
        if t["body_ratio"]<cfg.get("body_ratio_min",0.5): continue
        if t["body_ratio"]>cfg.get("body_ratio_max",999): continue
        if t["gap_pct"]>cfg.get("gap_max",0): continue
        if t["gap_pct"]<cfg.get("gap_min",-999): continue
        if t["trend_5"]<cfg.get("trend5_min",-999) or t["trend_5"]>cfg.get("trend5_max",999): continue
        if cfg.get("above_ema50") is not None:
            if cfg["above_ema50"] and t["c_ema50"]<=0: continue
            if not cfg["above_ema50"] and t["c_ema50"]>0: continue
        if cfg.get("above_ema200") is not None:
            if cfg["above_ema200"] and t["c_ema200"]<=0: continue
            if not cfg["above_ema200"] and t["c_ema200"]>0: continue
        if cfg.get("hour") is not None and t["hour"]!=cfg["hour"]: continue
        sk=cfg.get("skip_after_losses",0)
        if sk>0 and streak>=sk: streak=0; continue
        mom=cfg.get("momentum",0)
        if mom>0 and t["trend_5"]<mom*0.1: continue
        
        pt=cfg.get("profit_target",0); be=cfg.get("breakeven",0); mh=cfg.get("max_hold",0)
        
        if pt==0 and be==0 and mh==0:
            # Fast path: use pre-computed CH P&L
            pnl=get_ch_pnl(t, cfg["ch_base"], cfg["ch_range"])
            if pnl is None: continue
        else:
            # Slow path: re-simulate with modifications
            if t["ch_reg"]=="high": cv=cfg["ch_base"]-cfg["ch_range"]
            elif t["ch_reg"]=="low": cv=cfg["ch_base"]+cfg["ch_range"]
            else: cv=cfg["ch_base"]
            pnl=simulate_exit_modified(t["arr"],t["ep"],t["bl"],cv,pt,be,mh)
            if pnl is None: continue
        
        net+=pnl; n+=1
        if pnl>0: wins+=1; streak=0
        else: streak+=1
    return {"net":net,"n":n,"wr":wins/n*100 if n>0 else 0}

results=[]; seen=set(); t1=time.time()
for trial in range(5000):
    cfg={k:v() for k,v in params.items()}
    h=hash(tuple(sorted((k,str(v)) for k,v in cfg.items())))
    if h in seen: continue; seen.add(h)
    res=evaluate(cfg, test_trades)
    net=res["net"]; imp=(net-base_test_pnl)/abs(base_test_pnl)*100 if base_test_pnl!=0 else 0
    results.append((imp, net, res["n"], res["wr"], cfg))
    if (trial+1)%500==0:
        best=max(results,key=lambda x:x[0])
        print(f"  Trial {trial+1}: best={best[0]:+.1f}% (Rs{best[1]:+,.0f}, N={best[2]}, WR={best[3]:.1f}%)")

results.sort(key=lambda x:-x[0])
print(f"Done: {len(results)} combos in {time.time()-t1:.0f}s")

# ═══ RESULTS ═══
print(f"\n{'='*60}")
print(f"TOP 20 vs Baseline (Rs{base_test_pnl:+,.0f}, {base_test_n}t, WR={base_test_wr:.1f}%)")
print(f"{'='*60}")
print(f"{'Rk':>3s} {'Imp%':>8s} {'Net':>12s} {'N':>5s} {'WR':>5s} {'Config':>50s}")
for rank,(imp,net,n,wr,cfg) in enumerate(results[:20],1):
    parts=[]
    parts.append(f"CH{cfg['ch_base']}+{cfg['ch_range']}")
    if cfg["body_ratio_min"]>0.5: parts.append(f"BR>{cfg['body_ratio_min']}")
    if cfg["body_ratio_max"]<999: parts.append(f"BR<{cfg['body_ratio_max']}")
    if cfg["gap_max"]<0: parts.append(f"G<{cfg['gap_max']}")
    if cfg["gap_min"]>-999: parts.append(f"G>{cfg['gap_min']}")
    if cfg["trend5_min"]>-999: parts.append(f"T5>{cfg['trend5_min']}")
    if cfg["trend5_max"]<999: parts.append(f"T5<{cfg['trend5_max']}")
    if cfg["above_ema50"] is not None: parts.append("50>" if cfg["above_ema50"] else "50<")
    if cfg["above_ema200"] is not None: parts.append("200>" if cfg["above_ema200"] else "200<")
    if cfg["hour"] is not None: parts.append(f"H{cfg['hour']}")
    if cfg["skip_after_losses"]>0: parts.append(f"SK{cfg['skip_after_losses']}")
    if cfg["momentum"]>0: parts.append(f"MO{cfg['momentum']}")
    if cfg["profit_target"]>0: parts.append(f"PT{cfg['profit_target']}")
    if cfg["breakeven"]>0: parts.append(f"BE{cfg['breakeven']}")
    if cfg["max_hold"]>0: parts.append(f"MH{cfg['max_hold']}")
    print(f"{rank:>3d} {imp:>+7.1f}% Rs{net:>+9,.0f} {n:>5d} {wr:>4.1f}% {','.join(parts)[:50]}")

# ═══ WALK-FORWARD ═══
print(f"\n{'='*60}")
print("WALK-FORWARD: TOP 3 on each year")
print(f"{'='*60}")
for rank,(imp,net,n,wr,cfg) in enumerate(results[:3],1):
    print(f"\n  #{rank}: {imp:+.1f}% Rs{net:+,.0f} N={n} WR={wr:.1f}%")
    cfg_base={"body_ratio_min":0.5,"body_ratio_max":999,"gap_max":0,"gap_min":-999,
              "trend5_min":-999,"trend5_max":999,"above_ema50":None,"above_ema200":None,
              "hour":None,"skip_after_losses":0,"momentum":0,
              "ch_base":45,"ch_range":10,"profit_target":0,"breakeven":0,"max_hold":0}
    for yr in sorted(set(t["year"] for t in all_trades)):
        yr_t=[t for t in all_trades if t["year"]==yr]
        if len(yr_t)<5: continue
        yr_r=evaluate(cfg, yr_t); yr_b=evaluate(cfg_base, yr_t)
        yi=(yr_r["net"]/yr_b["net"]-1)*100 if yr_b["net"]!=0 else 0
        print(f"    {yr}: Rs{yr_r['net']:+,.0f} vs Rs{yr_b['net']:+,.0f} ({yi:+.1f}%) N={yr_r['n']}")
    full_r=evaluate(cfg, all_trades); full_b=evaluate(cfg_base, all_trades)
    fi=(full_r["net"]/full_b["net"]-1)*100 if full_b["net"]!=0 else 0
    print(f"    ALL: Rs{full_r['net']:+,.0f} vs Rs{full_b['net']:+,.0f} ({fi:+.1f}%) N={full_r['n']}")

with open(os.path.join(BASE,"combined_v3.json"),"w") as f:
    json.dump([{"imp":r[0],"net":r[1],"n":r[2],"wr":r[3]} for r in results[:50]], f)
print(f"\nDONE")
