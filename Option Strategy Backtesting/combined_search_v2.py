"""
Combined search v2: ALL filters+exits+optimizations simultaneously.
Pre-computes base P&L, then parameter search over 5000 combos.
"""
import pandas as pd, numpy as np, os, warnings, random, json, time
warnings.filterwarnings("ignore")
BASE=r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()
random.seed(42); np.random.seed(42)

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Pre-computing trade data...")
all_trades=[]
t0=time.time()

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
        
        # Store as numpy for speed: [high, low, close, atr, timestamp_unix]
        n_candles=min(5000, len(m5)-r)  # store up to 5000 candles (~17 days)
        arr=np.zeros((n_candles, 5))
        for jj in range(n_candles):
            j=r+jj
            arr[jj]=[hi[j], lo[j], cl[j], atr5[j], m5_dt[j].astype(np.int64)//10**9]
        
        # Determine CH regime
        atr14_v=h1["atr14"].iloc[i]; atr_ma_v=h1["atr_ma20"].iloc[i]
        if not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v:
            ch_regime="high_vol"
        elif not pd.isna(atr14_v):
            ch_regime="low_vol"
        else:
            ch_regime="normal"
        
        all_trades.append({
            "sym":sym,"bl":bl,"entry":ep,"arr":arr,"ch_regime":ch_regime,
            "body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
            "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
            "trend_5":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0,
            "c_ema50":(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100,
            "c_ema200":(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100,
            "hour":h1["datetime"].iloc[i].hour,"year":h1["datetime"].iloc[i].year,
        })

print(f"Total: {len(all_trades)} trades ({time.time()-t0:.0f}s)")

def simulate_fast(arr, entry, bl, ch_val, pt=0, be=0, mh=0):
    """Fast exit simulation on numpy array: arr[:,high,low,close,atr,time]"""
    he=entry
    for j in range(len(arr)):
        h,l,c,a,ts=arr[j]
        if np.isnan(a): continue
        if h>he: he=h
        if c<he-ch_val*a:
            return (c-entry)*bl-20
        if pt>0 and h>=entry+pt*a:
            return (entry+pt*a-entry)*bl-20
        if mh>0 and (ts-arr[0,4])>mh*3600:
            return (c-entry)*bl-20
        if be>0 and he>=entry+be*a and c<entry:
            return -20
    return None

# ═══ BASELINE ═══
print("\nComputing baseline (all trades, CH45+10)...")
base_pnls=[]
for t in all_trades:
    ch_val=45
    if t["ch_regime"]=="high_vol": ch_val=35
    elif t["ch_regime"]=="low_vol": ch_val=55
    pnl=simulate_fast(t["arr"],t["entry"],t["bl"],ch_val)
    if pnl is not None:
        t["base_pnl"]=pnl
        base_pnls.append(pnl)
    else:
        t["base_pnl"]=None

base_pnls=np.array(base_pnls)
print(f"Baseline: {len(base_pnls)} trades, Net=Rs{base_pnls.sum():+,.0f}, WR={(base_pnls>0).mean()*100:.1f}%")

# ═══ SPLIT ═══
# Sort by year for chronological split
year_order=sorted(set(t["year"] for t in all_trades if t["base_pnl"] is not None))
# 70% split
valid_trades=[t for t in all_trades if t["base_pnl"] is not None]
split=int(len(valid_trades)*0.7)
# Use year-based split for proper chronological
split_year=2021  # split at 2021

train_trades=[t for t in valid_trades if t["year"]<=split_year]
test_trades=[t for t in valid_trades if t["year"]>split_year]

# Adjust split to be exact 70/30 if needed
if len(train_trades)/len(valid_trades)<0.6:
    # Add some 2022
    pass
elif len(train_trades)/len(valid_trades)>0.8:
    split_year=2022

train_trades=[t for t in valid_trades if t["year"]<=split_year]
test_trades=[t for t in valid_trades if t["year"]>split_year]

# Fall back to simple index split
train_trades=valid_trades[:split]
test_trades=valid_trades[split:]
print(f"Train: {len(train_trades)} (2015-{split_year}), Test: {len(test_trades)} ({split_year+1}-2026)")

def evaluate_config(cfg, trades_list):
    """Evaluate a config on a list of trades (each has pre-computed base_pnl)."""
    net=0; n=0; wins=0; loss_streak=0
    
    for t in trades_list:
        # PRE-ENTRY FILTERS
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
        
        # SKIP AFTER LOSSES
        sk=cfg.get("skip_after_losses",0)
        if sk>0 and loss_streak>=sk:
            loss_streak=0; continue
        
        # MOMENTUM
        mom=cfg.get("momentum",0)
        if mom>0 and t["trend_5"]<mom*0.1: continue
        
        # EXIT SIMULATION
        ch_base=cfg.get("ch_base",45); ch_range=cfg.get("ch_range",10)
        if t["ch_regime"]=="high_vol": ch_val=ch_base-ch_range
        elif t["ch_regime"]=="low_vol": ch_val=ch_base+ch_range
        else: ch_val=ch_base
        
        pnl=simulate_fast(t["arr"],t["entry"],t["bl"],ch_val,
                         cfg.get("profit_target",0),cfg.get("breakeven",0),
                         cfg.get("max_hold",0))
        if pnl is None: continue
        
        net+=pnl; n+=1
        if pnl>0: wins+=1; loss_streak=0
        else: loss_streak+=1
    
    return {"net":net,"n":n,"wr":wins/n*100 if n>0 else 0,"wins":wins}

# BASELINE ON TEST
base_test=evaluate_config(
    {"body_ratio_min":0.5,"body_ratio_max":999,"gap_max":0,"gap_min":-999,
     "trend5_min":-999,"trend5_max":999,"above_ema50":None,"above_ema200":None,
     "hour":None,"skip_after_losses":0,"momentum":0,
     "ch_base":45,"ch_range":10,"profit_target":0,"breakeven":0,"max_hold":0},
    test_trades)
print(f"\nBaseline (test): Net=Rs{base_test['net']:+,.0f} N={base_test['n']} WR={base_test['wr']:.1f}%")

# ═══ RANDOM SEARCH ═══
print(f"\n{'='*60}")
print("RANDOM SEARCH: 5000 combinations")
print(f"{'='*60}")

param_space={
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

results=[]
seen=set()
t_start=time.time()

for trial in range(5000):
    cfg={k:v() for k,v in param_space.items()}
    
    # Skip excessive restrictions
    est=1.0
    if cfg["body_ratio_min"]>5: est*=0.34
    if cfg["body_ratio_max"]<5: est*=0.66
    if cfg["gap_max"]<-0.3: est*=0.3
    if cfg["gap_min"]>-0.05: est*=0.3
    if cfg["trend5_min"]>0.5: est*=0.45
    if cfg["trend5_max"]< -0.5: est*=0.3
    if cfg["above_ema50"] is not None: est*=0.69
    if cfg["above_ema200"] is not None: est*=0.69
    if cfg["hour"] is not None: est*=0.13
    if est<0.05: continue
    
    h=hash(tuple(sorted(cfg.items())))
    if h in seen: continue
    seen.add(h)
    
    res=evaluate_config(cfg, test_trades)
    net=res["net"]; base=base_test["net"]
    imp=(net/base-1)*100 if base!=0 else 0
    
    results.append((imp, net, res["n"], res["wr"], cfg))
    
    if (trial+1)%500==0:
        best=max(results,key=lambda x:x[0])
        print(f"  Trial {trial+1}: best={best[0]:+.1f}% (Rs{best[1]:+,.0f}, N={best[2]}, WR={best[3]:.1f}%)")

results.sort(key=lambda x:-x[0])
elapsed=time.time()-t_start

print(f"\n{'='*60}")
print(f"RESULTS: {len(results)} combos in {elapsed:.0f}s")
print(f"{'='*60}")

print(f"\n--- TOP 20 CONFIGURATIONS (vs baseline Rs{base_test['net']:+,.0f}) ---")
print(f"{'Rank':>4s} {'Imp%':>8s} {'Net':>12s} {'N':>5s} {'WR%':>6s} {'Summary':>50s}")
for rank,(imp,net,n,wr,cfg) in enumerate(results[:20],1):
    parts=[]
    parts.append(f"CH{cfg['ch_base']}+{cfg['ch_range']}")
    if cfg["body_ratio_min"]>0.5: parts.append(f"BR>{cfg['body_ratio_min']}")
    if cfg["body_ratio_max"]<999: parts.append(f"BR<{cfg['body_ratio_max']}")
    if cfg["gap_max"]<0: parts.append(f"G<{cfg['gap_max']}")
    if cfg["gap_min"]>-999: parts.append(f"G>{cfg['gap_min']}")
    if cfg["trend5_min"]>-999: parts.append(f"T5>{cfg['trend5_min']}")
    if cfg["trend5_max"]<999: parts.append(f"T5<{cfg['trend5_max']}")
    if cfg["above_ema50"] is not None: parts.append("EMA50>" if cfg["above_ema50"] else "EMA50<")
    if cfg["above_ema200"] is not None: parts.append("EMA200>" if cfg["above_ema200"] else "EMA200<")
    if cfg["hour"] is not None: parts.append(f"H{cfg['hour']}")
    if cfg["skip_after_losses"]>0: parts.append(f"SK{cfg['skip_after_losses']}")
    if cfg["momentum"]>0: parts.append(f"MO{cfg['momentum']}")
    if cfg["profit_target"]>0: parts.append(f"PT{cfg['profit_target']}")
    if cfg["breakeven"]>0: parts.append(f"BE{cfg['breakeven']}")
    if cfg["max_hold"]>0: parts.append(f"MH{cfg['max_hold']}")
    print(f"{rank:>4d} {imp:>+7.1f}% Rs{net:>+9,.0f} {n:>5d} {wr:>5.1f}% {','.join(parts)[:50]}")

# ═══ WALK-FORWARD ═══
print(f"\n{'='*60}")
print("WALK-FORWARD: TOP 5 on each year")
print(f"{'='*60}")

for rank,(imp,net,n,wr,cfg) in enumerate(results[:5],1):
    print(f"\n  #{rank}: {imp:+.1f}% Net=Rs{net:+,.0f} N={n} WR={wr:.1f}%")
    for yr in sorted(set(t["year"] for t in valid_trades)):
        yr_t=[t for t in valid_trades if t["year"]==yr]
        if len(yr_t)<5: continue
        yr_res=evaluate_config(cfg, yr_t)
        yr_base=evaluate_config(
            {"body_ratio_min":0.5,"body_ratio_max":999,"gap_max":0,"gap_min":-999,
             "trend5_min":-999,"trend5_max":999,"above_ema50":None,"above_ema200":None,
             "hour":None,"skip_after_losses":0,"momentum":0,
             "ch_base":45,"ch_range":10,"profit_target":0,"breakeven":0,"max_hold":0},
            yr_t)
        yb=yr_base["net"]
        yi=(yr_res["net"]/yb-1)*100 if yb!=0 else 0
        print(f"    {yr}: Rs{yr_res['net']:+,.0f} vs Rs{yb:+,.0f} ({yi:+.1f}%) N={yr_res['n']}")
    
    # Overall
    full_res=evaluate_config(cfg, valid_trades)
    full_base=evaluate_config(
        {"body_ratio_min":0.5,"body_ratio_max":999,"gap_max":0,"gap_min":-999,
         "trend5_min":-999,"trend5_max":999,"above_ema50":None,"above_ema200":None,
         "hour":None,"skip_after_losses":0,"momentum":0,
         "ch_base":45,"ch_range":10,"profit_target":0,"breakeven":0,"max_hold":0},
        valid_trades)
    fi=(full_res["net"]/full_base["net"]-1)*100
    print(f"    ALL: Rs{full_res['net']:+,.0f} vs Rs{full_base['net']:+,.0f} ({fi:+.1f}%) N={full_res['n']}")

# ═══ SAVE ═══
with open(os.path.join(BASE,"combined_search_v2.json"),"w") as f:
    json.dump([{"imp":r[0],"net":r[1],"n":r[2],"wr":r[3],"cfg":{k:(str(v) if not isinstance(v,(int,float,bool,type(None))) else v) for k,v in r[4].items()}} for r in results[:50]], f, indent=2)
print(f"\nSaved to combined_search_v2.json")
print(f"DONE")
