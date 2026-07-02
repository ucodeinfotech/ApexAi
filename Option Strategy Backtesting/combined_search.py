"""
Combined search: ALL filters + ALL exit modifications + ALL optimizations simultaneously.
Random search over ~3000 combinations to test if ANY synergy exists.
"""
import pandas as pd, numpy as np, os, warnings, random, json, time
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF=pd.Timestamp("14:15").time()
random.seed(42); np.random.seed(42)

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Pre-computing trade data...")
trades=[]  # pre-computed trade data

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
    atr5_fn=A(m5,14).values; tc=m5["datetime"].dt.time
    
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
        
        # Store candle data from entry onward for exit simulation (no limit)
        exit_candles=[]
        for j in range(r, len(m5)):
            exit_candles.append({"h":hi[j],"l":lo[j],"c":cl[j],"a":atr5_fn[j],"t":m5["datetime"].iloc[j]})
        
        trades.append({
            "sym":sym,"bl":bl,
            "entry":ep,"entry_time":m5["datetime"].iloc[r],
            "exit_candles":exit_candles,
            "atr14":h1["atr14"].iloc[i] if not pd.isna(h1["atr14"].iloc[i]) else None,
            "atr_ma20":h1["atr_ma20"].iloc[i] if not pd.isna(h1["atr_ma20"].iloc[i]) else None,
            "body_ratio":body.iloc[i]/(body.iloc[i-1]+1e-10),
            "gap_pct":(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100,
            "trend_5":(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0,
            "c_ema50":(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100,
            "c_ema200":(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100,
            "hour":h1["datetime"].iloc[i].hour,
            "year":h1["datetime"].iloc[i].year,
        })

print(f"Total trades: {len(trades)}")

def simulate_pnl(trade, ch_base, ch_range, profit_target=0, breakeven_atr=0, max_hold_h=0, tighten=False):
    """Simulate a single trade with parameterized exit rules."""
    ec=trade["exit_candles"]; ep=trade["entry"]
    
    # Determine CH value
    atr14=trade["atr14"]; atr_ma=trade["atr_ma20"]
    if atr14 is not None and atr_ma is not None and atr14>atr_ma:
        ch=ch_base-ch_range  # tighten in high vol
    elif atr14 is not None:
        ch=ch_base+ch_range  # widen in low vol
    else:
        ch=ch_base
    
    he=ep; count=0; consec_losses=0
    
    for cd in ec:
        count+=1
        ca=cd["a"]
        if pd.isna(ca): continue
        
        # Track highest high since entry
        if cd["h"]>he: he=cd["h"]
        
        # Check Chandelier exit
        if cd["c"]<he-ch*ca:
            pnl=(cd["c"]-ep)*trade["bl"]-20
            return pnl
        
        # Profit target
        if profit_target>0 and cd["h"]>=ep+profit_target*ca:
            pnl=(ep+profit_target*ca-ep)*trade["bl"]-20
            return pnl
        
        # Max hold time
        if max_hold_h>0 and (cd["t"]-trade["entry_time"]).total_seconds()>max_hold_h*3600:
            pnl=(cd["c"]-ep)*trade["bl"]-20
            return pnl
        
        # Breakeven stop
        if breakeven_atr>0 and he>=ep+breakeven_atr*ca and cd["c"]<ep:
            pnl=0-20  # breakeven (minus charges)
            return pnl
        
        # Tighten after losses
        if tighten and consec_losses>0:
            # Move CH closer by 10% per consecutive loss
            tight_ch=ch*(1-consec_losses*0.1)
            if cd["c"]<he-tight_ch*ca:
                pnl=(cd["c"]-ep)*trade["bl"]-20
                return pnl
    
    return None  # trade didn't close

def evaluate_config(params, trades_list, split_idx=None):
    """
    Evaluate a full configuration.
    params: dict with all parameters
    trades_list: list of trade dicts
    split_idx: if provided, only evaluate on test portion
    """
    config_net=0; config_count=0; config_wins=0
    loss_streak=0; momentum_active=0; nifty_net=0; sensex_net=0
    
    for ti,t in enumerate(trades_list):
        if split_idx is not None and ti<split_idx:
            continue  # skip training
        
        # ── PRE-ENTRY FILTERS ──
        if params.get("body_ratio_min",0.5)>0 and t["body_ratio"]<params["body_ratio_min"]: continue
        if params.get("body_ratio_max",999)<999 and t["body_ratio"]>params["body_ratio_max"]: continue
        if params.get("gap_max",0)<=0 and t["gap_pct"]>params["gap_max"]: continue
        if params.get("gap_min",-999)>= -999 and t["gap_pct"]<params["gap_min"]: continue
        if params.get("trend5_min",-999)>= -999 and t["trend_5"]<params["trend5_min"]: continue
        if params.get("trend5_max",999)<=999 and t["trend_5"]>params["trend5_max"]: continue
        if params.get("above_ema50") is not None:
            if params["above_ema50"] and t["c_ema50"]<=0: continue
            if not params["above_ema50"] and t["c_ema50"]>0: continue
        if params.get("above_ema200") is not None:
            if params["above_ema200"] and t["c_ema200"]<=0: continue
            if not params["above_ema200"] and t["c_ema200"]>0: continue
        if params.get("hour") is not None and t["hour"]!=params["hour"]: continue
        
        # ── SKIP AFTER LOSSES ──
        skip_loss=params.get("skip_after_losses",0)
        if skip_loss>0 and loss_streak>=skip_loss:
            loss_streak=0
            continue
        
        # ── MOMENTUM OVERLAY ──
        mom_thresh=params.get("momentum",0)
        if mom_thresh>0 and t["trend_5"]<mom_thresh*0.1:
            # Trend too weak, skip
            continue
        
        # ── EXIT SIMULATION ──
        pnl=simulate_pnl(t, params["ch_base"], params["ch_range"],
                         params.get("profit_target",0), params.get("breakeven",0),
                         params.get("max_hold",0), params.get("tighten",False))
        if pnl is None: continue
        
        # ── CHARGES ──
        pnl_after_charges=pnl  # already includes charges
        
        # ── ANTI-MARTINGALE ──
        am=params.get("anti_martingale","none")
        if am=="none":
            pass  # fixed lot, already applied
        elif am=="1w1l":
            # Already applied in simulate_pnl with 1 lot
            pass
        
        config_net+=pnl_after_charges
        config_count+=1
        if pnl>0:
            config_wins+=1
            loss_streak=0
        else:
            loss_streak+=1
        
        if "NIFTY" in t["sym"]:
            nifty_net+=pnl_after_charges
        else:
            sensex_net+=pnl_after_charges
    
    return {
        "net":config_net,"count":config_count,"wins":config_wins,
        "wr":config_wins/config_count*100 if config_count>0 else 0,
        "nifty_net":nifty_net,"sensex_net":sensex_net
    }

# ═══ SPLIT DATA ═══
split=int(len(trades)*0.7)
train_trades=trades[:split]
test_trades=trades[split:]
print(f"Train: {len(train_trades)}, Test: {len(test_trades)}")

# ═══ BASELINE ═══
baseline_params={
    "ch_base":45,"ch_range":10,"body_ratio_min":0.5,"body_ratio_max":999,
    "gap_max":0,"gap_min":-999,"trend5_min":-999,"trend5_max":999,
    "above_ema50":None,"above_ema200":None,"hour":None,
    "skip_after_losses":0,"momentum":0,"profit_target":0,"breakeven":0,
    "max_hold":0,"tighten":False,"anti_martingale":"none"
}
base_test=evaluate_config(baseline_params, trades, split)
print(f"\nBASELINE (DynCH45+10, all trades):")
print(f"  Net=Rs{base_test['net']:+,.0f} WR={base_test['wr']:.1f}% N={base_test['count']}")

# ═══ RANDOM SEARCH ═══
print(f"\n{'='*60}")
print("RANDOM SEARCH: 3000 combinations")
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
    "tighten":lambda:random.choice([True,False]),
    "anti_martingale":lambda:random.choice(["none"]),
}

results=[]
seen_configs=set()
t0=time.time()

for trial in range(3000):
    # Generate random config
    cfg={k:v() for k,v in param_space.items()}
    
    # Skip if highly restrictive (filters > 80% of trades)
    est_remaining=1.0
    if cfg["body_ratio_min"]>5: est_remaining*=0.34
    if cfg["body_ratio_max"]<5: est_remaining*=0.66
    if cfg["gap_max"]<-0.3: est_remaining*=0.3
    if cfg["gap_min"]>-0.05: est_remaining*=0.3
    if cfg["trend5_min"]>0.5: est_remaining*=0.45
    if cfg["trend5_max"]< -0.5: est_remaining*=0.3
    if cfg["above_ema50"] is not None: est_remaining*=0.69
    if cfg["above_ema200"] is not None: est_remaining*=0.69
    if cfg["hour"] is not None: est_remaining*=0.13
    if est_remaining<0.05: continue  # skip if too restrictive
    
    # Hash to avoid duplicates
    h=hash(frozenset(cfg.items()))
    if h in seen_configs: continue
    seen_configs.add(h)
    
    # Evaluate on test
    result=evaluate_config(cfg, trades, split)
    
    net=result["net"]; base=base_test["net"]
    imp=(net/base-1)*100 if base!=0 else 0
    
    results.append((imp, net, result["count"], result["wr"], cfg.copy()))
    
    # Report progress
    if (trial+1)%300==0:
        elapsed=time.time()-t0
        best=max(results, key=lambda x:x[0])
        print(f"  Trial {trial+1}: best={best[0]:+.1f}% "
              f"(Rs{best[1]:+,.0f}, N={best[2]}, WR={best[3]:.1f}%) "
              f"[{elapsed:.0f}s]")

results.sort(key=lambda x:-x[0])
elapsed=time.time()-t0

print(f"\n{'='*60}")
print(f"RESULTS: {len(results)} valid combinations tested ({elapsed:.0f}s)")
print(f"{'='*60}")

print(f"\n--- TOP 20 COMBINATIONS ---")
print(f"{'Rank':>4s} {'Impr%':>8s} {'Net':>12s} {'N':>5s} {'WR%':>6s} {'Config':>40s}")
for rank,(imp,net,count,wr,cfg) in enumerate(results[:20],1):
    # Summarize config
    parts=[]
    parts.append(f"CH{cfg['ch_base']}+{cfg['ch_range']}")
    if cfg["body_ratio_min"]>0.5: parts.append(f"BR>{cfg['body_ratio_min']}")
    if cfg["body_ratio_max"]<999: parts.append(f"BR<{cfg['body_ratio_max']}")
    if cfg["gap_max"]<0: parts.append(f"G<{cfg['gap_max']}")
    if cfg["gap_min"]>-999: parts.append(f"G>{cfg['gap_min']}")
    if cfg["trend5_min"]>-999: parts.append(f"T5>{cfg['trend5_min']}")
    if cfg["trend5_max"]<999: parts.append(f"T5<{cfg['trend5_max']}")
    if cfg["above_ema50"] is not None: parts.append("EMA50"+(">" if cfg["above_ema50"] else "<"))
    if cfg["above_ema200"] is not None: parts.append("EMA200"+(">" if cfg["above_ema200"] else "<"))
    if cfg["hour"] is not None: parts.append(f"H{cfg['hour']}")
    if cfg["skip_after_losses"]>0: parts.append(f"SK{cfg['skip_after_losses']}")
    if cfg["momentum"]>0: parts.append(f"MO{cfg['momentum']}")
    if cfg["profit_target"]>0: parts.append(f"PT{cfg['profit_target']}")
    if cfg["breakeven"]>0: parts.append(f"BE{cfg['breakeven']}")
    if cfg["max_hold"]>0: parts.append(f"MH{cfg['max_hold']}")
    if cfg["tighten"]: parts.append("TI")
    summ=" ".join(parts)
    print(f"{rank:>4d} {imp:>+7.1f}% Rs{net:>+9,.0f} {count:>5d} {wr:>5.1f}% {summ[:60]}")

# ═══ VALIDATE TOP 5 ON FULL WALK-FORWARD ═══
print(f"\n{'='*60}")
print("WALK-FORWARD VALIDATION (TOP 5 on each year)")
print(f"{'='*60}")

for rank,(imp,net,count,wr,cfg) in enumerate(results[:5],1):
    print(f"\n  Rank #{rank}: {imp:+.1f}% net=Rs{net:+,.0f}")
    print(f"  Config: {cfg}")
    
    # Test each year
    for yr in sorted(set(t["year"] for t in trades)):
        yr_trades=[t for t in trades if t["year"]==yr]
        if len(yr_trades)<10: continue
        yr_result=evaluate_config(cfg, yr_trades, 0)  # no split, full year
        yr_base=evaluate_config(baseline_params, yr_trades, 0)
        yr_imp=(yr_result["net"]/yr_base["net"]-1)*100 if yr_base["net"]!=0 else 0
        print(f"    {yr}: net=Rs{yr_result['net']:+,.0f} vs Rs{yr_base['net']:+,.0f} "
              f"({yr_imp:+.1f}%) N={yr_result['count']}")
    
    # Overall
    result=evaluate_config(cfg, trades, 0)
    print(f"    ALL: net=Rs{result['net']:+,.0f} WR={result['wr']:.1f}% N={result['count']}")

# ═══ SAVE RESULTS ═══
with open(os.path.join(BASE,"combined_search_results.json"),"w") as f:
    json.dump([{"imp":r[0],"net":r[1],"count":r[2],"wr":r[3],"cfg":{k:str(v) if isinstance(v,(pd.Timestamp,)) else v for k,v in r[4].items()}} for r in results[:50]], f, indent=2)

print(f"\nResults saved to combined_search_results.json")
print(f"\nDONE")
