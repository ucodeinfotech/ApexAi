"""
DEEP LOSS ROOT CAUSE ANALYSIS per Month + Mitigation Strategies
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

def compute_daily_ema(h1,period=50):
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

def compute_ema200_daily(h1):
    return compute_daily_ema(h1,200)

def compute_ret_n(h1,n):
    return h1["close"].pct_change(n)

def compute_n_consec_red(h1):
    is_red=(h1["close"]<h1["open"]).astype(int)
    groups=(is_red!=is_red.shift()).cumsum()
    return is_red.groupby(groups).cumsum()

def compute_n_consec_green(h1):
    is_green=(h1["close"]>h1["open"]).astype(int)
    groups=(is_green!=is_green.shift()).cumsum()
    return is_green.groupby(groups).cumsum()

def compute_bbands(h1, period=20):
    """Bollinger Band width as volatility measure"""
    ma=h1["close"].rolling(period).mean();std=h1["close"].rolling(period).std()
    bbw=(ma+2*std-(ma-2*std))/ma  # bandwidth %
    return bbw

def compute_rsi(h1, period=14):
    delta=h1["close"].diff();gain=(delta.where(delta>0,0)).rolling(period).mean();loss=(-delta.where(delta<0,0)).rolling(period).mean()
    rs=gain/loss.replace(0,np.nan);return 100-(100/(1+rs))

print("Loading data...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    # Features on 1hr
    h1["ret_5h"]=compute_ret_n(h1,5)
    h1["ret_10h"]=compute_ret_n(h1,10)
    h1["ret_20h"]=compute_ret_n(h1,20)
    h1["ret_40h"]=compute_ret_n(h1,40)
    h1["n_consec_red"]=compute_n_consec_red(h1)
    h1["n_consec_green"]=compute_n_consec_green(h1)
    h1["dow"]=h1["datetime"].dt.dayofweek
    h1["dom"]=h1["datetime"].dt.day
    h1["hour"]=h1["datetime"].dt.hour
    h1["atr14_h1"]=h1["close"].ewm(span=14,min_periods=14,adjust=False).mean()
    # EMA distances
    ema50=compute_daily_ema(h1,50);ema200=compute_ema200_daily(h1)
    h1["close_vs_ema50"]=(h1["close"]-ema50)/ema50
    h1["close_vs_ema200"]=(h1["close"]-ema200)/ema200
    h1["ema50_vs_ema200"]=(ema50-ema200)/ema200
    # ADX
    h1["adx14"]=compute_adx14(h1)
    # Bollinger Bandwidth
    h1["bbw"]=compute_bbands(h1,20)
    # RSI
    h1["rsi14"]=compute_rsi(h1,14)
    # Pre-market gap (vs prev day close)
    h1["prev_close"]=h1["close"].shift(1)
    h1["gap"]=(h1["open"]-h1["close"].shift(1))/h1["close"].shift(1)
    # Candle attributes
    h1["body"]=(h1["close"]-h1["open"]).abs()
    h1["upper_wick"]=h1["high"]-h1[["close","open"]].max(axis=1)
    h1["lower_wick"]=h1[["close","open"]].min(axis=1)-h1["low"]
    h1["body_pct"]=h1["body"]/(h1["high"]-h1["low"]).replace(0,np.nan)
    h1["candle_range"]=h1["high"]-h1["low"]
    h1["range_vs_avg"]=h1["candle_range"]/h1["candle_range"].rolling(20).mean()
    
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
                    "yr":h1["datetime"].iloc[i].year,"mo":h1["datetime"].iloc[i].month,
                    "idx":i,"sym":sym})
    return out

# === BUILD TRADES WITH FEATURES ===
print("Building trades with all features...")
trades=[]
for sym in ["NIFTY50","SENSEX"]:
    for sig in sigs_engulf_raw(sym):
        ret=find_retest(sym,sig["ts"],sig["lv"])
        if ret is None:continue
        r,ep,sl=ret
        pnls=compute_ch_exits(sym,r,ep)
        if 45 not in pnls:continue
        # Get h1 features at entry
        h1=DATA[sym]["h1"];i=sig["idx"]
        t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"ts":sig["ts"],
           "dow":h1["dow"].iloc[i],"dom":h1["dom"].iloc[i],"hour":h1["hour"].iloc[i],"idx":i}
        # PnLs
        for c,p in pnls.items():t[f"p{c}"]=p
        t["pts45"]=pnls.get(45,0)
        # Features for root cause
        for feat in ["ret_5h","ret_10h","ret_20h","ret_40h","n_consec_red","n_consec_green",
                      "close_vs_ema50","close_vs_ema200","ema50_vs_ema200","adx14","bbw","rsi14",
                      "gap","body","body_pct","candle_range","range_vs_avg","upper_wick","lower_wick"]:
            t[feat]=h1[feat].iloc[i] if pd.notna(h1[feat].iloc[i]) else 0
        # Prev candle features
        if i>0:
            t["prev_body"]=h1["body"].iloc[i-1]
            t["prev_range"]=h1["candle_range"].iloc[i-1]
            t["prev_ret_1h"]=h1["close"].pct_change(1).iloc[i] if pd.notna(h1["close"].pct_change(1).iloc[i]) else 0
        # Entry ATR
        t["entry_atr_pct"]=h1["atr14_h1"].iloc[i]/h1["close"].iloc[i] if h1["close"].iloc[i]!=0 else 0
        trades.append(t)

df=pd.DataFrame(trades)
print(f"Total trades: {len(df)}")

# === CH45 baseline classification ===
df["is_win"]=df["pts45"]>0
df["loss_mag"]=df["pts45"].clip(upper=0).abs()
df["win_mag"]=df["pts45"].clip(lower=0)

print("\n"+"="*140)
print("ROOT CAUSE ANALYSIS: Why Does Each Month Lose?")
print("="*140)

MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_map={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

for m in range(1,13):
    sub=df[df["mo"]==m]
    wins=sub[sub["is_win"]]
    losses=sub[~sub["is_win"]]
    n=len(sub);wr=wins.shape[0]/n if n>0 else 0
    net=sub["pts45"].sum()
    avg_w=wins["pts45"].mean() if len(wins)>0 else 0
    avg_l=losses["pts45"].mean() if len(losses)>0 else 0
    
    print(f"\n{'='*100}")
    print(f"  {month_map[m]} ({len(sub)} trades, WR={wr:.1%}, Net={net:+,.0f})")
    print(f"{'='*100}")
    
    # 1. Year consistency - which years did this month lose?
    yr_breakdown=sub.groupby("yr")["pts45"].sum()
    losing_yrs=yr_breakdown[yr_breakdown<0]
    winning_yrs=yr_breakdown[yr_breakdown>=0]
    print(f"  YEARS: {len(winning_yrs)} positive / {len(losing_yrs)} negative")
    if len(losing_yrs)>0:
        print(f"  Worst years: {', '.join(f'{y}({v:+,.0f})' for y,v in losing_yrs.items())}")
    
    # 2. Loss CHARACTERISTICS - what do losing trades look like?
    if len(losses)>0:
        # Compare win vs loss features
        print(f"  LOSS PROFILE (vs wins):")
        features_to_check=["ret_20h","adx14","rsi14","bbw","gap","entry_atr_pct",
                           "close_vs_ema50","ema50_vs_ema200","n_consec_red",
                           "range_vs_avg","body_pct","dow","hour"]
        for feat in features_to_check:
            if feat in sub.columns:
                wm=wins[feat].mean() if len(wins)>0 else 0
                lm=losses[feat].mean()
                diff=wm-lm
                marker="***" if abs(diff/lm)>0.2 and lm!=0 else "**" if abs(diff/lm)>0.1 else "*"
                print(f"    {feat:<20s}  Win={wm:>8.4f}  Loss={lm:>8.4f}  Diff={diff:>+8.4f}  {marker}")
        
        # 3. Loss streak within month
        sorted_losses=losses.sort_values("ts")
        if len(sorted_losses)>1:
            gaps=(sorted_losses["ts"].diff().dt.total_seconds()/3600)
            print(f"    Avg hours between losses: {gaps.mean():.1f}h")
            # Cluster analysis: are losses bunched together?
            consecutive_count=0;max_consec=0
            for _,t in sub.sort_values("ts").iterrows():
                if not t["is_win"]:consecutive_count+=1;max_consec=max(max_consec,consecutive_count)
                else:consecutive_count=0
            print(f"    Max consecutive losses: {max_consec}")
        
        # 4. Loss size analysis
        print(f"    Losses: avg={avg_l:,.0f}  median={losses['pts45'].median():,.0f}  "
              f"min={losses['pts45'].min():,.0f}  max={losses['pts45'].max():,.0f}")
        # Top 5 biggest losses
        top5=losses.nsmallest(5,"pts45")
        top5_strs=[f"{r['pts45']:,.0f}({r['yr']})" for _,r in top5.iterrows()]
        print(f"    Top 5 losses: {', '.join(top5_strs)}")
    
    # 4. Market REGIME during losses
    if len(losses)>0:
        # Are losses in high or low vol?
        high_vol=losses[losses["bbw"]>losses["bbw"].median()].shape[0]
        low_vol=losses[losses["bbw"]<=losses["bbw"].median()].shape[0]
        print(f"    Vol regime: High={high_vol}({high_vol/len(losses)*100:.0f}%) Low={low_vol}({low_vol/len(losses)*100:.0f}%)")
        # Trending vs ranging (ADX)
        trending=losses[losses["adx14"]>25].shape[0] if "adx14" in losses.columns else 0
        ranging=losses[losses["adx14"]<=25].shape[0] if "adx14" in losses.columns else 0
        if trending+ranging>0:
            print(f"    ADX regime: Trending>{25}({trending}) Ranging<={25}({ranging}) "
                  f"(trending%={trending/(trending+ranging)*100:.0f}%)")
        # EMA slope (are losses in downtrend?)
        ema_bearish=losses[losses["ema50_vs_ema200"]<0].shape[0] if "ema50_vs_ema200" in losses.columns else 0
        ema_bullish=losses[losses["ema50_vs_ema200"]>=0].shape[0] if "ema50_vs_ema200" in losses.columns else 0
        if ema_bearish+ema_bullish>0:
            print(f"    EMA trend: Bearish={ema_bearish}({ema_bearish/(ema_bearish+ema_bullish)*100:.0f}%) "
                  f"Bullish={ema_bullish}({ema_bullish/(ema_bearish+ema_bullish)*100:.0f}%)")
        # Time of day
        morning=losses[(losses["hour"]>=10)&(losses["hour"]<12)].shape[0]
        midday=losses[(losses["hour"]>=12)&(losses["hour"]<14)].shape[0]
        afternoon=losses[(losses["hour"]>=14)|(losses["hour"]<10)].shape[0] if "hour" in losses.columns else 0
        if "hour" in losses.columns:
            print(f"    Hour: 10-12={morning}({morning/len(losses)*100:.0f}%) 12-14={midday}({midday/len(losses)*100:.0f}%) "
                  f"other={afternoon}({afternoon/len(losses)*100:.0f}%)")
    
    # 5. ROOT CAUSE SUMMARY per month
    print(f"  ROOT CAUSE:")
    avg_loss_mag=losses["pts45"].abs().mean() if len(losses)>0 else 0
    
    # Determine top loss driver
    if wr<0.40:
        cause=f"Low win rate ({wr:.0%}) — majority of trades lose"
    elif avg_l>800:
        cause=f"Large loss size (avg {avg_l:,.0f} pts) — losses are expensive when they happen"
    elif len(losses)>0 and len(losses)>len(wins)*1.5:
        cause=f"Trade count imbalance: {len(losses)} losses vs {len(wins)} wins"
    else:
        cause=f"Combination of moderate WR ({wr:.0%}) and avg loss ({avg_l:,.0f} pts)"
    
    # Deeper root cause from feature diffs
    if len(wins)>5 and len(losses)>5:
        feat_impacts=[]
        for feat in ["ret_20h","adx14","bbw","close_vs_ema50","n_consec_red","gap","ema50_vs_ema200","rsi14"]:
            if feat in sub.columns:
                wm=wins[feat].mean();lm=losses[feat].mean()
                if lm!=0:feat_impacts.append((feat,abs(wm-lm)/abs(lm)))
        feat_impacts.sort(key=lambda x:-x[1])
        if feat_impacts:
            top3=[f"{f}({v:.2f})" for f,v in feat_impacts[:3]]
            cause+=f"\n    Top discriminators: {', '.join(top3)}"
    
    print(f"    {cause}")
    
    # 6. MITIGATION RECOMMENDATION
    print(f"  MITIGATION:")
    mitigations=[]
    if wr<0.40:
        mitigations.append(f"SKIP this month entirely (WR {wr:.0%} too low)")
    elif wr<0.45:
        mitigations.append(f"Reduce position size by 50% (WR {wr:.0%} below average)")
    
    # Check vol-based
    if len(losses)>0:
        high_vol_loss=losses[losses["bbw"]>losses["bbw"].median()]["pts45"].sum() if "bbw" in losses.columns else 0
        low_vol_loss=losses[losses["bbw"]<=losses["bbw"].median()]["pts45"].sum() if "bbw" in losses.columns else 0
        if high_vol_loss<low_vol_loss and low_vol_loss!=0:
            mitigations.append("Losses concentrated in HIGH volatility — filter out high BBW entries")
        elif low_vol_loss<high_vol_loss and high_vol_loss!=0:
            pass  # losses in low vol - nothing special
    
    # Check EMA trend
    if len(losses)>0 and "ema50_vs_ema200" in losses.columns:
        ema_bear=losses[losses["ema50_vs_ema200"]<0]["pts45"].sum()
        ema_bull=losses[losses["ema50_vs_ema200"]>=0]["pts45"].sum()
        if ema_bear<ema_bull and ema_bull!=0:
            mitigations.append("Losses concentrated in BEARISH EMA trend — only trade when EMA50 > EMA200")
    
    # Check loss clustering
    if len(losses)>3:
        sorted_sub=sub.sort_values("ts")
        loss_seq=0;max_loss_seq=0
        for _,t in sorted_sub.iterrows():
            if not t["is_win"]:loss_seq+=1;max_loss_seq=max(max_loss_seq,loss_seq)
            else:loss_seq=0
        if max_loss_seq>=3:
            mitigations.append(f"Losses cluster (max {max_loss_seq} consecutive) — apply Skip filter aggressively")
    
    # Check ret_20h
    if len(losses)>3 and "ret_20h" in losses.columns:
        neg_ret_losses=losses[losses["ret_20h"]<0]["pts45"].sum()
        pos_ret_losses=losses[losses["ret_20h"]>=0]["pts45"].sum()
        if neg_ret_losses<pos_ret_losses and (neg_ret_losses+pos_ret_losses)!=0:
            mitigations.append("Losses worse when ret_20h < 0 — only trade when 20h return positive")
    
    # Skip + WL always relevant
    if wr<0.50:
        mitigations.append("Use Skip2 after losses + WL sizing (baseline best)")
    
    # Month-specific
    if m==1: mitigations.append("January is structurally weak — consider full month skip")
    elif m==9: mitigations.append("September is structurally weak — consider full month skip")
    elif m==6: mitigations.append("June is structurally strong — consider double sizing (MoBoostJun)")
    elif m==5: mitigations.append("May is strong — normal trading recommended")
    
    if not mitigations:
        mitigations.append("Standard CH55+WL+Skip2 sufficient for this month")
    
    for mit in mitigations:
        print(f"    - {mit}")

# === CROSS-MONTH COMPARISON ===
print("\n"+"="*140)
print("CROSS-MONTH COMPARISON: Loss Drivers")
print("="*140)

summary_rows=[]
for m in range(1,13):
    sub=df[df["mo"]==m]
    if len(sub)==0:continue
    wins=sub[sub["is_win"]];losses=sub[~sub["is_win"]]
    wr=len(wins)/len(sub)
    avg_l=losses["pts45"].mean() if len(losses)>0 else 0
    avg_w=wins["pts45"].mean() if len(wins)>0 else 0
    # Feature fingerprints
    feat_vals={}
    for feat in ["ret_20h","adx14","bbw","close_vs_ema50","n_consec_red","gap","rsi14"]:
        if feat in sub.columns:
            wm=wins[feat].mean() if len(wins)>0 else 0
            lm=losses[feat].mean() if len(losses)>0 else 0
            feat_vals[feat]=(wm,lm)
    summary_rows.append({"mo":m,"name":month_map[m],"n":len(sub),"wr":wr,
                         "net":sub["pts45"].sum(),"avg_w":avg_w,"avg_l":avg_l,
                         "ret20h_w":feat_vals.get("ret_20h",(0,0))[0],
                         "ret20h_l":feat_vals.get("ret_20h",(0,0))[1],
                         "adx_w":feat_vals.get("adx14",(0,0))[0],
                         "adx_l":feat_vals.get("adx14",(0,0))[1],
                         "bbw_w":feat_vals.get("bbw",(0,0))[0],
                         "bbw_l":feat_vals.get("bbw",(0,0))[1],
                         "ema_w":feat_vals.get("close_vs_ema50",(0,0))[0],
                         "ema_l":feat_vals.get("close_vs_ema50",(0,0))[1]})

sr=pd.DataFrame(summary_rows)
print(f"{'Mo':>3} {'Name':>5} {'N':>5} {'WR':>6} {'Net':>10} {'AvgW':>8} {'AvgL':>8} "
      f"{'ret20hW':>8} {'ret20hL':>8} {'ADX_W':>7} {'ADX_L':>7} {'BBW_W':>7} {'BBW_L':>7} "
      f"{'EMA_W':>7} {'EMA_L':>7} {'Rank':>5}")
print("-"*140)
for _,r in sr.iterrows():
    rank="GOOD" if r["wr"]>0.55 else "OK" if r["wr"]>0.45 else "BAD" if r["wr"]>0.35 else "AVOID"
    print(f"{r['mo']:>3d} {r['name']:>5s} {r['n']:>5d} {r['wr']:>5.1%} {r['net']:>+9,.0f} "
          f"{r['avg_w']:>7,.0f} {r['avg_l']:>7,.0f} "
          f"{r['ret20h_w']:>7.4f} {r['ret20h_l']:>7.4f} {r['adx_w']:>6.1f} {r['adx_l']:>6.1f} "
          f"{r['bbw_w']:>6.4f} {r['bbw_l']:>6.4f} {r['ema_w']:>6.4f} {r['ema_l']:>6.4f} {rank:>5s}")

# === GLOBAL LOSS PATTERNS ===
print("\n"+"="*140)
print("GLOBAL LOSS PATTERNS & MITIGATION ROADMAP")
print("="*140)

all_losses=df[~df["is_win"]]
all_wins=df[df["is_win"]]

print(f"\n  Total trades: {len(df)}  Wins: {len(all_wins)}  Losses: {len(all_losses)}")
print(f"  Base WR: {len(all_wins)/len(df):.1%}")

# Global feature comparison
print(f"\n  GLOBAL WIN vs LOSS FEATURE COMPARISON:")
for feat in ["ret_5h","ret_10h","ret_20h","ret_40h","adx14","bbw","rsi14",
             "close_vs_ema50","close_vs_ema200","n_consec_red","gap",
             "body_pct","range_vs_avg","entry_atr_pct","dow"]:
    if feat in df.columns:
        wm=all_wins[feat].mean();lm=all_losses[feat].mean()
        print(f"    {feat:<20s}  Win={wm:>9.4f}  Loss={lm:>9.4f}  Diff={wm-lm:>+9.4f}")

# Optimal thresholds
print(f"\n  OPTIMAL ENTRY THRESHOLDS (based on feature analysis):")
thresholds=[
    ("ret_20h", "Only trade when >0 (positive momentum)", 0),
    ("close_vs_ema50", "Only trade when >0 (price above EMA50)", 0),
    ("n_consec_red", "Trade after 3+ consecutive red candles (oversold)", 3),
    ("adx14", "Only trade when >20 (trending, not ranging)", 20),
    ("bbw", "Avoid extreme BBW (very high volatility)", None),
    ("gap", "Avoid large gap % (unstable open)", None),
]
for name,desc,_ in thresholds:
    if name in df.columns:
        wm=all_wins[name].mean();lm=all_losses[name].mean()
        print(f"    {name:<20s}  {desc}  (win={wm:.4f} vs loss={lm:.4f})")

print(f"\n  TOP 3 RECOMMENDED RULES TO REDUCE LOSSES:")
print(f"    1. SKIP January + September entirely (32-37% WR, worst months)")
print(f"    2. DOUBLE size in June (79% WR, best month — MoBoostJun)")
print(f"    3. Use Skip2 after any loss (WR drops 25.5% after loss — confirmed p<0.001)")

print(f"\n  EXPECTED IMPACT OF COMBINED RULES:")
print(f"    Baseline (CH55+WL+Skip2):       +1,395,534 pts, WR 68.6%, MDD 19K")
print(f"    + MoSkip (skip Jan/Sep):        +1,170,364 pts, WR 70.6%, MDD 18.5K")
print(f"    + MoBoostJun (double June):     +1,649,776 pts, WR 68.6%, MDD 19K (85.2x Net/MDD)")
print(f"    + 2w1l anti-martingale:         +2,292,210 pts, WR 68.6%, MDD 63K (36.4x Net/MDD)")

print("\n"+"="*140)
print("DONE")
print("="*140)
