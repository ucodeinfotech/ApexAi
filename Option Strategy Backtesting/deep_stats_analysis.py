"""
DEEP STATISTICAL ANALYSIS: Why trades win or lose
All strategies x all features - statistical + ML analysis
"""
import pandas as pd, numpy as np, os, warnings, json, itertools
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
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
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

# ═══ LOAD DATA ═══
print("Loading data...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    tr_h1=pd.concat([h1["high"]-h1["low"],abs(h1["high"]-h1["close"].shift(1)),abs(h1["low"]-h1["close"].shift(1))],axis=1).max(axis=1)
    atr14_h1=tr_h1.ewm(span=14,min_periods=14,adjust=False).mean()
    DATA[sym]={"h1":h1,"m5":m5,"atr5v":atr5.values,"m5_hi":m5["high"].values,"m5_lo":m5["low"].values,
               "m5_cl":m5["close"].values,"m5_ep":m5["datetime"].astype('int64').values,
               "tc":pd.Series(m5["datetime"]).dt.time.values,"atr14_h1":atr14_h1.values}
CUT=pd.Timestamp("14:15").time()

def find_retest(sym, t, lv):
    d=DATA[sym];epc=d["m5_ep"];cl=d["m5_cl"];lo=d["m5_lo"];tc=d["tc"]
    t_ep=t.asm8.view('int64')
    idx=np.searchsorted(epc,t_ep,side="right")
    if idx>=len(cl):return None
    b=idx
    while b<len(cl) and cl[b]<=lv:b+=1
    if b>=len(cl)-1:return None
    r=b+1
    while r<len(cl):
        if lo[r]<lv and cl[r]>lv and tc[r]<CUT:break
        r+=1
    if r>=len(cl):return None
    ep=cl[r];sl=lo[r]
    if ep-sl<=0:return None
    return (r, ep, sl, idx)

def compute_ch_exits(sym, r, ep, max_bars=500):
    d=DATA[sym];atr5v=d["atr5v"];hi=d["m5_hi"];cl=d["m5_cl"];lo=d["m5_lo"]
    pnls={};exits={}
    for cv in CH_VALS:
        he=ep;exit_pts=None;exit_bar=0
        for j in range(r, min(r+max_bars, len(cl))):
            ca=atr5v[j]
            if pd.isna(ca):continue
            if hi[j]>he:he=hi[j]
            if cl[j]<he-cv*ca:
                exit_pts=round(cl[j]-ep,1);exit_bar=j-r;break
        if exit_pts is not None:
            pnls[cv]=exit_pts;exits[cv]=exit_bar
    return pnls, exits

# ═══ ENTRY SIGNALS ═══
def sigs_engulf_raw(sym):
    h1=DATA[sym]["h1"];b=(h1["close"]-h1["open"]).abs();g=h1["close"]>h1["open"];r=h1["close"]<h1["open"]
    out=[]
    for i in range(1,len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]:continue
        if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9:continue
        out.append({"ts":h1["datetime"].iloc[i],"lv":h1["high"].iloc[i],"yr":h1["datetime"].iloc[i].year,
                    "mo":h1["datetime"].iloc[i].month,"i":i})
    return out

# ═══ BUILD TRADE DATABASE WITH FEATURES ═══
print("Building Engulf_Raw trades with features...")
all_trades=[]
for sym in ["NIFTY50","SENSEX"]:
    d=DATA[sym];h1=d["h1"];m5=d["m5"];atr14_h1=d["atr14_h1"]
    # Pre-compute h1 features
    body=(h1["close"]-h1["open"]).abs();upper_wick=h1["high"]-h1[["open","close"]].max(axis=1)
    lower_wick=h1[["open","close"]].min(axis=1)-h1["low"]
    h1_range=h1["high"]-h1["low"];h1_ret=h1["close"].pct_change()
    h1_vol=h1["volume"].values if "volume" in h1.columns else np.zeros(len(h1))
    # 5-min atr for entry volatility
    m5_atr14=compute_atr(m5);m5_atr5=m5_atr14.rolling(5).mean().values
    
    for sig in sigs_engulf_raw(sym):
        ret=find_retest(sym, sig["ts"], sig["lv"])
        if ret is None:continue
        r,ep,sl,entry_idx=ret
        pnls,exits=compute_ch_exits(sym, r, ep)
        if 45 not in pnls:continue
        i=sig["i"]
        
        # ── FEATURE ENGINEERING ──
        f={}
        f["sym"]=sym;f["yr"]=sig["yr"];f["mo"]=sig["mo"]
        
        # Entry candle features
        f["body"]=body.iloc[i];f["prev_body"]=body.iloc[i-1]
        f["body_ratio"]=body.iloc[i]/body.iloc[i-1] if body.iloc[i-1]>0 else 0
        f["upper_wick"]=upper_wick.iloc[i];f["lower_wick"]=lower_wick.iloc[i]
        f["candle_range"]=h1_range.iloc[i]
        f["prev_range"]=h1_range.iloc[i-1]
        f["range_ratio"]=h1_range.iloc[i]/h1_range.iloc[i-1] if h1_range.iloc[i-1]>0 else 0
        f["entry_hour"]=h1["datetime"].iloc[i].hour
        f["entry_minute"]=h1["datetime"].iloc[i].minute
        f["day_of_week"]=h1["datetime"].iloc[i].dayofweek
        
        # Volatility features
        f["atr14"]=atr14_h1[i] if i<len(atr14_h1) and not pd.isna(atr14_h1[i]) else 0
        f["atr14_pct"]=f["atr14"]/h1["close"].iloc[i]*100 if h1["close"].iloc[i]>0 else 0
        f["m5_atr_entry"]=m5_atr5[r] if r<len(m5_atr5) and not pd.isna(m5_atr5[r]) else 0
        
        # Trend features
        f["ema50"]=compute_daily_ema(h1,50)[i] if i<len(compute_daily_ema(h1,50)) else 0
        f["ema200"]=compute_daily_ema(h1,200)[i] if i<len(compute_daily_ema(h1,200)) else 0
        f["close_vs_ema50"]=h1["close"].iloc[i]/f["ema50"]-1 if f["ema50"]>0 else 0
        f["close_vs_ema200"]=h1["close"].iloc[i]/f["ema200"]-1 if f["ema200"]>0 else 0
        
        # Return features
        f["ret_1h"]=h1_ret.iloc[i] if not pd.isna(h1_ret.iloc[i]) else 0
        f["ret_prev_1h"]=h1_ret.iloc[i-1] if i>0 and not pd.isna(h1_ret.iloc[i-1]) else 0
        f["ret_5h"]=h1["close"].iloc[i]/h1["close"].iloc[max(0,i-5)]-1 if h1["close"].iloc[max(0,i-5)]>0 else 0
        f["ret_20h"]=h1["close"].iloc[i]/h1["close"].iloc[max(0,i-20)]-1 if h1["close"].iloc[max(0,i-20)]>0 else 0
        
        # Consecutive
        f["prev_green"]=1 if h1["close"].iloc[i-1]>h1["open"].iloc[i-1] else 0
        f["n_consec_red"]=0
        for ci in range(i-1, max(0,i-10), -1):
            if h1["close"].iloc[ci]<h1["open"].iloc[ci]:f["n_consec_red"]+=1
            else:break
        
        # Seasonality
        f["quarter"]=(sig["mo"]-1)//3+1
        f["month_sin"]=np.sin(2*np.pi*sig["mo"]/12)
        f["month_cos"]=np.cos(2*np.pi*sig["mo"]/12)
        
        # Market condition proxies  
        f["range_20_avg"]=h1_range.rolling(20).mean().iloc[i] if i>=20 else h1_range.iloc[i]
        f["range_vs_avg"]=h1_range.iloc[i]/f["range_20_avg"] if f["range_20_avg"]>0 else 1
        f["vix"]=h1_range.iloc[i]/h1_range.iloc[max(0,i-20):i+1].mean() if i>=20 else 1  # proxy
        
        # Volume (if available)
        if "volume" in h1.columns:
            f["volume"]=h1["volume"].iloc[i] if not pd.isna(h1["volume"].iloc[i]) else 0
            f["vol_ma20"]=h1["volume"].iloc[max(0,i-20):i+1].mean()
            f["vol_ratio"]=f["volume"]/f["vol_ma20"] if f["vol_ma20"]>0 else 1
        else:
            f["volume"]=0;f["vol_ma20"]=0;f["vol_ratio"]=1
        
        # PnL outcomes
        f["pts45"]=pnls[45]
        f["is_win"]=pnls[45]>0
        f["pnl_magnitude"]=abs(pnls[45])
        for cv in CH_VALS:
            if cv in pnls:f[f"ch{cv}"]=pnls[cv]
        
        # Exit timing
        f["exit_bar_45"]=exits.get(45,0)
        f["exit_bars_cv"]={cv:exits.get(cv,0) for cv in CH_VALS}
        
        all_trades.append(f)

trades=pd.DataFrame(all_trades)
print(f"Total trades: {len(trades)}")
print(f"Wins: {(trades['is_win']).sum()} Losses: {(~trades['is_win']).sum()}")
print(f"WR: {(trades['is_win']).mean():.1%}")

# ═══════════════════════════════════════════════════════════
# 1. MONTHLY ANALYSIS — Deep stats per month
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 1: MONTHLY DEEP ANALYSIS")
print("="*120)

mon_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_stats=[]
for m in range(1,13):
    sub=trades[trades["mo"]==m]
    if len(sub)<10:continue
    win=sub[sub["is_win"]]
    loss=sub[~sub["is_win"]]
    net=sub["pts45"].sum()
    wr=sub["is_win"].mean()
    aw=win["pts45"].mean() if len(win)>0 else 0
    al=abs(loss["pts45"].mean()) if len(loss)>0 else 0
    wl=aw/al if al>0 else 999
    mdd=sub["pts45"].cumsum().cummax().sub(sub["pts45"].cumsum()).max()
    # Win/loss consistency
    win_yrs=sub.groupby("yr")["is_win"].mean()
    yr_consistency=(win_yrs>0.4).mean()
    monthly_stats.append({"month":m,"name":mon_names[m-1],"n":len(sub),"net":net,"wr":wr,"wl":wl,
                         "avg_win":aw,"avg_loss":al,"mdd":mdd,"yr_consistency":yr_consistency,
                         "best_yr":sub.groupby("yr")["pts45"].sum().max(),"worst_yr":sub.groupby("yr")["pts45"].sum().min()})
    # Feature means for wins vs losses
    w_body=win["body"].mean() if len(win)>0 else 0
    l_body=loss["body"].mean() if len(loss)>0 else 0
    w_atr=win["atr14_pct"].mean() if len(win)>0 else 0
    l_atr=loss["atr14_pct"].mean() if len(loss)>0 else 0
    print(f"  {mon_names[m-1]:>4s}: {len(sub):4d} trades Net={net:>+8,.0f} WR={wr:.0%} W/L={wl:.1f}x MDD={mdd:>+6,.0f} "
          f"Body(W/L)={w_body:.0f}/{l_body:.0f} ATR%(W/L)={w_atr:.1f}%/{l_atr:.1f}%")

# ═══════════════════════════════════════════════════════════
# 2. WIN VS LOSS FEATURE COMPARISON
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 2: WINNER vs LOSER — Feature Comparison")
print("="*120)

win=trades[trades["is_win"]]
loss=trades[~trades["is_win"]]

features=["body","prev_body","body_ratio","candle_range","prev_range","range_ratio",
          "atr14","atr14_pct","ret_prev_1h","ret_5h","ret_20h",
          "close_vs_ema50","close_vs_ema200","n_consec_red",
          "entry_hour","day_of_week","range_vs_avg","vol_ratio"]

print(f"  {'Feature':<25s} {'Winner Mean':>12s} {'Loser Mean':>12s} {'Diff':>12s} {'p-value':>10s} {'Sig':>5s}")
print("  "+"-"*76)
for feat in features:
    if feat not in trades.columns:continue
    wm=win[feat].mean();lm=loss[feat].mean()
    diff=wm-lm
    try:
        t_stat,p_val=stats.ttest_ind(win[feat].dropna(),loss[feat].dropna(),equal_var=False)
        sig="***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "ns"
    except:
        p_val=1.0;sig="ns"
    print(f"  {feat:<25s} {wm:>+11.2f}  {lm:>+11.2f}  {diff:>+11.2f}  {p_val:.4f}  {sig:>5s}")

# ═══════════════════════════════════════════════════════════
# 3. CORRELATION MATRIX (selected features vs is_win)
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 3: FEATURE CORRELATION WITH TRADE OUTCOME")
print("="*120)
corr_features=features+["is_win"]
corr_df=trades[corr_features].dropna().corr()
win_corr=corr_df["is_win"].drop("is_win").sort_values(key=abs, ascending=False)
print(f"  {'Feature':<25s} {'Corr':>10s} {'Abs':>10s}")
print("  "+"-"*45)
for feat,corr_val in win_corr.items():
    print(f"  {feat:<25s} {corr_val:>+9.4f}  {abs(corr_val):>9.4f}")

# ═══════════════════════════════════════════════════════════
# 4. CH VALUE ANALYSIS — Which CH is best when
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 4: CH VALUE ANALYSIS — Best CH by market condition")
print("="*120)

# For each trade, find which CH would have been best
for cv in CH_VALS:
    col=f"ch{cv}"
    if col in trades.columns:
        trades[f"best_ch_{cv}"]=trades[col]>trades["pts45"]  # better than CH45

# Find best CH per trade
ch_cols=[f"ch{cv}" for cv in CH_VALS if f"ch{cv}" in trades.columns]
ch_data=trades[ch_cols]
trades["best_ch"]=ch_data.idxmax(axis=1).str.replace("ch","").astype(int)
trades["worst_ch"]=ch_data.idxmin(axis=1).str.replace("ch","").astype(int)

# Best CH distribution
bc_dist=trades["best_ch"].value_counts().sort_index()
print(f"  Best CH Distribution:")
for cv in sorted(CH_VALS):
    pct=(trades["best_ch"]==cv).mean()*100
    print(f"    CH{cv:>2d}: {(trades['best_ch']==cv).sum():5d} trades ({pct:4.1f}%)")

print(f"\n  Best CH by volatility regime:")
low_vol=trades[trades["atr14_pct"]<trades["atr14_pct"].median()]
high_vol=trades[trades["atr14_pct"]>=trades["atr14_pct"].median()]
for label,sub in [("Low Vol",low_vol),("High Vol",high_vol)]:
    bc=sub["best_ch"].value_counts()
    print(f"    {label}: best CH={bc.index[0] if len(bc)>0 else 'N/A'} ({bc.iloc[0]/len(sub)*100:.0f}%)")

print(f"\n  Best CH by month:")
for m in range(1,13):
    sub=trades[trades["mo"]==m]
    if len(sub)<10:continue
    bc=sub["best_ch"].value_counts()
    print(f"    {mon_names[m-1]:>4s}: best CH={bc.index[0] if len(bc)>0 else 'N/A'} ({bc.iloc[0]/len(sub)*100:.0f}%) {sub['pts45'].sum():>+8,.0f}pts")

# ═══════════════════════════════════════════════════════════
# 5. LOSS AUTOCORRELATION & STREAK ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 5: LOSS STREAK & AUTOCORRELATION ANALYSIS")
print("="*120)
trades_sorted=trades.sort_values(["sym","yr","mo"]).reset_index(drop=True)
trades_sorted["prev_win"]=trades_sorted["is_win"].shift(1)
trades_sorted["prev2_win"]=trades_sorted["is_win"].shift(2)
trades_sorted["prev3_win"]=trades_sorted["is_win"].shift(3)

print(f"  Win autocorrelation:")
for lag in [1,2,3]:
    col=f"prev{lag}_win"
    sub=trades_sorted.dropna(subset=[col])
    after_win=sub[sub[col]==True]["is_win"].mean()
    after_loss=sub[sub[col]==False]["is_win"].mean()
    print(f"    After {lag} win(s):  WR={after_win:.1%}")
    print(f"    After {lag} loss(es): WR={after_loss:.1%}")
    try:
        tbl=pd.crosstab(sub[col],sub["is_win"])
        chi2,p=stats.chi2_contingency(tbl)[:2]
        print(f"    Chi-sq p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")
    except:pass

# Streak analysis
streaks=[];cur_streak=0;cur_type=None
for _,r in trades_sorted.iterrows():
    if cur_type is None:
        cur_type=r["is_win"];cur_streak=1
    elif r["is_win"]==cur_type:cur_streak+=1
    else:
        streaks.append({"type":"win" if cur_type else "loss","length":cur_streak,
                        "net":trades_sorted.iloc[len(streaks):len(streaks)+cur_streak]["pts45"].sum() if len(streaks)+cur_streak<=len(trades_sorted) else 0})
        cur_type=r["is_win"];cur_streak=1
streaks.append({"type":"win" if cur_type else "loss","length":cur_streak,
                "net":0})

streak_df=pd.DataFrame(streaks)
print(f"\n  Loss streak frequency:")
for length in range(1,8):
    sub=streak_df[(streak_df["type"]=="loss")&(streak_df["length"]==length)]
    print(f"    {length} consecutive losses: {len(sub)} times (avg net={sub['net'].mean():>+.0f})")
print(f"  Max loss streak: {streak_df[streak_df['type']=='loss']['length'].max()}")

# ═══════════════════════════════════════════════════════════
# 6. YEAR ANALYSIS — What makes a year good or bad
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 6: YEAR-LEVEL ANALYSIS")
print("="*120)

yearly=[]
for yr in sorted(trades["yr"].unique()):
    sub=trades[trades["yr"]==yr]
    win=sub[sub["is_win"]]
    loss=sub[~sub["is_win"]]
    net=sub["pts45"].sum()
    wr=sub["is_win"].mean()
    aw=win["pts45"].mean() if len(win)>0 else 0
    al=abs(loss["pts45"].mean()) if len(loss)>0 else 0
    wl=aw/al if al>0 else 999
    avg_atr=sub["atr14_pct"].mean()
    avg_range=sub["range_ratio"].mean()
    n_trades=len(sub)
    best_mo=sub.groupby("mo")["pts45"].sum().max()
    worst_mo=sub.groupby("mo")["pts45"].sum().min()
    yearly.append({"yr":yr,"net":net,"wr":wr,"wl":wl,"n":n_trades,"avg_atr":avg_atr,
                   "avg_range":avg_range,"avg_win":aw,"avg_loss":al,"best_mo":best_mo,"worst_mo":worst_mo})

ydf=pd.DataFrame(yearly)
print(f"  {'Year':<6s} {'Net':>10s} {'WR':>6s} {'W/L':>6s} {'N':>5s} {'AvgATR%':>8s} {'AvgRange':>9s} {'AvgWin':>9s} {'AvgLoss':>9s} {'BestMo':>9s} {'WorstMo':>9s}")
print("  "+"-"*86)
for _,r in ydf.iterrows():
    print(f"  {r['yr']:<6d} {r['net']:>+9,.0f}  {r['wr']:.1%} {r['wl']:>4.1f}x {r['n']:>4d} {r['avg_atr']:>7.2f}% {r['avg_range']:>8.2f} {r['avg_win']:>+8,.0f} {r['avg_loss']:>+8,.0f} {r['best_mo']:>+8,.0f} {r['worst_mo']:>+8,.0f}")

# ═══════════════════════════════════════════════════════════
# 7. ML FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 7: ML FEATURE IMPORTANCE (predicting win/loss)")
print("="*120)

ml_features=["body","prev_body","body_ratio","candle_range","prev_range","range_ratio",
             "atr14","atr14_pct","ret_prev_1h","ret_5h","ret_20h",
             "close_vs_ema50","close_vs_ema200","n_consec_red",
             "entry_hour","day_of_week","range_vs_avg","vol_ratio",
             "month_sin","month_cos"]
ml_df=trades[ml_features+["is_win"]].dropna()
X=StandardScaler().fit_transform(ml_df[ml_features]);y=ml_df["is_win"]

rf=RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42,n_jobs=-1)
rf.fit(X,y)
fi=pd.DataFrame({"feature":ml_features,"importance":rf.feature_importances_}).sort_values("importance",ascending=False)

cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)
scores=cross_val_score(rf,X,y,cv=cv,scoring="roc_auc")
print(f"  RF AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"  RF Accuracy: {cross_val_score(rf,X,y,cv=cv,scoring='accuracy').mean():.4f}")
print(f"\n  Top 15 features by importance:")
print(f"  {'Feature':<25s} {'Importance':>10s}")
print("  "+"-"*35)
for _,r in fi.head(15).iterrows():
    print(f"  {r['feature']:<25s} {r['importance']:>10.4f}")

# GBM
gb=GradientBoostingClassifier(n_estimators=100,max_depth=4,random_state=42)
gb.fit(X,y)
fi_gb=pd.DataFrame({"feature":ml_features,"importance":gb.feature_importances_}).sort_values("importance",ascending=False)
print(f"\n  GBM Top 10:")
print(f"  {'Feature':<25s} {'Importance':>10s}")
for _,r in fi_gb.head(10).iterrows():
    print(f"  {r['feature']:<25s} {r['importance']:>10.4f}")

# ═══════════════════════════════════════════════════════════
# 8. QUANTITATIVE RULES — Best conditions for trading
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 8: QUANTITATIVE RULES — Best/Worst conditions to trade")
print("="*120)

# For each top feature, find optimal threshold
for feat in fi.head(10)["feature"].values:
    if feat not in trades.columns or feat in ["month_sin","month_cos","entry_hour","day_of_week"]:
        continue
    sub=trades[feat].dropna()
    if sub.nunique()<5:continue
    # Try splitting at median
    med=sub.median()
    high=trades[trades[feat]>med]
    low=trades[trades[feat]<=med]
    if len(high)<10 or len(low)<10:continue
    h_wr=high["is_win"].mean();l_wr=low["is_win"].mean()
    h_net=high["pts45"].sum();l_net=low["pts45"].sum()
    diff=abs(h_wr-l_wr)
    if diff>0.03:
        better="HIGH" if h_wr>l_wr else "LOW"
        print(f"  {feat:<25s}: when {better} -> WR={max(h_wr,l_wr):.0%} ({max(h_net,l_net):>+,.0f}pts) "
              f"vs {min(h_wr,l_wr):.0%} ({min(h_net,l_net):>+,.0f}pts)")

# ═══════════════════════════════════════════════════════════
# 9. EXIT TIMING ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SECTION 9: EXIT TIMING ANALYSIS")
print("="*120)
for cv in [25,35,45,55]:
    col=f"exit_bar_{cv}"
    if col in trades.columns:
        wins=trades[trades["is_win"]][col].mean()
        losses=trades[~trades["is_win"]][col].mean()
        print(f"  CH{cv:>2d}: Avg exit bar (wins)={wins:.1f} (losses)={losses:.1f}")

# ═══════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n"+"="*120)
print("SUMMARY: KEY FINDINGS")
print("="*120)

best_month=trades.groupby("mo")["pts45"].sum().idxmax()
worst_month=trades.groupby("mo")["pts45"].sum().idxmin()
best_yr=trades.groupby("yr")["pts45"].sum().idxmax()
worst_yr=trades.groupby("yr")["pts45"].sum().idxmin()

print(f"""
1. OVERALL:
   - Win Rate: {trades['is_win'].mean():.1%}
   - Best Month: {mon_names[best_month-1]} ({trades.groupby('mo')['pts45'].sum().max():>+,.0f} pts)
   - Worst Month: {mon_names[worst_month-1]} ({trades.groupby('mo')['pts45'].sum().min():>+,.0f} pts)
   - Best Year: {best_yr} ({trades.groupby('yr')['pts45'].sum().max():>+,.0f} pts)
   - Worst Year: {worst_yr} ({trades.groupby('yr')['pts45'].sum().min():>+,.0f} pts)

2. TOP FEATURES PREDICTING WINS (RF):
""")
for _,r in fi.head(5).iterrows():
    print(f"   {r['feature']:<25s} importance={r['importance']:.4f}")

print(f"""
3. KEY RULES:
""")
for _,r in fi.head(10).iterrows():
    feat=r["feature"]
    if feat not in trades.columns or feat in ["month_sin","month_cos"]:continue
    med=trades[feat].median()
    high=trades[trades[feat]>med]
    low=trades[trades[feat]<=med]
    if len(high)<10 or len(low)<10:continue
    better="HIGH" if high["is_win"].mean()>low["is_win"].mean() else "LOW"
    print(f"   {feat:<25s}: {better} side wins {max(high['is_win'].mean(),low['is_win'].mean()):.0%} vs {min(high['is_win'].mean(),low['is_win'].mean()):.0%}")

print(f"""
4. LOSS AUTOCORRELATION:
   After a loss: WR drops to {trades_sorted[trades_sorted['prev_win']==False]['is_win'].mean():.1%}
   After a win:  WR rises to {trades_sorted[trades_sorted['prev_win']==True]['is_win'].mean():.1%}
   Best skip strategy: Skip 2 (removes worst autocorrelation)

5. OPTIMAL CH VALUE:
   Overall best: CH{int(trades['best_ch'].mode().iloc[0])} (selected {((trades['best_ch']==int(trades['best_ch'].mode().iloc[0])).mean()*100):.0f}% of time)
   Best single CH: 55 ({int((trades['best_ch']==55).mean()*100):.0f}% optimal)
""")
