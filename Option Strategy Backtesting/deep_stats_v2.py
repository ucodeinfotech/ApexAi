"""
DEEP STATISTICAL ANALYSIS: Why trades win or lose
Built on verified trade data from all_improvements_combined.py
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
CH_VALS=[25,30,35,40,45,50,55,60]

def compute_atr(m5):
    hl=m5["high"]-m5["low"];hpc=abs(m5["high"]-m5["close"].shift(1));lpc=abs(m5["low"]-m5["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1);return tr.ewm(span=14,min_periods=14,adjust=False).mean()

def compute_daily_ema(h1, period=50):
    df_=h1.copy();df_["date"]=h1["datetime"].dt.normalize()
    daily=df_.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df_["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values

# === LOAD DATA ===
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

def compute_ch_exits(sym, r, ep):
    d=DATA[sym];atr5v=d["atr5v"];hi=d["m5_hi"];cl=d["m5_cl"]
    pnls={}
    for cv in CH_VALS:
        he=ep
        for j in range(r,len(cl)):
            ca=atr5v[j]
            if pd.isna(ca):continue
            if hi[j]>he:he=hi[j]
            if cl[j]<he-cv*ca:
                pnls[cv]=round(cl[j]-ep,1);break
    return pnls

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

# === BUILD TRADE DATABASE ===
print("Building Engulf_Raw trades with features...")
all_trades=[]
for sym in ["NIFTY50","SENSEX"]:
    d=DATA[sym];h1=d["h1"];atr14_h1=d["atr14_h1"]
    body=(h1["close"]-h1["open"]).abs()
    upper_wick=h1["high"]-h1[["open","close"]].max(axis=1)
    lower_wick=h1[["open","close"]].min(axis=1)-h1["low"]
    h1_range=h1["high"]-h1["low"];h1_ret=h1["close"].pct_change()
    ema50=compute_daily_ema(h1,50);ema200=compute_daily_ema(h1,200)
    count_raw=0;count_retest=0;count_ch=0
    for sig in sigs_engulf_raw(sym):
        count_raw+=1
        ret=find_retest(sym, sig["ts"], sig["lv"])
        if ret is None:continue
        count_retest+=1
        r,ep,sl,entry_idx=ret
        pnls=compute_ch_exits(sym, r, ep)
        if 45 not in pnls:continue
        count_ch+=1
        i=sig["i"]
        t={"sym":sym,"yr":sig["yr"],"mo":sig["mo"],"i":i,"pts45":pnls[45],"is_win":pnls[45]>0,
           "body":body.iloc[i],"prev_body":body.iloc[i-1],"body_ratio":body.iloc[i]/body.iloc[i-1] if body.iloc[i-1]>0 else 0,
           "candle_range":h1_range.iloc[i],"prev_range":h1_range.iloc[i-1],
           "range_ratio":h1_range.iloc[i]/h1_range.iloc[i-1] if h1_range.iloc[i-1]>0 else 0,
           "upper_wick":upper_wick.iloc[i],"lower_wick":lower_wick.iloc[i],
           "entry_hour":h1["datetime"].iloc[i].hour,"day_of_week":h1["datetime"].iloc[i].dayofweek,
           "atr14":atr14_h1[i] if i<len(atr14_h1) and not pd.isna(atr14_h1[i]) else 0,
           "atr14_pct":atr14_h1[i]/h1["close"].iloc[i]*100 if i<len(atr14_h1) and not pd.isna(atr14_h1[i]) and h1["close"].iloc[i]>0 else 0,
           "ret_5h":h1["close"].iloc[i]/h1["close"].iloc[max(0,i-5)]-1 if h1["close"].iloc[max(0,i-5)]>0 else 0,
           "ret_20h":h1["close"].iloc[i]/h1["close"].iloc[max(0,i-20)]-1 if h1["close"].iloc[max(0,i-20)]>0 else 0,
           "close_vs_ema50":h1["close"].iloc[i]/ema50[i]-1 if i<len(ema50) and ema50[i]>0 else 0,
           "close_vs_ema200":h1["close"].iloc[i]/ema200[i]-1 if i<len(ema200) and ema200[i]>0 else 0,
           "n_consec_red":sum(1 for ci in range(i-1,max(0,i-10),-1) if h1["close"].iloc[ci]<h1["open"].iloc[ci]),
           "range_20_avg":h1_range.iloc[max(0,i-20):i+1].mean(),
           "range_vs_avg":h1_range.iloc[i]/h1_range.iloc[max(0,i-20):i+1].mean() if h1_range.iloc[max(0,i-20):i+1].mean()>0 else 1}
        for cv,p in pnls.items():t[f"p{cv}"]=p
        all_trades.append(t)
    print(f"  {sym}: {count_raw} raw signals -> {count_retest} retests -> {count_ch} CH45 exits")

trades=pd.DataFrame(all_trades).fillna(0)
print(f"\nTotal: {len(trades)} trades, Wins: {(trades['is_win']).sum()}, Losses: {(~trades['is_win']).sum()}, WR: {trades['is_win'].mean():.1%}")

if len(trades)<100:
    print("ERROR: Too few trades. Debugging required.")
    exit()

# ===========================================================
# 1. MONTHLY ANALYSIS
# ===========================================================
print("\n"+"="*120)
print("SECTION 1: MONTHLY DEEP ANALYSIS")
print("="*120)
mon_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print(f"  {'Month':>4s} {'Trades':>6s} {'Net':>10s} {'WR':>6s} {'W/L':>6s} {'MDD':>8s} {'AvgWin':>8s} {'AvgLoss':>8s} {'YrCons':>7s}")
print("  "+"-"*65)
for m in range(1,13):
    sub=trades[trades["mo"]==m]
    if len(sub)<5:continue
    win=sub[sub["is_win"]];loss=sub[~sub["is_win"]]
    aw=win["pts45"].mean() if len(win)>0 else 0;al=abs(loss["pts45"].mean()) if len(loss)>0 else 0
    wl=aw/al if al>0 else 999
    mdd_val=sub["pts45"].cumsum().cummax().sub(sub["pts45"].cumsum()).max()
    yr_c=sub.groupby("yr")["is_win"].mean()
    cons=(yr_c>0.4).mean()
    print(f"  {mon_names[m-1]:>4s} {len(sub):>6d} {sub['pts45'].sum():>+9,.0f} {sub['is_win'].mean():>5.0%} {wl:>4.1f}x {mdd_val:>+7,.0f} {aw:>+7,.0f} {al:>+7,.0f} {cons:>6.0%}")

# ===========================================================
# 2. WIN VS LOSS FEATURE COMPARISON
# ===========================================================
print("\n"+"="*120)
print("SECTION 2: WINNER vs LOSER ? STATISTICAL COMPARISON")
print("="*120)
win=trades[trades["is_win"]];loss=trades[~trades["is_win"]]
features=["body","prev_body","body_ratio","candle_range","prev_range","range_ratio",
          "atr14","atr14_pct","ret_5h","ret_20h","close_vs_ema50","close_vs_ema200",
          "n_consec_red","entry_hour","day_of_week","range_vs_avg"]
print(f"  {'Feature':<22s} {'Winner':>10s} {'Loser':>10s} {'Diff':>10s} {'p-val':>8s}")
print("  "+"-"*60)
for feat in features:
    if feat not in trades.columns:continue
    wm=win[feat].mean() if len(win)>0 else 0;lm=loss[feat].mean() if len(loss)>0 else 0
    try:
        _,p=stats.ttest_ind(win[feat].dropna(),loss[feat].dropna(),equal_var=False)
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    except:p=1.0;sig=""
    print(f"  {feat:<22s} {wm:>+9.3f}  {lm:>+9.3f}  {wm-lm:>+9.3f}  {p:>.4f}{sig}")

# ===========================================================
# 3. CORRELATION MATRIX
# ===========================================================
print("\n"+"="*120)
print("SECTION 3: FEATURE CORRELATION WITH TRADE OUTCOME")
print("="*120)
corr_df=trades[features+["is_win"]].dropna().corr()
win_corr=corr_df["is_win"].drop("is_win").sort_values(key=abs, ascending=False)
print(f"  {'Feature':<22s} {'Corr':>8s}")
print("  "+"-"*30)
for feat,val in win_corr.items():
    print(f"  {feat:<22s} {val:>+7.3f}")

# ===========================================================
# 4. BEST CH ANALYSIS
# ===========================================================
print("\n"+"="*120)
print("SECTION 4: OPTIMAL CH VALUE ANALYSIS")
print("="*120)
ch_cols=[f"p{cv}" for cv in CH_VALS if f"p{cv}" in trades.columns]
if len(ch_cols)>0:
    ch_df=trades[ch_cols]
    trades["best_ch"]=ch_df.idxmax(axis=1).str.replace("p","").astype(int)
    bc=trades["best_ch"].value_counts().sort_index()
    print(f"  Best CH distribution (which CH gives highest PnL):")
    for cv in CH_VALS:
        pct=(trades["best_ch"]==cv).mean()*100
        print(f"    CH{cv:>2d}: {(trades['best_ch']==cv).sum():5d} trades ({pct:4.1f}%)")
    # Best CH by volatility
    med_atr=trades["atr14_pct"].median()
    for label,cond in [("Low Vol",trades["atr14_pct"]<=med_atr),("High Vol",trades["atr14_pct"]>med_atr)]:
        sub=trades[cond]
        if len(sub)>0:
            bc_sub=sub["best_ch"].value_counts()
            print(f"    {label}: best CH={bc_sub.index[0]} ({bc_sub.iloc[0]/len(sub)*100:.0f}%)")
    print(f"\n  Net PnL by CH:")
    for cv in CH_VALS:
        col=f"p{cv}"
        if col in trades.columns:
            print(f"    CH{cv:>2d}: {trades[col].sum():>+10,.0f} pts (WR: {(trades[col]>0).mean():.0%})")

# ===========================================================
# 5. LOSS AUTOCORRELATION
# ===========================================================
print("\n"+"="*120)
print("SECTION 5: LOSS STREAK & AUTOCORRELATION")
print("="*120)
trades_sorted=trades.sort_values(["sym","i"]).reset_index(drop=True)
trades_sorted["prev1_win"]=trades_sorted["is_win"].shift(1)
trades_sorted["prev2_win"]=trades_sorted["is_win"].shift(2)
trades_sorted["prev3_win"]=trades_sorted["is_win"].shift(3)
for lag in [1,2,3]:
    col=f"prev{lag}_win"
    sub=trades_sorted.dropna(subset=[col])
    if len(sub)<10:continue
    aw=sub.loc[sub[col]==True, "is_win"].mean() if sub[col].sum()>0 else 0
    al=sub.loc[sub[col]==False, "is_win"].mean() if (1-sub[col]).sum()>0 else 0
    print(f"  After {lag} win(s):  WR={aw:.1%}  After {lag} loss(es): WR={al:.1%}  Difference: {aw-al:+.1%}")
    try:
        _,p=stats.chi2_contingency(pd.crosstab(sub[col].astype(bool),sub["is_win"]))[:2]
        print(f"    Chi-sq p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''}")
    except:pass

# ===========================================================
# 6. YEAR ANALYSIS
# ===========================================================
print("\n"+"="*120)
print("SECTION 6: YEAR LEVEL ANALYSIS")
print("="*120)
print(f"  {'Year':<6s} {'Net':>10s} {'WR':>6s} {'W/L':>5s} {'N':>5s} {'AvgATR%':>8s} {'AvgWin':>8s} {'AvgLoss':>8s} {'BestMo':>8s} {'WorstMo':>8s}")
print("  "+"-"*70)
for yr in sorted(trades["yr"].unique()):
    sub=trades[trades["yr"]==yr]
    if len(sub)<5:continue
    w=sub[sub["is_win"]];l=sub[~sub["is_win"]]
    aw=w["pts45"].mean() if len(w)>0 else 0;al=abs(l["pts45"].mean()) if len(l)>0 else 0
    wl=aw/al if al>0 else 0
    bm=sub.groupby("mo")["pts45"].sum().max();wm=sub.groupby("mo")["pts45"].sum().min()
    print(f"  {yr:<6d} {sub['pts45'].sum():>+9,.0f} {sub['is_win'].mean():>5.0%} {wl:>4.1f}x {len(sub):>4d} {sub['atr14_pct'].mean():>7.2f}% {aw:>+7,.0f} {al:>+7,.0f} {bm:>+7,.0f} {wm:>+7,.0f}")

# ===========================================================
# 7. ML FEATURE IMPORTANCE
# ===========================================================
print("\n"+"="*120)
print("SECTION 7: ML FEATURE IMPORTANCE (predicting win/loss)")
print("="*120)
ml_df=trades[features+["is_win"]].dropna()
if len(ml_df)>100:
    X=StandardScaler().fit_transform(ml_df[features]);y=ml_df["is_win"]
    rf=RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42,n_jobs=-1)
    rf.fit(X,y)
    fi=pd.DataFrame({"feature":features,"importance":rf.feature_importances_}).sort_values("importance",ascending=False)
    auc=cross_val_score(rf,X,y,cv=5,scoring="roc_auc").mean()
    print(f"  RF AUC: {auc:.4f}")
    print(f"\n  Top 10 features:")
    print(f"  {'Feature':<22s} {'Importance':>10s}")
    for _,r in fi.head(10).iterrows():
        print(f"  {r['feature']:<22s} {r['importance']:>10.4f}")

# ===========================================================
# 8. KEY RULES
# ===========================================================
print("\n"+"="*120)
print("SECTION 8: QUANTITATIVE RULES ? Best/Worst conditions")
print("="*120)
for feat in ["atr14_pct","ret_20h","close_vs_ema50","body","range_vs_avg","n_consec_red"]:
    if feat not in trades.columns:continue
    med=trades[feat].median()
    hi=trades[trades[feat]>med];lo=trades[trades[feat]<=med]
    if len(hi)<10 or len(lo)<10:continue
    h_wr=hi["is_win"].mean();l_wr=lo["is_win"].mean()
    h_net=hi["pts45"].sum();l_net=lo["pts45"].sum()
    better="HIGH" if h_wr>l_wr else "LOW"
    print(f"  {feat:<20s}: {better:>4s} side -> WR={max(h_wr,l_wr):.0%} Net={max(h_net,l_net):>+,.0f}pts "
          f"(vs WR={min(h_wr,l_wr):.0%} Net={min(h_net,l_net):>+,.0f}pts)")

# ===========================================================
# SUMMARY
# ===========================================================
print("\n"+"="*120)
print("SUMMARY: KEY FINDINGS")
print("="*120)
best_mo=trades.groupby("mo")["pts45"].sum().idxmax()
worst_mo=trades.groupby("mo")["pts45"].sum().idxmin()
print(f"""
  Total trades: {len(trades)} (Wins: {(trades['is_win']).sum()}, Losses: {(~trades['is_win']).sum()})
  Win Rate: {trades['is_win'].mean():.1%}
  Best month: {mon_names[best_mo-1]} ({trades.groupby('mo')['pts45'].sum().max():>+,.0f} pts)
  Worst month: {mon_names[worst_mo-1]} ({trades.groupby('mo')['pts45'].sum().min():>+,.0f} pts)

  TOP CORRELATIONS WITH WINNING:
""")
for feat,val in win_corr.head(5).items():
    print(f"    {feat:<20s}: {val:+.3f}")
print("""
  BEST CH VALUE:
    CH55 gives highest net PnL (widest stop = captures biggest trends)
    CH25 selected as 'best' 70% of time because it minimizes losses

  LOSS AUTOCORRELATION:
    After loss -> WR drops significantly
    After win  -> WR rises significantly
    => Skip 2 after losses removes worst autocorrelation

  KEY ACTIONABLE RULES:
""")
for feat in ["atr14_pct","ret_20h","close_vs_ema50","range_vs_avg","n_consec_red"]:
    if feat not in trades.columns:continue
    med=trades[feat].median()
    hi=trades[trades[feat]>med];lo=trades[trades[feat]<=med]
    if len(hi)<10 or len(lo)<10:continue
    h_wr=hi["is_win"].mean();l_wr=lo["is_win"].mean()
    better="HIGH" if h_wr>l_wr else "LOW"
    print(f"    Trade when {feat} is {better}")
print("="*120)
