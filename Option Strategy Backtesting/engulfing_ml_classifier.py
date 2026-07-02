"""
ML Classifier — Predict winning trades before entry.
Precomputes all indicators once for speed, extracts 40+ features per signal.
"""
import pandas as pd, numpy as np, os, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 0.50

def compute_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift(1)).abs(), (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

# ── Precompute all indicators on 1H dataframe ──
def precompute_indicators(h1):
    """Add indicator columns directly to h1."""
    h1 = h1.copy()
    # RSI
    delta=h1["close"].diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); h1["rsi14"]=100-(100/(1+rs))
    h1["rsi30"] = h1["rsi14"]  # will use 30-period below
    # RSI30
    delta30=h1["close"].diff(); g30=delta30.clip(lower=0).rolling(30).mean(); l30=(-delta30.clip(upper=0)).rolling(30).mean()
    rs30=g30/l30.replace(0,np.nan); h1["rsi30"]=100-(100/(1+rs30))
    # MACD
    ef=h1["close"].ewm(span=12).mean(); es=h1["close"].ewm(span=26).mean()
    h1["macd"]=ef-es; h1["macd_sig"]=h1["macd"].ewm(span=9).mean(); h1["macd_hist"]=h1["macd"]-h1["macd_sig"]
    # Bollinger
    bb_mid=h1["close"].rolling(20).mean(); bb_std=h1["close"].rolling(20).std()
    h1["bb_upper"]=bb_mid+2*bb_std; h1["bb_lower"]=bb_mid-2*bb_std; h1["bb_mid"]=bb_mid; h1["bbw"]=h1["bb_upper"]-h1["bb_lower"]
    h1["bb_pctB"]=np.where(h1["bbw"]>0, (h1["close"]-h1["bb_lower"])/h1["bbw"], 0.5)
    # Stochastic
    l14=h1["low"].rolling(14).min(); h14=h1["high"].rolling(14).max()
    h1["stoch_k"]=100*(h1["close"]-l14)/(h14-l14).replace(0,np.nan); h1["stoch_d"]=h1["stoch_k"].rolling(3).mean()
    # ADX
    atr14=compute_atr(h1,14)
    pdm=h1["high"].diff().clip(lower=0); mdm=h1["low"].diff().clip(upper=0).abs()
    h1["pdi"]=100*(pdm.rolling(14).mean()/atr14.replace(0,np.nan)); h1["mdi"]=100*(mdm.rolling(14).mean()/atr14.replace(0,np.nan))
    dx=100*((h1["pdi"]-h1["mdi"]).abs()/(h1["pdi"]+h1["mdi"]).replace(0,np.nan)); h1["adx"]=dx.rolling(14).mean()
    # ATR
    h1["atr"]=atr14; h1["atr_ma20"]=atr14.rolling(20).mean()
    # EMAs
    h1["ema50"]=h1["close"].ewm(span=50).mean(); h1["ema200"]=h1["close"].ewm(span=200).mean()
    h1["sma20"]=h1["close"].rolling(20).mean(); h1["sma50"]=h1["close"].rolling(50).mean()
    # Slopes
    h1["ema50_slope"]=h1["ema50"].diff(5)/5; h1["ema200_slope"]=h1["ema200"].diff(5)/5
    # 20-period high/low
    h1["hi20"]=h1["high"].rolling(20).max(); h1["lo20"]=h1["low"].rolling(20).min()
    return h1

def extract_features(h1, i):
    """Read features from precomputed h1 at index i."""
    f = {}
    f["body_c"] = abs(h1["close"].iloc[i] - h1["open"].iloc[i])
    f["body_p"] = abs(h1["close"].iloc[i-1] - h1["open"].iloc[i-1])
    f["body_ratio"] = f["body_c"] / f["body_p"] if f["body_p"]>0 else 0
    f["range_c"] = h1["high"].iloc[i] - h1["low"].iloc[i]
    f["range_p"] = h1["high"].iloc[i-1] - h1["low"].iloc[i-1]
    f["range_ratio"] = f["range_c"] / f["range_p"] if f["range_p"]>0 else 0
    f["gap_pct"] = (h1["open"].iloc[i] / h1["close"].iloc[i-1] - 1)*100
    f["gap_down"] = 1 if h1["open"].iloc[i] < h1["close"].iloc[i-1] else 0
    f["upper_wick"] = h1["high"].iloc[i] - max(h1["open"].iloc[i], h1["close"].iloc[i])
    f["lower_wick"] = min(h1["open"].iloc[i], h1["close"].iloc[i]) - h1["low"].iloc[i]
    f["close_pct"] = (h1["close"].iloc[i] / h1["open"].iloc[i] - 1)*100
    # Consecutive bearish
    cb=0
    for k in range(i-1, max(0,i-5), -1):
        if h1["close"].iloc[k] < h1["open"].iloc[k]: cb+=1
        else: break
    f["consec_bearish"] = cb
    # RSI
    f["rsi"] = h1["rsi14"].iloc[i] if not pd.isna(h1["rsi14"].iloc[i]) else 50
    f["rsi_p"] = h1["rsi14"].iloc[i-1] if not pd.isna(h1["rsi14"].iloc[i-1]) else 50
    f["rsi_chg"] = f["rsi"] - f["rsi_p"]
    f["rsi_os"] = 1 if f["rsi"] < 30 else 0
    f["rsi_b40"] = 1 if f["rsi"] < 40 else 0
    f["rsi_ma5"] = h1["rsi14"].iloc[max(0,i-4):i+1].mean() if i>=4 else f["rsi"]
    f["rsi30"] = h1["rsi30"].iloc[i] if not pd.isna(h1["rsi30"].iloc[i]) else 50
    # MACD
    f["macd"] = h1["macd"].iloc[i] if not pd.isna(h1["macd"].iloc[i]) else 0
    f["macds"] = h1["macd_sig"].iloc[i] if not pd.isna(h1["macd_sig"].iloc[i]) else 0
    f["mhist"] = h1["macd_hist"].iloc[i] if not pd.isna(h1["macd_hist"].iloc[i]) else 0
    f["mhist_p"] = h1["macd_hist"].iloc[i-1] if not pd.isna(h1["macd_hist"].iloc[i-1]) else 0
    f["mhist_inc"] = 1 if f["mhist"] > f["mhist_p"] else 0
    f["macd_abv"] = 1 if f["macd"] > f["macds"] else 0
    f["macd_b0"] = 1 if f["macd"] < 0 else 0
    # BB
    f["bb_pctB"] = h1["bb_pctB"].iloc[i] if not pd.isna(h1["bb_pctB"].iloc[i]) else 0.5
    f["bbw"] = h1["bbw"].iloc[i] / h1["bb_mid"].iloc[i]*100 if not pd.isna(h1["bb_mid"].iloc[i]) and h1["bb_mid"].iloc[i]>0 else 0
    f["bb_near_l"] = 1 if f["bb_pctB"] < 0.25 else 0
    f["bb_near_u"] = 1 if f["bb_pctB"] > 0.75 else 0
    # Stoch
    f["sk"] = h1["stoch_k"].iloc[i] if not pd.isna(h1["stoch_k"].iloc[i]) else 50
    f["sd"] = h1["stoch_d"].iloc[i] if not pd.isna(h1["stoch_d"].iloc[i]) else 50
    f["sk_os"] = 1 if f["sk"] < 20 else 0
    f["sk_b30"] = 1 if f["sk"] < 30 else 0
    # ADX
    f["adx"] = h1["adx"].iloc[i] if not pd.isna(h1["adx"].iloc[i]) else 20
    f["adx_s"] = 1 if f["adx"] > 25 else 0
    f["adx_t"] = 1 if f["adx"] > 20 else 0
    f["pdi"] = h1["pdi"].iloc[i] if not pd.isna(h1["pdi"].iloc[i]) else 20
    f["mdi"] = h1["mdi"].iloc[i] if not pd.isna(h1["mdi"].iloc[i]) else 20
    # ATR
    f["atr"] = h1["atr"].iloc[i] if not pd.isna(h1["atr"].iloc[i]) else 0
    f["atr_r"] = f["atr"] / h1["atr_ma20"].iloc[i] if not pd.isna(h1["atr_ma20"].iloc[i]) and h1["atr_ma20"].iloc[i]>0 else 1
    f["atr_h"] = 1 if not pd.isna(h1["atr_ma20"].iloc[i]) and f["atr"] > h1["atr_ma20"].iloc[i] else 0
    # EMAs
    f["pv_ema50"] = (h1["close"].iloc[i] / h1["ema50"].iloc[i] - 1)*100 if not pd.isna(h1["ema50"].iloc[i]) else 0
    f["pv_ema200"] = (h1["close"].iloc[i] / h1["ema200"].iloc[i] - 1)*100 if not pd.isna(h1["ema200"].iloc[i]) else 0
    f["pv_sma20"] = (h1["close"].iloc[i] / h1["sma20"].iloc[i] - 1)*100 if not pd.isna(h1["sma20"].iloc[i]) else 0
    f["ema50_abv200"] = 1 if h1["ema50"].iloc[i] > h1["ema200"].iloc[i] else 0
    f["p_abv50"] = 1 if h1["close"].iloc[i] > h1["ema50"].iloc[i] else 0
    f["p_abv200"] = 1 if h1["close"].iloc[i] > h1["ema200"].iloc[i] else 0
    f["e50_slp"] = h1["ema50_slope"].iloc[i] if not pd.isna(h1["ema50_slope"].iloc[i]) else 0
    # Extremes
    f["pv_20hi"] = (h1["close"].iloc[i] / h1["hi20"].iloc[i] - 1)*100 if not pd.isna(h1["hi20"].iloc[i]) else 0
    f["pv_20lo"] = (h1["close"].iloc[i] / h1["lo20"].iloc[i] - 1)*100 if not pd.isna(h1["lo20"].iloc[i]) else 0
    f["n20lo"] = 1 if abs(f["pv_20lo"]) < 2 else 0
    # Time
    dt = h1["datetime"].iloc[i]
    f["hour"] = dt.hour; f["dow"] = dt.dayofweek; f["mon"] = dt.month
    f["mon"] = 1 if dt.dayofweek==0 else 0
    f["fri"] = 1 if dt.dayofweek==4 else 0
    return f

def get_trade_outcome(sig, h1, m5):
    """Execute trade, return (points, hold_hours) or None."""
    tc=m5["datetime"].dt.time
    atr5=compute_atr(m5,14)
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    tu=int(pd.to_datetime(sig["trigger_time"]).timestamp()); lv=sig["level"]
    idx=np.searchsorted(du, tu, side="right")
    if idx>=len(m5): return None
    broke=idx
    while broke<len(m5) and cl[broke]<=lv: broke+=1
    if broke>=len(m5): return None
    retest=broke+1
    while retest<len(m5):
        if lo[retest]<lv and cl[retest]>lv and tc.iloc[retest]<CUTOFF_TIME: break
        retest+=1
    if retest>=len(m5): return None
    entry=cl[retest]; sl=lo[retest]
    if entry-sl<=0 or m5["datetime"].iloc[retest].hour==9: return None
    highest=entry
    for j in range(retest+1,len(m5)):
        ca=atr5.iloc[j]
        if pd.isna(ca): continue
        if hi[j]>highest: highest=hi[j]
        if cl[j]<highest-15*ca:
            return (cl[j]-entry, (m5["datetime"].iloc[j]-m5["datetime"].iloc[retest]).total_seconds()/3600)
    return None

print("="*100)
print("ML CLASSIFIER — PREDICT WINNING TRADES")
print("="*100)

for sym in ["NIFTY50","SENSEX"]:
    print(f"\n{'='*60}")
    print(f"  {sym}")
    print(f"{'='*60}")
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)

    # Precompute indicators once
    h1=precompute_indicators(h1)

    # Detect signals
    body=(h1["close"]-h1["open"]).abs()
    is_red=h1["close"]<h1["open"]; is_green=h1["close"]>h1["open"]
    sigs=[]
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*MIN_BODY_PCT: continue
        sigs.append({"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i})
    print(f"  Signals: {len(sigs)}")

    # Extract features and outcomes
    feat_list=[]; outcomes=[]
    fsigs=sigs[:]  # take all
    for s in fsigs:
        feat_list.append(extract_features(h1, s["idx"]))
        out=get_trade_outcome(s, h1, m5)
        outcomes.append(out[0] if out is not None else 0)

    df=pd.DataFrame(feat_list)
    df["win"]=np.array([o>0 for o in outcomes], dtype=int)
    df["points"]=outcomes
    print(f"  Trades: {len(df)}, WR: {df['win'].mean()*100:.1f}%")

    # Filter signals that actually executed
    mask=np.array([o is not None for o in outcomes])
    df=df[mask].reset_index(drop=True)
    print(f"  Executed: {len(df)}, WR: {df['win'].mean()*100:.1f}%")

    if len(df)<100:
        print("  Too few trades, skipping")
        continue

    # Walk-forward 5-fold
    n=len(df); fs=n//5
    fcols=[c for c in df.columns if c not in ["win","points"]]
    res=[]
    for fold in range(5):
        vs=fold*fs; ve=(fold+1)*fs if fold<4 else n
        ti=list(range(0,vs))+list(range(ve,n)); vi=list(range(vs,ve))
        X_tr=df[fcols].iloc[ti].fillna(0).values; y_tr=df["win"].iloc[ti].values
        X_v=df[fcols].iloc[vi].fillna(0).values; y_v=df["win"].iloc[vi].values
        scaler=StandardScaler()
        X_tr_s=scaler.fit_transform(X_tr); X_v_s=scaler.transform(X_v)
        rf=RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr); y_p=rf.predict(X_v_s)
        acc=accuracy_score(y_v,y_p); prec=precision_score(y_v,y_p,zero_division=0); rec=recall_score(y_v,y_p,zero_division=0)
        f1=f1_score(y_v,y_p,zero_division=0)
        # What if we only take model-selected trades?
        dv=df.iloc[vi].copy(); dv["pred"]=y_p
        wt=dv[dv["pred"]==1]
        wp=wt["points"].sum() if len(wt)>0 else 0
        ap=dv["points"].sum()
        res.append({"fold":fold+1,"acc":acc,"prec":prec,"rec":rec,"f1":f1,"wp":wp,"ap":ap,
                     "wn":len(wt),"an":len(dv),"wwr":wt["win"].mean()*100 if len(wt)>0 else 0})
        print(f"    F{fold+1}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | "
              f"ML: {len(wt)}t {wp:.0f}pts WR={wt['win'].mean()*100:.0f}% | All: {len(dv)}t {ap:.0f}pts")

    rd=pd.DataFrame(res)
    print(f"\n  MEAN: Acc={rd['acc'].mean():.3f} Prec={rd['prec'].mean():.3f} Rec={rd['rec'].mean():.3f} F1={rd['f1'].mean():.3f}")
    print(f"  Total ML: {rd['wn'].sum()}t {rd['wp'].sum():.0f}pts  All: {rd['an'].sum()}t {rd['ap'].sum():.0f}pts")
    print(f"  Improvement: {((rd['wp'].sum()/rd['ap'].sum())-1)*100 if rd['ap'].sum()!=0 else 0:+.1f}%")

    # Feature importance
    X_all=StandardScaler().fit_transform(df[fcols].fillna(0))
    rf_all=RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_all.fit(X_all, df["win"])
    imp=pd.DataFrame({"f":fcols,"imp":rf_all.feature_importances_}).sort_values("imp",ascending=False)
    print(f"\n  Top 10 features:")
    for _,r_ in imp.head(10).iterrows():
        print(f"    {r_['f']:15s}  {r_['imp']:.4f}")

print(f"\n{'='*60}")
print(f"PORTFOLIO TEST (NIFTY+SENSEX, 60/40 split)")
print(f"{'='*60}")

all_dfs=[]
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True); m5=m5.sort_values("datetime").reset_index(drop=True)
    h1=precompute_indicators(h1)
    body=(h1["close"]-h1["open"]).abs()
    is_red=h1["close"]<h1["open"]; is_green=h1["close"]>h1["open"]
    lot=50 if "NIFTY" in sym else 10
    for i in range(1,len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*MIN_BODY_PCT: continue
        sig={"trigger_time":h1["datetime"].iloc[i],"level":h1["high"].iloc[i],"idx":i}
        feat=extract_features(h1,i)
        out=get_trade_outcome(sig,h1,m5)
        if out is not None:
            feat["points"]=out[0]; feat["win"]=1 if out[0]>0 else 0
            feat["sym"]=sym; feat["lot"]=lot; feat["hh"]=out[1]
            for c in fcols:
                if c not in feat: feat[c]=0
            all_dfs.append(feat)

pdf=pd.DataFrame(all_dfs)
print(f"  Total: {len(pdf)}t, WR: {pdf['win'].mean()*100:.1f}%")

n=len(pdf); sp=int(n*0.6)
tr=pdf.iloc[:sp]; te=pdf.iloc[sp:]
X_tr=tr[[c for c in fcols if c in tr.columns]].fillna(0).values
y_tr=tr["win"].values
X_te=te[[c for c in fcols if c in te.columns]].fillna(0).values
y_te=te["win"].values
scl=StandardScaler()
X_tr_s=scl.fit_transform(X_tr); X_te_s=scl.transform(X_te)
rf=RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr); y_p=rf.predict(X_te_s)
te=te.copy(); te["pred"]=y_p

ml_t=te[te["pred"]==1].copy()
ml_t["pnl_rs"]=ml_t["points"]*ml_t["lot"]-20
al_t=te.copy()
al_t["pnl_rs"]=al_t["points"]*al_t["lot"]-20

def lf(df):
    lc=0; k=np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if lc>=2: k[i]=False; lc=0; continue
        if df["pnl_rs"].iloc[i]<=0: lc+=1
        else: lc=0
    return df[k].reset_index(drop=True)

ml_f=lf(ml_t) if len(ml_t)>0 else ml_t
al_f=lf(al_t)
mn=ml_f["pnl_rs"].sum() if len(ml_f)>0 else 0
an_=al_f["pnl_rs"].sum()
print(f"  ML-selected:   {len(ml_f):4d}t, Rs{mn:>+9,.0f}, WR={ml_f['pnl_rs'].gt(0).mean()*100:.1f}%")
print(f"  All trades:    {len(al_f):4d}t, Rs{an_:>+9,.0f}, WR={al_f['pnl_rs'].gt(0).mean()*100:.1f}%")
print(f"  Improvement:   {((mn/an_)-1)*100 if an_!=0 else 0:+.1f}%")
print(f"  Accuracy:      {accuracy_score(y_te,y_p):.3f}")
print(f"  Precision:     {precision_score(y_te,y_p,zero_division=0):.3f}")

out_dir=os.path.join(BASE,"backtest_results","ml_classifier")
os.makedirs(out_dir,exist_ok=True)
print(f"\nResults: {out_dir}")
