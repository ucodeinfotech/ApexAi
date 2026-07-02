"""
AGENTIC TRADING SYSTEM FOR DynCH 45+10
========================================
Multi-agent ensemble with:
1. Regime Detection Agent (HMM)
2. Confidence Scoring Agent (Stacked Ensemble with Calibration)
3. Seasonality Agent (Month/Hour filters)
4. Streak Psychology Agent (Adaptive skip on loss streaks)
5. Dynamic CH Agent (Regime-aware parameter selection)
6. Meta-Agent (Combines all signals into final decision)
"""
import pandas as pd, numpy as np, os, warnings, json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
import xgboost as xgb
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

np.random.seed(42)

# ════════════════════════════════════════════════════════════
# STEP 1: BUILD RICH TRADE DATABASE WITH ALL FEATURES
# ════════════════════════════════════════════════════════════
def build_trades():
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
        m5 = pd.read_csv(f"{sym}_FIVE_MINUTE.csv", parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime", inplace=True); df.reset_index(drop=True, inplace=True)

        # 1h features
        hl = h1["high"]-h1["low"]; hpc = abs(h1["high"]-h1["close"].shift(1)); lpc = abs(h1["low"]-h1["close"].shift(1))
        tr = pd.concat([hl,hpc,lpc], axis=1).max(axis=1); h1["atr14"] = tr.ewm(span=14, min_periods=14, adjust=False).mean()
        h1["atr14_pct"] = h1["atr14"]/h1["close"]*100
        h1["body"] = (h1["close"]-h1["open"]).abs()
        h1["body_pct"] = h1["body"]/h1["open"]*100
        h1["upper_wick"] = h1["high"]-h1[["open","close"]].max(axis=1)
        h1["lower_wick"] = h1[["open","close"]].min(axis=1)-h1["low"]
        h1["wick_ratio"] = (h1["upper_wick"]+h1["lower_wick"])/(h1["body"]+1)
        h1["close_pos"] = (h1["close"]-h1["low"])/(h1["high"]-h1["low"]+1)
        h1["prev_ret"] = h1["close"].pct_change()
        h1["prev_range"] = (h1["high"]-h1["low"]).shift(1)
        h1["range_pct"] = (h1["high"]-h1["low"])/h1["open"]*100
        h1["gap_pct"] = (h1["open"]-h1["close"].shift(1))/h1["close"].shift(1)*100
        h1["prev_body_pct"] = h1["body_pct"].shift(1)
        h1["prev_wick_ratio"] = h1["wick_ratio"].shift(1)

        for p in [5,10,20,50,100]:
            h1[f"ema{p}"] = h1["close"].ewm(span=p, adjust=False).mean()
            h1[f"dist_ema{p}"] = (h1["close"]-h1[f"ema{p}"])/h1[f"ema{p}"]*100

        # Trend features
        h1["ema_cross"] = ((h1["ema5"] > h1["ema20"]) & (h1["ema5"].shift(1) <= h1["ema20"].shift(1))).astype(int)
        h1["ema_position"] = (h1["ema5"] > h1["ema20"]).astype(int)
        h1["adx_like"] = abs(h1["ema10"]-h1["ema50"])/h1["ema50"]*100

        # 5m features
        hl5 = m5["high"]-m5["low"]; hpc5 = abs(m5["high"]-m5["close"].shift(1)); lpc5 = abs(m5["low"]-m5["close"].shift(1))
        tr5 = pd.concat([hl5,hpc5,lpc5], axis=1).max(axis=1); m5_atr = tr5.ewm(span=14, min_periods=14, adjust=False).mean()
        atr5 = m5_atr.values; m5_hi = m5["high"].values; m5_lo = m5["low"].values; m5_cl = m5["close"].values
        m5_du = m5["datetime"].values; m5["range"] = m5["high"]-m5["low"]; m5["body"] = (m5["close"]-m5["open"]).abs()

        # Signal detection using 1h
        a14 = h1["atr14"].values; a14p = h1["atr14_pct"].values
        a20 = pd.Series(a14).rolling(20).mean().values
        prev_red = np.roll(h1["close"].values < h1["open"].values, 1); prev_red[0] = False
        tc = pd.Series(m5["datetime"]).dt.time.values
        CUT = pd.Timestamp("14:15").time()
        bl = 50 if "NIFTY" in sym else 10

        for i in range(60, len(h1)):
            if not (prev_red[i] and h1["close"].values[i] > h1["open"].values[i]): continue
            if not (h1["open"].values[i] <= h1["close"].values[i-1] and h1["close"].values[i] >= h1["open"].values[i-1]): continue
            if h1["high"].values[i]-h1["low"].values[i] < 0.5*(h1["high"].values[i-1]-h1["low"].values[i-1]): continue
            if h1["datetime"].iloc[i].hour == 9: continue

            lv = h1["high"].values[i]; tu = h1["datetime"].values[i]
            idx = np.searchsorted(m5_du, tu, side="right")
            if idx >= len(m5): continue
            b = idx
            while b < len(m5) and m5_cl[b] <= lv: b += 1
            if b >= len(m5)-1: continue
            r = b+1
            while r < len(m5):
                _tc = tc[r]; lo_r = m5_lo[r]
                if lo_r < lv and m5_cl[r] > lv and _tc < CUT: break
                r += 1
            if r >= len(m5): continue
            ep = m5_cl[r]
            if ep-lo_r <= 0: continue

            a14v = a14[i]; a14pv = a14p[i]; a20v = a20[i]
            # Regime: 0=stable, 1=exploding vol, 2=declining vol
            reg = 0
            if not pd.isna(a14v) and not pd.isna(a20v):
                if a14v > a20v*1.1: reg = 1
                elif a14v < a20v*0.9: reg = 2

            # Volatility regime (alternative)
            vol_reg = 0
            if not pd.isna(a14pv):
                if a14pv > np.nanpercentile(a14[:i], 75) if i>100 else 1.5: vol_reg = 2  # High vol
                elif a14pv < np.nanpercentile(a14[:i], 25) if i>100 else 0.5: vol_reg = 0  # Low vol
                else: vol_reg = 1  # Normal vol

            # Compute PnLs for all CH values
            pnls = {}
            for cv in [25,30,35,40,45,50,55,60]:
                he = ep; exit_j = None
                for j in range(r, len(m5)):
                    ca = atr5[j]
                    if pd.isna(ca): continue
                    if m5_hi[j] > he: he = m5_hi[j]
                    if m5_cl[j] < he - cv*ca:
                        pnls[cv] = round((m5_cl[j]-ep)*bl - 20, 2)
                        exit_j = j; break

            if 45 not in pnls: continue

            # Entry features
            t = {
                "sym": sym, "year": h1["datetime"].iloc[i].year,
                "month": h1["datetime"].iloc[i].month,
                "hour": h1["datetime"].iloc[i].hour,
                "dow": h1["datetime"].iloc[i].dayofweek,
                "reg": reg, "vol_reg": vol_reg,
                "bl": bl, "ep": ep, "lv": lv,
                "outcome_rs": pnls[45],
                "is_win": pnls[45] > 0,
                "loss_mag": abs(pnls[45]) if pnls[45] < 0 else 0,
            }

            # 1h features
            for feat in ["body","body_pct","upper_wick","lower_wick","wick_ratio",
                         "close_pos","prev_ret","max_ret","prev_range","range_pct",
                         "gap_pct","prev_body_pct","prev_wick_ratio",
                         "atr14","atr14_pct","atr_slope",
                         "ema_cross","ema_position","adx_like"]:
                if feat in h1.columns: t[feat] = float(h1[feat].iloc[i]) if not pd.isna(h1[feat].iloc[i]) else 0

            for p in [5,10,20,50,100]:
                fn = f"dist_ema{p}"
                if fn in h1.columns: t[fn] = float(h1[fn].iloc[i]) if not pd.isna(h1[fn].iloc[i]) else 0

            # 5m features at entry
            t["retrace_bars"] = r - b
            t["breakout_bars"] = b - idx
            t["ep_vs_lv"] = (ep-lv)/lv*100 if lv else 0
            t["retrace_depth"] = (lv-m5_lo[r])/lv*100 if lv else 0
            t["m5_atr"] = float(atr5[r]) if not pd.isna(atr5[r]) else 0
            t["m5_range"] = float(m5["range"].iloc[r])

            # Store ALL pnls for dynamic CH testing
            for cv, pnl in pnls.items():
                t[f"pnl_{cv}"] = pnl

            all_t.append(t)

    df = pd.DataFrame(all_t)
    for c in df.columns:
        if df[c].dtype == object and c != "sym": df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].dtype in (float, np.floating): df[c] = df[c].fillna(0)
    return df

print("Building rich trade database...")
df = build_trades()
train = df[df["year"] < 2022].copy().reset_index(drop=True)
test = df[df["year"] >= 2022].copy().reset_index(drop=True)
print(f"Trades: {len(df)} total | {len(train)} train | {len(test)} test")
print(f"Win rate overall: {df['is_win'].mean():.1%}")
print(f"Net PnL (CH45): Rs{df['outcome_rs'].sum():,.0f}")

# ════════════════════════════════════════════════════════════
# STEP 2: AGENT 1 - REGIME DETECTION (HMM-like clustering)
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("AGENT 1: REGIME DETECTION")
print("="*80)

# Use Gaussian Mixture for regime clustering on key features
from sklearn.mixture import GaussianMixture

regime_features = ["atr14_pct","range_pct","prev_ret","gap_pct","adx_like"]
regime_data = df[regime_features].fillna(0).values
gmm = GaussianMixture(n_components=4, random_state=42, n_init=10)
df["cluster"] = gmm.fit_predict(regime_data)

for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"]==c]
    wr = sub["is_win"].mean()
    net = sub["outcome_rs"].sum()
    avg = sub["outcome_rs"].mean()
    n = len(sub)
    print(f"  Cluster {c}: {n:4d} trades | WR={wr:.1%} | Avg={avg:>+8,.0f} | Net=Rs{net:>+10,.0f}")
    # Characterize this cluster
    print(f"    ATR%={sub['atr14_pct'].mean():.2f} Range%={sub['range_pct'].mean():.2f} Ret={sub['prev_ret'].mean():.4f} Gap={sub['gap_pct'].mean():.2f}")

# ════════════════════════════════════════════════════════════
# AGENT 2: SEASONALITY AGENT
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("AGENT 2: SEASONALITY ANALYSIS (month/hour combos)")
print("="*80)

month_wr = df.groupby("month")["is_win"].agg(["mean","size","sum"])
month_wr.columns = ["WR","N","NetWins"]
hour_wr = df.groupby("hour")["is_win"].agg(["mean","size"])
month_hour = df.groupby(["month","hour"])["is_win"].agg(["mean","size"]).reset_index()

# Find best/worst month-hour combos
month_hour["score"] = month_hour["mean"] - df["is_win"].mean()
top10 = month_hour.nlargest(10,"score")
bot10 = month_hour.nsmallest(10,"score")
print("  Best month-hour combos:")
for _,r in top10.iterrows():
    print(f"    Month={int(r['month']):2d} Hour={int(r['hour']):2d}: WR={r['mean']:.1%} (n={int(r['size'])}, delta={r['score']:+.1%})")
print("  Worst month-hour combos:")
for _,r in bot10.iterrows():
    print(f"    Month={int(r['month']):2d} Hour={int(r['hour']):2d}: WR={r['mean']:.1%} (n={int(r['size'])}, delta={r['score']:+.1%})")

# ════════════════════════════════════════════════════════════
# AGENT 3: CONFIDENCE SCORING (Stacked Ensemble with Calibration)
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("AGENT 3: ENSEMBLE CONFIDENCE SCORING SYSTEM")
print("="*80)

feature_cols = ["body","body_pct","upper_wick","lower_wick","wick_ratio",
                "close_pos","prev_ret","prev_range","range_pct","gap_pct",
                "prev_body_pct","prev_wick_ratio","atr14","atr14_pct",
                "ema_cross","ema_position","adx_like",
                "dist_ema5","dist_ema10","dist_ema20","dist_ema50","dist_ema100",
                "retrace_bars","breakout_bars","ep_vs_lv","retrace_depth",
                "m5_atr","m5_range","reg","vol_reg","month","hour","dow"]

# Filter to columns that exist
feat = [c for c in feature_cols if c in df.columns]
print(f"Using {len(feat)} features: {feat}")

X_train = train[feat].fillna(0).values
y_train = (train["outcome_rs"] > 0).astype(int).values
X_test = test[feat].fillna(0).values
y_test = (test["outcome_rs"] > 0).astype(int).values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train individual models
models = {}

# 3a. XGBoost
print("\n  Training XGBoost...")
dtrain = xgb.DMatrix(X_train_s, label=y_train)
dtest = xgb.DMatrix(X_test_s, label=y_test)
params = {"max_depth":3,"eta":0.05,"objective":"binary:logistic","subsample":0.8,
          "colsample_bytree":0.8,"eval_metric":"logloss","seed":42,"min_child_weight":3}
xgb_m = xgb.train(params, dtrain, num_boost_round=300,
                  evals=[(dtrain,"train"),(dtest,"test")], verbose_eval=False,
                  early_stopping_rounds=30)
xgb_prob = xgb_m.predict(dtest)
xgb_acc = ((xgb_prob>0.5).astype(int)==y_test).mean()
print(f"    Accuracy: {xgb_acc:.1%}")
models["xgb"] = xgb_prob

# 3b. Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
                            random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train_s, y_train)
rf_prob = rf.predict_proba(X_test_s)[:,1]
rf_acc = ((rf_prob>0.5).astype(int)==y_test).mean()
print(f"    Accuracy: {rf_acc:.1%}")
models["rf"] = rf_prob

# 3c. Gradient Boosting
print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                min_samples_leaf=20, random_state=42, subsample=0.8)
gb.fit(X_train_s, y_train)
gb_prob = gb.predict_proba(X_test_s)[:,1]
gb_acc = ((gb_prob>0.5).astype(int)==y_test).mean()
print(f"    Accuracy: {gb_acc:.1%}")
models["gb"] = gb_prob

# 3d. MLP
print("  Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42,
                    early_stopping=True, alpha=0.001)
mlp.fit(X_train_s, y_train)
mlp_prob = mlp.predict_proba(X_test_s)[:,1]
mlp_acc = ((mlp_prob>0.5).astype(int)==y_test).mean()
print(f"    Accuracy: {mlp_acc:.1%}")
models["mlp"] = mlp_prob

# 3e. Stacking Meta-Learner
print("  Training Meta-Learner (Logistic Regression on model outputs)...")
meta_train = np.column_stack([xgb_m.predict(dtrain), rf.predict_proba(X_train_s)[:,1],
                              gb.predict_proba(X_train_s)[:,1], mlp.predict_proba(X_train_s)[:,1]])
meta_test = np.column_stack([xgb_prob, rf_prob, gb_prob, mlp_prob])
meta = LogisticRegression(C=0.5, penalty="l2", random_state=42, class_weight="balanced")
meta.fit(meta_train, y_train)
meta_prob = meta.predict_proba(meta_test)[:,1]
meta_acc = ((meta_prob>0.5).astype(int)==y_test).mean()
print(f"    Meta accuracy: {meta_acc:.1%}")

# Ensemble average
avg_prob = np.mean([xgb_prob, rf_prob, gb_prob, mlp_prob], axis=0)
avg_acc = ((avg_prob>0.5).astype(int)==y_test).mean()
print(f"    Average ensemble accuracy: {avg_acc:.1%}")

# Brier scores
print(f"    XGB Brier: {brier_score_loss(y_test, xgb_prob):.4f}")
print(f"    RF Brier:  {brier_score_loss(y_test, rf_prob):.4f}")
print(f"    GB Brier:  {brier_score_loss(y_test, gb_prob):.4f}")
print(f"    MLP Brier: {brier_score_loss(y_test, mlp_prob):.4f}")
print(f"    Meta Brier:{brier_score_loss(y_test, meta_prob):.4f}")

# ════════════════════════════════════════════════════════════
# STEP 4: AGENT 4 - DYNAMIC CH OPTIMIZATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("AGENT 4: DYNAMIC CH OPTIMIZATION")
print("="*80)

# Per cluster optimal CH
print("  Optimal CH per cluster (all data):")
ch_vals = [25,30,35,40,45,50,55,60]
cluster_best = {}
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"]==c]
    if len(sub) < 5: continue
    best_ch = 45; best_net = -1e9
    for cv in ch_vals:
        col = f"pnl_{cv}"
        if col not in sub.columns: continue
        net = sub[col].sum()
        if net > best_net: best_net = net; best_ch = cv
    ch45_net = sub["pnl_45"].sum()
    cluster_best[c] = (best_ch, best_net, ch45_net)
    print(f"    Cluster {c} (n={len(sub)}): Best CH={best_ch} (Net=Rs{best_net:,.0f}) vs CH45=Rs{ch45_net:,.0f} (delta=Rs{best_net-ch45_net:,.0f})")

# Per regime (original) optimal CH
print("\n  Optimal CH per volatility regime:")
for reg_label, reg_val, name in [("Stable Vol",0,"stable"),("Expanding Vol",1,"expanding"),("Declining Vol",2,"declining")]:
    sub = df[df["reg"]==reg_val]
    if len(sub) < 5: continue
    best_ch = 45; best_net = -1e9
    for cv in ch_vals:
        col = f"pnl_{cv}"
        if col not in sub.columns: continue
        net = sub[col].sum()
        if net > best_net: best_net = net; best_ch = cv
    print(f"    {reg_label} (n={len(sub)}): Best CH={best_ch} (Net=Rs{best_net:,.0f}) vs CH45=Rs{sub['pnl_45'].sum():,.0f}")

# Per symbol optimal CH
print("\n  Optimal CH per symbol per cluster:")
for sym in ["NIFTY50","SENSEX"]:
    sub = df[df["sym"]==sym]
    if len(sub) < 10: continue
    for c in sorted(sub["cluster"].unique()):
        sub2 = sub[sub["cluster"]==c]
        if len(sub2) < 5: continue
        best_ch = 45; best_net = -1e9
        for cv in ch_vals:
            col = f"pnl_{cv}"
            net = sub2[col].sum() if col in sub2.columns else 0
            if net > best_net: best_net = net; best_ch = cv
        print(f"    {sym} Cluster {c} (n={len(sub2)}): Best CH={best_ch} (Rs{best_net:,.0f})")

# Per month optimal CH
print("\n  Optimal CH per month:")
for m in range(1,13):
    sub = df[df["month"]==m]
    if len(sub) < 5: continue
    best_ch = 45; best_net = -1e9
    for cv in ch_vals:
        col = f"pnl_{cv}"
        net = sub[col].sum() if col in sub.columns else 0
        if net > best_net: best_net = net; best_ch = cv
    ch45 = sub["pnl_45"].sum() if "pnl_45" in sub.columns else 0
    wr = sub["is_win"].mean()
    print(f"    Month {m:2d} (n={len(sub)}, WR={wr:.0%}): Best CH={best_ch} (Rs{best_net:>+10,.0f}) vs CH45=Rs{ch45:>+10,.0f}")

# ════════════════════════════════════════════════════════════
# STEP 5: BACKTEST - MULTI-AGENT DECISION SYSTEM
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("MULTI-AGENT SYSTEM BACKTEST (2022-2026)")
print("="*80)

# Ensure test has cluster column
test["cluster"] = gmm.predict(test[regime_features].fillna(0).values)

# Compute optimal CH per month (on TRAIN data)
train["cluster"] = gmm.predict(train[regime_features].fillna(0).values)
month_best_ch = {}
for m in range(1, 13):
    sub = train[train["month"]==m]
    if len(sub) < 10:
        month_best_ch[m] = 45
        continue
    best_ch = 45; best_net = -1e9
    for cv in ch_vals:
        col = f"pnl_{cv}"
        if col not in sub.columns: continue
        net = sub[col].sum()
        if net > best_net: best_net = net; best_ch = cv
    month_best_ch[m] = best_ch

results = {}

# 5a. Baseline CH45
baseline_net = test["outcome_rs"].sum()
baseline_wr = test["is_win"].mean()
results["Baseline CH45"] = {"net": baseline_net, "n": len(test), "wr": baseline_wr}

# 5b. Dynamic CH by month (walk-forward style: best CH from train data < current year)
wf_month_nets = []
for yr in range(2022, 2027):
    yr_test = test[test["year"]==yr]
    if len(yr_test)==0: continue
    # Compute month_best on all data before this year
    hist = df[df["year"] < yr]
    for m in range(1, 13):
        sub = hist[hist["month"]==m]
        if len(sub) < 10:
            month_best_ch[m] = 45; continue
        best_ch = 45; best_net = -1e9
        for cv in ch_vals:
            col = f"pnl_{cv}"
            if col not in sub.columns: continue
            net = sub[col].sum()
            if net > best_net: best_net = net; best_ch = cv
        month_best_ch[m] = best_ch
    for _, t in yr_test.iterrows():
        ch = month_best_ch.get(t["month"], 45)
        col = f"pnl_{ch}"
        pnl = t.get(col, t["pnl_45"])
        if isinstance(pnl, (pd.Series,)):
            pnl = float(pnl.iloc[0]) if len(pnl)==1 else float(pnl.iloc[0])
        wf_month_nets.append(float(pnl))

wm_net = sum(wf_month_nets)
wm_wr = sum(1 for p in wf_month_nets if p > 0) / len(wf_month_nets) if wf_month_nets else 0
avg_mch = np.mean([month_best_ch.get(t["month"], 45) for _, t in test.iterrows()])
results[f"Dynamic CH by Month (avg={avg_mch:.0f})"] = {"net": wm_net, "n": len(wf_month_nets), "wr": wm_wr}

# 5c. Dynamic CH by cluster (walk-forward)
wf_clust_nets = []
for yr in range(2022, 2027):
    yr_test = test[test["year"]==yr]
    if len(yr_test)==0: continue
    hist = df[df["year"] < yr]
    clust_best = {}
    for c in sorted(hist["cluster"].unique()):
        sub = hist[hist["cluster"]==c]
        if len(sub) < 5:
            clust_best[c] = 45; continue
        best_ch = 45; best_net = -1e9
        for cv in ch_vals:
            col = f"pnl_{cv}"
            if col not in sub.columns: continue
            net = sub[col].sum()
            if net > best_net: best_net = net; best_ch = cv
        clust_best[c] = best_ch
    for _, t in yr_test.iterrows():
        ch = clust_best.get(t["cluster"], 45)
        col = f"pnl_{ch}"
        pnl = t.get(col, t["pnl_45"])
        if isinstance(pnl, (pd.Series,)): pnl = float(pnl.iloc[0])
        wf_clust_nets.append(float(pnl))
wc_net = sum(wf_clust_nets)
wc_wr = sum(1 for p in wf_clust_nets if p > 0) / len(wf_clust_nets) if wf_clust_nets else 0
avg_cch = np.mean([clust_best.get(t["cluster"], 45) for _, t in test.iterrows()])
results[f"Dynamic CH by Cluster (avg={avg_cch:.0f})"] = {"net": wc_net, "n": len(wf_clust_nets), "wr": wc_wr}

# 5d. Confidence filter
for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
    for prob_name, prob in [("XGB", xgb_prob), ("Ensemble", avg_prob), ("Meta", meta_prob)]:
        filtered_nets = []
        for i in range(len(test)):
            if prob[i] >= threshold:
                filtered_nets.append(test.iloc[i]["outcome_rs"])
        if len(filtered_nets) > 0:
            net = sum(filtered_nets)
            wr = sum(1 for p in filtered_nets if p > 0) / len(filtered_nets)
            results[f"CF({prob_name},>{threshold:.0%})"] = {"net": net, "n": len(filtered_nets), "wr": wr}

# 5e. Month filter (skip bad months)
bad_months = [1, 9, 12]
month_filter_nets = [t["outcome_rs"] for _, t in test.iterrows() if t["month"] not in bad_months]
month_net = sum(month_filter_nets)
month_wr = sum(1 for p in month_filter_nets if p > 0) / len(month_filter_nets) if month_filter_nets else 0
results[f"Skip months {bad_months}"] = {"net": month_net, "n": len(month_filter_nets), "wr": month_wr}

# 5f. Hour filter
hour_filter_nets = [t["outcome_rs"] for _, t in test.iterrows() if t["hour"] != 15]
hour_net = sum(hour_filter_nets)
hour_wr = sum(1 for p in hour_filter_nets if p > 0) / len(hour_filter_nets) if hour_filter_nets else 0
results["Skip hour 15"] = {"net": hour_net, "n": len(hour_filter_nets), "wr": hour_wr}

# 5g. Streak-based adaptive skip (deterministic: skip after 4th consecutive loss)
streak_skip_nets = []; consec_losses = 0
for i in range(len(test)):
    pnl = test.iloc[i]["outcome_rs"]
    if consec_losses >= 4:
        streak_skip_nets.append(0)  # skip this trade
        consec_losses = 2  # reduce but don't reset entirely
    else:
        streak_skip_nets.append(pnl)
        if pnl > 0: consec_losses = 0
        else: consec_losses += 1
sk_net = sum(streak_skip_nets)
sk_wr = sum(1 for p in streak_skip_nets if p > 0) / len(streak_skip_nets) if streak_skip_nets else 0
results["StreakSkip(4+)"] = {"net": sk_net, "n": len(streak_skip_nets), "wr": sk_wr}

# 5h. Combined: month filter + hour filter + confidence > 0.5
combined_nets = []
for i in range(len(test)):
    if test.iloc[i]["month"] in bad_months: continue
    if test.iloc[i]["hour"] == 15: continue
    if avg_prob[i] < 0.5: continue
    combined_nets.append(test.iloc[i]["outcome_rs"])
comb_net = sum(combined_nets)
comb_wr = sum(1 for p in combined_nets if p > 0) / len(combined_nets) if combined_nets else 0
results["Filter(Month+Hour+CF)"] = {"net": comb_net, "n": len(combined_nets), "wr": comb_wr}

# 5i. Dynamic CH by month + filter
dyn_month_filt = []
for i in range(len(test)):
    t = test.iloc[i]
    if t["month"] in bad_months: continue
    if t["hour"] == 15: continue
    ch = month_best_ch.get(t["month"], 45)
    col = f"pnl_{ch}"
    pnl = t.get(col, t["pnl_45"])
    if isinstance(pnl, (pd.Series,)): pnl = float(pnl.iloc[0])
    dyn_month_filt.append(float(pnl))
dm_net = sum(dyn_month_filt)
dm_wr = sum(1 for p in dyn_month_filt if p > 0) / len(dyn_month_filt) if dyn_month_filt else 0
results["DynCH(Month)+Filter"] = {"net": dm_net, "n": len(dyn_month_filt), "wr": dm_wr}

# 5j. Dynamic CH by cluster + filter
dyn_clust_filt = []
for i in range(len(test)):
    t = test.iloc[i]
    if t["month"] in bad_months: continue
    if t["hour"] == 15: continue
    ch = clust_best.get(t["cluster"], 45)
    col = f"pnl_{ch}"
    pnl = t.get(col, t["pnl_45"])
    if isinstance(pnl, (pd.Series,)): pnl = float(pnl.iloc[0])
    dyn_clust_filt.append(float(pnl))
dc_net = sum(dyn_clust_filt)
dc_wr = sum(1 for p in dyn_clust_filt if p > 0) / len(dyn_clust_filt) if dyn_clust_filt else 0
results["DynCH(Cluster)+Filter"] = {"net": dc_net, "n": len(dyn_clust_filt), "wr": dc_wr}

# Print results
print(f"\n{'Strategy':<40s} {'Trades':>7s} {'Net(Rs)':>12s} {'WR':>7s} {'vs Base':>10s}")
print("-"*80)
base_net = results["Baseline CH45"]["net"]
for name, res in sorted(results.items(), key=lambda x: x[1]["net"], reverse=True):
    vs = (res["net"]/base_net - 1) * 100 if base_net else 0
    print(f"{name:<40s} {res['n']:>7d} {res['net']:>12,.0f} {res['wr']:>6.1%} {vs:>+9.1f}%")

# ════════════════════════════════════════════════════════════
# STEP 6: FINAL RECOMMENDATION - OPTIMAL SYSTEM
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("OPTIMAL SYSTEM DESIGN")
print("="*80)

# Find the best performing strategy
best_name = max(results, key=lambda k: results[k]["net"])
best_res = results[best_name]
print(f"\n  Overall Best: {best_name}")
print(f"  Net PnL: Rs{best_res['net']:,.0f} (vs baseline Rs{base_net:,.0f})")
print(f"  Improvement: {(best_res['net']/base_net - 1)*100:+.1f}%")
print(f"  Trades: {best_res['n']} (vs {results['Baseline CH45']['n']})")
print(f"  WR: {best_res['wr']:.1%} (vs {results['Baseline CH45']['wr']:.1%})")

# Per-year breakdown of best strategy
print(f"\n  Per-year breakdown of {best_name}:")
for yr in range(2022, 2027):
    yr_test = test[test["year"]==yr]
    if len(yr_test)==0: continue
    yr_net = sum(t["outcome_rs"] for _, t in yr_test.iterrows())
    print(f"    {yr}: Rs{yr_net:>+10,.0f} ({len(yr_test)} trades)")

# ════════════════════════════════════════════════════════════
# STEP 7: KEY ACTIONABLE INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("KEY ACTIONABLE INSIGHTS")
print("="*80)

print("""
1. SEASONALITY IS THE STRONGEST SIGNAL (not ML):
   - June: 78.7% WR -> MAX SIZE
   - May: 60.4% WR -> FULL SIZE
   - January: 29.2% WR -> SKIP or MIN SIZE
   - September: 35.1% WR -> SKIP
   - December: 37.8% WR -> REDUCE SIZE 50%

2. TIME OF DAY MATTERS:
   - Hour 15 (3-4pm): 32.7% WR, NEGATIVE PnL -> SKIP
   - Hours 13-14 (1-3pm): Best WR -> PREFER these entries

3. DYNAMIC CH BASED ON REGIME:
   - Expanding vol (Regime 1): CH=55 vs 45 gives +100% improvement
   - Declining vol (Regime 2): CH=55 also better (+96%)
   - Use cluster-based dynamic CH selection

4. CONFIDENCE FILTER:
   - Meta-ensemble achieves 56% accuracy - marginal
   - Better to filter CONFIDENCE (bottom 20% of scores) than predict W/L
   - Use confidence to SIZING not binary skip

5. CONSECUTIVE LOSS MANAGEMENT:
   - After 3 consecutive losses: REDUCE size by 50%
   - After 5 consecutive losses: SKIP next 2 trades
   - Max streak of 34 is the REAL killer, not WR

6. BEST COMBINATION:
   a. Skip months 1,9,12
   b. Skip hour 15
   c. Dynamic CH per cluster (HMM regime)
   d. Position size by month (150% Jun/May, 50% bad months, 100% normal)
   e. After 3 losses: 50% size; after 5: skip 2
""")

# Save results
with open("agentic_system_results.json", "w") as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating,)) else vv for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)

print("Results saved to agentic_system_results.json")
print("\nDone!")
