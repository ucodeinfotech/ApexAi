"""
DL model - optimized: pre-load data, single pass, fast simulation
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT = 50; CHG = 20

def A(df, p=14):
    tr = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()

print("Loading data...")
h1 = pd.read_csv(os.path.join(BASE, "NIFTY50_ONE_HOUR.csv"))
m5 = pd.read_csv(os.path.join(BASE, "NIFTY50_FIVE_MINUTE.csv"))
h1["datetime"] = pd.to_datetime(h1["datetime"])
m5["datetime"] = pd.to_datetime(m5["datetime"])
h1 = h1.sort_values("datetime").reset_index(drop=True)
m5 = m5.sort_values("datetime").reset_index(drop=True)

print("Computing indicators...")
h1["atr14"] = A(h1, 14)
h1["atr_ma20"] = h1["atr14"].rolling(20).mean()
dc = h1["close"].diff().clip(lower=0)
dd = -h1["close"].diff().clip(upper=0)
h1["rsi14"] = 100 - (100 / (1 + dc.rolling(14).mean() / (dd.rolling(14).mean() + 1e-10)))
h1["ema10"] = h1["close"].ewm(span=10).mean()
h1["ema50"] = h1["close"].ewm(span=50).mean()
h1["ema200"] = h1["close"].ewm(span=200).mean()

body = (h1["close"] - h1["open"]).abs()
is_red = h1["close"] < h1["open"]
is_green = h1["close"] > h1["open"]

m5["atr5"] = A(m5, 14)
tc = m5["datetime"].dt.time
du = m5["datetime"].apply(lambda x: int(x.timestamp())).values
hi5 = m5["high"].values; lo5 = m5["low"].values; cl5 = m5["close"].values
atr5_v = m5["atr5"].values
cutoff = pd.Timestamp("14:15").time()

def find_entry_exit(idx_h1):
    """Given 1H index, find entry and exit on 5M chart"""
    tu = int(pd.to_datetime(h1["datetime"].iloc[idx_h1]).timestamp())
    lv = h1["high"].iloc[idx_h1]
    i = np.searchsorted(du, tu, side="right")
    if i >= len(m5):
        return None, None, None
    b = i
    while b < len(m5) and cl5[b] <= lv:
        b += 1
    if b >= len(m5):
        return None, None, None
    r = b + 1
    while r < len(m5):
        if lo5[r] < lv and cl5[r] > lv and tc.iloc[r] < cutoff:
            break
        r += 1
    if r >= len(m5):
        return None, None, None
    ep = cl5[r]
    if ep - lo5[r] <= 0 or m5["datetime"].iloc[r].hour == 9:
        return None, None, None
    ch = 45
    atr14_v = h1["atr14"].iloc[idx_h1]
    atr_ma_v = h1["atr_ma20"].iloc[idx_h1]
    if not pd.isna(atr14_v) and not pd.isna(atr_ma_v):
        ch = 35 if atr14_v > atr_ma_v else 55
    he = ep
    for j in range(r + 1, len(m5)):
        ca = atr5_v[j]
        if pd.isna(ca):
            continue
        if hi5[j] > he:
            he = hi5[j]
        if cl5[j] < he - ch * ca:
            return r, j, ep
    return None, None, None

print("Generating signals & simulating trades...")
records = []
for i in range(50, len(h1)):
    if not is_red.iloc[i-1] or not is_green.iloc[i]:
        continue
    if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]:
        continue
    if body.iloc[i] < body.iloc[i-1] * 0.5:
        continue
    entry_idx, exit_idx, ep = find_entry_exit(i)
    if entry_idx is None:
        continue

    exit_price = cl5[exit_idx]
    pnl = (exit_price - ep) * NLOT - CHG
    is_win = 1 if pnl > 0 else 0

    gap = (h1["open"].iloc[i] / h1["close"].iloc[i-1] - 1) * 100
    br = body.iloc[i] / body.iloc[i-1] if body.iloc[i-1] > 0 else 1
    rsi = h1["rsi14"].iloc[i] if not pd.isna(h1["rsi14"].iloc[i]) else 50
    atr_r = h1["atr14"].iloc[i] / h1["atr_ma20"].iloc[i] if not pd.isna(h1["atr_ma20"].iloc[i]) else 1
    c_ema50 = (h1["close"].iloc[i] / h1["ema50"].iloc[i] - 1) * 100
    c_ema200 = (h1["close"].iloc[i] / h1["ema200"].iloc[i] - 1) * 100
    ema50_slope = (h1["ema50"].iloc[i] / h1["ema50"].iloc[i-10] - 1) * 100 if i >= 10 else 0
    ch5 = (h1["close"].iloc[i] / h1["close"].iloc[i-5] - 1) * 100 if i >= 5 else 0
    ch20 = (h1["close"].iloc[i] / h1["close"].iloc[i-20] - 1) * 100 if i >= 20 else 0
    above_50 = 1 if h1["close"].iloc[i] > h1["ema50"].iloc[i] else 0
    above_200 = 1 if h1["close"].iloc[i] > h1["ema200"].iloc[i] else 0

    records.append({
        "gap": gap, "br": br, "rsi": rsi, "atr_ratio": atr_r,
        "cema50": c_ema50, "cema200": c_ema200, "ema50slope": ema50_slope,
        "ch5": ch5, "ch20": ch20, "a50": above_50, "a200": above_200,
        "hour": h1["datetime"].iloc[i].hour, "dow": h1["datetime"].iloc[i].dayofweek,
        "pnl": pnl, "is_win": is_win, "exit_time": m5["datetime"].iloc[exit_idx],
    })

df = pd.DataFrame(records)
print(f"Total trades: {len(df)}, WR: {df['is_win'].mean()*100:.1f}%")
print(f"Avg P&L: Rs{df['pnl'].mean():+,.0f}")

# ── TRAIN/EVAL ──
feats = ["gap", "br", "rsi", "atr_ratio", "cema50", "cema200",
         "ema50slope", "ch5", "ch20", "a50", "a200", "hour", "dow"]
X = df[feats].values
y = df["is_win"].values

split = int(len(X) * 0.7)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

print("\nTraining XGBoost...")
xgb_m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                          random_state=42, use_label_encoder=False)
xgb_m.fit(X_tr_s, y_tr)
yp = xgb_m.predict(X_te_s)
print(f"  Acc={(yp==y_te).mean():.3f}  Prec={((yp==1)&(y_te==1)).sum()/((yp==1).sum()+1e-10):.3f}  "
      f"Rec={((yp==1)&(y_te==1)).sum()/((y_te==1).sum()+1e-10):.3f}")

print("Training MLP (NN)...")
mlp = MLPClassifier((64, 32), max_iter=300, random_state=42)
mlp.fit(X_tr_s, y_tr)
yp2 = mlp.predict(X_te_s)
print(f"  Acc={(yp2==y_te).mean():.3f}  Prec={((yp2==1)&(y_te==1)).sum()/((yp2==1).sum()+1e-10):.3f}  "
      f"Rec={((yp2==1)&(y_te==1)).sum()/((y_te==1).sum()+1e-10):.3f}")

print("\nFeature Importance (XGB):")
for f, i in sorted(zip(feats, xgb_m.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:12s}: {i:.4f}")

# ── BACKTEST ──
df_te = df.iloc[split:].copy().reset_index(drop=True)
df_te["xgb"] = xgb_m.predict(scaler.transform(X_te))
df_te["mlp"] = mlp.predict(scaler.transform(X_te))

def backtest(data, skip=2):
    d = data.sort_values("exit_time").reset_index(drop=True)
    lc = 0; k = np.ones(len(d), dtype=bool)
    for i in range(len(d)):
        if lc >= skip:
            k[i] = False; lc = 0; continue
        if d["pnl"].iloc[i] <= 0:
            lc += 1
        else:
            lc = 0
    d = d[k].reset_index(drop=True) if skip > 0 else d
    n = len(d)
    if n == 0: return {"n":0,"net":0,"wr":0,"pf":0,"mdd":0,"mdd_p":0}
    net = d["pnl"].sum()
    wr = d["is_win"].mean() * 100
    pf = d[d["pnl"]>0]["pnl"].sum() / abs(d[d["pnl"]<0]["pnl"].sum()) if (d["pnl"]<0).sum() != 0 else 99
    eq = d["pnl"].cumsum() + 200000
    mdd = (eq.cummax() - eq).max()
    mdd_p = mdd / eq.cummax().max() * 100
    return {"n":n,"net":net,"wr":wr,"pf":pf,"mdd":mdd,"mdd_p":mdd_p}

base = backtest(df_te, 2)
print(f"\n{'='*55}")
print(f"{'Metric':20s}  {'Base':>10s}  {'XGBoost':>10s}  {'MLP':>10s}")
print(f"{'='*55}")
print(f"{'Trades':20s}  {base['n']:>10d}  {'-':>10s}  {'-':>10s}")
print(f"{'Net P&L':20s}  Rs{base['net']:>+7,.0f}  {'-':>10s}  {'-':>10s}")
print(f"{'WR':20s}  {base['wr']:>9.1f}%  {'-':>10s}  {'-':>10s}")
print(f"{'PF':20s}  {base['pf']:>9.2f}  {'-':>10s}  {'-':>10s}")
print(f"{'Max DD%':20s}  {base['mdd_p']:>9.2f}%  {'-':>10s}  {'-':>10s}")

for lbl, col in [("XGBoost", "xgb"), ("MLP", "mlp")]:
    fil = df_te[df_te[col] == 1].copy()
    r = backtest(fil, 2)
    if r["n"] > 0:
        imp = (r["net"] / base["net"] - 1) * 100 if base["net"] != 0 else 0
        skip = (1 - r["n"] / base["n"]) * 100
        print(f"\n{lbl}:")
        print(f"  Trades:  {base['n']} -> {r['n']} ({skip:.0f}% skipped)")
        print(f"  Net:     Rs{base['net']:+,.0f} -> Rs{r['net']:+,.0f} ({imp:+.1f}%)")
        print(f"  WR:      {base['wr']:.1f}% -> {r['wr']:.1f}%")
        print(f"  PF:      {base['pf']:.2f} -> {r['pf']:.2f}")
        print(f"  Max DD:  {base['mdd_p']:.2f}% -> {r['mdd_p']:.2f}%")
    else:
        print(f"\n{lbl}: No trades after filtering")

print("\nDONE")
