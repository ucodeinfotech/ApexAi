"""
Check if Transformer/ML/RL models reduce LOSSES even if total P&L drops.
Key metrics: total losses captured, total losses avoided, avg loss size, worst loss.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"

# Reload and generate all trades with P&Ls
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ═══ TRANSFORMER ═══
def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

print("Loading data...")
h1d={}; m5d={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(os.path.join(BASE,f"{sym}_ONE_HOUR.csv"))
    m5=pd.read_csv(os.path.join(BASE,f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"]=pd.to_datetime(h1["datetime"]); m5["datetime"]=pd.to_datetime(m5["datetime"])
    h1=h1.sort_values("datetime").reset_index(drop=True)
    m5=m5.sort_values("datetime").reset_index(drop=True)
    h1["body"]=(h1["close"]-h1["open"]).abs()
    h1["range"]=h1["high"]-h1["low"]
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-h1["close"].shift(1)).abs(),(h1["low"]-h1["close"].shift(1)).abs()],axis=1).max(axis=1)
    h1["atr14"]=tr.rolling(14,min_periods=14).mean()
    h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1d[sym]=h1; m5d[sym]=m5

CUTOFF=pd.Timestamp("14:15").time()
SEQ_LEN=15
FEATURES=["open","high","low","close","volume","body","range"]

all_pnls = []
all_seqs = []
all_single_features = []
all_dates = []

for sym in ["NIFTY50","SENSEX"]:
    h1=h1d[sym]; m5=m5d[sym]
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=50 if "NIFTY" in sym else 10
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; tc=m5["datetime"].dt.time

    for i in range(SEQ_LEN+60, len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*0.5: continue
        
        seq = []
        for j in range(i-SEQ_LEN, i):
            seq.append(h1[FEATURES].iloc[j].values.astype(np.float64))
        all_seqs.append(seq)
        
        tu=int(pd.to_datetime(h1["datetime"].iloc[i]).timestamp()); lv=h1["high"].iloc[i]
        idx=np.searchsorted(du,tu,side="right")
        if idx>=len(m5): all_seqs.pop(); continue
        b=idx
        while b<len(m5) and cl[b]<=lv: b+=1
        if b>=len(m5): all_seqs.pop(); continue
        r=b+1
        while r<len(m5):
            if lo[r]<lv and cl[r]>lv and tc.iloc[r]<CUTOFF: break
            r+=1
        if r>=len(m5): all_seqs.pop(); continue
        ep=cl[r]
        if ep-lo[r]<=0 or m5["datetime"].iloc[r].hour==9: all_seqs.pop(); continue
        
        atr14_v=h1["atr14"].iloc[i]; atr_ma_v=h1["atr_ma20"].iloc[i]
        ch=35 if (not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v) else 55 if (not pd.isna(atr14_v)) else 45
        he=ep
        for j in range(r+1,len(m5)):
            ca=atr5[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch*ca:
                pnl=(cl[j]-ep)*bl*1-20*1
                all_pnls.append(pnl)
                all_dates.append(h1["datetime"].iloc[i])
                break
        else:
            all_seqs.pop()

print(f"\nTotal trades: {len(all_pnls)}")
print(f"Total P&L: Rs{sum(all_pnls):+,.0f}")
print(f"Winners: {(np.array(all_pnls)>0).sum()}, Losers: {(np.array(all_pnls)<0).sum()}")
print(f"WR: {np.mean(np.array(all_pnls)>0)*100:.1f}%")

# ═══ BASELINE LOSS ANALYSIS ═══
pnls = np.array(all_pnls)
wins = pnls[pnls > 0]
losses = pnls[pnls < 0]
print(f"\n{'='*65}")
print("BASELINE LOSS ANALYSIS")
print(f"{'='*65}")
print(f"  Total Win P&L:  Rs{wins.sum():+,.0f}")
print(f"  Total Loss P&L: Rs{losses.sum():+,.0f}")
print(f"  Net P&L:        Rs{pnls.sum():+,.0f}")
print(f"  Avg Win:        Rs{wins.mean():+,.0f}")
print(f"  Avg Loss:       Rs{losses.mean():+,.0f}")
print(f"  Max Loss:       Rs{losses.min():+,.0f}")
print(f"  Worst 5 Losses: {', '.join(f'Rs{x:,.0f}' for x in sorted(losses)[:5])}")
print(f"  Loss Count:     {len(losses)} / {len(pnls)}")

# ═══ SPLIT ═══
split = int(len(pnls) * 0.7)
pnls_tr, pnls_te = pnls[:split], pnls[split:]

X_seq = np.array(all_seqs, dtype=np.float64)
B, T, F = X_seq.shape
X_flat = X_seq.reshape(-1, F)
scaler = StandardScaler()
X_flat_n = scaler.fit_transform(X_flat)
X_norm = X_flat_n.reshape(B, T, F)

X_tr, X_te = X_norm[:split], X_norm[split:]

# ═══ 1. TRANSFORMER + XGBOOST ═══
print(f"\n{'='*65}")
print("TRANSFORMER SEQUENTIAL MODEL")
print(f"{'='*65}")

# Simple Transformer implementation
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)

def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True); v = x.var(axis=-1, keepdims=True)
    return g * (x - m) / np.sqrt(v + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

class MHA:
    def __init__(self, d, h):
        self.d=d; self.h=h; self.dk=d//h
        s=d**-0.5
        self.Wq=np.random.RandomState(1).uniform(-s,s,(d,d))
        self.Wk=np.random.RandomState(2).uniform(-s,s,(d,d))
        self.Wv=np.random.RandomState(3).uniform(-s,s,(d,d))
        self.Wo=np.random.RandomState(4).uniform(-s,s,(d,d))
    def forward(self, x):
        B,T,D=x.shape
        Q=x@self.Wq; K=x@self.Wk; V=x@self.Wv
        Q=Q.reshape(B,T,self.h,self.dk).transpose(0,2,1,3)
        K=K.reshape(B,T,self.h,self.dk).transpose(0,2,1,3)
        V=V.reshape(B,T,self.h,self.dk).transpose(0,2,1,3)
        S=Q@K.transpose(0,1,3,2)/self.dk**0.5
        A=softmax(S)
        O=A@V
        O=O.transpose(0,2,1,3).reshape(B,T,D)
        return O@self.Wo

class FF:
    def __init__(self, d, dff):
        s=d**-0.5
        self.W1=np.random.RandomState(5).uniform(-s,s,(d,dff))
        self.b1=np.zeros(dff)
        self.W2=np.random.RandomState(6).uniform(-s,s,(dff,d))
        self.b2=np.zeros(d)
    def forward(self, x):
        return gelu(x@self.W1+self.b1)@self.W2+self.b2

class TB:
    def __init__(self, d, h, dff):
        self.a=MHA(d,h); self.f=FF(d,dff)
        self.g1=np.ones(d); self.b1=np.zeros(d)
        self.g2=np.ones(d); self.b2=np.zeros(d)
    def forward(self, x):
        x=layer_norm(x+self.a.forward(x),self.g1,self.b1)
        x=layer_norm(x+self.f.forward(x),self.g2,self.b2)
        return x

class TF:
    def __init__(self, sl, nf, d=64, h=4, nl=2, dff=128):
        self.d=d
        s=nf**-0.5
        self.emb=np.random.RandomState(7).uniform(-s,s,(nf,d))
        self.emb_b=np.zeros(d)
        self.pe=self._pe(sl,d)
        self.blocks=[TB(d,h,dff) for _ in range(nl)]
    def _pe(self, sl, d):
        pe=np.zeros((sl,d))
        for p in range(sl):
            for i in range(0,d,2):
                pe[p,i]=np.sin(p/10000**(i/d))
                if i+1<d: pe[p,i+1]=np.cos(p/10000**(i/d))
        return pe
    def forward(self, x):
        B,T,F=x.shape
        x=x@self.emb+self.emb_b+self.pe[:T]
        for b in self.blocks: x=b.forward(x)
        return x.mean(axis=1)

print("Extracting Transformer features...")
tf_rand = TF(SEQ_LEN, F, d=64, h=4, nl=2, dff=128)
X_tf = np.array([tf_rand.forward(x.reshape(1,T,F)) for x in X_norm]).reshape(B,64)
X_tf_tr, X_tf_te = X_tf[:split], X_tf[split:]

dtf = xgb.DMatrix(X_tf_tr, label=(pnls_tr>0).astype(int))
def_te = xgb.DMatrix(X_tf_te)
def_te_lbl = xgb.DMatrix(X_tf_te, label=(pnls_te>0).astype(int))
mtf = xgb.train({"max_depth":3,"eta":0.03,"objective":"binary:logistic","seed":42,"verbosity":0},
                dtf, num_boost_round=50, verbose_eval=0)
tf_probs = mtf.predict(def_te)

# Try multiple thresholds - pick the one that minimizes losses
print("\n  Threshold sweep for Transformer:")
results = []
for th in np.arange(0.05, 0.96, 0.05):
    enter = tf_probs > th
    skip = ~enter
    if enter.sum() == 0 or skip.sum() == 0:
        continue
    pnl_enter = pnls_te[enter].sum()
    pnl_skip = pnls_te[skip].sum()
    pnl_all = pnls_te.sum()
    loss_enter = pnls_te[enter][pnls_te[enter] < 0].sum() if (pnls_te[enter] < 0).sum() > 0 else 0
    loss_skip = pnls_te[skip][pnls_te[skip] < 0].sum() if (pnls_te[skip] < 0).sum() > 0 else 0
    loss_all = pnls_te[pnls_te < 0].sum()
    wr_enter = (pnls_te[enter] > 0).mean() * 100
    n_enter = enter.sum()
    n_skip = skip.sum()
    loss_reduction = (loss_all - loss_enter) / abs(loss_all) * 100 if loss_all < 0 else 0
    results.append((th, n_enter, pnl_enter, pnl_skip, wr_enter, loss_enter, loss_skip, loss_reduction))

# Sort by loss reduction
results.sort(key=lambda r: -r[7])
print(f"  {'Thresh':>6s} {'Enter':>6s} {'P&L Enter':>12s} {'P&L Skip':>12s} {'WR%':>6s} {'Loss Enter':>12s} {'Loss Red%':>10s}")
print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*6} {'-'*12} {'-'*10}")
for th, n, pe, ps, wr, le, ls, lr in results[:8]:
    print(f"  {th:>5.2f} {n:>6d} Rs{pe:>+9,.0f} Rs{ps:>+9,.0f} {wr:>5.1f}% Rs{le:>+9,.0f} {lr:>+8.1f}%")
print(f"  {'ALL':>6s} {len(pnls_te):>6d} Rs{pnls_te.sum():>+9,.0f} {'---':>12s} {(pnls_te>0).mean()*100:>5.1f}% Rs{pnls_te[pnls_te<0].sum():>+9,.0f} {'---':>10s}")

# ═══ 2. SIMPLE ML CLASSIFIER ═══  
print(f"\n{'='*65}")
print("SIMPLE XGBOOST (single candle features)")
print(f"{'='*65}")

# Build single-candle features
X_single = X_seq.reshape(B, -1)  # flat sequence
X_s_tr, X_s_te = X_single[:split], X_single[split:]
ds = xgb.DMatrix(X_s_tr, label=(pnls_tr>0).astype(int))
de = xgb.DMatrix(X_s_te, label=(pnls_te>0).astype(int))
ms = xgb.train({"max_depth":6,"eta":0.05,"objective":"binary:logistic","seed":42,"verbosity":0},
               ds, num_boost_round=100, verbose_eval=0)
s_probs = ms.predict(de)

print("\n  Threshold sweep for XGBoost:")
for th in np.arange(0.05, 0.96, 0.05):
    enter = s_probs > th
    if enter.sum() == 0: continue
    pnl_e = pnls_te[enter].sum()
    wr_e = (pnls_te[enter] > 0).mean() * 100
    loss_e = pnls_te[enter][pnls_te[enter] < 0].sum() if (pnls_te[enter] < 0).sum() > 0 else 0
    loss_skip = pnls_te[~enter][pnls_te[~enter] < 0].sum() if (~enter).sum() > 0 and (pnls_te[~enter] < 0).sum() > 0 else 0
    loss_all = pnls_te[pnls_te < 0].sum()
    lr = (loss_all - loss_e) / abs(loss_all) * 100 if loss_all < 0 else 0
    print(f"  {th:>5.2f} {enter.sum():>6d} Rs{pnl_e:>+9,.0f} {wr_e:>5.1f}% Rs{loss_e:>+9,.0f} {lr:>+7.1f}%")

# ═══ 3. BEST POSSIBLE LOSS REDUCTION (Oracle) ═══
print(f"\n{'='*65}")
print("ORACLE ANALYSIS: What's theoretically possible?")
print(f"{'='*65}")

pnls_sorted = np.sort(pnls_te)
cumulative = np.cumsum(pnls_sorted)
# Find the optimal skip point
best_skip_count = 0
best_net = pnls_te.sum()
for skip_n in range(1, min(100, len(pnls_sorted))):
    skipped_losses = pnls_sorted[:skip_n]
    kept = pnls_sorted[skip_n:]
    new_net = kept.sum()
    if new_net > best_net:
        best_net = new_net
        best_skip_count = skip_n

print(f"  Baseline net P&L:      Rs{pnls_te.sum():+,.0f}")
print(f"  Oracle best net P&L:   Rs{best_net:+,.0f}")
print(f"  Oracle skip count:     {best_skip_count} trades ({best_skip_count/len(pnls_te)*100:.1f}%)")
print(f"  Oracle improvement:    {(best_net/pnls_te.sum()-1)*100:+>+.1f}%")
print(f"  Oracle max possible:   Rs{pnls_te[pnls_te>0].sum():+,.0f} (skip ALL losers)")

# ═══ 4. LOSS REDUCTION vs WIN RETENTION TRADE-OFF ═══  
print(f"\n{'='*65}")
print("KEY QUESTION: Do any models reduce losses more than they sacrifice wins?")
print(f"{'='*65}")

def measure_model(name, probs, pnls, threshold=0.5):
    enter = probs > threshold
    skip = ~enter
    n_ent = enter.sum()
    n_skip = skip.sum()
    if n_ent == 0:
        print(f"\n  {name}: No trades entered at threshold {threshold}")
        return
    
    win_ent = pnls[enter][pnls[enter] > 0].sum() if (pnls[enter] > 0).sum() > 0 else 0
    loss_ent = pnls[enter][pnls[enter] < 0].sum() if (pnls[enter] < 0).sum() > 0 else 0
    win_skipped = pnls[skip][pnls[skip] > 0].sum() if (pnls[skip] > 0).sum() > 0 else 0
    loss_skipped = pnls[skip][pnls[skip] < 0].sum() if (pnls[skip] < 0).sum() > 0 else 0
    win_all = pnls[pnls > 0].sum()
    loss_all = pnls[pnls < 0].sum()
    
    win_retention = win_ent / win_all * 100
    loss_avoidance = (loss_all - loss_ent) / abs(loss_all) * 100 if loss_all < 0 else 0
    
    print(f"\n  {name} @ thresh={threshold:.2f}:")
    print(f"    Enters: {n_ent}/{n_ent+n_skip} ({n_ent/(n_ent+n_skip)*100:.1f}%)")
    print(f"    Wins retained:   Rs{win_ent:+,.0f} ({win_retention:.1f}% of total wins Rs{win_all:+,.0f})")
    print(f"    Losses captured: Rs{loss_ent:+,.0f} ({100-loss_avoidance:.1f}% of total losses Rs{abs(loss_all):,.0f})")
    print(f"    Losses avoided:  Rs{loss_skipped:+,.0f}")
    print(f"    Win retention vs Loss avoidance: {win_retention:.1f}% vs {loss_avoidance:.1f}%")
    if win_retention > loss_avoidance:
        print(f"    VERDICT: {'SKIPS MORE WINS THAN LOSSES -- HARMFUL'}")
    else:
        print(f"    VERDICT: {'Avoids more losses than it sacrifices wins -- BENEFICIAL'}")

# Test each model at its "best" threshold
print("\n  --- Testing each model at multiple thresholds ---")
print(f"  {'Model':>20s} {'Thresh':>7s} {'Enter':>7s} {'Net P&L':>12s} {'WR%':>7s} {'Loss Red%':>10s} {'Win Ret%':>10s} {'Beneficial?':>12s}")
print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*12}")

# DQN-style: random skip (baseline for comparison)
for skip_pct in [0.25, 0.50, 0.75]:
    np.random.seed(42)
    skip_mask = np.random.random(len(pnls_te)) < skip_pct
    enter = ~skip_mask
    if enter.sum() == 0: continue
    w_ret = pnls_te[enter][pnls_te[enter] > 0].sum() / pnls_te[pnls_te > 0].sum() * 100
    l_avd = (pnls_te[pnls_te < 0].sum() - pnls_te[enter][pnls_te[enter] < 0].sum()) / abs(pnls_te[pnls_te < 0].sum()) * 100
    beneficial = "YES" if w_ret > l_avd else "NO"
    print(f"  {'Random Skip '+str(int(skip_pct*100))+'%':>20s} {'---':>7s} {enter.sum():>7d} Rs{pnls_te[enter].sum():>+9,.0f} {(pnls_te[enter]>0).mean()*100:>6.1f}% {l_avd:>+9.1f}% {w_ret:>+9.1f}% {beneficial:>12s}")

# Transformer at various thresholds
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    enter = tf_probs > th
    if enter.sum() < 2: continue
    w_ret = pnls_te[enter][pnls_te[enter] > 0].sum() / pnls_te[pnls_te > 0].sum() * 100
    l_avd = (pnls_te[pnls_te < 0].sum() - pnls_te[enter][pnls_te[enter] < 0].sum()) / abs(pnls_te[pnls_te < 0].sum()) * 100
    beneficial = "NO" if w_ret < l_avd else "YES"
    print(f"  {'Transformer':>20s} {th:>6.2f} {enter.sum():>7d} Rs{pnls_te[enter].sum():>+9,.0f} {(pnls_te[enter]>0).mean()*100:>6.1f}% {l_avd:>+9.1f}% {w_ret:>+9.1f}% {beneficial:>12s}")

# XGBoost at various thresholds
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    enter = s_probs > th
    if enter.sum() < 2: continue
    w_ret = pnls_te[enter][pnls_te[enter] > 0].sum() / pnls_te[pnls_te > 0].sum() * 100
    l_avd = (pnls_te[pnls_te < 0].sum() - pnls_te[enter][pnls_te[enter] < 0].sum()) / abs(pnls_te[pnls_te < 0].sum()) * 100
    beneficial = "NO" if w_ret < l_avd else "YES"
    print(f"  {'XGBoost':>20s} {th:>6.2f} {enter.sum():>7d} Rs{pnls_te[enter].sum():>+9,.0f} {(pnls_te[enter]>0).mean()*100:>6.1f}% {l_avd:>+9.1f}% {w_ret:>+9.1f}% {beneficial:>12s}")

# Always-trade baseline
w_ret = 100.0
l_avd = 0.0
print(f"  {'Baseline (all)':>20s} {'---':>7s} {len(pnls_te):>7d} Rs{pnls_te.sum():>+9,.0f} {(pnls_te>0).mean()*100:>6.1f}% {l_avd:>+9.1f}% {w_ret:>+9.1f}% {'---':>12s}")

# ═══ 5. ANALYSIS: WHAT DO CORRECTLY SKIPPED LOSSES LOOK LIKE? ═══
print(f"\n{'='*65}")
print("DETAILED: Trades correctly skipped vs incorrectly skipped")
print(f"{'='*65}")

# Use Transformer at threshold 0.5
th=0.5
enter = tf_probs > th
skip = ~enter

correct_skip = (pnls_te < 0) & skip  # Good: skipped a loser
wrong_skip = (pnls_te > 0) & skip    # Bad: skipped a winner
correct_enter = (pnls_te > 0) & enter # Good: entered a winner
wrong_enter = (pnls_te < 0) & enter   # Bad: entered a loser

print(f"\n  Transformer @ thresh={th}:")
print(f"    Correctly entered (win): {correct_enter.sum()} trades, Rs{pnls_te[correct_enter].sum():+,.0f}")
print(f"    Wrongly entered (loss):  {wrong_enter.sum()} trades, Rs{pnls_te[wrong_enter].sum():+,.0f}")
print(f"    Correctly skipped (loss):{correct_skip.sum()} trades, Rs{pnls_te[correct_skip].sum():+,.0f}")
print(f"    Wrongly skipped (win):   {wrong_skip.sum()} trades, Rs{pnls_te[wrong_skip].sum():+,.0f}")
print(f"    Rs lost by wrong skips:  Rs{pnls_te[wrong_skip].sum():+,.0f}")
print(f"    Rs saved by correct skip:Rs{abs(pnls_te[correct_skip].sum()):+,.0f}")
print(f"    Net benefit of skipping: Rs{pnls_te[correct_skip].sum() - pnls_te[wrong_skip].sum():+,.0f}")

# Check if there are SPECIFIC patterns in correctly skipped losses
print(f"\n  Correctly skipped losses (n={correct_skip.sum()}):")
if correct_skip.sum() > 0:
    skip_losses = pnls_te[correct_skip]
    print(f"    Count: {len(skip_losses)}, Total: Rs{skip_losses.sum():+,.0f}, Avg: Rs{skip_losses.mean():+,.0f}")
    print(f"    Top 5: {', '.join(f'Rs{x:,.0f}' for x in sorted(skip_losses)[:5])}")

print(f"\n  Wrongly skipped winners (n={wrong_skip.sum()}):")
if wrong_skip.sum() > 0:
    skip_wins = pnls_te[wrong_skip]
    print(f"    Count: {len(skip_wins)}, Total: Rs{skip_wins.sum():+,.0f}, Avg: Rs{skip_wins.mean():+,.0f}")
    print(f"    Top 5: {', '.join(f'Rs{x:,.0f}' for x in sorted(skip_wins, reverse=True)[:5])}")

# ═══ 6. FINAL VERDICT ═══
print(f"\n{'='*65}")
print("FINAL VERDICT")
print(f"{'='*65}")

# Check if ANY threshold of ANY model gives win_retention > loss_avoidance
best_balance = 0
best_desc = ""
for model_name, probs in [("Transformer", tf_probs), ("XGBoost", s_probs)]:
    for th in np.arange(0.05, 0.96, 0.05):
        enter = probs > th
        if enter.sum() < 5: continue
        w_ret = pnls_te[enter][pnls_te[enter] > 0].sum() / pnls_te[pnls_te > 0].sum() * 100
        l_avd = (pnls_te[pnls_te < 0].sum() - pnls_te[enter][pnls_te[enter] < 0].sum()) / abs(pnls_te[pnls_te < 0].sum()) * 100
        balance = l_avd - w_ret  # negative = bad, positive = good
        if balance > best_balance:
            best_balance = balance
            best_desc = f"{model_name} @ th={th:.2f}: enter={enter.sum()}, win_ret={w_ret:.1f}%, loss_avd={l_avd:.1f}%, net_loss_red_vs_sacrifice=Rs{pnls_te[correct_skip].sum() if 'correct_skip' in dir() else 0:,.0f}"

print(f"\n  Best found: {best_desc}")
if best_balance > 0:
    print(f"\n  RESULT: Models CAN reduce losses more than they sacrifice wins!")
    print(f"          But the absolute P&L is still lower than baseline.")
else:
    print(f"\n  RESULT: Even at optimal thresholds, ALL models lose more in skipped")
    print(f"          winners than they save in avoided losses.")

print(f"\nDONE")
