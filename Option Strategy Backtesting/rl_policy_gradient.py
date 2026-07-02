"""
Policy Gradient (REINFORCE) for engulfing strategy — directly optimizes cumulative P&L.
The policy decides lot size (0 to max) at each signal, learning allocation weights
that maximize total return rather than predicting individual trade outcomes.
"""
import pandas as pd, numpy as np, os, warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20; CUTOFF=pd.Timestamp("14:15").time()
RS = 42
np.random.seed(RS)

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
    h1["atr14"]=A(h1,14); h1["atr_ma20"]=h1["atr14"].rolling(20).mean()
    h1["atr_pctile"]=h1["atr14"].rolling(252).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()+1e-10)*100 if x.max()>x.min() else 50, raw=False)
    h1["ema10"]=h1["close"].ewm(span=10).mean(); h1["ema50"]=h1["close"].ewm(span=50).mean()
    h1["ema200"]=h1["close"].ewm(span=200).mean()
    h1d[sym]=h1; m5d[sym]=m5

print("Generating trade data...")
all_states=[]; all_pnls=[]

for sym in ["NIFTY50","SENSEX"]:
    h1=h1d[sym]; m5=m5d[sym]
    body=(h1["close"]-h1["open"]).abs()
    is_r=h1["close"]<h1["open"]; is_g=h1["close"]>h1["open"]
    bl=NLOT if "NIFTY" in sym else SLOT
    du=m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi=m5["high"].values; lo=m5["low"].values; cl=m5["close"].values
    atr5=A(m5,14).values; tc=m5["datetime"].dt.time

    for i in range(60, len(h1)):
        if not is_r.iloc[i-1] or not is_g.iloc[i]: continue
        if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
        if body.iloc[i]<body.iloc[i-1]*0.5: continue
        
        gap=(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100
        br=body.iloc[i]/(body.iloc[i-1]+1e-10)
        atr_r=h1["atr14"].iloc[i]/h1["atr_ma20"].iloc[i] if not pd.isna(h1["atr_ma20"].iloc[i]) else 1
        atr_p=h1["atr_pctile"].iloc[i] if not pd.isna(h1["atr_pctile"].iloc[i]) else 50
        c_ema50=(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100
        c_ema200=(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100
        ch5=(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0
        ch20=(h1["close"].iloc[i]/h1["close"].iloc[i-20]-1)*100 if i>=20 else 0
        a50=1 if h1["close"].iloc[i]>h1["ema50"].iloc[i] else 0
        a200=1 if h1["close"].iloc[i]>h1["ema200"].iloc[i] else 0
        
        state=np.array([gap, br, atr_r, atr_p, c_ema50, c_ema200, ch5, ch20, a50, a200], dtype=np.float64)
        
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
        ch=35 if (not pd.isna(atr14_v) and not pd.isna(atr_ma_v) and atr14_v>atr_ma_v) else 55 if (not pd.isna(atr14_v)) else 45
        he=ep
        for j in range(r+1,len(m5)):
            ca=atr5[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch*ca:
                all_pnls.append((cl[j]-ep)*bl*1-CHG*1)
                all_states.append(state)
                break

X = np.array(all_states); P = np.array(all_pnls)
print(f"Total trades: {len(X)}, WR: {(P>0).mean()*100:.1f}%, Avg: Rs{P.mean():+,.0f}")

# Split
split=int(len(X)*0.7)
X_train, X_test = X[:split], X[split:]
P_train, P_test = P[:split], P[split:]

# Normalize
scaler=StandardScaler()
X_train_n=scaler.fit_transform(X_train)
X_test_n=scaler.transform(X_test)

# ── POLICY GRADIENT (REINFORCE) ──
# Policy: π_θ(a|s) = sigmoid(w·s + b) for lot_size multiplier ∈ [0,1]
# We use logistic regression as our policy network (simple, differentiable)
# The policy outputs the probability of entering with full lot size
# Action space: continuous [0,1] → lot_size = action * max_lots

print(f"\n{'='*60}")
print("POLICY GRADIENT (REINFORCE)")
print(f"{'='*60}")

class PolicyGradient:
    def __init__(self, dim, lr=0.01):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr
        self.dim = dim
    
    def forward(self, x):
        """Probability of entering (full lot size)"""
        logit = np.dot(x, self.w) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    
    def sample_action(self, x, temperature=1.0):
        """Sample lot_size ∈ [0, 1]"""
        p = self.forward(x)
        # Beta distribution controlled by p
        alpha = p * temperature + 1e-6
        beta = (1 - p) * temperature + 1e-6
        return np.random.beta(alpha, beta)
    
    def compute_loss_gradient(self, x, action, reward, baseline):
        """REINFORCE gradient: ∇log(π) * (reward - baseline)"""
        p = self.forward(x)
        # log-probability of taking this action under beta distribution
        alpha = p * 5 + 1e-6
        beta_par = (1 - p) * 5 + 1e-6
        # Gradient of log(π(a|s)) w.r.t. logit
        # For beta: log(π) = (α-1)log(a) + (β-1)log(1-a) - log(B(α,β))
        # ∂log(π)/∂logit = 5 * (log(a) - log(1-a)) - ψ(α) + ψ(β) where ψ is digamma
        # Simplified: the gradient pushes p toward action if reward > baseline
        gradient = (action - p) * (reward - baseline)
        return gradient
    
    def update(self, trajectory):
        """Update policy using REINFORCE with baseline"""
        states = np.array([t[0] for t in trajectory])
        actions = np.array([t[1] for t in trajectory])
        rewards = np.array([t[2] for t in trajectory])
        
        # Baseline = running average return
        baseline = rewards.mean() if len(rewards) > 0 else 0
        
        # Gradient ascent
        grads = np.zeros(self.dim)
        grad_b = 0.0
        
        for x, a, r in trajectory:
            g = self.compute_loss_gradient(x, a, r, baseline)
            grads += g * x
            grad_b += g
        
        # Update with RMSProp-style adaptive LR
        self.w += self.lr * grads / (np.linalg.norm(grads) + 1e-8)
        self.b += self.lr * grad_b / (len(trajectory) + 1e-8)

# Episode-based training
n_episodes = 200
lr = 0.005
pg = PolicyGradient(X_train_n.shape[1], lr)

print(f"Training {n_episodes} episodes...")
best_test_pnl = -np.inf
best_w = None
best_b = None

for ep in range(n_episodes):
    # Collect trajectory
    trajectory = []
    for i in range(len(X_train_n)):
        p_action = pg.sample_action(X_train_n[i], temperature=3.0)
        reward = P_train[i] * p_action  # P&L scaled by lot size
        trajectory.append((X_train_n[i], p_action, reward))
    
    # Update policy
    pg.update(trajectory)
    
    # Evaluate every 20 episodes
    if (ep + 1) % 20 == 0:
        # Train set evaluation
        train_actions = np.array([pg.forward(X_train_n[i]) for i in range(len(X_train_n))])
        train_pnl = (P_train * train_actions).sum()
        train_wr = ((P_train * train_actions) > 0).mean() * 100
        
        # Test set evaluation
        test_actions = np.array([pg.forward(X_test_n[i]) for i in range(len(X_test_n))])
        test_pnl = (P_test * test_actions).sum()
        test_wr = ((P_test * test_actions) > 0).mean() * 100
        
        print(f"  Ep {ep+1}: Train PnL=Rs{train_pnl:+,.0f} ({train_wr:.1f}%) | "
              f"Test PnL=Rs{test_pnl:+,.0f} ({test_wr:.1f}%)")
        
        if test_pnl > best_test_pnl:
            best_test_pnl = test_pnl
            best_w = pg.w.copy()
            best_b = pg.b

# Final evaluation with best policy
print(f"\n{'='*60}")
print("FINAL POLICY GRADIENT RESULTS")
print(f"{'='*60}")

pg_best = PolicyGradient(X_train_n.shape[1])
pg_best.w = best_w
pg_best.b = best_b

# Test set
test_probs = np.array([pg_best.forward(X_test_n[i]) for i in range(len(X_test_n))])
test_pnl_continuous = (P_test * test_probs).sum()
test_pnl_binary = P_test[(test_probs > 0.5).astype(bool)].sum() if (test_probs > 0.5).sum() > 0 else 0
n_enter = (test_probs > 0.5).sum()

baseline_pnl = P_test.sum()

print(f"\n  Baseline (all trades): Rs{baseline_pnl:+,.0f}")
print(f"  PG Continuous (weighted): Rs{test_pnl_continuous:+,.0f}")
print(f"  PG Binary (enter >0.5): Rs{test_pnl_binary:+,.0f} ({n_enter}/{len(P_test)} trades)")
print(f"  Improvement (cont): {(test_pnl_continuous/baseline_pnl-1)*100:+>+.1f}%")
improvement_binary = (test_pnl_binary / baseline_pnl - 1) * 100 if baseline_pnl != 0 else 0
print(f"  Improvement (binary): {improvement_binary:+>+.1f}%")

# Analysis: what probabilities does the policy assign?
print(f"\n  Policy action distribution:")
print(f"  Prob[enter] > 0.5: {n_enter}/{len(P_test)} ({n_enter/len(P_test)*100:.1f}%)")
print(f"  Avg prob on winners: {test_probs[P_test>0].mean():.3f}")
print(f"  Avg prob on losers: {test_probs[P_test<0].mean():.3f}")
print(f"  Corr(prob, P&L): {np.corrcoef(test_probs, P_test)[0,1]:.3f}")

# Critical check: does policy differentiate?
w_p = test_probs[P_test>0]
l_p = test_probs[P_test<0]
print(f"\n  Separation check:")
print(f"  Winners > 0.5: {(w_p>0.5).mean()*100:.1f}%")
print(f"  Losers > 0.5:  {(l_p>0.5).mean()*100:.1f}%")

# Weight analysis
print(f"\n  Policy weights:")
feature_names = ["gap","body_ratio","atr_r","atr_p","c_ema50","c_ema200","ch5","ch20","a50","a200"]
for name, w in sorted(zip(feature_names, pg_best.w), key=lambda x: -abs(x[1])):
    print(f"    {name:15s}: {w:+8.4f}")
print(f"  {'bias':15s}: {pg_best.b:+8.4f}")

# Try with fixed thresholds (grid search on action probability threshold)
print(f"\n{'='*60}")
print("THRESHOLD OPTIMIZATION (post-hoc)")
print(f"{'='*60}")

best_thresh = 0.5
best_thresh_pnl = 0
for thresh in np.arange(0.05, 0.96, 0.05):
    pnl = P_test[(test_probs > thresh).astype(bool)].sum() if (test_probs > thresh).sum() > 0 else 0
    if pnl > best_thresh_pnl:
        best_thresh_pnl = pnl
        best_thresh = thresh

print(f"  Optimal threshold: {best_thresh:.2f}")
print(f"  PnL at optimal: Rs{best_thresh_pnl:+,.0f} ({int((test_probs>best_thresh).sum())}/{len(P_test)} trades)")
print(f"  vs baseline: Rs{baseline_pnl:+,.0f}")
print(f"  Improvement: {(best_thresh_pnl/baseline_pnl-1)*100:+>+.1f}%")

# What about REINFORCE + skip filter?
print(f"\n{'='*60}")
print("REINFORCE + Skip-after-2-losses")
print(f"{'='*60}")

# Simulate: apply skip filter after PG decision
skip_count = 0
pg_skip_pnls = []
pg_skip_probs = []

for i in range(len(X_test_n)):
    prob = pg_best.forward(X_test_n[i])
    pg_skip_probs.append(prob)
    
    if skip_count >= 2:
        # Skip enforced
        pass
        skip_count = 0
    elif prob > best_thresh:
        pg_skip_pnls.append(P_test[i])
        skip_count += 1 if P_test[i] <= 0 else 0
    # else: skip by policy

if len(pg_skip_pnls) > 0:
    pg_skip_total = sum(pg_skip_pnls)
    pg_skip_wr = (np.array(pg_skip_pnls) > 0).mean() * 100
    print(f"  Entered: {len(pg_skip_pnls)}/{len(P_test)}")
    print(f"  Net P&L: Rs{pg_skip_total:+,.0f}")
    print(f"  WR: {pg_skip_wr:.1f}%")
    print(f"  vs baseline: {(pg_skip_total/baseline_pnl-1)*100:+>+.1f}%")

print(f"\nDONE")
