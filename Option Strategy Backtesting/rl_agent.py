"""
RL Agent for Engulfing Strategy — Q-Learning with Neural Network
Trains an agent to decide Enter/Skip at each engulfing signal to maximize cumulative P&L.
Uses sklearn MLPRegressor as Q-function approximator (no PyTorch needed).
"""
import pandas as pd, numpy as np, os, warnings, copy, random
from collections import deque
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
NLOT=50; SLOT=10; CHG=20; CUTOFF=pd.Timestamp("14:15").time()
RS = 42
random.seed(RS); np.random.seed(RS)

def A(df,p=14):
    tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p,min_periods=p).mean()

# ── PRE-LOAD DATA ──
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
    h1["rsi14"]=100-(100/(1+h1["close"].diff().clip(lower=0).rolling(14).mean()/(-h1["close"].diff().clip(upper=0).rolling(14).mean()+1e-10)))
    h1d[sym]=h1; m5d[sym]=m5

# ── GENERATE ALL TRADES WITH STATE FEATURES ──
print("Generating trade data...")
all_episodes = []  # List of episodes, each episode is a list of (state, action, reward, next_state) tuples
all_states = []
all_pnls = []

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
        
        # Extract state features at signal time
        gap=(h1["open"].iloc[i]/h1["close"].iloc[i-1]-1)*100
        br=body.iloc[i]/(body.iloc[i-1]+1e-10)
        rsi=h1["rsi14"].iloc[i] if not pd.isna(h1["rsi14"].iloc[i]) else 50
        atr_r=h1["atr14"].iloc[i]/h1["atr_ma20"].iloc[i] if not pd.isna(h1["atr_ma20"].iloc[i]) else 1
        atr_p=h1["atr_pctile"].iloc[i] if not pd.isna(h1["atr_pctile"].iloc[i]) else 50
        c_ema50=(h1["close"].iloc[i]/h1["ema50"].iloc[i]-1)*100
        c_ema200=(h1["close"].iloc[i]/h1["ema200"].iloc[i]-1)*100
        ema50_slope=(h1["ema50"].iloc[i]/h1["ema50"].iloc[i-10]-1)*100 if i>=10 else 0
        ch5=(h1["close"].iloc[i]/h1["close"].iloc[i-5]-1)*100 if i>=5 else 0
        ch20=(h1["close"].iloc[i]/h1["close"].iloc[i-20]-1)*100 if i>=20 else 0
        a50=1 if h1["close"].iloc[i]>h1["ema50"].iloc[i] else 0
        a200=1 if h1["close"].iloc[i]>h1["ema200"].iloc[i] else 0
        
        state = np.array([gap, br, rsi, atr_r, atr_p, c_ema50, c_ema200, ema50_slope, ch5, ch20, a50, a200,
                          h1["datetime"].iloc[i].hour, h1["datetime"].iloc[i].dayofweek], dtype=np.float64)
        
        # Simulate trade to get P&L
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
        
        he=ep; pnl=None
        for j in range(r+1,len(m5)):
            ca=atr5[j]
            if pd.isna(ca): continue
            if hi[j]>he: he=hi[j]
            if cl[j]<he-ch*ca:
                pnl=(cl[j]-ep)*bl*1-CHG*1
                break
        
        if pnl is None: continue
        
        all_states.append(state)
        all_pnls.append(pnl)

X_states = np.array(all_states)
y_pnls = np.array(all_pnls)
y_win = (y_pnls > 0).astype(int)

print(f"Total engulfing signals: {len(X_states)}")
print(f"Win rate: {y_win.mean()*100:.1f}%")
print(f"Avg P&L: Rs{y_pnls.mean():+,.0f}")

# ── RL ENVIRONMENT ──
class EngulfingEnv:
    """Environment: sequential engulfing signals, agent decides enter/skip"""
    def __init__(self, states, pnls, wins):
        self.states = states
        self.pnls = pnls
        self.wins = wins
        self.n = len(states)
        self.idx = 0
    
    def reset(self):
        self.idx = 0
        return self.states[0] if self.n > 0 else None
    
    def step(self, action):
        """action: 1=enter, 0=skip. Returns: next_state, reward, done"""
        if action == 1:  # Enter
            reward = self.pnls[self.idx]
        else:  # Skip
            reward = 0
        
        self.idx += 1
        done = self.idx >= self.n
        next_state = self.states[min(self.idx, self.n-1)] if not done else None
        return next_state, reward, done

# ── TRAIN RL AGENT (DQN-style with experience replay) ──
print(f"\n{'='*60}")
print("TRAINING RL AGENT (Q-Learning + Neural Network)")
print(f"{'='*60}")

# Chronological split
split_idx = int(len(X_states) * 0.7)
X_train, X_test = X_states[:split_idx], X_states[split_idx:]
pnls_train, pnls_test = y_pnls[:split_idx], y_pnls[split_idx:]
wins_train, wins_test = y_win[:split_idx], y_win[split_idx:]

# Normalize
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
all_norm = scaler.transform(X_states)

# Train Q-network: MLP that predicts expected future reward for each action
# Action 0 (skip) always gives 0 reward
# Action 1 (enter) gives trade P&L
# Q(s, a=0) = 0 (always)
# Q(s, a=1) = expected P&L given state

print("\nPhase 1: Train expected P&L predictor (Q-value for action=1)...")
q_net = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                     solver="adam", alpha=0.001, max_iter=500,
                     random_state=RS, early_stopping=True, validation_fraction=0.15)
q_net.fit(X_train_norm, pnls_train)

# Evaluate Q-network
q_preds_train = q_net.predict(X_train_norm)
q_preds_test = q_net.predict(X_test_norm)

# Policy: enter if Q(s, a=1) > 0 (expected P&L positive)
train_pred_enter = (q_preds_train > 0).astype(int)
test_pred_enter = (q_preds_test > 0).astype(int)

# Results on train
train_enter_idx = train_pred_enter == 1
train_skip_idx = train_pred_enter == 0
test_enter_idx = test_pred_enter == 1
test_skip_idx = test_pred_enter == 0

def print_results(name, enter_mask, pnls_arr, states_arr):
    n = len(pnls_arr)
    n_enter = enter_mask.sum()
    n_skip = n - n_enter
    
    if n_enter > 0:
        net_enter = pnls_arr[enter_mask].sum()
        wr_enter = (pnls_arr[enter_mask] > 0).mean() * 100
        aw_enter = pnls_arr[enter_mask][pnls_arr[enter_mask] > 0].mean() if (pnls_arr[enter_mask] > 0).sum() > 0 else 0
        al_enter = pnls_arr[enter_mask][pnls_arr[enter_mask] < 0].mean() if (pnls_arr[enter_mask] < 0).sum() > 0 else 0
        pf_enter = (pnls_arr[enter_mask][pnls_arr[enter_mask] > 0].sum() / 
                    abs(pnls_arr[enter_mask][pnls_arr[enter_mask] < 0].sum())) if (pnls_arr[enter_mask] < 0).sum() != 0 else 99
    else:
        net_enter = 0; wr_enter = 0; aw_enter = 0; al_enter = 0; pf_enter = 0
    
    net_all = pnls_arr.sum()
    wr_all = (pnls_arr > 0).mean() * 100
    pf_all = (pnls_arr[pnls_arr > 0].sum() / abs(pnls_arr[pnls_arr < 0].sum())) if (pnls_arr < 0).sum() != 0 else 99
    
    print(f"\n  {name}:")
    print(f"  {'':20s} {'All':>10s} {'Enter':>10s} {'Skip':>10s}")
    print(f"  {'Trades':20s} {n:>10d} {n_enter:>10d} {n_skip:>10d}")
    print(f"  {'Net P&L':20s} Rs{net_all:>+7,.0f} Rs{net_enter:>+7,.0f} Rs{pnls_arr[~enter_mask].sum() if n_skip>0 else 0:>+7,.0f}")
    print(f"  {'WR%':20s} {wr_all:>9.1f}% {wr_enter:>9.1f}% -")
    print(f"  {'PF':20s} {pf_all:>9.2f} {pf_enter:>9.2f} -")
    
    if n_enter > 0:
        improvement = (net_enter / net_all - 1) * 100 if net_all != 0 else 0
        print(f"  {'Improvement vs All':20s} {'-':>10s} {improvement:>+9.1f}% {'-':>10s}")

print_results("TRAIN", train_enter_idx, pnls_train, X_train_norm)
print_results("TEST", test_enter_idx, pnls_test, X_test_norm)

# ── PHASE 2: EXPERIENCE REPLAY DQN ──
print(f"\n{'='*60}")
print("Phase 2: Iterative DQN with Experience Replay")
print(f"{'='*60}")

class DQNAgent:
    def __init__(self, state_dim, hidden=(128, 64)):
        self.q_net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                   solver="adam", alpha=0.001, max_iter=200,
                                   random_state=RS, warm_start=True)
        self.replay_buffer = deque(maxlen=5000)
        self.state_dim = state_dim
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.fitted = False
        self.scaler = StandardScaler()
    
    def get_q_values(self, states):
        """Get Q(s, a=1) for each state"""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        s_norm = self.scaler.transform(states)
        return self.q_net.predict(s_norm)
    
    def act(self, state, training=True):
        """Epsilon-greedy: 1=enter, 0=skip"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        q_val = self.get_q_values(state.reshape(1, -1))[0]
        return 1 if q_val > 0 else 0
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        
        states = np.array([s for s, a, r, ns, d in batch])
        targets = []
        
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                ns = next_state.reshape(1, -1)
                next_q = 0
                if self.fitted:
                    try:
                        next_q = self.get_q_values(ns)[0]
                        next_q = max(0, next_q)  # only action=1 matters
                    except:
                        next_q = 0
                target = reward + self.gamma * next_q
            targets.append(target)
        
        targets = np.array(targets)
        
        # Normalize and fit
        if not self.fitted:
            s_norm = self.scaler.fit_transform(states)
            self.fitted = True
        else:
            s_norm = self.scaler.transform(states)
        
        self.q_net.fit(s_norm, targets)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train DQN
print("Training DQN agent with experience replay...")
dqn = DQNAgent(X_train_norm.shape[1])
dqn.q_net.fit(X_train_norm[:10], pnls_train[:10])  # initialize
dqn.scaler.fit(X_train_norm)
dqn.fitted = True

n_episodes = 50
batch_size = 64

for ep in range(n_episodes):
    env = EngulfingEnv(X_train_norm, pnls_train, wins_train)
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = dqn.act(state, training=True)
        next_state, reward, done = env.step(action)
        dqn.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    dqn.replay(batch_size)
    
    if (ep + 1) % 10 == 0:
        # Evaluate
        env_eval = EngulfingEnv(X_train_norm, pnls_train, wins_train)
        s = env_eval.reset()
        d = False
        total_enter = 0; total_pnl = 0
        while not d:
            a = dqn.act(s, training=False)
            ns, r, d = env_eval.step(a)
            total_enter += a
            total_pnl += r
            s = ns if ns is not None else s
        print(f"  Episode {ep+1}: epsilon={dqn.epsilon:.3f}, "
              f"entered={total_enter}, pnl={total_pnl:+,.0f}, "
              f"replay={len(dqn.replay_buffer)}")

# Evaluate final DQN on test set
print(f"\n{'='*60}")
print("FINAL DQN EVALUATION (TEST SET)")
print(f"{'='*60}")

env_test = EngulfingEnv(X_test_norm, pnls_test, wins_test)
s = env_test.reset()
d = False
dqn_actions = []
while not d:
    a = dqn.act(s, training=False)
    ns, r, d = env_test.step(a)
    dqn_actions.append(a)
    s = ns if ns is not None else s

dqn_actions = np.array(dqn_actions)
dqn_enter = dqn_actions == 1
dqn_skip = dqn_actions == 0

n_all = len(pnls_test)
n_enter = dqn_enter.sum()

if n_enter > 0:
    net_enter = pnls_test[dqn_enter].sum()
    wr_enter = (pnls_test[dqn_enter] > 0).mean() * 100
    aw_enter = pnls_test[dqn_enter][pnls_test[dqn_enter] > 0].mean() if (pnls_test[dqn_enter] > 0).sum() > 0 else 0
    al_enter = pnls_test[dqn_enter][pnls_test[dqn_enter] < 0].mean() if (pnls_test[dqn_enter] < 0).sum() > 0 else 0
    pf_enter = (pnls_test[dqn_enter][pnls_test[dqn_enter] > 0].sum() / 
                abs(pnls_test[dqn_enter][pnls_test[dqn_enter] < 0].sum())) if (pnls_test[dqn_enter] < 0).sum() != 0 else 99
else:
    net_enter = 0; wr_enter = 0; pf_enter = 0

net_all = pnls_test.sum()
wr_all = wins_test.mean() * 100
pf_all = (pnls_test[pnls_test > 0].sum() / abs(pnls_test[pnls_test < 0].sum())) if (pnls_test < 0).sum() != 0 else 99

print(f"\n  {'':20s} {'All Trades':>12s} {'DQN Enter':>12s} {'DQN Skip':>12s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")  
print(f"  {'Trades':20s} {n_all:>12d} {n_enter:>12d} {n_all-n_enter:>12d}")
print(f"  {'Net P&L':20s} Rs{net_all:>+9,.0f} Rs{net_enter:>+9,.0f} Rs{pnls_test[~dqn_enter].sum() if n_all-n_enter>0 else 0:>+9,.0f}")
print(f"  {'WR%':20s} {wr_all:>11.1f}% {wr_enter:>11.1f}% {'-':>12s}")
print(f"  {'PF':20s} {pf_all:>11.2f} {pf_enter:>11.2f} {'-':>12s}")

improvement = (net_enter / net_all - 1) * 100 if net_all != 0 else 0
print(f"  {'Improvement':20s} {'-':>12s} {improvement:>+10.1f}% {'-':>12s}")

# ── PHASE 3: RL with Skip Filter (combined) ──  
print(f"\n{'='*60}")
print("Phase 3: DQN + Skip-after-2-Losses combined")
print(f"{'='*60}")

class DQNSkipEnv:
    """Environment with skip filter built in"""
    def __init__(self, states, pnls, wins):
        self.states = states
        self.pnls = pnls
        self.wins = wins
        self.n = len(states)
        self.idx = 0
        self.consec_loss = 0
    
    def reset(self):
        self.idx = 0
        self.consec_loss = 0
        return self.states[0] if self.n > 0 else None
    
    def step(self, action):
        """action: 1=enter (if permitted by skip filter), 0=skip"""
        # Skip filter: after 2 consecutive losses, force skip
        if self.consec_loss >= 2:
            action = 0
            self.consec_loss = 0
        
        if action == 1:
            reward = self.pnls[self.idx]
            if self.pnls[self.idx] <= 0:
                self.consec_loss += 1
            else:
                self.consec_loss = 0
        else:
            reward = 0
        
        self.idx += 1
        done = self.idx >= self.n
        next_state = self.states[min(self.idx, self.n-1)] if not done else None
        return next_state, reward, done

# Train DQN with skip environment
print("Training DQN with integrated skip-after-2-losses...")
dqn2 = DQNAgent(X_train_norm.shape[1])
dqn2.q_net.fit(X_train_norm[:10], pnls_train[:10])
dqn2.scaler.fit(X_train_norm)
dqn2.fitted = True

for ep in range(30):
    env = DQNSkipEnv(X_train_norm, pnls_train, wins_train)
    state = env.reset()
    done = False
    while not done:
        action = dqn2.act(state, training=True)
        next_state, reward, done = env.step(action)
        dqn2.remember(state, action, reward, next_state, done)
        state = next_state
    dqn2.replay(64)

# Evaluate on test
env_te = DQNSkipEnv(X_test_norm, pnls_test, wins_test)
s = env_te.reset()
d = False
dqn2_actions = []
while not d:
    a = dqn2.act(s, training=False)
    ns, r, d = env_te.step(a)
    dqn2_actions.append(a)
    s = ns if ns is not None else s

dqn2_actions = np.array(dqn2_actions)
dqn2_enter = dqn2_actions == 1

if dqn2_enter.sum() > 0:
    net_e2 = pnls_test[dqn2_enter].sum()
    wr_e2 = (pnls_test[dqn2_enter] > 0).mean() * 100
    pf_e2 = (pnls_test[dqn2_enter][pnls_test[dqn2_enter] > 0].sum() / 
             abs(pnls_test[dqn2_enter][pnls_test[dqn2_enter] < 0].sum())) if (pnls_test[dqn2_enter] < 0).sum() != 0 else 99
    imp2 = (net_e2 / net_all - 1) * 100 if net_all != 0 else 0
else:
    net_e2 = 0; wr_e2 = 0; pf_e2 = 0; imp2 = 0

print(f"\n  DQN+Skip: Entered={dqn2_enter.sum()}, Net=Rs{net_e2:+,.0f}, WR={wr_e2:.1f}%, PF={pf_e2:.2f}, vsAll={imp2:+.1f}%")

# ── FINAL COMPARISON ──
print(f"\n{'='*70}")
print("FINAL COMPARISON — TEST SET")
print(f"{'='*70}")
print(f"{'Strategy':25s}  {'Trades':>8s}  {'Net P&L':>12s}  {'WR%':>6s}  {'PF':>6s}  {'vsAll':>8s}")
print(f"{'-'*70}")
print(f"{'All Trades':25s}  {n_all:>8d}  Rs{net_all:>+9,.0f}  {wr_all:>5.1f}%  {pf_all:>5.2f}  {'-':>8s}")

# Supervised ML baseline
print(f"{'ML Predict Enter':25s}  {test_enter_idx.sum():>8d}  Rs{pnls_test[test_enter_idx].sum():>+9,.0f}  "
      f"{(pnls_test[test_enter_idx]>0).mean()*100 if test_enter_idx.sum()>0 else 0:>5.1f}%  "
      f"{'?':>6s}  {((pnls_test[test_enter_idx].sum()/net_all)-1)*100 if net_all!=0 and test_enter_idx.sum()>0 else 0:>+7.1f}%")

print(f"{'DQN Enter':25s}  {n_enter:>8d}  Rs{net_enter:>+9,.0f}  {wr_enter:>5.1f}%  {pf_enter:>5.2f}  {improvement:>+7.1f}%")
print(f"{'DQN+Skip':25s}  {dqn2_enter.sum():>8d}  Rs{net_e2:>+9,.0f}  {wr_e2:>5.1f}%  {pf_e2:>5.2f}  {imp2:>+7.1f}%")

# ── ANALYSIS ──  
print(f"\n{'='*70}")
print("ANALYSIS: What RL learned vs What Skip Filter Does")
print(f"{'='*70}")

# Compare RL decisions with actual outcomes
# For each state in test, get Q-value
q_test = dqn.get_q_values(X_test_norm)
q_enter = (q_test > 0).astype(int)

# What if we just take all trades? = baseline
# What does Q-value distribution look like?
q_winners = q_test[wins_test == 1]
q_losers = q_test[wins_test == 0]

print(f"\n  Q-value stats:")
print(f"  Winners: mean={q_winners.mean():+.1f}, median={np.median(q_winners):+.1f}, "
      f"min={q_winners.min():+.1f}, max={q_winners.max():+.1f}")
print(f"  Losers:  mean={q_losers.mean():+.1f}, median={np.median(q_losers):+.1f}, "
      f"min={q_losers.min():+.1f}, max={q_losers.max():+.1f}")
print(f"  % Winners with Q>0: {(q_winners>0).mean()*100:.1f}%")
print(f"  % Losers with Q>0:  {(q_losers>0).mean()*100:.1f}%")

# Upper bound: what if we only took profitable trades?
print(f"\n  Upper bound (oracle):")
oracle_pnl = pnls_test[pnls_test > 0].sum()
print(f"  Only profitable trades: Rs{oracle_pnl:+,.0f} ({len(pnls_test[pnls_test>0])} of {n_all})")
print(f"  Current strategy captures: {net_all/oracle_pnl*100:.1f}% of oracle profit")

print(f"\nDONE")
