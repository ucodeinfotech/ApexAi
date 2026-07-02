"""
================================================================================
 HIGH GAINER PREDICTOR — Full ML Lifecycle
 Target: Next-day open-to-close return > 2% (binary classification)
 Universe: 475+ Indian stocks, daily frequency 2017-2026
================================================================================

Pipeline:
  1. Problem Definition & Setup
  2. Data Collection & Integration
  3. Exploratory Data Analysis
  4. Feature Engineering & Selection
  5. Model Development (multiple models + tuning)
  6. Walkforward Evaluation
  7. Performance Analysis
  8. Error Analysis
  9. Model Interpretation (SHAP)
 10. Final Model & Recommendations
================================================================================
"""
import duckdb, pandas as pd, numpy as np, warnings, pickle, time, json, os
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'high_gainer_full_ml'
OUT.mkdir(exist_ok=True)
REPORTS = OUT / 'reports'
REPORTS.mkdir(exist_ok=True)
CKPT = OUT / 'checkpoint.pkl'
t0 = time.time()

SKIP_COMPUTE = False  # always run full pipeline; checkpoint is for crash recovery

# ─── TRADE COST CONSTANTS ───
STT=0.001; BRK=0.0003; MIN_BRK=20; EXCH=0.0000345; SEBI=1e-6; GST=0.18; STAMP=3e-5; SLIP=0.0005
TOTAL_POS=110000

con = duckdb.connect(str(DB), read_only=True)

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 1: PROBLEM DEFINITION & SETUP')
print(f'{"="*70}')
# ─── Problem Statement ───
print(f'''
  PROBLEM: Binary classification
    Input:  Daily features for stock i on day T
            (price, volume, technicals, intraday metrics, market regime)
    Output: Probability that stock i's next-day open-to-close return > 2%
    Target: gainer_flag = 1 if (close_{{T+1}} / open_{{T+1}} - 1) > 0.02 else 0

  WHY 2%?
    - 90th percentile of daily ret_1d ~ 2.71% (from feature_store)
    - Represents a meaningful, tradeable move
    - Only 12.4% of stock-days meet this criterion (imbalanced)

  EVALUATION METRICS:
    - ROC-AUC (ranking quality)
    - Average Precision (precision across thresholds)
    - Precision@K (practical: how many of top 10 picks actually gain >2%)
    - Hit rate (fraction of all gainers captured in top K)
    - Profitability simulation (what if we bought top K each day)
''')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 2: DATA COLLECTION & INTEGRATION')
print(f'{"="*70}')


# ─── 2a: Daily Feature Store ───
print('\n[2a] Loading daily feature store...')
fs = con.execute("SELECT * FROM feature_store WHERE timeframe='1day' ORDER BY symbol, datetime").fetchdf()
fs['datetime'] = pd.to_datetime(fs['datetime'])
fs['date'] = fs['datetime'].dt.normalize()
print(f'  Rows: {len(fs):,} | Symbols: {fs["symbol"].nunique():,} | Date range: {fs["date"].min().date()} to {fs["date"].max().date()}')

# ─── 2b: 60min Intraday data ───
print('\n[2b] Loading raw 60min data for intraday features...')
m60 = con.execute("""
    SELECT symbol, datetime, datetime::DATE as date, close, high, low, volume
    FROM raw_market WHERE timeframe='60min' ORDER BY symbol, datetime
""").fetchdf()
m60['datetime'] = pd.to_datetime(m60['datetime'])
m60['ret_60'] = m60.groupby('symbol')['close'].transform(lambda x: x.pct_change() * 100)
m60['range_60'] = (m60['high'] - m60['low']) / m60['close'] * 100

# Aggregate 60min bars to daily intraday features
idf = m60.groupby(['symbol', 'date']).agg(
    intra_morn_ret=('ret_60', 'first'),
    intra_close_ret=('ret_60', 'last'),
    intra_max_gain=('ret_60', 'max'),
    intra_max_loss=('ret_60', 'min'),
    intra_volatility=('range_60', 'mean'),
    intra_range_sum=('range_60', 'sum'),
    intra_vol_sum=('volume', 'sum'),
    intra_n_bars=('close', 'count'),
).reset_index()
# Last hour return alias
idf['intra_last_ret'] = idf['intra_close_ret']
# Momentum: close vs open of day from 60min perspective
idf['intra_trend'] = idf['intra_close_ret'] - idf['intra_morn_ret']
idf = idf.fillna(0)
print(f'  Rows: {len(idf):,} | Symbols: {idf["symbol"].nunique():,}')

# ─── 2c: VIX & Market Regime ───
print('\n[2c] Loading VIX and market regime...')
vix = con.execute("SELECT datetime::DATE as date, vix_close, vix_change, vix_ratio_5, vix_zscore_20 FROM vix_data ORDER BY datetime").fetchdf()
vix['date'] = pd.to_datetime(vix['date'])

regimes = con.execute("SELECT datetime::DATE as date, regime_label, volatility_regime FROM market_regimes WHERE timeframe='1day' ORDER BY datetime").fetchdf()
regimes['date'] = pd.to_datetime(regimes['date'])
regimes['is_bull'] = (regimes['regime_label'] == 'bull').astype(float)
regimes['is_bear'] = (regimes['regime_label'] == 'bear').astype(float)
regimes['is_high_vol'] = (regimes['volatility_regime'] == 'high_vol').astype(float)

# ─── 2d: Delivery Data ───
print('\n[2d] Loading delivery data...')
delivery = con.execute("SELECT symbol, date, delivery_pct FROM delivery_data ORDER BY symbol, date").fetchdf()
delivery['date'] = pd.to_datetime(delivery['date'])
delivery['del_ma5'] = delivery.groupby('symbol')['delivery_pct'].transform(lambda x: x.rolling(5, min_periods=1).mean())
delivery['del_high'] = (delivery['delivery_pct'] > 75).astype(float)
print(f'  Rows: {len(delivery):,} | Symbols: {delivery["symbol"].nunique():,}')

# ─── 2e: Merge everything into master dataset ───
print('\n[2e] Merging all data sources...')

# Register all tables in DuckDB for efficient joining
con.register('fs_tbl', fs)
con.register('idf_tbl', idf)
con.register('vix_tbl', vix)
con.register('reg_tbl', regimes)
con.register('del_tbl', delivery)

df = con.execute("""
    SELECT fs_tbl.*,
           idf_tbl.intra_morn_ret, idf_tbl.intra_close_ret, idf_tbl.intra_max_gain,
           idf_tbl.intra_max_loss, idf_tbl.intra_volatility, idf_tbl.intra_range_sum,
           idf_tbl.intra_vol_sum, idf_tbl.intra_last_ret, idf_tbl.intra_trend,
           vix_tbl.vix_close, vix_tbl.vix_change, vix_tbl.vix_ratio_5, vix_tbl.vix_zscore_20,
           reg_tbl.is_bull, reg_tbl.is_bear, reg_tbl.is_high_vol,
           del_tbl.delivery_pct, del_tbl.del_ma5, del_tbl.del_high
    FROM fs_tbl
    LEFT JOIN idf_tbl ON fs_tbl.symbol = idf_tbl.symbol AND fs_tbl.date = idf_tbl.date
    LEFT JOIN vix_tbl ON fs_tbl.date = vix_tbl.date
    LEFT JOIN reg_tbl ON fs_tbl.date = reg_tbl.date
    LEFT JOIN del_tbl ON fs_tbl.symbol = del_tbl.symbol AND fs_tbl.date = del_tbl.date
    ORDER BY fs_tbl.symbol, fs_tbl.datetime
""").fetchdf()

for tbl in ['fs_tbl', 'idf_tbl', 'vix_tbl', 'reg_tbl', 'del_tbl']:
    con.unregister(tbl)

# Penny filter (avg close < 50 over 2024+)
penny = con.execute("""
    SELECT symbol, AVG(close) as avg_c FROM feature_store
    WHERE timeframe='1day' AND datetime >= '2024-01-01' GROUP BY symbol
""").fetchdf()
penny_syms = set(penny[penny['avg_c'] < 50]['symbol'])
df = df[~df['symbol'].isin(penny_syms)].copy()

# Fill NaN in all float columns
for c in df.columns:
    if df[c].dtype in ('float64', 'float32'):
        df[c] = df[c].fillna(0).astype(np.float32)

print(f'  Final shape: {df.shape} | Symbols: {df["symbol"].nunique():,}')
print(f'  Columns: {len(df.columns)} ({sum(1 for c in df.columns if c.startswith("intra_"))} intraday, {sum(1 for c in df.columns if c.startswith("vix_"))} vix, {sum(1 for c in df.columns if c.startswith("del"))} delivery, {sum(1 for c in df.columns if c.startswith("is_"))} regime)')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 3: EXPLORATORY DATA ANALYSIS')
print(f'{"="*70}')

# ─── 3a: Target Distribution ───
print('\n[3a] Target distribution (next-day open-close > 2%)...')
df['next_close'] = df.groupby('symbol')['close'].shift(-1)
df['next_open'] = df.groupby('symbol')['open'].shift(-1)
df['target_ret'] = df['next_close'] / df['next_open'] - 1
df = df.dropna(subset=['target_ret']).copy()
df['target'] = (df['target_ret'] > 0.02).astype(int)
pos_rate = df['target'].mean()
n_pos, n_total = df['target'].sum(), len(df)
print(f'  Positive class (>2% gainers): {n_pos:,}/{n_total:,} = {pos_rate:.1%}')
print(f'  Negative class: {n_total - n_pos:,} = {(1-pos_rate):.1%}')
print(f'  Imbalance ratio: {(1-pos_rate)/max(pos_rate,0.001):.1f}:1')

# Target return distribution
ret_percentiles = np.percentile(df['target_ret'].values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
print(f'\n  Target return percentiles:')
for p, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], ret_percentiles):
    print(f'    {p:>2d}th: {v*100:+.2f}%')

# ─── 3b: Target by year ───
print(f'\n[3b] Target distribution by year:')
df['year'] = pd.to_datetime(df['datetime']).dt.year
yearly_pos = df.groupby('year').agg(
    total=('target', 'count'), pos=('target', 'sum'),
    rate=('target', 'mean')).reset_index()
for _, r in yearly_pos.iterrows():
    print(f'  {int(r["year"]):>4d}: {r["pos"]:>6,}/{r["total"]:>6,} = {r["rate"]*100:>5.1f}%')

# ─── 3c: Missing data analysis ───
print(f'\n[3c] Missing data analysis (columns with >0% zeros caused by fillna):')
exclude_cols = {'symbol', 'datetime', 'date', 'next_close', 'next_open', 'target_ret', 'target', 'year', 'open', 'high', 'low', 'close', 'volume'}
feat_candidates = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

# Check which features actually have non-zero variance
zero_var = [c for c in feat_candidates if df[c].std() < 1e-10]
if zero_var:
    print(f'  Zero-variance columns (dropping): {zero_var}')

# ─── 3d: Feature-target correlation ───
print(f'\n[3d] Feature-target correlation (top 20 by absolute correlation):')
corrs = df[feat_candidates + ['target']].corr()['target'].drop('target').abs().sort_values(ascending=False).head(20)
for c, v in corrs.items():
    direct_corr = df[c].corr(df['target'])
    print(f'  {c:<30s} |corr|={v:.4f}  direct={direct_corr:+.4f}')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 4: FEATURE ENGINEERING & SELECTION')
print(f'{"="*70}')

# ─── 4a: Feature engineering ───
print('\n[4a] Creating engineered features...')

# Interaction: volatility * volume (explosive moves)
df['vol_x_vol'] = df['atr_14'] * df['vol_ratio_10']
df['range_x_vol'] = df['range_5'] * df['vol_ratio_10']

# Momentum + volatility (breakout potential)
df['mom_x_vol'] = df['ret_1d'] * df['hv_20']

# Distance from moving averages (oversold/overbought)
df['dist_sma_5'] = (df['close'] / df['sma_5'] - 1) * 100
df['dist_sma_20'] = (df['close'] / df['sma_20'] - 1) * 100

# Intraday + daily combined
df['intra_x_daily_vol'] = df['intra_volatility'] * df['range_5']
df['intra_morn_signal'] = df['intra_morn_ret'] * (df['intra_volatility'] > df['intra_volatility'].median()).astype(float)

# VIX interaction with stock features
df['vix_x_vol'] = df['vix_close'] * df['hv_20'] / 100

# Delivery interaction
if 'delivery_pct' in df.columns:
    df['delivery_x_ret'] = df['delivery_pct'] * df['ret_1d']
    df['delivery_surge'] = (df['delivery_pct'] > 70).astype(float) * df['ret_1d']

print(f'  Created {len(df.columns) - len(feat_candidates) - len(exclude_cols)} new features')

# ─── 4b: Feature selection ───
print('\n[4b] Selecting final feature set...')
all_feats = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

# Remove zero-variance
all_feats = [c for c in all_feats if df[c].std() > 1e-10]

# Remove highly correlated features (|rho| > 0.95)
corr_matrix = df[all_feats].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
all_feats = [c for c in all_feats if c not in high_corr]
print(f'  Removed {len(high_corr)} highly correlated features')
print(f'  Final feature count: {len(all_feats)}')

# Feature categories
cats = {
    'Price/Momentum': [c for c in all_feats if any(x in c for x in ['sma_', 'ema_', 'macd_', 'adx', 'rsi_14', 'plus_di', 'minus_di', 'stoch_', 'williams_r', 'mfi', 'uo', 'cci', 'trix', 'roc_', 'ret_1d', 'log_ret', 'close_vs_', 'swing_'])]}
cats['Volatility'] = [c for c in all_feats if any(x in c for x in ['atr_14', 'hv_20', 'bb_', 'kc_', 'dc_', 'range_', 'zscore_', 'volatility'])]
cats['Volume'] = [c for c in all_feats if any(x in c for x in ['obv', 'cmf', 'eom', 'fi', 'vpt', 'vol_ratio', 'vol_sum'])]
cats['Intraday'] = [c for c in all_feats if c.startswith('intra_')]
cats['Market'] = [c for c in all_feats if any(x in c for x in ['vix_', 'is_bull', 'is_bear', 'is_high_vol'])]
cats['Delivery'] = [c for c in all_feats if c.startswith('del')]
cats['Interaction'] = [c for c in all_feats if c not in sum(cats.values(), [])]

for cat, feats in cats.items():
    print(f'  {cat}: {len(feats)} features')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 5: MODEL DEVELOPMENT')
print(f'{"="*70}')

# ─── 5a: Train/val/test split (time-aware) ───
print('\n[5a] Creating time-aware splits...')
years = sorted(df['year'].unique())
# Train: 2017-2021, Val: 2022-2023, Test: 2024-2026
train_mask = df['year'].isin(years[:5])
val_mask = df['year'].isin([2022, 2023])
test_mask = df['year'] >= 2024
print(f'  Train:   {train_mask.sum():>8,} rows ({df[train_mask]["year"].min()}-{df[train_mask]["year"].max()})')
print(f'  Val:     {val_mask.sum():>8,} rows ({df[val_mask]["year"].min()}-{df[val_mask]["year"].max()})')
print(f'  Test:    {test_mask.sum():>8,} rows ({df[test_mask]["year"].min()}-{df[test_mask]["year"].max()})')
print(f'  Pos rate: Train={df[train_mask]["target"].mean():.1%} Val={df[val_mask]["target"].mean():.1%} Test={df[test_mask]["target"].mean():.1%}')

# Prepare data
def prep_data(mask):
    d = df[mask].copy()
    d = d.dropna(subset=all_feats)
    return d[all_feats].values, d['target'].values, d

X_train, y_train, train_data = prep_data(train_mask)
X_val, y_val, val_data = prep_data(val_mask)
X_test, y_test, test_data = prep_data(test_mask)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

scale_pos = (1 - pos_rate) / max(pos_rate, 0.001)
print(f'  Scale pos weight: {scale_pos:.1f}')

# ─── 5b: Model comparison ───
print('\n[5b] Training and comparing multiple models...')
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

models = {}

# XGBoost
print('  Training XGBoost...')
xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
    gamma=0.1, reg_alpha=0.01, reg_lambda=1.0,
    scale_pos_weight=scale_pos,
    random_state=42, n_jobs=-1, verbosity=0, device='cuda', tree_method='hist')
xgb_model.fit(X_train_s, y_train,
              eval_set=[(X_val_s, y_val)], verbose=False)
models['XGBoost'] = xgb_model

# LightGBM
print('  Training LightGBM...')
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
    class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1,
    device='gpu' if False else 'cpu')
lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)])
models['LightGBM'] = lgb_model

# Random Forest
print('  Training Random Forest...')
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train_s, y_train)
models['RandomForest'] = rf_model

# Logistic Regression (baseline)
print('  Training Logistic Regression...')
lr_model = LogisticRegression(
    class_weight='balanced', C=1.0, max_iter=1000,
    random_state=42, n_jobs=-1)
lr_model.fit(X_train_s, y_train)
models['LogisticReg'] = lr_model

# ─── 5c: Compare models ───
print(f'\n[5c] Model comparison on validation set:')
results = []
for name, model in models.items():
    prob = model.predict_proba(X_val_s)[:, 1]
    auc = roc_auc_score(y_val, prob)
    ap = average_precision_score(y_val, prob)
    # Top-10 hit rate
    val_tmp = val_data.copy()
    val_tmp['prob'] = prob
    top10_hits = val_tmp.groupby('date').apply(
        lambda d: d.nlargest(10, 'prob')['target'].sum()).mean()
    results.append({'model': name, 'AUC': auc, 'AP': ap, 'top10_hits': top10_hits})
    print(f'  {name:<15s} AUC={auc:.4f}  AP={ap:.4f}  Top10_hits={top10_hits:.2f}/day')

results_df = pd.DataFrame(results)
best_model_name = results_df.sort_values('AUC', ascending=False).iloc[0]['model']
print(f'\n  Best model: {best_model_name}')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 6: HYPERPARAMETER TUNING')
print(f'{"="*70}')
print('\n[6a] Tuning XGBoost via grid search on validation set...')

# Parameter grid (reduced for speed: 4×3×2×2×2×2 = 192 trials)
import gc
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [3, 5],
    'reg_alpha': [0, 0.01],
}

best_auc = 0; best_params = None
n_trials = 0
for md in param_grid['max_depth']:
    for lr in param_grid['learning_rate']:
        for ss in param_grid['subsample']:
            for cs in param_grid['colsample_bytree']:
                for mcw in param_grid['min_child_weight']:
                    for ra in param_grid['reg_alpha']:
                        n_trials += 1
                        model = xgb.XGBClassifier(
                            n_estimators=500, max_depth=md, learning_rate=lr,
                            subsample=ss, colsample_bytree=cs, min_child_weight=mcw,
                            gamma=0.1, reg_alpha=ra, reg_lambda=1.0,
                            scale_pos_weight=scale_pos,
                            random_state=42, n_jobs=-1, verbosity=0,
                            device='cuda', tree_method='hist')
                        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
                        prob = model.predict_proba(X_val_s)[:, 1]
                        auc = roc_auc_score(y_val, prob)
                        if auc > best_auc:
                            best_auc = auc
                            best_params = {'max_depth': md, 'learning_rate': lr,
                                           'subsample': ss, 'colsample_bytree': cs,
                                           'min_child_weight': mcw, 'reg_alpha': ra}
                            best_model = model
                        print(f'  Trial {n_trials:>3d}: md={md} lr={lr} ss={ss} cs={cs} mcw={mcw} ra={ra} -> AUC={auc:.4f}{" *" if auc == best_auc else ""}')
                        del model; gc.collect()  # free GPU memory between trials

print(f'\n  Best params: {best_params}')
print(f'  Val AUC: {best_auc:.4f}')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 7: WALKFORWARD EVALUATION (OOS)')
print(f'{"="*70}')

# Use best model with best params
print(f'\n[7a] Training best model with walkforward (2018-2026)...')
windows = [(df['year'].isin(years[:i]), df['year'] == years[i]) for i in range(2, len(years))]

final_model_params = {
    'n_estimators': 500,
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'min_child_weight': best_params['min_child_weight'],
    'reg_alpha': best_params['reg_alpha'],
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'scale_pos_weight': scale_pos,
    'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    'device': 'cuda', 'tree_method': 'hist',
}

wf_results = []
for wi, (tr_mask, te_mask) in enumerate(windows):
    train = df[tr_mask].copy()
    test = df[te_mask].copy()
    if len(test) < 500:
        continue

    test_start = test['datetime'].min()
    train = train[train['datetime'] < test_start - pd.Timedelta(days=7)].copy()

    feat_use = [c for c in all_feats if train[c].notna().all() and train[c].std() > 1e-10]
    train = train.dropna(subset=feat_use)
    test = test.dropna(subset=feat_use)
    if len(train) < 500 or len(test) < 50:
        continue

    scaler_w = StandardScaler()
    X_tr = scaler_w.fit_transform(train[feat_use].values)
    X_te = scaler_w.transform(test[feat_use].values)
    y_tr = train['target'].values; y_te = test['target'].values

    model = xgb.XGBClassifier(**final_model_params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    prob = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, prob)
    ap = average_precision_score(y_te, prob)

    yr_str = test['year'].iloc[0]
    print(f'  [{wi+1:2d}/{len(windows)}] {yr_str}: AUC={auc:.4f} AP={ap:.4f} n={len(y_te):,} pos={y_te.mean():.1%}')

    for i in range(len(test)):
        wf_results.append({
            'dt': test['datetime'].iloc[i], 'sym': test['symbol'].iloc[i],
            'prob': prob[i], 'act': y_te[i], 'ret': test['target_ret'].iloc[i],
        })

# Save last model
best_final_model = model
final_feat_use = feat_use

wf_df = pd.DataFrame(wf_results)
wf_auc = roc_auc_score(wf_df['act'], wf_df['prob'])
wf_ap = average_precision_score(wf_df['act'], wf_df['prob'])
print(f'\n  Walkforward OVERALL: AUC={wf_auc:.4f}  AP={wf_ap:.4f}  Predictions={len(wf_df):,}')

# Save checkpoint
import copy
ckpt_data = dict(wf_df=wf_df, final_feat_use=final_feat_use, best_final_model=best_final_model,
                 scaler=scaler, df=df, feat_candidates=feat_candidates, all_feats=all_feats,
                 pos_rate=pos_rate, final_model_params=final_model_params,
                 wf_auc=wf_auc, wf_ap=wf_ap, X_test_s=X_test_s, test_data=test_data)
pickle.dump(ckpt_data, open(CKPT, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
del copy
print(f'  Checkpoint saved to {CKPT} ({os.path.getsize(CKPT)/1e6:.0f} MB)')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 8: PERFORMANCE ANALYSIS')
print(f'{"="*70}')

# ─── 8a: Top-K analysis ───
print(f'\n[8a] Top-K analysis (daily ranking):')
top10_precision = None; top10_hitrate = None
for k in [3, 5, 10, 20, 50]:
    topk_hits = wf_df.groupby('dt').apply(
        lambda d: d.nlargest(k, 'prob')['act'].sum())
    actual = wf_df.groupby('dt')['act'].sum()
    hit_rate = (topk_hits / actual.replace(0, np.nan)).dropna()
    precision_k = topk_hits.mean() / k
    if k == 10:
        top10_precision = precision_k
        top10_hitrate = hit_rate
    print(f'  Top {k:>3d}: precision={precision_k:.1%} captures={hit_rate.mean():.1%} of gainers ({topk_hits.mean():.1f}/day)')

# ─── 8b: Classification metrics ───
print(f'\n[8b] Classification at various thresholds:')
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for th in thresholds:
    pred = (wf_df['prob'] >= th).astype(int)
    prec = precision_score(wf_df['act'], pred, zero_division=0)
    rec = recall_score(wf_df['act'], pred, zero_division=0)
    n_pred = pred.sum()
    print(f'  Threshold={th:.1f}: Prec={prec:.1%} Rec={rec:.1%} Signals={n_pred:,}/day')

# ─── 8c: Yearly breakdown ───
print(f'\n[8c] Yearly performance:')
wf_df['year'] = pd.to_datetime(wf_df['dt']).dt.year
for yr in sorted(wf_df['year'].unique()):
    sub = wf_df[wf_df['year'] == yr]
    auc_y = roc_auc_score(sub['act'], sub['prob'])
    ap_y = average_precision_score(sub['act'], sub['prob'])
    top10 = sub.groupby('dt').apply(
        lambda d: d.nlargest(10, 'prob')['act'].sum()).mean()
    print(f'  {int(yr):>4d}: AUC={auc_y:.4f} AP={ap_y:.4f} Top10_hits={top10:.1f}/day  n={len(sub):,}')

# ─── 8d: Profitability simulation ───
print(f'\n[8d] Profitability simulation (buy top 5 each day):')
MULT = 100
def cost_rt(pos_size):
    if pos_size <= 0: return 1.0
    brk_side = max(BRK * pos_size, MIN_BRK) / pos_size; brk_total = brk_side * 2
    gst_base = brk_total + EXCH*2 + SEBI*2
    return brk_total + STT + EXCH*2 + SEBI*2 + STAMP + gst_base * GST + SLIP * 2

# Simulate: each day, buy top 5 by probability, hold for next-day open-to-close
sim_trades = []
for dt_uniq in sorted(wf_df['dt'].unique()):
    day = wf_df[wf_df['dt'] == dt_uniq]
    top5 = day.nlargest(5, 'prob')
    avg_ret = top5['ret'].mean()
    cost = cost_rt(TOTAL_POS) * 2 * MULT / 5  # enter + exit for each of 5 positions
    sim_trades.append({'dt': dt_uniq, 'ret': avg_ret, 'net': avg_ret - cost, 'n': len(top5)})

sim_df = pd.DataFrame(sim_trades)
# Monthly aggregation
sim_df['month'] = pd.PeriodIndex(pd.to_datetime(sim_df['dt']), freq='M')
monthly = sim_df.groupby('month').agg(
    n_days=('ret', 'count'),
    ret=('ret', lambda x: (1 + x/100).prod() - 1),
    net=('net', lambda x: (1 + x/100).prod() - 1),
).reset_index()
monthly['ret'] *= 100; monthly['net'] *= 100

print(f'  Trades: {len(sim_df):,} over {sim_df["month"].nunique()} months')
print(f'  Avg daily return: gross={sim_df["ret"].mean():+.4f}% net={sim_df["net"].mean():+.4f}%')
print(f'  Monthly win rate: {(monthly["net"]>0).mean():.0%}')
print(f'  Best month: {monthly.loc[monthly["net"].idxmax(), "month"]} ({monthly["net"].max():+.2f}%)')
print(f'  Worst month: {monthly.loc[monthly["net"].idxmin(), "month"]} ({monthly["net"].min():+.2f}%)')
print(f'  Total gross return: {(1+sim_df["ret"].mean()/100*len(sim_df)):.1%}')
print(f'  Total net return: {(1+sim_df["net"].mean()/100*len(sim_df)):.1%}')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 9: ERROR ANALYSIS')
print(f'{"="*70}')

# ─── 9a: Calibration ───
print('\n[9a] Probability calibration (expected vs actual gainer rate):')
wf_df['prob_bin'] = pd.cut(wf_df['prob'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
calibration = wf_df.groupby('prob_bin').agg(
    n=('act', 'count'), actual_rate=('act', 'mean')).reset_index()
calibration['expected_rate'] = calibration['prob_bin'].apply(lambda x: x.mid)
print(f'  {"Prob range":<12s} {"N":>6s} {"Expected":>9s} {"Actual":>9s} {"Bias":>9s}')
for _, r in calibration.iterrows():
    bias = r['actual_rate'] - r['expected_rate']
    print(f'  {str(r["prob_bin"]):<12s} {r["n"]:>6,} {r["expected_rate"]:>7.1%} {r["actual_rate"]:>7.1%} {bias:>+7.1%}')

# ─── 9b: Error analysis by feature ───
print(f'\n[9b] Error analysis by feature quartiles (top features):')
top_feats = pd.DataFrame({'feat': final_feat_use, 'imp': best_final_model.feature_importances_})
top_feats = top_feats.sort_values('imp', ascending=False).head(5)['feat'].tolist()
wf_df_feat = wf_df.merge(df[['datetime', 'symbol'] + top_feats], on=['datetime', 'symbol'], how='left')
for feat in top_feats:
    valid = wf_df_feat[feat].dropna()
    if len(valid) < 100:
        continue
    wf_df_feat['feat_bin'] = pd.qcut(wf_df_feat[feat].fillna(0), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    err = wf_df_feat.groupby('feat_bin').agg(
        n=('act', 'count'), pos_rate=('act', 'mean'),
        pred_prob=('prob', 'mean')).reset_index()
    print(f'  {feat}:')
    for _, r in err.iterrows():
        print(f'    {r["feat_bin"]:>4s}: n={r["n"]:>6,} actual={r["pos_rate"]:.1%} pred={r["pred_prob"]:.1%}')

# ─── 9c: Performance by market regime ───
print(f'\n[9c] Performance by market regime:')
wf_df_reg = wf_df.merge(df[['datetime', 'symbol', 'is_bull', 'is_bear']], on=['datetime', 'symbol'], how='left')
for regime, col in [('Bull', 'is_bull'), ('Bear', 'is_bear'), ('Sideways', None)]:
    if col:
        mask = wf_df_reg[col] > 0.5
    else:
        mask = (wf_df_reg['is_bull'] <= 0.5) & (wf_df_reg['is_bear'] <= 0.5)
    sub = wf_df_reg[mask]
    if len(sub) < 100:
        continue
    auc_r = roc_auc_score(sub['act'], sub['prob'])
    ap_r = average_precision_score(sub['act'], sub['prob'])
    print(f'  {regime:<10s}: AUC={auc_r:.4f} AP={ap_r:.4f} n={len(sub):,} pos={sub["act"].mean():.1%}')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 10: MODEL INTERPRETATION (SHAP)')
print(f'{"="*70}')

print('\n[10a] Computing SHAP values (sample of 10K rows)...')
import shap
shap_sample = X_test_s[np.random.choice(len(X_test_s), min(10000, len(X_test_s)), replace=False)]
explainer = shap.TreeExplainer(best_final_model)
shap_vals = explainer(shap_sample)

print(f'\n  Top 15 features by mean |SHAP|:')
shap_importance = np.abs(shap_vals.values).mean(axis=0)
shap_feat_order = np.argsort(shap_importance)[::-1][:15]
for i, idx in enumerate(shap_feat_order):
    print(f'  {i+1:>2d}. {final_feat_use[idx]:<30s} mean|SHAP|={shap_importance[idx]:.6f}')

# ─── SHAP dependence plot data (text summary) ───
print(f'\n  Top feature SHAP range (intra_volatility):')
feat_idx = final_feat_use.index('intra_volatility') if 'intra_volatility' in final_feat_use else 0
feat_vals = shap_sample[:, feat_idx]
feat_bins = pd.qcut(pd.Series(feat_vals), q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
mean_shap = pd.Series(shap_vals.values[:, feat_idx]).groupby(feat_bins).mean()
for b, v in mean_shap.items():
    print(f'    {str(b):>12s}: avg SHAP = {v:+.6f}')

# ════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f' PHASE 11: FINAL MODEL & RESULTS SUMMARY')
print(f'{"="*70}')

print(f'''
  FINAL MODEL: XGBoost Classifier (tuned)
  Best params: {json.dumps(best_params, indent=2)}

  WALKFORWARD PERFORMANCE:
    ROC-AUC:     {wf_auc:.4f}
    Avg Precision: {wf_ap:.4f}
    Predictions: {len(wf_df):,}
    Baseline (random): AUC=0.500 AP={pos_rate:.4f}

  TOP-10 DAILY PICKS:
    Precision:   {top10_precision:.1%} (when k=10)
    Gainers captured: {top10_hitrate.mean():.1%} of all daily gainers
    
  PROFITABILITY (Top 5, daily):
    Avg daily net: {sim_df['net'].mean():+.4f}%
    Monthly win rate: {(monthly['net']>0).mean():.0%}
    Total net return over period: {(1+sim_df['net'].mean()/100*len(sim_df)):.1%}

  TOP 10 FEATURES:
''')
fi = pd.DataFrame({'feature': final_feat_use, 'importance': best_final_model.feature_importances_})
fi = fi.sort_values('importance', ascending=False).head(10)
for _, r in fi.iterrows():
    print(f'    {r["feature"]:<30s} {r["importance"]:.4f}')

print(f'''
  SAVED OUTPUTS:
    - {OUT / "final_model.pkl"} (trained XGBoost model)
    - {OUT / "scaler.pkl"} (StandardScaler)
    - {OUT / "features.txt"} (feature list)
    - {OUT / "walkforward_results.pkl"} (all OOS predictions)
    - {OUT / "reports/"} (detailed reports)
  Total time: {time.time()-t0:.0f}s
''')

# ─── Save all outputs ───
pickle.dump(best_final_model, open(OUT / 'final_model.pkl', 'wb'))
pickle.dump(scaler, open(OUT / 'scaler.pkl', 'wb'))
with open(OUT / 'features.txt', 'w') as f:
    f.write('\n'.join(final_feat_use))
pickle.dump(wf_df, open(OUT / 'walkforward_results.pkl', 'wb'))

# Summary json
summary = {
    'auc': float(wf_auc), 'ap': float(wf_ap),
    'pos_rate': float(pos_rate), 'n_predictions': len(wf_df),
    'n_symbols': int(df['symbol'].nunique()),
    'n_features': len(final_feat_use),
    'best_params': best_params,
    'avg_daily_net': float(sim_df['net'].mean()),
    'monthly_win_rate': float((monthly['net']>0).mean()),
    'top10_precision': float(top10_precision),
    'total_time_s': time.time() - t0,
}
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)

con.close()
print(f'{"="*70}')
print(f' ML LIFECYCLE COMPLETE — Results saved to {OUT}')
print(f'{"="*70}')
