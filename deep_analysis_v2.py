# Deep Comprehensive Analysis v2 - Optimized
import duckdb, pandas as pd, numpy as np, time, warnings, json, gc
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
OUT = BASE / 'deep_analysis_report'
OUT.mkdir(exist_ok=True); (OUT/'charts').mkdir(exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)
t0 = time.time()

print('='*60)
print(' DEEP COMPREHENSIVE ANALYSIS v2')
print('='*60)

con = duckdb.connect(str(DB), read_only=True)

# ─── LOAD ───
print('\n[0] Loading data...')
fs = con.execute("""
    SELECT symbol, datetime::DATE as date, open, high, low, close, volume,
           ret_1d, range_5, hv_20, rsi_14, vol_ratio_5, bb_width, adx
    FROM feature_store WHERE timeframe='1day'
    ORDER BY symbol, date
""").fetchdf()
fs = fs.sort_values(['symbol','date'])
fs['next_close'] = fs.groupby('symbol')['close'].shift(-1)
fs['next_open'] = fs.groupby('symbol')['open'].shift(-1)
fs['target_ret'] = fs['next_close']/fs['next_open']-1
fs['target'] = (fs['target_ret']>0.02).astype(int)
fs = fs.dropna(subset=['target'])
print(f'  {len(fs):,} rows, {fs["symbol"].nunique()} symbols, {fs["target"].mean():.1%} pos rate')

daily = fs.groupby('date').agg(gainer_rate=('target','mean'), n_stocks=('symbol','count'),
    avg_ret=('ret_1d','mean'), avg_range=('range_5','mean'), avg_hv=('hv_20','mean')).reset_index()
daily = daily.sort_values('date'); daily['gainer_rate'] = daily['gainer_rate'].fillna(0)
print(f'  {len(daily)} trading days')

# ════════════════════════════════════════════════════════════════
# SECTION 1: TIME SERIES ANALYSIS
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}\n SECTION 1: TIME SERIES ANALYSIS\n{"="*60}')
ts_results = {}; t1 = time.time()

def compute_acf(series, nlags=40):
    n=len(series); m=np.mean(series); v=np.var(series,ddof=0)
    if v==0: return np.zeros(nlags+1)
    c0=np.sum((series-m)**2)/n; acf=np.ones(nlags+1)
    for k in range(1,nlags+1): acf[k]=np.sum((series[:-k]-m)*(series[k:]-m))/n/c0
    return acf

def compute_pacf(series, nlags=40):
    acf=compute_acf(series,nlags); pacf=np.zeros(nlags+1); pacf[0]=1.0
    for k in range(1,nlags+1):
        if k==1: pacf[k]=acf[1]
        else:
            phi=np.zeros(k+1); phi[1]=acf[1]
            for i in range(2,k+1):
                num=acf[i]-np.sum(phi[1:i]*acf[i-1:0:-1])
                denom=1-np.sum(phi[1:i]*acf[1:i])
                phi[i]=num/denom if abs(denom)>1e-10 else 0
                for j in range(1,i): phi[j]=phi[j]-phi[i]*phi[i-j]
            pacf[k]=phi[k]
    return pacf

# 1a. ACF/PACF
print('\n[1a] ACF/PACF...')
ts = daily['gainer_rate'].values
acf = compute_acf(ts,40); pacf = compute_pacf(ts,40)
se_95 = 1.96/np.sqrt(len(ts))
sig_acf = [i for i in range(1,41) if abs(acf[i])>se_95]
sig_pacf = [i for i in range(1,41) if abs(pacf[i])>se_95]

# ACF/PACF chart
fig, axes = plt.subplots(2,1,figsize=(14,8))
for ax,vals,title,color in zip(axes,[acf,pacf],['ACF - Daily Gainer Rate','PACF - Daily Gainer Rate'],['steelblue','darkorange']):
    ax.bar(range(1,41),vals[1:],color=color,width=0.6)
    ax.axhline(se_95,color='red',ls='--',alpha=0.5); ax.axhline(-se_95,color='red',ls='--',alpha=0.5)
    ax.axhline(0,color='black',lw=0.5); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.set_xlabel('Lag (days)'); ax.set_ylabel('Value'); ax.grid(True,alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'ts_acf_pacf.png',dpi=150,bbox_inches='tight'); plt.close()

acf_table = pd.DataFrame({'lag':range(1,41),'ACF':acf[1:],'PACF':pacf[1:],
    'sig_ACF':[abs(acf[i])>se_95 for i in range(1,41)],'sig_PACF':[abs(pacf[i])>se_95 for i in range(1,41)]})
acf_table.to_csv(OUT/'tables'/'acf_pacf.csv',index=False)
ts_results['acf'] = {'sig_acf_lags':sig_acf,'sig_pacf_lags':sig_pacf,
    'key_lags':{str(l):{'acf':float(acf[l]),'pacf':float(pacf[l])} for l in [1,2,3,4,5,10,16,17,18,20]}}
print(f'  ACF lags: {sig_acf[:10]}... PACF lags: {sig_pacf[:10]}...')

# 1b. Stationarity (market level only)
print('\n[1b] Stationarity...')
from statsmodels.tsa.stattools import adfuller
stat_results = []
for feat in ['gainer_rate','avg_ret','avg_hv','avg_range']:
    s = daily[feat].dropna().values
    adf_p = adfuller(s,maxlag=20)[1] if len(s)>100 else 1
    stat_results.append({'feature':feat,'adf_pval':float(adf_p),'stationary':adf_p<0.05})
pd.DataFrame(stat_results).to_csv(OUT/'tables'/'stationarity.csv',index=False)
ts_results['stationarity'] = {r['feature']:r['adf_pval'] for r in stat_results}

# 1c. Granger (limited to 100 symbols for speed)
print('\n[1c] Granger Causality (100 symbols)...')
from statsmodels.tsa.stattools import grangercausalitytests
gc_feats = ['ret_1d','range_5','hv_20','rsi_14','vol_ratio_5','bb_width','adx']
gc_data = []
for sym in fs['symbol'].unique()[:100]:
    sd = fs[fs['symbol']==sym].sort_values('date').dropna(subset=['target']+gc_feats)
    if len(sd)<100: continue
    for feat in gc_feats:
        try:
            g = grangercausalitytests(sd[[feat,'target']].values,maxlag=5,verbose=False)
            bl = min(range(1,6),key=lambda l:g[l][0]['ssr_chi2test'][1])
            gc_data.append({'symbol':sym,'feature':feat,'best_pval':float(g[bl][0]['ssr_chi2test'][1]),
                'best_lag':bl,'significant':g[bl][0]['ssr_chi2test'][1]<0.05})
        except: pass
gc_df = pd.DataFrame(gc_data)
gc_agg = gc_df.groupby('feature').agg(pct_sig=('significant','mean'),median_pval=('best_pval','median'),
    mean_lag=('best_lag','mean')).reset_index()
gc_agg.to_csv(OUT/'tables'/'granger_detailed.csv',index=False)
gc_df.to_csv(OUT/'tables'/'granger_per_symbol.csv',index=False)
ts_results['granger'] = {r['feature']:{'pct_sig':float(r['pct_sig']),'median_pval':float(r['median_pval']),
    'mean_lag':float(r['mean_lag'])} for _,r in gc_agg.iterrows()}
for _,r in gc_agg.iterrows():
    print(f'    {r["feature"]:<15s} {r["pct_sig"]:.0%} sig  p={r["median_pval"]:.2e}  lag={r["mean_lag"]:.1f}')

# 1d. Window optimization
print('\n[1d] Window optimization (100 symbols)...')
windows = [3,5,10,15,21,30,50]
win_data = []
for sym in fs['symbol'].unique()[:100]:
    sd = fs[fs['symbol']==sym].sort_values('date').dropna(subset=['target'])
    if len(sd)<200: continue
    for feat in ['ret_1d','range_5','hv_20','vol_ratio_5']:
        for w in windows:
            r = sd[feat].rolling(w,min_periods=max(3,w//3)).mean()
            c = r.corr(sd['target'])
            win_data.append({'symbol':sym,'feature':feat,'window':w,'corr':float(c) if not np.isnan(c) else 0})

wdf = pd.DataFrame(win_data)
wag = wdf.groupby(['feature','window']).agg(corr_mean=('corr','mean'),corr_std=('corr','std')).reset_index()
wag.to_csv(OUT/'tables'/'window_optimization.csv',index=False)

fig,ax=plt.subplots(figsize=(10,6))
for feat in ['ret_1d','range_5','hv_20','vol_ratio_5']:
    fw=wag[wag['feature']==feat]; ax.plot(fw['window'],fw['corr_mean'],marker='o',label=feat,lw=2)
ax.axhline(0,color='gray',ls='--',alpha=0.5); ax.set_xlabel('Window (days)'); ax.set_ylabel('Mean Correlation')
ax.set_title('Window Optimization',fontsize=13,fontweight='bold'); ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'ts_window_optimization.png',dpi=150,bbox_inches='tight'); plt.close()

opt_wins = {}
for feat in ['ret_1d','range_5','hv_20','vol_ratio_5']:
    fw=wag[wag['feature']==feat]
    b=fw.loc[fw['corr_mean'].abs().idxmax()]; opt_wins[feat]={'window':int(b['window']),'corr':float(b['corr_mean'])}
ts_results['optimal_windows']=opt_wins
print('  Optimal:',opt_wins)

# 1e. GMM Regimes
print('\n[1e] GMM Regimes...')
rf = daily[['avg_ret','avg_hv','avg_range']].dropna().values
rf_s = StandardScaler().fit_transform(rf)
bic_scores = []
for n in range(2,8):
    g=GaussianMixture(n_components=n,random_state=42,n_init=10,max_iter=500).fit(rf_s)
    bic_scores.append({'n':n,'bic':g.bic(rf_s)})
best_n = int(min(bic_scores,key=lambda x:x['bic'])['n'])
gmm = GaussianMixture(n_components=best_n,random_state=42,n_init=15,max_iter=500)
daily['regime'] = np.nan
daily.loc[daily[['avg_ret','avg_hv','avg_range']].dropna().index,'regime'] = gmm.fit_predict(rf_s)
daily = daily.dropna(subset=['regime']); daily['regime']=daily['regime'].astype(int)

rp = daily.groupby('regime').agg(gainer_rate=('gainer_rate','mean'),avg_ret=('avg_ret','mean'),
    avg_hv=('avg_hv','mean'),n_days=('date','count')).reset_index()

# BIC chart
fig,ax=plt.subplots(figsize=(10,6))
ax.plot([b['n'] for b in bic_scores],[b['bic'] for b in bic_scores],marker='o',lw=2,color='steelblue')
ax.axvline(best_n,color='red',ls='--',alpha=0.7,label=f'Optimal (n={best_n})')
ax.set_xlabel('Regimes'); ax.set_ylabel('BIC'); ax.set_title('GMM Regime Selection',fontsize=13,fontweight='bold')
ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'ts_gmm_bic.png',dpi=150,bbox_inches='tight'); plt.close()

# Transition matrix
n_reg=best_n; tm=np.zeros((n_reg,n_reg)); ra=daily['regime'].values
for t in range(1,len(ra)): tm[ra[t-1],ra[t]]+=1
tm=tm/(tm.sum(axis=1,keepdims=True)+1e-10)

fig,ax=plt.subplots(figsize=(9,8))
im=ax.imshow(tm,cmap='YlOrRd',aspect='auto',vmin=0,vmax=1)
for i in range(n_reg):
    for j in range(n_reg):
        ax.text(j,i,f'{tm[i,j]:.2f}',ha='center',va='center',fontsize=9,color='black' if tm[i,j]<0.5 else 'white')
ax.set_xticks(range(n_reg)); ax.set_yticks(range(n_reg))
ax.set_xlabel('To Regime'); ax.set_ylabel('From Regime')
ax.set_title(f'Regime Transition Matrix (n={best_n})',fontsize=13,fontweight='bold')
plt.colorbar(im,fraction=0.046); plt.tight_layout()
fig.savefig(OUT/'charts'/'ts_regime_transition.png',dpi=150,bbox_inches='tight'); plt.close()

daily[['date','avg_ret','avg_hv','avg_range','gainer_rate','n_stocks','regime']].to_csv(
    OUT/'tables'/'regime_data_full.csv',index=False)
rp.to_csv(OUT/'tables'/'regime_profiles.csv',index=False)
np.savetxt(OUT/'tables'/'transition_matrix.csv',tm,delimiter=',',fmt='%.4f')
ts_results['regimes']={'n_regimes':best_n,'profiles':{int(r['regime']):{'gainer_rate':float(r['gainer_rate']),
    'avg_ret':float(r['avg_ret']),'avg_hv':float(r['avg_hv']),'n_days':int(r['n_days'])} for _,r in rp.iterrows()}}
print(f'  Regimes: {best_n}')
for _,r in rp.iterrows(): print(f'    R{int(r["regime"])}: n={int(r["n_days"])} gainer={r["gainer_rate"]:.1%} ret={r["avg_ret"]:.2f}%')

# 1f. Seasonality
print('\n[1f] Seasonality...')
daily['dow']=pd.to_datetime(daily['date']).dt.dayofweek
daily['month']=pd.to_datetime(daily['date']).dt.month
daily['year']=pd.to_datetime(daily['date']).dt.year
dow_eff=daily.groupby('dow')['gainer_rate'].agg(['mean','std','count'])
mon_eff=daily.groupby('month')['gainer_rate'].agg(['mean','std','count'])
yr_eff=daily.groupby('year')['gainer_rate'].agg(['mean','std','count'])
dow_g=[daily[daily['dow']==d]['gainer_rate'].values for d in range(5)]
dow_a=stats.f_oneway(*dow_g); dow_k=stats.kruskal(*dow_g)
mon_g=[daily[daily['month']==m]['gainer_rate'].values for m in range(1,13) if m in daily['month'].unique()]
mon_a=stats.f_oneway(*mon_g) if len(mon_g)>=2 else (0,1); mon_k=stats.kruskal(*mon_g) if len(mon_g)>=2 else (0,1)

fig,axes=plt.subplots(2,2,figsize=(14,10))
dn=['Mon','Tue','Wed','Thu','Fri']; mn=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[0,0].bar(dn,[dow_eff.loc[d,'mean'] for d in range(5)],color='steelblue',alpha=0.8)
axes[0,0].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5,label=f'μ={daily["gainer_rate"].mean():.1%}')
axes[0,0].set_title(f'Day-of-Week (ANOVA p={dow_a[1]:.3f})',fontsize=12,fontweight='bold'); axes[0,0].legend(); axes[0,0].grid(True,alpha=0.3)
axes[0,1].bar([mn[m-1] for m in range(1,13) if m in mon_eff.index],[mon_eff.loc[m,'mean'] for m in range(1,13) if m in mon_eff.index],color='darkorange',alpha=0.8)
axes[0,1].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5)
axes[0,1].set_title(f'Month (ANOVA p={mon_a[1]:.3f})',fontsize=12,fontweight='bold'); axes[0,1].grid(True,alpha=0.3)
for l in axes[0,1].get_xticklabels(): l.set_rotation(45)
axes[1,0].bar([str(int(y)) for y in yr_eff.index],[yr_eff.loc[y,'mean'] for y in yr_eff.index],color='forestgreen',alpha=0.8)
axes[1,0].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5)
axes[1,0].set_title('Year-over-Year',fontsize=12,fontweight='bold'); axes[1,0].grid(True,alpha=0.3)
qrt=daily.groupby(pd.to_datetime(daily['date']).dt.quarter)['gainer_rate'].mean()
axes[1,1].bar([f'Q{q}' for q in range(1,5)],[qrt.get(q,0) for q in range(1,5)],color='purple',alpha=0.8)
axes[1,1].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5)
axes[1,1].set_title('Quarterly',fontsize=12,fontweight='bold'); axes[1,1].grid(True,alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'ts_seasonality.png',dpi=150,bbox_inches='tight'); plt.close()

ts_results['seasonality']={'dow':{dn[d]:float(dow_eff.loc[d,'mean']) for d in range(5)},
    'dow_anova':float(dow_a[1]),'dow_kruskal':float(dow_k[1]),
    'month':{mn[m-1]:float(mon_eff.loc[m,'mean']) for m in range(1,13) if m in mon_eff.index},
    'month_anova':float(mon_a[1]),'month_kruskal':float(mon_k[1]),
    'yearly':{str(int(y)):{'mean':float(yr_eff.loc[y,'mean']),'n':int(yr_eff.loc[y,'count'])} for y in yr_eff.index}}
print(f'  DOW p={dow_a[1]:.4f} KW p={dow_k[1]:.4f} | Month p={mon_a[1]:.4f} KW p={mon_k[1]:.4f}')

# 1g. Cross-sectional spread
print('\n[1g] Cross-sectional spread...')
cs_feats=['ret_1d','range_5','hv_20','vol_ratio_5','bb_width']
cs_res=[]
fig,axes=plt.subplots(2,3,figsize=(15,10)); axes=axes.flatten()
for i,feat in enumerate(cs_feats):
    if i>=len(axes): break
    fs['_r']=fs.groupby('date')[feat].rank(pct=True)
    fs['_q']=pd.qcut(fs['_r'].rank(method='first'),5,labels=False,duplicates='drop')
    qr=fs.groupby('_q')['target'].mean()
    sp=qr.max()-qr.min()
    cs_res.append({'feature':feat,'q1':float(qr.iloc[0]),'q5':float(qr.iloc[-1]),'spread':float(sp)})
    axes[i].bar(range(len(qr)),qr.values,color='steelblue',alpha=0.8)
    axes[i].axhline(fs['target'].mean(),color='red',ls='--',alpha=0.5,label=f'μ={fs["target"].mean():.1%}')
    axes[i].set_title(f'{feat} (spread={sp:.1%})',fontsize=11,fontweight='bold')
    axes[i].set_xticks(range(len(qr))); axes[i].set_xticklabels([f'Q{j+1}' for j in range(len(qr))])
    axes[i].grid(True,alpha=0.3)
for j in range(i+1,len(axes)): axes[j].set_visible(False)
plt.tight_layout(); fig.savefig(OUT/'charts'/'ts_cross_sectional.png',dpi=150,bbox_inches='tight'); plt.close()
fs=fs.drop(columns=[c for c in fs.columns if c.startswith('_')],errors='ignore')
pd.DataFrame(cs_res).to_csv(OUT/'tables'/'cross_sectional_spread.csv',index=False)
ts_results['cross_sectional']={r['feature']:r for r in cs_res}

# save TSA
with open(OUT/'tables'/'ts_analysis_results.json','w') as f: json.dump(ts_results,f,indent=2)
print(f'  TSA done in {time.time()-t1:.0f}s')

# ════════════════════════════════════════════════════════════════
# SECTION 2: DATA MINING (Phase 4)
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}\n SECTION 2: DATA MINING (Phase 4)\n{"="*60}')
t2=time.time()

import sys; sys.path.insert(0,str(BASE))
from src.patterns.candlestick import detect_patterns as detect_candle
from src.patterns.chart_patterns import detect_chart_patterns

# Load all 1day data for 200 symbols in one query
syms = [r[0] for r in con.execute(
    "SELECT symbol, COUNT(*) as cnt FROM raw_market WHERE timeframe='1day' GROUP BY symbol ORDER BY cnt DESC LIMIT 200"
).fetchall()]

# Batch load all symbols
all_ohlc = con.execute("""
    SELECT symbol, datetime, open, high, low, close, volume
    FROM raw_market WHERE timeframe='1day' AND symbol IN ({})
    ORDER BY symbol, datetime
""".format(','.join(["'"+s+"'" for s in syms]))).fetchdf()
print(f'  Loaded {len(all_ohlc):,} rows for {all_ohlc["symbol"].nunique()} symbols')

# Process patterns per symbol
pat_list = []
for sym in all_ohlc['symbol'].unique():
    df = all_ohlc[all_ohlc['symbol']==sym].copy()
    if len(df)<100: continue
    try:
        cm = detect_candle(df); cht = detect_chart_patterns(df)
        combined = pd.concat([cm,cht],axis=1)
        for col in combined.columns:
            occ = int(combined[col].sum())
            if occ > 0:
                pat_list.append({'symbol':sym,'pattern':col,'occurrences':occ,'frequency':occ/len(df)})
    except: pass

pdf = pd.DataFrame(pat_list)
pf = pdf.groupby('pattern').agg(total_occ=('occurrences','sum'),n_symbols=('symbol','nunique'),
    avg_freq=('frequency','mean')).sort_values('total_occ',ascending=False).reset_index()
pf.to_csv(OUT/'tables'/'pattern_frequency.csv',index=False)
print(f'  {len(pf)} patterns detected')
for _,r in pf.head(10).iterrows():
    print(f'    {r["pattern"]:<25s} occ={r["total_occ"]:>8,} sym={r["n_symbols"]} freq={r["avg_freq"]:.2%}')

# Pattern forward performance
perf_list = []
for sym in all_ohlc['symbol'].unique()[:50]:
    df = all_ohlc[all_ohlc['symbol']==sym].sort_values('datetime').copy()
    if len(df)<100: continue
    df['fwd_1d']=df['close'].pct_change().shift(-1)
    df['fwd_3d']=df['close'].pct_change(3).shift(-3)
    df['fwd_5d']=df['close'].pct_change(5).shift(-5)
    try:
        cm=detect_candle(df); cht=detect_chart_patterns(df)
        combined=pd.concat([cm,cht],axis=1)
        for col in combined.columns:
            mask=combined[col].astype(bool)
            if mask.sum()<5: continue
            perf_list.append({'pattern':col,'symbol':sym,'n':int(mask.sum()),
                'fwd1d':float(df.loc[mask,'fwd_1d'].mean()),'fwd3d':float(df.loc[mask,'fwd_3d'].mean()),
                'wr1d':float((df.loc[mask,'fwd_1d']>0).mean())})
    except: pass

pef=pd.DataFrame(perf_list)
if len(pef)>0:
    pa=pef.groupby('pattern').agg(n=('n','sum'),fwd1d=('fwd1d','mean'),fwd3d=('fwd3d','mean'),wr1d=('wr1d','mean')).reset_index()
    pa.to_csv(OUT/'tables'/'pattern_performance.csv',index=False)
    print('\n  Top by fwd1d return:')
    for _,r in pa.sort_values('fwd1d',ascending=False).head(10).iterrows():
        print(f'    {r["pattern"]:<25s} fwd1d={r["fwd1d"]:+.2%} wr1d={r["wr1d"]:.1%} fwd3d={r["fwd3d"]:+.2%} n={int(r["n"])}')

# Pattern co-occurrence
top_p = pf.head(12)['pattern'].tolist()
fig,ax=plt.subplots(figsize=(10,9))
cm_data=pd.DataFrame(0,index=top_p,columns=top_p)
for sym in all_ohlc['symbol'].unique()[:100]:
    sp=pdf[pdf['symbol']==sym]
    pp=sp[sp['pattern'].isin(top_p)]['pattern'].tolist()
    for i in range(len(pp)):
        for j in range(i+1,len(pp)):
            if pp[i] in top_p and pp[j] in top_p:
                cm_data.loc[pp[i],pp[j]]+=1; cm_data.loc[pp[j],pp[i]]+=1
im=ax.imshow(cm_data.values,cmap='YlOrRd',aspect='auto')
ax.set_xticks(range(len(top_p))); ax.set_yticks(range(len(top_p)))
ax.set_xticklabels(top_p,rotation=45,ha='right',fontsize=8); ax.set_yticklabels(top_p,fontsize=8)
ax.set_title('Pattern Co-occurrence',fontsize=13,fontweight='bold')
plt.colorbar(im,fraction=0.046); plt.tight_layout()
fig.savefig(OUT/'charts'/'pattern_cooccurrence.png',dpi=150,bbox_inches='tight'); plt.close()
print(f'  Co-occurrence matrix done')

# Association with market structure features
print('\n  Association of patterns with target on cleaned features...')
ef = pd.read_parquet(BASE / 'cleaned_features.parquet')
struct_cols = [c for c in ef.columns if c in (
    'fvg_bullish','fvg_bearish','ob_bullish','ob_bearish','liq_sweep_high','liq_sweep_low',
    'bos_up','bos_down','choch_sell','choch_buy','wyckoff_spring','wyckoff_upthrust',
    'mkt_in_value_area','vol_profile_high_vol_node','vol_profile_low_vol_node')]
struct_analysis = []
for col in struct_cols:
    if col not in ef.columns: continue
    n=ef[col].sum()
    if n<50: continue
    hr=ef.loc[ef[col]==1,'target'].mean()
    br=ef['target'].mean()
    lift=hr/br if br>0 else 0
    try:
        tbl=pd.crosstab(ef[col],ef['target'])
        if tbl.shape==(2,2):
            from scipy.stats import chi2_contingency
            chi2,p_val,_,_=chi2_contingency(tbl)
        else: p_val=1
    except: p_val=1
    struct_analysis.append({'feature':col,'n':int(n),'hit_rate':float(hr),'lift':float(lift),'pval':float(p_val)})

sa_df=pd.DataFrame(struct_analysis).sort_values('lift',ascending=False)
sa_df.to_csv(OUT/'tables'/'structure_feature_association.csv',index=False)
print(f'  Structure features with significant lift: {(sa_df["pval"]<0.05).sum()}/{len(sa_df)}')
for _,r in sa_df.head(10).iterrows():
    print(f'    {r["feature"]:<30s} lift={r["lift"]:.2f} hit={r["hit_rate"]:.1%} p={r["pval"]:.4f}')

print(f'  Data Mining done in {time.time()-t2:.0f}s')

# ════════════════════════════════════════════════════════════════
# SECTION 3: EDA (Phase 5)
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}\n SECTION 3: EDA (Phase 5)\n{"="*60}')
t3=time.time()

feat_cols=[c for c in ef.columns if c not in ('symbol','datetime','date','target','target_ret','next_close','next_open','year')]
num_cols=[c for c in feat_cols if ef[c].dtype in ('float64','float32','int64','int32')]
print(f'  Features: {len(feat_cols)}, numeric: {len(num_cols)}')

# Feature-target correlations (Pearson, Spearman, MI)
corr_data=[]
for c in num_cols:
    if ef[c].nunique()<2: continue
    s=ef[c].values; t=ef['target'].values
    pcorr=np.corrcoef(s,t)[0,1] if s.std()>0 else 0
    mi=mutual_info_classif(s.reshape(-1,1),t,random_state=42,discrete_features=False)[0]
    sp=stats.spearmanr(s,t)[0]
    corr_data.append({'feature':c,'pearson':float(pcorr),'spearman':float(sp),'mutual_info':float(mi)})

cdf=pd.DataFrame(corr_data).sort_values('mutual_info',ascending=False)
cdf.to_csv(OUT/'tables'/'feature_target_correlation.csv',index=False)

fig,ax=plt.subplots(figsize=(12,8))
top=cdf.head(20)
x=range(len(top))
ax.bar(x,top['mutual_info'].values,color='steelblue',alpha=0.8,label='Mutual Info')
ax2=ax.twinx()
ax2.plot(top['pearson'].values,'ro-',ms=5,lw=1.5,label='Pearson',alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(top['feature'].values,rotation=45,ha='right',fontsize=8)
ax.set_ylabel('Mutual Information'); ax2.set_ylabel('Pearson')
ax.set_title('Top 20 Features by MI with Target',fontsize=13,fontweight='bold')
l1,l2=ax.get_legend_handles_labels(); l3,l4=ax2.get_legend_handles_labels()
ax.legend(l1+l3,l2+l4,loc='upper right'); ax.grid(True,alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'eda_feature_importance.png',dpi=150,bbox_inches='tight'); plt.close()
print(f'  Top features by MI: {top.iloc[0]["feature"]} ({top.iloc[0]["mutual_info"]:.4f}), {top.iloc[1]["feature"]} ({top.iloc[1]["mutual_info"]:.4f})')

# Correlation heatmap
top_n=min(25,len(num_cols))
top_f=cdf.head(top_n)['feature'].tolist()
fig,ax=plt.subplots(figsize=(14,12))
sub=ef[top_f].corr()
im=ax.imshow(sub,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(len(top_f))); ax.set_yticks(range(len(top_f)))
ax.set_xticklabels(top_f,rotation=90,fontsize=7); ax.set_yticklabels(top_f,fontsize=7)
ax.set_title(f'Correlation Matrix (top {top_n})',fontsize=13,fontweight='bold')
plt.colorbar(im,fraction=0.046); plt.tight_layout()
fig.savefig(OUT/'charts'/'eda_correlation_heatmap.png',dpi=150,bbox_inches='tight'); plt.close()

# Highly correlated pairs
corr_m=ef[num_cols].corr().abs()
up=corr_m.where(np.triu(np.ones(corr_m.shape),k=1).astype(bool))
hpairs=[]
for i in range(len(up.columns)):
    for j in range(i+1,len(up.columns)):
        if up.iloc[i,j]>0.95:
            hpairs.append({'feat1':up.columns[i],'feat2':up.columns[j],'corr':float(corr_m.iloc[i,j])})
hpdf=pd.DataFrame(hpairs).sort_values('corr',ascending=False)
hpdf.to_csv(OUT/'tables'/'highly_correlated_features.csv',index=False)
print(f'  Highly correlated (r>0.95): {len(hpairs)} pairs')

# Symbol ranking
sym_t=ef.groupby('symbol')['target'].agg(['mean','sum','count']).sort_values('mean',ascending=False)
fig,axes=plt.subplots(1,2,figsize=(16,8))
for ax,d,color,title in zip(axes,
    [sym_t.head(20),sym_t.tail(20).sort_values('mean',ascending=True)],
    ['forestgreen','crimson'],
    ['Top 20 Symbols','Bottom 20 Symbols']):
    ax.barh(range(len(d)),d['mean'].values,color=color,alpha=0.8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index,fontsize=7)
    ax.axvline(ef['target'].mean(),color='red',ls='--',alpha=0.5,label=f'μ={ef["target"].mean():.1%}')
    ax.set_title(title,fontsize=12,fontweight='bold'); ax.set_xlabel('Gainer Rate'); ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout(); fig.savefig(OUT/'charts'/'eda_symbol_ranking.png',dpi=150,bbox_inches='tight'); plt.close()

# Target decile analysis for top features
best_f=cdf.head(5)['feature'].tolist()
fig,axes=plt.subplots(2,3,figsize=(15,10)); axes=axes.flatten()
for i,feat in enumerate(best_f):
    if i>=len(axes): break
    ef['_d']=pd.qcut(ef[feat].rank(method='first'),10,labels=False,duplicates='drop')
    d10=ef.groupby('_d')['target'].mean()
    axes[i].bar(range(len(d10)),d10.values,color='steelblue',alpha=0.8)
    axes[i].axhline(ef['target'].mean(),color='red',ls='--',alpha=0.5)
    axes[i].set_title(f'{feat} (spread={d10.max()-d10.min():.1%})',fontsize=10)
    axes[i].set_xlabel('Decile'); axes[i].set_ylabel('Gainer Rate'); axes[i].grid(True,alpha=0.3)
for j in range(i+1,len(axes)): axes[j].set_visible(False)
plt.tight_layout(); fig.savefig(OUT/'charts'/'eda_decile_analysis.png',dpi=150,bbox_inches='tight'); plt.close()
ef=ef.drop(columns=['_d'],errors='ignore')

# Feature distributions (sample 20)
samp_f=num_cols[:20]; n_cols=5; n_rows=4
fig,axes=plt.subplots(n_rows,n_cols,figsize=(20,16)); axes=axes.flatten()
for i,feat in enumerate(samp_f):
    if i>=len(axes): break
    s=ef[feat].dropna()
    axes[i].hist(s,bins=80,color='steelblue',alpha=0.7,density=True)
    axes[i].axvline(s.mean(),color='red',ls='--',label=f'μ={s.mean():.2f}')
    axes[i].set_title(f'{feat}\nskew={s.skew():.1f} kurt={s.kurtosis():.1f}',fontsize=8)
    axes[i].tick_params(axis='x',labelsize=6)
for i in range(len(samp_f),len(axes)): axes[i].set_visible(False)
plt.tight_layout(); fig.savefig(OUT/'charts'/'eda_feature_distributions.png',dpi=150,bbox_inches='tight'); plt.close()

# PCA
print('\n  PCA...')
pca_f=num_cols[:30]
pca_data=ef[pca_f].fillna(0).values
pca=StandardScaler().fit_transform(pca_data)
pca_obj=PCA(n_components=10).fit(pca)
var_exp=pca_obj.explained_variance_ratio_; cum_var=np.cumsum(var_exp)
pca_proj=pca_obj.transform(pca)

fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].bar(range(1,11),var_exp[:10],alpha=0.7,color='steelblue',label='Individual')
axes[0].plot(range(1,11),cum_var[:10],'ro-',ms=6,label='Cumulative')
axes[0].axhline(0.8,color='green',ls='--',alpha=0.5,label='80%')
axes[0].set_title('PCA Variance Explained',fontsize=12,fontweight='bold'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
colors=['crimson' if t==1 else 'lightgray' for t in ef['target'].values[:len(pca_proj)]]
axes[1].scatter(pca_proj[:5000,0],pca_proj[:5000,1],c=colors[:5000],alpha=0.5,s=3)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA (5K, red=gainer)',fontsize=12,fontweight='bold')
plt.tight_layout(); fig.savefig(OUT/'charts'/'eda_pca.png',dpi=150,bbox_inches='tight'); plt.close()
print(f'  PCA: PC1={var_exp[0]:.1%} PC2={var_exp[1]:.1%} Cum10={cum_var[9]:.1%} n80pct={np.searchsorted(cum_var,0.8)+1}')

# Year-over-year feature drift
ef['year']=pd.to_datetime(ef['datetime']).dt.year
yr_feat=ef.groupby('year')[num_cols].mean()
yr_feat.to_csv(OUT/'tables'/'features_by_year.csv',index=False)
yr_target=ef.groupby('year')['target'].mean()
top_drift=yr_feat.std().sort_values(ascending=False).head(10)
print('  Top drifting features YoY:')
for c in top_drift.index[:10]:
    print(f'    {c:<30s} std={top_drift[c]:.4f}')

# Outlier analysis (3 methods)
print('\n  Outlier analysis...')
out_data=[]
for c in num_cols:
    s=ef[c].dropna()
    if len(s)<100: continue
    q1,q3=np.percentile(s,[25,75]); iqr=q3-q1
    if iqr==0: continue
    n_iqr=((s<q1-3*iqr)|(s>q3+3*iqr)).sum()
    z=np.abs((s-s.mean())/s.std()) if s.std()>0 else np.zeros(len(s))
    n_z=(z>3).sum()
    mad=np.median(np.abs(s-np.median(s)))
    n_mad=int((0.6745*np.abs(s-np.median(s))/mad>3.5).sum()) if mad>0 else 0
    out_data.append({'feature':c,'pct_iqr':float(n_iqr/len(s)*100),'pct_z':float(n_z/len(s)*100),'pct_mad':float(n_mad/len(s)*100)})
odf=pd.DataFrame(out_data).sort_values('pct_iqr',ascending=False)
odf.to_csv(OUT/'tables'/'outlier_comparison.csv',index=False)
print(f'  >5% IQR outliers: {(odf["pct_iqr"]>5).sum()} features')

print(f'  EDA done in {time.time()-t3:.0f}s')

# ════════════════════════════════════════════════════════════════
# SECTION 4: DATA CLEANING (Phase 6)
# ════════════════════════════════════════════════════════════════
print(f'\n{"="*60}\n SECTION 4: DATA CLEANING (Phase 6)\n{"="*60}')
t4=time.time()

raw_ef=pd.read_parquet(BASE/'engineered_features.parquet')
clean_ef=pd.read_parquet(BASE/'cleaned_features.parquet')

# Before/After comparison for key features
key_f=['range_5','hv_20','ret_1d','bb_width','vol_ratio_5']
fig,axes=plt.subplots(2,3,figsize=(15,10)); axes=axes.flatten()
for i,feat in enumerate(key_f):
    if feat not in raw_ef.columns or feat not in clean_ef.columns: continue
    rs=raw_ef[feat].dropna(); cs=clean_ef[feat].dropna()
    axes[i].hist(rs,bins=80,alpha=0.5,color='red',density=True,label=f'Raw μ={rs.mean():.2f}')
    axes[i].hist(cs,bins=80,alpha=0.5,color='steelblue',density=True,label=f'Clean μ={cs.mean():.2f}')
    axes[i].set_title(feat,fontsize=10,fontweight='bold'); axes[i].legend(fontsize=7)
for j in range(i+1,len(axes)): axes[j].set_visible(False)
plt.tight_layout(); fig.savefig(OUT/'charts'/'cleaning_before_after.png',dpi=150,bbox_inches='tight'); plt.close()

# Data quality scoring
qual_data=[]
for c in num_cols:
    s=clean_ef[c].dropna()
    if len(s)==0: continue
    miss_pct=(1-len(s)/len(clean_ef))*100
    skew_pen=min(50,abs(s.skew())*5)
    qual=max(0,100-miss_pct*2-skew_pen)
    qual_data.append({'feature':c,'missing_pct':float(miss_pct),'skew':float(s.skew()),
        'quality_score':float(qual)})
qdf=pd.DataFrame(qual_data).sort_values('quality_score')
qdf.to_csv(OUT/'tables'/'feature_quality_scores.csv',index=False)
print(f'  Mean quality: {qdf["quality_score"].mean():.1f}')
print(f'  Low quality (<50): {(qdf["quality_score"]<50).sum()} features')

# Cleaning summary
print(f'\n  Data Cleaning Summary:')
print(f'    Raw: {raw_ef.shape}')
print(f'    Cleaned: {clean_ef.shape}')
print(f'    Features removed: {len(raw_ef.columns)-len(clean_ef.columns)}')
removed=[c for c in raw_ef.columns if c not in clean_ef.columns]
print(f'    Removed: {removed}')

print(f'  Data Cleaning done in {time.time()-t4:.0f}s')

# ════════════════════════════════════════════════════════════════
# COMPLETE
# ════════════════════════════════════════════════════════════════
con.close()
print(f'\n{"="*60}')
print(f' ALL ANALYSES COMPLETE in {time.time()-t0:.0f}s')
print(f' Output: {OUT}')
print(f' Total charts: {len(list((OUT/"charts").glob("*.png")))}')
print(f' Total tables: {len(list((OUT/"tables").glob("*.csv")))} + {len(list((OUT/"tables").glob("*.json")))}')
print('='*60)
