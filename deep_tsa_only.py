# TSA only - executes quickly
import duckdb,pandas as pd,numpy as np,time,warnings,json
from pathlib import Path; from scipy import stats
from sklearn.mixture import GaussianMixture; from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
BASE=Path(r'C:\Users\pc\Downloads\stock hist data')
DB=BASE/'warehouse'/'market_data.duckdb'; OUT=BASE/'deep_analysis_report'
OUT.mkdir(exist_ok=True);(OUT/'charts').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)
t0=time.time()
print('='*60,'\n TSA ONLY','\n','='*60)
con=duckdb.connect(str(DB),read_only=True)

fs=con.execute("SELECT symbol,datetime::DATE as date,open,high,low,close,volume,ret_1d,range_5,hv_20,rsi_14,vol_ratio_5,bb_width,adx FROM feature_store WHERE timeframe='1day' ORDER BY symbol,date").fetchdf()
fs=fs.sort_values(['symbol','date'])
fs['next_close']=fs.groupby('symbol')['close'].shift(-1);fs['next_open']=fs.groupby('symbol')['open'].shift(-1)
fs['target_ret']=fs['next_close']/fs['next_open']-1;fs['target']=(fs['target_ret']>0.02).astype(int)
fs=fs.dropna(subset=['target'])
daily=fs.groupby('date').agg(gainer_rate=('target','mean'),n_stocks=('symbol','count'),avg_ret=('ret_1d','mean'),avg_range=('range_5','mean'),avg_hv=('hv_20','mean')).reset_index().sort_values('date')
daily['gainer_rate']=daily['gainer_rate'].fillna(0)

def cacf(s,nl=40):
    n=len(s);m=np.mean(s);v=np.var(s,ddof=0)
    if v==0:return np.zeros(nl+1)
    c0=np.sum((s-m)**2)/n;acf=np.ones(nl+1)
    for k in range(1,nl+1):acf[k]=np.sum((s[:-k]-m)*(s[k:]-m))/n/c0
    return acf
def cpacf(s,nl=40):
    acf=cacf(s,nl);pacf=np.zeros(nl+1);pacf[0]=1.0
    for k in range(1,nl+1):
        if k==1:pacf[k]=acf[1]
        else:
            phi=np.zeros(k+1);phi[1]=acf[1]
            for i in range(2,k+1):
                n_=acf[i]-np.sum(phi[1:i]*acf[i-1:0:-1]);d_=1-np.sum(phi[1:i]*acf[1:i])
                phi[i]=n_/d_ if abs(d_)>1e-10 else 0
                for j in range(1,i):phi[j]=phi[j]-phi[i]*phi[i-j]
            pacf[k]=phi[k]
    return pacf

# 1a ACF/PACF
print('[1a] ACF/PACF...')
ts=daily['gainer_rate'].values
acf=cacf(ts,40);pacf=cpacf(ts,40);se95=1.96/np.sqrt(len(ts))
sig_acf=[i for i in range(1,41) if abs(acf[i])>se95];sig_pacf=[i for i in range(1,41) if abs(pacf[i])>se95]
fig,axes=plt.subplots(2,1,figsize=(14,8))
for ax,vals,title,clr in zip(axes,[acf,pacf],['ACF - Gainer Rate','PACF - Gainer Rate'],['steelblue','darkorange']):
    ax.bar(range(1,41),vals[1:],color=clr,width=0.6)
    ax.axhline(se95,color='red',ls='--',alpha=0.5);ax.axhline(-se95,color='red',ls='--',alpha=0.5);ax.axhline(0,color='black',lw=0.5)
    ax.set_title(title,fontsize=13,fontweight='bold');ax.set_xlabel('Lag');ax.set_ylabel('Value');ax.grid(True,alpha=0.3)
plt.tight_layout();fig.savefig(OUT/'charts'/'ts_acf_pacf.png',dpi=150,bbox_inches='tight');plt.close()
pd.DataFrame({'lag':range(1,41),'ACF':acf[1:],'PACF':pacf[1:],'sig_acf':[abs(acf[i])>se95 for i in range(1,41)],'sig_pacf':[abs(pacf[i])>se95 for i in range(1,41)]}).to_csv(OUT/'tables'/'acf_pacf.csv',index=False)
print(f'  ACF: {sig_acf[:10]}; PACF: {sig_pacf[:10]}')

# 1b Stationarity
print('[1b] Stationarity...')
from statsmodels.tsa.stattools import adfuller,kpss
stat_r=[]
for feat in ['gainer_rate','avg_ret','avg_hv','avg_range']:
    s=daily[feat].dropna().values;adf_p=adfuller(s,maxlag=20)[1] if len(s)>100 else 1
    try:kpss_p=kpss(s,regression='c',nlags='auto')[1]
    except:kpss_p=1
    stat_r.append({'feature':feat,'adf_pval':float(adf_p),'kpss_pval':float(kpss_p),'adf_stationary':adf_p<0.05,'kpss_stationary':kpss_p>=0.05})
pd.DataFrame(stat_r).to_csv(OUT/'tables'/'stationarity.csv',index=False)

# 1c Granger
print('[1c] Granger...')
from statsmodels.tsa.stattools import grangercausalitytests
gc_feats=['ret_1d','range_5','hv_20','rsi_14','vol_ratio_5','bb_width','adx'];gc_d=[]
for sym in fs['symbol'].unique()[:100]:
    sd=fs[fs['symbol']==sym].sort_values('date').dropna(subset=['target']+gc_feats)
    if len(sd)<100:continue
    for feat in gc_feats:
        try:
            g=grangercausalitytests(sd[[feat,'target']].values,maxlag=5,verbose=False)
            bl=min(range(1,6),key=lambda l:g[l][0]['ssr_chi2test'][1])
            gc_d.append({'symbol':sym,'feature':feat,'best_pval':float(g[bl][0]['ssr_chi2test'][1]),'best_lag':bl,'significant':g[bl][0]['ssr_chi2test'][1]<0.05})
        except:pass
gc_df=pd.DataFrame(gc_d)
gc_agg=gc_df.groupby('feature').agg(pct_sig=('significant','mean'),median_pval=('best_pval','median'),mean_lag=('best_lag','mean')).reset_index()
gc_agg.to_csv(OUT/'tables'/'granger_detailed.csv',index=False);gc_df.to_csv(OUT/'tables'/'granger_per_symbol.csv',index=False)
for _,r in gc_agg.iterrows():print(f'  {r["feature"]:<15s} {r["pct_sig"]:.0%} sig p={r["median_pval"]:.2e} lag={r["mean_lag"]:.1f}')

# 1d Windows
print('[1d] Windows...')
windows=[3,5,10,15,21,30,50];win_d=[]
for sym in fs['symbol'].unique()[:100]:
    sd=fs[fs['symbol']==sym].sort_values('date').dropna(subset=['target'])
    if len(sd)<200:continue
    for feat in ['ret_1d','range_5','hv_20','vol_ratio_5']:
        for w in windows:
            r=sd[feat].rolling(w,min_periods=max(3,w//3)).mean();c=r.corr(sd['target'])
            win_d.append({'symbol':sym,'feature':feat,'window':w,'corr':float(c) if not np.isnan(c) else 0})
wdf=pd.DataFrame(win_d);wag=wdf.groupby(['feature','window']).agg(corr_mean=('corr','mean'),corr_std=('corr','std')).reset_index()
wag.to_csv(OUT/'tables'/'window_optimization.csv',index=False)
fig,ax=plt.subplots(figsize=(10,6))
for feat in ['ret_1d','range_5','hv_20','vol_ratio_5']:
    fw=wag[wag['feature']==feat];ax.plot(fw['window'],fw['corr_mean'],marker='o',label=feat,lw=2)
ax.axhline(0,color='gray',ls='--',alpha=0.5);ax.set_xlabel('Window');ax.set_ylabel('Corr');ax.set_title('Window Optimization',fontsize=13,fontweight='bold');ax.legend();ax.grid(True,alpha=0.3)
plt.tight_layout();fig.savefig(OUT/'charts'/'ts_window_optimization.png',dpi=150,bbox_inches='tight');plt.close()
opt_wins={}
for feat in ['ret_1d','range_5','hv_20','vol_ratio_5']:
    fw=wag[wag['feature']==feat];b=fw.loc[fw['corr_mean'].abs().idxmax()];opt_wins[feat]={'window':int(b['window']),'corr':float(b['corr_mean'])}
print(f'  Optimal: {opt_wins}')

# 1e Regimes
print('[1e] Regimes...')
rf=daily[['avg_ret','avg_hv','avg_range']].dropna().values;rf_s=StandardScaler().fit_transform(rf)
bic=[];
for n in range(2,8):g=GaussianMixture(n_components=n,random_state=42,n_init=10,max_iter=500).fit(rf_s);bic.append({'n':n,'bic':g.bic(rf_s)})
best_n=int(min(bic,key=lambda x:x['bic'])['n'])
gmm=GaussianMixture(n_components=best_n,random_state=42,n_init=15,max_iter=500)
daily['regime']=np.nan;daily.loc[daily[['avg_ret','avg_hv','avg_range']].dropna().index,'regime']=gmm.fit_predict(rf_s)
daily=daily.dropna(subset=['regime']);daily['regime']=daily['regime'].astype(int)
rp=daily.groupby('regime').agg(gainer_rate=('gainer_rate','mean'),avg_ret=('avg_ret','mean'),avg_hv=('avg_hv','mean'),n_days=('date','count')).reset_index()
fig,ax=plt.subplots(figsize=(10,6));ax.plot([b['n'] for b in bic],[b['bic'] for b in bic],marker='o',lw=2,color='steelblue')
ax.axvline(best_n,color='red',ls='--',alpha=0.7,label=f'n={best_n}');ax.set_xlabel('Regimes');ax.set_ylabel('BIC');ax.set_title('GMM Regime Selection',fontsize=13,fontweight='bold');ax.legend();ax.grid(True,alpha=0.3)
plt.tight_layout();fig.savefig(OUT/'charts'/'ts_gmm_bic.png',dpi=150,bbox_inches='tight');plt.close()
nr=best_n;tm=np.zeros((nr,nr));ra=daily['regime'].values
for t in range(1,len(ra)):tm[ra[t-1],ra[t]]+=1
tm=tm/(tm.sum(axis=1,keepdims=True)+1e-10)
fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(tm,cmap='YlOrRd',aspect='auto',vmin=0,vmax=1)
for i in range(nr):
    for j in range(nr):ax.text(j,i,f'{tm[i,j]:.2f}',ha='center',va='center',fontsize=9,color='black' if tm[i,j]<0.5 else 'white')
ax.set_xticks(range(nr));ax.set_yticks(range(nr));ax.set_xlabel('To');ax.set_ylabel('From')
ax.set_title(f'Regime Transition (n={best_n})',fontsize=13,fontweight='bold')
plt.colorbar(im,fraction=0.046);plt.tight_layout();fig.savefig(OUT/'charts'/'ts_regime_transition.png',dpi=150,bbox_inches='tight');plt.close()
daily[['date','avg_ret','avg_hv','avg_range','gainer_rate','n_stocks','regime']].to_csv(OUT/'tables'/'regime_data_full.csv',index=False)
rp.to_csv(OUT/'tables'/'regime_profiles.csv',index=False);np.savetxt(OUT/'tables'/'transition_matrix.csv',tm,delimiter=',',fmt='%.4f')
for _,r in rp.iterrows():print(f'  R{int(r["regime"])}: n={int(r["n_days"])} gainer={r["gainer_rate"]:.1%} ret={r["avg_ret"]:.2f}%')

# 1f Seasonality
print('[1f] Seasonality...')
daily['dow']=pd.to_datetime(daily['date']).dt.dayofweek;daily['month']=pd.to_datetime(daily['date']).dt.month;daily['year']=pd.to_datetime(daily['date']).dt.year
de=daily.groupby('dow')['gainer_rate'].agg(['mean','std','count']);me=daily.groupby('month')['gainer_rate'].agg(['mean','std','count'])
ye=daily.groupby('year')['gainer_rate'].agg(['mean','std','count'])
dg=[daily[daily['dow']==d]['gainer_rate'].values for d in range(5)];da=stats.f_oneway(*dg);dk=stats.kruskal(*dg)
mg=[daily[daily['month']==m]['gainer_rate'].values for m in range(1,13) if m in daily['month'].unique()]
ma=stats.f_oneway(*mg) if len(mg)>=2 else (0,1);mk=stats.kruskal(*mg) if len(mg)>=2 else (0,1)
dn=['Mon','Tue','Wed','Thu','Fri'];mn=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig,axes=plt.subplots(2,2,figsize=(14,10))
axes[0,0].bar(dn,[de.loc[d,'mean'] for d in range(5)],color='steelblue',alpha=0.8)
axes[0,0].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5,label=f'mu={daily["gainer_rate"].mean():.1%}')
axes[0,0].set_title(f'Day-of-Week (ANOVA p={da[1]:.3f})',fontsize=12,fontweight='bold');axes[0,0].legend();axes[0,0].grid(True,alpha=0.3)
axes[0,1].bar([mn[m-1] for m in range(1,13) if m in me.index],[me.loc[m,'mean'] for m in range(1,13) if m in me.index],color='darkorange',alpha=0.8)
axes[0,1].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5)
axes[0,1].set_title(f'Month (ANOVA p={ma[1]:.3f})',fontsize=12,fontweight='bold');axes[0,1].grid(True,alpha=0.3)
for l in axes[0,1].get_xticklabels():l.set_rotation(45)
axes[1,0].bar([str(int(y)) for y in ye.index],[ye.loc[y,'mean'] for y in ye.index],color='forestgreen',alpha=0.8)
axes[1,0].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5);axes[1,0].set_title('Year-over-Year',fontsize=12,fontweight='bold');axes[1,0].grid(True,alpha=0.3)
qrt=daily.groupby(pd.to_datetime(daily['date']).dt.quarter)['gainer_rate'].mean()
axes[1,1].bar([f'Q{q}' for q in range(1,5)],[qrt.get(q,0) for q in range(1,5)],color='purple',alpha=0.8)
axes[1,1].axhline(daily['gainer_rate'].mean(),color='red',ls='--',alpha=0.5);axes[1,1].set_title('Quarterly',fontsize=12,fontweight='bold');axes[1,1].grid(True,alpha=0.3)
plt.tight_layout();fig.savefig(OUT/'charts'/'ts_seasonality.png',dpi=150,bbox_inches='tight');plt.close()
print(f'  DOW p={da[1]:.4f} KW p={dk[1]:.4f} | Month p={ma[1]:.4f} KW p={mk[1]:.4f}')

# 1g Cross-sectional
print('[1g] Cross-sectional...')
cs_feats=['ret_1d','range_5','hv_20','vol_ratio_5','bb_width'];cs_r=[]
fig,axes=plt.subplots(2,3,figsize=(15,10));axes=axes.flatten()
for i,feat in enumerate(cs_feats):
    if i>=len(axes):break
    fs['_r']=fs.groupby('date')[feat].rank(pct=True);fs['_q']=pd.qcut(fs['_r'].rank(method='first'),5,labels=False,duplicates='drop')
    qr=fs.groupby('_q')['target'].mean();sp=qr.max()-qr.min()
    cs_r.append({'feature':feat,'q1':float(qr.iloc[0]),'q5':float(qr.iloc[-1]),'spread':float(sp)})
    axes[i].bar(range(len(qr)),qr.values,color='steelblue',alpha=0.8)
    axes[i].axhline(fs['target'].mean(),color='red',ls='--',alpha=0.5,label=f'mu={fs["target"].mean():.1%}')
    axes[i].set_title(f'{feat} (spread={sp:.1%})',fontsize=11,fontweight='bold');axes[i].set_xticks(range(len(qr)))
    axes[i].set_xticklabels([f'Q{j+1}' for j in range(len(qr))]);axes[i].grid(True,alpha=0.3)
for j in range(i+1,len(axes)):axes[j].set_visible(False)
plt.tight_layout();fig.savefig(OUT/'charts'/'ts_cross_sectional.png',dpi=150,bbox_inches='tight');plt.close()
fs=fs.drop(columns=[c for c in fs.columns if c.startswith('_')],errors='ignore')
pd.DataFrame(cs_r).to_csv(OUT/'tables'/'cross_sectional_spread.csv',index=False)
for r in cs_r:print(f'  {r["feature"]:<15s} spread={r["spread"]:.1%}')

# Save
ts_r={'acf':{'sig_acf_lags':sig_acf,'sig_pacf_lags':sig_pacf},'stationarity':{r['feature']:r['adf_pval'] for r in stat_r},
    'granger':{r['feature']:{'pct_sig':float(r['pct_sig']),'median_pval':float(r['median_pval']),'mean_lag':float(r['mean_lag'])} for _,r in gc_agg.iterrows()},
    'optimal_windows':opt_wins,'regimes':{'n_regimes':best_n,'profiles':{int(r['regime']):{'gainer_rate':float(r['gainer_rate']),'n_days':int(r['n_days'])} for _,r in rp.iterrows()}},
    'seasonality':{'dow':{dn[d]:float(de.loc[d,'mean']) for d in range(5)},'month':{mn[m-1]:float(me.loc[m,'mean']) for m in range(1,13) if m in me.index}}}
with open(OUT/'tables'/'ts_analysis_results.json','w') as f:json.dump(ts_r,f,indent=2)

con.close()
print(f'\nTSA complete in {time.time()-t0:.0f}s')
