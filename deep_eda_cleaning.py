# EDA + Cleaning combined
import pandas as pd,numpy as np,time,warnings,json
from pathlib import Path;from scipy import stats;from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA;from sklearn.preprocessing import StandardScaler
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
BASE=Path(r'C:\Users\pc\Downloads\stock hist data');OUT=BASE/'deep_analysis_report'
OUT.mkdir(exist_ok=True);(OUT/'charts').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)
t0=time.time()
print('='*60,'\n EDA + CLEANING\n','='*60)

ef=pd.read_parquet(BASE/'cleaned_features.parquet')
feat_cols=[c for c in ef.columns if c not in ('symbol','datetime','date','target','target_ret','next_close','next_open','year')]
num_cols=[c for c in feat_cols if ef[c].dtype in ('float64','float32','int64','int32')]
print(f'  Loaded: {ef.shape}, features: {len(num_cols)}')

# ─── EDA ───
print('\n--- EDA ---')

# Feature-target correlation (Pearson, Spearman, MI)
corr_data=[]
for c in num_cols:
    if ef[c].nunique()<2:continue
    s=ef[c].values;t=ef['target'].values
    pcorr=np.corrcoef(s,t)[0,1] if s.std()>0 else 0
    mi=mutual_info_classif(s.reshape(-1,1),t,random_state=42,discrete_features=False)[0]
    sp=stats.spearmanr(s,t)[0]
    corr_data.append({'feature':c,'pearson':float(pcorr),'spearman':float(sp),'mutual_info':float(mi)})
cdf=pd.DataFrame(corr_data).sort_values('mutual_info',ascending=False)
cdf.to_csv(OUT/'tables'/'feature_target_correlation.csv',index=False)

fig,ax=plt.subplots(figsize=(12,8))
top=cdf.head(20);x=range(len(top))
ax.bar(x,top['mutual_info'].values,color='steelblue',alpha=0.8,label='MI')
ax2=ax.twinx();ax2.plot(top['pearson'].values,'ro-',ms=5,lw=1.5,label='Pearson',alpha=0.7)
ax.set_xticks(x);ax.set_xticklabels(top['feature'].values,rotation=45,ha='right',fontsize=8)
ax.set_ylabel('MI');ax2.set_ylabel('Pearson')
ax.set_title('Top 20 Features by Mutual Info',fontsize=13,fontweight='bold')
l1,l2=ax.get_legend_handles_labels();l3,l4=ax2.get_legend_handles_labels()
ax.legend(l1+l3,l2+l4,loc='upper right');ax.grid(True,alpha=0.3)
plt.tight_layout();fig.savefig(OUT/'charts'/'eda_feature_importance.png',dpi=150,bbox_inches='tight');plt.close()
print(f'  Top features: {top.iloc[0]["feature"]} (MI={top.iloc[0]["mutual_info"]:.4f})')

# Correlation heatmap
top20=cdf.head(20)['feature'].tolist()
fig,ax=plt.subplots(figsize=(12,10))
sub=ef[top20].corr();im=ax.imshow(sub,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(len(top20)));ax.set_yticks(range(len(top20)))
ax.set_xticklabels(top20,rotation=90,fontsize=8);ax.set_yticklabels(top20,fontsize=8)
ax.set_title('Correlation Heatmap (Top 20)',fontsize=13,fontweight='bold')
plt.colorbar(im,fraction=0.046);plt.tight_layout()
fig.savefig(OUT/'charts'/'eda_correlation_heatmap.png',dpi=150,bbox_inches='tight');plt.close()

# Highly correlated pairs
corr_m=ef[num_cols].corr().abs()
up=corr_m.where(np.triu(np.ones(corr_m.shape),k=1).astype(bool))
hp=[]
for i in range(len(up.columns)):
    for j in range(i+1,len(up.columns)):
        if up.iloc[i,j]>0.95:hp.append({'feat1':up.columns[i],'feat2':up.columns[j],'corr':float(corr_m.iloc[i,j])})
hpdf=pd.DataFrame(hp).sort_values('corr',ascending=False)
hpdf.to_csv(OUT/'tables'/'highly_correlated_features.csv',index=False)
print(f'  Highly correlated (r>0.95): {len(hp)} pairs')

# Symbol ranking
sym_t=ef.groupby('symbol')['target'].agg(['mean','sum','count']).sort_values('mean',ascending=False)
fig,axes=plt.subplots(1,2,figsize=(16,8))
for ax_,d,color,title in zip(axes,[sym_t.head(20),sym_t.tail(20).sort_values('mean',ascending=True)],['forestgreen','crimson'],['Top 20','Bottom 20']):
    ax_.barh(range(len(d)),d['mean'].values,color=color,alpha=0.8)
    ax_.set_yticks(range(len(d)));ax_.set_yticklabels(d.index,fontsize=7)
    ax_.axvline(ef['target'].mean(),color='red',ls='--',alpha=0.5,label=f'mu={ef["target"].mean():.1%}')
    ax_.set_title(title,fontsize=12,fontweight='bold');ax_.set_xlabel('Gainer Rate');ax_.legend();ax_.grid(True,alpha=0.3)
plt.tight_layout();fig.savefig(OUT/'charts'/'eda_symbol_ranking.png',dpi=150,bbox_inches='tight');plt.close()

# Decile analysis for top features
best_f=cdf.head(5)['feature'].tolist()
fig,axes=plt.subplots(2,3,figsize=(15,10));axes=axes.flatten()
for i,feat in enumerate(best_f):
    if i>=len(axes):break
    ef['_d']=pd.qcut(ef[feat].rank(method='first'),10,labels=False,duplicates='drop')
    d10=ef.groupby('_d')['target'].mean()
    axes[i].bar(range(len(d10)),d10.values,color='steelblue',alpha=0.8)
    axes[i].axhline(ef['target'].mean(),color='red',ls='--',alpha=0.5)
    axes[i].set_title(f'{feat} spread={d10.max()-d10.min():.1%}',fontsize=10)
    axes[i].set_xlabel('Decile');axes[i].set_ylabel('Gainer Rate');axes[i].grid(True,alpha=0.3)
for j in range(i+1,len(axes)):axes[j].set_visible(False)
plt.tight_layout();fig.savefig(OUT/'charts'/'eda_decile_analysis.png',dpi=150,bbox_inches='tight');plt.close()
ef=ef.drop(columns=['_d'],errors='ignore')

# Feature distributions
samp=num_cols[:20]
fig,axes=plt.subplots(4,5,figsize=(20,16));axes=axes.flatten()
for i,feat in enumerate(samp):
    if i>=len(axes):break
    s=ef[feat].dropna()
    axes[i].hist(s,bins=80,color='steelblue',alpha=0.7,density=True)
    axes[i].axvline(s.mean(),color='red',ls='--',label=f'mu={s.mean():.2f}')
    axes[i].set_title(f'{feat[:20]} skew={s.skew():.1f} kurt={s.kurtosis():.1f}',fontsize=7)
    axes[i].tick_params(axis='x',labelsize=5)
for i in range(len(samp),len(axes)):axes[i].set_visible(False)
plt.tight_layout();fig.savefig(OUT/'charts'/'eda_feature_distributions.png',dpi=150,bbox_inches='tight');plt.close()

# PCA
print('  PCA...')
pca_f=num_cols[:30];pd_=ef[pca_f].fillna(0).values;ps_=StandardScaler().fit_transform(pd_)
pca=PCA(n_components=10).fit(ps_);ve=pca.explained_variance_ratio_;cv_=np.cumsum(ve);pp=pca.transform(ps_)
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].bar(range(1,11),ve[:10],alpha=0.7,color='steelblue',label='Individual')
axes[0].plot(range(1,11),cv_[:10],'ro-',ms=6,label='Cumulative')
axes[0].axhline(0.8,color='green',ls='--',alpha=0.5,label='80%');axes[0].set_title('PCA Variance Explained',fontsize=12,fontweight='bold');axes[0].legend();axes[0].grid(True,alpha=0.3)
clrs=['crimson' if t==1 else 'lightgray' for t in ef['target'].values[:len(pp)]]
axes[1].scatter(pp[:5000,0],pp[:5000,1],c=clrs[:5000],alpha=0.5,s=3)
axes[1].set_xlabel('PC1');axes[1].set_ylabel('PC2');axes[1].set_title('PCA (5K, red=gainer)',fontsize=12,fontweight='bold')
plt.tight_layout();fig.savefig(OUT/'charts'/'eda_pca.png',dpi=150,bbox_inches='tight');plt.close()
print(f'  PC1={ve[0]:.1%} PC2={ve[1]:.1%} Cum10={cv_[9]:.1%} 80% at {np.searchsorted(cv_,0.8)+1}')

# Year-over-year drift
ef['year']=pd.to_datetime(ef['datetime']).dt.year
yr_feat=ef.groupby('year')[num_cols].mean();yr_feat.to_csv(OUT/'tables'/'features_by_year.csv',index=False)
yr_target=ef.groupby('year')['target'].mean()
top_drift=yr_feat.std().sort_values(ascending=False).head(10)
print('  YoY drift:')
for c in top_drift.index[:10]:print(f'    {c:<30s} std={top_drift[c]:.4f}')

# Outlier analysis
print('  Outliers...')
od_=[]
for c in num_cols:
    s=ef[c].dropna()
    if len(s)<100:continue
    q1,q3=np.percentile(s,[25,75]);iqr=q3-q1
    if iqr==0:continue
    n_iqr=((s<q1-3*iqr)|(s>q3+3*iqr)).sum()
    z=np.abs((s-s.mean())/s.std()) if s.std()>0 else np.zeros(len(s))
    n_z=(z>3).sum()
    mad=np.median(np.abs(s-np.median(s)))
    n_mad=int((0.6745*np.abs(s-np.median(s))/mad>3.5).sum()) if mad>0 else 0
    od_.append({'feature':c,'pct_iqr':float(n_iqr/len(s)*100),'pct_z':float(n_z/len(s)*100),'pct_mad':float(n_mad/len(s)*100)})
odf=pd.DataFrame(od_).sort_values('pct_iqr',ascending=False)
odf.to_csv(OUT/'tables'/'outlier_comparison.csv',index=False)
print(f'  >5% IQR outliers: {(odf["pct_iqr"]>5).sum()} features')

# ─── CLEANING ANALYSIS ───
print('\n--- CLEANING ---')
raw_ef=pd.read_parquet(BASE/'engineered_features.parquet')
clean_ef=pd.read_parquet(BASE/'cleaned_features.parquet')

key_f=['range_5','hv_20','ret_1d','bb_width','vol_ratio_5']
fig,axes=plt.subplots(2,3,figsize=(15,10));axes=axes.flatten()
for i,feat in enumerate(key_f):
    if feat not in raw_ef.columns or feat not in clean_ef.columns:continue
    rs=raw_ef[feat].dropna();cs=clean_ef[feat].dropna()
    axes[i].hist(rs,bins=80,alpha=0.5,color='red',density=True,label=f'Raw mu={rs.mean():.2f}')
    axes[i].hist(cs,bins=80,alpha=0.5,color='steelblue',density=True,label=f'Clean mu={cs.mean():.2f}')
    axes[i].set_title(feat,fontsize=10,fontweight='bold');axes[i].legend(fontsize=7)
for j in range(i+1,len(axes)):axes[j].set_visible(False)
plt.tight_layout();fig.savefig(OUT/'charts'/'cleaning_before_after.png',dpi=150,bbox_inches='tight');plt.close()

# Quality scores
qd_=[]
for c in num_cols:
    s=clean_ef[c].dropna()
    if len(s)==0:continue
    mp=(1-len(s)/len(clean_ef))*100;sp_=min(50,abs(s.skew())*5);qual=max(0,100-mp*2-sp_)
    qd_.append({'feature':c,'missing_pct':float(mp),'skew':float(s.skew()),'quality_score':float(qual)})
qdf=pd.DataFrame(qd_).sort_values('quality_score')
qdf.to_csv(OUT/'tables'/'feature_quality_scores.csv',index=False)
print(f'  Mean quality: {qdf["quality_score"].mean():.1f}, low (<50): {(qdf["quality_score"]<50).sum()}')

print(f'\nEDA+Cleaning done in {time.time()-t0:.0f}s')
print(f'Charts: {len(list((OUT/"charts").glob("*.png")))}')
print(f'Tables: {len(list((OUT/"tables").glob("*.csv")))}')
