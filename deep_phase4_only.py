# Phase 4 only
import duckdb,pandas as pd,numpy as np,time,warnings,json
from pathlib import Path; import sys
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
BASE=Path(r'C:\Users\pc\Downloads\stock hist data')
DB=BASE/'warehouse'/'market_data.duckdb';OUT=BASE/'deep_analysis_report'
OUT.mkdir(exist_ok=True);(OUT/'charts').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)
t0=time.time()
print('='*60,'\n PHASE 4: DATA MINING ONLY\n','='*60)

sys.path.insert(0,str(BASE))
from src.patterns.candlestick import detect_patterns as detect_candle
from src.patterns.chart_patterns import detect_chart_patterns

con=duckdb.connect(str(DB),read_only=True)
syms=[r[0] for r in con.execute("SELECT symbol,COUNT(*) as cnt FROM raw_market WHERE timeframe='1day' GROUP BY symbol ORDER BY cnt DESC LIMIT 200").fetchall()]
all_ohlc=con.execute("SELECT symbol,datetime,open,high,low,close,volume FROM raw_market WHERE timeframe='1day' AND symbol IN ({}) ORDER BY symbol,datetime".format(','.join(["'"+s+"'" for s in syms]))).fetchdf()
con.close()
print(f'  Loaded {len(all_ohlc):,} rows, {all_ohlc["symbol"].nunique()} symbols')

# Pattern frequency
print('[2a] Pattern frequency...')
pat_list=[]
for sym in all_ohlc['symbol'].unique():
    df=all_ohlc[all_ohlc['symbol']==sym].copy()
    if len(df)<100:continue
    try:
        cm=detect_candle(df);cht=detect_chart_patterns(df);combined=pd.concat([cm,cht],axis=1)
        for col in combined.columns:
            occ=int(combined[col].sum())
            if occ>0:pat_list.append({'symbol':sym,'pattern':col,'occurrences':occ,'frequency':occ/len(df)})
    except:pass
pdf=pd.DataFrame(pat_list)
pf=pdf.groupby('pattern').agg(total_occ=('occurrences','sum'),n_symbols=('symbol','nunique'),avg_freq=('frequency','mean')).sort_values('total_occ',ascending=False).reset_index()
pf.to_csv(OUT/'tables'/'pattern_frequency.csv',index=False)
print(f'  {len(pf)} patterns')
for _,r in pf.head(10).iterrows():print(f'    {r["pattern"]:<25s} occ={r["total_occ"]:>8,} sym={r["n_symbols"]} freq={r["avg_freq"]:.2%}')

# Pattern forward performance
print('[2b] Pattern performance...')
perf_list=[]
for sym in all_ohlc['symbol'].unique()[:50]:
    df=all_ohlc[all_ohlc['symbol']==sym].sort_values('datetime').copy()
    if len(df)<100:continue
    df['fwd_1d']=df['close'].pct_change().shift(-1);df['fwd_3d']=df['close'].pct_change(3).shift(-3);df['fwd_5d']=df['close'].pct_change(5).shift(-5)
    try:
        cm=detect_candle(df);cht=detect_chart_patterns(df);combined=pd.concat([cm,cht],axis=1)
        for col in combined.columns:
            mask=combined[col].astype(bool)
            if mask.sum()<5:continue
            perf_list.append({'pattern':col,'symbol':sym,'n':int(mask.sum()),'fwd1d':float(df.loc[mask,'fwd_1d'].mean()),'fwd3d':float(df.loc[mask,'fwd_3d'].mean()),'wr1d':float((df.loc[mask,'fwd_1d']>0).mean())})
    except:pass
pef=pd.DataFrame(perf_list)
if len(pef)>0:
    pa=pef.groupby('pattern').agg(n=('n','sum'),fwd1d=('fwd1d','mean'),fwd3d=('fwd3d','mean'),wr1d=('wr1d','mean')).reset_index()
    pa.to_csv(OUT/'tables'/'pattern_performance.csv',index=False)
    print('  Top by fwd1d:')
    for _,r in pa.sort_values('fwd1d',ascending=False).head(10).iterrows():print(f'    {r["pattern"]:<25s} fwd1d={r["fwd1d"]:+.2%} wr1d={r["wr1d"]:.1%} fwd3d={r["fwd3d"]:+.2%} n={int(r["n"])}')

# Co-occurrence
print('[2c] Co-occurrence...')
top_p=pf.head(12)['pattern'].tolist()
fig,ax=plt.subplots(figsize=(10,9))
cm_=pd.DataFrame(0,index=top_p,columns=top_p)
for sym in all_ohlc['symbol'].unique()[:100]:
    sp=pdf[pdf['symbol']==sym];pp=sp[sp['pattern'].isin(top_p)]['pattern'].tolist()
    for i in range(len(pp)):
        for j in range(i+1,len(pp)):
            if pp[i] in top_p and pp[j] in top_p:cm_.loc[pp[i],pp[j]]+=1;cm_.loc[pp[j],pp[i]]+=1
im=ax.imshow(cm_.values,cmap='YlOrRd',aspect='auto')
ax.set_xticks(range(len(top_p)));ax.set_yticks(range(len(top_p)))
ax.set_xticklabels(top_p,rotation=45,ha='right',fontsize=8);ax.set_yticklabels(top_p,fontsize=8)
ax.set_title('Pattern Co-occurrence',fontsize=13,fontweight='bold')
plt.colorbar(im,fraction=0.046);plt.tight_layout();fig.savefig(OUT/'charts'/'pattern_cooccurrence.png',dpi=150,bbox_inches='tight');plt.close()

# Structure feature association
print('[2d] Structure association...')
ef=pd.read_parquet(BASE/'cleaned_features.parquet')
struct_cols=[c for c in ef.columns if c in('fvg_bullish','fvg_bearish','ob_bullish','ob_bearish','liq_sweep_high','liq_sweep_low','bos_up','bos_down','choch_sell','choch_buy','wyckoff_spring','wyckoff_upthrust','mkt_in_value_area','vol_profile_high_vol_node','vol_profile_low_vol_node')]
sa=[]
for col in struct_cols:
    if col not in ef.columns:continue
    n=int(ef[col].sum())
    if n<50:continue
    hr=ef.loc[ef[col]==1,'target'].mean();br=ef['target'].mean();lift=hr/br if br>0 else 0
    try:
        tbl=pd.crosstab(ef[col],ef['target'])
        if tbl.shape==(2,2):
            from scipy.stats import chi2_contingency
            chi2,p_val,_,_=chi2_contingency(tbl)
        else:p_val=1
    except:p_val=1
    sa.append({'feature':col,'n':int(n),'hit_rate':float(hr),'lift':float(lift),'pval':float(p_val)})
sa_df=pd.DataFrame(sa).sort_values('lift',ascending=False)
sa_df.to_csv(OUT/'tables'/'structure_feature_association.csv',index=False)
print(f'  Structure features with lift>1: {(sa_df["lift"]>1).sum()}/{len(sa_df)}')
for _,r in sa_df.head(10).iterrows():print(f'    {r["feature"]:<30s} lift={r["lift"]:.2f} hit={r["hit_rate"]:.1%} p={r["pval"]:.4f}')

print(f'Phase 4 done in {time.time()-t0:.0f}s')
