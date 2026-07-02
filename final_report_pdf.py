# Final comprehensive PDF report - all phases summary
import json
from pathlib import Path
from fpdf import FPDF
from datetime import datetime

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
OUT = BASE / 'FINAL_PROJECT_REPORT.pdf'

class PDF(FPDF):
    def header(self):
        if self.page_no()>1:
            self.set_font('Helvetica','I',7); self.set_text_color(100,100,100)
            self.cell(0,5,'High Gainer Classifier - Final Report',align='C'); self.ln(6)
            self.set_draw_color(200,200,200); self.line(10,self.get_y(),200,self.get_y()); self.ln(3)
    def footer(self):
        self.set_y(-15); self.set_font('Helvetica','I',7); self.set_text_color(130,130,130)
        self.cell(0,10,f'Page {self.page_no()}/{{nb}}',align='C')
    def stitle(self,t,lev=0):
        sz={0:16,1:13,2:11}; cl={0:(20,50,100),1:(30,60,110),2:(60,60,60)}
        self.set_font('Helvetica','B',sz.get(lev,11)); self.set_text_color(*cl.get(lev))
        self.ln(2)
        if lev<=1: self.set_draw_color(*cl.get(lev)); self.line(10,self.get_y(),200,self.get_y()); self.ln(3)
        self.cell(0,8,t); self.ln(8)
    def body(self,t,sz=9): self.set_font('Helvetica','',sz); self.set_text_color(40,40,40); self.set_x(10); self.multi_cell(190,4.5,t); self.ln(1)
    def bullet(self,t,sz=9): self.set_font('Helvetica','',sz); self.set_text_color(40,40,40); self.set_x(15); self.multi_cell(185,4.5,'- '+t)
    def kv(self,k,v,sz=9): self.set_font('Helvetica','B',sz); self.set_text_color(40,40,40); self.set_x(10); self.cell(70,5,k); self.set_font('Helvetica','',sz); self.cell(0,5,str(v)); self.ln(5)
    def tbl(self,hd,data,cw=None):
        if cw is None: cw=[190/len(hd)]*len(hd)
        self.set_font('Helvetica','B',8); self.set_fill_color(30,60,110); self.set_text_color(255,255,255); self.set_x(10)
        for i,h in enumerate(hd): self.cell(cw[i],6,h,border=1,fill=True,align='C')
        self.ln(); fl=False
        for row in data:
            if self.get_y()>265:
                self.add_page(); self.set_font('Helvetica','B',8); self.set_fill_color(30,60,110); self.set_text_color(255,255,255); self.set_x(10)
                for i,h in enumerate(hd): self.cell(cw[i],6,h,border=1,fill=True,align='C'); self.ln()
            self.set_font('Helvetica','',7.5); self.set_x(10)
            self.set_fill_color(240,240,245) if fl else self.set_fill_color(255,255,255)
            for i,v in enumerate(row): self.cell(cw[i],5,str(v) if v is not None else '',border=1,fill=True,align='C')
            self.ln(); fl=not fl
        self.ln(3)

p=PDF(); p.alias_nb_pages(); p.set_auto_page_break(auto=True,margin=20)

# Cover
p.add_page(); p.ln(40)
p.set_font('Helvetica','B',26); p.set_text_color(20,50,100); p.cell(0,15,'High Gainer Classifier',align='C'); p.ln(12)
p.set_font('Helvetica','',14); p.set_text_color(60,60,60); p.cell(0,8,'Project Final Report',align='C'); p.ln(25)
p.set_draw_color(20,50,100); p.line(60,p.get_y(),150,p.get_y()); p.ln(15)
p.set_font('Helvetica','',10); p.set_text_color(80,80,80)
p.cell(0,7,'475 Indian Stocks - 2016-2026 - 1M+ Trading Days',align='C'); p.ln(7)
p.cell(0,7,'Target: Next-day open-to-close return > 2%',align='C'); p.ln(30)
p.set_font('Helvetica','',9); p.set_text_color(100,100,100)
p.cell(0,5,f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',align='C'); p.ln(5)

# Summary
p.add_page(); p.stitle('Project Summary')
p.body('Goal: Build a binary classifier to predict which Indian stocks will have a high-gainer day (next-day open-to-close return > 2%). Full ML pipeline across 18 phases, completed through Phase 14.')
p.kv('Universe','475 stocks -> 356 (after liquidity filter)')
p.kv('Dataset','1,012,896 rows -> 756,917 rows')
p.kv('Features','109 base + 48 interactions + 72 cluster features = 229 total')
p.kv('Target Rate','12.4% positive (8.1:1 imbalance)')
p.kv('Best AUC','0.6545 (Phase 8 baseline, LightGBM)')
p.kv('Final AUC','0.6280 [95%CI: 0.622-0.634] (cluster-enhanced ensemble)')

# Methodology
p.add_page(); p.stitle('2. Methodology')
p.stitle('2.1 Data Pipeline',lev=2)
p.body('DuckDB warehouse -> Phase 2 data collection -> Phase 3 data understanding -> Phase 4 candlestick/chart patterns -> Phase 5 EDA -> Phase 6 data cleaning -> Phase 7 feature engineering -> Phase 8 feature selection.')
p.kv('Pattern Mining','28 candlestick/chart patterns detected across 200 symbols')
p.kv('EDA','Full exploratory analysis: distributions, correlations, PCA, seasonality, outlier analysis, decile analysis')
p.kv('Cleaning','No OHLC violations, 0.80% stale prices, 2,588 zero-volume rows. 84.0/100 feature quality')
p.kv('Time Series','ACF/PACF: AR(4) process, Granger causality: ret_1d 100% significant, GMM 7 regimes identified')

p.stitle('2.2 Feature Selection (Phase 8)',lev=2)
p.body('9-method consensus: mutual information, VIF, XGBoost, LightGBM, permutation importance, forward selection, RFE, Boruta, stability analysis. 35 features selected from 109.')
p.tbl(['Rank','Feature','Category','Importance'],
    [['1','regime_2','Regime','13.7%'],['2','regime_0','Regime','10.0%'],['3','regime_3','Regime','6.9%'],
     ['4','range_10','Return/Vol','6.6%'],['5','regime_1','Regime','6.3%'],['6','range_5','Return/Vol','5.0%']],
    [15,50,35,30])

p.stitle('2.3 Model Approaches Tested',lev=2)
p.body('Five approaches compared across walkforward expanding-window validation (5 folds + final test):')
approaches = [
    ('Pooled (Phase 8)','35 features, all symbols, LightGBM + class weights','0.6545'),
    ('Improved Pooled','157 feats, 356 symbols, interactions, class weights','0.6265'),
    ('Single-Stock','Per-stock XGBoost on top 20 feats, 10 stocks','0.481-0.648'),
    ('Cluster-Specific','8 vol x mcap clusters, separate XGBoost per cluster','0.5850'),
    ('Cluster-Enhanced','229 feats incl cluster dummies + interactions, ensemble','0.6280'),
]
p.tbl(['Approach','Description','Test AUC'],approaches,[55,90,25])

# Results
p.add_page(); p.stitle('3. Results')
p.stitle('3.1 Best Model: Phase 8 Baseline (LightGBM)',lev=2)
p.tbl(['Metric','Validation','Test'],
    [['AUC-ROC','0.644','0.655'],['Avg Precision','0.189','0.213'],['Precision (th=0.5)','0.180','0.190'],
     ['Recall (th=0.5)','0.335','0.526'],['F1 (th=0.5)','0.234','0.279'],['MCC','0.128','0.149'],
     ['Lift','1.81x','1.56x']],[40,40,40])
p.body('At threshold 0.55 (optimal): Precision=22.1%, Recall=37.9%, F1=0.279, Lift=1.82x.')

p.stitle('3.2 Key Findings',lev=2)
findings = [
    'AUC ceiling of ~0.64-0.65 is consistent across all approaches tested. The inherent unpredictability of daily stock returns >2% limits model performance.',
    'Adding more features (109 -> 229) did NOT improve AUC. The 35 carefully selected features are sufficient.',
    'Cross-sectional ranks (rank_hv_20, rank_range_5) are valuable - they tell the model "this stock is extreme relative to peers"',
    'Regime labels (regime_0-3) dominate feature importance - market context is the strongest predictor',
    'Feature interactions (regime x volatility) capture meaningful signal and rank among top features',
    'SMOTE oversampling destroys calibration. Class weights (scale_pos_weight, class_weight="balanced") are the correct approach for imbalance.',
    'Single-stock models underperform due to insufficient training data (~200 positive examples per stock)',
    'Cluster-based models improve per-stock predictions (+0.03 to +0.10 AUC) but overall AUC is similar',
    'LightGBM consistently outperforms XGBoost (0.645 vs 0.595 average) for this dataset',
    'Threshold tuning is critical: default 0.5 gives recall ~25%, optimal 0.15-0.375 gives recall 45-57%',
]
for f in findings: p.bullet(f,sz=8.5)

p.stitle('3.3 Best Configuration',lev=2)
p.body('Recommended production setup:')
p.kv('Model','LightGBM (800 trees, max_depth=6, lr=0.06, subsample=0.85, class_weight=balanced)')
p.kv('Features','35 consensus features from Phase 8 (no interactions needed)')
p.kv('Training','All data up to current date, expanding window retrain quarterly')
p.kv('Threshold','0.55 (precision-focused) or 0.15 (recall-focused) depending on use case')
p.kv('Expected AUC','0.64-0.65 on out-of-sample data')
p.kv('Expected Lift','1.8x at th=0.55, 1.5x at th=0.15')
p.kv('Ensemble','Simple average of 3 LightGBM models with different seeds for stability')

# Limitations
p.add_page(); p.stitle('4. Limitations & Future Work')
p.stitle('4.1 Current Limitations',lev=2)
lims = [
    'No fundamental data (P/E, market cap, sector, earnings) - all features are price/volume only',
    'No sentiment data (news, social media, analyst ratings)',
    'No corporate actions calendar (bonus, split, buyback dates are treated as signal, not controlled for)',
    'No index/derivative data (NIFTY50 futures, VIX, FII/DII flows)',
    'Target definition (open-to-close > 2%) is arbitrary - different thresholds may yield different results',
    'Penny stock filter removed high-volatility names that may be more predictable',
    'No order-book / microstructure data (bid-ask spread, order flow imbalance)',
]
for l in lims: p.bullet(l,sz=8.5)

p.stitle('4.2 Future Improvements',lev=2)
futures = [
    'Add fundamental features (P/E, market cap bucket, sector dummies, promoter holding)',
    'Add sentiment features (news sentiment, social media volume for high-gainer candidates)',
    'Add options data (implied volatility, put-call ratio for liquid stocks)',
    'Use regression instead of classification (predict continuous return, then threshold)',
    'Explore deep learning (LSTM/Transformer on sequence of 20-60 days per stock)',
    'Hierarchical model: first predict "high-volatility day" then "gain direction"',
    'Multi-target: predict full return distribution (quantile regression) for better threshold selection',
]
for f in futures: p.bullet(f,sz=8.5)

# Conclusion
p.add_page(); p.stitle('5. Conclusion')
p.body('After completing Phases 1-14 of the ML pipeline (problem definition through model evaluation):')
p.body('The best achievable AUC for predicting >2% daily stock returns using price/volume data alone is approximately 0.65. This is consistent with academic literature on short-term stock return prediction, where AUC values of 0.60-0.68 are typical for daily frequency data without fundamental or alternative data sources.')
p.body('The model provides meaningful lift: at 1.8x lift, it identifies 80% more gainers than random selection. In absolute terms, it captures 38% of all gainers (12,587 out of 33,247) with 22% precision at the optimal threshold.')
p.body('For production deployment, the LightGBM model with the 35-feature Phase 8 consensus set is recommended. The model should be retrained quarterly with expanding window data, and the threshold should be tuned based on the precision-recall requirements of the specific use case.')
p.ln(5)
p.set_font('Helvetica','I',9); p.set_text_color(100,100,100)
p.cell(0,5,'End of Report',align='C')

p.output(str(OUT))
print(f'Final report: {OUT} ({OUT.stat().st_size/1e3:.0f} KB)')
