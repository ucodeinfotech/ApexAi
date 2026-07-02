# Generate Comprehensive PDF Report
import pandas as pd, numpy as np, json, textwrap, time
from pathlib import Path
from fpdf import FPDF
from datetime import datetime

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
IN = BASE / 'deep_analysis_report'
OUT = BASE / 'deep_analysis_report' / 'COMPREHENSIVE_REPORT.pdf'

t0 = time.time()

class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(100,100,100)
            self.cell(0, 5, 'Stock High Gainer Classifier - Comprehensive Analysis Report', align='C')
            self.ln(6)
            self.set_draw_color(200,200,200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(130,130,130)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title, level=0):
        sizes = {0:16, 1:13, 2:11}
        colors = {0:(20,50,100), 1:(30,60,110), 2:(60,60,60)}
        self.set_font('Helvetica', 'B', sizes.get(level, 11))
        self.set_text_color(*colors.get(level, (60,60,60)))
        self.set_draw_color(*colors.get(level, (60,60,60)))
        self.ln(4)
        if level <= 1:
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
        self.cell(0, 8, title)
        self.ln(8)

    def body_text(self, text, size=9):
        self.set_font('Helvetica', '', size)
        self.set_text_color(40,40,40)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def bullet(self, text, size=9):
        self.set_font('Helvetica', '', size)
        self.set_text_color(40,40,40)
        x0 = self.l_margin + 5
        self.set_x(x0)
        self.multi_cell(0, 4.5, '- ' + text)

    def key_value(self, key, value, size=9):
        self.set_font('Helvetica', 'B', size)
        self.set_text_color(40,40,40)
        self.cell(70, 5, key)
        self.set_font('Helvetica', '', size)
        self.cell(0, 5, value)
        self.ln(5)

    def add_chart(self, path, caption='', w=170):
        if path.exists():
            self.image(str(path), x=15, w=w)
            if caption:
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(80,80,80)
                self.cell(0, 4, caption, align='C')
                self.ln(4)
            self.ln(3)

    def add_small_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(30, 60, 110)
        self.set_text_color(255,255,255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align='C')
        self.ln()
        self.set_font('Helvetica', '', 7.5)
        self.set_text_color(40,40,40)
        fill = False
        for row in data:
            if self.get_y() > 265:
                self.add_page()
                self.set_font('Helvetica', 'B', 8)
                self.set_fill_color(30, 60, 110)
                self.set_text_color(255,255,255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 6, h, border=1, fill=True, align='C')
                self.ln()
                self.set_font('Helvetica', '', 7.5)
                self.set_text_color(40,40,40)
            if fill:
                self.set_fill_color(240,240,245)
            else:
                self.set_fill_color(255,255,255)
            for i, val in enumerate(row):
                txt = str(val) if val is not None else ''
                self.cell(col_widths[i], 5, txt, border=1, fill=True, align='C')
            self.ln()
            fill = not fill
        self.ln(3)


pdf = ReportPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ══════════════════════════════════════════════════════════════
# COVER PAGE
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica', 'B', 28)
pdf.set_text_color(20, 50, 100)
pdf.cell(0, 15, 'High Gainer Classifier', align='C')
pdf.ln(12)
pdf.set_font('Helvetica', '', 16)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 8, 'Depth Analysis Report', align='C')
pdf.ln(20)
pdf.set_draw_color(20, 50, 100)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(15)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(80,80,80)
pdf.cell(0, 7, 'Predicting Next-Day Open-to-Close Returns > 2%', align='C')
pdf.ln(7)
pdf.cell(0, 7, '475 Stocks :: 2016-2026 :: 1M+ Trading Days', align='C')
pdf.ln(30)
pdf.set_font('Helvetica', '', 9)
pdf.set_text_color(100,100,100)
pdf.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')
pdf.ln(5)
pdf.cell(0, 5, '18-Phase ML Pipeline :: Phases 4-5-6 + Time Series Analysis', align='C')

# ══════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('Table of Contents')
toc_items = [
    ('1', 'Executive Summary'),
    ('2', 'Dataset Overview'),
    ('3', 'Time Series Analysis'),
    ('4', '  3.1 ACF/PACF Analysis'),
    ('5', '  3.2 Stationarity Tests'),
    ('6', '  3.3 Granger Causality'),
    ('7', '  3.4 Rolling Window Optimization'),
    ('8', '  3.5 GMM Regime Clustering'),
    ('9', '  3.6 Seasonality Analysis'),
    ('10', '  3.7 Cross-Sectional Feature Spread'),
    ('11', 'Phase 4: Data Mining'),
    ('12', '  4.1 Pattern Detection & Frequency'),
    ('13', '  4.2 Pattern Forward Performance'),
    ('14', '  4.3 Pattern Co-occurrence'),
    ('15', '  4.4 Market Structure Feature Association'),
    ('16', 'Phase 5: Exploratory Data Analysis'),
    ('17', '  5.1 Feature-Target Relationship'),
    ('18', '  5.2 Correlation Analysis'),
    ('19', '  5.3 Symbol-Level Analysis'),
    ('20', '  5.4 PCA Dimensionality Reduction'),
    ('21', '  5.5 Outlier Detection'),
    ('22', 'Phase 6: Data Cleaning'),
    ('23', '  6.1 Cleaning Summary'),
    ('24', '  6.2 Feature Quality Scores'),
    ('25', '  6.3 Before vs After Comparison'),
    ('26', 'Key Findings & Recommendations'),
]
for num, title in toc_items:
    w = 10 if num.startswith(' ') else 0
    pdf.set_font('Helvetica', 'B' if w == 0 else '', 10)
    pdf.set_text_color(40,40,40)
    pdf.cell(w + 5, 6, num.strip())
    pdf.cell(0, 6, title)
    pdf.ln(6)

# ══════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('1. Executive Summary')

pdf.body_text(
    'This report presents a comprehensive depth analysis of the High Gainer Classifier project, '
    'spanning Time Series Analysis (phases 2-3), Data Mining (Phase 4), Exploratory Data Analysis '
    '(Phase 5), and Data Cleaning (Phase 6). The dataset comprises 475 stocks with 1,012,896 trading '
    'days from 2016 to 2026, targeting next-day open-to-close returns exceeding 2% (12.4% positive rate).'
)

pdf.ln(2)
pdf.section_title('Key Metrics', level=2)
pdf.key_value('Dataset Size', '1,012,896 rows x 116 columns')
pdf.key_value('Trading Days', '2,626')
pdf.key_value('Symbols', '475 (post penny-filter)')
pdf.key_value('Target Rate', '12.4% (125,418 positive events)')
pdf.key_value('Class Imbalance Ratio', '8.1:1 (non-gainer:gainer)')
pdf.key_value('Date Range', '2016-01-01 to 2026-06-24')

pdf.ln(2)
pdf.section_title('Top Findings', level=2)
pdf.bullet('Time Series: Daily gainer rate shows significant short-term autocorrelation (lag-1 ACF=0.161) and a bi-weekly cycle (lags 16-18).')
pdf.bullet('Stationarity: All features are ADF-stationary at both market and per-symbol level, confirming the validity of standard ML approaches.')
pdf.bullet('Granger Causality: ret_1d, rsi_14, and vol_ratio_5 Granger-cause the next-day gainer in 100% of tested symbols (p << 0.001).')
pdf.bullet('Regimes: 7 GMM regimes identified. Extreme regimes (R3: crash, R5: strong bull) show 20-27% gainer rates vs 9.7% for normal regime (R0).')
pdf.bullet('Seasonality: Month effect is statistically significant (ANOVA p=0.001). April shows highest gainer rate (14.4%), July lowest (10.7%).')
pdf.bullet('Cross-sectional Spread: hv_20 and range_5 have ~9% spreads between top and bottom quintiles, making them strong discriminators.')
pdf.bullet('Pattern Mining: double_bottom leads to +1.44% average next-day return (81.3% win rate), the strongest predictive pattern found.')
pdf.bullet('Structure Features: Bearish Fair Value Gap (fvg_bearish) has the highest lift for next-day gainers at 1.52 (18.9% hit rate).')

# ══════════════════════════════════════════════════════════════
# 2. DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('2. Dataset Overview')

pdf.body_text(
    'The primary data source is the DuckDB warehouse containing 1.04M daily rows across 490 symbols. '
    'After penny-filtering (avg close < Rs.50 since 2024), 475 symbols remain. Features are sourced from '
    'the pre-computed feature_store table (99 technical indicators) plus 33 market structure features '
    '(Phase 4) and 12+ engineered features (Phase 7 cross-sectional ranks, rolling windows, regime labels).'
)

pdf.ln(2)
pdf.section_title('Feature Categories', level=2)
feat_cats = [
    ('Technical Indicators', '~55', 'SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.'),
    ('Returns & Range', '~12', 'ret_1d, range_5/10/20, log_ret_1d, etc.'),
    ('Volume', '~10', 'OBV, CMF, MFI, vol_ratio_5/10/20, EOM, FI, VPT'),
    ('Cross-sectional Ranks', '4', 'rank_hv_20, rank_range_5, rank_bb_width, rank_vol_ratio_5'),
    ('Rolling Windows', '6', 'ret_1d_ma_5, range_5_ma_21, hv_20_ma_3, etc.'),
    ('Regime Labels', '5', 'regime_0 through regime_4 (one-hot from GMM)'),
    ('Temporal', '17', 'dow_0-4, month_1-12'),
    ('Lagged', '12', 'ret_1d_lag1-3, range_5_lag1-3, vol_ratio_5_lag1-3, hv_20_lag1, etc.'),
    ('Market Structure', '33', 'FVG, Order Blocks, Wyckoff, BOS, Liquidity Sweeps'),
]
pdf.add_small_table(
    ['Category', 'Count', 'Description'],
    feat_cats,
    [35, 15, 140]
)

# ══════════════════════════════════════════════════════════════
# 3. TIME SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('3. Time Series Analysis')
pdf.body_text(
    'Time series analysis examines the temporal structure of daily gainer rates to determine '
    'optimal window lengths, feature transformations, and regime-aware modeling approaches.'
)

# 3.1 ACF/PACF
pdf.section_title('3.1 ACF/PACF Analysis', level=2)
pdf.body_text(
    'The autocorrelation function (ACF) and partial autocorrelation function (PACF) of the daily '
    'gainer rate reveal significant short-term momentum and bi-weekly cycles.'
)

pdf.add_chart(IN/'charts'/'ts_acf_pacf.png', 'Figure 1: ACF and PACF of Daily Gainer Rate (lags 1-40)')

try:
    acf_df = pd.read_csv(IN/'tables'/'acf_pacf.csv')
    sig_acf = acf_df[acf_df['sig_ACF'] == True]['lag'].tolist()
    sig_pacf = acf_df[acf_df['sig_PACF'] == True]['lag'].tolist()
    pdf.body_text(f'Significant ACF lags: {sig_acf[:15]}')
    pdf.body_text(f'Significant PACF lags: {sig_pacf[:15]}')

    key_lags = [1, 2, 3, 5, 10, 16, 17, 18, 20]
    lag_data = []
    for l in key_lags:
        row = acf_df[acf_df['lag'] == l]
        if len(row) > 0:
            r = row.iloc[0]
            lag_data.append([str(l), f'{r["ACF"]:.4f}', f'{r["PACF"]:.4f}',
                            'Yes' if r['sig_ACF'] else 'No', 'Yes' if r['sig_PACF'] else 'No'])
    pdf.add_small_table(['Lag', 'ACF', 'PACF', 'Sig ACF', 'Sig PACF'], lag_data, [20, 40, 40, 40, 40])

    pdf.body_text(
        'Interpretation: ACF decays slowly with significant values at lags 1-4 (short-term momentum) '
        'and 16-17 (bi-weekly). PACF cuts off after lag 4, suggesting an AR(4) process. '
        'The bi-weekly pattern (lags 16-18) suggests settlement cycles or expiry effects.'
    )
except: pass

# 3.2 Stationarity
pdf.add_page()
pdf.section_title('3.2 Stationarity Tests', level=2)
pdf.body_text(
    'ADF and KPSS tests were applied to both market-level aggregate series and per-symbol features '
    'to determine if differencing is needed.'
)

try:
    stat_df = pd.read_csv(IN/'tables'/'stationarity.csv')
    pdf.body_text('Market-level stationarity:')
    st_data = []
    for _, r in stat_df.iterrows():
        adf_v = 'Stationary' if r['adf_stationary'] else 'Non-stationary'
        kpss_v = 'Stationary' if r['kpss_stationary'] else 'Non-stationary'
        verdict = 'Stationary' if r['adf_stationary'] and r['kpss_stationary'] else 'Borderline'
        st_data.append([r['feature'], f'{r["adf_pval"]:.6f}', adf_v, f'{r["kpss_pval"]:.6f}', kpss_v, verdict])
    pdf.add_small_table(['Series', 'ADF p-val', 'ADF Verdict', 'KPSS p-val', 'KPSS Verdict', 'Final'],
                        st_data, [28, 28, 28, 28, 28, 28])

    pdf.body_text(
        'All features pass ADF stationarity at p<0.05. KPSS shows borderline results for gainer_rate '
        '(p=0.045) and avg_hv (p=0.036), suggesting conditional heteroskedasticity rather than '
        'unit root issues. Per-symbol tests confirm 100% of symbols have stationary features.'
    )
except: pass

# 3.3 Granger Causality
pdf.add_page()
pdf.section_title('3.3 Granger Causality', level=2)
pdf.body_text(
    'Granger causality tests determine whether lagged values of each feature help predict the '
    'target (next-day gainer) beyond the target\'s own history. Tests used maxlag=5 with SSR '
    'chi-square test statistic.'
)

try:
    gc_df = pd.read_csv(IN/'tables'/'granger_detailed.csv')
    gc_data = []
    for _, r in gc_df.iterrows():
        gc_data.append([r['feature'], f'{r["pct_sig"]:.0%}', f'{r["median_pval"]:.2e}', f'{r["mean_lag"]:.1f}'])
    pdf.add_small_table(['Feature', '% Symbols Sig', 'Median p-value', 'Mean Opt Lag'],
                        gc_data, [35, 45, 55, 45])

    pdf.body_text(
        'Key findings: (1) ret_1d is the single strongest predictor - yesterday\'s return Granger-causes '
        'today\'s gainer in 100% of symbols with median p=10^-220. (2) vol_ratio_5 has the longest '
        'optimal lag (4.4 days), suggesting volume buildup precedes gainers. (3) hv_20 has the shortest '
        'optimal lag (1.5 days), meaning recent volatility spikes are immediately informative. '
        'These patterns validate the feature-specific window approach implemented in Phase 7.'
    )
except: pass

# 3.4 Rolling Window Optimization
pdf.section_title('3.4 Rolling Window Optimization', level=2)
pdf.add_chart(IN/'charts'/'ts_window_optimization.png', 'Figure 2: Feature-Window Correlation and Mutual Information with Target')

try:
    win_df = pd.read_csv(IN/'tables'/'window_optimization.csv')
    pdf.body_text('Optimal rolling windows (max |corr| with target):')
    win_data = []
    for feat in ['ret_1d', 'range_5', 'hv_20', 'vol_ratio_5']:
        fw = win_df[win_df['feature'] == feat]
        if len(fw) > 0:
            best = fw.loc[fw['corr_mean'].abs().idxmax()]
            win_data.append([feat, str(int(best['window'])), f'{best["corr_mean"]:+.4f}', f'{best["corr_std"]:.4f}'])
    pdf.add_small_table(['Feature', 'Optimal Window', 'Mean Correlation', 'Std Dev'], win_data, [35, 40, 50, 45])

    pdf.body_text(
        'range_5 with 3-day window shows the strongest correlation (+0.090). hv_20 similarly prefers '
        'short windows (3 days). ret_1d shows negative correlation at 5-days (-0.012), suggesting '
        'mean-reversion at weekly horizons. vol_ratio_5 peaks at 30 days, indicating sustained volume '
        'trends are more informative than short-term spikes.'
    )
except: pass

# 3.5 GMM Regime Clustering
pdf.add_page()
pdf.section_title('3.5 GMM Regime Clustering', level=2)
pdf.body_text(
    'Gaussian Mixture Models were applied to daily market-level features (avg_ret, avg_hv, avg_range) '
    'to identify distinct market regimes. BIC optimization selected 7 regimes.'
)

pdf.add_chart(IN/'charts'/'ts_gmm_bic.png', 'Figure 3: GMM Regime Selection via BIC/AIC')

try:
    rp_df = pd.read_csv(IN/'tables'/'regime_profiles.csv')
    pdf.body_text('Regime profiles (sorted by gainer rate):')
    rp_data = []
    for _, r in rp_df.sort_values('gainer_rate', ascending=False).iterrows():
        rp_data.append([f'R{int(r["regime"])}', f'{r["n_days"]} ({r["n_days"]/2626*100:.0f}%)',
                        f'{r["gainer_rate"]:.1%}', f'{r["avg_ret"]:+.2f}%'])
    pdf.add_small_table(['Regime', 'Days', 'Gainer Rate', 'Avg Return'], rp_data, [20, 50, 50, 50])

    pdf.body_text(
        'The 7 regimes range from R5 (Strong Bull: 27% gainer rate, 13 days) to R0 (Normal: 9.7% gainer '
        'rate, 995 days). The extreme regimes (R3, R4, R5) are rare (<2% of days combined) but show '
        'dramatically elevated gainer rates (21-27%). This validates the regime-aware feature engineering '
        'approach and suggests regime-conditional models may outperform single-model approaches.'
    )
except: pass

pdf.add_chart(IN/'charts'/'ts_regime_transition.png', 'Figure 4: Regime Transition Matrix (7 regimes)')

pdf.body_text(
    'The transition matrix reveals high persistence in normal regimes (R0: 39%, R6: 76% self-transition) '
    'and rapid mean-reversion in extreme regimes. Regime R5 (Strong Bull) has only 30% persistence, '
    'suggesting these opportunities are short-lived.'
)

# 3.6 Seasonality
pdf.add_page()
pdf.section_title('3.6 Seasonality Analysis', level=2)
pdf.add_chart(IN/'charts'/'ts_seasonality.png', 'Figure 5: Seasonality Analysis (DOW, Month, Year, Quarter)')

try:
    ts_r = json.load(open(IN/'tables'/'ts_analysis_results.json'))
    seas = ts_r.get('seasonality', {})
    dow = seas.get('dow', {})
    mon = seas.get('month', {})

    dow_data = [[d, f'{v:.1%}'] for d, v in dow.items()]
    mon_data = [[m, f'{v:.1%}'] for m, v in mon.items()]

    pdf.body_text(f'Day-of-Week (ANOVA p={seas.get("dow_anova", "N/A"):.4f}):')
    pdf.add_small_table(['Day', 'Gainer Rate'], dow_data, [50, 50])
    pdf.body_text(f'Month (ANOVA p={seas.get("month_anova", "N/A"):.4f}, KW p={seas.get("month_kruskal", "N/A"):.4f}):')
    pdf.add_small_table(['Month', 'Gainer Rate'], mon_data, [50, 50])

    pdf.body_text(
        'Day-of-week effects are not statistically significant (ANOVA p=0.473), while month effects '
        'are significant (ANOVA p=0.001, Kruskal-Wallis p=0.038). April shows the highest gainer rate '
        '(14.4%) potentially due to FY-start effects. July-September shows a seasonal trough (10.4-11.6%). '
        '2020 had the highest yearly rate (15.8%) due to COVID volatility.'
    )
except Exception as e:
    pdf.body_text(f'Could not load seasonality data: {str(e)}')

# 3.7 Cross-Sectional Feature Spread
pdf.section_title('3.7 Cross-Sectional Feature Spread', level=2)
pdf.add_chart(IN/'charts'/'ts_cross_sectional.png', 'Figure 6: Cross-Sectional Feature Spread (quintile analysis)')

try:
    cs_df = pd.read_csv(IN/'tables'/'cross_sectional_spread.csv')
    cs_data = []
    for _, r in cs_df.iterrows():
        cs_data.append([r['feature'], f'{r["q1"]:.1%}', f'{r["q5"]:.1%}', f'{r["spread"]:.1%}'])
    pdf.add_small_table(['Feature', 'Q1 Rate', 'Q5 Rate', 'Spread'], cs_data, [35, 45, 45, 45])

    pdf.body_text(
        'Cross-sectional ranking (how a stock ranks vs its peers TODAY) is the strongest signal family. '
        'Stocks in the top quintile of hv_20 (highest volatility) have 16.9% chance of being gainers '
        'tomorrow vs 8.0% for bottom quintile - a 8.9% absolute spread. This underscores the importance '
        'of the percentile-rank features engineered in Phase 7.'
    )
except: pass

# ══════════════════════════════════════════════════════════════
# 4. DATA MINING (Phase 4)
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('4. Phase 4: Data Mining')

pdf.body_text(
    'Data mining covers candlestick pattern detection, chart pattern detection, pattern forward '
    'performance analysis, co-occurrence patterns, and market structure feature association with the target.'
)

pdf.section_title('4.1 Pattern Detection & Frequency', level=2)
try:
    pf = pd.read_csv(IN/'tables'/'pattern_frequency.csv')
    pf_data = []
    for _, r in pf.head(15).iterrows():
        pf_data.append([r['pattern'][:20], f'{int(r["total_occ"]):,}', f'{r["n_symbols"]}', f'{r["avg_freq"]:.2%}'])
    pdf.add_small_table(['Pattern', 'Total Occ', 'Symbols', 'Avg Freq'], pf_data, [50, 40, 30, 40])

    pdf.body_text(
        '28 patterns were detected across 200 symbols. The "channel" pattern is pervasive (94% frequency), '
        'followed by spinning_top (30%), nr4 (27%), and pennant (25%). Three_black_crows (12%) and '
        'doji (11%) are the most frequent candlestick-specific patterns.'
    )
except: pass

# 4.2 Pattern Forward Performance
pdf.section_title('4.2 Pattern Forward Performance', level=2)
try:
    pa = pd.read_csv(IN/'tables'/'pattern_performance.csv')
    pdf.body_text('Top patterns by next-day forward return:')
    pp_data = []
    for _, r in pa.sort_values('fwd1d', ascending=False).head(12).iterrows():
        pp_data.append([r['pattern'][:22], f'{r["fwd1d"]:+.2%}', f'{r["wr1d"]:.1%}', f'{r["fwd3d"]:+.2%}', f'{int(r["n"]):,}'])
    pdf.add_small_table(['Pattern', 'Fwd1D', 'WinRate1D', 'Fwd3D', 'N'], pp_data, [50, 25, 30, 30, 25])

    pdf.body_text(
        'double_bottom dominates with +1.44% average next-day return and 81.3% win rate - this is '
        'the single strongest pattern discovered. darvas_box_up (+0.37%), volatility_contraction (+0.28%), '
        'and darvas_box_down (+0.24%) also show positive predictive value. Notably, reversal patterns '
        '(hammer, morning_star, marubozu) outperform continuation patterns.'
    )
except: pass

# 4.3 Pattern Co-occurrence
pdf.section_title('4.3 Pattern Co-occurrence', level=2)
pdf.add_chart(IN/'charts'/'pattern_cooccurrence.png', 'Figure 7: Pattern Co-occurrence Matrix (top 12)')
pdf.body_text(
    'The co-occurrence matrix reveals strong clustering: continuation patterns (flag, pennant, channel) '
    'tend to co-occur, while reversal patterns (double_bottom, three_black_crows, morning_star) form '
    'a separate cluster. This suggests a factor structure underlying pattern occurrences.'
)

# 4.4 Market Structure Features
pdf.section_title('4.4 Market Structure Feature Association', level=2)
try:
    sa_df = pd.read_csv(IN/'tables'/'structure_feature_association.csv')
    sa_data = []
    for _, r in sa_df.iterrows():
        sa_data.append([r['feature'][:25], f'{r["lift"]:.2f}', f'{r["hit_rate"]:.1%}', f'{r["n"]:,}', f'{r["pval"]:.4f}'])
    pdf.add_small_table(['Feature', 'Lift', 'Hit Rate', 'N', 'p-value'], sa_data, [55, 25, 30, 30, 30])

    pdf.body_text(
        'Bearish Fair Value Gap (fvg_bearish) shows the strongest association with next-day gainers '
        '(lift=1.52, 18.9% hit rate). Break of Structure downward (bos_down) follows at lift=1.32. '
        'These SMC (Smart Money Concept) features significantly enhance the feature set and were '
        'integrated into the engineered feature store in Phase 7.'
    )
except: pass

# ══════════════════════════════════════════════════════════════
# 5. EDA (Phase 5)
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('5. Phase 5: Exploratory Data Analysis')

pdf.section_title('5.1 Feature-Target Relationship', level=2)
pdf.add_chart(IN/'charts'/'eda_feature_importance.png', 'Figure 8: Top 20 Features by Mutual Information with Target')

try:
    cdf = pd.read_csv(IN/'tables'/'feature_target_correlation.csv')
    top_corr = cdf.head(20)
    ct_data = []
    for _, r in top_corr.iterrows():
        ct_data.append([r['feature'][:25], f'{r["mutual_info"]:.4f}', f'{r["pearson"]:+.4f}', f'{r["spearman"]:+.4f}'])
    pdf.add_small_table(['Feature', 'MI', 'Pearson', 'Spearman'], ct_data, [55, 30, 35, 35])

    pdf.body_text(
        'regime_3 (bullish regime) has the highest mutual information with target (0.108), confirming '
        'the regime-aware approach. range_5_ma_21, rank_hv_20, and rank_range_5 follow - all are '
        'cross-sectional or rolling-window features engineered in Phase 7. This validates the feature '
        'engineering decisions made earlier.'
    )
except: pass

pdf.add_chart(IN/'charts'/'eda_decile_analysis.png', 'Figure 9: Decile Analysis of Top Features vs Target Rate')

# 5.2 Correlation Analysis
pdf.section_title('5.2 Correlation Analysis', level=2)
pdf.add_chart(IN/'charts'/'eda_correlation_heatmap.png', 'Figure 10: Feature Correlation Matrix (Top 20)')

pdf.body_text(
    '83 feature pairs show correlation > 0.95, concentrated among price-based indicators '
    '(SMA family, price levels) and volume-based indicators (VWAP family). This multicollinearity '
    'will be addressed in Phase 8 Feature Selection via VIF analysis and correlation-based filtering.'
)

# 5.3 Symbol-Level Analysis
pdf.section_title('5.3 Symbol-Level Analysis', level=2)
pdf.add_chart(IN/'charts'/'eda_symbol_ranking.png', 'Figure 11: Top and Bottom Symbol Ranking by Gainer Rate')

pdf.body_text(
    'GMRINFRA has the highest gainer rate (20.0%) while SENSEX has the lowest (0.9%). High-gainer '
    'symbols are predominantly mid/small-cap with higher volatility profiles, while low-gainer '
    'symbols are large-cap index heavyweights and defensive stocks (HDFCBANK 4.0%, ITC 4.4%, '
    'NESTLEIND 4.6%). This suggests symbol-specific model calibration may be beneficial.'
)

# 5.4 PCA
pdf.section_title('5.4 PCA Dimensionality Reduction', level=2)
pdf.add_chart(IN/'charts'/'eda_pca.png', 'Figure 12: PCA Analysis - Variance Explained and Projection')

pdf.body_text(
    'PCA on 30 features reveals that 5 components explain 80% of variance, and 10 components '
    'explain 96%. The PCA projection shows significant overlap between gainer and non-gainer classes, '
    'confirming the challenge of the classification task. No clear linear separation exists, '
    'justifying the use of non-linear models like XGBoost.'
)

# 5.5 Outlier Detection
pdf.section_title('5.5 Outlier Detection', level=2)
try:
    odf = pd.read_csv(IN/'tables'/'outlier_comparison.csv')
    od_top = odf.head(10)
    od_data = []
    for _, r in od_top.iterrows():
        od_data.append([r['feature'][:22], f'{r["pct_iqr"]:.1f}%', f'{r["pct_z"]:.1f}%', f'{r["pct_mad"]:.1f}%'])
    pdf.add_small_table(['Feature', 'IQR (3x)', 'Z-score (3)', 'MAD (3.5)'], od_data, [50, 35, 35, 35])

    pdf.body_text(
        '21 features have >5% IQR-based outliers. macd_hist tops the list (11.0%), followed by volume '
        '(9.8%) and obv (8.0%). These were capped at 0.5/99.5 percentiles in Phase 6. The MAD method '
        'consistently flags fewer outliers, suggesting IQR-based capping is appropriate.'
    )
except: pass

# ══════════════════════════════════════════════════════════════
# 6. DATA CLEANING (Phase 6)
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('6. Phase 6: Data Cleaning')

pdf.section_title('6.1 Cleaning Summary', level=2)
pdf.body_text('Actions taken during Data Cleaning Phase:')
pdf.bullet('Missing Values: None found (all filled via median/imputation in Phase 7)')
pdf.bullet('Outlier Capping: 69 features winsorized at 0.5/99.5 percentiles (597,798 values capped)')
pdf.bullet('Duplicate Rows: None found')
pdf.bullet('Low-Variance Features Removed: 8 (regime_4, rs_vs_market, rs_ratio_market, rs_vs_sector, rs_ratio_sector, rs_momentum_10/20, rs_peer_rank)')
pdf.bullet('Final Shape: 1,012,896 rows x 116 columns (114 features + target + metadata)')

# 6.2 Feature Quality
pdf.section_title('6.2 Feature Quality Scores', level=2)
try:
    qdf = pd.read_csv(IN/'tables'/'feature_quality_scores.csv')
    q_bot = qdf.head(10)
    pdf.body_text(f'Mean quality score: {qdf["quality_score"].mean():.1f}/100')
    pdf.body_text(f'Features with quality < 50: {(qdf["quality_score"]<50).sum()}')
    pdf.body_text('Bottom 10 by quality score:')
    qd_data = []
    for _, r in q_bot.iterrows():
        qd_data.append([r['feature'][:25], f'{r["quality_score"]:.1f}', f'{r["missing_pct"]:.1f}%', f'{r["skew"]:.1f}'])
    pdf.add_small_table(['Feature', 'Score', 'Missing%', 'Skew'], qd_data, [60, 25, 35, 35])

    pdf.body_text(
        'No features have quality scores below 50. The lowest-scoring features are price-level '
        'indicators (vol_profile_vwap, vol_profile_vpoc) which have high skew due to the wide range '
        'of stock prices (Rs.50 to Rs.50,000+). These are retained as tree-based models handle '
        'skew well.'
    )
except: pass

# 6.3 Before/After
pdf.section_title('6.3 Before vs After Cleaning', level=2)
pdf.add_chart(IN/'charts'/'cleaning_before_after.png', 'Figure 13: Feature Distributions Before and After Cleaning')

pdf.body_text(
    'The before/after comparison shows that extreme tails were trimmed while preserving the central '
    'distribution shape. For example, range_5 was capped from [0.01, 175] to [1.87, 31.77], removing '
    'extreme outliers that likely represent data errors or corporate actions. The mean and median '
    'of most features remain largely unchanged, confirming that capping removed noise, not signal.'
)

# ══════════════════════════════════════════════════════════════
# KEY FINDINGS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('Key Findings & Recommendations')

pdf.section_title('Top 10 Findings', level=2)
findings = [
    ('Granger Causality Confirmed', 'ret_1d, rsi_14, and vol_ratio_5 Granger-cause next-day gainers in 100% of symbols, validating their use as predictive features.'),
    ('Regime Awareness Critical', 'Gainers are 2.8x more likely in extreme regimes (R3/R4/R5: 21-27%) vs normal (R0: 9.7%). Regime-conditional modeling is recommended.'),
    ('Cross-Sectional > Absolute', 'Percentile ranks (rank_hv_20 spread=8.9%) outperform absolute values (hv_20 spread=10.9% but confounded by symbol identity).'),
    ('Pattern Mining Uncovered Strong Signals', 'double_bottom (+1.44% fwd1d, 81.3% WR) and fvg_bearish (lift=1.52) are powerful standalone predictors.'),
    ('Month Effect is Real', 'Seasonal patterns are statistically significant (p=0.001). April yield is 40% higher than July.'),
    ('Window Length Matters', 'Feature-specific windows (3-5 days for vol/range, 30 days for volume) outperform fixed windows by 15-25% in correlation.'),
    ('Bi-weekly Cycle Present', 'ACF shows significant lags at 16-18 days, suggesting settlement/F&O expiry effects.'),
    ('No Unit Root Issues', '100% of features are ADF-stationary, allowing standard ML without differencing.'),
    ('High Multicollinearity', '83 feature pairs with r>0.95 require aggressive feature selection in Phase 8.'),
    ('Class Imbalance is Manageable', '8.1:1 ratio with 125K positive examples is sufficient for XGBoost with scale_pos_weight~7.1.'),
]

for i, (title, desc) in enumerate(findings, 1):
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 5, f'{i}. {title}')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 4.5, desc)
    pdf.ln(2)

pdf.section_title('Recommendations for Phase 8+', level=2)
recs = [
    'Feature Selection: Drop highly correlated pairs (r>0.95), target 30-40 features via mutual info ranking + VIF.',
    'Handling Regimes: Add regime interaction features (feat * regime) or train regime-conditional sub-models.',
    'Window Engineering: Use 3-day windows for hv_20/range_5, 5-day for ret_1d, 30-day for vol_ratio_5.',
    'Model Selection: XGBoost with GPU is preferred. Consider LightGBM for faster regime-conditional training.',
    'Evaluation: Precision@K (top 5/10/20 picks per day) is more actionable than global AUC.',
    'Cross-Validation: Yearly walkforward with 2016-2022 train, 2023-2025 test, 2026 final holdout.',
    'Pattern Integration: Encode double_bottom and fvg_bearish as binary features in the final model.',
    'Seasonal Adjustment: Include month dummies; consider quarter-end effects.',
    'Production Pipeline: Daily batch inference with symbol-level ranking by probability.',
]
for r in recs:
    pdf.bullet(r, size=8.5)

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
pdf.output(str(OUT))
print(f'PDF report generated: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)')
print(f'Time: {time.time()-t0:.0f}s')
