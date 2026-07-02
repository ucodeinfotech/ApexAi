"""
Generate comprehensive PDF report of all backtesting results
"""
from fpdf import FPDF
import os

OUTPUT = os.path.join(os.path.dirname(__file__) or ".", "Backtest_Report.pdf")

class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Bullish Engulfing Strategy - Comprehensive Backtest Report", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 60, 110)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 60, 110)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, txt):
        self.set_font("Courier", "", 9)
        self.multi_cell(0, 4.5, txt)
        self.ln(2)

    def body_text_bold(self, txt):
        self.set_font("Courier", "B", 9)
        self.multi_cell(0, 4.5, txt)
        self.ln(2)

    def key_value(self, key, val):
        self.set_font("Courier", "", 9)
        self.cell(70, 5, f"  {key}:")
        self.set_font("Courier", "B", 9)
        self.cell(0, 5, val, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(30, 60, 110)
        self.set_text_color(255, 255, 255)
        for i, (c, w) in enumerate(zip(cols, widths)):
            self.cell(w, 6, c, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, vals, widths, bold=False, highlight=False):
        self.set_font("Courier", "B" if bold else "", 7)
        if highlight:
            self.set_fill_color(220, 235, 255)
        else:
            self.set_fill_color(255, 255, 255)
        for v, w in zip(vals, widths):
            align = "R" if isinstance(v, str) and v.startswith(("+", "-", "$")) else "L" if not isinstance(v, (int, float)) else "R"
            self.cell(w, 4.5, str(v), border=1, fill=True, align=align)
        self.ln()


def build_report():
    pdf = Report("P", "mm", "A4")
    pdf.alias_nb_pages()

    # ============================================================
    # TITLE PAGE
    # ============================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 110)
    pdf.cell(0, 15, "Bullish Engulfing Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 18)
    pdf.cell(0, 12, "Comprehensive Backtest Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "All Strategies  x  All Improvement Variants  x  Deep Statistical Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.cell(0, 8, "Instruments: NIFTY50 + SENSEX  |  2015 - 2026  |  5-min Data", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 7, "Generated: June 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ============================================================
    # EXECUTIVE SUMMARY
    # ============================================================
    pdf.add_page()
    pdf.section_title("1. EXECUTIVE SUMMARY")

    pdf.sub_title("Overall Recommendation")
    pdf.body_text(
        "  Engulf_Raw + CH55 + WL + Skip2\n"
        "  ========================================\n"
        "  Net PnL:     +1,395,534 pts\n"
        "  Win Rate:    69%\n"
        "  W/L Ratio:   5.8x\n"
        "  Max DD:      19,362 pts\n"
        "  Net/MDD:     72.1x\n"
        "  Trades:      564 over 12 years (~47/yr)\n"
        "  Profitable Years: 9 of 12"
    )

    pdf.sub_title("Key Findings")
    pdf.body_text(
        "  - Best risk-adjusted return: CH55 + WL + Skip2 on Engulf_Raw (Net/MDD 72.1x)\n"
        "  - Best raw return:           CH55 + 2w1l on Engulf_Raw (+1,889,157 pts, MDD 223K)\n"
        "  - CH55 dominates CH45:       +1,235,116 pts vs +660,523 pts (87% improvement)\n"
        "  - Loss autocorrelation confirmed (chi-sq p<0.001): WR drops 25.5% after a loss\n"
        "  - Skip2 directly addresses loss streaks: breaks the autocorrelation chain\n"
        "  - WL (anti-martingale) sizing: W/L ratio jumps from 2.5x to 4.7x with position sizing\n"
        "  - 60 configurations tested across 5 entry strategies"
    )

    # ============================================================
    # FINAL RANKING TABLE
    # ============================================================
    pdf.add_page()
    pdf.section_title("2. FINAL CONSOLIDATED RANKING (Top 30)")

    cols = ["#", "Strategy", "Config", "Trades", "Net Pts", "WR", "W/L", "MDD", "Net/MDD"]
    widths = [8, 30, 40, 14, 24, 12, 12, 20, 18]

    # Top 30 data
    top30 = [
        (1,"Engulf_Raw","CH55+2w1l",1671,"+1,889,157","52%","2.6x","+223,397","8.5x"),
        (2,"Engulf_Raw","CH55+Skip2",987,"+1,522,621","69%","3.1x","+53,351","28.5x"),
        (3,"Engulf_Raw","CH55+WL+Skip2",564,"+1,395,534","69%","5.8x","+19,362","72.1x"),
        (4,"Engulf_Raw","CH55+WL",1671,"+1,296,991","52%","4.7x","+66,227","19.6x"),
        (5,"Engulf_Raw","CH55",1671,"+1,235,116","52%","2.5x","+153,981","8.0x"),
        (6,"Engulf_Raw","CH60",1671,"+1,201,005","51%","2.4x","+223,397","5.4x"),
        (7,"Engulf_Raw","CH45+2w1l",1671,"+1,014,604","48%","2.3x","+155,411","6.5x"),
        (8,"Engulf_Raw","CH45+3w2l",1671,"+1,009,163","48%","2.3x","+155,411","6.5x"),
        (9,"Engulf_Raw","DynCH+Skip2",987,"+984,455","64%","2.6x","+65,478","15.0x"),
        (10,"Engulf_Raw","CH45+Skip2",987,"+879,465","65%","2.4x","+35,145","25.0x"),
        (11,"Engulf_Raw","DynCH+WL",1671,"+850,951","46%","4.2x","+54,147","15.7x"),
        (12,"Engulf_Raw","DynCH+WL+Skip2",564,"+838,156","64%","4.4x","+23,755","35.3x"),
        (13,"Comb_OR","CH55+2w1l",731,"+834,160","49%","2.8x","+119,127","7.0x"),
        (14,"Engulf_Raw","CH45+Skip3",853,"+832,395","67%","2.4x","+32,930","25.3x"),
        (15,"Engulf_Raw","CH45+Skip1",1183,"+831,717","58%","2.3x","+53,276","15.6x"),
        (16,"Engulf_Raw","CH45+WL",1671,"+775,569","48%","4.3x","+49,789","15.6x"),
        (17,"Engulf_Filt","CH55+2w1l",652,"+754,263","50%","2.7x","+111,559","6.8x"),
        (18,"Engulf_Raw","DynCH",1671,"+733,946","46%","2.4x","+171,793","4.3x"),
        (19,"Engulf_Raw","CH45_base",1671,"+660,523","48%","2.2x","+105,693","6.2x"),
        (20,"Comb_OR","CH55+Skip2",440,"+654,023","64%","2.8x","+23,367","28.0x"),
        (21,"BigCandle","CH55+2w1l",480,"+645,452","53%","2.7x","+61,044","10.6x"),
        (22,"Engulf_Raw","CH35",1671,"+618,593","48%","2.3x","+76,173","8.1x"),
        (23,"Engulf_Filt","CH55+Skip2",397,"+608,828","65%","2.8x","+19,422","31.3x"),
        (24,"Comb_OR","CH55+WL",731,"+601,656","48%","4.2x","+63,840","9.4x"),
        (25,"Engulf_Filt","CH55+WL+Skip2",240,"+593,445","67%","4.9x","+9,093","65.3x"),
        (26,"Comb_OR","CH55+WL+Skip2",251,"+567,124","66%","4.7x","+8,627","65.7x"),
        (27,"Comb_OR","CH55",731,"+542,860","48%","2.4x","+113,708","4.8x"),
        (28,"Engulf_Filt","CH55+WL",652,"+525,938","47%","4.3x","+37,380","14.1x"),
        (29,"Engulf_Filt","CH55",652,"+492,170","47%","2.4x","+85,568","5.8x"),
        (30,"BigCandle","CH55+WL+Skip2",181,"+477,748","66%","5.4x","+8,571","55.7x"),
    ]

    # Check if we need to squeeze: use smaller font
    pdf.set_font("Courier", "", 6.5)
    pdf.table_header(cols, widths)
    for r in top30:
        hl = r[0] <= 3
        pdf.table_row([str(r[0]), r[1], r[2], str(r[3]), r[4], r[5], r[6], r[7], r[8]], widths, bold=hl, highlight=hl)
    pdf.ln(3)

    pdf.sub_title("Legend")
    pdf.body_text(
        "  Strategy:    Engulf_Raw = pure bullish engulfing  |  Engulf_Filt = filtered (EMA50>200, ADX>20)\n"
        "               BigCandle = 1.5x body reversal  |  Comb_OR = engulf OR big candle  |  Sir = ATR body\n"
        "  Config:      CH55 = 55-point trailing stop  |  WL = win/loss anti-martingale sizing\n"
        "               Skip2 = skip 2 trades after loss  |  2w1l = double after win, halve after loss\n"
        "               DynCH = monthly CH from best historical value  |  base = CH45 baseline\n"
        "  MDD:         Intra-swarm maximum drawdown (points)"
    )

    # ============================================================
    # BEST PER CATEGORY
    # ============================================================
    pdf.add_page()
    pdf.section_title("3. BEST CONFIGURATION PER METRIC")

    metrics = [
        ("BEST RAW RETURN", "Engulf_Raw + CH55 + 2w1l", "+1,889,157 pts", "52% WR, 2.6x W/L, MDD 223K"),
        ("BEST RISK-ADJUSTED", "Engulf_Raw + CH55 + WL + Skip2", "+1,395,534 pts", "69% WR, 5.8x W/L, MDD 19K, Net/MDD 72.1x"),
        ("BEST WIN RATE", "Engulf_Raw + CH55 + Skip2", "+1,522,621 pts", "69% WR, 3.1x W/L, MDD 53K"),
        ("BEST FILTERED WIN RATE", "Engulf_Filt + CH55 + WL + Skip2", "+593,445 pts", "67% WR, 4.9x W/L, MDD 9K"),
        ("BEST W/L RATIO", "Engulf_Raw + CH55 + WL + Skip2", "+1,395,534 pts", "5.8x W/L, 69% WR, MDD 19K"),
        ("LOWEST MDD", "Sir + CH55 + Skip2", "+59,068 pts", "MDD 5K, 54% WR, 4.2x W/L"),
        ("BEST FILTERED RETURN", "Engulf_Filt + CH55 + 2w1l", "+754,263 pts", "50% WR, 2.7x W/L, MDD 112K"),
        ("BEST BIGCANDLE", "BigCandle + CH55 + WL + Skip2", "+477,748 pts", "66% WR, 5.4x W/L, MDD 9K"),
        ("BEST COMBO (OR)", "Comb_OR + CH55 + WL + Skip2", "+567,124 pts", "66% WR, 4.7x W/L, MDD 9K"),
        ("BEST SIR", "Sir + CH55 + 2w1l", "+65,710 pts", "44% WR, 2.9x W/L, MDD 20K"),
    ]

    widths2 = [50, 65, 30, 45]
    pdf.table_header(["Category", "Configuration", "Net Pts", "Details"], widths2)
    for r in metrics:
        pdf.table_row(list(r), widths2, highlight=r[1].startswith("Engulf_Raw + CH55 + WL + Skip2"))
    pdf.ln(5)

    pdf.section_title("4. BEST PER ENTRY STRATEGY")
    pdf.sub_title("Maximum Return Configuration")
    strategies_raw = [
        ("Engulf_Raw",   "CH55+2w1l",      "+1,889,157", "52%", "2.6x", "+223,397"),
        ("Engulf_Filt",  "CH55+2w1l",        "+754,263", "50%", "2.7x", "+111,559"),
        ("BigCandle",    "CH55+2w1l",        "+645,452", "53%", "2.7x",  "+61,044"),
        ("Comb_OR",      "CH55+2w1l",        "+834,160", "49%", "2.8x", "+119,127"),
        ("Sir",          "CH55+2w1l",         "+65,710", "44%", "2.9x",  "+20,300"),
    ]
    widths3 = [26, 28, 22, 12, 12, 20]
    pdf.table_header(["Strategy", "Config", "Net Pts", "WR", "W/L", "MDD"], widths3)
    for r in strategies_raw:
        pdf.table_row(list(r), widths3)

    pdf.ln(3)
    pdf.sub_title("Best Risk-Adjusted Configuration")
    strategies_ra = [
        ("Engulf_Raw",   "CH55+WL+Skip2", "+1,395,534", "69%", "5.8x", "+19,362", "72.1x"),
        ("Engulf_Filt",  "CH55+WL+Skip2",   "+593,445", "67%", "4.9x",  "+9,093", "65.3x"),
        ("BigCandle",    "CH55+WL+Skip2",   "+477,748", "66%", "5.4x",  "+8,571", "55.7x"),
        ("Comb_OR",      "CH55+WL+Skip2",   "+567,124", "66%", "4.7x",  "+8,627", "65.7x"),
        ("Sir",          "CH55+Skip2",       "+59,068", "54%", "4.2x",  "+4,996", "11.8x"),
    ]
    widths4 = [26, 28, 22, 12, 12, 18, 18]
    pdf.table_header(["Strategy", "Config", "Net Pts", "WR", "W/L", "MDD", "Net/MDD"], widths4)
    for r in strategies_ra:
        pdf.table_row(list(r), widths4)

    # ============================================================
    # YEAR-BY-YEAR COMPARISON
    # ============================================================
    pdf.add_page()
    pdf.section_title("5. YEAR-BY-YEAR: TOP 5 CONFIGURATIONS")

    pdf.body_text(
        "  Year   Eng_Raw+CH55+2w1l   Eng_Raw+CH55+Skip2   Eng_Raw+CH55+WL+Skip2   Eng_Raw+CH55+WL     Eng_Raw+CH55\n"
        "  ------------------------------------------------------------------------------------------------------"
    )
    years = [
        ("2015","-15,176","-2,903","-1,129","-4,048","-9,911"),
        ("2016","+127,731","+76,178","+61,995","+72,137","+86,433"),
        ("2017","+61,526","+61,982","+56,362","+45,888","+43,017"),
        ("2018","+25,206","+39,792","+29,330","+26,588","+12,606"),
        ("2019","+28,876","+31,930","+30,318","+16,748","+19,172"),
        ("2020","+357,900","+241,112","+215,757","+215,967","+237,012"),
        ("2021","+494,230","+355,086","+359,080","+357,254","+326,474"),
        ("2022","-44,697","+39,553","+3,895","-43,032","-45,814"),
        ("2023","+522,847","+358,902","+283,337","+275,867","+347,349"),
        ("2024","+470,654","+354,789","+366,117","+362,197","+317,120"),
        ("2025","-47,005","-5,170","-1,926","-13,213","-33,269"),
        ("2026","-92,935","-28,630","-7,602","-15,361","-65,074"),
        ("Total","+1,889,157","+1,522,621","+1,395,534","+1,296,991","+1,235,116"),
    ]
    ywidths = [16, 34, 34, 34, 30, 28]
    pdf.table_header(["Year","CH55+2w1l","CH55+Skip2","CH55+WL+Skip2","CH55+WL","CH55"], ywidths)
    for r in years:
        hl = r[0] == "Total"
        pdf.table_row(list(r), ywidths, bold=hl, highlight=hl)

    pdf.ln(3)
    pdf.body_text(
        "  Notes:\n"
        "  - 2020-2021-2024 were exceptional years (COVID recovery + bull run)\n"
        "  - 2022 was challenging (rising rates, bear market) - only Skip2 variants stayed positive\n"
        "  - 2025-2026 partial data shows losses; CH55+WL+Skip2 best at limiting downside\n"
        "  - 2023 was the best year overall (+522K with CH55+2w1l)"
    )

    # ============================================================
    # DEEP STATISTICAL ANALYSIS
    # ============================================================
    pdf.add_page()
    pdf.section_title("6. DEEP STATISTICAL ANALYSIS (1,671 Trades)")

    pdf.sub_title("6.1 Monthly Breakdown")
    pdf.body_text(
        "  Month Trades      Net    WR   W/L     MDD  AvgWin  AvgLoss WinCons\n"
        "  ----------------------------------------------------------------"
    )
    months = [
        ("Jan","152","-37,385","32%","1.1x","+45,448","+787","+723","33%"),
        ("Feb","145","+10,856","40%","1.7x","+31,883","+1,337","+767","42%"),
        ("Mar","130","+3,605","44%","1.3x","+19,878","+1,318","+980","58%"),
        ("Apr","134","+50,090","47%","2.0x","+15,587","+1,805","+896","50%"),
        ("May","151","+192,744","61%","3.4x","+18,272","+2,586","+766","67%"),
        ("Jun","129","+190,147","79%","2.7x","+6,000","+2,069","+775","91%"),
        ("Jul","146","+93,671","55%","2.6x","+16,319","+1,687","+661","45%"),
        ("Aug","153","+40,856","52%","2.0x","+6,541","+950","+482","45%"),
        ("Sep","143","-7,400","37%","1.6x","+30,536","+1,497","+964","27%"),
        ("Oct","112","+60,340","49%","2.3x","+15,801","+2,011","+882","55%"),
        ("Nov","127","+55,700","43%","3.4x","+8,520","+1,713","+504","36%"),
        ("Dec","149","+7,301","38%","1.8x","+17,711","+1,189","+657","36%"),
    ]
    mwidths = [12, 12, 20, 12, 12, 20, 14, 14, 16]
    pdf.table_header(["Mo","Trd","Net","WR","W/L","MDD","AvgWin","AvgLos","WinCon"], mwidths)
    for r in months:
        hl = r[0] in ("May","Jun")
        pdf.table_row(list(r), mwidths, highlight=hl)
    pdf.ln(2)
    pdf.body_text(
        "  Key: WinCons = % of years where month was positive. May/Jun dominate:\n"
        "       - June: 79% WR, 91% year consistency (best month)\n"
        "       - January: 32% WR, 33% consistency (worst month)\n"
        "       - September: 37% WR, 27% consistency"
    )

    pdf.sub_title("6.2 Loss Autocorrelation")
    pdf.body_text(
        "  Condition          WR    Delta vs Base\n"
        "  -----------------------------------------\n"
        "  After 1 win       75.5%    +27.6%\n"
        "  After 1 loss      22.4%    -25.5%\n"
        "  After 2 wins      70.0%    +22.1%\n"
        "  After 2 losses    27.5%    -20.4%\n"
        "  After 3 wins      65.0%    +17.1%\n"
        "  After 3 losses    32.0%    -15.9%\n\n"
        "  Chi-square test: p < 0.001 *** (highly significant)\n"
        "  Base WR: 47.9%\n\n"
        "  => This confirms the Skip2 logic: skipping 2 trades after a loss\n"
        "     breaks the autocorrelation chain and avoids the 22.4% WR zone."
    )

    pdf.sub_title("6.3 Optimal CH Value Analysis")
    pdf.body_text(
        "  CH Value    Net PnL       WR    Best Fit %\n"
        "  --------------------------------------------\n"
        "  CH25      +293,688      46%     31.1%\n"
        "  CH30      +380,340      47%     16.6%\n"
        "  CH35      +618,593      48%      9.1%\n"
        "  CH40      +599,076      49%      6.8%\n"
        "  CH45      +660,523      48%      3.0%\n"
        "  CH50      +804,842      50%     11.8%\n"
        "  CH55    +1,235,116      52%     13.4%\n"
        "  CH60    +1,201,005      51%      8.2%\n\n"
        "  CH55 gives highest net PnL (widest stop = captures biggest trends).\n"
        "  CH25 selected as 'best' 31% of time because it minimizes individual losses.\n"
        "  Wider stops (CH55/60) capture trends that tight stops miss."
    )

    pdf.add_page()
    pdf.sub_title("6.4 Winner vs Loser Feature Comparison")
    pdf.body_text(
        "  Feature               Winner    Loser     Diff    p-val  Signif\n"
        "  ---------------------------------------------------------------\n"
        "  body                 +79.6    +76.6     +3.0   0.6249   -\n"
        "  prev_body            +31.2    +28.8     +2.4   0.3902   -\n"
        "  body_ratio            +8.0    +10.2     -2.2   0.2472   -\n"
        "  candle_range        +122.1   +119.6     +2.5   0.7452   -\n"
        "  atr14               +118.3   +123.1     -4.8   0.3581   -\n"
        "  ret_20h              +0.005   +0.003   +0.002 0.0102   *\n"
        "  close_vs_ema50       +0.016   +0.012   +0.003 0.0514   ns\n"
        "  n_consec_red         +5.04    +4.92    +0.12  0.0951   ns\n"
        "  day_of_week          +2.12    +1.98    +0.15  0.0305   *\n"
        "  range_vs_avg         +1.08    +1.05    +0.03  0.1785   -\n\n"
        "  Only 2 features significant (p<0.05): ret_20h and day_of_week.\n"
        "  Most candle features are NOT predictive.\n"
        "  => This is a market regime strategy, not a pattern-recognition one."
    )

    pdf.sub_title("6.5 ML Feature Importance (Random Forest)")
    pdf.body_text(
        "  Model AUC: 0.546 (barely above random)\n\n"
        "  Rank  Feature           Importance\n"
        "  -----------------------------------\n"
        "  1     ret_20h             0.113\n"
        "  2     close_vs_ema50      0.110\n"
        "  3     close_vs_ema200     0.099\n"
        "  4     range_vs_avg        0.088\n"
        "  5     atr14_pct           0.085\n"
        "  6     range_ratio         0.078\n"
        "  7     atr14               0.072\n"
        "  8     body_ratio          0.062\n\n"
        "  No single feature dominates. The strategy's edge comes from\n"
        "  market regime x position management, not candle patterns."
    )

    pdf.sub_title("6.6 Year Level Analysis")
    pdf.body_text(
        "  Year        Net      WR   W/L    N  AvgATR%  AvgWin  AvgLoss  BestMo   WorstMo\n"
        "  ----------------------------------------------------------------------\n"
        "  2015     -7,256   30%  0.9x   81   0.45%    +183    +204     +932   -2,111\n"
        "  2016    +49,506   52%  2.7x  175   0.38%    +824    +303  +20,812   -4,503\n"
        "  2017    +18,347   53%  1.2x  182   0.27%    +761    +653  +17,884   -8,799\n"
        "  2018    +28,671   50%  1.7x  163   0.36%    +908    +547  +18,686   -9,527\n"
        "  2019     -1,385   39%  1.5x  159   0.37%    +676    +446  +16,097  -11,519\n"
        "  2020   +207,225   64%  3.6x  149   0.62%  +2,594    +726  +60,256   -4,540\n"
        "  2021   +224,251   53%  3.4x  154   0.40%  +3,677  +1,073  +91,494  -14,401\n"
        "  2022    -11,395   41%  1.3x  146   0.45%  +1,783  +1,377  +31,701  -27,136\n"
        "  2023   +178,938   64%  3.4x  129   0.28%  +2,623    +769  +48,594   -3,710\n"
        "  2024    +40,100   40%  2.2x  174   0.36%  +1,808    +832  +74,161  -13,742\n"
        "  2025    -18,350   39%  1.1x  117   0.32%    +871    +823   +9,482   -8,235\n"
        "  2026    -48,129   24%  0.6x   42   0.44%  +1,068  +1,838   -2,447  -14,057"
    )

    pdf.sub_title("6.7 Quantitative Trading Rules")
    pdf.body_text(
        "  Condition                Trade when       WR     Net\n"
        "  -----------------------------------------------------\n"
        "  atr14_pct                 LOW (vol<0.28%)  51%  +336K\n"
        "  ret_20h                   LOW (dipped)     48%  +379K\n"
        "  close_vs_ema50            HIGH (bullish)   49%  +342K\n"
        "  body                      LOW (small cdl)  48%  +427K\n"
        "  range_vs_avg              HIGH (expanding) 49%  +366K\n"
        "  n_consec_red              HIGH (oversold)  51%  +348K"
    )

    # ============================================================
    # RECOMMENDED STRATEGY DETAIL
    # ============================================================
    pdf.add_page()
    pdf.section_title("7. RECOMMENDED STRATEGY: Engulf_Raw + CH55 + WL + Skip2")

    pdf.key_value("Entry", "Bullish Engulfing (close > open, prev close < prev open)")
    pdf.key_value("Exit", "Trailing stop: CH55 (55 points from highest close since entry)")
    pdf.key_value("Position Sizing", "Win/Loss anti-martingale: 2 contracts after win, 1 after loss")
    pdf.key_value("Loss Filter", "Skip2: skip next 2 trades after any loss")
    pdf.key_value("Instruments", "NIFTY50 + SENSEX")
    pdf.key_value("Timeframe", "5-min data, any session")
    pdf.key_value("Avg Trades/Year", "~47")
    pdf.key_value("Net PnL (12yr)", "+1,395,534 pts")
    pdf.key_value("Win Rate", "69%")
    pdf.key_value("W/L Ratio", "5.8x")
    pdf.key_value("Max Drawdown", "19,362 pts")
    pdf.key_value("Net/MDD", "72.1x")
    pdf.key_value("Profitable Years", "9 out of 12")

    pdf.ln(5)
    pdf.sub_title("Year-by-Year Performance")
    yyw = [("2015","-1,129","31%","1.0x","35"),("2016","+61,995","71%","3.5x","111"),
           ("2017","+56,362","69%","3.7x","114"),("2018","+29,330","64%","1.9x","95"),
           ("2019","+30,318","65%","4.3x","93"),("2020","+215,757","81%","5.2x","108"),
           ("2021","+359,080","78%","12.0x","107"),("2022","+3,895","52%","1.1x","75"),
           ("2023","+283,337","82%","5.7x","95"),("2024","+366,117","81%","8.7x","127"),
           ("2025","-1,926","48%","0.8x","58"),("2026","-7,602","7%","0.8x","15"),
           ("Total","+1,395,534","69%","5.8x","564")]
    yw = [12, 22, 12, 12, 14]
    pdf.table_header(["Year","Net","WR","W/L","Trades"], yw)
    for r in yyw:
        hl = r[0] == "Total"
        pdf.table_row(list(r), yw, bold=hl, highlight=hl)

    pdf.ln(5)
    pdf.sub_title("Why This Configuration Wins")
    pdf.body_text(
        "  1. CH55 (wide stop): Captures bigger trends that tighter stops miss.\n"
        "     Net improves from +660K (CH45) to +1.24M (CH55) for the same entries.\n\n"
        "  2. WL Sizing (anti-martingale): After wins, position size doubles to\n"
        "     capitalize on momentum. After losses, position size halves to conserve\n"
        "     capital. W/L ratio jumps from 2.5x to 4.7x vs equal sizing.\n\n"
        "  3. Skip2 (loss filter): Loss autocorrelation is statistically significant\n"
        "     (p<0.001). WR drops from 47.9% to 22.4% after a loss. Skip2 breaks\n"
        "     this chain by sitting out the next 2 trades, avoiding the highest-risk\n"
        "     period. Net/MDD improves from 8.0x (CH55 only) to 72.1x.\n\n"
        "  4. Monthly patterns: June (79% WR) and May (61% WR) are best months.\n"
        "     Consider reducing size or skipping January (32% WR) and September (37%).\n\n"
        "  5. Year patterns: Strategy excels in trending years (2020-2024).\n"
        "     Skip2 variants stay positive even in difficult years (2022)."
    )

    # ============================================================
    # METHODOLOGY
    # ============================================================
    pdf.add_page()
    pdf.section_title("8. METHODOLOGY & DATA")

    pdf.sub_title("Data Sources")
    pdf.body_text(
        "  - Instruments: NIFTY50 Index, SENSEX Index\n"
        "  - Timeframe: 5-minute candles\n"
        "  - Period: January 2015 - June 2026 (~12 years)\n"
        "  - Data points: ~350,000 5-min bars per instrument"
    )

    pdf.sub_title("Entry Strategies")
    pdf.body_text(
        "  1. Engulf_Raw: Bullish engulfing pattern (current close > current open AND\n"
        "     previous close < previous open). No additional filters.\n\n"
        "  2. Engulf_Filt: Same engulfing pattern + EMA50 > EMA200 (uptrend) +\n"
        "     ADX(14) > 20 (trending) + 9:15-14:45 session filter.\n\n"
        "  3. BigCandle: Current body > 1.5x average body of last 20 candles,\n"
        "     closing near high (body > 0.6x range). Reversal context.\n\n"
        "  4. Sir: Body > 1.0x ATR(14) for directional conviction + all filters.\n\n"
        "  5. Comb_OR: Engulfing OR BigCandle with filters."
    )

    pdf.sub_title("Exit Strategies")
    pdf.body_text(
        "  - CH(N): Trailing stop N points from highest close since entry\n"
        "  - DynCH: Monthly CH value chosen from best historical performance\n"
        "  - FixTP: Fixed take-profit (unprofitable, included for completeness)\n"
        "  - Values tested: CH7, CH15, CH25, CH30, CH35, CH40, CH45, CH50, CH55, CH60"
    )

    pdf.sub_title("Improvement Variants")
    pdf.body_text(
        "  - WL (Win/Loss sizing): 2 contracts after win, 1 after loss\n"
        "  - Skip1/2/3: Skip N trades following a loss\n"
        "  - 2w1l: Double after win, halve after loss (aggressive anti-martingale)\n"
        "  - 3w2l: Triple after win, halve after loss\n"
        "  - WL + Skip2: Combined W/L sizing with skip-2 loss filter"
    )

    pdf.sub_title("Performance Metrics")
    pdf.body_text(
        "  - Net PnL: Total points profit/loss (instrument-agnostic)\n"
        "  - WR: Win Rate = winning trades / total trades\n"
        "  - W/L: Average win size / average loss size\n"
        "  - MDD: Intra-swarm maximum drawdown (peak-to-trough)\n"
        "  - Net/MDD: Risk-adjusted return ratio\n"
        "  - WinCons: % of years where a given month was net positive"
    )

    pdf.sub_title("Statistical Methods")
    pdf.body_text(
        "  - Chi-square test: Loss autocorrelation significance\n"
        "  - Welch's t-test: Winner vs loser feature differences\n"
        "  - Random Forest: Feature importance (100 estimators, AUC metric)\n"
        "  - Pearson correlation: Feature-outcome relationships\n"
        "  - All tests use baseline CH45 exits on Engulf_Raw (1,671 trades)"
    )

    # ============================================================
    # OUTPUT
    # ============================================================
    pdf.output(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build_report()
    print(f"PDF generated: {path}")
    print(f"Size: {os.path.getsize(path):,} bytes")
