"""Generate professional PDF report for pivot breakout backtest"""
import os, glob
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime

OUTPUT_DIR = "backtest_results"
PDF_PATH = os.path.join(OUTPUT_DIR, "backtest_report_v2.pdf")

# --- Collect all stats ---
all_stats = []
for f in sorted(glob.glob(f"{OUTPUT_DIR}/*_trades_v2.csv")):
    sym = os.path.basename(f).replace("_trades_v2.csv", "")
    df = pd.read_csv(f)

    total = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = round(wc/total*100, 2) if total else 0
    gp = round(wins["net_pnl"].sum(), 2) if wc else 0
    gl = round(losses["net_pnl"].sum(), 2) if lc else 0
    np_ = round(df["net_pnl"].sum(), 2)
    pf = round(abs(gp/gl), 2) if gl != 0 else 0
    avg_r = round(df["r"].mean(), 2)

    longs = df[df["type"] == "LONG"]
    shorts = df[df["type"] == "SHORT"]
    sl_hit = df[df["reason"] == "SL"]
    tp_hit = df[df["reason"] == "TP"]

    df_s = df.sort_values("exit_time").reset_index(drop=True)
    df_s["cum"] = df_s["net_pnl"].cumsum()
    df_s["peak"] = df_s["cum"].cummax()
    df_s["dd"] = df_s["peak"] - df_s["cum"]
    mdd = round(df_s["dd"].max(), 2)

    avg_w = round(wins["net_pnl"].mean(), 2) if wc else 0
    avg_l = round(losses["net_pnl"].mean(), 2) if lc else 0
    sharpe = round(df["r"].mean()/df["r"].std()*np.sqrt(total), 2) if df["r"].std() > 0 else 0
    charges_total = round(df["charges"].sum(), 2)

    all_stats.append({
        "symbol": sym,
        "trades": total,
        "wins": wc,
        "losses": lc,
        "win_rate": wr,
        "net_pnl": np_,
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": pf,
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "avg_r": avg_r,
        "max_dd": mdd,
        "sharpe": sharpe,
        "charges": charges_total,
        "long_pct": round(len(longs)/total*100,1) if total else 0,
        "short_pct": round(len(shorts)/total*100,1) if total else 0,
        "sl_pct": round(len(sl_hit)/total*100,1) if total else 0,
        "tp_pct": round(len(tp_hit)/total*100,1) if total else 0
    })

all_stats.sort(key=lambda x: x["net_pnl"])

# Totals
t_trades = sum(s["trades"] for s in all_stats)
t_wins = sum(s["wins"] for s in all_stats)
t_losses = sum(s["losses"] for s in all_stats)
t_wr = round(t_wins/t_trades*100, 2)
t_np = round(sum(s["net_pnl"] for s in all_stats), 2)
t_gp = round(sum(s["gross_profit"] for s in all_stats), 2)
t_gl = round(sum(s["gross_loss"] for s in all_stats), 2)
t_pf = round(abs(t_gp/t_gl), 2) if t_gl != 0 else 0
t_charges = round(sum(s["charges"] for s in all_stats), 2)

# Combined trade analysis
all_trades = []
for f in glob.glob(f"{OUTPUT_DIR}/*_trades_v2.csv"):
    all_trades.append(pd.read_csv(f))
combined = pd.concat(all_trades, ignore_index=True)

longs = combined[combined["type"] == "LONG"]
shorts = combined[combined["type"] == "SHORT"]
sl_trades = combined[combined["reason"] == "SL"]
tp_trades = combined[combined["reason"] == "TP"]

# --- PDF Class ---
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100,100,100)
        self.cell(0, 6, "Pivot Breakout v2 (Optimized) | Oct 2016 - Jun 2026 | Nifty 50 Stocks", align="C")
        self.ln(8)
        self.set_draw_color(200,200,200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128,128,128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 60, 110)
        self.ln(4)
        self.cell(0, 10, title)
        self.ln(8)
        self.set_draw_color(30, 60, 110)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title)
        self.ln(7)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def colored_cell(self, w, h, txt, align="C", fill=False, color=None):
        if color:
            self.set_fill_color(*color)
        self.cell(w, h, txt, align=align, fill=fill)

    def add_table(self, headers, data, col_widths, flag_row=None, highlight_col=None):
        """Render a table with optional row flags and highlight column index."""
        # Header row
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(30, 60, 110)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 7)
        for ri, row in enumerate(data):
            # Determine row color
            fill = False
            if flag_row:
                flag = flag_row(ri, row)
                if flag == "negative":
                    self.set_fill_color(255, 235, 235)
                    fill = True
                elif flag == "positive":
                    self.set_fill_color(235, 255, 235)
                    fill = True
                elif flag == "header":
                    self.set_fill_color(240, 245, 255)
                    fill = True
                elif flag == "worst":
                    self.set_fill_color(255, 215, 215)
                    fill = True
                elif flag == "best":
                    self.set_fill_color(215, 255, 215)
                    fill = True

            self.set_text_color(40, 40, 40)
            for i, val in enumerate(row):
                txt = str(val)
                # Highlight the worst column
                if highlight_col and i == highlight_col:
                    self.set_text_color(200, 0, 0)
                else:
                    self.set_text_color(40, 40, 40)
                self.cell(col_widths[i], 5.5, txt, border=1, align="C", fill=fill)
            self.ln()

            # Page break if near end
            if self.get_y() > 265:
                self.add_page()
                # Re-print header
                self.set_font("Helvetica", "B", 7.5)
                self.set_fill_color(30, 60, 110)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
                self.ln()
                self.set_font("Helvetica", "", 7)


# --- Build PDF ---
pdf = PDF(orientation="L", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ------- PAGE 1: COVER -------
pdf.add_page()
pdf.ln(30)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(30, 60, 110)
pdf.cell(0, 15, "PIVOT BREAKOUT STRATEGY V2", align="C")
pdf.ln(12)
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Optimized Backtest Results Report", align="C")
pdf.ln(20)

pdf.set_draw_color(30, 60, 110)
pdf.line(60, pdf.get_y(), 240, pdf.get_y())
pdf.ln(20)

pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(60, 60, 60)
summary_lines = [
    f"Universe:   50 Nifty 50 Stocks",
    f"Period:     October 2016 - June 2026 (~10 years)",
    f"Timeframes: 15-min signal / 1-min entry",
    f"Total Trades:  {t_trades:,}",
    f"Generated:  {datetime.now().strftime('%d-%b-%Y %I:%M %p')}",
]
for line in summary_lines:
    pdf.cell(0, 8, line, align="C")
    pdf.ln(6)

pdf.ln(10)
pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(130, 130, 130)
pdf.cell(0, 6, "Disclaimer: This is a historical backtest for research purposes only.", align="C")
pdf.ln(5)
pdf.cell(0, 6, "Past performance does not guarantee future results.", align="C")

# ------- PAGE 2: EXECUTIVE SUMMARY -------
pdf.add_page(orientation="L")
pdf.section_title("1. EXECUTIVE SUMMARY")

pdf.body_text(
    f"A comprehensive backtest of the optimized Pivot Breakout Strategy v2 was conducted across all 50 Nifty 50 stocks "
    f"over approximately 10 years of intraday data. Seven key improvements were applied (close-trigger, market entry, "
    f"ATR-based SL, trend filter, volume confirmation, Rs5 brokerage, partial profit booking). "
    f"The strategy generated a total of {t_trades:,} trades, of which only {t_wins:,} ({t_wr}%) were profitable. "
    f"The net result is a loss of Rs {t_np:,.2f} across all stocks, with total charges amounting to Rs {t_charges:,.2f}."
)

# Key metrics box
pdf.set_draw_color(30, 60, 110)
pdf.set_fill_color(245, 248, 255)
pdf.rect(12, pdf.get_y(), 272, 55, style="D")
box_y = pdf.get_y() + 2

metrics = [
    ("Total Trades", f"{t_trades:,}", "Win Rate", f"{t_wr}%"),
    ("Winning Trades", f"{t_wins:,}", "Losing Trades", f"{t_losses:,}"),
    ("Net P&L", f"Rs {t_np:,.2f}", "Profit Factor", f"{t_pf}"),
    ("Total Charges", f"Rs {t_charges:,.2f}", "Avg R Multiple", f"{sum(s['avg_r'] for s in all_stats)/len(all_stats):.2f}"),
    ("Profitable Stocks", "0 / 50", "Worst Loss (Stock)", f"Rs {max(s['net_pnl'] for s in all_stats):,.2f}"),
]

pdf.set_xy(18, box_y + 2)
pdf.set_font("Helvetica", "B", 12)
pdf.set_text_color(30, 60, 110)
pdf.cell(0, 8, "Key Results")
pdf.ln(10)

pdf.set_xy(18, pdf.get_y())
for i, (label1, val1, label2, val2) in enumerate(metrics):
    col = 18 if i % 2 == 0 else 155
    y = box_y + 16 + (i // 2) * 12
    
    pdf.set_xy(col, y)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(30, 6, label1)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(50, 6, val1)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(30, 6, label2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(50, 6, val2)

pdf.set_y(box_y + 55)
pdf.ln(4)

pdf.sub_title("2. TRADE ANALYSIS")
metrics_rows = [
    ("Long Trades", f"{len(longs):,}", f"{len(longs)/len(combined)*100:.1f}%", f"Rs {longs['net_pnl'].mean():.2f}"),
    ("Short Trades", f"{len(shorts):,}", f"{len(shorts)/len(combined)*100:.1f}%", f"Rs {shorts['net_pnl'].mean():.2f}"),
    ("SL Hit", f"{len(sl_trades):,}", f"{len(sl_trades)/len(combined)*100:.1f}%", f"Rs {sl_trades['net_pnl'].mean():.2f}"),
    ("TP Hit", f"{len(tp_trades):,}", f"{len(tp_trades)/len(combined)*100:.1f}%", f"Rs {tp_trades['net_pnl'].mean():.2f}"),
]
mh = ["Metric", "Count", "% of Total", "Avg P&L"]
mw = [55, 60, 60, 70]
pdf.add_table(mh, metrics_rows, mw)
pdf.ln(4)

pdf.sub_title("3. STOCK PERFORMANCE RANKING")

# Top 5 / Bottom 5
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(60,60,60)
pdf.cell(130, 7, "Best 5 Stocks (Least Negative)")
pdf.cell(130, 7, "Worst 5 Stocks")
pdf.ln(8)

best5 = all_stats[:5]
worst5 = all_stats[-5:]

# Mini table header
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_fill_color(30, 60, 110)
pdf.set_text_color(255,255,255)
mini_h = ["Symbol", "Trades", "Win%", "Net P&L", "AvgR", "Charges"]
for i, h in enumerate(mini_h):
    pdf.cell(20, 6, h, border=1, align="C", fill=True)
pdf.cell(8, 6, "", border=0)  # gap
for i, h in enumerate(mini_h):
    pdf.cell(20, 6, h, border=1, align="C", fill=True)
pdf.ln()

pdf.set_font("Helvetica", "", 7)
for bi, wi in zip(best5, worst5):
    pdf.set_text_color(40,40,40)
    # Best
    cells = [bi["symbol"], f"{bi['trades']:,}", f"{bi['win_rate']:.1f}%", f"Rs{bi['net_pnl']:,.0f}", f"{bi['avg_r']:.2f}", f"Rs{bi['charges']:,.0f}"]
    for ci, c in enumerate(cells):
        if ci == 3:
            pdf.set_text_color(0, 128, 0)
        else:
            pdf.set_text_color(40, 40, 40)
        pdf.cell(20, 5.5, c, border=1, align="C")
    pdf.cell(8, 5.5, "", border=0)
    # Worst
    cells2 = [wi["symbol"], f"{wi['trades']:,}", f"{wi['win_rate']:.1f}%", f"Rs{wi['net_pnl']:,.0f}", f"{wi['avg_r']:.2f}", f"Rs{wi['charges']:,.0f}"]
    for ci, c in enumerate(cells2):
        if ci == 3:
            pdf.set_text_color(200, 0, 0)
        else:
            pdf.set_text_color(40, 40, 40)
        pdf.cell(20, 5.5, c, border=1, align="C")
    pdf.ln()

pdf.ln(6)

# ------- FULL STOCK TABLE -------
pdf.add_page(orientation="L")
pdf.section_title("4. COMPLETE STOCK-WISE RESULTS (ALL 50 STOCKS)")

headers = [
    "#", "Symbol", "Trades", "Wins", "Loss", "Win%",
    "Net P&L", "Gross Profit", "Gross Loss", "PF",
    "Avg R", "Avg Win", "Avg Loss", "Max DD", "Sharpe", "Charges"
]
col_widths = [6, 16, 10, 8, 8, 10, 16, 16, 16, 8, 8, 12, 12, 14, 9, 14]

# Build data rows
rows_data = []
for idx, s in enumerate(all_stats, 1):
    rows_data.append([
        idx, s["symbol"], f"{s['trades']:,}", f"{s['wins']:,}", f"{s['losses']:,}", f"{s['win_rate']:.1f}%",
        f"Rs{s['net_pnl']:,.0f}", f"Rs{s['gross_profit']:,.0f}", f"Rs{s['gross_loss']:,.0f}", f"{s['profit_factor']:.2f}",
        f"{s['avg_r']:.2f}", f"Rs{s['avg_win']:,.0f}", f"Rs{s['avg_loss']:,.0f}", f"Rs{s['max_dd']:,.0f}", f"{s['sharpe']:.1f}", f"Rs{s['charges']:,.0f}"
    ])

def row_flag(ri, row):
    if row[6].startswith("Rs-") or (row[6].startswith("Rs") and float(row[6].replace("Rs", "").replace(",","")) < 0):
        return "negative"
    elif row[6] == "Rs0":
        return "neutral"
    return "positive"

def worst_flag(ri, row):
    # First 5 rows = worst
    if ri < 5:
        return "worst"
    if ri >= len(rows_data) - 5:
        return "best"
    return None

pdf.add_table(headers, rows_data, col_widths, flag_row=worst_flag, highlight_col=6)

# ------- TOTAL ROW -------
pdf.set_font("Helvetica", "B", 8)
pdf.set_fill_color(30, 60, 110)
pdf.set_text_color(255, 255, 255)
total_row = ["", "TOTAL", f"{t_trades:,}", f"{t_wins:,}", f"{t_losses:,}", f"{t_wr:.1f}%",
             f"Rs{t_np:,.0f}", f"Rs{t_gp:,.0f}", f"Rs{t_gl:,.0f}", f"{t_pf:.2f}",
             "", "", "", "", "", f"Rs{t_charges:,.0f}"]
for i, c in enumerate(total_row):
    pdf.cell(col_widths[i], 6, c, border=1, align="C", fill=True)
pdf.ln(6)

# ------- LAST PAGE: CONCLUSION -------
pdf.add_page(orientation="L")
pdf.section_title("5. CONCLUSION & RECOMMENDATIONS")

pdf.sub_title("Optimization Results vs Original (v1)")
improvement_data = [
    ("Metric", "Original (v1)", "Optimized (v2)", "Improvement"),
    ("Total Trades", "540,999", "65,969", "-88% (fewer)"),
    ("Win Rate", "1.28%", "2.13%", "+66% relative"),
    ("Net P&L", "Rs-17,502,592", "Rs-1,478,672", "-91.5% loss"),
    ("Total Charges", "Rs14,052,861", "Rs884,082", "-93.7%"),
    ("Avg R Multiple", "-0.73", "-0.69", "+5% relative"),
    ("Profitable Stocks", "0/50", "0/50", "No change"),
]
mh = ["Metric", "Original (v1)", "Optimized (v2)", "Improvement"]
mw = [50, 50, 50, 50]
pdf.add_table(mh, improvement_data, mw)
pdf.ln(4)

pdf.sub_title("Why the Strategy Still Fails")
reasons = [
    "1. 'Close above R1' is better than 'touch' but still reverts ~98% of the time - pivots act as magnets, not breakouts.",
    "2. Entering at the trigger candle close avoids extreme prices, but doesn't solve the core directional problem.",
    "3. ATR-based SL allows breathing room but most trades still reverse before reaching TP.",
    "4. Despite 7 optimizations, the fundamental signal is not predictive - price crossing R1/S1 has no edge.",
    "5. Even with Rs5 brokerage, charges consume Rs884k out of Rs1,479k total loss (~60% of loss is friction).",
    "6. Without charges, the strategy still loses Rs595k - the edge is negative, not eaten by friction.",
]
for r in reasons:
    pdf.body_text("  " + r)

pdf.ln(2)
pdf.sub_title("Next Steps - What Might Actually Work")
improvements = [
    "1. Different pivot formula: Use Woodie's or Camarilla pivots (not traditional) - they may give better levels.",
    "2. Multi-timeframe confluence: Require R1 breakout ALIGNED with 1-hr pivot level breakout.",
    "3. Machine learning filter: Train a classifier on 50+ features to predict whether R1 breakout will succeed.",
    "4. Regime detection: Only trade breakouts during high-volatility regimes (identify with VIX or ATR expansion).",
    "5. Completely different strategy: Pivot breakouts don't work on Indian markets - try gap-and-go, VWAP, or order-flow.",
    "6. Higher timeframe: Test on daily/weekly pivots with daily entries instead of intraday.",
]
for imp in improvements:
    pdf.body_text("  " + imp)

pdf.ln(2)
pdf.sub_title("Observations on v2 Optimizations")
observations = [
    "Trades reduced from 540,999 to 65,969 (-88%) due to stricter filters - fewer but higher quality signals.",
    "Win rate improved from 1.28% to 2.13% - still far from profitable threshold of ~35%.",
    "Charges collapsed from Rs14Cr to Rs0.88Cr - Rs5 brokerage + fewer trades = massive savings.",
    "MARUTI had highest win rate (14.4%) but still lost Rs93k - high win rate doesn't mean profitability.",
    "No stock was profitable even with all 7 optimizations applied simultaneously.",
]
for obs in observations:
    pdf.body_text("  " + obs)

pdf.ln(4)
pdf.set_draw_color(30, 60, 110)
pdf.line(10, pdf.get_y(), 200, pdf.get_y())
pdf.ln(6)

pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 5.5,
    "This report was generated automatically as part of a quantitative research project. "
    "The data spans October 2016 through June 2026, using 15-minute candle data for signal generation "
    "and 1-minute candle data for entry simulation. All calculations include realistic brokerage, "
    "STT, exchange fees, SEBI charges, GST, and stamp duty as applicable to Indian equity intraday trading."
)

# ------- SAVE -------
pdf.output(PDF_PATH)
print(f"PDF report saved to: {PDF_PATH}")
print(f"File size: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")
