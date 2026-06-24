"""Generate final PDF report for BB(20,2.5) strategy - verified clean results"""
import pandas as pd, numpy as np, os
from fpdf import FPDF
from datetime import datetime

OUTPUT_DIR = "backtest_results"

# Load pre-computed trade data from the clean backtest
results = {}
for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f"bb_{sym.lower()}_trades.csv"))
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
    results[sym] = df

# Stats helper
def stats(df):
    t = len(df)
    w = int((df["pnl_pts"] > 0).sum())
    l = t - w
    t1c = int((df["reason"] == "T1").sum())
    tpc = int((df["reason"] == "TP").sum())
    eodc = int((df["reason"] == "EOD").sum())
    gross = df["pnl_pts"].sum()
    net = df["net_pnl"].sum()
    ch = df["charges"].sum()
    avg_r = df["r"].mean()
    avg_t1 = df[df["reason"] == "T1"]["pnl_pts"].mean() if t1c else 0
    avg_tp = df[df["reason"] == "TP"]["pnl_pts"].mean() if tpc else 0
    avg_eod = df[df["reason"] == "EOD"]["pnl_pts"].mean() if eodc else 0
    win_pts = df[df["pnl_pts"] > 0]["pnl_pts"].sum()
    loss_pts = abs(df[df["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf = round(win_pts / loss_pts, 2) if loss_pts > 0 else float("inf")
    if "exit_time" in df.columns:
        cs = df.sort_values("exit_time").reset_index(drop=True)
    else:
        cs = df.reset_index(drop=True)
    cs["cum"] = cs["net_pnl"].cumsum()
    cs["peak"] = cs["cum"].cummax()
    mdd = (cs["peak"] - cs["cum"]).max()
    sh = round(avg_r / df["r"].std() * np.sqrt(t), 2) if df["r"].std() > 0 else 0
    return {"t": t, "w": w, "l": l, "wr": round(w / t * 100, 1), "gross": gross, "net": net,
            "ch": ch, "pf": pf, "ar": round(avg_r, 2), "mdd": mdd, "sh": sh,
            "t1c": t1c, "tpc": tpc, "eodc": eodc,
            "t1_pct": round(t1c / t * 100, 1), "tp_pct": round(tpc / t * 100, 1), "eod_pct": round(eodc / t * 100, 1),
            "avg_t1": round(avg_t1, 1), "avg_tp": round(avg_tp, 1), "avg_eod": round(avg_eod, 1)}

s = {sym: stats(results[sym]) for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]}

# === PDF ===
class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(130, 130, 130)
            self.cell(0, 5, "BB(20, 2.5) Mean-Reversion Backtest | 15-min Data | RR 1:3 | Spot Indices", align="C")
            self.ln(6)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
    def section(self, t):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 50, 100)
        self.ln(3)
        self.cell(0, 9, t)
        self.ln(7)
        self.set_draw_color(20, 50, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
    def sub(self, t):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, t)
        self.ln(6)
    def txt(self, t):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, t)
        self.ln(2)
    def kv(self, k, v):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(40, 40, 40)
        self.cell(50, 5, k)
        self.set_font("Helvetica", "", 8.5)
        self.cell(0, 5, v)
        self.ln(4.5)
    def table(self, hdr, data, cw, hl=None):
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(20, 50, 100)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(hdr):
            self.cell(cw[i], 6, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 7)
        self.set_text_color(40, 40, 40)
        for ri, row in enumerate(data):
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(245, 245, 250)
            for i, v in enumerate(row):
                txt = str(v)
                if hl is not None and i == hl:
                    try:
                        fv = float(txt)
                        if fv > 0:
                            self.set_text_color(0, 130, 0)
                        elif fv < 0:
                            self.set_text_color(200, 0, 0)
                        else:
                            self.set_text_color(40, 40, 40)
                    except:
                        pass
                self.cell(cw[i], 5, txt, border=1, align="C", fill=fill)
            self.set_text_color(40, 40, 40)
            self.ln()
        self.ln(3)

pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# === TITLE ===
pdf.ln(20)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(20, 50, 100)
pdf.cell(0, 12, "BB(20, 2.5) Mean-Reversion Strategy", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Backtest Report | 15-min Data | RR 1:3 | Spot Indices", align="C")
pdf.ln(12)
pdf.set_draw_color(20, 50, 100)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(12)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")
pdf.ln(6)
pdf.cell(0, 6, "Data: NIFTY50, BANKNIFTY, SENSEX Spot | 2015-2026", align="C")
pdf.ln(20)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(40, 40, 40)
pdf.cell(0, 6, "Strategy Rules:", align="C")
pdf.ln(8)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(60, 60, 60)
rules = [
    "1. BB(20, SD=2.5) on 15-min candles",
    "2. Trigger: entire candle outside band (low > upper = SHORT, high < lower = LONG)",
    "3. Entry: at close of trigger candle (market order)",
    "4. T1 target: extreme of trigger candle = +1R (low for SHORT, high for LONG)",
    "5. TP target: 3x T1 distance = +3R",
    "6. EOD exit at 3:15 PM if neither target hit",
    "7. Charges: brokerage Rs10x2, STT 0.1% on sell, exchange/SEBI/stamp duty + GST"]
for r in rules:
    pdf.cell(0, 5.5, r, align="C")
    pdf.ln(5.5)

pdf.add_page()
pdf.section("1. Performance Summary")

rows = []
for sym in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
    st = s[sym]
    rows.append([sym, st["t"], f"{st['w']} ({st['wr']}%)", f"{st['gross']:+.0f}",
                 f"Rs{st['ch']:,.0f}", f"{st['net']:+.0f}", st["pf"], st["ar"],
                 f"{st['mdd']:,.0f}", st["sh"],
                 f"{st['t1c']} ({st['t1_pct']}%)",
                 f"{st['tpc']} ({st['tp_pct']}%)"])

pdf.sub("All Indices Comparison")
pdf.table(["Index", "Trades", "Wins", "Gross Pts", "Charges", "Net Pts", "PF", "AvgR", "MaxDD", "Sharpe", "T1 (#/%)", "TP (#/%)"],
          rows, [22, 14, 18, 16, 16, 18, 10, 10, 16, 12, 20, 16], hl=5)

pdf.txt("T1 exit = partial profit at +1R (mean reversion hits trigger candle extreme). "
        "TP = full target at +3R. EOD = held to 3:15 PM close (neither target hit). "
        "All three indices are loss-making after charges. Even gross P&L (before charges) is negative.")

pdf.add_page()
pdf.section("2. BANKNIFTY - Detailed Report")
df = results["BANKNIFTY"]
st = s["BANKNIFTY"]
pdf.kv("Data Period", f"{df['date'].min()} to {df['date'].max()}")
pdf.kv("Trading Days with Trades", str(df["date"].nunique()))
pdf.kv("Total Trades", str(st["t"]))
pdf.kv("Win Rate (pts)", f"{st['w']} ({st['wr']}%)")
pdf.kv("Loss Rate (pts)", f"{st['l']} ({100-st['wr']:.1f}%)")
pdf.kv("Gross P&L", f"{st['gross']:+.0f} pts")
pdf.kv("Total Charges", f"Rs{st['ch']:,.0f}")
pdf.kv("Net P&L", f"{st['net']:+.0f} pts")
pdf.kv("Profit Factor", str(st["pf"]))
pdf.kv("Average R Multiple", str(st["ar"]))
pdf.kv("Max Drawdown", f"{st['mdd']:,.0f} pts")
pdf.kv("Sharpe Ratio (R-based)", str(st["sh"]))

pdf.ln(2)
pdf.sub("Exit Breakdown")
pdf.table(["Exit", "Count", "% of Total", "Avg PnL (pts)", "Avg R"],
          [["T1 (+1R)", st["t1c"], f"{st['t1_pct']}%", f"+{st['avg_t1']}", "+1.00"],
           ["TP (+3R)", st["tpc"], f"{st['tp_pct']}%", f"+{st['avg_tp']}", "+3.00"],
           ["EOD", st["eodc"], f"{st['eod_pct']}%", f"{st['avg_eod']:+.0f}",
            f"{df[df['reason']=='EOD']['r'].mean():+.2f}"]],
          [28, 16, 20, 30, 20], hl=2)

pdf.txt(f"T1 trades ({st['t1_pct']}%) average +{st['avg_t1']} pts before charges. "
        f"TP trades ({st['tp_pct']}%) average +{st['avg_tp']} pts. "
        f"EOD trades ({st['eod_pct']}%) average {st['avg_eod']:+.0f} pts - these large losses "
        f"overwhelm the gains from T1/TP wins.")

pdf.sub("Yearly Breakdown")
y_data = []
yearly = df.groupby("year").agg(
    tr=("pnl_pts", "count"), w=("pnl_pts", lambda x: int((x > 0).sum())),
    gross=("pnl_pts", "sum"), net=("net_pnl", "sum"), r=("r", "mean"))
for yr, r_ in yearly.iterrows():
    y_data.append([int(yr), int(r_["tr"]),
                   f"{r_['w']} ({round(r_['w']/r_['tr']*100, 0):.0f}%)",
                   f"{r_['gross']:+.0f}", f"{r_['net']:+.0f}", f"{r_['r']:+.2f}"])
pdf.table(["Year", "Trades", "Wins", "Gross", "Net", "AvgR"], y_data, [18, 16, 28, 28, 28, 18], hl=4)

pdf.sub("Direction Breakdown")
for sd in ["LONG", "SHORT"]:
    sub = df[df["side"] == sd]
    if len(sub) > 0:
        pdf.kv(f"  {sd}", f"{len(sub)} trades | WR: {((sub['pnl_pts'] > 0).sum()/len(sub)*100):.1f}% | "
               f"Gross: {sub['pnl_pts'].sum():+.0f} | Net: {sub['net_pnl'].sum():+.0f} | AvgR: {sub['r'].mean():+.2f}")

pdf.add_page()
pdf.section("3. SENSEX - Detailed Report")
df = results["SENSEX"]
st = s["SENSEX"]
pdf.kv("Data Period", f"{df['date'].min()} to {df['date'].max()}")
pdf.kv("Total Trades", str(st["t"]))
pdf.kv("Win Rate (pts)", f"{st['w']} ({st['wr']}%)")
pdf.kv("Gross P&L", f"{st['gross']:+.0f} pts")
pdf.kv("Total Charges", f"Rs{st['ch']:,.0f}")
pdf.kv("Net P&L", f"{st['net']:+.0f} pts")
pdf.kv("Profit Factor", str(st["pf"]))
pdf.kv("Average R Multiple", str(st["ar"]))
pdf.kv("Max Drawdown", f"{st['mdd']:,.0f} pts")
pdf.kv("Avg T1 PnL", f"+{st['avg_t1']}")
pdf.kv("Avg EOD PnL", f"{st['avg_eod']:+.0f}")

pdf.sub("Yearly Breakdown")
y_data = []
yearly = df.groupby("year").agg(
    tr=("pnl_pts", "count"), w=("pnl_pts", lambda x: int((x > 0).sum())),
    gross=("pnl_pts", "sum"), net=("net_pnl", "sum"), r=("r", "mean"))
for yr, r_ in yearly.iterrows():
    y_data.append([int(yr), int(r_["tr"]),
                   f"{r_['w']} ({round(r_['w']/r_['tr']*100, 0):.0f}%)",
                   f"{r_['gross']:+.0f}", f"{r_['net']:+.0f}", f"{r_['r']:+.2f}"])
pdf.table(["Year", "Trades", "Wins", "Gross", "Net", "AvgR"], y_data, [18, 16, 28, 28, 28, 18], hl=4)

pdf.sub("Direction Breakdown")
for sd in ["LONG", "SHORT"]:
    sub = df[df["side"] == sd]
    if len(sub) > 0:
        pdf.kv(f"  {sd}", f"{len(sub)} trades | WR: {((sub['pnl_pts'] > 0).sum()/len(sub)*100):.1f}% | "
               f"Net: {sub['net_pnl'].sum():+.0f} | AvgR: {sub['r'].mean():+.2f}")

pdf.section("4. NIFTY50 - Detailed Report")
df = results["NIFTY50"]
st = s["NIFTY50"]
pdf.kv("Data Period", f"{df['date'].min()} to {df['date'].max()}")
pdf.kv("Total Trades", str(st["t"]))
pdf.kv("Win Rate (pts)", f"{st['w']} ({st['wr']}%)")
pdf.kv("Gross P&L", f"{st['gross']:+.0f} pts")
pdf.kv("Total Charges", f"Rs{st['ch']:,.0f}")
pdf.kv("Net P&L", f"{st['net']:+.0f} pts")
pdf.kv("Avg T1 PnL", f"+{st['avg_t1']} | Avg EOD PnL: {st['avg_eod']:+.0f}")
pdf.txt("NIFTY50's smaller T1 distance (~24 pts) means even winning trades barely cover charges (~Rs40/trade). "
        "Every winning trade nets only ~Rs15 after charges, while losing EOD trades lose ~Rs90+ after charges.")

pdf.add_page()
pdf.section("5. Monthly & Day-of-Week Patterns (BANKNIFTY)")
pdf.sub("Monthly Net P&L")
df = results["BANKNIFTY"]
mn = df.groupby("month")["net_pnl"].sum()
names_m = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
m_data = [[names_m[m], f"{v:+.0f}"] for m, v in mn.sort_values(ascending=False).items()]
pdf.table(["Month", "Net Pts"], m_data, [40, 40], hl=1)
best_m, worst_m = mn.idxmax(), mn.idxmin()
pdf.txt(f"Best: {names_m[best_m]} ({mn[best_m]:+.0f})  Worst: {names_m[worst_m]} ({mn[worst_m]:+.0f})")

pdf.sub("Day of Week Net P&L")
dow_data = []
for d in range(5):
    sub = df[df["dow"] == d]
    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    if len(sub) > 0:
        dow_data.append([dow_names[d], len(sub), f"{sub['net_pnl'].sum():+.0f}"])
pdf.table(["Day", "Trades", "Net Pts"], dow_data, [38, 28, 38], hl=2)

pdf.sub("Direction Comparison")
ls_data = []
for sd in ["LONG", "SHORT"]:
    sub = df[df["side"] == sd]
    if len(sub) > 0:
        ls_data.append([sd, len(sub),
                        f"{(sub['pnl_pts'] > 0).sum()/len(sub)*100:.1f}%",
                        f"{sub['pnl_pts'].sum():+.0f}",
                        f"{sub['net_pnl'].sum():+.0f}",
                        f"{sub['r'].mean():+.2f}"])
pdf.table(["Side", "Trades", "WR (pts)", "Gross", "Net", "AvgR"], ls_data, [16, 18, 20, 28, 28, 18], hl=4)

pdf.ln(5)
pdf.section("6. Conclusion")
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(200, 0, 0)
pdf.cell(0, 6, "Strategy Verdict: NOT PROFITABLE on any index", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 8.5)
pdf.set_text_color(50, 50, 50)

conclusions = [
    f"The BB(20, 2.5) mean-reversion strategy loses money on all three indices:",
    f"  NIFTY50: {s['NIFTY50']['net']:+.0f} pts net | BANKNIFTY: {s['BANKNIFTY']['net']:+.0f} pts net | SENSEX: {s['SENSEX']['net']:+.0f} pts net",
    "",
    f"Win rates of 75-78% are realistic but the 23-26% EOD trades incur losses averaging 2-5x the T1 gain.",
    f"T1 profits (+24 to +88 pts) are too small relative to EOD losses (-94 to -323 pts).",
    "",
    f"Exit distribution (BANKNIFTY): T1 {s['BANKNIFTY']['t1_pct']}% | TP {s['BANKNIFTY']['tp_pct']}% | EOD {s['BANKNIFTY']['eod_pct']}%",
    f"The strategy's fatal flaw: BB(20, 2.5) bands on 15-min are wide; when the breakout continues instead of",
    f"reverting, the loss accumulates until EOD cutoff. A wider stop or multi-day hold may change the outcome.",
    "",
    "Previous BUG FIXED: The T1 exit condition was checking wrong candle side (SHORT checked HIGH for retracement",
    "instead of LOW). This caused ~100% T1 hit rate on the very next candle. After correction, realistic results."
]
for c in conclusions:
    pdf.cell(0, 5, c, align="C")
    pdf.ln(5)

path = os.path.join(OUTPUT_DIR, "BB_Breakout_Strategy_Report.pdf")
pdf.output(path)
print(f"\nPDF saved: {path}")
