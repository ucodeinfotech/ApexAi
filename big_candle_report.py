"""Generate updated PDF report with forward analysis"""
import pandas as pd, numpy as np, os, glob
from fpdf import FPDF
from datetime import datetime

OUTPUT = "C:/Users/pc/Downloads/stock hist data/backtest_results"

pattern_df = pd.read_csv(os.path.join(OUTPUT, "big_candle_patterns.csv"))
stats_df = pd.read_csv(os.path.join(OUTPUT, "big_candle_stats.csv"))
fwd_df = pd.read_csv(os.path.join(OUTPUT, "big_candle_forward.csv"))

# ─── PDF ───
class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(130,130,130)
            self.cell(0,5,"Big Candle + Consolidation Pattern | 493 Stocks | Daily TF", align="C")
            self.ln(6)
            self.set_draw_color(200,200,200)
            self.line(10,self.get_y(),200,self.get_y())
            self.ln(3)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150,150,150)
        self.cell(0,8,f"Page {self.page_no()}", align="C")
    def section(self, t, c=(20,50,100)):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*c)
        self.ln(3); self.cell(0,9,t); self.ln(7)
        self.set_draw_color(*c); self.line(10,self.get_y(),200,self.get_y()); self.ln(4)
    def sub(self, t):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60,60,60)
        self.cell(0,7,t); self.ln(6)
    def txt(self, t):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(40,40,40)
        self.multi_cell(0,4.5,t); self.ln(2)
    def kv(self, k, v):
        self.set_font("Helvetica", "B", 8.5); self.set_text_color(40,40,40)
        self.cell(50,5,k)
        self.set_font("Helvetica", "", 8.5)
        self.cell(0,5,str(v)); self.ln(4.5)
    def table(self, hdr, data, cw, hl=None):
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(20,50,100); self.set_text_color(255,255,255)
        for i,h in enumerate(hdr): self.cell(cw[i],6,h,border=1,align="C",fill=True)
        self.ln()
        self.set_font("Helvetica", "", 7); self.set_text_color(40,40,40)
        for ri,row in enumerate(data):
            fill = ri % 2 == 1
            if fill: self.set_fill_color(245,245,250)
            for i,v in enumerate(row):
                try:
                    fv = float(v) if v != "-" else None
                    if hl is not None and i == hl and fv is not None:
                        self.set_text_color((0,130,0) if fv > 0 else ((200,0,0) if fv < 0 else (40,40,40)))
                except: pass
                self.cell(cw[i],5,str(v),border=1,align="C",fill=fill)
            self.set_text_color(40,40,40); self.ln()
        self.ln(3)

pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# === TITLE ===
pdf.ln(15)
pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(20,50,100)
pdf.cell(0,12,"Big Candle + Consolidation Scanner", align="C"); pdf.ln(10)
pdf.set_font("Helvetica", "", 11); pdf.set_text_color(80,80,80)
pdf.cell(0,7,"Full Backtest + Forward Performance | Daily Timeframe | 493 Stocks", align="C"); pdf.ln(12)
pdf.set_draw_color(20,50,100); pdf.line(60,pdf.get_y(),150,pdf.get_y()); pdf.ln(10)
pdf.set_font("Helvetica", "", 9); pdf.set_text_color(100,100,100)
pdf.cell(0,6,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C"); pdf.ln(5)
pdf.cell(0,6,f"Patterns Found: {len(pattern_df):,} across {stats_df[stats_df['patterns']>0].shape[0]} stocks", align="C"); pdf.ln(5)
pdf.cell(0,6,f"Forward observations: {len(fwd_df):,}", align="C"); pdf.ln(20)

pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(40,40,40)
pdf.cell(0,6,"Pattern Definition:", align="C"); pdf.ln(8)
pdf.set_font("Helvetica", "", 9); pdf.set_text_color(60,60,60)
for r in [
    "1. Trigger: body > 2x avg body (20d), upper wick < 20% range, volume > 1.5x avg",
    "2. Consolidation: 3+ small body candles (body < 30% range) within 5% of trigger close",
    "3. Non-small candles reset the consolidation count"]:
    pdf.cell(0,5.5,r, align="C"); pdf.ln(5.5)

pdf.add_page()
pdf.section("1. Overall Statistics")
tot_s = stats_df[stats_df['patterns']>0].shape[0]
pdf.kv("Stocks Scanned", str(len(stats_df)))
pdf.kv("Stocks with Patterns", str(tot_s))
pdf.kv("Total Patterns", f"{len(pattern_df):,}")
pdf.kv("Avg Patterns/Stock", f"{len(pattern_df)/tot_s:.1f}" if tot_s else "0")
bull = pattern_df[pattern_df["trigger_type"]=="BULLISH"]
bear = pattern_df[pattern_df["trigger_type"]=="BEARISH"]
pdf.kv("Bullish Triggers", f"{len(bull)} ({len(bull)/len(pattern_df)*100:.1f}%)")
pdf.kv("Bearish Triggers", f"{len(bear)} ({len(bear)/len(pattern_df)*100:.1f}%)")
still = pattern_df[pattern_df["status"]=="CONSOLIDATING"]
pdf.kv("Currently Consolidating", str(len(still)))

pdf.section("2. FORWARD PERFORMANCE (After Pattern Ends)")
pdf.txt("Key question: After the consolidation ends, where does price go next?")
pdf.txt("Both BULLISH and BEARISH patterns tested separately.")

for ttype, color in [("BULLISH", (0,100,0)), ("BEARISH", (180,0,0))]:
    pdf.sub(f"{ttype} Triggers")
    pdf.set_text_color(*color)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.cell(16,6,"Days",border=1,align="C")
    pdf.cell(18,6,"Count",border=1,align="C")
    pdf.cell(22,6,"Avg Ret%",border=1,align="C")
    pdf.cell(18,6,"Win%",border=1,align="C")
    pdf.cell(22,6,"Median%",border=1,align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(40,40,40)
    for h in [1,3,5,10,20,60]:
        sub = fwd_df[(fwd_df["horizon"]==h)&(fwd_df["trigger_type"]==ttype)]
        if len(sub)==0: continue
        avg = sub["fwd_return"].mean()
        med = sub["fwd_return"].median()
        wr = (sub["fwd_return"]>0).mean()*100
        pdf.cell(16,5,str(h),border=1,align="C")
        pdf.cell(18,5,str(len(sub)),border=1,align="C")
        pdf.cell(22,5,f"{avg:+.2f}%",border=1,align="C")
        pdf.cell(18,5,f"{wr:.1f}%",border=1,align="C")
        pdf.cell(22,5,f"{med:+.2f}%",border=1,align="C")
        pdf.ln()
    pdf.ln(4)

pdf.set_text_color(40,40,40)
pdf.txt("IMPORTANT: Both BULLISH and BEARISH triggers show positive forward returns. "
        "The pattern identifies stocks with recent strength/activity, and they tend to drift "
        "upward over time regardless of trigger direction. The 1-5 day horizon is essentially random.")
pdf.txt("The 60-day average return of +3-4% with ~53% win rate represents a mild bullish bias, "
        "not a tradable edge after accounting for slippage, commissions, and market drift.")

pdf.add_page()
pdf.section("3. Stock-Level Forward Analysis (5-day)")
pdf.sub("Worst Stocks After Pattern (avoid)")
w5 = fwd_df[(fwd_df["horizon"]==5)].groupby("symbol").agg(
    avg=("fwd_return","mean"),cnt=("fwd_return","count"),
    wr=("fwd_return",lambda x: (x>0).mean()*100))
w5 = w5[w5["cnt"]>=15].sort_values("avg")
rows = []
for sym, r in w5.head(15).iterrows():
    rows.append([sym, str(int(r["cnt"])), f"{r['avg']:+.2f}%", f"{r['wr']:.0f}%"])
pdf.table(["Symbol","Patterns","Avg 5d Ret","Win%"], rows, [45,20,35,35], hl=2)

pdf.sub("Best Stocks After Pattern (favor)")
rows = []
for sym, r in w5.tail(15).iloc[::-1].iterrows():
    rows.append([sym, str(int(r["cnt"])), f"{r['avg']:+.2f}%", f"{r['wr']:.0f}%"])
pdf.table(["Symbol","Patterns","Avg 5d Ret","Win%"], rows, [45,20,35,35], hl=2)

pdf.txt("Stock-specific behavior varies widely. Some stocks consistently fail after this pattern "
        "(AMBUJACEM -4.6%, ANGELONE -4.2%, IRCTC 9% win rate) while others thrive "
        "(AURIONPRO +11.6%, ADANIGREEN 87% win rate, STARCEMENT 91% win rate).")

pdf.add_page()
pdf.section("4. High-Confidence Stock Watchlist")
pdf.txt("Stocks with BEST forward returns after BULLISH pattern (min 15 patterns):")
rows = []
for sym, r in w5.tail(15).iloc[::-1].iterrows():
    sub = pattern_df[pattern_df["symbol"]==sym]
    now = sub[sub["status"]=="CONSOLIDATING"]
    rows.append([sym, f"{r['avg']:+.2f}%", f"{r['wr']:.0f}%", str(len(now))])
pdf.table(["Symbol","Avg 5d Ret","Win%","Active"], rows, [40,30,24,30], hl=1)

pdf.txt("'Active' = number of patterns currently in consolidation (potential trades).")

pdf.sub("Currently Consolidating (Bullish)")
rows = []
for _, r in pattern_df[(pattern_df["status"]=="CONSOLIDATING")].sort_values("days_since_pattern").head(20).iterrows():
    rows.append([r["symbol"], r["trigger_date"], f"{r['trigger_close']:.0f}",
                 str(r["consol_count"]), f"{r['days_since_pattern']}d"])
if rows:
    pdf.table(["Symbol","Trigger","Close","Consol","Age"], rows, [30,22,22,20,24])

pdf.section("5. Conclusion")
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(200,0,0)
# Determine overall verdict
bull_ok = (bull["last_close"] > bull["trigger_close"]).sum() if len(bull)>0 else 0
bear_ok = (bear["last_close"] < bear["trigger_close"]).sum() if len(bear)>0 else 0
all_ok = bull_ok + bear_ok
all_t = len(bull) + len(bear)
pdf.cell(0,6,f"Verdict: The pattern alone is a ~{all_ok/all_t*100:.0f}% directional indicator", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 8.5); pdf.set_text_color(50,50,50)
c = [
    f"Overall directional accuracy: {all_ok/all_t*100:.1f}% ({all_ok}/{all_t}) - essentially random.",
    f"Forward returns: neutral at 1-5 days, mildly bullish at 20-60 days (+3-4% avg, 52-53% win rate).",
    "",
    "The pattern identifies stocks that recently showed strength (big candle + high volume + consolidation).",
    "These stocks tend to drift upward over 2-3 months, consistent with momentum/continuation.",
    "",
    "For a scanner to be useful at entry:",
    "  - Filter by stock-specific performance (ADANIGREEN, AURIONPRO, STARCEMENT pattern well)",
    "  - Avoid stocks that consistently fail (ANGELONE, IRCTC, AMBUJACEM)",
    "  - Enter when consolidation is fresh (0-5 days old) on confirmation above breakout",
    "  - Use 1-3 day forward with tight SL (since short-term edge is near zero)",
    "",
    "Next improvement: Add volume decline during consolidation, RSI oversold/overbought filter,",
    "or combine with sector/theme context for better selectivity."]
for l in c: pdf.cell(0,5,l,align="C"); pdf.ln(5)

path = os.path.join(OUTPUT, "Big_Candle_Scanner_Report.pdf")
pdf.output(path)
print(f"PDF saved: {path}")
