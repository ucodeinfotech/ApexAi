"""
PDF Report: Squeeze Breakout Scanner Results
"""
import pandas as pd, numpy as np, os
from fpdf import FPDF

DATA = "C:/Users/pc/Downloads/stock hist data/backtest_results"
OUT_PATH = os.path.join(DATA, "Squeeze_Breakout_Report.pdf")

df = pd.read_csv(os.path.join(DATA, "squeeze_breakout_patterns.csv"))
fwd = pd.read_csv(os.path.join(DATA, "squeeze_breakout_fwd.csv"))

class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Courier", "", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

pdf.set_font("Courier", "B", 14)
pdf.cell(0, 10, "SQUEEZE BREAKOUT SCANNER", align="C")
pdf.ln(3)
pdf.cell(0, 10, "Consolidation -> Big Candle Breakout", align="C")
pdf.ln(8)
pdf.set_font("Courier", "", 10)
pdf.cell(0, 8, f"Generated: July 2, 2026")
pdf.ln(6)
pdf.cell(0, 8, f"Data: 493 stocks, daily, ~2016-2026")
pdf.ln(12)

# ─── SUMMARY ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== SUMMARY ===")
pdf.ln(8)
pdf.set_font("Courier", "", 10)

bull = df[df["type"]=="BULLISH"]
bear = df[df["type"]=="BEARISH"]
n = len(df)
nb = len(bull)
nbe = len(bear)

lines = [
    f"  Total patterns: {n} across {df['symbol'].nunique()} stocks",
    f"  Bullish: {nb} ({nb/n*100:.1f}%)",
    f"  Bearish: {nbe} ({nbe/n*100:.1f}%)",
    "",
    f"  Avg consolidation: {df['consol_days'].mean():.1f} days",
    f"  Avg consol range: {df['consol_range_pct'].mean():.2f}%",
    f"  Avg breakout body ratio: {df['body_ratio'].mean():.2f}x avg",
    f"  Avg breakout volume: {df['vol_ratio'].mean():.2f}x avg",
]
for l in lines:
    pdf.cell(0, 5.5, l)
    pdf.ln(5.5)
pdf.ln(3)

# ─── FORWARD RETURNS ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== FORWARD RETURNS ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

col_w = [22, 12, 14, 14, 10]
pdf.set_font("Courier", "B", 9)
for c, w in zip(["Type", "N", "AvgRet", "WinRate", "MedRet"], col_w):
    pdf.cell(w, 7, c, border=1)
pdf.ln()
pdf.set_font("Courier", "", 9)

for h in [1, 3, 5, 10, 20, 60]:
    pdf.set_font("Courier", "B", 9)
    pdf.cell(col_w[0], 6, f"--- {h}d ---")
    pdf.cell(sum(col_w[1:]), 6, "")
    pdf.ln()
    pdf.set_font("Courier", "", 9)
    for ttype in ["BULLISH", "BEARISH"]:
        sub = fwd[(fwd["horizon"]==h) & (fwd["type"]==ttype)]
        nsub = len(sub)
        if nsub == 0: continue
        avg = sub["fwd_return"].mean()
        med = sub["fwd_return"].median()
        wr = (sub["fwd_return"]>0).mean()*100
        vals = [ttype, f"{nsub}", f"{avg:+.2f}%", f"{wr:.0f}%", f"{med:+.2f}%"]
        for v, w in zip(vals, col_w):
            pdf.cell(w, 5.5, v)
        pdf.ln()
pdf.ln(4)

# ─── DIRECTIONAL ACCURACY TABLE ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== DIRECTIONAL ACCURACY vs HOLDING PERIOD ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

# Compute directional accuracy at each horizon
dir_res = []
import glob as gm
files = sorted(gm.glob("C:/Users/pc/Downloads/stock hist data/comprehensive_data/*_ONE_DAY.csv"))
files_map = {}
for f in files:
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    d = pd.read_csv(f)
    d["datetime"] = pd.to_datetime(d["datetime"])
    files_map[sym] = d.sort_values("datetime").reset_index(drop=True)

for _, p in df.iterrows():
    sym = p["symbol"]
    idx = p["idx"]
    close = p["close"]
    ttype = p["type"]
    d = files_map.get(sym)
    if d is None: continue
    for h in [1,3,5,10,20,60]:
        fwd_idx = idx + h
        if fwd_idx >= len(d): continue
        fclose = d.iloc[fwd_idx]["close"]
        correct = (ttype=="BULLISH" and fclose>close) or (ttype=="BEARISH" and fclose<close)
        dir_res.append({"horizon":h, "type":ttype, "correct":correct})
dir_df = pd.DataFrame(dir_res)

col_w2 = [12, 20, 20]
pdf.set_font("Courier", "B", 9)
for c, w in zip(["Horiz", "Bull Acc", "Bear Acc"], col_w2):
    pdf.cell(w, 7, c, border=1)
pdf.ln()
pdf.set_font("Courier", "", 9)
for h in [1,3,5,10,20,60]:
    pdf.cell(col_w2[0], 6, f"{h:>4d}")
    for ttype in ["BULLISH", "BEARISH"]:
        sub = dir_df[(dir_df["horizon"]==h) & (dir_df["type"]==ttype)]
        acc = sub["correct"].mean()*100 if len(sub) > 0 else 0
        pdf.cell(col_w2[1], 6, f"{acc:.1f}%")
    pdf.ln()
pdf.ln(5)

# ─── STOCK ANALYSIS ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== STOCK PERFORMANCE (5d forward) ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

fwd5 = fwd[fwd["horizon"]==5]
stocks = fwd5.groupby("symbol")["fwd_return"].agg(["mean","count",lambda x: (x>0).mean()*100])
stocks = stocks[stocks["count"]>=5].sort_values("mean")

col_w3 = [22, 6, 14, 10]
pdf.set_font("Courier", "B", 9)
for c, w in zip(["SYMBOL", "N", "AVG RET", "WIN%"], col_w3):
    pdf.cell(w, 7, c, border=1)
pdf.ln()
pdf.set_font("Courier", "", 8)

pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "WORST 15:")
pdf.ln()
pdf.set_font("Courier", "", 8)
for sym, r in stocks.head(15).iterrows():
    pdf.cell(col_w3[0], 5, sym[:20])
    pdf.cell(col_w3[1], 5, f"{int(r['count'])}")
    pdf.cell(col_w3[2], 5, f"{r['mean']:+.2f}%")
    pdf.cell(col_w3[3], 5, f"{r['<lambda_0>']:.0f}%")
    pdf.ln()

pdf.ln(3)
pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "BEST 15:")
pdf.ln()
pdf.set_font("Courier", "", 8)
for sym, r in stocks.tail(15).iloc[::-1].iterrows():
    pdf.cell(col_w3[0], 5, sym[:20])
    pdf.cell(col_w3[1], 5, f"{int(r['count'])}")
    pdf.cell(col_w3[2], 5, f"{r['mean']:+.2f}%")
    pdf.cell(col_w3[3], 5, f"{r['<lambda_0>']:.0f}%")
    pdf.ln()

pdf.ln(5)

# ─── COMPARISON WITH BIG CANDLE PATTERN ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== COMPARISON: SQUEEZE vs BIG CANDLE PATTERN ===")
pdf.ln(9)
pdf.set_font("Courier", "", 10)

comp_lines = [
    "  SQUEEZE BREAKOUT (consolidation -> big candle):",
    "    Bullish 5d: +0.46%, 49% WR | 20d: +1.93%, 54% WR | 60d: +5.70%, 55% WR",
    "    1d accuracy: 48.0% (near random short-term)",
    "    Medium-term drift: STRONG (+5.7% at 60d)",
    "",
    "  BIG CANDLE + CONSOLIDATION (big candle -> consolidation):",
    "    Bullish 5d: +0.29%, 50% WR | 20d: +1.22%, 50% WR | 60d: +3.94%, 52% WR",
    "    1d accuracy: 45.2% (below random)",
    "    Medium-term drift: MODERATE (+3.9% at 60d)",
    "",
    "  KEY DIFFERENCE:",
    "    - Squeeze breakout waits for confirmation (big candle AFTER cooldown)",
    "    - Big Candle pattern catches the move BEFORE consolidation",
    "    - Squeeze has 74% bullish patterns vs Big Candle's 57%",
    "    - Squeeze 60d return is 45% higher than Big Candle (+5.7% vs +3.9%)",
    "",
    "  BOTTOM LINE: Squeeze breakout is the better pattern for medium-term",
    "  bullish bias. But both patterns still rely on general market drift",
    "  rather than genuine short-term directional edge.",
]
for l in comp_lines:
    pdf.cell(0, 5.5, l)
    pdf.ln(5.5)

pdf.ln(5)

# ─── TOP/BOTTOM STOCKS ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== RELIABLE STOCKS ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

always_win = stocks[stocks['<lambda_0>']==100]
always_lose = stocks[stocks['<lambda_0>']==0]

pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "100% WIN RATE (n>=5):")
pdf.ln()
pdf.set_font("Courier", "", 8)
if len(always_win) > 0:
    for sym, r in always_win.iterrows():
        pdf.cell(0, 5, f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}%")
        pdf.ln()
else:
    pdf.cell(0, 5, "  (none with n>=5)")
    pdf.ln()

pdf.ln(3)
pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "0% WIN RATE (n>=5):")
pdf.ln()
pdf.set_font("Courier", "", 8)
if len(always_lose) > 0:
    for sym, r in always_lose.iterrows():
        pdf.cell(0, 5, f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}%")
        pdf.ln()
else:
    pdf.cell(0, 5, "  (none with n>=5)")
    pdf.ln()

pdf.output(OUT_PATH)
print(f"PDF saved: {OUT_PATH}")
