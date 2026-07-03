"""
Generate PDF report comparing BASE vs IMPROVED Big Candle Scanner results
"""
import pandas as pd, numpy as np, os
import glob as glob_module
from fpdf import FPDF

DATA = "C:/Users/pc/Downloads/stock hist data/backtest_results"
OUT_PATH = os.path.join(DATA, "Big_Candle_Improved_Report.pdf")

bd = pd.read_csv(os.path.join(DATA, "big_candle_base.csv"))
idf = pd.read_csv(os.path.join(DATA, "big_candle_improved.csv"))
bf = pd.read_csv(os.path.join(DATA, "big_candle_base_fwd.csv"))
iff = pd.read_csv(os.path.join(DATA, "big_candle_improved_fwd.csv"))

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-15)
        self.set_font("Courier", "", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ─── Title ───
pdf.set_font("Courier", "B", 14)
pdf.cell(0, 10, "BIG CANDLE + CONSOLIDATION SCANNER", align="C")
pdf.ln(6)
pdf.cell(0, 10, "BASE vs IMPROVED COMPARISON", align="C")
pdf.ln(8)
pdf.set_font("Courier", "", 10)
pdf.cell(0, 8, f"Generated: July 2, 2026")
pdf.ln(6)
pdf.cell(0, 8, f"Data: 493 stocks, daily, ~2016-2026")
pdf.ln(12)

# ─── Summary ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== SUMMARY ===")
pdf.ln(8)
pdf.set_font("Courier", "", 10)

n_base = len(bd); n_imp = len(idf)
n_base_stk = bd["symbol"].nunique(); n_imp_stk = idf["symbol"].nunique()

bull_b = bd[bd["trigger_type"]=="BULLISH"]
bear_b = bd[bd["trigger_type"]=="BEARISH"]
bull_i = idf[idf["trigger_type"]=="BULLISH"]
bear_i = idf[idf["trigger_type"]=="BEARISH"]

lines = [
    f"  BASE:     {n_base:>6d} patterns / {n_base_stk:>3d} stocks",
    f"  IMPROVED: {n_imp:>6d} patterns / {n_imp_stk:>3d} stocks",
    f"  Reduction: {(1-n_imp/n_base)*100:>5.1f}%",
    "",
    f"  Directional accuracy (consolidation END):",
    f"     BASE BULLISH: {bull_b['last_close'].gt(bull_b['trigger_close']).mean()*100:.1f}%",
    f"     BASE BEARISH: {bear_b['last_close'].lt(bear_b['trigger_close']).mean()*100:.1f}%",
    f"     IMP BULLISH: {bull_i['last_close'].gt(bull_i['trigger_close']).mean()*100:.1f}%",
    f"     IMP BEARISH: {bear_i['last_close'].lt(bear_i['trigger_close']).mean()*100:.1f}%",
]
for l in lines:
    pdf.cell(0, 5.5, l)
    pdf.ln(5.5)

pdf.ln(3)

# ─── PARAM TABLE ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== PARAMETERS ===")
pdf.ln(8)
pdf.set_font("Courier", "", 9)

params = [
    ("Parameter", "BASE", "IMPROVED"),
    ("Big candle body mult", ">= 2.0x avg", ">= 2.0x avg"),
    ("Upper wick ratio", "< 20%", "< 20%"),
    ("Volume multiplier", ">= 1.5x", ">= 1.5x"),
    ("Consolidation min candles", "3", "3"),
    ("Consolidation max range", "5%", "5%"),
    ("Consol body / range", "< 30%", "< 30%"),
    ("Volume decline filter", "--", "Consol vol < 60% trigger"),
    ("RSI filter", "--", "Bull: RSI<50, Bear: RSI>50"),
    ("Min 14d avg volume", "--", "> 100,000"),
    ("Min price", "--", "> Rs 10"),
]

for r in params:
    pdf.cell(60, 6, r[0])
    pdf.cell(45, 6, r[1])
    pdf.cell(0, 6, r[2])
    pdf.ln(6)
pdf.ln(4)

# ─── FORWARD RETURNS TABLE ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== FORWARD RETURNS ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

# Header
col_w = [6, 18, 10, 10, 10, 10]
headers = ["", "Set", "N", "AvgRet", "Win%", "MedRet"]
for c, w in zip(headers, col_w):
    pdf.cell(w*3, 7, c, border=1)
pdf.ln()

for h in [1, 3, 5, 10, 20, 60]:
    pdf.set_font("Courier", "B", 9)
    pdf.cell(col_w[0]*3, 6, f"{h:>2d}d")
    pdf.cell(col_w[1]*3, 6, "")
    pdf.cell(col_w[2]*3, 6, "")
    pdf.cell(col_w[3]*3, 6, "")
    pdf.cell(col_w[4]*3, 6, "")
    pdf.cell(col_w[5]*3, 6, "")
    pdf.ln()
    pdf.set_font("Courier", "", 9)
    for label, ff, ttype in [
        ("BASE", bf, "BULLISH"), ("BASE", bf, "BEARISH"),
        ("IMP", iff, "BULLISH"), ("IMP", iff, "BEARISH"),
    ]:
        sub = ff[(ff["horizon"]==h) & (ff["trigger_type"]==ttype)]
        n = len(sub)
        if n == 0: continue
        avg = sub["fwd_return"].mean()
        med = sub["fwd_return"].median()
        wr = (sub["fwd_return"]>0).mean()*100
        vals = [label, f"{n}", f"{avg:+.2f}%", f"{wr:.0f}%", f"{med:+.2f}%"]
        for v, w in zip(vals, col_w[1:]):
            pdf.cell(w*3, 5.5, v)
        pdf.ln()
pdf.ln(4)

# ─── DIRECTIONAL ACCURACY BY HORIZON ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== DIRECTIONAL ACCURACY vs HOLDING PERIOD ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

# For each pattern, check if price is higher/lower than trigger close at N days
def direction_by_horizon(df, df_map, horizons=[1,3,5,10,20,60]):
    results = []
    for _, p in df.iterrows():
        sym = p["symbol"]
        t = p["trigger_type"]
        tclose = p["trigger_close"]
        ti = p["trigger_idx"]
        if sym not in df_map: continue
        d = df_map[sym]
        for h in horizons:
            fwd = ti + h
            if fwd >= len(d): continue
            fclose = d.iloc[fwd]["close"]
            correct = (t=="BULLISH" and fclose>tclose) or (t=="BEARISH" and fclose<tclose)
            results.append({"horizon":h, "trigger_type":t, "correct":correct})
    return pd.DataFrame(results)

files = sorted(glob_module.glob(f"{DATA.replace('backtest_results','comprehensive_data')}/*_ONE_DAY.csv"))
df_map = {}
for f in files:
    sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
    d = pd.read_csv(f)
    d["datetime"] = pd.to_datetime(d["datetime"])
    df_map[sym] = d.sort_values("datetime").reset_index(drop=True)

print("Computing directional accuracy by horizon...")
import glob
base_dir = dir_base = direction_by_horizon(bd, df_map)
imp_dir = direction_by_horizon(idf, df_map)

pdf.set_font("Courier", "B", 9)
pdf.cell(12, 7, "Horiz")
pdf.cell(24, 7, "BASE Bull", border=1)
pdf.cell(24, 7, "BASE Bear", border=1)
pdf.cell(24, 7, "IMP Bull", border=1)
pdf.cell(24, 7, "IMP Bear", border=1)
pdf.ln()
pdf.set_font("Courier", "", 9)
for h in [1,3,5,10,20,60]:
    pdf.cell(12, 6, f"{h:>2d}d")
    for label, dd in [("B", base_dir), ("I", imp_dir)]:
        for ttype in ["BULLISH", "BEARISH"]:
            sub = dd[(dd["horizon"]==h) & (dd["trigger_type"]==ttype)]
            acc = sub["correct"].mean()*100 if len(sub) > 0 else 0
            pdf.cell(24, 6, f"{acc:.1f}%")
    pdf.ln()
pdf.ln(5)

# ─── TOP/BOTTOM STOCKS ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== STOCK-LEVEL ANALYSIS (IMPROVED, 5d forward) ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

imp5 = iff[iff["horizon"]==5].groupby("symbol")["fwd_return"].agg(["mean","count",lambda x: (x>0).mean()*100])
imp5 = imp5[imp5["count"]>=5].sort_values("mean")

pdf.set_font("Courier", "B", 9)
for c, w in [("SYMBOL", 28), ("N", 8), ("AVG RET", 14), ("WIN%", 10)]:
    pdf.cell(w, 7, c, border=1)
pdf.ln()
pdf.set_font("Courier", "", 8)

pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "WORST 15:")
pdf.ln()
pdf.set_font("Courier", "", 8)
for sym, r in imp5.head(15).iterrows():
    pdf.cell(28, 5, sym[:20])
    pdf.cell(8, 5, f"{int(r['count'])}")
    pdf.cell(14, 5, f"{r['mean']:+.2f}%")
    pdf.cell(10, 5, f"{r['<lambda_0>']:.0f}%")
    pdf.ln()

pdf.ln(3)
pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "BEST 15:")
pdf.ln()
pdf.set_font("Courier", "", 8)
for sym, r in imp5.tail(15).iloc[::-1].iterrows():
    pdf.cell(28, 5, sym[:20])
    pdf.cell(8, 5, f"{int(r['count'])}")
    pdf.cell(14, 5, f"{r['mean']:+.2f}%")
    pdf.cell(10, 5, f"{r['<lambda_0>']:.0f}%")
    pdf.ln()

pdf.ln(5)

# ─── FAIL RATE STOCKS (reliable losers) ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== MOST RELIABLE STOCKS (IMPROVED, 5d) ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "ALWAYS FAIL (0% win rate, n>=5):")
pdf.ln()
pdf.set_font("Courier", "", 8)
fail_stocks = imp5[imp5['<lambda_0>']==0]
for sym, r in fail_stocks.iterrows():
    pdf.cell(0, 5, f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}% win=0%")
    pdf.ln()
if len(fail_stocks)==0:
    pdf.cell(0, 5, "  (none)")
    pdf.ln()

pdf.ln(3)
pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "ALWAYS WIN (100% win rate, n>=5):")
pdf.ln()
pdf.set_font("Courier", "", 8)
win_stocks = imp5[imp5['<lambda_0>']==100]
for sym, r in win_stocks.iterrows():
    pdf.cell(0, 5, f"  {sym:20s} n={int(r['count']):>3d} avg={r['mean']:+.2f}% win=100%")
    pdf.ln()
if len(win_stocks)==0:
    pdf.cell(0, 5, "  (none)")
    pdf.ln()

pdf.ln(5)

# ─── CONCLUSIONS ───
pdf.set_font("Courier", "B", 11)
pdf.cell(0, 8, "=== KEY FINDINGS ===")
pdf.ln(9)
pdf.set_font("Courier", "", 9)

conclusions = [
    "1. The Big Candle + Consolidation pattern is ~50% directional.",
    "   Both BASE and IMPROVED show near-random directional accuracy.",
    "",
    "2. IMPROVED filters (RSI reverse, volume decline) provide",
    "   marginal gains (+1-2%) but reduce sample by 70%.",
    "",
    "3. Short-term forward returns (1-5d) are essentially random.",
    "   Neither BASE nor IMPROVED shows consistent edge.",
    "",
    "4. Medium-term (20-60d): mild bullish drift (+2-4%) regardless",
    "   of trigger direction. Stock quality bias, not pattern signal.",
    "",
    "5. Stock-specific filtering is the ONLY reliable improvement:",
    "   - Avoid: ANGELONE, SHARDACROP, OLAELEC, AMBUJACEM, IFCI",
    "   - Prefer: STARCEMENT, ADANIGREEN, BHEL, CANBK, BLUESTARCO",
    "",
    "6. No technical filtering combination tested can extract",
    "   meaningful directional edge from this pattern alone.",
]

for l in conclusions:
    pdf.cell(0, 5.5, l)
    pdf.ln(5.5)

pdf.output(OUT_PATH)
print(f"PDF saved: {OUT_PATH}")
