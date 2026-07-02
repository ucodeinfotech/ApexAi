"""Comparison Report: Engulfing vs Big Candle - All Variants"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "comparison")
os.makedirs(OUT, exist_ok=True)
PLOT = os.path.join(OUT, "plots"); os.makedirs(PLOT, exist_ok=True)
plt.rcParams["figure.dpi"] = 150

def load_csv(path):
    p = os.path.join(BASE, path)
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_csv(p)
    if "exit_time" in df.columns: df["exit_time"] = pd.to_datetime(df["exit_time"])
    if "entry_time" in df.columns: df["entry_time"] = pd.to_datetime(df["entry_time"])
    return df

def calc(df):
    if df.empty or "points" not in df.columns: return {}
    df=df.copy(); t=len(df); w=df[df["points"]>0]; l=df[df["points"]<=0]
    wc=len(w); lc=len(l); gp=w["points"].sum() if wc else 0; gl=l["points"].sum() if lc else 0
    d=df.sort_values("exit_time").reset_index(drop=True) if "exit_time" in df.columns else df
    d["cum"]=d["points"].cumsum(); d["peak"]=d["cum"].cummax(); d["dd"]=d["peak"]-d["cum"]
    aw=w["points"].mean() if wc else 0; al=l["points"].mean() if lc else 0; 
    mw=w["points"].max() if wc else 0; ml=l["points"].min() if lc else 0
    return {"trades":t,"wins":wc,"losses":lc,"wr":round(wc/t*100,1) if t else 0,
            "net":round(df["points"].sum(),2),"pf":round(abs(gp/gl),2) if gl!=0 else (999 if gp>0 else 0),
            "avg_w":round(aw,2),"avg_l":round(al,2),"max_w":round(mw,2),"max_l":round(ml,2),
            "mdd":round(d["dd"].max(),2),"mdd_pct":round(d["dd"].max()/d["peak"].max()*100,1) if d["peak"].max()>0 else 0}

# ── Load all trade books ──
data = {}

# Engulfing
for sym in ["NIFTY50","SENSEX"]:
    for variant in ["Raw_FixTP","Filter_FixTP","Raw_Chan7","Filter_Chan7"]:
        fname = f"Engulf_{variant}" 
        df = load_csv(f"backtest_results/engulfing/{sym}_{fname}.csv") if os.path.exists(os.path.join(BASE,f"backtest_results/engulfing/{sym}_{fname}.csv")) else pd.DataFrame()
        data[f"Eng_{variant}_{sym}"] = (df, calc(df))

# Big Candle
bc_variants = [
    ("BC_PrevOpt_FixTP", "backtest_results/improvements/{}_OPTIMAL.csv"),
    ("BC_Sir_Chan7", "backtest_results/sir_strategy/{}_Sir_Trades.csv"),
    ("BC_Base_Chan7", "backtest_results/sir_strategy/{}_Base_Chandelier.csv"),
    ("BC_Base_FixTP", "backtest_results/sir_strategy/{}_Baseline.csv"),
]
for label, pattern in bc_variants:
    for sym in ["NIFTY50","SENSEX"]:
        df = load_csv(pattern.format(sym))
        data[f"{label}_{sym}"] = (df, calc(df))

# ── Global metrics ──
def combined_net(label, syms=["NIFTY50","SENSEX"]):
    return sum(data[f"{label}_{s}"][1].get("net",0) or 0 for s in syms)

def combined_mdd(label, syms=["NIFTY50","SENSEX"]):
    return max(data[f"{label}_{s}"][1].get("mdd",0) or 0 for s in syms)

def combined_tr(label, syms=["NIFTY50","SENSEX"]):
    return sum(data[f"{label}_{s}"][1].get("trades",0) or 0 for s in syms)

def combined_wr(label, syms=["NIFTY50","SENSEX"]):
    tr = combined_tr(label, syms)
    if not tr: return 0
    wr_sum = sum((data[f"{label}_{s}"][1].get("wr",0) or 0)*(data[f"{label}_{s}"][1].get("trades",0) or 0) for s in syms)
    return round(wr_sum/tr, 1)

# ── Plots ──
# Equity: best 4 strategies side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for sym_idx, sym in enumerate(["NIFTY50","SENSEX"]):
    ax = axes[sym_idx]
    lines = [
        ("Eng Raw+FixTP", f"Eng_{sym}_Raw_FixTP", "#E74C3C"),
        ("Eng Raw+Chan7", f"Eng_{sym}_Raw_Chan7", "#3498DB"),
        ("BC PrevOpt FIX", f"BC_PrevOpt_FixTP_{sym}", "#F39C12"),
        ("BC Base+Chan7", f"BC_Base_Chan7_{sym}", "#2ECC71"),
        ("BC Sir+Chan7", f"BC_Sir_Chan7_{sym}", "#9B59B6"),
    ]
    for label, key, clr in lines:
        df, m = data.get(key, (pd.DataFrame(), {}))
        if df.empty: continue
        dd = df.sort_values("exit_time").reset_index(drop=True)
        dd["cum"] = dd["points"].cumsum()
        ax.plot(dd.index, dd["cum"], label=f"{label} ({m.get('net',0):+.0f})", color=clr, lw=1.5, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"{sym}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade Sequence"); ax.set_ylabel("Points")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Equity Curve Comparison: Engulfing vs Big Candle", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(PLOT,"equity_compare.png"), bbox_inches="tight"); plt.close(fig)

# Bar chart: combined net
fig, ax = plt.subplots(figsize=(12, 5))
all_variants = [
    ("Eng Raw+FixTP", "Eng_Raw_FixTP"),
    ("Eng Raw+Chan7", "Eng_Raw_Chan7"),
    ("Eng Filter+FixTP", "Eng_Filter_FixTP"),
    ("Eng Filter+Chan7", "Eng_Filter_Chan7"),
    ("BC PrevOpt FIX", "BC_PrevOpt_FixTP"),
    ("BC Sir+Chan7", "BC_Sir_Chan7"),
    ("BC Base+Chan7", "BC_Base_Chan7"),
    ("BC Base+FIX", "BC_Base_FixTP"),
]
colors = ["#E74C3C","#3498DB","#F39C12","#2ECC71","#9B59B6","#1ABC9C","#E67E22","#95A5A6"]
nets = [combined_net(l) for _, l in all_variants]
bars = ax.barh(range(len(all_variants)), nets, color=colors, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(all_variants)))
ax.set_yticklabels([v[0] for v in all_variants], fontsize=8)
ax.set_xlabel("Combined Net Points", fontsize=11, fontweight="bold")
ax.set_title("Total Return Comparison: Engulfing vs Big Candle", fontsize=13, fontweight="bold")
ax.axvline(0, color="gray", ls="--", alpha=0.4)
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
for bar, val in zip(bars, nets):
    ax.text(val+(500 if val>0 else -2000), bar.get_y()+bar.get_height()/2,
            f"{val:+,.0f}", va="center", fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(PLOT,"bar_compare.png"), bbox_inches="tight"); plt.close(fig)

# MDD comparison bar
fig, ax = plt.subplots(figsize=(12, 5))
mdds = [combined_mdd(l) for _, l in all_variants]
bars = ax.barh(range(len(all_variants)), mdds, color=colors, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(all_variants)))
ax.set_yticklabels([v[0] for v in all_variants], fontsize=8)
ax.set_xlabel("Max Drawdown (pts)", fontsize=11, fontweight="bold")
ax.set_title("Max Drawdown Comparison", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
for bar, val in zip(bars, mdds):
    ax.text(val+200, bar.get_y()+bar.get_height()/2, f"{val:,.0f}", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(PLOT,"mdd_compare.png"), bbox_inches="tight"); plt.close(fig)

# Return/Drawdown ratio bar
fig, ax = plt.subplots(figsize=(12, 5))
ratios = [round(nets[i]/mdds[i],2) if mdds[i]>0 else 0 for i in range(len(all_variants))]
bars = ax.barh(range(len(all_variants)), ratios, color=colors, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(all_variants)))
ax.set_yticklabels([v[0] for v in all_variants], fontsize=8)
ax.set_xlabel("Return / Drawdown Ratio", fontsize=11, fontweight="bold")
ax.set_title("Risk-Adjusted Return (Net / MDD)", fontsize=13, fontweight="bold")
ax.axvline(0, color="gray", ls="--", alpha=0.4)
for bar, val in zip(bars, ratios):
    ax.text(val+0.1, bar.get_y()+bar.get_height()/2, f"{val:.2f}", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(PLOT,"ratio_compare.png"), bbox_inches="tight"); plt.close(fig)

# ── PDF Report ──
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9); self.set_text_color(100,100,100)
        self.cell(0,7,"Strategy Comparison: Engulfing vs Big Candle", align="L"); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("Helvetica","I",8); self.set_text_color(150,150,150)
        self.cell(0,10,f"Page {self.page_no()}/{{nb}}", align="C")
    def section(self,t):
        self.set_font("Helvetica","B",13); self.set_text_color(20,60,120)
        self.cell(0,9,t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20,60,120); self.line(15,self.get_y(),195,self.get_y()); self.ln(4)

pdf = PDF(); pdf.alias_nb_pages()

# ── Title ──
pdf.add_page(); pdf.ln(15)
pdf.set_font("Helvetica","B",22); pdf.set_text_color(20,60,120)
pdf.cell(0,12,"Strategy Comparison Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","",13); pdf.set_text_color(80,80,80)
pdf.cell(0,9,"Engulfing Pattern vs Big Candle Reversal", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica","",10); pdf.set_text_color(50,50,50)
pdf.cell(0,7,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,7,"NIFTY50 & SENSEX | 2015-2026 | BUY only | All variants", align="C", new_x="LMARGIN", new_y="NEXT")

pdf.ln(8)
pdf.set_fill_color(235,242,250); pdf.set_draw_color(20,60,120)
y0=pdf.get_y(); pdf.rect(12,y0,186,32,style="DF")
pdf.set_xy(16,y0+4); pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,6,"Strategies Covered", new_x="LMARGIN", new_y="NEXT")
descs = [
    ("Engulfing Pattern", "Bullish engulfing on 1H -> 5M breakout+retest entry"),
    ("Big Candle Rev.", "Big body + reversal candle on 1H -> 5M breakout+retest entry"),
    ("Exits", "Fixed 1:2 TP (SL at retest low) or Chandelier 7xATR"),
    ("Filters", "Optional: ADX>20, session 9:30-12:30, EMA50>200"),
]
for name, d in descs:
    pdf.set_xy(16,pdf.get_y()); pdf.set_font("Helvetica","B",8.5); pdf.set_text_color(40,40,40)
    pdf.cell(28,5.5,name)
    pdf.set_font("Helvetica","",8.5); pdf.set_text_color(80,80,80)
    pdf.cell(0,5.5,d, new_x="LMARGIN", new_y="NEXT")
pdf.set_y(y0+33)

# ── Combined Table ──
pdf.add_page(); pdf.section("Combined Results - All 8 Variants")
cols = [30,12,16,12,12,12,12,14]
hdr = ["Strategy","Tr","Net","WR%","PF","AvgW","AvgL","MDD"]
pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr,cols): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()

groups = [
    ("ENGULFING", ["Eng_Raw_FixTP","Eng_Raw_Chan7","Eng_Filter_FixTP","Eng_Filter_Chan7"]),
    ("BIG CANDLE", ["BC_Base_FixTP","BC_PrevOpt_FixTP","BC_Base_Chan7","BC_Sir_Chan7"]),
]
pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
for grp_name, keys in groups:
    pdf.set_font("Helvetica","B",8); pdf.set_text_color(20,60,120)
    pdf.cell(0,5.5,grp_name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
    for key in keys:
        net = combined_net(key)
        tr = combined_tr(key)
        wr = combined_wr(key)
        mdd = combined_mdd(key)
        n_m = data.get(f"{key}_NIFTY50",(pd.DataFrame(),{}))[1]
        s_m = data.get(f"{key}_SENSEX",(pd.DataFrame(),{}))[1]
        pf = round(((n_m.get("pf",0) or 0)+(s_m.get("pf",0) or 0))/2,2)
        aw = round(((n_m.get("avg_w",0) or 0)+(s_m.get("avg_w",0) or 0))/2,1)
        al = round(((n_m.get("avg_l",0) or 0)+(s_m.get("avg_l",0) or 0))/2,1)
        short = key.replace("Eng_","").replace("BC_","")
        vals = [short,str(tr),f"{net:+.0f}",f"{wr}%",f"{pf:.2f}",f"{aw:+.0f}",f"{al:+.0f}",f"{mdd:.0f}"]
        for v,c in zip(vals,cols): pdf.cell(c,5,str(v),border=1,align="C")
        pdf.ln()
    pdf.ln(2)

# Highlight best in each category
eng_best = max([(combined_net(k),k) for k in ["Eng_Raw_FixTP","Eng_Raw_Chan7","Eng_Filter_FixTP","Eng_Filter_Chan7"]])
bc_best  = max([(combined_net(k),k) for k in ["BC_Base_FixTP","BC_PrevOpt_FixTP","BC_Base_Chan7","BC_Sir_Chan7"]])
all_best = max([(combined_net(k),k) for k in ["Eng_Raw_FixTP","Eng_Raw_Chan7","Eng_Filter_FixTP","Eng_Filter_Chan7",
                                               "BC_Base_FixTP","BC_PrevOpt_FixTP","BC_Base_Chan7","BC_Sir_Chan7"]])
pdf.set_font("Helvetica","",9); pdf.set_text_color(20,60,120)
pdf.cell(0,6,f"Best Engulfing: {eng_best[1]} ({eng_best[0]:+,.0f})", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,6,f"Best Big Candle: {bc_best[1]} ({bc_best[0]:+,.0f})", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","B",9); pdf.set_text_color(20,120,60)
pdf.cell(0,6,f"Overall Best: {all_best[1]} ({all_best[0]:+,.0f})", new_x="LMARGIN", new_y="NEXT")

# Per-symbol
pdf.add_page(); pdf.section("Per-Symbol Detail")
cols2 = [30,12,16,12,12,12,12,14,12]
hdr2 = ["Strategy","Tr","Net","WR%","PF","AvgW","AvgL","MDD","Hold"]
for sym in ["NIFTY50","SENSEX"]:
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
    pdf.cell(0,6,f"{sym}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica","B",6.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
    for h,c in zip(hdr2,cols2): pdf.cell(c,5,h,border=1,align="C",fill=True)
    pdf.ln()
    pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
    for key_base in ["Eng_Raw_FixTP","Eng_Raw_Chan7","Eng_Filter_FixTP","Eng_Filter_Chan7",
                     "BC_Base_FixTP","BC_PrevOpt_FixTP","BC_Base_Chan7","BC_Sir_Chan7"]:
        df, m = data.get(f"{key_base}_{sym}", (pd.DataFrame(),{}))
        if not m: continue
        short = key_base.replace("Eng_","").replace("BC_","")
        vals = [short,str(m["trades"]),f'{m["net"]:+.0f}',f'{m["wr"]}%',f'{m["pf"]:.2f}',
                f'{m["avg_w"]:+.0f}',f'{m["avg_l"]:+.0f}',f'{m["mdd"]:.0f}',
                f'{m.get("avg_hold",0)}h']
        for v,c in zip(vals,cols2): pdf.cell(c,4.5,str(v),border=1,align="C")
        pdf.ln()
    pdf.ln(4)

# Plots
pdf.add_page(); pdf.section("Equity Curve Comparison")
pdf.image(os.path.join(PLOT,"equity_compare.png"),x=8,w=194)
pdf.add_page(); pdf.section("Total Return Comparison")
pdf.image(os.path.join(PLOT,"bar_compare.png"),x=8,w=194)
pdf.add_page(); pdf.section("Max Drawdown Comparison")
pdf.image(os.path.join(PLOT,"mdd_compare.png"),x=8,w=194)
pdf.add_page(); pdf.section("Risk-Adjusted Return (Net / MDD)")
pdf.image(os.path.join(PLOT,"ratio_compare.png"),x=8,w=194)

# Final insights
pdf.add_page(); pdf.section("Key Insights & Recommendations")
best_risk = max([(round(combined_net(k)/combined_mdd(k),4) if combined_mdd(k)>0 else 0, k) 
                 for k in ["Eng_Raw_FixTP","Eng_Raw_Chan7","Eng_Filter_FixTP","Eng_Filter_Chan7",
                           "BC_Base_FixTP","BC_PrevOpt_FixTP","BC_Base_Chan7","BC_Sir_Chan7"]])
insights = [
    ("Highest Net Points:", all_best[1], f"{all_best[0]:+,.0f} pts"),
    ("Best Risk-Adjusted (Net/MDD):", best_risk[1], f"{best_risk[0]:.2f}"),
    ("Best Engulfing Variant:", eng_best[1], f"{eng_best[0]:+,.0f} pts"),
    ("Best Big Candle Variant:", bc_best[1], f"{bc_best[0]:+,.0f} pts"),
    ("","",""),
    ("Key Takeaways:","",""),
    ("- Engulfing generates more trades (600/index) than Big Candle (170/index)","",""),
    ("- Engulfing Raw+Chan7 has highest return (+62,676) but MDD -12,278","",""),
    ("- Engulfing Raw+FixTP outperforms BC PrevOpt (+16,642 vs +10,612) with better PF","",""),
    ("- Filters hurt Engulfing more than Big Candle (especially on SENSEX)","",""),
    ("- Big Candle has better risk-adjusted (Net/MDD 4-14x vs Engulfing 2-3x)","",""),
    ("- Conservative: BC PrevOpt FixTP has best Net/MDD ratio","",""),
    ("- Aggressive: Engulfing Raw+FixTP best return without Chan7 blow-up risk","",""),
]
pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
for label, strat, val in insights:
    if not label and not strat and not val:
        pdf.ln(3)
    elif not strat and not val:
        pdf.set_font("Helvetica","B",9); pdf.set_text_color(20,60,120)
        pdf.cell(0,6,label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
    else:
        pdf.set_font("Helvetica","B",9); pdf.set_text_color(20,60,120)
        pdf.cell(0,6,f"{label} {strat} ({val})", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)

pdf.ln(8)
pdf.set_font("Helvetica","I",9); pdf.set_text_color(100,100,100)
pdf.cell(0,6,f"Report: {OUT}/Strategy_Comparison_v2.pdf", new_x="LMARGIN", new_y="NEXT")
pdf_path = os.path.join(OUT, "Strategy_Comparison_v2.pdf")
pdf.output(pdf_path)
print(f"Report: {pdf_path}")
