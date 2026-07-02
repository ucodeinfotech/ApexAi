"""Comparison Report: Prev Optimal vs Sir Strategy vs Base+Chandelier"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "comparison")
os.makedirs(OUT, exist_ok=True)
PLOT = os.path.join(OUT, "plots")
os.makedirs(PLOT, exist_ok=True)

plt.rcParams["figure.dpi"] = 150

# ── Load data ──
def load(path):
    df = pd.read_csv(os.path.join(BASE, path))
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df

n_opt = load("backtest_results/improvements/NIFTY50_OPTIMAL.csv")
s_opt = load("backtest_results/improvements/SENSEX_OPTIMAL.csv")
n_sir = load("backtest_results/sir_strategy/NIFTY50_Sir_Trades.csv")
s_sir = load("backtest_results/sir_strategy/SENSEX_Sir_Trades.csv")
n_bc  = load("backtest_results/sir_strategy/NIFTY50_Base_Chandelier.csv")
s_bc  = load("backtest_results/sir_strategy/SENSEX_Base_Chandelier.csv")

strats = {
    "Prev Optimal (1:2TP)": (n_opt, s_opt, "#E74C3C"),
    "Sir Strategy (Chan7)":  (n_sir, s_sir, "#2ECC71"),
    "Base+Chandelier 7x":    (n_bc,  s_bc,  "#3498DB"),
}

def calc(df):
    if df.empty or "points" not in df.columns: return {}
    df=df.copy(); t=len(df); w=df[df["points"]>0]; l=df[df["points"]<=0]
    wc=len(w); lc=len(l); gp=w["points"].sum() if wc else 0; gl=l["points"].sum() if lc else 0
    d=df.sort_values("exit_time").reset_index(drop=True)
    d["cum"]=d["points"].cumsum(); d["peak"]=d["cum"].cummax(); d["dd"]=d["peak"]-d["cum"]
    aw=w["points"].mean() if wc else 0; al=l["points"].mean() if lc else 0
    mw=w["points"].max() if wc else 0; ml=l["points"].min() if lc else 0
    return {"trades":t,"wins":wc,"losses":lc,"wr":round(wc/t*100,1) if t else 0,
            "net":round(df["points"].sum(),2),"pf":round(abs(gp/gl),2) if gl!=0 else (999 if gp>0 else 0),
            "avg_w":round(aw,2),"avg_l":round(al,2),"max_w":round(mw,2),"max_l":round(ml,2),
            "mdd":round(d["dd"].max(),2),
            "mdd_pct":round(d["dd"].max()/d["peak"].max()*100,1) if d["peak"].max()>0 else 0,
            "sharpe":round(df["points"].mean()/df["points"].std()*np.sqrt(t),2) if df["points"].std()>0 else 0,
            "avg_hold":round(df["hold_hours"].mean(),1) if "hold_hours" in df.columns else 0}

# ── Plots: Equity curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for sym_idx, (sym, label) in enumerate([("NIFTY50", "a"), ("SENSEX", "b")]):
    ax = axes[sym_idx]
    for name, (n, s, clr) in strats.items():
        d = n if sym == "NIFTY50" else s
        if d.empty: continue
        dd = d.sort_values("exit_time").reset_index(drop=True)
        dd["cum"] = dd["points"].cumsum()
        ax.plot(dd.index, dd["cum"], label=name, color=clr, lw=1.5, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"{sym} - Equity Curve Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade Sequence"); ax.set_ylabel("Cumulative Points")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOT, "equity_comparison.png"), bbox_inches="tight")
plt.close(fig)

# ── Drawdown comparison ──
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
for row, (name, (n, s, clr)) in enumerate(strats.items()):
    for col, (sym, df_data) in enumerate([("NIFTY50", n), ("SENSEX", s)]):
        ax = axes[row][col]
        d = df_data.sort_values("exit_time").reset_index(drop=True)
        d["cum"] = d["points"].cumsum(); d["peak"] = d["cum"].cummax()
        d["dd"] = d["cum"] - d["peak"]
        ax.fill_between(d.index, d["dd"], 0, color=clr, alpha=0.4)
        mdd = d["dd"].min()
        ax.annotate(f"Max: {mdd:,.0f}", xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8, color=clr, fontweight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec=clr, alpha=0.8))
        ax.set_title(f"{name} - {sym} DD", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
fig.tight_layout()
fig.savefig(os.path.join(PLOT, "dd_comparison.png"), bbox_inches="tight")
plt.close(fig)

# ── Side-by-side bar chart ──
fig, ax = plt.subplots(figsize=(10, 5))
names = list(strats.keys())
nets = [calc(strats[n][0])["net"] + calc(strats[n][1])["net"] for n in names]
colors = [strats[n][2] for n in names]
bars = ax.bar(names, nets, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, nets):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(500 if val>0 else -1500),
            f"{val:+,.0f}", ha="center", va="bottom" if val>0 else "top", fontsize=11, fontweight="bold")
ax.axhline(0, color="gray", ls="--", alpha=0.4)
ax.set_ylabel("Combined Net Points", fontsize=11, fontweight="bold")
ax.set_title("Total Return Comparison (NIFTY50 + SENSEX)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
fig.tight_layout()
fig.savefig(os.path.join(PLOT, "bar_comparison.png"), bbox_inches="tight")
plt.close(fig)

# ── PDF Report ──
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9); self.set_text_color(100,100,100)
        self.cell(0,7,"Strategy Comparison Report", align="L"); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("Helvetica","I",8); self.set_text_color(150,150,150)
        self.cell(0,10,f"Page {self.page_no()}/{{nb}}", align="C")
    def section(self,t):
        self.set_font("Helvetica","B",13); self.set_text_color(20,60,120)
        self.cell(0,9,t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20,60,120); self.line(15,self.get_y(),195,self.get_y()); self.ln(4)

pdf = PDF(); pdf.alias_nb_pages()

# ── Page 1: Title ──
pdf.add_page(); pdf.ln(20)
pdf.set_font("Helvetica","B",24); pdf.set_text_color(20,60,120)
pdf.cell(0,12,"Strategy Comparison Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","",13); pdf.set_text_color(80,80,80)
pdf.cell(0,9,"Previous Optimal vs Sir Strategy vs Base+Chandelier", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Helvetica","",10); pdf.set_text_color(50,50,50)
pdf.cell(0,7,f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | NIFTY50 & SENSEX | 2015-2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

# Description box
pdf.set_fill_color(235,242,250); pdf.set_draw_color(20,60,120)
y0=pdf.get_y(); pdf.rect(12,y0,186,56,style="DF")
desc = [
    ("Prev Optimal (1:2TP)", "Baseline signals (1.5xSMA20, BUY, skip9am, skip14+ for SENSEX) + Fixed 1:2 Risk-Reward exit"),
    ("Sir Strategy (Chan7)", "Sir filters (1.0xATR20, ADX>20, session 9:30-12:30, EMA50>200, BUY) + Chandelier Exit 7xATR"),
    ("Base+Chandelier 7x",  "Baseline signals (all, no filters) + Chandelier Exit 7xATR (close < highest_high - 7xATR)"),
]
pdf.set_xy(16,y0+4); pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,6,"Strategies Compared", new_x="LMARGIN", new_y="NEXT")
for name, d in desc:
    pdf.set_xy(16,pdf.get_y()); pdf.set_font("Helvetica","B",9); pdf.set_text_color(40,40,40)
    pdf.cell(48,6,name)
    pdf.set_font("Helvetica","",8.5); pdf.set_text_color(80,80,80)
    pdf.cell(0,6,d, new_x="LMARGIN", new_y="NEXT")

pdf.set_y(y0+57)
pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
pdf.cell(0,6,"Note: All strategies use BUY-only direction. BANKNIFTY excluded. Charges: Rs10/order.", new_x="LMARGIN", new_y="NEXT")

# ── Page 2: Combined Summary ──
pdf.add_page(); pdf.section("Combined Results (NIFTY50 + SENSEX)")
cols_w = [42,18,22,18,18,22,18,22]
headers = ["Strategy","Trades","Net Pts","WR%","PF","AvgW","AvgL","MDD"]
pdf.set_font("Helvetica","B",8); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(headers,cols_w): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",8); pdf.set_text_color(50,50,50)
for name, (n,s,_) in strats.items():
    mn,ms=calc(n),calc(s)
    net=(mn.get("net",0) or 0)+(ms.get("net",0) or 0)
    tr=(mn.get("trades",0) or 0)+(ms.get("trades",0) or 0)
    wr=round(((mn.get("wr",0) or 0)*(mn.get("trades",0) or 0)+(ms.get("wr",0) or 0)*(ms.get("trades",0) or 0))/tr,1)
    pf=round(((mn.get("pf",0) or 0)+(ms.get("pf",0) or 0))/2,2)
    aw=round(((mn.get("avg_w",0) or 0)+(ms.get("avg_w",0) or 0))/2,1)
    al=round(((mn.get("avg_l",0) or 0)+(ms.get("avg_l",0) or 0))/2,1)
    mdd=max(mn.get("mdd",0) or 0, ms.get("mdd",0) or 0)
    vals=[name,str(tr),f"{net:+.0f}",f"{wr}%",f"{pf}",f"{aw:+.0f}",f"{al:+.0f}",f"{mdd:.0f}"]
    for v,c in zip(vals,cols_w): pdf.cell(c,5.5,str(v),border=1,align="C")
    pdf.ln()
pdf.ln(3)
pdf.set_font("Helvetica","",9); pdf.set_text_color(20,60,120)
best=max((calc(strats[n][0])["net"]+calc(strats[n][1])["net"],n) for n in strats)
pdf.cell(0,6,f"Highest net: {best[1]} ({best[0]:+,.0f} pts)", new_x="LMARGIN", new_y="NEXT")

# ── Page 3: Per-symbol detail ──
for sym in ["NIFTY50", "SENSEX"]:
    pdf.add_page(); pdf.section(f"{sym} - Detailed Comparison")
    cols_w2 = [38,14,18,14,14,14,14,14,14,14,14]
    hdrs = ["Strategy","Tr","Net","WR%","PF","AvgW","AvgL","MaxW","MaxL","MDD","Hold"]
    pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
    for h,c in zip(hdrs,cols_w2): pdf.cell(c,5,h,border=1,align="C",fill=True)
    pdf.ln()
    pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
    for name, (n, s, _) in strats.items():
        d = n if sym == "NIFTY50" else s
        m = calc(d)
        if not m: continue
        vals = [name,str(m["trades"]),f'{m["net"]:+.0f}',f'{m["wr"]}%',f'{m["pf"]:.2f}',
                f'{m["avg_w"]:+.0f}',f'{m["avg_l"]:+.0f}',f'{m["max_w"]:+.0f}',f'{m["max_l"]:+.0f}',
                f'{m["mdd"]:.0f}',f'{m["avg_hold"]}h']
        for v,c in zip(vals,cols_w2): pdf.cell(c,4.5,str(v),border=1,align="C")
        pdf.ln()

# ── Page 5: Equity curves ──
pdf.add_page(); pdf.section("Equity Curve Comparison")
pdf.image(os.path.join(PLOT,"equity_comparison.png"),x=10,w=190)
pdf.ln(3)

# ── Page 6: Drawdown ──
pdf.add_page(); pdf.section("Drawdown Comparison")
pdf.image(os.path.join(PLOT,"dd_comparison.png"),x=10,w=190)
pdf.ln(3)

# ── Page 7: Bar chart + Insights ──
pdf.add_page(); pdf.section("Total Return Comparison")
pdf.image(os.path.join(PLOT,"bar_comparison.png"),x=20,w=170)
pdf.ln(5)
pdf.section("Summary & Recommendations")
insights = [
    "Key Findings:",
    "",
    f"1. Prev Optimal (1:2TP): {calc(n_opt)['net']+calc(s_opt)['net']:+,.0f} pts, MDD {max(calc(n_opt)['mdd'],calc(s_opt)['mdd']):.0f} pts",
    f"   - Best risk-adjusted returns with lowest drawdown",
    f"   - Ideal for conservative traders",
    "",
    f"2. Sir Strategy (Chan7): {calc(n_sir)['net']+calc(s_sir)['net']:+,.0f} pts, MDD {max(calc(n_sir)['mdd'],calc(s_sir)['mdd']):.0f} pts",
    f"   - Filters reduce trades 76% but also reduce net by 29% vs Prev Optimal",
    f"   - Chandelier exit captures extended moves but adds drawdown",
    "",
    f"3. Base+Chandelier: {calc(n_bc)['net']+calc(s_bc)['net']:+,.0f} pts, MDD {max(calc(n_bc)['mdd'],calc(s_bc)['mdd']):.0f} pts",
    f"   - Highest absolute returns (2.1x Prev Optimal)",
    f"   - But 5.4x the MDD of Prev Optimal",
    "",
    "Recommendation:",
    "- Aggressive: Base+Chandelier 7x (+22,454 pts, accept higher MDD)",
    "- Conservative: Prev Optimal with 1:2TP (+10,612 pts, lowest MDD)",
    "- Balanced: Sir Strategy filters + Chandelier (+7,546 pts, moderate profile)",
]
pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
for ln in insights: pdf.cell(0,5.5,ln, new_x="LMARGIN", new_y="NEXT")

pdf.ln(8)
pdf.set_font("Helvetica","I",9); pdf.set_text_color(100,100,100)
pdf.cell(0,6,f"Report saved: comparison/", new_x="LMARGIN", new_y="NEXT")

pdf_path = os.path.join(OUT, "Strategy_Comparison_Report.pdf")
pdf.output(pdf_path)
print(f"Report: {pdf_path}")
