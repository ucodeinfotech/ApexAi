"""Comprehensive Engulfing Strategy Analysis - All variants, P&L, Exclusions, Report"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "engulfing")
PLOT = os.path.join(OUT, "analysis_plots"); os.makedirs(PLOT, exist_ok=True)

CAP = 100000; NLOT = 50; SLOT = 10; CHG = 20


def load(sym, variant):
    """variant: Engulf_Raw_FixTP etc"""
    p = os.path.join(BASE, "backtest_results", "engulfing", f"{sym}_{variant}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["date"] = df["exit_time"].dt.date
        df["pnl"] = df["points"] * (NLOT if "NIFTY" in sym else SLOT) - CHG
        return df
    return pd.DataFrame()


def analyze(name, nifty, sensex):
    """Compute full metrics for a variant"""
    n = nifty.copy() if not nifty.empty else pd.DataFrame()
    s = sensex.copy() if not sensex.empty else pd.DataFrame()
    
    n_net = n["pnl"].sum() if not n.empty else 0
    s_net = s["pnl"].sum() if not s.empty else 0
    tot = n_net + s_net
    tr = (len(n) if not n.empty else 0) + (len(s) if not s.empty else 0)
    days = (set(n["date"]) if not n.empty else set()) | (set(s["date"]) if not s.empty else set())
    
    def stats(df):
        if df.empty: return {"wins":0,"losses":0,"wr":0,"gp":0,"gl":0,"pf":0}
        w=df[df["pnl"]>0]; l=df[df["pnl"]<=0]
        gp=w["pnl"].sum(); gl=abs(l["pnl"].sum())
        return {"wins":len(w),"losses":len(l),"wr":len(w)/len(df)*100,"gp":gp,"gl":gl,"pf":gp/gl if gl>0 else (999 if gp>0 else 0)}
    
    ns = stats(n); ss = stats(s)
    
    # Combined daily
    n_daily = n.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl":"n_pnl"}) if not n.empty else pd.DataFrame(columns=["date","n_pnl"])
    s_daily = s.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl":"s_pnl"}) if not s.empty else pd.DataFrame(columns=["date","s_pnl"])
    comb = pd.merge(n_daily, s_daily, on="date", how="outer").fillna(0).sort_values("date").reset_index(drop=True)
    comb["total"] = comb["n_pnl"] + comb["s_pnl"]
    comb["cum"] = comb["total"].cumsum()
    peak = comb["cum"].cummax(); dd = peak - comb["cum"]; mdd = dd.max()
    returns = comb["total"] / (CAP * 2)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    cagr = ((1 + tot/(CAP*2)) ** (1/10) - 1) * 100 if tot > -(CAP*2) else -100
    win_months = sum(1 for v in comb.groupby(pd.to_datetime(comb["date"]).dt.to_period("M"))["total"].sum() if v > 0)
    tot_months = len(comb.groupby(pd.to_datetime(comb["date"]).dt.to_period("M")))
    
    # Max consecutive losing days
    comb["win"] = comb["total"] > 0
    max_cl = 0; cur_cl = 0
    for w in comb["win"]:
        if not w: cur_cl += 1
        else: cur_cl = 0
        max_cl = max(max_cl, cur_cl)
    
    top = comb.nlargest(1, "total").iloc[0] if not comb.empty else None
    bot = comb.nsmallest(1, "total").iloc[0] if not comb.empty else None
    
    return {"name":name,"trades":tr,"days":len(days),"n_net":n_net,"s_net":s_net,"net":tot,
            "n_wr":ns["wr"],"s_wr":ss["wr"],"n_pf":ns["pf"],"s_pf":ss["pf"],
            "mdd":mdd,"mdd_pct":mdd/(CAP*2)*100,"sharpe":sharpe,"cagr":cagr,
            "win_months":win_months,"tot_months":tot_months,"max_cl":max_cl,
            "best_day":top["date"] if top is not None else "","best_val":top["total"] if top is not None else 0,
            "worst_day":bot["date"] if bot is not None else "","worst_val":bot["total"] if bot is not None else 0,
            "comb_df":comb}


def exclude_march(df):
    """Exclude all of March 2020"""
    if df.empty: return df
    mar = pd.Timestamp("2020-03-01").date()
    apr = pd.Timestamp("2020-03-31").date()
    return df[~((df["date"] >= mar) & (df["date"] <= apr))]


# ── All variants ──
variants = ["Engulf_Raw_FixTP", "Engulf_Raw_Chan7", "Engulf_Filter_FixTP", "Engulf_Filter_Chan7"]
all_results = {}

print("=" * 75)
print("COMPREHENSIVE ENGULFING STRATEGY ANALYSIS")
print("=" * 75)

for v in variants:
    n = load("NIFTY50", v)
    s = load("SENSEX", v)
    m = analyze(v, n, s)
    all_results[v] = {"full": m}
    
    # March 2020 excluded
    n_ex = exclude_march(n)
    s_ex = exclude_march(s)
    m_ex = analyze(f"{v} (no Mar2020)", n_ex, s_ex)
    all_results[v]["excl"] = m_ex
    
    print(f"\n  {v}:")
    print(f"    Full:     {m['trades']:3d} tr | Rs{m['net']:>+8,.0f} | CAGR {m['cagr']:+.1f}% | MDD {m['mdd_pct']:.1f}% | Sharpe {m['sharpe']:.2f}")
    print(f"    NoMar20:  {m_ex['trades']:3d} tr | Rs{m_ex['net']:>+8,.0f} | CAGR {m_ex['cagr']:+.1f}% | MDD {m_ex['mdd_pct']:.1f}% | Sharpe {m_ex['sharpe']:.2f}")

# ── Summary table ──
print(f"\n{'='*75}")
print("SUMMARY TABLE - ALL VARIANTS")
print(f"{'='*75}")
print(f"  {'Variant':<28s} {'Trades':>7s} {'Net P&L':>12s} {'CAGR':>7s} {'MDD%':>6s} {'Sharpe':>7s} {'WR%':>5s} {'PF':>5s}")
print(f"  {'-'*77}")
for v in variants:
    m = all_results[v]["full"]
    print(f"  {m['name']:<28s} {m['trades']:>5d}   Rs{m['net']:>+8,.0f} {m['cagr']:>+6.1f}% {m['mdd_pct']:>5.1f}% {m['sharpe']:>7.2f} {(m['n_wr']+m['s_wr'])/2:>5.1f}% {(m['n_pf']+m['s_pf'])/2:>5.2f}")
print(f"  {'-'*77}")
for v in variants:
    m = all_results[v]["excl"]
    print(f"  {m['name']:<28s} {m['trades']:>5d}   Rs{m['net']:>+8,.0f} {m['cagr']:>+6.1f}% {m['mdd_pct']:>5.1f}% {m['sharpe']:>7.2f} {(m['n_wr']+m['s_wr'])/2:>5.1f}% {(m['n_pf']+m['s_pf'])/2:>5.2f}")

# ── Plots ──
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for idx, (title, key) in enumerate([("Full Period", "full"), ("Excluding March 2020", "excl")]):
    ax = axes[idx]
    colors = {"Engulf_Raw_FixTP":"#E74C3C","Engulf_Raw_Chan7":"#3498DB","Engulf_Filter_FixTP":"#F39C12","Engulf_Filter_Chan7":"#2ECC71"}
    for v in variants:
        df = all_results[v][key]["comb_df"]
        if df.empty: continue
        ax.plot(range(len(df)), df["cum"], label=f"{v} (Rs{all_results[v][key]['net']:+,.0f})", color=colors[v], lw=1.5, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"Engulfing Strategy - {title}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative P&L (Rs)"); ax.set_xlabel("Trading Days")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("Rs{x:,.0f}"))
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PLOT, "equity_all.png"), bbox_inches="tight"); plt.close(fig)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(variants))
w = 0.35
nets_full = [all_results[v]["full"]["net"] for v in variants]
nets_excl = [all_results[v]["excl"]["net"] for v in variants]
bars1 = ax.bar(x - w/2, nets_full, w, label="Full Period", color="#E74C3C", alpha=0.8)
bars2 = ax.bar(x + w/2, nets_excl, w, label="Excl Mar 2020", color="#3498DB", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([v.replace("Engulf_","") for v in variants], fontsize=8)
ax.set_ylabel("Net P&L (Rs)"); ax.set_title("Engulfing - Full vs Excl March 2020", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("Rs{x:,.0f}"))
ax.legend()
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(10000 if bar.get_height()>0 else -50000), f"Rs{bar.get_height():+,.0f}", ha="center", fontsize=7, rotation=90)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(10000 if bar.get_height()>0 else -50000), f"Rs{bar.get_height():+,.0f}", ha="center", fontsize=7, rotation=90)
fig.tight_layout(); fig.savefig(os.path.join(PLOT, "bar_full_vs_excl.png"), bbox_inches="tight"); plt.close(fig)

# ── PDF Report ──
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9); self.set_text_color(100,100,100)
        self.cell(0,7,"Engulfing Strategy - Comprehensive Analysis", align="L"); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("Helvetica","I",8); self.set_text_color(150,150,150)
        self.cell(0,10,f"Page {self.page_no()}/{{nb}}", align="C")
    def section(self,t):
        self.set_font("Helvetica","B",13); self.set_text_color(20,60,120)
        self.cell(0,9,t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20,60,120); self.line(15,self.get_y(),195,self.get_y()); self.ln(4)

pdf = PDF(); pdf.alias_nb_pages()

# Title
pdf.add_page(); pdf.ln(15)
pdf.set_font("Helvetica","B",22); pdf.set_text_color(20,60,120)
pdf.cell(0,12,"Engulfing Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","",13); pdf.set_text_color(80,80,80)
pdf.cell(0,9,"Comprehensive Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica","",10); pdf.set_text_color(50,50,50)
pdf.cell(0,7,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,7,"NIFTY50 & SENSEX | Rs1L capital each | All 4 variants", align="C", new_x="LMARGIN", new_y="NEXT")

pdf.ln(8)
pdf.set_fill_color(235,242,250); pdf.set_draw_color(20,60,120)
y0=pdf.get_y(); pdf.rect(12,y0,186,32,style="DF")
pdf.set_xy(16,y0+4); pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,6,"Variants & Parameters", new_x="LMARGIN", new_y="NEXT")
vdesc = [
    ("Raw+FixTP", "No filters, Fixed 1:2 TP"),
    ("Raw+Chan7", "No filters, Chandelier Exit 7xATR"),
    ("Filter+FixTP", "ADX>20 + session + EMA50/200 + Fixed 1:2"),
    ("Filter+Chan7", "ADX>20 + session + EMA50/200 + Chandelier 7xATR"),
]
for name, d in vdesc:
    pdf.set_xy(16,pdf.get_y()); pdf.set_font("Helvetica","B",8.5); pdf.set_text_color(40,40,40)
    pdf.cell(22,5.5,name)
    pdf.set_font("Helvetica","",8.5); pdf.set_text_color(80,80,80)
    pdf.cell(0,5.5,d, new_x="LMARGIN", new_y="NEXT")
pdf.set_y(y0+33)

# Results table
pdf.add_page(); pdf.section("All Variants - Full Period")
cols = [36,10,16,12,10,10,16,12]
hdr = ["Variant","Tr","Net P&L","CAGR","MDD%","Shp","Best Day","Worst"]
pdf.set_font("Helvetica","B",6.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr,cols): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
for v in variants:
    m = all_results[v]["full"]
    short = v.replace("Engulf_","")
    vals = [short,str(m["trades"]),f'Rs{m["net"]:+,.0f}',f'{m["cagr"]:+.1f}%',f'{m["mdd_pct"]:.1f}%',f'{m["sharpe"]:.2f}',
            f'{m["best_day"]} ({m["best_val"]:+,.0f})',f'{m["worst_day"]} ({m["worst_val"]:+,.0f})']
    for vv,cc in zip(vals,cols): pdf.cell(cc,5,str(vv),border=1,align="C")
    pdf.ln()
pdf.ln(3)

# Excl March 2020
pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,7,"Excluding March 2020", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","B",6.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr,cols): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
for v in variants:
    m = all_results[v]["excl"]
    short = v.replace("Engulf_","") + " (no Mar)"
    vals = [short,str(m["trades"]),f'Rs{m["net"]:+,.0f}',f'{m["cagr"]:+.1f}%',f'{m["mdd_pct"]:.1f}%',f'{m["sharpe"]:.2f}',
            f'{m["best_day"]} ({m["best_val"]:+,.0f})',f'{m["worst_day"]} ({m["worst_val"]:+,.0f})']
    for vv,cc in zip(vals,cols): pdf.cell(cc,5,str(vv),border=1,align="C")
    pdf.ln()

# Per-symbol detail
pdf.add_page(); pdf.section("Per-Symbol Detail (Raw+FixTP)")
cols2 = [30,10,14,12,10,10,28]
hdr2 = ["Variant","Tr","Net","WR%","PF","MDD","Monthly Win"]
pdf.set_font("Helvetica","B",6.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr2,cols2): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
for v in ["Engulf_Raw_FixTP","Engulf_Raw_Chan7"]:
    m = all_results[v]["full"]
    for sym_lbl, n_net, s_net, wr, pf in [("NIFTY50",m["n_net"],None,m["n_wr"],m["n_pf"]),("SENSEX",None,m["s_net"],m["s_wr"],m["s_pf"])]:
        net = n_net if n_net is not None else s_net
        vals = [f"{v.replace('Engulf_','')} {sym_lbl}","-",f'Rs{net:+,.0f}',f'{wr:.1f}%',f'{pf:.2f}',"-",f'{m["win_months"]}/{m["tot_months"]}']
        for vv,cc in zip(vals,cols2): pdf.cell(cc,5,str(vv),border=1,align="C")
        pdf.ln()
    pdf.ln(2)

# Compare all 4 variants
pdf.add_page(); pdf.section("Variant Comparison")
cols3 = [32,12,16,10,10,10,10,12]
hdr3 = ["Variant (Full)","Tr","Net P&L","CAGR","MDD%","Shp","WR%","PF"]
pdf.set_font("Helvetica","B",6.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr3,cols3): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
for v in variants:
    m = all_results[v]["full"]
    wr_avg = (m["n_wr"]+m["s_wr"])/2
    pf_avg = (m["n_pf"]+m["s_pf"])/2
    vals = [v.replace("Engulf_",""),str(m["trades"]),f'Rs{m["net"]:+,.0f}',f'{m["cagr"]:+.1f}%',f'{m["mdd_pct"]:.1f}%',f'{m["sharpe"]:.2f}',f'{wr_avg:.1f}%',f'{pf_avg:.2f}']
    for vv,cc in zip(vals,cols3): pdf.cell(cc,5,str(vv),border=1,align="C")
    pdf.ln()

pdf.ln(5); pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,7,"Excluding March 2020", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","B",6.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr3,cols3): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",6.5); pdf.set_text_color(50,50,50)
for v in variants:
    m = all_results[v]["excl"]
    wr_avg = (m["n_wr"]+m["s_wr"])/2
    pf_avg = (m["n_pf"]+m["s_pf"])/2
    vals = [v.replace("Engulf_",""),str(m["trades"]),f'Rs{m["net"]:+,.0f}',f'{m["cagr"]:+.1f}%',f'{m["mdd_pct"]:.1f}%',f'{m["sharpe"]:.2f}',f'{wr_avg:.1f}%',f'{pf_avg:.2f}']
    for vv,cc in zip(vals,cols3): pdf.cell(cc,5,str(vv),border=1,align="C")
    pdf.ln()

# Plots
pdf.add_page(); pdf.section("Equity Curves - All 4 Variants")
pdf.image(os.path.join(PLOT,"equity_all.png"),x=8,w=194)
pdf.add_page(); pdf.section("Full Period vs Excl March 2020")
pdf.image(os.path.join(PLOT,"bar_full_vs_excl.png"),x=15,w=180)

# Best variant deep dive
best_v = "Engulf_Raw_FixTP"
pdf.add_page(); pdf.section(f"Deep Dive: {best_v.replace('Engulf_','')}")
m = all_results[best_v]["full"]
m_ex = all_results[best_v]["excl"]
details = [
    f"Full Period:",
    f"  Net P&L: Rs{m['net']:+,.0f} (NIFTY Rs{m['n_net']:+,.0f} + SENSEX Rs{m['s_net']:+,.0f})",
    f"  Return: {m['net']/(CAP*2)*100:+.1f}% | CAGR: {m['cagr']:+.1f}% | Sharpe: {m['sharpe']:.2f}",
    f"  Max DD: Rs{m['mdd']:+,.0f} ({m['mdd_pct']:.1f}%)",
    f"  Best Day: {m['best_day']} (Rs{m['best_val']:+,.0f}) | Worst: {m['worst_day']} (Rs{m['worst_val']:+,.0f})",
    f"  Max Cons Loss Days: {m['max_cl']} | Win Months: {m['win_months']}/{m['tot_months']}",
    f"",
    f"Excluding March 2020:",
    f"  Net P&L: Rs{m_ex['net']:+,.0f}",
    f"  Return: {m_ex['net']/(CAP*2)*100:+.1f}% | CAGR: {m_ex['cagr']:+.1f}% | Sharpe: {m_ex['sharpe']:.2f}",
    f"  Max DD: Rs{m_ex['mdd']:+,.0f} ({m_ex['mdd_pct']:.1f}%)",
    f"  Max Cons Loss Days: {m_ex['max_cl']} | Win Months: {m_ex['win_months']}/{m_ex['tot_months']}",
    f"",
    f"Verdict: Engulfing strategy is heavily dependent on COVID volatility.",
    f"Without Mar 2020, CAGR drops from {m['cagr']:+.1f}% to {m_ex['cagr']:+.1f}%.",
]
pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
for l in details: pdf.cell(0,6,l, new_x="LMARGIN", new_y="NEXT")

pdf.ln(8); pdf.set_font("Helvetica","I",9); pdf.set_text_color(100,100,100)
pdf.cell(0,6,f"Reports: {OUT}/", new_x="LMARGIN", new_y="NEXT")
pdf_path = os.path.join(OUT, "Engulfing_Comprehensive_Report.pdf")
pdf.output(pdf_path)
print(f"\nReport: {pdf_path}")
