"""Engulfing Pattern Strategy - Converted to our backtesting framework"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "engulfing")
os.makedirs(OUT, exist_ok=True)
PLOT = os.path.join(OUT, "plots"); os.makedirs(PLOT, exist_ok=True)
plt.rcParams["figure.dpi"] = 150

CUTOFF_TIME = pd.Timestamp("14:15").time()
MIN_BODY_PCT = 50.0  # Engulfing: current body >= 50% of prev body


def compute_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_adx(df, period=14):
    tr = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    up=df["high"]-df["high"].shift(1); down=df["low"].shift(1)-df["low"]
    pdm=((up>down)&(up>0))*up; ndm=((down>up)&(down>0))*down
    atr=tr.rolling(period).mean(); pdi=100*(pdm.rolling(period).mean()/atr); ndi=100*(ndm.rolling(period).mean()/atr)
    dx=100*(abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan))
    return dx.rolling(period).mean()


def compute_daily_ema(df_1h, period=50):
    df=df_1h.copy(); df["date"]=df["datetime"].dt.normalize()
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"})
    return df["date"].map(daily["close"].ewm(span=period,adjust=False).mean()).values


def detect_engulfing_signals(h1, filters=True):
    """Detect bullish engulfing on 1-hour chart"""
    body = (h1["close"]-h1["open"]).abs()
    atr20 = compute_atr(h1, 20)
    adx14 = compute_adx(h1, 14)
    ema50 = compute_daily_ema(h1, 50)
    ema200 = compute_daily_ema(h1, 200)

    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]

    signals = []
    for i in range(1, len(h1)):
        # Engulfing: prev bearish + current bullish + wraps prev
        if not is_red.iloc[i-1]:
            continue
        if not is_green.iloc[i]:
            continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1]:
            continue  # open not <= prev close
        if h1["close"].iloc[i] < h1["open"].iloc[i-1]:
            continue  # close not >= prev open
        if body.iloc[i] < body.iloc[i-1] * (MIN_BODY_PCT / 100):
            continue  # body filter

        if filters:
            if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):
                continue
            if ema50[i] <= ema200[i]:
                continue
            if adx14.iloc[i] <= 20:
                continue
            t_min = h1["datetime"].iloc[i].hour * 60 + h1["datetime"].iloc[i].minute
            if t_min < 570 or t_min > 750:  # 9:30-12:30
                continue

        signals.append({"trigger_time": h1["datetime"].iloc[i], "dir": "BUY", "level": h1["high"].iloc[i]})
    return signals


def exec_fixed_tp(sigs, m5):
    trades = []
    tc = m5["datetime"].dt.time
    for sig in sigs:
        t, lv = sig["trigger_time"], sig["level"]
        scan = m5[m5["datetime"] > t]
        if scan.empty: continue
        b = scan[scan["close"] > lv]
        if b.empty: continue
        pb = scan.loc[b.index[0]+1:]
        if pb.empty: continue
        r = (pb["low"]<lv)&(pb["close"]>lv)&(tc.loc[pb.index]<CUTOFF_TIME)
        if not r.any(): continue
        bar = scan.loc[r.idxmax()]; ep, sl = bar["close"], bar["low"]
        if ep-sl <= 0: continue
        if bar["datetime"].hour == 9: continue
        tp = ep + 2*(ep-sl)
        xs = scan.loc[bar.name+1:]
        for _, r2 in xs.iterrows():
            if r2["low"]<=sl:
                trades.append({"points":sl-ep,"exit_time":r2["datetime"],"hold_hours":(r2["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":"SL"})
                break
            if r2["high"]>=tp:
                trades.append({"points":tp-ep,"exit_time":r2["datetime"],"hold_hours":(r2["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":"TP"})
                break
    return pd.DataFrame(trades)


def exec_chandelier(sigs, m5, mult=7):
    trades = []; tc = m5["datetime"].dt.time
    atr5 = compute_atr(m5, 14)
    for sig in sigs:
        t, lv = sig["trigger_time"], sig["level"]
        scan = m5[m5["datetime"] > t]
        if scan.empty: continue
        b = scan[scan["close"] > lv]
        if b.empty: continue
        pb = scan.loc[b.index[0]+1:]
        if pb.empty: continue
        r = (pb["low"]<lv)&(pb["close"]>lv)&(tc.loc[pb.index]<CUTOFF_TIME)
        if not r.any(): continue
        bar = scan.loc[r.idxmax()]; ep, sl = bar["close"], bar["low"]
        if ep-sl <= 0: continue
        if bar["datetime"].hour == 9: continue
        xs = scan.loc[bar.name+1:]
        if xs.empty: continue
        hi = ep
        for idx, row in xs.iterrows():
            ca = atr5.loc[idx]
            if pd.isna(ca): continue
            if row["high"] > hi: hi = row["high"]
            st = hi - mult*ca
            if row["close"] < st:
                trades.append({"points":row["close"]-ep,"exit_time":row["datetime"],"hold_hours":(row["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":f"CH{mult}"})
                break
    return pd.DataFrame(trades)


def calc(df):
    if df.empty or "points" not in df.columns: return {}
    t=len(df); w=df[df["points"]>0]; l=df[df["points"]<=0]; wc=len(w); lc=len(l)
    gp=w["points"].sum() if wc else 0; gl=l["points"].sum() if lc else 0
    d=df.sort_values("exit_time").reset_index(drop=True)
    d["cum"]=d["points"].cumsum(); d["peak"]=d["cum"].cummax(); d["dd"]=d["peak"]-d["cum"]
    aw=w["points"].mean() if wc else 0; al=l["points"].mean() if lc else 0
    mw=w["points"].max() if wc else 0; ml=l["points"].min() if lc else 0
    return {"trades":t,"wins":wc,"losses":lc,"wr":round(wc/t*100,1) if t else 0,
            "net":round(df["points"].sum(),2),"pf":round(abs(gp/gl),2) if gl!=0 else (999 if gp>0 else 0),
            "avg_w":round(aw,2),"avg_l":round(al,2),"max_w":round(mw,2),"max_l":round(ml,2),
            "mdd":round(d["dd"].max(),2),"mdd_pct":round(d["dd"].max()/d["peak"].max()*100,1) if d["peak"].max()>0 else 0,
            "sharpe":round(df["points"].mean()/df["points"].std()*np.sqrt(t),2) if df["points"].std()>0 else 0}


# ── Main ──
print("=" * 70)
print("ENGULFING PATTERN STRATEGY BACKTEST")
print("=" * 70)

results = {}
for sym in ["NIFTY50", "SENSEX"]:
    print(f"\n--- {sym} ---")
    h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
    m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)

    # All 4 combos
    sigs_no_filter = detect_engulfing_signals(h1, filters=False)
    sigs_filter = detect_engulfing_signals(h1, filters=True)

    for name, sigs, ex_fn in [
        ("Engulf_Raw_FixTP",   sigs_no_filter, exec_fixed_tp),
        ("Engulf_Filter_FixTP", sigs_filter, exec_fixed_tp),
        ("Engulf_Raw_Chan7",   sigs_no_filter, lambda s,m: exec_chandelier(s,m,7)),
        ("Engulf_Filter_Chan7", sigs_filter,  lambda s,m: exec_chandelier(s,m,7)),
    ]:
        df = ex_fn(sigs, m5)
        m = calc(df)
        results[f"{sym}_{name}"] = (df, m)
        key = f"  {name:22s}"
        if m:
            print(f"{key}: {m['trades']:3d} tr | Net {m['net']:+7.0f} | WR {m['wr']}% | PF {m['pf']} | MDD {m['mdd']:5.0f}")
        else:
            print(f"{key}: 0 trades")

    # Save CSVs
    csv_dir = os.path.join(BASE, "backtest_results", "engulfing")
    os.makedirs(csv_dir, exist_ok=True)
    for name, sigs, ex_fn in [
        ("Engulf_Raw_FixTP",   sigs_no_filter, exec_fixed_tp),
        ("Engulf_Filter_FixTP", sigs_filter, exec_fixed_tp),
        ("Engulf_Raw_Chan7",   sigs_no_filter, lambda s,m: exec_chandelier(s,m,7)),
        ("Engulf_Filter_Chan7", sigs_filter,  lambda s,m: exec_chandelier(s,m,7)),
    ]:
        df = results[f"{sym}_{name}"][0]
        if not df.empty:
            df.to_csv(os.path.join(csv_dir, f"{sym}_{name}.csv"), index=False)

# ── Plots ──
for sym in ["NIFTY50", "SENSEX"]:
    fig, ax = plt.subplots(figsize=(10, 4))
    variants = [
        ("Raw+FixTP", f"{sym}_Engulf_Raw_FixTP", "#E74C3C"),
        ("Filter+FixTP", f"{sym}_Engulf_Filter_FixTP", "#F39C12"),
        ("Raw+Chan7", f"{sym}_Engulf_Raw_Chan7", "#3498DB"),
        ("Filter+Chan7", f"{sym}_Engulf_Filter_Chan7", "#2ECC71"),
    ]
    for label, key, clr in variants:
        df, m = results.get(key, (pd.DataFrame(), {}))
        if df.empty: continue
        dd = df.sort_values("exit_time").reset_index(drop=True)
        dd["cum"] = dd["points"].cumsum()
        ax.plot(dd.index, dd["cum"], label=f"{label} ({m.get('net',0):+.0f}pts)", color=clr, lw=1.5, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"{sym} - Engulfing Strategy Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade Seq"); ax.set_ylabel("Points")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT, f"{sym}_equity.png"), bbox_inches="tight"); plt.close(fig)


# ── PDF Report ──
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9); self.set_text_color(100,100,100)
        self.cell(0,7,"Engulfing Pattern Strategy Report", align="L"); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("Helvetica","I",8); self.set_text_color(150,150,150)
        self.cell(0,10,f"Page {self.page_no()}/{{nb}}", align="C")
    def section(self,t):
        self.set_font("Helvetica","B",13); self.set_text_color(20,60,120)
        self.cell(0,9,t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20,60,120); self.line(15,self.get_y(),195,self.get_y()); self.ln(4)

pdf = PDF(); pdf.alias_nb_pages()

# Title
pdf.add_page(); pdf.ln(20)
pdf.set_font("Helvetica","B",24); pdf.set_text_color(20,60,120)
pdf.cell(0,12,"Engulfing Pattern Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","",13); pdf.set_text_color(80,80,80)
pdf.cell(0,9,"Pine Script Conversion - Backtest Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Helvetica","",10); pdf.set_text_color(50,50,50)
pdf.cell(0,7,f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,7,"NIFTY50 & SENSEX | 2015-2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

# Description
pdf.set_fill_color(235,242,250); pdf.set_draw_color(20,60,120)
y0=pdf.get_y(); pdf.rect(12,y0,186,42,style="DF")
pdf.set_xy(16,y0+4); pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,6,"Pattern Definition (from Pine Script)", new_x="LMARGIN", new_y="NEXT")
lines = [
    "Bullish Engulfing: prev bearish candle + current bullish wraps prev body",
    "Bearish Engulfing: prev bullish candle + current bearish wraps prev body",
    "Body Filter: current body >= 50% of previous body (configurable in Pine)",
    "Entry (5-min): Breakout above signal candle high -> retest close -> enter",
    "Exits: Fixed 1:2 TP (SL at retest candle low) or Chandelier Exit 7xATR",
    "Filters (optional): ADX>20, session 9:30-12:30, EMA50>200 (same as Sir Strategy)",
]
for l in lines:
    pdf.set_xy(16,pdf.get_y()); pdf.set_font("Helvetica","",8.5); pdf.set_text_color(50,50,50)
    pdf.cell(0,5.5,l, new_x="LMARGIN", new_y="NEXT")
pdf.set_y(y0+43)

# Results table
pdf.add_page(); pdf.section("Results Summary")
cols = [38,14,18,14,14,14,14,14]
hdr = ["Version","Tr","Net","WR%","PF","AvgW","AvgL","MDD"]
pdf.set_font("Helvetica","B",7.5); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr,cols): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",7.5); pdf.set_text_color(50,50,50)

for label, keys in [
    ("NIFTY50", ["NIFTY50_Engulf_Raw_FixTP","NIFTY50_Engulf_Filter_FixTP","NIFTY50_Engulf_Raw_Chan7","NIFTY50_Engulf_Filter_Chan7"]),
    ("SENSEX",  ["SENSEX_Engulf_Raw_FixTP","SENSEX_Engulf_Filter_FixTP","SENSEX_Engulf_Raw_Chan7","SENSEX_Engulf_Filter_Chan7"]),
]:
    pdf.set_font("Helvetica","B",8); pdf.set_text_color(20,60,120)
    pdf.cell(0,6,label, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica","",7.5); pdf.set_text_color(50,50,50)
    for key in keys:
        _, m = results.get(key, (pd.DataFrame(), {}))
        if not m: continue
        short = key.replace("NIFTY50_","").replace("SENSEX_","")
        vals = [short,str(m["trades"]),f'{m["net"]:+.0f}',f'{m["wr"]}%',f'{m["pf"]:.2f}',f'{m["avg_w"]:+.0f}',f'{m["avg_l"]:+.0f}',f'{m["mdd"]:.0f}']
        for v,c in zip(vals,cols): pdf.cell(c,5,str(v),border=1,align="C")
        pdf.ln()
    pdf.ln(3)

# Equity
pdf.add_page(); pdf.section("Equity Curves")
for sym in ["NIFTY50", "SENSEX"]:
    pdf.image(os.path.join(PLOT,f"{sym}_equity.png"),x=12,w=186)
    pdf.ln(3)

pdf.add_page(); pdf.section("Comparison with Big Candle Strategy")
# Load big candle results for comparison
try:
    n_bc_fix = pd.read_csv(os.path.join(BASE, "backtest_results", "improvements", "NIFTY50_OPTIMAL.csv"))
    s_bc_fix = pd.read_csv(os.path.join(BASE, "backtest_results", "improvements", "SENSEX_OPTIMAL.csv"))
    n_bc_chan = pd.read_csv(os.path.join(BASE, "backtest_results", "sir_strategy", "NIFTY50_Base_Chandelier.csv"))
    s_bc_chan = pd.read_csv(os.path.join(BASE, "backtest_results", "sir_strategy", "SENSEX_Base_Chandelier.csv"))
    bc_fix_net = n_bc_fix["points"].sum() + s_bc_fix["points"].sum()
    bc_chan_net = n_bc_chan["points"].sum() + s_bc_chan["points"].sum()
    
    eng_fix = sum(results[k][1]["net"] for k in ["NIFTY50_Engulf_Raw_FixTP","SENSEX_Engulf_Raw_FixTP"] if k in results)
    eng_chan = sum(results[k][1]["net"] for k in ["NIFTY50_Engulf_Raw_Chan7","SENSEX_Engulf_Raw_Chan7"] if k in results)

    lines = [
        f"Big Candle + FixTP: {bc_fix_net:+.0f} pts",
        f"Engulfing + FixTP:  {eng_fix:+.0f} pts ({(eng_fix/bc_fix_net*100):.0f}% of Big Candle)",
        f"Big Candle + Chan7: {bc_chan_net:+.0f} pts",
        f"Engulfing + Chan7:  {eng_chan:+.0f} pts ({(eng_chan/bc_chan_net*100):.0f}% of Big Candle)",
    ]
    pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
    for l in lines: pdf.cell(0,7,l, new_x="LMARGIN", new_y="NEXT")
except: pass

pdf.ln(10)
pdf.set_font("Helvetica","I",9); pdf.set_text_color(100,100,100)
pdf.cell(0,6,f"Reports: {OUT}/", new_x="LMARGIN", new_y="NEXT")

pdf_path = os.path.join(OUT, "Engulfing_Strategy_Report.pdf")
pdf.output(pdf_path)
print(f"\nReport: {pdf_path}")
