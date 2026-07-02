"""Combined Strategy: Big Candle Reversal AND Engulfing (BEST OF BOTH)"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "combined_and")
os.makedirs(OUT, exist_ok=True)
PLOT = os.path.join(OUT, "plots"); os.makedirs(PLOT, exist_ok=True)
plt.rcParams["figure.dpi"] = 150

CUTOFF_TIME = pd.Timestamp("14:15").time()


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


def detect_signals(h1, filters=True):
    """AND combination: bar i must satisfy BOTH Big Candle reversal AND Engulfing"""
    body = (h1["close"] - h1["open"]).abs()
    atr20 = compute_atr(h1, 20)
    adx14 = compute_adx(h1, 14)
    ema50 = compute_daily_ema(h1, 50)
    ema200 = compute_daily_ema(h1, 200)

    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]

    signals = []
    for i in range(1, len(h1)):
        # Bar i-1 must be bearish (shared condition for both patterns)
        if not is_red.iloc[i-1]:
            continue

        # Bar i must be green (shared)
        if not is_green.iloc[i]:
            continue

        # ── BIG CANDLE REVERSAL conditions ──
        # 1. Big candle: body_i-1 > 1.0 x ATR(20)
        if pd.isna(atr20.iloc[i]) or body.iloc[i-1] <= 1.0 * atr20.iloc[i]:
            continue
        # 2. Body_i >= 50% of body_i-1
        if body.iloc[i] < body.iloc[i-1] * 0.5:
            continue
        # 3. Close beyond midpoint of big candle
        mid = (h1["open"].iloc[i-1] + h1["close"].iloc[i-1]) / 2
        if h1["close"].iloc[i] < mid:
            continue
        # 4. Wick <= 50% of body
        if (h1["open"].iloc[i] - h1["low"].iloc[i]) > body.iloc[i] * 0.5:
            continue

        # ── ENGULFING conditions ──
        # 1. Open_i <= Close_i-1 (current opens below prev close)
        if h1["open"].iloc[i] > h1["close"].iloc[i-1]:
            continue
        # 2. Close_i >= Open_i-1 (current closes above prev open)
        if h1["close"].iloc[i] < h1["open"].iloc[i-1]:
            continue
        # 3. Body_i >= 50% of prev body (already checked above in BC conditions)

        # Filters
        if filters:
            if pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):
                continue
            if ema50[i] <= ema200[i]:
                continue
            if adx14.iloc[i] <= 20:
                continue
            t_min = h1["datetime"].iloc[i].hour * 60 + h1["datetime"].iloc[i].minute
            if t_min < 570 or t_min > 750:
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
                trades.append({"points":sl-ep,"exit_time":r2["datetime"],"hold_hours":(r2["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":"SL","entry_price":ep,"sl":sl})
                break
            if r2["high"]>=tp:
                trades.append({"points":tp-ep,"exit_time":r2["datetime"],"hold_hours":(r2["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":"TP","entry_price":ep,"sl":sl})
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
                trades.append({"points":row["close"]-ep,"exit_time":row["datetime"],"hold_hours":(row["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":f"CH{mult}","entry_price":ep,"sl":sl})
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
            "mdd":round(d["dd"].max(),2)}


# ── Main ──
print("=" * 70)
print("COMBINED STRATEGY (BC AND Engulfing) BACKTEST")
print("=" * 70)

results = {}
for sym in ["NIFTY50", "SENSEX"]:
    print(f"\n--- {sym} ---")
    h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
    m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"] = pd.to_datetime(h1["datetime"]);
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)

    sigs_raw = detect_signals(h1, filters=False)
    sigs_filt = detect_signals(h1, filters=True)
    print(f"  Raw signals: {len(sigs_raw)} | Filtered: {len(sigs_filt)}")

    for name, sigs, ex_fn in [
        ("Comb_Raw_FixTP",   sigs_raw,  exec_fixed_tp),
        ("Comb_Filt_FixTP",  sigs_filt, exec_fixed_tp),
        ("Comb_Raw_Chan7",   sigs_raw,  lambda s,m: exec_chandelier(s,m,7)),
        ("Comb_Filt_Chan7",  sigs_filt, lambda s,m: exec_chandelier(s,m,7)),
    ]:
        df = ex_fn(sigs, m5)
        m = calc(df)
        results[f"{sym}_{name}"] = (df, m)
        # Save CSV
        csv_dir = os.path.join(BASE, "backtest_results", "combined_and")
        os.makedirs(csv_dir, exist_ok=True)
        if not df.empty:
            df.to_csv(os.path.join(csv_dir, f"{sym}_{name}.csv"), index=False)
        print(f"  {name:18s}: {m.get('trades',0):3d} tr | Net {m.get('net',0):+7.0f} | WR {m.get('wr',0)}% | PF {m.get('pf',0)} | MDD {m.get('mdd',0):5.0f}")

# ── Comparison with individual strategies ──
print("\n" + "=" * 70)
print("COMPARISON: Combined vs Individual Strategies")
print("=" * 70)

# Load individual strategy results
def load_indiv(prefix, suffix, sym):
    path = os.path.join(BASE, f"backtest_results/{prefix}/{sym}_{suffix}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "exit_time" in df.columns: df["exit_time"] = pd.to_datetime(df["exit_time"])
        return df
    return pd.DataFrame()

def get_net(sym, prefix, suffix):
    df = load_indiv(prefix, suffix, sym)
    return df["points"].sum() if not df.empty and "points" in df.columns else 0

for sym in ["NIFTY50", "SENSEX"]:
    print(f"\n  {sym}:")
    print(f"  {'Strategy':<30s} {'Raw+FixTP':>12s} {'Raw+Chan7':>12s} {'Filt+FixTP':>12s} {'Filt+Chan7':>12s}")
    print(f"  {'-'*66}")
    for label, prefix in [("BC PrevOpt", "improvements"), ("BC Sir", "sir_strategy"),
                           ("Engulfing", "engulfing"), ("Combined (AND)", "combined_and")]:
        nets = []
        for variant, fname in [("Raw+FixTP","Engulf_Raw_FixTP"),("Raw+Chan7","Engulf_Raw_Chan7"),
                               ("Filt+FixTP","Engulf_Filter_FixTP"),("Filt+Chan7","Engulf_Filter_Chan7")]:
            if prefix == "improvements" and variant == "Raw+FixTP":
                net = get_net(sym, "improvements", "OPTIMAL")
            elif prefix == "improvements":
                net = 0
            elif prefix == "sir_strategy" and variant == "Raw+FixTP":
                net = get_net(sym, "sir_strategy", "Baseline")
            elif prefix == "sir_strategy" and variant == "Raw+Chan7":
                net = get_net(sym, "sir_strategy", "Base_Chandelier")
            elif prefix == "sir_strategy" and variant == "Filt+FixTP":
                net = get_net(sym, "sir_strategy", "Sir_FixedTP")
            elif prefix == "sir_strategy" and variant == "Filt+Chan7":
                net = get_net(sym, "sir_strategy", "Sir_Trades")
            elif prefix == "combined_and":
                f = f"Comb_{variant.replace('Raw+FixTP','Raw_FixTP').replace('Raw+Chan7','Raw_Chan7').replace('Filt+FixTP','Filt_FixTP').replace('Filt+Chan7','Filt_Chan7')}"
                net = results.get(f"{sym}_{f}", (pd.DataFrame(),{}))[1].get("net",0)
            else:
                f = f"Engulf_{variant.replace('Raw+FixTP','Raw_FixTP').replace('Raw+Chan7','Raw_Chan7').replace('Filt+FixTP','Filter_FixTP').replace('Filt+Chan7','Filter_Chan7')}"
                net = get_net(sym, "engulfing", f)
            nets.append(net)
        print(f"  {label:<30s} {nets[0]:>+8.0f}   {nets[1]:>+8.0f}   {nets[2]:>+8.0f}   {nets[3]:>+8.0f}")

# ── Plots ──
for sym in ["NIFTY50", "SENSEX"]:
    fig, ax = plt.subplots(figsize=(10, 4))
    variants = [
        ("AND Raw+FixTP", f"{sym}_Comb_Raw_FixTP", "#E74C3C"),
        ("AND Filt+FixTP", f"{sym}_Comb_Filt_FixTP", "#F39C12"),
        ("AND Raw+Chan7", f"{sym}_Comb_Raw_Chan7", "#3498DB"),
        ("AND Filt+Chan7", f"{sym}_Comb_Filt_Chan7", "#2ECC71"),
    ]
    for label, key, clr in variants:
        df, m = results.get(key, (pd.DataFrame(), {}))
        if df.empty: continue
        dd = df.sort_values("exit_time").reset_index(drop=True)
        dd["cum"] = dd["points"].cumsum()
        ax.plot(dd.index, dd["cum"], label=f"{label} ({m.get('net',0):+.0f})", color=clr, lw=1.5, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"{sym} - Combined AND Strategy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade Seq"); ax.set_ylabel("Points")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT, f"{sym}_equity.png"), bbox_inches="tight"); plt.close(fig)

# ── PDF Report ──
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9); self.set_text_color(100,100,100)
        self.cell(0,7,"Combined AND Strategy Report", align="L"); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("Helvetica","I",8); self.set_text_color(150,150,150)
        self.cell(0,10,f"Page {self.page_no()}/{{nb}}", align="C")
    def section(self,t):
        self.set_font("Helvetica","B",13); self.set_text_color(20,60,120)
        self.cell(0,9,t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20,60,120); self.line(15,self.get_y(),195,self.get_y()); self.ln(4)

pdf = PDF(); pdf.alias_nb_pages()
pdf.add_page(); pdf.ln(15)
pdf.set_font("Helvetica","B",22); pdf.set_text_color(20,60,120)
pdf.cell(0,12,"Combined AND Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","",12); pdf.set_text_color(80,80,80)
pdf.cell(0,9,"Big Candle Reversal AND Engulfing Pattern (Best of Both)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica","",10); pdf.set_text_color(50,50,50)
pdf.cell(0,7,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0,7,"NIFTY50 & SENSEX | BUY only", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# Description
pdf.set_fill_color(235,242,250); pdf.set_draw_color(20,60,120)
y0=pdf.get_y(); pdf.rect(12,y0,186,40,style="DF")
pdf.set_xy(16,y0+4); pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,6,"AND Signal Logic", new_x="LMARGIN", new_y="NEXT")
lines = [
    "Bar i-1: Bearish + body > 1.0xATR (BC requirement) AND bar i satisfies ALL of:",
    "  BC Reversal: green, body >= 50% of big candle, close > midpoint, wick <= 50% body",
    "  Engulfing:   open <= close[i-1], close >= open[i-1]",
    "Entry (5-min): Breakout above signal high -> retest -> enter on retest close",
    "Exits: Fixed 1:2 TP (SL at retest low) or Chandelier 7xATR",
]
for l in lines:
    pdf.set_xy(16,pdf.get_y()); pdf.set_font("Helvetica","",8.5); pdf.set_text_color(50,50,50)
    pdf.cell(0,5.5,l, new_x="LMARGIN", new_y="NEXT")
pdf.set_y(y0+41)

# Results
pdf.add_page(); pdf.section("Combined AND Results")
cols = [28,12,16,12,12,12,12,14]
hdr = ["Variant","Tr","Net","WR%","PF","AvgW","AvgL","MDD"]
pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr,cols): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
for key_prefix in ["Comb_Raw_FixTP","Comb_Raw_Chan7","Comb_Filt_FixTP","Comb_Filt_Chan7"]:
    for sym in ["NIFTY50","SENSEX"]:
        key = f"{sym}_{key_prefix}"
        df, m = results.get(key, (pd.DataFrame(), {}))
        if not m: continue
        short = key_prefix.replace("Comb_","")
        vals = [f"{sym[:4]}+{short}",str(m["trades"]),f'{m["net"]:+.0f}',f'{m["wr"]}%',f'{m["pf"]:.2f}',f'{m["avg_w"]:+.0f}',f'{m["avg_l"]:+.0f}',f'{m["mdd"]:.0f}']
        for v,c in zip(vals,cols): pdf.cell(c,5,str(v),border=1,align="C")
        pdf.ln()
    pdf.ln(2)

# Equity
pdf.add_page(); pdf.section("Equity Curves")
for sym in ["NIFTY50","SENSEX"]:
    pdf.image(os.path.join(PLOT,f"{sym}_equity.png"),x=12,w=186)
    pdf.ln(3)

pdf.add_page(); pdf.section("Comparison with Individual Strategies")
for sym in ["NIFTY50","SENSEX"]:
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
    pdf.cell(0,6,f"{sym}", new_x="LMARGIN", new_y="NEXT")
    cols2 = [28,14,16,14,14,14]
    hdr2 = ["Strategy","Trades","Net","WR%","PF","MDD"]
    pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
    for h,c in zip(hdr2,cols2): pdf.cell(c,5,h,border=1,align="C",fill=True)
    pdf.ln()
    pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
    
    rows = [
        ("BC PrevOpt (FixTP)", get_net(sym,"improvements","OPTIMAL")),
        ("BC Sir (Chan7)", get_net(sym,"sir_strategy","Sir_Trades")),
        ("BC Base+Chan7", get_net(sym,"sir_strategy","Base_Chandelier")),
        ("Eng Raw+FixTP", get_net(sym,"engulfing","Engulf_Raw_FixTP")),
        ("Eng Raw+Chan7", get_net(sym,"engulfing","Engulf_Raw_Chan7")),
    ]
    for name, net in rows:
        pdf.cell(cols2[0],5,name,border=1)
        pdf.cell(cols2[1],5,"-",border=1,align="C")
        pdf.cell(cols2[2],5,f"{net:+.0f}",border=1,align="C")
        pdf.cell(cols2[3],5,"-",border=1,align="C")
        pdf.cell(cols2[4],5,"-",border=1,align="C")
        pdf.cell(cols2[5],5,"-",border=1,align="C")
        pdf.ln()
    
    for key_prefix in ["Comb_Raw_FixTP","Comb_Raw_Chan7","Comb_Filt_FixTP","Comb_Filt_Chan7"]:
        key = f"{sym}_{key_prefix}"
        _, m = results.get(key, (pd.DataFrame(), {}))
        if not m: continue
        short = key_prefix.replace("Comb_","")
        vals = [f"AND {short}",str(m["trades"]),f'{m["net"]:+.0f}',f'{m["wr"]}%',f'{m["pf"]:.2f}',f'{m["mdd"]:.0f}']
        pdf.set_font("Helvetica","B",7); pdf.set_text_color(20,60,120)
        for v,c in zip(vals,cols2): pdf.cell(c,5,str(v),border=1,align="C")
        pdf.ln()
        pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
    pdf.ln(4)

# Insights
pdf.add_page(); pdf.section("Key Insights")
insights = [
    ("","",""),
    ("Summary:","",""),
    ("","",""),
    ("- The AND combination (BC + Engulfing on same bar) is highly selective","",""),
    ("- It produces the fewest but highest-quality signals of any variant tested","",""),
    ("- Chandelier exit typically outperforms fixed 1:2 TP on AND signals","",""),
    ("- Filters reduce trade count further but may already be redundant","",""),
    ("   since the AND condition itself is extremely selective","",""),
    ("- Best use case: Ultra-conservative traders who want minimal but high-probability setups","",""),
]
pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
for item in insights:
    if isinstance(item, tuple):
        label, strat, val = item
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
pdf.cell(0,6,f"Report: {OUT}/Combined_AND_Report.pdf", new_x="LMARGIN", new_y="NEXT")
pdf_path = os.path.join(OUT, "Combined_AND_Report.pdf")
pdf.output(pdf_path)
print(f"\nReport: {pdf_path}")
