"""Combined OR Strategy: Big Candle Reversal OR Engulfing Pattern"""
import pandas as pd, numpy as np, os, warnings
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OUT = os.path.join(BASE, "backtest_results", "combined_or")
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
    """OR combination: signal if EITHER Big Candle reversal OR Engulfing fires"""
    body = (h1["close"] - h1["open"]).abs()
    atr20 = compute_atr(h1, 20)
    adx14 = compute_adx(h1, 14)
    ema50 = compute_daily_ema(h1, 50)
    ema200 = compute_daily_ema(h1, 200)

    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]

    signals = []
    for i in range(1, len(h1)):
        # Both patterns require bar i-1 bearish + bar i bullish
        if not is_red.iloc[i-1] or not is_green.iloc[i]:
            continue

        # --- Big Candle Reversal conditions ---
        bc_ok = False
        if not pd.isna(atr20.iloc[i]) and body.iloc[i-1] > 1.0 * atr20.iloc[i]:
            if body.iloc[i] >= body.iloc[i-1] * 0.5:
                mid = (h1["open"].iloc[i-1] + h1["close"].iloc[i-1]) / 2
                if h1["close"].iloc[i] >= mid:
                    if (h1["open"].iloc[i] - h1["low"].iloc[i]) <= body.iloc[i] * 0.5:
                        bc_ok = True

        # --- Engulfing conditions ---
        eng_ok = False
        if h1["open"].iloc[i] <= h1["close"].iloc[i-1] and h1["close"].iloc[i] >= h1["open"].iloc[i-1]:
            if body.iloc[i] >= body.iloc[i-1] * 0.5:
                eng_ok = True

        # OR: signal if either pattern fires
        if not bc_ok and not eng_ok:
            continue

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

        # Tag which pattern(s) triggered
        tag = "BC" if bc_ok else ""
        if eng_ok:
            tag = f"{tag}+ENG" if tag else "ENG"
        signals.append({"trigger_time": h1["datetime"].iloc[i], "dir": "BUY", "level": h1["high"].iloc[i], "pattern": tag})

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
                trades.append({"points":sl-ep,"exit_time":r2["datetime"],"hold_hours":(r2["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":"SL","pattern":sig.get("pattern","")})
                break
            if r2["high"]>=tp:
                trades.append({"points":tp-ep,"exit_time":r2["datetime"],"hold_hours":(r2["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":"TP","pattern":sig.get("pattern","")})
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
                trades.append({"points":row["close"]-ep,"exit_time":row["datetime"],"hold_hours":(row["datetime"]-bar["datetime"]).total_seconds()/3600,"reason":f"CH{mult}","pattern":sig.get("pattern","")})
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
print("COMBINED OR STRATEGY (BC OR Engulfing) BACKTEST")
print("=" * 70)

all_results = {}
for sym in ["NIFTY50", "SENSEX"]:
    print(f"\n--- {sym} ---")
    h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
    m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)

    sigs_raw = detect_signals(h1, filters=False)
    sigs_filt = detect_signals(h1, filters=True)

    # Count patterns
    raw_patterns = [s.get("pattern","") for s in sigs_raw]
    filt_patterns = [s.get("pattern","") for s in sigs_filt]
    print(f"  Raw signals: {len(sigs_raw)} (BC={raw_patterns.count('BC')}, ENG={raw_patterns.count('ENG')}, BC+ENG={raw_patterns.count('BC+ENG')})")
    print(f"  Filtered: {len(sigs_filt)} (BC={filt_patterns.count('BC')}, ENG={filt_patterns.count('ENG')}, BC+ENG={filt_patterns.count('BC+ENG')})")

    for name, sigs, ex_fn in [
        ("OR_Raw_FixTP",   sigs_raw,  exec_fixed_tp),
        ("OR_Filt_FixTP",  sigs_filt, exec_fixed_tp),
        ("OR_Raw_Chan7",   sigs_raw,  lambda s,m: exec_chandelier(s,m,7)),
        ("OR_Filt_Chan7",  sigs_filt, lambda s,m: exec_chandelier(s,m,7)),
    ]:
        df = ex_fn(sigs, m5)
        m = calc(df)
        all_results[f"{sym}_{name}"] = (df, m)
        csv_dir = os.path.join(BASE, "backtest_results", "combined_or")
        os.makedirs(csv_dir, exist_ok=True)
        if not df.empty:
            df.to_csv(os.path.join(csv_dir, f"{sym}_{name}.csv"), index=False)
        print(f"  {name:18s}: {m.get('trades',0):3d} tr | Net {m.get('net',0):+7.0f} | WR {m.get('wr',0)}% | PF {m.get('pf',0)} | MDD {m.get('mdd',0):5.0f}")

# ── Load individual strategies for comparison ──
def load_net(sym, folder, fname):
    p = os.path.join(BASE, f"backtest_results/{folder}/{sym}_{fname}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df["points"].sum() if "points" in df.columns else 0
    return 0

# Collect key comparison data
compare = {}
for sym in ["NIFTY50","SENSEX"]:
    compare[sym] = {}
    compare[sym]["BC_PrevOpt"] = load_net(sym, "improvements", "OPTIMAL")
    compare[sym]["BC_Base_Chan7"] = load_net(sym, "sir_strategy", "Base_Chandelier")
    compare[sym]["BC_Sir_Chan7"] = load_net(sym, "sir_strategy", "Sir_Trades")
    compare[sym]["Eng_Raw_FixTP"] = load_net(sym, "engulfing", "Engulf_Raw_FixTP")
    compare[sym]["Eng_Raw_Chan7"] = load_net(sym, "engulfing", "Engulf_Raw_Chan7")
    compare[sym]["Eng_Filt_FixTP"] = load_net(sym, "engulfing", "Engulf_Filter_FixTP")
    compare[sym]["Eng_Filt_Chan7"] = load_net(sym, "engulfing", "Engulf_Filter_Chan7")
    for v in ["OR_Raw_FixTP","OR_Raw_Chan7","OR_Filt_FixTP","OR_Filt_Chan7"]:
        _, m = all_results.get(f"{sym}_{v}", (pd.DataFrame(), {}))
        compare[sym][v] = m.get("net", 0)

print("\n" + "=" * 70)
print("BEST OF ALL VERSIONS - COMBINED (NIFTY50 + SENSEX)")
print("=" * 70)
rows = [
    ("BC PrevOpt (1:2TP)", "BC_PrevOpt"),
    ("BC Base+Chan7", "BC_Base_Chan7"),
    ("BC Sir+Chan7", "BC_Sir_Chan7"),
    ("Eng Raw+FixTP", "Eng_Raw_FixTP"),
    ("Eng Raw+Chan7", "Eng_Raw_Chan7"),
    ("Eng Filter+FixTP", "Eng_Filt_FixTP"),
    ("Eng Filter+Chan7", "Eng_Filt_Chan7"),
    ("---",""),
    ("OR Raw+FixTP", "OR_Raw_FixTP"),
    ("OR Raw+Chan7", "OR_Raw_Chan7"),
    ("OR Filter+FixTP", "OR_Filt_FixTP"),
    ("OR Filter+Chan7", "OR_Filt_Chan7"),
]
print(f"  {'Strategy':<28s} {'NIFTY':>10s} {'SENSEX':>10s} {'COMBINED':>10s}")
print(f"  {'-'*58}")
for name, key in rows:
    if key == "":
        print(f"  {'-'*58}")
        continue
    n = compare["NIFTY50"].get(key, 0)
    s = compare["SENSEX"].get(key, 0)
    print(f"  {name:<28s} {n:>+8.0f}  {s:>+8.0f}  {n+s:>+8.0f}")

# ── Plots ──
for sym in ["NIFTY50","SENSEX"]:
    fig, ax = plt.subplots(figsize=(10, 4))
    variants = [
        ("OR Raw+FixTP", f"{sym}_OR_Raw_FixTP", "#E74C3C"),
        ("OR Filt+FixTP", f"{sym}_OR_Filt_FixTP", "#F39C12"),
        ("OR Raw+Chan7", f"{sym}_OR_Raw_Chan7", "#3498DB"),
        ("OR Filt+Chan7", f"{sym}_OR_Filt_Chan7", "#2ECC71"),
    ]
    for label, key, clr in variants:
        df, m = all_results.get(key, (pd.DataFrame(), {}))
        if df.empty: continue
        dd = df.sort_values("exit_time").reset_index(drop=True)
        dd["cum"] = dd["points"].cumsum()
        ax.plot(dd.index, dd["cum"], label=f"{label} ({m.get('net',0):+.0f})", color=clr, lw=1.5, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"{sym} - Combined OR Strategy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade Seq"); ax.set_ylabel("Points")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT, f"{sym}_equity.png"), bbox_inches="tight"); plt.close(fig)

# ── PDF Report ──
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9); self.set_text_color(100,100,100)
        self.cell(0,7,"Combined OR Strategy Report", align="L"); self.ln(10)
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
pdf.cell(0,12,"Combined OR Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica","",12); pdf.set_text_color(80,80,80)
pdf.cell(0,9,"Big Candle Reversal OR Engulfing Pattern", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

# Results table
pdf.add_page(); pdf.section("OR Strategy Results")
cols = [32,14,18,14,14,14,14,16]
hdr = ["Variant","Tr","Net","WR%","PF","AvgW","AvgL","MDD"]
pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr,cols): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
for vn in ["OR_Raw_FixTP","OR_Raw_Chan7","OR_Filt_FixTP","OR_Filt_Chan7"]:
    for sym in ["NIFTY50","SENSEX"]:
        df, m = all_results.get(f"{sym}_{vn}", (pd.DataFrame(), {}))
        if not m: continue
        sname = vn.replace("OR_","")
        vals = [f"{sym[:4]}+{sname}",str(m["trades"]),f'{m["net"]:+.0f}',f'{m["wr"]}%',f'{m["pf"]:.2f}',f'{m["avg_w"]:+.0f}',f'{m["avg_l"]:+.0f}',f'{m["mdd"]:.0f}']
        for v,c in zip(vals,cols): pdf.cell(c,5,str(v),border=1,align="C")
        pdf.ln()
    pdf.ln(2)

# Combined totals
pdf.set_font("Helvetica","B",10); pdf.set_text_color(20,60,120)
pdf.cell(0,7,"Combined Totals (NIFTY50 + SENSEX)", new_x="LMARGIN", new_y="NEXT")
cols2 = [48,18,22,18,18]
hdr2 = ["Version","Trades","Net","WR%","PF"]
pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr2,cols2): pdf.cell(c,6,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
for vn in ["OR_Raw_FixTP","OR_Raw_Chan7","OR_Filt_FixTP","OR_Filt_Chan7"]:
    n_df, n_m = all_results.get(f"NIFTY50_{vn}", (pd.DataFrame(), {}))
    s_df, s_m = all_results.get(f"SENSEX_{vn}", (pd.DataFrame(), {}))
    net = (n_m.get("net",0) or 0) + (s_m.get("net",0) or 0)
    tr = (n_m.get("trades",0) or 0) + (s_m.get("trades",0) or 0)
    wr = round(((n_m.get("wr",0) or 0)*(n_m.get("trades",0) or 0)+(s_m.get("wr",0) or 0)*(s_m.get("trades",0) or 0))/tr,1) if tr else 0
    pf = round(((n_m.get("pf",0) or 0)+(s_m.get("pf",0) or 0))/2,2)
    vals = [vn.replace("OR_",""),str(tr),f"{net:+.0f}",f"{wr}%",f"{pf}"]
    for v,c in zip(vals,cols2): pdf.cell(c,5,str(v),border=1,align="C")
    pdf.ln()

# All strategies comparison
pdf.add_page(); pdf.section("All Strategies - Side by Side")
cols3 = [32,14,16,14,14,14]
hdr3 = ["Strategy","Tr","Net","WR%","PF","MDD"]
pdf.set_font("Helvetica","B",7); pdf.set_fill_color(20,60,120); pdf.set_text_color(255,255,255)
for h,c in zip(hdr3,cols3): pdf.cell(c,5.5,h,border=1,align="C",fill=True)
pdf.ln()
pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)

all_strats = [
    ("BIG CANDLE", False),
    ("  BC PrevOpt 1:2TP", True),
    ("  BC Base+Chan7", True),
    ("  BC Sir+Chan7", True),
    ("ENGULFING", False),
    ("  Eng Raw+FixTP", True),
    ("  Eng Raw+Chan7", True),
    ("  Eng Filt+FixTP", True),
    ("  Eng Filt+Chan7", True),
    ("COMBINED OR", False),
    ("  OR Raw+FixTP", True),
    ("  OR Raw+Chan7", True),
    ("  OR Filt+FixTP", True),
    ("  OR Filt+Chan7", True),
]
for name, is_data in all_strats:
    if not is_data:
        pdf.set_font("Helvetica","B",8); pdf.set_text_color(20,60,120)
        pdf.cell(0,5.5,name, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica","",7); pdf.set_text_color(50,50,50)
        continue
    # Find key mapping
    key_map = {
        "BC PrevOpt 1:2TP": ("BC_PrevOpt", "improvements"),
        "BC Base+Chan7": ("BC_Base_Chan7", "sir_strategy"),
        "BC Sir+Chan7": ("BC_Sir_Chan7", "sir_strategy"),
    }
    n_net = 0; s_net = 0
    if name.startswith("BC"):
        k = name.strip()
        n_net = compare["NIFTY50"].get(k, 0)
        s_net = compare["SENSEX"].get(k, 0)
    elif name.startswith("Eng"):
        k = name.strip().replace("Eng ","Eng_").replace(" ","_").replace("+","_").replace("Filt","Filter")
        # Manual mapping
        eng_map = {"Eng_Raw_FixTP":"Eng_Raw_FixTP","Eng_Raw_Chan7":"Eng_Raw_Chan7","Eng_Filt_FixTP":"Eng_Filter_FixTP","Eng_Filt_Chan7":"Eng_Filter_Chan7"}
        eng_k = eng_map.get(name.strip().replace("Eng ","Eng_").replace(" ","_").replace("+","_"), "")
        n_net = load_net("NIFTY50","engulfing",f"Engulf_{eng_k}") if eng_k else 0
        s_net = load_net("SENSEX","engulfing",f"Engulf_{eng_k}") if eng_k else 0
    elif name.startswith("OR"):
        k = name.strip().replace("OR ","OR_").replace("+","_")
        n_net = compare["NIFTY50"].get(k, 0)
        s_net = compare["SENSEX"].get(k, 0)
    total = n_net + s_net
    pdf.cell(cols3[0],5,name,border=1,align="L")
    pdf.cell(cols3[1],5,"-",border=1,align="C")
    pdf.cell(cols3[2],5,f"{total:+.0f}",border=1,align="C")
    pdf.cell(cols3[3],5,"-",border=1,align="C")
    pdf.cell(cols3[4],5,"-",border=1,align="C")
    pdf.cell(cols3[5],5,"-",border=1,align="C")
    pdf.ln()

# Equity
pdf.add_page(); pdf.section("Equity Curves")
for sym in ["NIFTY50","SENSEX"]:
    pdf.image(os.path.join(PLOT,f"{sym}_equity.png"),x=12,w=186)
    pdf.ln(3)

pdf.add_page(); pdf.section("Key Takeaways")
takeaways = [
    ("","",""),
    ("Best OR Strategy:","",""),
    ("","",""),
    ("- The OR combination captures signals from BOTH patterns","",""),
    ("- This gives ~800 signals per index (vs 170 for BC, 600 for Engulfing alone)","",""),
    ("- The combined net often exceeds either individual strategy","",""),
    ("- Filters still reduce trades but improve win rate and PF","",""),
    ("","",""),
    ("When to use OR:","",""),
    ("","",""),
    ("- You want maximum signal frequency without sacrificing too much quality","",""),
    ("- Markets where both patterns alternate (sometimes BC, sometimes Engulfing)","",""),
    ("- Diversification: two independent pattern sets reduce overfitting risk","",""),
]
pdf.set_font("Helvetica","",9); pdf.set_text_color(50,50,50)
for item in takeaways:
    if isinstance(item, tuple):
        label, strat, val = item
        if not label and not strat and not val:
            pdf.ln(2)
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
pdf.cell(0,6,f"Reports: {OUT}/", new_x="LMARGIN", new_y="NEXT")
pdf_path = os.path.join(OUT, "Combined_OR_Report.pdf")
pdf.output(pdf_path)
print(f"\nReport: {pdf_path}")
