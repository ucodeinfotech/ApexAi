"""Generate individual PDF reports for NIFTY50 and SENSEX"""
import pandas as pd
import numpy as np
import os, warnings
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from fpdf import FPDF
warnings.filterwarnings("ignore")

DATA_DIR = "."
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MULTIPLIER = 1.5
STRONG_BODY_PCT = 50.0
LENGTH = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()
CHARGES_PER_ORDER = 10

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def detect_signals(df_1h):
    body = (df_1h["close"] - df_1h["open"]).abs()
    avg_body = body.rolling(LENGTH, min_periods=LENGTH).mean()
    is_green = df_1h["close"] > df_1h["open"]
    is_red = df_1h["close"] < df_1h["open"]
    big_buy = is_green & (body > avg_body * MULTIPLIER)
    big_sell = is_red & (body > avg_body * MULTIPLIER)
    signals = []
    for i in range(1, len(df_1h)):
        if pd.isna(avg_body.iloc[i]):
            continue
        if big_buy.iloc[i - 1]:
            if not is_red.iloc[i]: continue
            prev_body = body.iloc[i - 1]
            curr_body = body.iloc[i]
            if curr_body < prev_body * (STRONG_BODY_PCT / 100): continue
            mid = (df_1h["open"].iloc[i - 1] + df_1h["close"].iloc[i - 1]) / 2
            if df_1h["close"].iloc[i] > mid: continue
            upper_wick = df_1h["high"].iloc[i] - df_1h["open"].iloc[i]
            if upper_wick > curr_body * 0.5: continue
            signals.append({"trigger_time": df_1h["datetime"].iloc[i], "dir": "SELL",
                            "level": df_1h["low"].iloc[i], "trigger_high": round(df_1h["high"].iloc[i], 2),
                            "trigger_low": round(df_1h["low"].iloc[i], 2)})
        elif big_sell.iloc[i - 1]:
            if not is_green.iloc[i]: continue
            prev_body = body.iloc[i - 1]
            curr_body = body.iloc[i]
            if curr_body < prev_body * (STRONG_BODY_PCT / 100): continue
            mid = (df_1h["open"].iloc[i - 1] + df_1h["close"].iloc[i - 1]) / 2
            if df_1h["close"].iloc[i] < mid: continue
            lower_wick = df_1h["open"].iloc[i] - df_1h["low"].iloc[i]
            if lower_wick > curr_body * 0.5: continue
            signals.append({"trigger_time": df_1h["datetime"].iloc[i], "dir": "BUY",
                            "level": df_1h["high"].iloc[i], "trigger_high": round(df_1h["high"].iloc[i], 2),
                            "trigger_low": round(df_1h["low"].iloc[i], 2)})
    return signals


def exec_trades(signals, df_5m, sym):
    trades = []
    time_col = df_5m["datetime"].dt.time
    for sig in signals:
        t = sig["trigger_time"]; level = sig["level"]; d = sig["dir"]
        scan = df_5m[df_5m["datetime"] > t]
        if scan.empty: continue
        if d == "BUY":
            b = scan[scan["close"] > level]
            if b.empty: continue
            b0 = b.index[0]; pb = scan.loc[b0 + 1:]
            if pb.empty: continue
            r = (pb["low"] < level) & (pb["close"] > level) & (time_col.loc[pb.index] < CUTOFF_TIME)
            if not r.any(): continue
            ri = r.idxmax(); bar = scan.loc[ri]
            ep, sl = bar["close"], bar["low"]; risk = ep - sl
            if risk <= 0: continue
            tp = ep + 2 * risk; xs = scan.loc[ri + 1:]
            if xs.empty: continue
            sh = xs["low"] <= sl; th = xs["high"] >= tp
            si = sh.idxmax() if sh.any() else None; ti = th.idxmax() if th.any() else None
            if si and (ti is None or si < ti):
                trades.append({"points": sl - ep, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[si, "datetime"],
                               "reason": "SL", "hold_hours": (scan.loc[si, "datetime"] - bar["datetime"]).total_seconds() / 3600, "dir": d, "symbol": sym})
            elif ti:
                trades.append({"points": tp - ep, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[ti, "datetime"],
                               "reason": "TP", "hold_hours": (scan.loc[ti, "datetime"] - bar["datetime"]).total_seconds() / 3600, "dir": d, "symbol": sym})
        else:
            b = scan[scan["close"] < level]
            if b.empty: continue
            b0 = b.index[0]; pb = scan.loc[b0 + 1:]
            if pb.empty: continue
            r = (pb["high"] > level) & (pb["close"] < level) & (time_col.loc[pb.index] < CUTOFF_TIME)
            if not r.any(): continue
            ri = r.idxmax(); bar = scan.loc[ri]
            ep, sl = bar["close"], bar["high"]; risk = sl - ep
            if risk <= 0: continue
            tp = ep - 2 * risk; xs = scan.loc[ri + 1:]
            if xs.empty: continue
            sh = xs["high"] >= sl; th = xs["low"] <= tp
            si = sh.idxmax() if sh.any() else None; ti = th.idxmax() if th.any() else None
            if si and (ti is None or si < ti):
                trades.append({"points": ep - sl, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[si, "datetime"],
                               "reason": "SL", "hold_hours": (scan.loc[si, "datetime"] - bar["datetime"]).total_seconds() / 3600, "dir": d, "symbol": sym})
            elif ti:
                trades.append({"points": ep - tp, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[ti, "datetime"],
                               "reason": "TP", "hold_hours": (scan.loc[ti, "datetime"] - bar["datetime"]).total_seconds() / 3600, "dir": d, "symbol": sym})
    return trades


def plot_equity_curve(df, path):
    d = df.sort_values("exit_time").reset_index(drop=True)
    d["cum"] = d["points"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(d.index, d["cum"], alpha=0.25, color="#2E86AB")
    ax.plot(d.index, d["cum"], color="#2E86AB", lw=1)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Equity Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trade Sequence"); ax.set_ylabel("Cumulative Points")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.annotate(f"Final: {d['cum'].iloc[-1]:+.0f}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#2E86AB", alpha=0.9))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_drawdown(df, path):
    d = df.sort_values("exit_time").reset_index(drop=True)
    d["cum"] = d["points"].cumsum(); d["peak"] = d["cum"].cummax(); d["dd"] = d["cum"] - d["peak"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(d.index, d["dd"], 0, color="#D64045", alpha=0.6)
    ax.set_title("Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (pts)"); ax.set_xlabel("Trade Sequence")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.annotate(f"Max: {d['dd'].min():,.0f}", xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", fontsize=11, fontweight="bold", color="#D64045",
                bbox=dict(boxstyle="round", fc="white", ec="#D64045", alpha=0.9))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_yearly(df, path):
    d = df.copy(); d["exit_time"] = pd.to_datetime(d["exit_time"])
    d["year"] = d["exit_time"].dt.year
    y = d.groupby("year").agg(net=("points","sum"), trades=("points","count"), wins=("points",lambda x:(x>0).sum())).reset_index()
    y["wr"] = y["wins"]/y["trades"]*100
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ECC71" if v>0 else "#E74C3C" for v in y["net"]]
    bars = ax.bar(y["year"].astype(str), y["net"], color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Net Points"); ax.set_xlabel("Year")
    ax.set_title("Yearly P&L", fontsize=13, fontweight="bold")
    for b, v in zip(bars, y["net"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+(15 if v>=0 else -50),
                f"{v:+.0f}", ha="center", fontsize=8, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(y["year"].astype(str), y["wr"], "D-", color="#3498DB", lw=2, ms=6)
    ax2.set_ylabel("Win Rate %", color="#3498DB"); ax2.tick_params(axis="y", colors="#3498DB")
    for _, r in y.iterrows():
        ax2.annotate(f"{r['wr']:.0f}%", (str(r["year"]), r["wr"]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#3498DB")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_pnl_dist(df, path):
    w = df[df["points"]>0]["points"]; l = df[df["points"]<=0]["points"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, color, label in zip(axes, [w, l], ["#2ECC71", "#E74C3C"], ["Wins", "Losses"]):
        ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="darkgreen" if label=="Wins" else "darkred", ls="--",
                   label=f"Mean: {data.mean():+.1f}")
        ax.set_title(f"{label} (n={len(data)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Points"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_duration(df, path):
    w = df[df["points"]>0]["hold_hours"]; l = df[df["points"]<=0]["hold_hours"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, color, label in zip(axes, [w, l], ["#2ECC71", "#E74C3C"], ["Wins", "Losses"]):
        ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="darkgreen" if label=="Wins" else "darkred", ls="--",
                   label=f"Avg: {data.mean():.1f}h")
        ax.set_title(f"{label} Duration (n={len(data)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hours"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_exit_reason(df, path):
    r = df["reason"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"TP": "#2ECC71", "SL": "#E74C3C"}
    ax.pie(r.values, labels=[f"{k}\n({v})" for k,v in r.items()],
           autopct="%1.1f%%", colors=[colors.get(k,"#95A5A6") for k in r.index],
           startangle=90, explode=[0.05]*len(r))
    ax.set_title("Exit Reason", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_dir_pnl(df, path):
    g = df.groupby("dir")["points"].sum()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#2ECC71" if v>0 else "#E74C3C" for v in g.values]
    ax.bar(g.index, g.values, color=colors, alpha=0.8, edgecolor="white", width=0.5)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Net Points"); ax.set_title("P&L by Direction", fontsize=13, fontweight="bold")
    for i, v in enumerate(g.values):
        ax.text(i, v+(10 if v>=0 else -30), f"{v:+.0f}", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def run_backtest(sym):
    h1 = pd.read_csv(f"{DATA_DIR}/{sym}_ONE_HOUR.csv")
    m5 = pd.read_csv(f"{DATA_DIR}/{sym}_FIVE_MINUTE.csv")
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)
    sigs = detect_signals(h1)
    trades = exec_trades(sigs, m5, sym)
    return pd.DataFrame(trades) if trades else pd.DataFrame()


def calc_metrics(df):
    if df.empty: return {}
    t = len(df); w = df[df["points"]>0]; l = df[df["points"]<=0]
    wc = len(w); lc = len(l)
    m = {
        "trades": t, "wins": wc, "losses": lc,
        "win_rate": round(wc/t*100,1) if t else 0,
        "net_pts": round(df["points"].sum(),2),
        "gross_profit": round(w["points"].sum(),2) if wc else 0,
        "gross_loss": round(l["points"].sum(),2) if lc else 0,
        "pf": round(abs(w["points"].sum()/l["points"].sum()),2) if lc and l["points"].sum()!=0 else (999 if wc else 0),
        "avg_win": round(w["points"].mean(),2) if wc else 0,
        "avg_loss": round(l["points"].mean(),2) if lc else 0,
        "max_win": round(w["points"].max(),2) if wc else 0,
        "max_loss": round(l["points"].min(),2) if lc else 0,
        "avg_hold": round(df["hold_hours"].mean(),1),
        "charges": t * CHARGES_PER_ORDER * 2,
        "buy_net": round(df[df["dir"]=="BUY"]["points"].sum(),2) if "BUY" in df["dir"].values else 0,
        "sell_net": round(df[df["dir"]=="SELL"]["points"].sum(),2) if "SELL" in df["dir"].values else 0,
    }
    d = df.sort_values("exit_time").reset_index(drop=True)
    d["cum"] = d["points"].cumsum(); d["peak"] = d["cum"].cummax(); d["dd"] = d["peak"] - d["cum"]
    m["mdd"] = round(d["dd"].max(),2)
    m["mdd_pct"] = round(m["mdd"]/d["peak"].max()*100,1) if d["peak"].max()>0 else 0
    m["sharpe"] = round(df["points"].mean()/df["points"].std()*np.sqrt(t),2) if df["points"].std()>0 else 0
    m["avg_r"] = round(df["points"].mean()/abs(m["avg_loss"]),2) if m["avg_loss"]!=0 else 0
    return m


def gen_report(sym):
    print(f"\n--- {sym} ---")
    df = run_backtest(sym)
    if df.empty:
        print("  No trades!")
        return
    df.to_csv(f"{OUTPUT_DIR}/{sym}_trade_book.csv", index=False)
    m = calc_metrics(df)

    # Plots
    pd_ = f"{OUTPUT_DIR}/plots"; os.makedirs(pd_, exist_ok=True)
    plot_equity_curve(df, f"{pd_}/{sym}_equity.png")
    plot_drawdown(df, f"{pd_}/{sym}_drawdown.png")
    plot_yearly(df, f"{pd_}/{sym}_yearly.png")
    plot_pnl_dist(df, f"{pd_}/{sym}_pnl_dist.png")
    plot_duration(df, f"{pd_}/{sym}_duration.png")
    plot_exit_reason(df, f"{pd_}/{sym}_exit_reason.png")
    plot_dir_pnl(df, f"{pd_}/{sym}_dir_pnl.png")

    # PDF
    pdf = FPDF(); pdf.alias_nb_pages()

    # Page 1 - Title
    pdf.add_page()
    pdf.ln(25)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 14, "Big Candle Reversal Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, f"Backtest Report - {sym}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Period: 2015-01 to 2026-06", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)

    # Strategy box
    pdf.set_fill_color(235, 242, 250)
    pdf.set_draw_color(20, 60, 120)
    pdf.rect(15, pdf.get_y(), 180, 60, style="DF")
    y0 = pdf.get_y() + 4
    pdf.set_xy(20, y0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 6, "Strategy Rules", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(20, pdf.get_y())
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    rules = [
        f"1-Hour: Big Candle (body > {MULTIPLIER}x SMA20) + opposite reversal (body >= {STRONG_BODY_PCT}% of big candle)",
        "Reversal close beyond big candle midpoint + wick <= 50% of reversal body",
        "5-Min: Break trigger high/low -> retest -> enter on retest close (before 2:15 PM)",
        "SL = retest candle extreme | TP = 1:2 R:R | No time exit | Rs 10/order charges",
    ]
    for r in rules:
        pdf.set_xy(20, pdf.get_y())
        pdf.cell(0, 5.5, r, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(y0 + 56)

    # Page 2 - Results
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "Performance Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 60, 120)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    metrics = [
        ("Total Trades", str(m["trades"])),
        ("Winning Trades", str(m["wins"])),
        ("Losing Trades", str(m["losses"])),
        ("Win Rate", f"{m['win_rate']}%"),
        ("Net Points", f"{m['net_pts']:+,.2f}"),
        ("Gross Profit", f"{m['gross_profit']:+,.2f}"),
        ("Gross Loss", f"{m['gross_loss']:+,.2f}"),
        ("Profit Factor", f"{m['pf']}"),
        ("Average Win", f"{m['avg_win']:+,.2f}"),
        ("Average Loss", f"{m['avg_loss']:+,.2f}"),
        ("Max Win", f"{m['max_win']:+,.2f}"),
        ("Max Loss", f"{m['max_loss']:+,.2f}"),
        ("Avg Hold Time", f"{m['avg_hold']} hrs"),
        ("Max Drawdown", f"{m['mdd']:+,.2f} pts ({m['mdd_pct']}%)"),
        ("Sharpe Ratio", f"{m['sharpe']}"),
        ("P&L - BUY Trades", f"{m['buy_net']:+,.2f}"),
        ("P&L - SELL Trades", f"{m['sell_net']:+,.2f}"),
        ("Total Charges (Rs)", f"Rs {m['charges']:,}"),
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    col_x = pdf.w / 2 + 8
    y_start = pdf.get_y()
    for i, (label, val) in enumerate(metrics):
        r = i // 2; c = i % 2
        x = 18 if c == 0 else col_x
        y = y_start + r * 7
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(75, 7, label)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(35, 7, val, new_x="LMARGIN", new_y="NEXT")

    # Page 3 - Equity
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 8, "Equity Curve & Drawdown", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.image(f"{pd_}/{sym}_equity.png", x=10, w=190)
    pdf.ln(2)
    pdf.image(f"{pd_}/{sym}_drawdown.png", x=10, w=190)

    # Page 4 - Yearly + Direction
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 8, "Yearly P&L & Direction Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.image(f"{pd_}/{sym}_yearly.png", x=10, w=190)
    pdf.ln(2)
    pdf.image(f"{pd_}/{sym}_dir_pnl.png", x=60, w=90)

    # Page 5 - Distribution
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 8, "Trade Distribution Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.image(f"{pd_}/{sym}_pnl_dist.png", x=10, w=190)
    pdf.ln(2)
    pdf.image(f"{pd_}/{sym}_duration.png", x=10, w=190)
    pdf.ln(2)
    pdf.image(f"{pd_}/{sym}_exit_reason.png", x=65, w=80)

    path = f"{OUTPUT_DIR}/{sym}_Backtest_Report.pdf"
    pdf.output(path)
    print(f"  Report: {path}")
    print(f"  Trades: {m['trades']} | Net: {m['net_pts']:+.1f} | WR: {m['win_rate']}% | PF: {m['pf']}")


if __name__ == "__main__":
    for s in ["NIFTY50", "SENSEX"]:
        gen_report(s)
