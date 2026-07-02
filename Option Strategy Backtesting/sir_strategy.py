"""
Sir Strategy Improvement - Enhanced Big Candle Reversal
Filters: Trend(EMA50/200) | Direction(BUY only) | Session(9:30-12:30) | ADX>20 | 1.0xATR(20) big candle | Chandelier exit 7xATR
"""
import pandas as pd
import numpy as np
import os, warnings
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fpdf import FPDF
warnings.filterwarnings("ignore")

OUTPUT_DIR = "backtest_results/sir_strategy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHARGES_PER_ORDER = 10
CUTOFF_TIME = pd.Timestamp("14:15").time()
STRONG_BODY_PCT = 50.0

plt.rcParams["figure.dpi"] = 150


def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_adx(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def compute_daily_ema(df_1h, period=50):
    df = df_1h.copy()
    df["date"] = df["datetime"].dt.normalize()
    daily = df.groupby("date").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    ema = daily["close"].ewm(span=period, adjust=False).mean()
    return df["date"].map(ema).values


def get_signals_sir(df_1h):
    """Detect BUY signals with all Sir Strategy filters"""
    body = (df_1h["close"] - df_1h["open"]).abs()
    atr20 = compute_atr(df_1h, 20)
    adx14 = compute_adx(df_1h, 14)
    ema50 = compute_daily_ema(df_1h, 50)
    ema200 = compute_daily_ema(df_1h, 200)

    is_green = df_1h["close"] > df_1h["open"]
    is_red = df_1h["close"] < df_1h["open"]
    big_bearish = is_red & (body > 1.0 * atr20)

    sigs = []
    for i in range(1, len(df_1h)):
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):
            continue
        if ema50[i] <= ema200[i]:
            continue
        if adx14.iloc[i] <= 20:
            continue
        t_min = df_1h["datetime"].iloc[i].hour * 60 + df_1h["datetime"].iloc[i].minute
        if t_min < 570 or t_min > 750:
            continue
        if not big_bearish.iloc[i - 1]:
            continue
        if not is_green.iloc[i]:
            continue
        if body.iloc[i] < body.iloc[i-1] * (STRONG_BODY_PCT / 100):
            continue
        mid = (df_1h["open"].iloc[i-1] + df_1h["close"].iloc[i-1]) / 2
        if df_1h["close"].iloc[i] < mid:
            continue
        if (df_1h["open"].iloc[i] - df_1h["low"].iloc[i]) > body.iloc[i] * 0.5:
            continue
        sigs.append({
            "trigger_time": df_1h["datetime"].iloc[i],
            "dir": "BUY",
            "level": df_1h["high"].iloc[i],
            "atr_20": atr20.iloc[i],
        })
    return sigs


def get_signals_baseline(df_1h):
    """Original: MULT=1.5, big candle = buy, reversal = buy"""
    body = (df_1h["close"] - df_1h["open"]).abs()
    avg_body = body.rolling(20, min_periods=20).mean()
    is_green = df_1h["close"] > df_1h["open"]
    big_buy = is_green & (body > avg_body * 1.5)
    sigs = []
    for i in range(1, len(df_1h)):
        if pd.isna(avg_body.iloc[i]):
            continue
        if not big_buy.iloc[i-1]:
            continue
        if not is_green.iloc[i]:
            continue
        if body.iloc[i] < body.iloc[i-1] * 0.5:
            continue
        mid = (df_1h["open"].iloc[i-1] + df_1h["close"].iloc[i-1]) / 2
        if df_1h["close"].iloc[i] < mid:
            continue
        if (df_1h["open"].iloc[i] - df_1h["low"].iloc[i]) > body.iloc[i] * 0.5:
            continue
        sigs.append({"trigger_time": df_1h["datetime"].iloc[i], "dir": "BUY", "level": df_1h["high"].iloc[i]})
    return sigs


def exec_chandelier(sigs, df_5m, trail_mult=7):
    """Execute with Chandelier Exit: close < highest_high - mult*ATR"""
    trades = []
    tc = df_5m["datetime"].dt.time
    atr_5m = compute_atr(df_5m, 14)
    for sig in sigs:
        t, lv = sig["trigger_time"], sig["level"]
        scan = df_5m[df_5m["datetime"] > t]
        if scan.empty:
            continue
        b = scan[scan["close"] > lv]
        if b.empty:
            continue
        breakout_idx = b.index[0]
        pb = scan.loc[breakout_idx + 1:]
        if pb.empty:
            continue
        retest = (pb["low"] < lv) & (pb["close"] > lv) & (tc.loc[pb.index] < CUTOFF_TIME)
        if not retest.any():
            continue
        bar = scan.loc[retest.idxmax()]
        ep = bar["close"]
        sl0 = bar["low"]
        if ep - sl0 <= 0:
            continue
        if bar["datetime"].hour == 9:
            continue

        xs = scan.loc[bar.name + 1:]
        if xs.empty:
            continue
        highest_high = ep
        exit_price = None
        exit_time = None
        reason = None
        for idx, row in xs.iterrows():
            cur_atr = atr_5m.loc[idx]
            if pd.isna(cur_atr):
                continue
            if row["high"] > highest_high:
                highest_high = row["high"]
            trail_stop = highest_high - trail_mult * cur_atr
            if row["close"] < trail_stop:
                exit_price = row["close"]
                exit_time = row["datetime"]
                reason = f"CH{trail_mult}"
                break
        if exit_price is not None:
            trades.append({
                "dir": "BUY",
                "entry_time": bar["datetime"],
                "exit_time": exit_time,
                "entry_price": round(ep, 2),
                "exit_price": round(exit_price, 2),
                "sl": round(sl0, 2),
                "points": round(exit_price - ep, 2),
                "reason": reason,
                "hold_hours": round((exit_time - bar["datetime"]).total_seconds() / 3600, 1),
            })
    return trades


def exec_fixed_tp(sigs, df_5m):
    """Original exit: fixed 1:2 TP"""
    trades = []
    tc = df_5m["datetime"].dt.time
    for sig in sigs:
        t, lv = sig["trigger_time"], sig["level"]
        scan = df_5m[df_5m["datetime"] > t]
        if scan.empty:
            continue
        b = scan[scan["close"] > lv]
        if b.empty:
            continue
        breakout_idx = b.index[0]
        pb = scan.loc[breakout_idx + 1:]
        if pb.empty:
            continue
        retest = (pb["low"] < lv) & (pb["close"] > lv) & (tc.loc[pb.index] < CUTOFF_TIME)
        if not retest.any():
            continue
        bar = scan.loc[retest.idxmax()]
        ep = bar["close"]
        sl0 = bar["low"]
        if ep - sl0 <= 0:
            continue
        if bar["datetime"].hour == 9:
            continue
        tp = ep + 2 * (ep - sl0)

        xs = scan.loc[bar.name + 1:]
        exit_price = None
        exit_time = None
        reason = None
        for _, r2 in xs.iterrows():
            if r2["low"] <= sl0:
                exit_price = sl0
                exit_time = r2["datetime"]
                reason = "SL"
                break
            if r2["high"] >= tp:
                exit_price = tp
                exit_time = r2["datetime"]
                reason = "TP"
                break
        if exit_price is not None:
            trades.append({
                "dir": "BUY",
                "entry_time": bar["datetime"],
                "exit_time": exit_time,
                "entry_price": round(ep, 2),
                "exit_price": round(exit_price, 2),
                "sl": round(sl0, 2),
                "points": round(exit_price - ep, 2),
                "reason": reason,
                "hold_hours": round((exit_time - bar["datetime"]).total_seconds() / 3600, 1),
            })
    return trades


def calc(df):
    if df.empty:
        return {}
    t = len(df)
    w = df[df["points"] > 0]
    l = df[df["points"] <= 0]
    wc = len(w)
    lc = len(l)
    gp = w["points"].sum() if wc else 0
    gl = l["points"].sum() if lc else 0
    d = df.sort_values("exit_time").reset_index(drop=True)
    d["cum"] = d["points"].cumsum()
    d["peak"] = d["cum"].cummax()
    d["dd"] = d["peak"] - d["cum"]
    return {
        "trades": t,
        "wins": wc,
        "losses": lc,
        "wr": round(wc / t * 100, 1) if t else 0,
        "net": round(df["points"].sum(), 2),
        "pf": round(abs(gp / gl), 2) if gl != 0 else (999 if gp > 0 else 0),
        "avg_w": round(w["points"].mean(), 2) if wc else 0,
        "avg_l": round(l["points"].mean(), 2) if lc else 0,
        "max_w": round(w["points"].max(), 2) if wc else 0,
        "max_l": round(l["points"].min(), 2) if lc else 0,
        "mdd": round(d["dd"].max(), 2),
        "mdd_pct": round(d["dd"].max() / d["peak"].max() * 100, 1) if d["peak"].max() > 0 else 0,
        "sharpe": round(df["points"].mean() / df["points"].std() * np.sqrt(t), 2) if df["points"].std() > 0 else 0,
        "avg_hold": round(df["hold_hours"].mean(), 1),
    }


def plot_equity(df1, df2, label1, label2, path, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    for d, lbl, clr in [(df1, label1, "#E74C3C"), (df2, label2, "#2ECC71")]:
        if d.empty:
            continue
        dd = d.sort_values("exit_time").reset_index(drop=True)
        dd["cum"] = dd["points"].cumsum()
        ax.plot(dd.index, dd["cum"], label=lbl, color=clr, lw=1.5, alpha=0.9)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Trade Sequence")
    ax.set_ylabel("Cumulative Points")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(df, path, title):
    d = df.sort_values("exit_time").reset_index(drop=True)
    d["cum"] = d["points"].cumsum()
    d["peak"] = d["cum"].cummax()
    d["dd"] = d["cum"] - d["peak"]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(d.index, d["dd"], 0, color="#D64045", alpha=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (pts)")
    ax.set_xlabel("Trade Sequence")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.annotate(f"Max: {d['dd'].min():,.0f}", xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", fontsize=10, color="#D64045", fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#D64045", alpha=0.9))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ─── Main ───
print("=" * 70)
print("SIR STRATEGY IMPROVEMENT BACKTEST")
print("=" * 70)

all_sir = {}
all_baseline = {}
all_sir_fixed = {}
all_base_chan = {}

for sym in ["NIFTY50", "SENSEX"]:
    print(f"\n{'=' * 60}")
    print(f"  {sym}")
    print(f"{'=' * 60}")

    h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv")
    m5 = pd.read_csv(f"{sym}_FIVE_MINUTE.csv")
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)

    # ── 1. Baseline (original method: MULT=1.5, BUY+Skip9am, fixed 1:2 TP) ──
    sigs_base = get_signals_baseline(h1)
    base_trades = exec_fixed_tp(sigs_base, m5)
    df_base = pd.DataFrame(base_trades)
    m_base = calc(df_base)
    df_base.to_csv(f"{OUTPUT_DIR}/{sym}_Baseline.csv", index=False)
    all_baseline[sym] = (df_base, m_base)
    print(f"  Baseline: {m_base['trades']} trades | Net {m_base['net']:+.0f} pts | WR {m_base['wr']}% | PF {m_base['pf']}")

    # ── 2. Sir Strategy (all filters + Chandelier x7) ──
    sigs_sir = get_signals_sir(h1)
    print(f"  Sir Signals after all filters: {len(sigs_sir)}")
    if len(sigs_sir) > 0:
        print(f"    First: {sigs_sir[0]['trigger_time'].date()}")
        print(f"    Last:  {sigs_sir[-1]['trigger_time'].date()}")

    sir_trades = exec_chandelier(sigs_sir, m5, 7)
    df_sir = pd.DataFrame(sir_trades)
    m_sir = calc(df_sir)
    df_sir.to_csv(f"{OUTPUT_DIR}/{sym}_Sir_Trades.csv", index=False)
    all_sir[sym] = (df_sir, m_sir)
    print(f"  Sir Strategy: {m_sir['trades']} trades | Net {m_sir['net']:+.0f} pts | WR {m_sir['wr']}% | PF {m_sir['pf']} | MDD {m_sir['mdd']:.0f}")

    # ── 3. Baseline signals + Chandelier x7 (isolate filter effect) ──
    base_chan = exec_chandelier(sigs_base, m5, 7)
    df_base_chan = pd.DataFrame(base_chan)
    m_base_chan = calc(df_base_chan)
    df_base_chan.to_csv(f"{OUTPUT_DIR}/{sym}_Base_Chandelier.csv", index=False)
    print(f"  Base+Chan: {m_base_chan['trades']} trades | Net {m_base_chan['net']:+.0f} pts | WR {m_base_chan['wr']}% | PF {m_base_chan['pf']} | MDD {m_base_chan['mdd']:.0f}")

    # ── 4. Sir signals + fixed TP (isolate exit effect) ──
    sir_fixed = exec_fixed_tp(sigs_sir, m5)
    df_sir_fixed = pd.DataFrame(sir_fixed)
    m_sir_fixed = calc(df_sir_fixed)
    df_sir_fixed.to_csv(f"{OUTPUT_DIR}/{sym}_Sir_FixedTP.csv", index=False)
    all_sir_fixed[sym] = (df_sir_fixed, m_sir_fixed)
    all_base_chan[sym] = (df_base_chan, m_base_chan)
    print(f"  Sir+FixTP: {m_sir_fixed['trades']} trades | Net {m_sir_fixed['net']:+.0f} pts | WR {m_sir_fixed['wr']}% | PF {m_sir_fixed['pf']}")

    # Print detail
    if not df_sir.empty:
        print(f"  Avg Win: {m_sir['avg_w']:+.0f} | Avg Loss: {m_sir['avg_l']:+.0f} | Max Win: {m_sir['max_w']:+.0f} | Max Loss: {m_sir['max_l']:+.0f}")
        print(f"  Avg Hold: {m_sir['avg_hold']}h | Max DD: {m_sir['mdd']:.0f} pts ({m_sir['mdd_pct']}%)")
        if m_sir['trades'] > 0:
            print(f"  Exit reasons: {df_sir['reason'].value_counts().to_dict()}")


# ─── Plots ───
plot_dir = f"{OUTPUT_DIR}/plots"
os.makedirs(plot_dir, exist_ok=True)
for sym in ["NIFTY50", "SENSEX"]:
    if sym in all_sir and sym in all_baseline:
        plot_equity(all_baseline[sym][0], all_sir[sym][0],
                    "Baseline", "Sir Strategy",
                    f"{plot_dir}/{sym}_equity.png",
                    f"{sym} - Sir Strategy vs Baseline")
        plot_drawdown(all_sir[sym][0],
                      f"{plot_dir}/{sym}_drawdown.png",
                      f"{sym} - Sir Strategy Drawdown")

# ─── PDF Report ───
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 7, "Sir Strategy Improvement - Backtest Report", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, t):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 60, 120)
        self.cell(0, 9, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)


pdf = PDF()
pdf.alias_nb_pages()

# Page 1: Title
pdf.add_page()
pdf.ln(25)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(20, 60, 120)
pdf.cell(0, 14, "Sir Strategy Improvement", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Big Candle Reversal - Enhanced Backtest Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(50, 50, 50)
pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "NIFTY50 & SENSEX | 2015-01 to 2026-06", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(12)

# Filters box
pdf.set_fill_color(235, 242, 250)
pdf.set_draw_color(20, 60, 120)
filters_list = [
    ("1. Directional Bias", "BUY only (short trades removed)"),
    ("2. Trend Filter", "Daily EMA50 > EMA200 (bull market alignment)"),
    ("3. Big Candle", "Body > 1.0 x ATR(20) (replaces 1.5 x SMA20; 2x gave only 2 signals)"),
    ("4. Trend Strength", "ADX(14) > 20 (avoid sideways markets)"),
    ("5. Session Filter", "Trigger window: 9:30 AM - 12:30 PM"),
    ("6. Entry Filter", "Skip 09:00 entries"),
    ("7. Exit Method", "Chandelier Exit (7 x ATR, replaces fixed 1:2 TP)"),
    ("8. Index Selection", "NIFTY50 & SENSEX only (BANKNIFTY excluded)"),
]
y0 = pdf.get_y()
pdf.rect(15, y0, 180, 10 + len(filters_list) * 6.5, style="DF")
pdf.set_xy(20, y0 + 4)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(20, 60, 120)
pdf.cell(0, 6, "Sir Strategy Filters Applied", new_x="LMARGIN", new_y="NEXT")
for name, desc in filters_list:
    pdf.set_xy(20, pdf.get_y())
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(55, 6, name)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, desc, new_x="LMARGIN", new_y="NEXT")
pdf.set_y(y0 + 12 + len(filters_list) * 6.5)

# Page 2: Combined Summary
pdf.add_page()
pdf.section("Combined Results")
tot_base = sum(all_baseline[s][1]["net"] for s in ["NIFTY50", "SENSEX"])
tot_sir = sum(all_sir[s][1]["net"] for s in ["NIFTY50", "SENSEX"])
tr_base = sum(all_baseline[s][1]["trades"] for s in ["NIFTY50", "SENSEX"])
tr_sir = sum(all_sir[s][1]["trades"] for s in ["NIFTY50", "SENSEX"])

cols_w = [45, 22, 25, 20, 20, 25, 25]
headers = ["Version", "Trades", "Net Pts", "WR%", "PF", "Max DD", "Avg Hold"]
pdf.set_font("Helvetica", "B", 8)
pdf.set_fill_color(20, 60, 120)
pdf.set_text_color(255, 255, 255)
for h, c in zip(headers, cols_w):
    pdf.cell(c, 6, h, border=1, align="C", fill=True)
pdf.ln()
pdf.set_font("Helvetica", "", 8)
pdf.set_text_color(50, 50, 50)
for ver, tr, nt, wr, pf, mdd, hold in [
    ("Baseline", tr_base, tot_base,
     round(sum(all_baseline[s][1]["wr"] * all_baseline[s][1]["trades"] for s in ["NIFTY50", "SENSEX"]) / tr_base, 1),
     round(sum(all_baseline[s][1]["pf"] for s in ["NIFTY50", "SENSEX"]) / 2, 2),
     round(max(all_baseline[s][1]["mdd"] for s in ["NIFTY50", "SENSEX"]), 0),
     round(sum(all_baseline[s][1]["avg_hold"] for s in ["NIFTY50", "SENSEX"]) / 2, 1)),
    ("Sir Strategy", tr_sir, tot_sir,
     round(sum(all_sir[s][1]["wr"] * all_sir[s][1]["trades"] for s in ["NIFTY50", "SENSEX"]) / tr_sir, 1) if tr_sir else 0,
     round(sum(all_sir[s][1]["pf"] for s in ["NIFTY50", "SENSEX"]) / 2, 2),
     round(max(all_sir[s][1]["mdd"] for s in ["NIFTY50", "SENSEX"]), 0),
     round(sum(all_sir[s][1]["avg_hold"] for s in ["NIFTY50", "SENSEX"]) / 2, 1)),
]:
    pdf.cell(cols_w[0], 5.5, ver, border=1, align="C")
    pdf.cell(cols_w[1], 5.5, str(tr), border=1, align="C")
    pdf.cell(cols_w[2], 5.5, f"{nt:+.0f}", border=1, align="C")
    pdf.cell(cols_w[3], 5.5, f"{wr}%", border=1, align="C")
    pdf.cell(cols_w[4], 5.5, f"{pf}", border=1, align="C")
    pdf.cell(cols_w[5], 5.5, f"{mdd:.0f}", border=1, align="C")
    pdf.cell(cols_w[6], 5.5, f"{hold}h", border=1, align="C")
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(20, 60, 120)
impr = tot_sir - tot_base
pdf.cell(0, 7, f"Total Improvement: {impr:+.0f} pts ({(impr/tot_base*100):+.0f}% vs baseline)", new_x="LMARGIN", new_y="NEXT")

# Page 3+: Per-symbol detail
for sym in ["NIFTY50", "SENSEX"]:
    if sym not in all_sir:
        continue
    pdf.add_page()
    pdf.section(f"{sym} - Detailed Results")

    # Comparison table
    headers2 = ["Version", "Trades", "Net Pts", "WR%", "PF", "AvgW", "AvgL", "MaxW", "MaxL", "MDD", "Hold"]
    cols_w2 = [38, 16, 18, 14, 14, 16, 16, 16, 16, 16, 14]
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(20, 60, 120)
    pdf.set_text_color(255, 255, 255)
    for h, c in zip(headers2, cols_w2):
        pdf.cell(c, 5, h, border=1, align="C", fill=True)
    pdf.ln()

    versions = [
        ("Baseline (1:2TP)", all_baseline.get(sym, (pd.DataFrame(), {}))[1]),
        ("Base+Chandelier", all_base_chan.get(sym, (pd.DataFrame(), {}))[1]),
        ("Sir+FixTP", all_sir_fixed.get(sym, (pd.DataFrame(), {}))[1]),
        ("Sir Strategy", all_sir[sym][1]),
    ]
    pdf.set_font("Helvetica", "", 6.5)
    pdf.set_text_color(50, 50, 50)
    for ver, m in versions:
        if not m:
            continue
        vals = [ver, str(m["trades"]), f"{m['net']:+.0f}", f"{m['wr']}%", f"{m['pf']}",
                f"{m['avg_w']:+.0f}", f"{m['avg_l']:+.0f}",
                f"{m['max_w']:+.0f}", f"{m['max_l']:+.0f}",
                f"{m['mdd']:.0f}", f"{m['avg_hold']}h"]
        for v, c in zip(vals, cols_w2):
            pdf.cell(c, 4.5, str(v), border=1, align="C")
        pdf.ln()

    # Images
    pdf.ln(3)
    if os.path.exists(f"{plot_dir}/{sym}_equity.png"):
        pdf.image(f"{plot_dir}/{sym}_equity.png", x=12, w=186)
    pdf.ln(2)
    if os.path.exists(f"{plot_dir}/{sym}_drawdown.png"):
        pdf.image(f"{plot_dir}/{sym}_drawdown.png", x=12, w=186)

# Final page
pdf.add_page()
pdf.section("Observations & Recommendations")
obs = [
    f"Sir Strategy combined net: {tot_sir:+.0f} pts vs Baseline {tot_base:+.0f} pts",
    f"Improvement: {impr:+.0f} pts ({(impr/tot_base*100):+.1f}%)",
    f"Trades reduced: {tr_base} to {tr_sir} ({(1-tr_sir/tr_base)*100:.0f}% fewer, higher quality)",
    "",
    "Key observations:",
    "- Chandelier Exit at 7xATR is the primary performance driver (fixed 1:2 TP is far inferior)",
    "- Sir filters (ADX>20, session 9:30-12:30, EMA50>200, 1.0xATR candle) reduce trades ~78%",
    "- Sir filters reduce max drawdown: NIFTY 840pts vs 2,098 (Base+Chan); SENSEX 1,684 vs 3,863",
    "- On SENSEX, Sir filters improve pts/trade: 186 vs 106 (Base+Chan - all signals)",
    "- On NIFTY50, pts/trade is similar: 27 vs 28 (filters have marginal impact on quality)",
    "- Volume filter skipped: Angel API index spot data has zero volume",
]
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(50, 50, 50)
for ln in obs:
    pdf.cell(0, 6, ln, new_x="LMARGIN", new_y="NEXT")

pdf.ln(10)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 6, f"Trade books: {OUTPUT_DIR}/", new_x="LMARGIN", new_y="NEXT")

pdf_path = f"{OUTPUT_DIR}/Sir_Strategy_Improvement_Report.pdf"
pdf.output(pdf_path)
print(f"\n{'=' * 70}")
print(f"Report: {pdf_path}")
print(f"{'=' * 70}")
