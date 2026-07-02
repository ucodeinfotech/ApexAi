"""
Generate comprehensive PDF report for Big Candle Reversal Strategy Backtest
"""
import pandas as pd
import numpy as np
import os, time, warnings
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from fpdf import FPDF
warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "."
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MULTIPLIER = 1.5
STRONG_BODY_PCT = 50.0
LENGTH = 20
CUTOFF_TIME = pd.Timestamp("14:15").time()
CHARGES_PER_ORDER = 10

INDICES = ["NIFTY50", "BANKNIFTY", "SENSEX"]

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


# ─── Signal Detection ────────────────────────────────────────────────────
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
            if not is_red.iloc[i]:
                continue
            prev_body = body.iloc[i - 1]
            curr_body = body.iloc[i]
            if curr_body < prev_body * (STRONG_BODY_PCT / 100):
                continue
            mid = (df_1h["open"].iloc[i - 1] + df_1h["close"].iloc[i - 1]) / 2
            if df_1h["close"].iloc[i] > mid:
                continue
            upper_wick = df_1h["high"].iloc[i] - df_1h["open"].iloc[i]
            if upper_wick > curr_body * 0.5:
                continue
            signals.append({
                "trigger_time": df_1h["datetime"].iloc[i],
                "date": df_1h["datetime"].iloc[i].date(),
                "dir": "SELL", "level": df_1h["low"].iloc[i],
                "trigger_high": round(df_1h["high"].iloc[i], 2),
                "trigger_low": round(df_1h["low"].iloc[i], 2),
            })
        elif big_sell.iloc[i - 1]:
            if not is_green.iloc[i]:
                continue
            prev_body = body.iloc[i - 1]
            curr_body = body.iloc[i]
            if curr_body < prev_body * (STRONG_BODY_PCT / 100):
                continue
            mid = (df_1h["open"].iloc[i - 1] + df_1h["close"].iloc[i - 1]) / 2
            if df_1h["close"].iloc[i] < mid:
                continue
            lower_wick = df_1h["open"].iloc[i] - df_1h["low"].iloc[i]
            if lower_wick > curr_body * 0.5:
                continue
            signals.append({
                "trigger_time": df_1h["datetime"].iloc[i],
                "date": df_1h["datetime"].iloc[i].date(),
                "dir": "BUY", "level": df_1h["high"].iloc[i],
                "trigger_high": round(df_1h["high"].iloc[i], 2),
                "trigger_low": round(df_1h["low"].iloc[i], 2),
            })
    return signals


def exec_trades(signals, df_5m, sym):
    trades = []
    time_col = df_5m["datetime"].dt.time
    for sig in signals:
        t = sig["trigger_time"]
        level = sig["level"]
        direction = sig["dir"]
        m = df_5m["datetime"] > t
        scan = df_5m[m]
        if scan.empty:
            continue
        if direction == "BUY":
            b_hit = scan["close"] > level
            if not b_hit.any():
                continue
            b0 = b_hit.idxmax()
            pb = scan.loc[b0 + 1:]
            if pb.empty:
                continue
            r_hit = (pb["low"] < level) & (pb["close"] > level) & (time_col.loc[pb.index] < CUTOFF_TIME)
            if not r_hit.any():
                continue
            r_idx = r_hit.idxmax()
            bar = scan.loc[r_idx]
            ep, sl = bar["close"], bar["low"]
            risk = ep - sl
            if risk <= 0:
                continue
            tp = ep + 2 * risk
            xs = scan.loc[r_idx + 1:]
            if xs.empty:
                continue
            sl_hit = xs["low"] <= sl
            tp_hit = xs["high"] >= tp
            sl_idx = sl_hit.idxmax() if sl_hit.any() else None
            tp_idx = tp_hit.idxmax() if tp_hit.any() else None
            if sl_idx is not None and (tp_idx is None or sl_idx < tp_idx):
                hold = (scan.loc[sl_idx, "datetime"] - bar["datetime"]).total_seconds() / 3600
                trades.append({**sig, "points": sl - ep, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[sl_idx, "datetime"],
                               "reason": "SL", "hold_hours": hold, "symbol": sym})
            elif tp_idx is not None:
                hold = (scan.loc[tp_idx, "datetime"] - bar["datetime"]).total_seconds() / 3600
                trades.append({**sig, "points": tp - ep, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[tp_idx, "datetime"],
                               "reason": "TP", "hold_hours": hold, "symbol": sym})
        else:
            b_hit = scan["close"] < level
            if not b_hit.any():
                continue
            b0 = b_hit.idxmax()
            pb = scan.loc[b0 + 1:]
            if pb.empty:
                continue
            r_hit = (pb["high"] > level) & (pb["close"] < level) & (time_col.loc[pb.index] < CUTOFF_TIME)
            if not r_hit.any():
                continue
            r_idx = r_hit.idxmax()
            bar = scan.loc[r_idx]
            ep, sl = bar["close"], bar["high"]
            risk = sl - ep
            if risk <= 0:
                continue
            tp = ep - 2 * risk
            xs = scan.loc[r_idx + 1:]
            if xs.empty:
                continue
            sl_hit = xs["high"] >= sl
            tp_hit = xs["low"] <= tp
            sl_idx = sl_hit.idxmax() if sl_hit.any() else None
            tp_idx = tp_hit.idxmax() if tp_hit.any() else None
            if sl_idx is not None and (tp_idx is None or sl_idx < tp_idx):
                hold = (scan.loc[sl_idx, "datetime"] - bar["datetime"]).total_seconds() / 3600
                trades.append({**sig, "points": ep - sl, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[sl_idx, "datetime"],
                               "reason": "SL", "hold_hours": hold, "symbol": sym})
            elif tp_idx is not None:
                hold = (scan.loc[tp_idx, "datetime"] - bar["datetime"]).total_seconds() / 3600
                trades.append({**sig, "points": ep - tp, "entry_price": ep, "sl": sl, "tp": tp,
                               "entry_time": bar["datetime"], "exit_time": scan.loc[tp_idx, "datetime"],
                               "reason": "TP", "hold_hours": hold, "symbol": sym})
    return trades


def compute_metrics(df):
    if df.empty:
        return {}
    total = len(df)
    wins = df[df["points"] > 0]
    losses = df[df["points"] <= 0]
    wc = len(wins); lc = len(losses)
    wr = wc / total * 100 if total else 0
    gp = wins["points"].sum() if wc else 0
    gl = losses["points"].sum() if lc else 0
    net = df["points"].sum()
    pf = abs(gp / gl) if gl != 0 else (999 if gp > 0 else 0)
    avg_w = wins["points"].mean() if wc else 0
    avg_l = losses["points"].mean() if lc else 0
    max_w = wins["points"].max() if wc else 0
    max_l = losses["points"].min() if lc else 0
    avg_hold = df["hold_hours"].mean()
    total_charges = int(total * CHARGES_PER_ORDER * 2)
    df_s = df.sort_values("exit_time").reset_index(drop=True)
    df_s["cum"] = df_s["points"].cumsum()
    df_s["peak"] = df_s["cum"].cummax()
    df_s["dd"] = df_s["peak"] - df_s["cum"]
    mdd = df_s["dd"].max()
    peak_val = df_s["peak"].max()
    mdd_pct = mdd / peak_val * 100 if peak_val > 0 else 0
    sharpe = (df["points"].mean() / df["points"].std() * np.sqrt(total)) if df["points"].std() > 0 else 0
    avg_r_multiple = df["points"].mean() / abs(avg_l) if avg_l != 0 else 0
    return {
        "total": total, "wins": wc, "losses": lc,
        "win_rate": round(wr, 1), "net_points": round(net, 2),
        "gross_profit": round(gp, 2), "gross_loss": round(gl, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_w, 2), "avg_loss": round(avg_l, 2),
        "max_win": round(max_w, 2), "max_loss": round(max_l, 2),
        "avg_hold_hours": round(avg_hold, 1),
        "max_dd": round(mdd, 2), "max_dd_pct": round(mdd_pct, 1),
        "sharpe": round(sharpe, 2),
        "avg_r": round(avg_r_multiple, 2),
        "total_charges_rs": total_charges,
    }


# ─── Plotting Functions ─────────────────────────────────────────────────
def plot_equity_curve(trades_df, output_path):
    df = trades_df.sort_values("exit_time").reset_index(drop=True)
    df["cumulative"] = df["points"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df.index, df["cumulative"], alpha=0.3, color="#2E86AB")
    ax.plot(df.index, df["cumulative"], color="#2E86AB", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Equity Curve (Cumulative Points)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade Sequence", fontsize=10)
    ax.set_ylabel("Cumulative Points", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    final = df["cumulative"].iloc[-1]
    ax.annotate(f"Final: {final:+.0f} pts", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#2E86AB", alpha=0.9))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(trades_df, output_path):
    df = trades_df.sort_values("exit_time").reset_index(drop=True)
    df["cumulative"] = df["points"].cumsum()
    df["peak"] = df["cumulative"].cummax()
    df["dd"] = df["cumulative"] - df["peak"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(df.index, df["dd"], 0, color="#D64045", alpha=0.6)
    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade Sequence", fontsize=10)
    ax.set_ylabel("Drawdown (pts)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    mdd_val = df["dd"].min()
    ax.annotate(f"Max DD: {mdd_val:,.0f} pts", xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", fontsize=11, fontweight="bold", color="#D64045",
                bbox=dict(boxstyle="round", fc="white", ec="#D64045", alpha=0.9))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pnl_distribution(trades_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    wins = trades_df[trades_df["points"] > 0]["points"]
    losses = trades_df[trades_df["points"] <= 0]["points"]
    ax = axes[0]
    ax.hist(wins, bins=30, color="#2ECC71", alpha=0.7, edgecolor="white")
    ax.axvline(wins.mean(), color="darkgreen", linestyle="--", label=f"Mean: {wins.mean():+.1f}")
    ax.set_title(f"Winning Trades Distribution (n={len(wins)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Points"); ax.set_ylabel("Frequency"); ax.legend()
    ax = axes[1]
    ax.hist(losses, bins=30, color="#E74C3C", alpha=0.7, edgecolor="white")
    ax.axvline(losses.mean(), color="darkred", linestyle="--", label=f"Mean: {losses.mean():+.1f}")
    ax.set_title(f"Losing Trades Distribution (n={len(losses)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Points"); ax.set_ylabel("Frequency"); ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(trades_df, output_path):
    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    df["month"] = df["exit_time"].dt.month
    monthly = df.groupby(["year", "month"])["points"].sum().unstack()
    monthly = monthly.fillna(0)
    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly) * 0.35)))
    sns.heatmap(monthly, cmap="RdYlGn", center=0, annot=True, fmt=".0f",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Points"})
    ax.set_title("Monthly P&L Heatmap", fontsize=14, fontweight="bold")
    ax.set_ylabel("Year"); ax.set_xlabel("Month")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticklabels(month_names, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_bars(trades_df, output_path):
    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    yearly = df.groupby("year").agg(
        net=("points", "sum"),
        trades=("points", "count"),
        wins=("points", lambda x: (x > 0).sum())
    ).reset_index()
    yearly["wr"] = yearly["wins"] / yearly["trades"] * 100
    fig, ax1 = plt.subplots(figsize=(12, 5))
    colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in yearly["net"]]
    bars = ax1.bar(yearly["year"].astype(str), yearly["net"], color=colors, alpha=0.8, edgecolor="white")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Net Points", fontsize=10, color="#2C3E50")
    ax1.set_xlabel("Year", fontsize=10)
    for bar, val in zip(bars, yearly["net"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (20 if val >= 0 else -80),
                f"{val:+.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(yearly["year"].astype(str), yearly["wr"], "D-", color="#3498DB", linewidth=2, markersize=6)
    ax2.set_ylabel("Win Rate (%)", fontsize=10, color="#3498DB")
    ax2.tick_params(axis="y", colors="#3498DB")
    for _, row in yearly.iterrows():
        ax2.annotate(f"{row['wr']:.0f}%", (str(row['year']), row['wr']),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#3498DB")
    ax1.set_title("Yearly P&L with Win Rate Overlay", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_trade_duration(trades_df, output_path):
    df = trades_df.copy()
    wins = df[df["points"] > 0]["hold_hours"]
    losses = df[df["points"] <= 0]["hold_hours"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(wins, bins=30, color="#2ECC71", alpha=0.7, edgecolor="white")
    ax.axvline(wins.mean(), color="darkgreen", linestyle="--", label=f"Avg: {wins.mean():.1f}h")
    ax.set_title(f"Win Duration (n={len(wins)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hold Hours"); ax.set_ylabel("Frequency"); ax.legend()
    ax = axes[1]
    ax.hist(losses, bins=30, color="#E74C3C", alpha=0.7, edgecolor="white")
    ax.axvline(losses.mean(), color="darkred", linestyle="--", label=f"Avg: {losses.mean():.1f}h")
    ax.set_title(f"Loss Duration (n={len(losses)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hold Hours"); ax.set_ylabel("Frequency"); ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_exit_reason_pie(trades_df, output_path):
    reasons = trades_df["reason"].value_counts()
    colors_pie = {"TP": "#2ECC71", "SL": "#E74C3C"}
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        reasons.values, labels=reasons.index, autopct="%1.1f%%",
        colors=[colors_pie.get(r, "#95A5A6") for r in reasons.index],
        startangle=90, explode=[0.05] * len(reasons),
        textprops={"fontsize": 12, "fontweight": "bold"}
    )
    ax.set_title("Exit Reason Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_direction_comparison(trades_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = trades_df.groupby(["symbol", "dir"])["points"].sum().unstack()
    x = np.arange(len(groups))
    w = 0.3
    ax.bar(x - w / 2, groups.get("BUY", 0), w, label="BUY", color="#2ECC71", alpha=0.85)
    ax.bar(x + w / 2, groups.get("SELL", 0), w, label="SELL", color="#E74C3C", alpha=0.85)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups.index, fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Points", fontsize=10)
    ax.set_title("P&L by Direction per Index", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    for i, idx in enumerate(groups.index):
        for j, d in enumerate(["BUY", "SELL"]):
            val = groups.get(d, pd.Series(0)).get(idx, 0)
            if val != 0:
                ax.text(i + (-1 if j == 0 else 1) * w / 2, val + (10 if val >= 0 else -30),
                        f"{val:+.0f}", ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ─── PDF Report Generator ───────────────────────────────────────────────
class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Big Candle Reversal Strategy - Backtest Report", align="L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(44, 62, 80)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, txt):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, txt)
        self.ln(2)

    def metric_row(self, label, value, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        px = self.l_margin + indent * 4
        self.cell(px - self.l_margin, 0, "")
        self.cell(90, 7, label)
        self.set_font("Helvetica", "B", 10)
        self.cell(40, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def build_report():
    print("Running backtest...")
    data = {}
    for sym in INDICES:
        h1 = pd.read_csv(f"{DATA_DIR}/{sym}_ONE_HOUR.csv")
        m5 = pd.read_csv(f"{DATA_DIR}/{sym}_FIVE_MINUTE.csv")
        h1["datetime"] = pd.to_datetime(h1["datetime"])
        m5["datetime"] = pd.to_datetime(m5["datetime"])
        h1 = h1.sort_values("datetime").reset_index(drop=True)
        m5 = m5.sort_values("datetime").reset_index(drop=True)
        data[sym] = (h1, m5)

    all_trades = []
    index_metrics = {}
    for sym in INDICES:
        df_1h, df_5m = data[sym]
        sigs = detect_signals(df_1h)
        trades = exec_trades(sigs, df_5m, sym)
        if trades:
            all_trades.extend(trades)
            index_metrics[sym] = compute_metrics(pd.DataFrame(trades))

    all_df = pd.DataFrame(all_trades)
    all_df.to_csv(f"{OUTPUT_DIR}/full_trade_book.csv", index=False)
    combined_metrics = compute_metrics(all_df)

    # Generate plots
    print("Generating plots...")
    plot_dir = f"{OUTPUT_DIR}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plot_equity_curve(all_df, f"{plot_dir}/equity_curve.png")
    plot_drawdown(all_df, f"{plot_dir}/drawdown.png")
    plot_pnl_distribution(all_df, f"{plot_dir}/pnl_distribution.png")
    plot_monthly_heatmap(all_df, f"{plot_dir}/monthly_heatmap.png")
    plot_yearly_bars(all_df, f"{plot_dir}/yearly_bars.png")
    plot_trade_duration(all_df, f"{plot_dir}/trade_duration.png")
    plot_exit_reason_pie(all_df, f"{plot_dir}/exit_reason.png")
    plot_direction_comparison(all_df, f"{plot_dir}/direction_comparison.png")

    # Build PDF
    print("Building PDF report...")
    pdf = PDFReport()
    pdf.alias_nb_pages()

    # ── Page 1: Title ──
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "Big Candle Reversal Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Backtest Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Data: NIFTY50, BANKNIFTY, SENSEX (2015-2026)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Strategy Overview", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    overview = (
        "The Big Candle Reversal Pattern identifies a large directional candle (Big Candle) "
        "followed by an opposite-colored reversal candle on the 1-hour timeframe. "
        "When the reversal candle closes beyond the midpoint of the Big Candle's body with "
        "a clean wick structure, a trigger is generated. "
        "Entry is executed on the 5-minute timeframe: price must first break out beyond the "
        "trigger candle's high/low, then retest that level before entering on the retest candle's close. "
        "Stop-loss is placed at the retest candle's high/low, with a 1:2 risk-reward take-profit."
    )
    pdf.multi_cell(0, 6, overview)
    pdf.ln(10)

    # ── Page 2: Parameters ──
    pdf.add_page()
    pdf.section_title("Strategy Parameters")
    params = [
        ("Signal Timeframe", "1-Hour"),
        ("Entry/Exit Timeframe", "5-Minute"),
        ("Big Candle Body Multiplier", f"{MULTIPLIER}x SMA({LENGTH})"),
        ("Reversal Body % of Big Candle", f"{STRONG_BODY_PCT}%"),
        ("Retracement Confirmation", "50% (midpoint of big candle body)"),
        ("Wick Filter", "Wick <= 50% of reversal body"),
        ("Entry Cutoff Time", "2:15 PM"),
        ("Risk-Reward", "1:2 (fixed)"),
        ("Charges", f"Rs {CHARGES_PER_ORDER}/order (Rs {CHARGES_PER_ORDER*2} round-trip)"),
        ("Data Period", "2015-01 to 2026-06"),
    ]
    for label, val in params:
        pdf.metric_row(label, val)
    pdf.ln(10)
    pdf.section_title("Signal Logic")
    logic = (
        "1. Compute average body size (SMA 20) of 1-hour candles.\n"
        "2. A 'Big Candle' is any candle with body > 1.5x the SMA(20).\n"
        "3. The next candle (reversal) must be opposite-colored, body >= 50% of Big Candle's body,\n"
        "   close beyond Big Candle's body midpoint, and have a wick <= 50% of its own body.\n"
        "4. On 5-min: price breaks trigger candle's high/low, retests it, then enter on retest close.\n"
        "5. SL at retest candle extreme, TP at 1:2 R:R. No time-based exit."
    )
    pdf.body_text(logic)

    # ── Page 3: Combined Results ──
    pdf.add_page()
    pdf.section_title("Combined Results (All Indices)")
    m = combined_metrics
    metrics_list = [
        ("Total Trades", m.get("total", "-")),
        ("Winning Trades", m.get("wins", "-")),
        ("Losing Trades", m.get("losses", "-")),
        ("Win Rate", f"{m.get('win_rate', '-')}%"),
        ("Net Points", f"{m.get('net_points', '-'):+,.2f}"),
        ("Gross Profit", f"{m.get('gross_profit', '-'):+,.2f}"),
        ("Gross Loss", f"{m.get('gross_loss', '-'):+,.2f}"),
        ("Profit Factor", m.get("profit_factor", "-")),
        ("Average Win", f"{m.get('avg_win', '-'):+,.2f} pts"),
        ("Average Loss", f"{m.get('avg_loss', '-'):+,.2f} pts"),
        ("Max Win", f"{m.get('max_win', '-'):+,.2f} pts"),
        ("Max Loss", f"{m.get('max_loss', '-'):+,.2f} pts"),
        ("Avg Hold Time", f"{m.get('avg_hold_hours', '-')} hours"),
        ("Max Drawdown", f"{m.get('max_dd', '-'):+,.2f} pts ({m.get('max_dd_pct', '-')}%)"),
        ("Sharpe Ratio", m.get("sharpe", "-")),
        ("Total Charges (Rs)", f"Rs {m.get('total_charges_rs', '-'):,}"),
    ]
    half = len(metrics_list) // 2 + 1
    col2_y = pdf.get_y()
    for i, (label, val) in enumerate(metrics_list):
        if i < half:
            pdf.metric_row(label, val)
        else:
            if i == half:
                pdf.set_xy(pdf.w / 2 + 5, col2_y)
            pdf.set_x(pdf.w / 2 + 5)
            pdf.cell(80, 7, label)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(40, 7, str(val), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Note: Points are gross index points. Charges are in INR (not deducted from points).",
             new_x="LMARGIN", new_y="NEXT")

    # Per-index table
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "Per-Index Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    col_w = [28, 20, 28, 20, 22, 22, 24, 24]
    headers = ["Index", "Trades", "Net Pts", "Win%", "PF", "Avg W", "Avg L", "Max DD"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    for h, cw in zip(headers, col_w):
        pdf.cell(cw, 7, h, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    for sym in INDICES:
        m2 = index_metrics.get(sym, {})
        row_data = [
            sym, str(m2.get("total", 0)), f"{m2.get('net_points', 0):+.0f}",
            f"{m2.get('win_rate', 0):.1f}%", f"{m2.get('profit_factor', 0):.2f}",
            f"{m2.get('avg_win', 0):+.0f}", f"{m2.get('avg_loss', 0):+.0f}",
            f"{m2.get('max_dd', 0):+.0f}"
        ]
        pdf.set_fill_color(245, 245, 245) if headers.index("Index") % 2 == 0 else None
        for val, cw in zip(row_data, col_w):
            pdf.cell(cw, 6, val, border=1, align="C")
        pdf.ln()

    # ── Pages 4+: Plots ──
    plots = [
        ("Equity Curve", "equity_curve.png"),
        ("Drawdown", "drawdown.png"),
        ("Win/Loss Distribution", "pnl_distribution.png"),
        ("Yearly P&L", "yearly_bars.png"),
        ("Monthly P&L Heatmap", "monthly_heatmap.png"),
        ("Trade Duration Analysis", "trade_duration.png"),
        ("Exit Reason (SL vs TP)", "exit_reason.png"),
        ("P&L by Direction per Index", "direction_comparison.png"),
    ]
    for title, fname in plots:
        pdf.add_page()
        pdf.section_title(title)
        path = f"{plot_dir}/{fname}"
        if os.path.exists(path):
            img_w = 180
            pdf.image(path, x=pdf.w / 2 - img_w / 2, w=img_w)

    # Save PDF
    pdf_path = f"{OUTPUT_DIR}/Big_Candle_Reversal_Backtest_Report.pdf"
    pdf.output(pdf_path)
    print(f"\n{'='*60}")
    print(f"Report saved: {pdf_path}")
    print(f"Trade book:   {OUTPUT_DIR}/full_trade_book.csv")
    print(f"Plots:        {plot_dir}/")
    print(f"{'='*60}")
    print(f"\nCombined Results:")
    print(f"  Trades: {m['total']} | Net: {m['net_points']:+.1f} pts | WR: {m['win_rate']}%")
    print(f"  PF: {m['profit_factor']} | Avg Win: {m['avg_win']:+.1f} | Avg Loss: {m['avg_loss']:+.1f}")
    print(f"  Sharpe: {m['sharpe']} | Max DD: {m['max_dd']:+.1f} pts ({m['max_dd_pct']}%)")


if __name__ == "__main__":
    build_report()
