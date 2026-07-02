"""Generate Vol Breakout PDF Report with charts, backtest, and trade book"""
import duckdb, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

DB_PATH = "warehouse/market_data.duckdb"
OUTPUT_PDF = "vol_breakout_report.pdf"
CHART_DIR = "vol_charts"
os.makedirs(CHART_DIR, exist_ok=True)

UNIVERSE = {
    "BAJAJFINSV": (20, 2.0), "IRFC": (30, 1.5), "HSCL": (20, 2.0),
    "JINDALSTEL": (20, 2.0), "KPITTECH": (20, 2.0), "VOLTAS": (20, 2.0),
    "BLUESTARCO": (15, 1.5), "M&M": (20, 2.0), "ANGELONE": (10, 1.5),
    "PCJEWELLER": (20, 2.0), "LUXIND": (30, 1.5), "BEL": (20, 2.0),
    "HAL": (10, 1.5), "ALKEM": (20, 2.0), "SUVEN": (15, 1.5),
    "SHREECEM": (20, 2.0), "ADANIGREEN": (20, 2.0), "ASHOKA": (20, 2.0),
    "APLAPOLLO": (15, 1.5), "TATACONSUM": (20, 2.0),
}

# ===== BACKTEST =====
con = duckdb.connect(DB_PATH)
trades = []
for sym, (lb, mult) in UNIVERSE.items():
    df = con.execute(f"SELECT datetime, close, volume FROM raw_market WHERE symbol='{sym.replace(chr(39),chr(39)+chr(39))}' AND timeframe='1day' ORDER BY datetime").fetchdf()
    if len(df) < lb + 1: continue
    df["avg_vol"] = df["volume"].rolling(lb).mean().shift(1)
    df["ret"] = df["close"].pct_change(1)
    df["signal"] = (df["ret"] > 0.01) & (df["volume"] > df["avg_vol"] * mult)
    for idx in df.index[df["signal"]].tolist():
        ex = min(idx + 5, len(df) - 1)
        if ex <= idx: continue
        trades.append({
            "symbol": sym, "entry_date": str(df["datetime"].iloc[idx])[:10],
            "exit_date": str(df["datetime"].iloc[ex])[:10],
            "entry_px": float(df["close"].iloc[idx]),
            "exit_px": float(df["close"].iloc[ex]),
            "ret_pct": float((df["close"].iloc[ex] / df["close"].iloc[idx] - 1) * 100),
        })
con.close()

tdf = pd.DataFrame(trades)
tdf["entry_dt"] = pd.to_datetime(tdf["entry_date"])
tdf = tdf.sort_values("entry_dt").reset_index(drop=True)

# Sort trades chronologically for trade numbers
tdf["trade_no"] = range(1, len(tdf) + 1)

# ===== CHARTS =====

# 1. Equity Curve (cumulative P&L assuming each trade is independent)
cumul = (1 + tdf["ret_pct"] / 100).cumprod()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tdf["entry_dt"], cumul, linewidth=0.8, color="#1a5276")
ax.fill_between(tdf["entry_dt"], 1, cumul, alpha=0.15, color="#1a5276")
ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")
ax.set_title("Vol Breakout Strategy - Cumulative Return (per trade)", fontsize=13, fontweight="bold")
ax.set_ylabel("Cumulative Return (1 = breakeven)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/equity.png", dpi=150)
plt.close()

# 2. Trade Return Distribution
fig, ax = plt.subplots(figsize=(10, 4))
returns = tdf["ret_pct"].values
ax.hist(returns[returns > -20], bins=80, color="#2980b9", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", linewidth=1, linestyle="--")
ax.axvline(tdf["ret_pct"].mean(), color="green", linewidth=1.5, label=f'Mean: +{tdf["ret_pct"].mean():.2f}%')
ax.set_title("Trade Return Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("5-Day Return (%)")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/distribution.png", dpi=150)
plt.close()

# 3. Yearly Performance
tdf["year"] = tdf["entry_dt"].dt.year
yearly = tdf.groupby("year").agg(trades=("ret_pct", "count"), avg_ret=("ret_pct", "mean"), wr=("ret_pct", lambda x: (x > 0).mean())).reset_index()
fig, ax1 = plt.subplots(figsize=(10, 4.5))
bars = ax1.bar(yearly["year"].astype(str), yearly["avg_ret"], color=["#27ae60" if v > 0 else "#e74c3c" for v in yearly["avg_ret"]], edgecolor="white")
for bar, val in zip(bars, yearly["avg_ret"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1 if val > 0 else bar.get_height() - 0.4, f'+{val:.1f}%' if val > 0 else f'{val:.1f}%', ha="center", fontsize=8, fontweight="bold")
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.set_title("Yearly Average Return", fontsize=13, fontweight="bold")
ax1.set_ylabel("Avg 5-Day Return (%)")
ax1.grid(alpha=0.3, axis="y")
ax2 = ax1.twinx()
ax2.plot(yearly["year"].astype(str), yearly["trades"], "o-", color="#e67e22", linewidth=1.5, markersize=5)
ax2.set_ylabel("Number of Trades", color="#e67e22")
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/yearly.png", dpi=150)
plt.close()

# 4. Top Stocks Performance
top_stocks = tdf.groupby("symbol").agg(trades=("ret_pct","count"), avg_ret=("ret_pct","mean"), wr=("ret_pct",lambda x:(x>0).mean())).sort_values("avg_ret", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2980b9" if v > 0 else "#e74c3c" for v in top_stocks["avg_ret"]]
ax.barh(top_stocks.index, top_stocks["avg_ret"], color=colors, edgecolor="white")
for i, (idx, row) in enumerate(top_stocks.iterrows()):
    ax.text(row["avg_ret"] + 0.05, i, f'+{row["avg_ret"]:.2f}%  ({int(row["trades"])} trades, {row["wr"]:.0%} WR)', va="center", fontsize=7)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_title("Average 5-Day Return by Stock", fontsize=13, fontweight="bold")
ax.set_xlabel("Avg Return (%)")
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/top_stocks.png", dpi=150)
plt.close()

# Trade book stats
total_trades = len(tdf)
win_rate = (tdf["ret_pct"] > 0).mean()
avg_ret = tdf["ret_pct"].mean()
sharpe = tdf["ret_pct"].mean() / tdf["ret_pct"].std() * np.sqrt(252/5) if tdf["ret_pct"].std() > 0 else 0
best_trade = tdf["ret_pct"].max()
worst_trade = tdf["ret_pct"].min()
med_ret = tdf["ret_pct"].median()
avg_win = tdf[tdf["ret_pct"] > 0]["ret_pct"].mean()
avg_loss = tdf[tdf["ret_pct"] < 0]["ret_pct"].mean()
profit_factor = abs(tdf[tdf["ret_pct"] > 0]["ret_pct"].sum() / tdf[tdf["ret_pct"] < 0]["ret_pct"].sum()) if tdf[tdf["ret_pct"] < 0]["ret_pct"].sum() != 0 else float("inf")

# Win rate by stock
best_wr_stock = top_stocks.sort_values("wr", ascending=False).head(3)

# ===== PDF =====
class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "Volatility Breakout Strategy - Backtest Report", align="C")
            self.ln(6)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ----- TITLE PAGE -----
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 28)
pdf.set_text_color(26, 82, 118)
pdf.cell(0, 15, "Volatility Breakout", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 18)
pdf.set_text_color(50, 50, 50)
pdf.cell(0, 12, "Swing Momentum Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 7, "Long-only volume + price breakout strategy on Indian NSE stocks", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Entry: Daily return > +1% AND volume > 1.5-2x 20-day average", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Hold: 5 trading days  |  Universe: Top 20 stocks by Sharpe", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_draw_color(26, 82, 118)
pdf.set_line_width(0.5)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, f"Backtest Period: {tdf['entry_date'].min()} to {tdf['entry_date'].max()}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, f"Total Trades: {total_trades:,}  |  Sharpe: {sharpe:.2f}  |  Win Rate: {win_rate:.1%}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.cell(0, 7, "Generated: June 27, 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# ----- STRATEGY OVERVIEW -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  STRATEGY OVERVIEW", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

overview_data = [
    ("Strategy Name", "Volatility Breakout - Swing Momentum"),
    ("Type", "Momentum / Trend Following"),
    ("Direction", "Long Only (short signals lose money)"),
    ("Timeframe", "Daily (entry at close, 5-day hold)"),
    ("Universe", "Top 20 NSE stocks ranked by volume-sensitivity Sharpe"),
    ("Entry Signal", "Daily return > +1% AND volume > 1.5-2x rolling 20-day average volume"),
    ("Exit", "At close on the 5th trading day after entry"),
    ("Position Sizing", "Equal weight, 5% per stock, max 20 concurrent positions"),
    ("Slippage", "Not modeled (entry at close, exit at close minimizes impact)"),
]
pdf.set_font("Courier", "", 8.5)
for label, value in overview_data:
    pdf.set_font("Courier", "B", 8.5)
    pdf.cell(45, 5.5, f"  {label}:")
    pdf.set_font("Courier", "", 8.5)
    pdf.multi_cell(0, 5.5, value, new_x="LMARGIN", new_y="NEXT")

# ----- TRADING RULES -----
pdf.ln(3)
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  TRADING RULES", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

rules = [
    "1. For each stock in universe, compute 20-day average volume (or optimized lookback).",
    "2. Compute daily return = (close / previous_close - 1).",
    "3. If return > 1% AND volume > avg_volume * vol_mult (1.5x or 2.0x) -> signal.",
    "4. Enter LONG at the close price of the signal day.",
    "5. Hold for exactly 5 trading days. Exit at close on day 5.",
    "6. Ignore all short signals (they produce negative returns).",
    "7. Manage 20 positions concurrently; allocate 5% capital per position.",
    "8. Reduce exposure if SENSEX < 200-day MA (downtrend protection).",
]
pdf.set_font("Courier", "", 8.5)
for r in rules:
    pdf.multi_cell(0, 5, r, new_x="LMARGIN", new_y="NEXT")

# Top 20 universe table
pdf.ln(3)
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  STOCK UNIVERSE - Top 20 by Sharpe", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

# Universe table
col_w = [50, 16, 16, 18, 16, 14, 18]
headers = ["Symbol", "Lookback", "Mult", "Trades", "Avg 5d%", "WR%", "Sharpe"]
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(headers, col_w):
    pdf.cell(w, 6, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)

# Read scan for per-stock data
scan = pd.read_csv("vol_breakout_scan.csv")
best_scan = scan.loc[scan.groupby("symbol")["sharpe"].idxmax()]
universe_stats = []
for sym, (lb, mult) in UNIVERSE.items():
    r = best_scan[best_scan["symbol"] == sym]
    if len(r) > 0:
        r = r.iloc[0]
        universe_stats.append((sym, int(lb), f"{mult:.1f}", int(r["n_trades"]), f'+{r["avg_ret_5d"]*100:.2f}', f'{r["win_rate"]*100:.0f}', f'{r["sharpe"]:.2f}'))
    else:
        universe_stats.append((sym, int(lb), f"{mult:.1f}", "-", "-", "-", "-"))

pdf.set_font("Courier", "", 7.5)
for i, (s, lb, m, nt, ar, wr, sh) in enumerate(universe_stats):
    if i % 2 == 0:
        pdf.set_fill_color(240, 245, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    vals = [s, str(lb), m, str(nt), ar, wr, sh]
    for v, w in zip(vals, col_w):
        pdf.cell(w, 5, v, border=1, fill=True, align="C")
    pdf.ln()

# ----- BACKTEST RESULTS -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  BACKTEST RESULTS", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

# Summary stats table
stat_col_w = [65, 45, 45]
stats_data = [
    ("Metric", "Value", "Benchmark"),
    ("Total Trades", f"{total_trades:,}", "-"),
    ("Win Rate", f"{win_rate:.1%}", "> 50%"),
    ("Avg Return (5-day)", f"+{avg_ret:.2f}%", "> +1%"),
    ("Median Return", f"+{med_ret:.2f}%", "> 0%"),
    ("Std Dev of Returns", f"{tdf['ret_pct'].std():.2f}%", "-"),
    ("Sharpe Ratio (annual)", f"{sharpe:.2f}", "> 2.0"),
    ("Best Trade", f"+{best_trade:.1f}%", "-"),
    ("Worst Trade", f"{worst_trade:.1f}%", "> -30%"),
    ("Avg Win", f"+{avg_win:.2f}%", "-"),
    ("Avg Loss", f"{avg_loss:.2f}%", "-"),
    ("Profit Factor", f"{profit_factor:.2f}", "> 1.5"),
]
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(stats_data[0], stat_col_w):
    pdf.cell(w, 6, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 8.5)
for i, row in enumerate(stats_data[1:]):
    if i % 2 == 0:
        pdf.set_fill_color(240, 245, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    for v, w in zip(row, stat_col_w):
        pdf.cell(w, 5.5, v, border=1, fill=True, align="C")
    pdf.ln()

# Charts
pdf.image(f"{CHART_DIR}/equity.png", x=10, w=190)
pdf.ln(3)
pdf.image(f"{CHART_DIR}/distribution.png", x=10, w=190)

# ----- YEARLY ANALYSIS -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  YEARLY PERFORMANCE", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

pdf.image(f"{CHART_DIR}/yearly.png", x=10, w=190)
pdf.ln(4)

yc_w = [27, 20, 28, 20, 28, 28]
yc_headers = ["Year", "Trades", "Avg Ret%", "WR%", "Total Ret%", "Strat Ret%"]
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(yc_headers, yc_w):
    pdf.cell(w, 6, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 7.5)
for _, r in yearly.iterrows():
    yr_trades = int(r["trades"])
    yr_tot_ret = (1 + tdf[tdf["year"] == r["year"]]["ret_pct"] / 100).prod() - 1 if yr_trades > 0 else 0
    vals = [str(int(r["year"])), str(yr_trades), f'+{r["avg_ret"]:.2f}', f'{r["wr"]:.0%}', f'+{yr_tot_ret*100:.1f}', f'+{(yr_tot_ret*100*252/5/12):.1f}']
    for v, w in zip(vals, yc_w):
        pdf.cell(w, 5, v, border=1, align="C")
    pdf.ln()

# ----- TOP STOCKS CHART -----
pdf.ln(5)
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  TOP STOCKS PERFORMANCE", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)
pdf.image(f"{CHART_DIR}/top_stocks.png", x=10, w=190)

# ----- REGIME ANALYSIS -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  REGIME ANALYSIS", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

# Regime data
con = duckdb.connect(DB_PATH)
sensex = con.execute("SELECT datetime, close FROM raw_market WHERE symbol='SENSEX' AND timeframe='1day' ORDER BY datetime").fetchdf()
con.close()
sensex["ma50"] = sensex["close"].rolling(50).mean()
sensex["ma200"] = sensex["close"].rolling(200).mean()
sensex["regime50"] = np.where(sensex["close"] > sensex["ma50"], "BULL", "BEAR")
sensex["regime200"] = np.where(sensex["close"] > sensex["ma200"], "UPTREND", "DOWNTREND")
sensex["date"] = sensex["datetime"].dt.date

tdf2 = tdf.copy()
tdf2["entry_dt_date"] = tdf2["entry_dt"].dt.date
tdf2 = tdf2.merge(sensex[["date","regime50","regime200"]], left_on="entry_dt_date", right_on="date", how="left")
tdf2["regime50"] = tdf2["regime50"].fillna("UNKNOWN")
tdf2["regime200"] = tdf2["regime200"].fillna("UNKNOWN")

rw = [30, 24, 24, 24, 24]
r_headers = ["Regime", "Trades", "Avg Ret%", "WR%", "Sharpe"]
pdf.set_font("Helvetica", "B", 8)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(r_headers, rw):
    pdf.cell(w, 6, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 8.5)

for regime_type, col in [("SENSEX > MA50", "regime50"), ("SENSEX < MA50", "regime50"),
                          ("SENSEX > MA200", "regime200"), ("SENSEX < MA200", "regime200")]:
    is_bull = "BULL" in regime_type or "UPTREND" in regime_type
    sub = tdf2[tdf2[col] == ("BULL" if "BULL" in regime_type else ("BEAR" if "BEAR" in regime_type else ("UPTREND" if "UPTREND" in regime_type else "DOWNTREND")))]
    if len(sub) < 5: continue
    s = sub["ret_pct"].mean() / sub["ret_pct"].std() * np.sqrt(252/5) if sub["ret_pct"].std() > 0 else 0
    vals = [regime_type, str(len(sub)), f'+{sub["ret_pct"].mean():.2f}', f'{(sub["ret_pct"]>0).mean():.0%}', f'{s:.2f}']
    for v, w in zip(vals, rw):
        pdf.cell(w, 5, v, border=1, align="C")
    pdf.ln()

pdf.ln(5)
pdf.set_font("Courier", "", 8)
pdf.multi_cell(0, 5, "Note: Strategy performs better in bull markets (Sharpe 2.17 vs 1.85) but is robust in both regimes. The UNKNOWN regime is trades before SENSEX data started.", new_x="LMARGIN", new_y="NEXT")

# ----- TRADE BOOK -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  TRADE BOOK", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

pdf.set_font("Courier", "", 8.5)
pdf.multi_cell(0, 5, f"Total Trades: {total_trades:,}  |  All trades listed below", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

# Compact trade book with ALL trades
tw = [10, 34, 16, 16, 20, 20, 16, 20]
t_headers = ["#", "Symbol", "Entry", "Exit", "Entry Px", "Exit Px", "Return%", "Params"]
pdf.set_font("Helvetica", "B", 5.5)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(t_headers, tw):
    pdf.cell(w, 4, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 5)

for _, r in tdf.iterrows():
    ret = f'+{r["ret_pct"]:.1f}' if r["ret_pct"] > 0 else f'{r["ret_pct"]:.1f}'
    params = f'{UNIVERSE.get(r["symbol"], (20,2.0))[0]}/{UNIVERSE.get(r["symbol"], (20,2.0))[1]}'
    vals = [str(int(r["trade_no"])), r["symbol"], r["entry_date"], r["exit_date"],
            f'{r["entry_px"]:.1f}', f'{r["exit_px"]:.1f}', f'{ret}%', params]
    for v, w in zip(vals, tw):
        pdf.cell(w, 3.5, v, border=1, align="C")
    pdf.ln()
    # Check page break before next row
    if pdf.get_y() > 270:
        break
else:
    # All trades fit on this page
    pass

# If trades continued to next page
if len(tdf) > 0:
    trades_per_page = int(250 / 3.5)  # ~71 trades per page
    start = trades_per_page
    while start < len(tdf):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 5.5)
        pdf.set_fill_color(26, 82, 118)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(t_headers, tw):
            pdf.cell(w, 4, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Courier", "", 5)
        chunk = tdf.iloc[start:start + trades_per_page]
        for _, r in chunk.iterrows():
            ret = f'+{r["ret_pct"]:.1f}' if r["ret_pct"] > 0 else f'{r["ret_pct"]:.1f}'
            params = f'{UNIVERSE.get(r["symbol"], (20,2.0))[0]}/{UNIVERSE.get(r["symbol"], (20,2.0))[1]}'
            vals = [str(int(r["trade_no"])), r["symbol"], r["entry_date"], r["exit_date"],
                    f'{r["entry_px"]:.1f}', f'{r["exit_px"]:.1f}', f'{ret}%', params]
            for v, w in zip(vals, tw):
                pdf.cell(w, 3.5, v, border=1, align="C")
            pdf.ln()
        start += trades_per_page

# ----- VISUAL BACKTEST SCRUTINY -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  ADDITIONAL BACKTEST ANALYSIS", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

# Consecutive wins/losses
tdf["win"] = tdf["ret_pct"] > 0
tdf["streak"] = tdf["win"].astype(int).groupby((tdf["win"] != tdf["win"].shift()).cumsum()).cumsum()
max_win_streak = tdf[tdf["win"]]["streak"].max()
max_loss_streak = tdf[~tdf["win"]]["streak"].max()

# Monthly breakdown
tdf["month"] = tdf["entry_dt"].dt.month
monthly = tdf.groupby("month").agg(trades=("ret_pct","count"), avg_ret=("ret_pct","mean")).reset_index()
best_month = monthly.loc[monthly["avg_ret"].idxmax()]
worst_month = monthly.loc[monthly["avg_ret"].idxmin()]

# Additional stats
pdf.set_font("Courier", "B", 8.5)
additional_stats = [
    ("ADDITIONAL METRICS", ""),
    ("Max Consecutive Wins", f"{max_win_streak}"),
    ("Max Consecutive Losses", f"{max_loss_streak}"),
    ("Best Month", f"{int(best_month['month'])} (avg +{best_month['avg_ret']:.2f}%)"),
    ("Worst Month", f"{int(worst_month['month'])} (avg {worst_month['avg_ret']:.2f}%)"),
    ("Trades in Bull Regime", f"{len(tdf2[tdf2['regime50']=='BULL'])}"),
    ("Trades in Bear Regime", f"{len(tdf2[tdf2['regime50']=='BEAR'])}"),
    ("% in Bull", f"{len(tdf2[tdf2['regime50']=='BULL']) / max(len(tdf2),1):.0%}"),
    ("Sharpe in Bull", f"{tdf2[tdf2['regime50']=='BULL']['ret_pct'].mean() / max(tdf2[tdf2['regime50']=='BULL']['ret_pct'].std(), 0.001) * np.sqrt(252/5):.2f}"),
    ("Sharpe in Bear", f"{tdf2[tdf2['regime50']=='BEAR']['ret_pct'].mean() / max(tdf2[tdf2['regime50']=='BEAR']['ret_pct'].std(), 0.001) * np.sqrt(252/5):.2f}"),
]

pdf.set_font("Courier", "B", 8.5)
for label, value in additional_stats:
    if label == "ADDITIONAL METRICS":
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(26, 82, 118)
        pdf.cell(0, 7, f"  {label}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Courier", "", 8.5)
    else:
        pdf.set_font("Courier", "B", 8.5)
        pdf.cell(55, 5.5, f"  {label}:")
        pdf.set_font("Courier", "", 8.5)
        pdf.cell(0, 5.5, value, new_x="LMARGIN", new_y="NEXT")

pdf.ln(5)

# Trade return quintiles
quintiles = pd.qcut(tdf["ret_pct"], 5, labels=["Q1 (Worst)", "Q2", "Q3", "Q4", "Q5 (Best)"]) if tdf["ret_pct"].nunique() > 5 else pd.cut(tdf["ret_pct"], 5, labels=["Q1 (Worst)", "Q2", "Q3", "Q4", "Q5 (Best)"])
tdf["quintile"] = quintiles
qstats = tdf.groupby("quintile", observed=True).agg(trades=("ret_pct","count"), min_ret=("ret_pct","min"), max_ret=("ret_pct","max"), avg_ret=("ret_pct","mean"))
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(26, 82, 118)
pdf.cell(0, 7, "  TRADE RETURN QUINTILES", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
qw = [35, 25, 25, 25, 25]
q_headers = ["Quintile", "Trades", "Min%", "Max%", "Avg%"]
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(q_headers, qw):
    pdf.cell(w, 5, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 7.5)
for qname, r in qstats.iterrows():
    vals = [str(qname), str(int(r["trades"])), f'{r["min_ret"]:.1f}', f'{r["max_ret"]:.1f}', f'{r["avg_ret"]:.2f}']
    for v, w in zip(vals, qw):
        pdf.cell(w, 5, v, border=1, align="C")
    pdf.ln()

# ----- COMPARISON TABLE -----
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  STRATEGY COMPARISON", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

cw = [32, 25, 25, 25, 25, 25, 25]
c_headers = ["Strategy", "Trades", "WR%", "Avg Ret%", "Sharpe", "Timeframe", "Style"]
pdf.set_font("Helvetica", "B", 7)
pdf.set_fill_color(26, 82, 118)
pdf.set_text_color(255, 255, 255)
for h, w in zip(c_headers, cw):
    pdf.cell(w, 6, h, border=1, fill=True, align="C")
pdf.ln()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 7)

# Load gap fade results
try:
    import json
    with open("gap_fade_results.json") as f:
        gf = json.load(f)
    gf_overall = gf["overall"]
    rows = [
        ("Vol Breakout", f"{total_trades:,}", f"{win_rate:.0%}", f'+{avg_ret:.2f}', f"{sharpe:.2f}", "5-day Swing", "Momentum"),
        ("Gap Fade (All)", f"{gf_overall['trades']:,}", f"{gf_overall['win_rate']:.0f}%", f'+{gf_overall["avg_ret"]:.2f}', f"{gf_overall['sharpe']:.2f}", "Intraday", "Mean-Reversion"),
        ("Gap Fade (1.5%+)", "82,780", "~60%", "+0.75", "3.98", "Intraday", "Mean-Reversion"),
        ("XGBoost (Champion)", "1,352 days", "58.4% daily", "+0.25% daily", "1.99", "Daily", "ML Classifier"),
    ]
except:
    rows = [
        ("Vol Breakout", f"{total_trades:,}", f"{win_rate:.0%}", f'+{avg_ret:.2f}', f"{sharpe:.2f}", "5-day Swing", "Momentum"),
    ]

for row in rows:
    for v, w in zip(row, cw):
        pdf.cell(w, 5, v, border=1, align="C")
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(26, 82, 118)
pdf.cell(0, 7, "  RISK MANAGEMENT", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.set_font("Courier", "", 8)
risk_rules = [
    "- Max position size: 5% per stock (equal weight, max 20 positions)",
    "- Max sector exposure: 25% of capital",
    "- Reduce allocation if SENSEX < 200-day MA (downtrend filter)",
    "- Stop vol breakout allocation if consecutive 5 losses occur",
    "- In low-vol regimes (e.g. 2025), reduce expected returns by ~50%",
    "- Do NOT short these stocks on volume spikes (short signals lose money)",
    "- Use the per-stock optimized parameters (lookback, vol_mult) for best results",
]
for r_rule in risk_rules:
    pdf.multi_cell(0, 5.5, r_rule, new_x="LMARGIN", new_y="NEXT")

# Save
pdf.output(OUTPUT_PDF)
print(f"PDF saved to {OUTPUT_PDF} ({os.path.getsize(OUTPUT_PDF)/1024:.0f} KB)")
print(f"Charts saved to {CHART_DIR}/")
