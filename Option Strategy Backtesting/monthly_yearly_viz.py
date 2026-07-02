"""
Monthly heatmaps + yearly report for all 5 strategies with proper visualization
"""
import pandas as pd
import numpy as np
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

CH_VALS = [25, 30, 35, 40, 45, 50, 55, 60]
MON_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

plt.rcParams.update({
    "font.family": "Segoe UI, Arial, sans-serif",
    "font.size": 9,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

def compute_atr(m5):
    hl = m5["high"] - m5["low"]
    hpc = abs(m5["high"] - m5["close"].shift(1))
    lpc = abs(m5["low"] - m5["close"].shift(1))
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=14, min_periods=14, adjust=False).mean()

def compute_atr20(h1):
    tr = pd.concat([
        h1["high"] - h1["low"],
        abs(h1["high"] - h1["close"].shift(1)),
        abs(h1["low"] - h1["close"].shift(1))
    ], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=20).mean()

def compute_adx14(h1):
    tr = pd.concat([
        h1["high"] - h1["low"],
        abs(h1["high"] - h1["close"].shift(1)),
        abs(h1["low"] - h1["close"].shift(1))
    ], axis=1).max(axis=1)
    up = h1["high"] - h1["high"].shift(1)
    down = h1["low"].shift(1) - h1["low"]
    pdm = ((up > down) & (up > 0)) * up
    ndm = ((down > up) & (down > 0)) * down
    atr14 = tr.rolling(14, min_periods=14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr14)
    ndi = 100 * (ndm.rolling(14).mean() / atr14)
    dx = 100 * (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan))
    return dx.rolling(14).mean()

def compute_daily_ema(h1, period=50):
    df_ = h1.copy()
    df_["date"] = h1["datetime"].dt.normalize()
    daily = df_.groupby("date").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return df_["date"].map(daily["close"].ewm(span=period, adjust=False).mean()).values

print("Loading data...")
DATA = {}
for sym in ["NIFTY50", "SENSEX"]:
    h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
    m5 = pd.read_csv(f"{sym}_FIVE_MINUTE.csv", parse_dates=["datetime"])
    for df in [h1, m5]:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
    atr5 = compute_atr(m5)
    DATA[sym] = {
        "h1": h1, "m5": m5,
        "m5_epoch": m5["datetime"].astype("int64").values,
        "m5_hi": m5["high"].values, "m5_lo": m5["low"].values,
        "m5_cl": m5["close"].values, "m5_atr": atr5.values,
        "tc": pd.Series(m5["datetime"]).dt.time.values,
    }

CUT = pd.Timestamp("14:15").time()

def find_retest(sym, t, lv):
    d = DATA[sym]
    m5_epoch = d["m5_epoch"]
    m5_cl = d["m5_cl"]
    m5_lo = d["m5_lo"]
    tc = d["tc"]
    t_ep = t.asm8.view("int64")
    idx = np.searchsorted(m5_epoch, t_ep, side="right")
    if idx >= len(m5_cl):
        return None
    b = idx
    while b < len(m5_cl) and m5_cl[b] <= lv:
        b += 1
    if b >= len(m5_cl) - 1:
        return None
    r = b + 1
    while r < len(m5_cl):
        if m5_lo[r] < lv and m5_cl[r] > lv and tc[r] < CUT:
            break
        r += 1
    if r >= len(m5_cl):
        return None
    ep = m5_cl[r]
    sl = m5_lo[r]
    if ep - sl <= 0:
        return None
    return (r, ep, sl)

def compute_ch_exits(sym, r, ep):
    d = DATA[sym]
    m5_cl = d["m5_cl"]
    m5_hi = d["m5_hi"]
    m5_atr = d["m5_atr"]
    pnls = {}
    for cv in CH_VALS:
        he = ep
        for j in range(r, len(m5_cl)):
            ca = m5_atr[j]
            if pd.isna(ca):
                continue
            if m5_hi[j] > he:
                he = m5_hi[j]
            if m5_cl[j] < he - cv * ca:
                pnls[cv] = round(m5_cl[j] - ep, 1)
                break
    return pnls

# ====== ENTRY FUNCTIONS ======
def sigs_engulf_raw(sym):
    h1 = DATA[sym]["h1"]
    b = (h1["close"] - h1["open"]).abs()
    g = h1["close"] > h1["open"]
    r = h1["close"] < h1["open"]
    out = []
    for i in range(1, len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:
            continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]:
            continue
        if b.iloc[i] < b.iloc[i-1] * 0.5 or h1["datetime"].iloc[i].hour == 9:
            continue
        out.append({
            "ts": h1["datetime"].iloc[i],
            "lv": h1["high"].iloc[i],
            "yr": h1["datetime"].iloc[i].year,
            "mo": h1["datetime"].iloc[i].month,
            "sym": sym
        })
    return out

def sigs_engulf_filt(sym):
    h1 = DATA[sym]["h1"]
    b = (h1["close"] - h1["open"]).abs()
    g = h1["close"] > h1["open"]
    r = h1["close"] < h1["open"]
    atr20 = compute_atr20(h1)
    adx14 = compute_adx14(h1)
    ema50 = compute_daily_ema(h1, 50)
    ema200 = compute_daily_ema(h1, 200)
    out = []
    for i in range(1, len(h1)):
        if not r.iloc[i-1] or not g.iloc[i]:
            continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]:
            continue
        if b.iloc[i] < b.iloc[i-1] * 0.5:
            continue
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):
            continue
        if ema50[i] <= ema200[i] or adx14.iloc[i] <= 20:
            continue
        t_min = h1["datetime"].iloc[i].hour * 60 + h1["datetime"].iloc[i].minute
        if t_min < 570 or t_min > 750 or h1["datetime"].iloc[i].hour == 9:
            continue
        out.append({
            "ts": h1["datetime"].iloc[i],
            "lv": h1["high"].iloc[i],
            "yr": h1["datetime"].iloc[i].year,
            "mo": h1["datetime"].iloc[i].month,
            "sym": sym
        })
    return out

def sigs_big_candle(sym):
    h1 = DATA[sym]["h1"]
    b = (h1["close"] - h1["open"]).abs()
    ab = b.rolling(20, min_periods=20).mean()
    g = h1["close"] > h1["open"]
    r = h1["close"] < h1["open"]
    out = []
    for i in range(1, len(h1)):
        if pd.isna(ab.iloc[i]) or not r.iloc[i-1] or not g.iloc[i]:
            continue
        if b.iloc[i-1] <= ab.iloc[i-1] * 1.5 or b.iloc[i] < b.iloc[i-1] * 0.5:
            continue
        mid = (h1["open"].iloc[i-1] + h1["close"].iloc[i-1]) / 2
        if h1["close"].iloc[i] < mid or (h1["open"].iloc[i] - h1["low"].iloc[i]) > b.iloc[i] * 0.5:
            continue
        out.append({
            "ts": h1["datetime"].iloc[i],
            "lv": h1["high"].iloc[i],
            "yr": h1["datetime"].iloc[i].year,
            "mo": h1["datetime"].iloc[i].month,
            "sym": sym
        })
    return out

def sigs_sir(sym):
    h1 = DATA[sym]["h1"]
    b = (h1["close"] - h1["open"]).abs()
    g = h1["close"] > h1["open"]
    atr20 = compute_atr20(h1)
    adx14 = compute_adx14(h1)
    ema50 = compute_daily_ema(h1, 50)
    ema200 = compute_daily_ema(h1, 200)
    out = []
    for i in range(1, len(h1)):
        if pd.isna(atr20.iloc[i]) or pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):
            continue
        if not g.iloc[i] or ema50[i] <= ema200[i] or adx14.iloc[i] <= 20:
            continue
        t_min = h1["datetime"].iloc[i].hour * 60 + h1["datetime"].iloc[i].minute
        if t_min < 570 or t_min > 750:
            continue
        if not (h1["close"].iloc[i-1] < h1["open"].iloc[i-1]):
            continue
        if b.iloc[i-1] <= 1.0 * atr20.iloc[i] or b.iloc[i] < b.iloc[i-1] * 0.5:
            continue
        mid = (h1["open"].iloc[i-1] + h1["close"].iloc[i-1]) / 2
        if h1["close"].iloc[i] < mid or (h1["open"].iloc[i] - h1["low"].iloc[i]) > b.iloc[i] * 0.5:
            continue
        out.append({
            "ts": h1["datetime"].iloc[i],
            "lv": h1["high"].iloc[i],
            "yr": h1["datetime"].iloc[i].year,
            "mo": h1["datetime"].iloc[i].month,
            "sym": sym
        })
    return out

def sigs_comb_or(sym):
    h1 = DATA[sym]["h1"]
    b = (h1["close"] - h1["open"]).abs()
    g = h1["close"] > h1["open"]
    atr20 = compute_atr20(h1)
    adx14 = compute_adx14(h1)
    ema50 = compute_daily_ema(h1, 50)
    ema200 = compute_daily_ema(h1, 200)
    out = []
    for i in range(1, len(h1)):
        if not (g.iloc[i] and h1["close"].iloc[i-1] < h1["open"].iloc[i-1]):
            continue
        bc = False
        if not pd.isna(atr20.iloc[i]) and b.iloc[i-1] > 1.0 * atr20.iloc[i] and b.iloc[i] >= b.iloc[i-1] * 0.5:
            mid = (h1["open"].iloc[i-1] + h1["close"].iloc[i-1]) / 2
            if h1["close"].iloc[i] >= mid and (h1["open"].iloc[i] - h1["low"].iloc[i]) <= b.iloc[i] * 0.5:
                bc = True
        en = (h1["open"].iloc[i] <= h1["close"].iloc[i-1] and
              h1["close"].iloc[i] >= h1["open"].iloc[i-1] and
              b.iloc[i] >= b.iloc[i-1] * 0.5)
        if not (bc or en):
            continue
        if pd.isna(adx14.iloc[i]) or pd.isna(ema50[i]) or pd.isna(ema200[i]):
            continue
        if ema50[i] <= ema200[i] or adx14.iloc[i] <= 20:
            continue
        t_min = h1["datetime"].iloc[i].hour * 60 + h1["datetime"].iloc[i].minute
        if t_min < 570 or t_min > 750:
            continue
        out.append({
            "ts": h1["datetime"].iloc[i],
            "lv": h1["high"].iloc[i],
            "yr": h1["datetime"].iloc[i].year,
            "mo": h1["datetime"].iloc[i].month,
            "sym": sym
        })
    return out

strategies = [
    ("Engulf_Raw", "Pure bullish engulfing", sigs_engulf_raw),
    ("Engulf_Filt", "Engulf + EMA50>200 + ADX>20", sigs_engulf_filt),
    ("BigCandle", "Big candle reversal", sigs_big_candle),
    ("Sir", "Strong impulse reversal", sigs_sir),
    ("Comb_OR", "Engulfing OR big candle", sigs_comb_or),
]

print("Building trade sets...")
TRADE_SETS = {}
for sname, sdesc, sfunc in strategies:
    print(f"  {sname}...", end=" ", flush=True)
    rows = []
    for sym in ["NIFTY50", "SENSEX"]:
        for sig in sfunc(sym):
            ret = find_retest(sym, sig["ts"], sig["lv"])
            if ret is None:
                continue
            r, ep, sl = ret
            pnls = compute_ch_exits(sym, r, ep)
            if 55 not in pnls:
                continue
            t = {
                "sym": sym, "yr": sig["yr"], "mo": sig["mo"],
                "ts": sig["ts"], "pts55": pnls[55]
            }
            rows.append(t)
    TRADE_SETS[sname] = pd.DataFrame(rows).fillna(0)
    print(f"{len(TRADE_SETS[sname])} trades")

def metrics(pnl):
    pnl = np.array(pnl)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        return {"n": 0, "wr": 0, "aw": 0, "al": 0, "wl": 0, "pf": 0, "net": 0, "mdd": 0}
    n = len(pnl)
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wr = len(w) / n
    aw = w.mean() if len(w) > 0 else 0
    al = l.mean() if len(l) > 0 else 0
    wl_r = aw / abs(al) if al != 0 else 999
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else 999
    net = pnl.sum()
    cum = np.cumsum(pnl)
    mx = np.maximum.accumulate(cum)
    dd = mx - cum
    mdd = dd.max()
    return {"n": n, "wr": wr, "net": net, "aw": aw, "al": al, "wl": wl_r, "pf": pf, "mdd": mdd}

# ====== BUILD MONTHLY / YEARLY AGGREGATES ======
years = list(range(2015, 2027))

ALL_STRAT_MONTHLY = {}
ALL_STRAT_YEARLY = {}

for sname in [s[0] for s in strategies]:
    df = TRADE_SETS[sname]
    pnl_col = "pts55"
    mo_data = np.full((12,), np.nan)
    mo_wr = np.full((12,), np.nan)
    mo_n = np.full((12,), 0)
    for m in range(1, 13):
        sub = df[df["mo"] == m][pnl_col].values
        if len(sub) > 0:
            mo_data[m-1] = sub.sum()
            mo_wr[m-1] = (sub > 0).sum() / len(sub)
            mo_n[m-1] = len(sub)
    ALL_STRAT_MONTHLY[sname] = {"net": mo_data, "wr": mo_wr, "n": mo_n}

    yr_data = {}
    for yr in years:
        sub = df[df["yr"] == yr][pnl_col].values
        if len(sub) > 0:
            m = metrics(sub)
            yr_data[yr] = m
    ALL_STRAT_YEARLY[sname] = yr_data

# ====== BUILD COMPARISON YEARLY (ALL 5 STRATEGIES) ======
comp_yr = {yr: {} for yr in years}
for sname in [s[0] for s in strategies]:
    for yr in years:
        if yr in ALL_STRAT_YEARLY[sname]:
            comp_yr[yr][sname] = ALL_STRAT_YEARLY[sname][yr]["net"]
        else:
            comp_yr[yr][sname] = 0

NIFTY_MONTHLY = np.array([3120, 4100, 5103, 6370, 6568, 3934, 6238, -1006, -520, 21, 3945, 5158])
SENSEX_MONTHLY = np.array([1613, 1038, 4592, 76, 3853, 3122, 3168, 5815, -3984, -608, 2630, -3225])

OUTPUT_PDF = os.path.join(os.path.dirname(__file__) or ".", "Monthly_Yearly_Visualization.pdf")

with PdfPages(OUTPUT_PDF) as pdf:
    # ====== PAGE 1: MONTHLY NET HEATMAP (all 5 strats) ======
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes_flat = axes.flatten()
    for idx, (sname, sdesc, _) in enumerate(strategies):
        ax = axes_flat[idx]
        md = ALL_STRAT_MONTHLY[sname]
        net_data = md["net"].reshape(1, 12)
        vmax = max(abs(np.nanmax(net_data)), abs(np.nanmin(net_data))) or 1
        cmap = sns.diverging_palette(10, 130, s=85, l=45, as_cmap=True)
        sns.heatmap(
            net_data,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt="+,.0f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Net Pts"},
            xticklabels=MON_NAMES,
            yticklabels=False,
        )
        ax.set_title(f"{sname}: Monthly Net Pts (CH55)", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=45)
    axes_flat[4].axis("off")
    axes_flat[5].axis("off")
    fig.suptitle("MONTHLY NET RETURN HEATMAPS - All 5 Strategies (CH55, 2015-2026)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 2: MONTHLY WIN RATE HEATMAP ======
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes_flat = axes.flatten()
    for idx, (sname, sdesc, _) in enumerate(strategies):
        ax = axes_flat[idx]
        md = ALL_STRAT_MONTHLY[sname]
        wr_data = md["wr"].reshape(1, 12)
        sns.heatmap(
            wr_data,
            ax=ax,
            cmap=sns.color_palette("RdYlGn", 20),
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".0%",
            linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Win Rate"},
            xticklabels=MON_NAMES,
            yticklabels=False,
        )
        ax.set_title(f"{sname}: Monthly Win Rate (CH55)", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=45)
    axes_flat[4].axis("off")
    axes_flat[5].axis("off")
    fig.suptitle("MONTHLY WIN RATE HEATMAPS - All 5 Strategies (CH55, 2015-2026)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 3: COMBINED MONTHLY (NIFTY + SENSEX) ======
    fig, ax = plt.subplots(figsize=(12, 6))
    all_net = np.array([ALL_STRAT_MONTHLY[s[0]]["net"] for s in strategies])
    x = np.arange(12)
    w = 0.15
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (sname, _, _) in enumerate(strategies):
        bars = ax.bar(x + i*w, all_net[i], w, label=sname, color=colors[i], alpha=0.85)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(MON_NAMES)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Monthly Net Return Comparison - All Strategies (CH55)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Net Points")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 4: YEARLY BAR CHART (ALL 5 STRATEGIES) ======
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(years))
    w = 0.15
    for i, (sname, _, _) in enumerate(strategies):
        vals = [comp_yr[yr].get(sname, 0) for yr in years]
        bars = ax.bar(x + i*w, vals, w, label=sname, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            if abs(v) > 20000:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (5000 if v >= 0 else -15000),
                        f"{v/1000:.0f}k", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=6, rotation=90)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(years)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Yearly Net Return - All Strategies (CH55, 2015-2026)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Net Points")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 5: YEARLY CUMULATIVE LINE CHART ======
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (sname, _, _) in enumerate(strategies):
        cum = 0
        cvals = []
        for yr in years:
            net = comp_yr[yr].get(sname, 0)
            cum += net
            cvals.append(cum)
        ax.plot(years, cvals, "o-", label=sname, color=colors[i], linewidth=2, markersize=5)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Cumulative Net Return Over Years - All Strategies (CH55)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Net Points")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 6: TRADE COUNT PER MONTH ======
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes_flat = axes.flatten()
    for idx, (sname, sdesc, _) in enumerate(strategies):
        ax = axes_flat[idx]
        md = ALL_STRAT_MONTHLY[sname]
        n_data = md["n"].reshape(1, 12)
        sns.heatmap(
            n_data,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Trades"},
            xticklabels=MON_NAMES,
            yticklabels=False,
        )
        ax.set_title(f"{sname}: Trades Per Month (CH55)", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=45)
    axes_flat[4].axis("off")
    axes_flat[5].axis("off")
    fig.suptitle("TRADE COUNT PER MONTH - All 5 Strategies (CH55, 2015-2026)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 7: YEARLY TRADE DISTRIBUTION ======
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(years))
    w = 0.15
    for i, (sname, _, _) in enumerate(strategies):
        vals = [ALL_STRAT_YEARLY[sname].get(yr, {}).get("n", 0) for yr in years]
        ax.bar(x + i*w, vals, w, label=sname, color=colors[i], alpha=0.85)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(years)
    ax.set_title("Yearly Trade Count - All Strategies (CH55)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Trades")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 8: YEARLY WIN RATE COMPARISON ======
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (sname, _, _) in enumerate(strategies):
        vals = [ALL_STRAT_YEARLY[sname].get(yr, {}).get("wr", 0) for yr in years]
        ax.plot(years, vals, "o-", label=sname, color=colors[i], linewidth=2, markersize=5)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Yearly Win Rate - All Strategies (CH55)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Win Rate")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 9: YEARLY W/L RATIO ======
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (sname, _, _) in enumerate(strategies):
        vals = [ALL_STRAT_YEARLY[sname].get(yr, {}).get("wl", 0) for yr in years]
        vals = [v if v < 20 else 0 for v in vals]  # cap outliers
        ax.plot(years, vals, "o-", label=sname, color=colors[i], linewidth=2, markersize=5)
    ax.axhline(y=1, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Yearly Win/Loss Ratio - All Strategies (CH55)", fontsize=13, fontweight="bold")
    ax.set_ylabel("W/L Ratio")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 10: BEST 5 STRATEGY MONTHLY RANKING ======
    fig, ax = plt.subplots(figsize=(14, 8))
    rank_data = []
    for sname in [s[0] for s in strategies]:
        row_net = ALL_STRAT_MONTHLY[sname]["net"]
        for m_idx, net_v in enumerate(row_net):
            if not np.isnan(net_v):
                rank_data.append({"Strategy": sname, "Month": MON_NAMES[m_idx], "Net": net_v})
    rank_df = pd.DataFrame(rank_data)
    pivot = rank_df.pivot_table(index="Strategy", columns="Month", values="Net", aggfunc="sum")
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.diverging_palette(10, 130, s=85, l=45, as_cmap=True),
        center=0,
        annot=True,
        fmt="+,.0f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Net Pts"},
    )
    ax.set_title("All Strategies × Month Net Return (CH55, 2015-2026)", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 11: NIFTY vs SENSEX MONTHLY COMPARISON ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    sns.heatmap(NIFTY_MONTHLY.reshape(1, 12), ax=ax1, cmap=sns.diverging_palette(10, 130, s=85, l=45, as_cmap=True),
                center=0, annot=True, fmt="+,.0f", linewidths=0.5,
                xticklabels=MON_NAMES, yticklabels=["NIFTY50"],
                cbar_kws={"shrink": 0.5, "label": "Net Pts"})
    ax1.set_title("NIFTY50 Monthly Net (Engulf_Raw CH55)", fontsize=11, fontweight="bold")
    sns.heatmap(SENSEX_MONTHLY.reshape(1, 12), ax=ax2, cmap=sns.diverging_palette(10, 130, s=85, l=45, as_cmap=True),
                center=0, annot=True, fmt="+,.0f", linewidths=0.5,
                xticklabels=MON_NAMES, yticklabels=["SENSEX"],
                cbar_kws={"shrink": 0.5, "label": "Net Pts"})
    ax2.set_title("SENSEX Monthly Net (Engulf_Raw CH55)", fontsize=11, fontweight="bold")
    fig.suptitle("NIFTY50 vs SENSEX Monthly Net Returns", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 12: YEARLY BAR WITH TOTALS ======
    fig, ax = plt.subplots(figsize=(14, 7))
    sname = "Engulf_Raw"
    yr_data = ALL_STRAT_YEARLY[sname]
    yr_vals = [yr_data.get(yr, {}).get("net", 0) for yr in years]
    colors_yr = ["#2ecc71" if v >= 0 else "#e74c3c" for v in yr_vals]
    bars = ax.bar(years, yr_vals, color=colors_yr, alpha=0.85, edgecolor="gray", linewidth=0.5)
    for bar, v in zip(bars, yr_vals):
        ypos = bar.get_height() + 8000 if v >= 0 else bar.get_height() - 15000
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{v:+,.0f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(f"{sname} - Yearly Net Return (CH55, 2015-2026)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Net Points")
    ax.set_xlabel("Year")
    total = sum(yr_vals)
    ax.text(0.5, 0.95, f"TOTAL: {total:+,.0f} pts",
            transform=ax.transAxes, ha="center", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    plt.close()

    # ====== PAGE 13: MONTHLY CUMULATIVE BY STRATEGY ======
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes_flat = axes.flatten()
    for idx, (sname, sdesc, _) in enumerate(strategies):
        ax = axes_flat[idx]
        md = ALL_STRAT_MONTHLY[sname]
        net_data = md["net"]
        cum_vals = np.nancumsum(net_data)
        colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in cum_vals]
        ax.bar(MON_NAMES, cum_vals, color=colors_bar, alpha=0.85)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"{sname}: Cumulative Monthly Net", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes_flat[4].axis("off")
    axes_flat[5].axis("off")
    fig.suptitle("CUMULATIVE MONTHLY NET (CH55, 2015-2026)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=200)
    plt.close()

print(f"\nSaved: {OUTPUT_PDF}")
print(f"Size: {os.path.getsize(OUTPUT_PDF):,} bytes")

# ====== TEXT SUMMARY ======
print("\n===== DATA SUMMARY =====")
print(f"\n{'Strategy':<15} {'Trades':>8} {'Net':>12} {'WR':>8} {'W/L':>8} {'Yearly%':>10}")
print("="*65)
for sname, _, _ in strategies:
    df = TRADE_SETS[sname]
    pnl = df["pts55"].values
    m = metrics(pnl)
    yr_prof = sum(1 for yr in years if ALL_STRAT_YEARLY[sname].get(yr, {}).get("net", 0) > 0)
    print(f"{sname:<15} {m['n']:>8} {m['net']:>+12,.0f} {m['wr']:>7.1%} {m['wl']:>7.1f}x {yr_prof/12:>9.0%}")

print("\n===== TOP MONTHS (Avg Net across all strats) =====")
all_mon_avg = np.zeros(12)
for sname, _, _ in strategies:
    all_mon_avg += np.nan_to_num(ALL_STRAT_MONTHLY[sname]["net"])
all_mon_avg /= 5
for m_idx in np.argsort(-all_mon_avg):
    print(f"  {MON_NAMES[m_idx]:5s}: {all_mon_avg[m_idx]:+9,.0f} avg pts")

print("\n===== WORST MONTHS =====")
for m_idx in np.argsort(all_mon_avg):
    print(f"  {MON_NAMES[m_idx]:5s}: {all_mon_avg[m_idx]:+9,.0f} avg pts")

print(f"\nPDF saved to: {OUTPUT_PDF}")
