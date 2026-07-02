"""
COMPREHENSIVE PER-SYMBOL TEST: NIFTY50 vs SENSEX
All 20 CH versions × 1-lot/1w1l × with/without skip × point-based metrics
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF = pd.Timestamp("14:15").time()

def A(df, p=14):
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()

VER = {
    "DynCH 25+10": (25, 10), "DynCH 30+10": (30, 10), "DynCH 30+15": (30, 15),
    "DynCH 35+10": (35, 10), "DynCH 35+15": (35, 15),
    "DynCH 40+5": (40, 5), "DynCH 40+10": (40, 10), "DynCH 40+12": (40, 12),
    "DynCH 45+5": (45, 5), "DynCH 45+8": (45, 8), "DynCH 45+10": (45, 10),
    "DynCH 45+12": (45, 12), "DynCH 45+15": (45, 15),
    "DynCH 50+8": (50, 8), "DynCH 50+10": (50, 10), "DynCH 50+12": (50, 12),
    "DynCH 55+10": (55, 10), "DynCH 55+15": (55, 15),
    "DynCH 60+10": (60, 10), "DynCH 60+15": (60, 15),
}
VN = list(VER.keys()); CB = [VER[v][0] for v in VN]; CR = [VER[v][1] for v in VN]
NV = len(VN); CH_VALS = sorted(set(CB))

print("=" * 140)
print("PER-SYMBOL ANALYSIS: NIFTY50 vs SENSEX — ALL STRATEGIES × ALL MODES")
print("=" * 140)

def build_trades_for_symbol(sym):
    """Build trades for ONE symbol, returns list of trade dicts."""
    h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
    m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)
    tr = pd.concat([h1["high"] - h1["low"],
                    (h1["high"] - h1["close"].shift(1)).abs(),
                    (h1["low"] - h1["close"].shift(1)).abs()], axis=1).max(axis=1)
    h1["atr14"] = tr.rolling(14, min_periods=14).mean()
    h1["atr_ma20"] = h1["atr14"].rolling(20).mean()
    body = (h1["close"] - h1["open"]).abs()
    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]
    du = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi = m5["high"].values; lo = m5["low"].values; cl = m5["close"].values
    atr5 = A(m5, 14).values; tc = m5["datetime"].dt.time
    trades = []
    for i in range(60, len(h1)):
        if not is_red.iloc[i-1] or not is_green.iloc[i]:
            continue
        if h1["open"].iloc[i] > h1["close"].iloc[i-1] or h1["close"].iloc[i] < h1["open"].iloc[i-1]:
            continue
        if body.iloc[i] < body.iloc[i-1] * 0.5:
            continue
        tu = int(h1["datetime"].iloc[i].timestamp())
        lv = h1["high"].iloc[i]
        idx = np.searchsorted(du, tu, side="right")
        if idx >= len(m5):
            continue
        b = idx
        while b < len(m5) and cl[b] <= lv:
            b += 1
        if b >= len(m5):
            continue
        r = b + 1
        while r < len(m5):
            if lo[r] < lv and cl[r] > lv and tc.iloc[r] < CUTOFF:
                break
            r += 1
        if r >= len(m5):
            continue
        ep = cl[r]
        if ep - lo[r] <= 0 or m5["datetime"].iloc[r].hour == 9:
            continue
        a14 = h1["atr14"].iloc[i]; a20 = h1["atr_ma20"].iloc[i]
        reg = 0
        if not pd.isna(a14) and not pd.isna(a20) and a14 > a20:
            reg = 1
        elif not pd.isna(a14):
            reg = 2
        pnls = {}
        for cv in CH_VALS:
            he = ep
            for j in range(r, len(m5)):
                ca = atr5[j]
                if pd.isna(ca):
                    continue
                if hi[j] > he:
                    he = hi[j]
                if cl[j] < he - cv * ca:
                    pnls[cv] = (cl[j] - ep)  # POINTS (no *bl, no -20 fee)
                    break
        if 45 not in pnls:
            continue
        trades.append({
            "dt": h1["datetime"].iloc[i], "sym": sym,
            "year": h1["datetime"].iloc[i].year,
            "ep": ep, "reg": reg, "pnls": pnls.copy()
        })
    return trades

def get_pts(t, ch_b, ch_r):
    """Return point P&L (without lot multiplication or fees)."""
    if t["reg"] == 1:
        cv = ch_b - ch_r
    elif t["reg"] == 2:
        cv = ch_b + ch_r
    else:
        cv = ch_b
    nearest = min(CH_VALS, key=lambda x: abs(x - cv))
    return t["pnls"].get(nearest)

def compute_metrics(points_list, use_1w1l=False, use_skip=False, train_losses=None):
    """
    Given list of (point_gain, dt), compute comprehensive metrics.
    Returns dict with all metrics.
    """
    # Convert points to net P&L using lot sizes and fee
    # We'll compute both: point stats AND rupee stats
    if len(points_list) == 0:
        return {"N": 0, "Net": 0, "WR": 0, "PF": 0, "MDD": 0, "AvgGain": 0, "AvgLoss": 0,
                "MaxWin": 0, "MaxLoss": 0, "StDev": 0, "Sharpe": 0, "RoMaD": 0,
                "WinPts": 0, "LossPts": 0, "AvgWinPts": 0, "AvgLossPts": 0}

    if use_1w1l:
        pos = 1
    net_list = []
    pts_list = []
    for idx, (pts, dt) in enumerate(points_list):
        if use_skip and train_losses is not None and idx > 0:
            # Check skip: was prev a loss AND small loss?
            prev_pts = points_list[idx-1][0]
            prev_net = prev_pts  # in points for skip logic
            prior = [x for x in train_losses if x < 0]
            th = np.median(prior) if len(prior) > 5 else -10
            if prev_net < 0 and prev_net < th:
                continue  # skip this trade
        if use_1w1l:
            net = pts * pos
            pts_list.append(pts * pos)
            net_list.append(pts * pos)
            if pts > 0:
                pos = 2
            else:
                pos = 1
        else:
            net = pts
            pts_list.append(pts)
            net_list.append(pts)

    n = len(net_list)
    if n == 0:
        return {"N": 0, "Net": 0, "WR": 0, "PF": 0, "MDD": 0, "AvgGain": 0, "AvgLoss": 0,
                "MaxWin": 0, "MaxLoss": 0, "StDev": 0, "Sharpe": 0, "RoMaD": 0,
                "WinPts": 0, "LossPts": 0, "AvgWinPts": 0, "AvgLossPts": 0}

    total_net = sum(net_list)
    wins = [x for x in net_list if x > 0]
    losses = [x for x in net_list if x < 0]
    wr = len(wins) / n * 100
    pf = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    max_win = max(wins) if wins else 0
    max_loss = min(losses) if losses else 0
    stdev = np.std(net_list)
    sharpe = (np.mean(net_list) / stdev * np.sqrt(252)) if stdev > 0 else 0
    cum = 0; peak = 0; mdd = 0
    for x in net_list:
        cum += x
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
    romad = (total_net / mdd) if mdd > 0 else 0

    # Point stats (unscaled, pre-lot)
    wins_p = [x for x in pts_list if x > 0]
    losses_p = [x for x in pts_list if x < 0]
    total_win_pts = sum(wins_p)
    total_loss_pts = sum(losses_p)
    avg_win_pts = np.mean(wins_p) if wins_p else 0
    avg_loss_pts = np.mean(losses_p) if losses_p else 0

    return {
        "N": n, "Net": total_net, "WR": wr, "PF": pf,
        "MDD": mdd, "AvgGain": avg_win, "AvgLoss": avg_loss,
        "MaxWin": max_win, "MaxLoss": max_loss, "StDev": stdev,
        "Sharpe": sharpe, "RoMaD": romad,
        "WinPts": total_win_pts, "LossPts": total_loss_pts,
        "AvgWinPts": avg_win_pts, "AvgLossPts": avg_loss_pts
    }

# ─────────────── BUILD TRADES FOR BOTH SYMBOLS ───────────────
print("\nBuilding trades for NIFTY50...")
nifty_trades = build_trades_for_symbol("NIFTY50")
print(f"  NIFTY50: {len(nifty_trades)} trades")

print("Building trades for SENSEX...")
sensex_trades = build_trades_for_symbol("SENSEX")
print(f"  SENSEX: {len(sensex_trades)} trades")

cut = pd.Timestamp("2022-01-01").tz_localize("Asia/Kolkata")

def analyze_symbol(all_trades, sym_name, lot_size, fee_per_trade):
    """Analyze all strategies for one symbol. Returns results list."""
    train = [t for t in all_trades if t["dt"] < cut]
    test = [t for t in all_trades if t["dt"] >= cut]
    print(f"\n  {sym_name}: {len(train)} train, {len(test)} test")

    results = []
    for v in range(NV):
        vn = VN[v]; cb = CB[v]; cr = CR[v]
        # Build point lists
        train_pts = []
        for t in train:
            p = get_pts(t, cb, cr)
            if p is not None:
                train_pts.append((p, t["dt"]))
        test_pts = []
        for t in test:
            p = get_pts(t, cb, cr)
            if p is not None:
                test_pts.append((p, t["dt"]))

        if len(test_pts) == 0:
            continue

        train_losses = [p for p, dt in train_pts if p < 0]

        # 1-lot Base
        m1 = compute_metrics(test_pts, False, False)
        # 1-lot Skip
        m2 = compute_metrics(test_pts, False, True, train_losses)
        # 1w1l Base
        m3 = compute_metrics(test_pts, True, False)
        # 1w1l Skip
        m4 = compute_metrics(test_pts, True, True, train_losses)

        results.append((vn, cb, cr, m1, m2, m3, m4, test_pts, train_losses))
    return results

print("\n" + "=" * 140)
print("ANALYZING NIFTY50")
nifty_results = analyze_symbol(nifty_trades, "NIFTY50", 50, 20)

print("\n" + "=" * 140)
print("ANALYZING SENSEX")
sensex_results = analyze_symbol(sensex_trades, "SENSEX", 10, 20)

# ─────────────── PRINT COMPARISON TABLE ───────────────
def print_comparison_table(nifty_res, sensex_res, mode_name, use_1w1l, use_skip):
    """Print side-by-side comparison for a given mode."""
    print(f"\n{'=' * 140}")
    print(f"MODE: {mode_name}")
    print(f"{'=' * 140}")
    hdr = f"{'Version':>35s} | {'NIFTY N':>6s} {'NetPts':>8s} {'WR%':>5s} {'PF':>6s} {'MDD':>8s} {'Sharpe':>7s} | {'SENSEX N':>6s} {'NetPts':>8s} {'WR%':>5s} {'PF':>6s} {'MDD':>8s} {'Sharpe':>7s} | {'DiffPts':>8s}"
    print(hdr)
    print("-" * 140)

    # Tuple: (vn, cb, cr, m1, m2, m3, m4, test_pts, train_losses)
    # m1=idx3 (1-lot base), m2=idx4 (1-lot skip), m3=idx5 (1w1l base), m4=idx6 (1w1l skip)
    base_idx = 5 if use_1w1l else 3
    m_idx = base_idx + (1 if use_skip else 0)

    rows = []
    for nr, sr in zip(nifty_res, sensex_res):
        vn = nr[0]
        n = nr[m_idx]
        s = sr[m_idx]
        diff = n["Net"] - s["Net"]
        rows.append((n["Net"] + s["Net"], vn, n, s, diff))

    rows.sort(key=lambda x: -x[0])
    for _, vn, n, s, diff in rows:
        print(f"{vn:>35s} | {n['N']:>6d} {n['Net']:>+8.1f} {n['WR']:>4.1f}% {n['PF']:>5.2f} {n['MDD']:>+8.1f} {n['Sharpe']:>6.2f} | {s['N']:>6d} {s['Net']:>+8.1f} {s['WR']:>4.1f}% {s['PF']:>5.2f} {s['MDD']:>+8.1f} {s['Sharpe']:>6.2f} | {diff:>+8.1f}")

    # Best overall
    best_row = rows[0]
    print(f"\n  BEST: {best_row[1]} — Combined NetPts={best_row[0]:.1f}")

modes = [
    ("1-LOT BASE", False, False),
    ("1-LOT WITH SKIP", False, True),
    ("1W1L BASE", True, False),
    ("1W1L WITH SKIP", True, True),
]

for mode_name, use_1w1l, use_skip in modes:
    print_comparison_table(nifty_results, sensex_results, mode_name, use_1w1l, use_skip)

# ─────────────── DETAILED PER-SYMBOL YEAR BREAKDOWN ───────────────
print(f"\n{'=' * 140}")
print("YEAR BREAKDOWN: BEST VERSION (DynCH 60+10)")
print(f"{'=' * 140}")

for sym_name, all_trades, lot_size in [("NIFTY50", nifty_trades, 50), ("SENSEX", sensex_trades, 10)]:
    v = VN.index("DynCH 60+10")
    cb = CB[v]; cr = CR[v]
    train = [t for t in all_trades if t["dt"] < cut]
    test = [t for t in all_trades if t["dt"] >= cut]

    print(f"\n  {sym_name} (lot={lot_size}):")

    # Compute train losses
    train_pts = []
    for t in train:
        p = get_pts(t, cb, cr)
        if p is not None: train_pts.append((p, t["dt"]))
    train_losses = [p for p, dt in train_pts if p < 0]

    for yr in sorted(set(t["year"] for t in test)):
        yr_t = [t for t in test if t["year"] == yr]
        yr_pts = []
        for t in yr_t:
            p = get_pts(t, cb, cr)
            if p is not None: yr_pts.append((p, t["dt"]))
        if len(yr_pts) < 2: continue

        # 1-lot base
        mb = compute_metrics(yr_pts, False, False)
        # 1-lot skip (uses ALL prior losses, not just this year)
        ms = compute_metrics(yr_pts, False, True, train_losses)
        # 1w1l base
        mw = compute_metrics(yr_pts, True, False)
        # 1w1l skip
        mws = compute_metrics(yr_pts, True, True, train_losses)

        l1 = f"1-lot Base={mb['Net']:>+7.1f}pts -> Skip={ms['Net']:>+7.1f}pts"
        l2 = f"1w1l Base={mw['Net']:>+7.1f}pts -> Skip={mws['Net']:>+7.1f}pts"
        print(f"    {yr}: {l1} | {l2}")

    # Totals
    test_pts = [(get_pts(t, cb, cr), t["dt"]) for t in test if get_pts(t, cb, cr) is not None]
    mb = compute_metrics(test_pts, False, False)
    ms = compute_metrics(test_pts, False, True, train_losses)
    mw = compute_metrics(test_pts, True, False)
    mws = compute_metrics(test_pts, True, True, train_losses)
    tl1 = f"1-lot Base={mb['Net']:>+7.1f}pts -> Skip={ms['Net']:>+7.1f}pts"
    tl2 = f"1w1l Base={mw['Net']:>+7.1f}pts -> Skip={mws['Net']:>+7.1f}pts"
    print(f"    TOTAL: {tl1} | {tl2}")

# ─────────────── ALL METRICS FOR BEST ───────────────
v = VN.index("DynCH 60+10")
cb = CB[v]; cr = CR[v]

for sym_name, all_trades, lot_size in [("NIFTY50", nifty_trades, 50), ("SENSEX", sensex_trades, 10)]:
    train = [t for t in all_trades if t["dt"] < cut]
    test = [t for t in all_trades if t["dt"] >= cut]
    test_pts = [(get_pts(t, cb, cr), t["dt"]) for t in test if get_pts(t, cb, cr) is not None]
    train_pts = [(get_pts(t, cb, cr), t["dt"]) for t in train if get_pts(t, cb, cr) is not None]
    train_losses = [p for p, dt in train_pts if p < 0]

    m1 = compute_metrics(test_pts, False, False)
    m2 = compute_metrics(test_pts, False, True, train_losses)
    m3 = compute_metrics(test_pts, True, False)
    m4 = compute_metrics(test_pts, True, True, train_losses)

    if sym_name == "NIFTY50":
        nm, nw, ns1, nsw = m1, m3, m2, m4
    else:
        sm, sw, ss1, ssw = m1, m3, m2, m4

def fmtv(v):
    if isinstance(v, float) and abs(v) > 1e9:
        return f"{v:.1e}"
    if isinstance(v, float):
        return f"{v:>+10.1f}"
    return f"{v:>10}"

metrics_list = ["N", "Net", "WR", "PF", "MDD", "AvgGain", "AvgLoss", "MaxWin", "MaxLoss", "StDev", "Sharpe", "RoMaD"]
print(f"\nDetailed metrics for DynCH 60+10 (test set):")
for m in metrics_list:
    print(f"  {m:>12s}: Lot1 N={fmtv(nm[m])} S={fmtv(sm[m])} | 1w1l N={fmtv(nw[m])} S={fmtv(sw[m])} | Skip1 N={fmtv(ns1[m])} S={fmtv(ss1[m])} | SkipW N={fmtv(nsw[m])} S={fmtv(ssw[m])}")

# ─────────────── SYMMETRY ANALYSIS ───────────────
print(f"\n{'=' * 140}")
print("SYMMETRY: Does skip help both symbols equally?")
print(f"{'=' * 140}")

for mode_name, use_1w1l, use_skip in modes:
    base_idx = 5 if use_1w1l else 3
    skip_idx = base_idx + 1
    m_idx = base_idx if not use_skip else skip_idx

    nifty_best = sorted([(nr[m_idx]["Net"], nr[0]) for nr in nifty_results],
                        key=lambda x: -x[0])
    sensex_best = sorted([(sr[m_idx]["Net"], sr[0]) for sr in sensex_results],
                         key=lambda x: -x[0])

    print(f"\n  {mode_name}:")
    print(f"  NIFTY50 best: {nifty_best[0][1]} ({nifty_best[0][0]:+.1f} pts)")
    print(f"  SENSEX best:  {sensex_best[0][1]} ({sensex_best[0][0]:+.1f} pts)")

    # Same version?
    if nifty_best[0][1] == sensex_best[0][1]:
        print(f"  SAME version for both")
    else:
        print(f"  DIFFERENT versions")

    # Skip improvement%
    for sym_res, sym_name in [(nifty_results, "NIFTY50"), (sensex_results, "SENSEX")]:
        base_idx = 5 if use_1w1l else 3
        skip_idx = base_idx + 1
        gains = []
        for nr in sym_res:
            base_net = nr[base_idx]["Net"]
            skip_net = nr[skip_idx]["Net"]
            if base_net != 0:
                gains.append((skip_net / base_net - 1) * 100)
        avg_impr = np.mean(gains) if gains else 0
        print(f"  {sym_name}: avg skip improvement = {avg_impr:+.1f}%")

print(f"\n{'=' * 100}")
print("COMPLETE - ALL SYMBOLS x ALL VERSIONS x ALL MODES")
print(f"{'=' * 100}")
