"""
FINAL COMPREHENSIVE TEST: ALL strategy versions × 1-lot/1w1l × with/without skip
Correct Python (no continue;stmt on same line bug).
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

# All 20 CH variants
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

print("=" * 100)
print("FINAL COMPREHENSIVE TEST: ALL VERSIONS × 1-LOT / 1W1L × WITH/WITHOUT SKIP")
print("=" * 100)

# ─── STEP 1: Build all trades ───
print("\nBuilding trades...")
trades = []
for sym in ["NIFTY50", "SENSEX"]:
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
    bl = 50 if "NIFTY" in sym else 10
    du = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi = m5["high"].values; lo = m5["low"].values; cl = m5["close"].values
    atr5 = A(m5, 14).values; tc = m5["datetime"].dt.time

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
        # Precompute P&L for all CH values
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
                    pnls[cv] = (cl[j] - ep) * bl - 20
                    break
        if 45 not in pnls:
            continue
        trades.append({
            "dt": h1["datetime"].iloc[i], "sym": sym,
            "year": h1["datetime"].iloc[i].year, "bl": bl,
            "ep": ep, "reg": reg, "pnls": pnls.copy()
        })

print(f"Total trades: {len(trades)}")
cut = pd.Timestamp("2022-01-01").tz_localize("Asia/Kolkata")
train = [t for t in trades if t["dt"] < cut]
test = [t for t in trades if t["dt"] >= cut]
print(f"Train: {len(train)}, Test: {len(test)}")

def get_pnl(t, ch_b, ch_r):
    if t["reg"] == 1:
        cv = ch_b - ch_r
    elif t["reg"] == 2:
        cv = ch_b + ch_r
    else:
        cv = ch_b
    nearest = min(CH_VALS, key=lambda x: abs(x - cv))
    return t["pnls"].get(nearest)

# ─── STEP 2: Evaluate ───
def evaluate(v_idx, use_1w1l=False, use_skip=False):
    """Returns (base_net, skip_net, base_n, skip_n, base_wr, skip_wr, base_max_drawdown, skip_max_drawdown)"""
    vn = VN[v_idx]; cb = CB[v_idx]; cr = CR[v_idx]
    base_p = []  # (net, dt)
    for t in test:
        p = get_pnl(t, cb, cr)
        if p is not None:
            base_p.append((p, t["dt"]))
    base_n = len(base_p)
    if base_n == 0:
        return (0, 0, 0, 0, 0, 0, 0, 0)

    if use_1w1l:
        pos = 1  # current position size (1 or 2)
        base_net_list = []
        for i in range(base_n):
            pnl = base_p[i][0]
            net = pnl * pos
            base_net_list.append(net)
            if pnl > 0:
                pos = 2
            else:
                pos = 1
        base_net = sum(base_net_list)
        base_wr = sum(1 for x in base_net_list if x > 0) / base_n * 100
        cum = 0; peak = 0; mdd = 0
        for x in base_net_list:
            cum += x
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > mdd: mdd = dd
    else:
        base_net_list = [p[0] for p in base_p]
        base_net = sum(base_net_list)
        base_wr = sum(1 for x in base_net_list if x > 0) / base_n * 100
        cum = 0; peak = 0; mdd = 0
        for x in base_net_list:
            cum += x
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > mdd: mdd = dd

    # Train loss median
    train_losses = []
    for t in train:
        p = get_pnl(t, cb, cr)
        if p is not None and p < 0:
            train_losses.append(p)
    loss_med = np.median(train_losses) if len(train_losses) > 5 else -5000

    if not use_skip:
        return (base_net, base_net, base_n, base_n, base_wr, base_wr, mdd, mdd)

    # With skip filter
    if use_1w1l:
        pos = 1
        skip_list = []
        prior_l = list(train_losses.copy())
        for i in range(base_n):
            pnl = base_p[i][0]
            net = pnl * pos
            prior = [x for x in prior_l if x < 0]
            th = np.median(prior) if len(prior) > 5 else loss_med
            skip = False
            if i > 0:
                prev_net = base_net_list[i-1] if use_1w1l else base_p[i-1][0]
                if prev_net < 0 and prev_net < th:
                    skip = True
            if not skip:
                skip_list.append(net)
                prior_l.append(pnl)
                if pnl > 0:
                    pos = 2
                else:
                    pos = 1
            else:
                if pnl > 0:
                    pos = 2
                else:
                    pos = 1
        skip_net = sum(skip_list); skip_n = len(skip_list)
        skip_wr = sum(1 for x in skip_list if x > 0) / skip_n * 100 if skip_n > 0 else 0
        cum = 0; peak = 0; smdd = 0
        for x in skip_list:
            cum += x
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > smdd: smdd = dd
    else:
        skip_list = []
        prior_l = list(train_losses.copy())
        for i in range(base_n):
            pnl = base_p[i][0]
            prior = [x for x in prior_l if x < 0]
            th = np.median(prior) if len(prior) > 5 else loss_med
            skip = False
            if i > 0 and base_p[i-1][0] < 0 and base_p[i-1][0] < th:
                skip = True
            if not skip:
                skip_list.append(pnl)
                prior_l.append(pnl)
        skip_net = sum(skip_list); skip_n = len(skip_list)
        skip_wr = sum(1 for x in skip_list if x > 0) / skip_n * 100 if skip_n > 0 else 0
        cum = 0; peak = 0; smdd = 0
        for x in skip_list:
            cum += x
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > smdd: smdd = dd

    return (base_net, skip_net, base_n, skip_n, base_wr, skip_wr, mdd, smdd)

# ─── STEP 3: Run ALL combinations ───
modes = [
    ("1-LOT", False, False, "Base"),
    ("1-LOT", False, True, "Skip"),
    ("1W1L", True, False, "Base"),
    ("1W1L", True, True, "Skip"),
]

for sz_name, use_1w1l, use_skip, label in modes:
    print(f"\n{'=' * 100}")
    print(f"  {sz_name} {label}")
    print(f"{'=' * 100}")
    print(f"{'VERSION':>40s} {'Net':>12s} {'N':>5s} {'WR%':>6s} {'MDD':>10s}")
    print(f"{'-'*40} {'-'*12} {'-'*5} {'-'*6} {'-'*10}")

    results = []
    for v in range(NV):
        bn, sn, bn_n, sn_n, bwr, swr, bmdd, smdd = evaluate(v, use_1w1l, use_skip)
        net = sn if use_skip else bn
        n = sn_n if use_skip else bn_n
        wr = swr if use_skip else bwr
        mdd = smdd if use_skip else bmdd
        results.append((net, VN[v], n, wr, mdd, bn, sn, bn_n, sn_n))
        print(f"{VN[v]:>40s} Rs{net:>+9,.0f} {n:>5d} {wr:>5.1f}% Rs{mdd:>+8,.0f}")

    # Ranked
    print(f"\n  RANKED:")
    results.sort(key=lambda x: -x[0])
    for rank, (net, vn, n, wr, mdd, bn, sn, bn_n, sn_n) in enumerate(results, 1):
        print(f"  {rank:>2d}. {vn:>35s} Rs{net:>+9,.0f} {n:>4d} trades {wr:>5.1f}% WR MDD=Rs{mdd:>+8,.0f}")

# ─── STEP 4: WALK-FORWARD FOR TOP COMBOS ───
print(f"\n{'=' * 100}")
print("WALK-FORWARD: Top 3 in each sizing × mode")
print(f"{'=' * 100}")

for sz_name, use_1w1l, use_skip in [("1-LOT", False, True), ("1W1L", True, True)]:
    print(f"\n--- {sz_name} WITH SKIP ---")
    sub = []
    for v in range(NV):
        bn, sn, bn_n, sn_n, bwr, swr, bmdd, smdd = evaluate(v, use_1w1l, use_skip)
        sub.append((sn, VN[v], bn, sn, bn_n, sn_n, bwr, swr))
    sub.sort(key=lambda x: -x[0])

    for rank in range(min(3, len(sub))):
        sn, vn, bn, s, bn_n, sn_n, bwr, swr = sub[rank]
        v = VN.index(vn)
        cb = CB[v]; cr = CR[v]

        print(f"\n  [{vn}] Net=Rs{bn:+,.0f}->Rs{s:+,.0f} ({s/bn*100-100:+.1f}%) WR={bwr:.0f}%->{swr:.0f}%")

        # Per-year breakdown
        train_losses = []
        for t in train:
            p = get_pnl(t, cb, cr)
            if p is not None and p < 0:
                train_losses.append(p)
        loss_med0 = np.median(train_losses) if len(train_losses) > 5 else -5000

        for yr in sorted(set(t["year"] for t in test)):
            yr_t = [t for t in test if t["year"] == yr]
            yr_p = []
            for t in yr_t:
                p = get_pnl(t, cb, cr)
                if p is not None:
                    yr_p.append((p, t["dt"]))
            if len(yr_p) < 3:
                continue

            # Base
            if use_1w1l:
                pos = 1; base_yr_list = []
                for pnl, dt in yr_p:
                    net = pnl * pos; base_yr_list.append(net)
                    if pnl > 0: pos = 2
                    else: pos = 1
            else:
                base_yr_list = [p for p, dt in yr_p]
            base_yr = sum(base_yr_list)

            # With skip
            prior = [t for t in train if t["year"] < yr]
            prior_losses = []
            for t in prior:
                p = get_pnl(t, cb, cr)
                if p is not None and p < 0:
                    prior_losses.append(p)
            th0 = np.median(prior_losses) if len(prior_losses) > 5 else loss_med0

            if use_1w1l:
                pos = 1; skip_yr_list = []; pl = list(prior_losses.copy())
                for idx in range(len(yr_p)):
                    pnl, dt = yr_p[idx]
                    net = pnl * pos
                    pr = [x for x in pl if x < 0]
                    th = np.median(pr) if len(pr) > 5 else th0
                    sk = False
                    if idx > 0 and base_yr_list[idx-1] < 0 and base_yr_list[idx-1] < th:
                        sk = True
                    if not sk:
                        skip_yr_list.append(net)
                        pl.append(pnl)
                    if pnl > 0: pos = 2
                    else: pos = 1
            else:
                skip_yr_list = []; pl = list(prior_losses.copy())
                for idx in range(len(yr_p)):
                    pnl, dt = yr_p[idx]
                    pr = [x for x in pl if x < 0]
                    th = np.median(pr) if len(pr) > 5 else th0
                    sk = False
                    if idx > 0 and yr_p[idx-1][0] < 0 and yr_p[idx-1][0] < th:
                        sk = True
                    if not sk:
                        skip_yr_list.append(pnl)
                        pl.append(pnl)

            skip_yr = sum(skip_yr_list)
            imp = (skip_yr / base_yr - 1) * 100 if base_yr != 0 else 0
            print(f"    {yr}: Base=Rs{base_yr:>+9,.0f} Skip=Rs{skip_yr:>+9,.0f} ({imp:>+.1f}%) N={len(skip_yr_list)}")

        print(f"    TOTAL: Rs{bn:>+9,.0f} -> Rs{s:>+9,.0f} ({s/bn*100-100:+.1f}%)")

print(f"\n{'=' * 100}")
print("DONE - ALL STRATEGIES TESTED")
print(f"{'=' * 100}")
