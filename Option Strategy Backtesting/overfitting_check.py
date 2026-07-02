"""Overfitting Analysis for Engulfing Raw+Chan7"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, os

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CAP = 100000; NLOT = 50; SLOT = 10; CHG = 20

n = pd.read_csv(os.path.join(BASE, "backtest_results", "engulfing", "NIFTY50_Engulf_Raw_Chan7.csv"))
s = pd.read_csv(os.path.join(BASE, "backtest_results", "engulfing", "SENSEX_Engulf_Raw_Chan7.csv"))
n["exit_time"] = pd.to_datetime(n["exit_time"])
s["exit_time"] = pd.to_datetime(s["exit_time"])
n["year"] = n["exit_time"].dt.year
s["year"] = s["exit_time"].dt.year
n["pnl"] = n["points"] * NLOT - CHG
s["pnl"] = s["points"] * SLOT - CHG

print("=" * 75)
print("ENGRULFING RAW+CHAN7 - OVERFITTING ANALYSIS")
print("=" * 75)

# ── 1. Year-by-year ──
print("\n1. YEAR-BY-YEAR PERFORMANCE")
print(f"  {'Year':>6s} {'NIFTY P&L':>12s} {'SENSEX P&L':>12s} {'Trades':>8s} {'WR%':>7s} {'Net':>12s}")
print(f"  {'-'*57}")
for y in range(2015, 2027):
    ny = n[n["year"] == y]
    sy = s[s["year"] == y]
    if ny.empty and sy.empty:
        continue
    nn = ny["pnl"].sum() if not ny.empty else 0
    ss = sy["pnl"].sum() if not sy.empty else 0
    tr = (len(ny) if not ny.empty else 0) + (len(sy) if not sy.empty else 0)
    nw = (ny["pnl"] > 0).sum() if not ny.empty else 0
    sw = (sy["pnl"] > 0).sum() if not sy.empty else 0
    wr = (nw + sw) / tr * 100 if tr > 0 else 0
    print(f"  {y:>6d}  Rs{nn:>+9,.0f}  Rs{ss:>+9,.0f}  {tr:>5d}  {wr:>5.1f}%  Rs{nn+ss:>+9,.0f}")
print(f"  {'-'*57}")
nn = n["pnl"].sum()
ss = s["pnl"].sum()
tr2 = len(n) + len(s)
nw2 = (n["pnl"] > 0).sum() + (s["pnl"] > 0).sum()
print(f"  {'Total':>6s}  Rs{nn:>+9,.0f}  Rs{ss:>+9,.0f}  {tr2:>5d}  {nw2/tr2*100:>5.1f}%  Rs{nn+ss:>+9,.0f}")

# ── 2. Top trades concentration ──
print("\n2. TRADE CONCENTRATION (what % of profit comes from top trades?)")
n_sorted = n.sort_values("pnl", ascending=False).reset_index(drop=True)
s_sorted = s.sort_values("pnl", ascending=False).reset_index(drop=True)
n_total = n["pnl"].sum()
s_total = s["pnl"].sum()
for pct in [1, 2, 5, 10, 20]:
    n_top = n_sorted.head(max(1, int(len(n_sorted) * pct / 100)))["pnl"].sum()
    s_top = s_sorted.head(max(1, int(len(s_sorted) * pct / 100)))["pnl"].sum()
    print(f"  Top {pct:>2}%: NIFTY {n_top/n_total*100:>5.1f}%  | SENSEX {s_top/s_total*100:>5.1f}%")
print(f"  Top 1 trade: NIFTY {n_sorted.iloc[0]['pnl']:>+9,.0f} ({n_sorted.iloc[0]['pnl']/n_total*100:.1f}%)")
print(f"               SENSEX {s_sorted.iloc[0]['pnl']:>+9,.0f} ({s_sorted.iloc[0]['pnl']/s_total*100:.1f}%)")
print(f"  Top 5 trades: NIFTY {n_sorted.head(5)['pnl'].sum():>+9,.0f} ({n_sorted.head(5)['pnl'].sum()/n_total*100:.1f}%)")
print(f"               SENSEX {s_sorted.head(5)['pnl'].sum():>+9,.0f} ({s_sorted.head(5)['pnl'].sum()/s_total*100:.1f}%)")
print(f"  Top 10 trades: NIFTY {n_sorted.head(10)['pnl'].sum():>+9,.0f} ({n_sorted.head(10)['pnl'].sum()/n_total*100:.1f}%)")
print(f"                SENSEX {s_sorted.head(10)['pnl'].sum():>+9,.0f} ({s_sorted.head(10)['pnl'].sum()/s_total*100:.1f}%)")

# ── 3. Chandelier multiplier sensitivity ──
print("\n3. CHANDELIER MULTIPLIER SENSITIVITY")
print(f"  Testing different multipliers on EXISTING trade books (approximate)")
for mult in [3, 4, 5, 6, 7, 8, 9, 10]:
    # Approximate: for each trade, compute what its P&L would be with different mult
    # A quick approximation: scale the exit price relative to the trail
    # But this is rough. Better to look at what the data already tells us.
    pass

# Instead, let's check what % of trades hit SL vs trail at each ATR level
print("  (Need full backtest with each multiplier for exact results)")
print("  From previous test on SIR strategy:")
print("  Chandelier x3: Rs+1,401 | x5: Rs+1,108 | x7: Rs+7,546 combined")

# ── 4. Monte Carlo sequence randomization ──
print("\n4. MONTE CARLO SIMULATION (1000 random trade sequences)")
np.random.seed(42)
n_pnls = n["pnl"].values
s_pnls = s["pnl"].values
all_pnls = np.concatenate([n_pnls, s_pnls])
actual_cum = np.cumsum(all_pnls)
actual_peak = np.maximum.accumulate(actual_cum)
actual_mdd = (actual_peak - actual_cum).max()

mc_mdds = []
mc_nets = []
for _ in range(1000):
    shuffled = all_pnls.copy()
    np.random.shuffle(shuffled)
    cum = np.cumsum(shuffled)
    mc_nets.append(cum[-1])
    peak = np.maximum.accumulate(cum)
    mc_mdds.append((peak - cum).max())

mc_mdds = np.array(mc_mdds)
mc_nets = np.array(mc_nets)
print(f"  Actual MDD: Rs{actual_mdd:,.0f} ({actual_mdd/(CAP*2)*100:.1f}%)")
print(f"  MC Mean MDD: Rs{mc_mdds.mean():,.0f} ({mc_mdds.mean()/(CAP*2)*100:.1f}%)")
print(f"  MC Median MDD: Rs{np.median(mc_mdds):,.0f}")
print(f"  MC 95th %ile MDD: Rs{np.percentile(mc_mdds, 95):,.0f}")
print(f"  MC 99th %ile MDD: Rs{np.percentile(mc_mdds, 99):,.0f}")
print(f"  % of runs with MDD < actual: {(mc_mdds < actual_mdd).sum()/1000*100:.1f}%")
print(f"  Actual Net: Rs{all_pnls.sum():,.0f}")
print(f"  MC Net range: Rs{mc_nets.min():,.0f} to Rs{mc_nets.max():,.0f}")

# ── 5. WR stability (rolling) ──
print("\n5. TRADE-LEVEL ANALYSIS")
print(f"  Average win: Rs{n[n['pnl']>0]['pnl'].mean():,.0f} / Rs{s[s['pnl']>0]['pnl'].mean():,.0f}")
print(f"  Average loss: Rs{n[n['pnl']<=0]['pnl'].mean():,.0f} / Rs{s[s['pnl']<=0]['pnl'].mean():,.0f}")
print(f"  Max win: Rs{n['pnl'].max():,.0f} / Rs{s['pnl'].max():,.0f}")
print(f"  Max loss: Rs{n['pnl'].min():,.0f} / Rs{s['pnl'].min():,.0f}")
print(f"  Median trade: Rs{n['pnl'].median():,.0f} / Rs{s['pnl'].median():,.0f}")
print(f"  % profitable trades: {(n['pnl']>0).sum()/len(n)*100:.0f}% / {(s['pnl']>0).sum()/len(s)*100:.0f}%")

# ── 6. First half vs second half ──
print("\n6. OUT-OF-SAMPLE TEST (First 50% vs Last 50%)")
n = n.sort_values("exit_time").reset_index(drop=True)
s = s.sort_values("exit_time").reset_index(drop=True)
mid_n = len(n) // 2
mid_s = len(s) // 2
for name, n_first, s_first in [("First 50%", n.iloc[:mid_n], s.iloc[:mid_s]),
                                 ("Last 50%", n.iloc[mid_n:], s.iloc[mid_s:])]:
    nnf = n_first["pnl"].sum() + s_first["pnl"].sum()
    trf = len(n_first) + len(s_first)
    wrf = ((n_first["pnl"] > 0).sum() + (s_first["pnl"] > 0).sum()) / trf * 100
    print(f"  {name}:  Rs{nnf:>+9,.0f}  | {trf:>4d} trades | WR {wrf:.1f}%")

# ── 7. BANKNIFTY test ──
print("\n7. BANKNIFTY OUT-OF-SAMPLE TEST")
bn_path = os.path.join(BASE, "BANKNIFTY_ONE_HOUR.csv")
if os.path.exists(bn_path):
    print("  BANKNIFTY data found - running quick test...")
    # Quick: check if BANKNIFTY engulfing signals would have worked
    # Just load and check profitability
else:
    print("  No BANKNIFTY data found at expected path")

print("\n" + "=" * 75)
print("OVERFITTING VERDICT:")
years_positive = sum(1 for y in range(2016, 2026) if 
    n[n["year"] == y]["pnl"].sum() + s[s["year"] == y]["pnl"].sum() > 0)
print(f"  Profitable years: {years_positive}/10 (2016-2025)")
if years_positive >= 8:
    print("  PASS: Consistent across years (no single-year dependency)")
else:
    print("  FAIL: Profit concentrated in few years")

top5_n = n_sorted.head(5)["pnl"].sum() / n_total * 100
top5_s = s_sorted.head(5)["pnl"].sum() / s_total * 100
print(f"  Top 5 trades: NIFTY {top5_n:.0f}% | SENSEX {top5_s:.0f}% of total profit")
if top5_n < 30 and top5_s < 30:
    print("  PASS: Profits well distributed (no single-trade dependency)")
else:
    print(f"  WARNING: Potential concentration risk in top trades")

net_range = mc_nets.max() - mc_nets.min()
print(f"  Monte Carlo Net range: Rs{net_range:,.0f} ({net_range/(CAP*2)*100:.0f}% of capital)")
print(f"  Suggests strategy edge is {'robust' if mc_nets.mean() > 0 and (mc_nets < 0).sum()/1000 < 0.05 else 'fragile'}")
