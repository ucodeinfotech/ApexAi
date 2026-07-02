"""OR Raw+FixTP - Excluding March 2020 to check real edge"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
OR_DIR = os.path.join(BASE, "backtest_results", "combined_or")

nifty = pd.read_csv(os.path.join(OR_DIR, "NIFTY50_OR_Raw_FixTP.csv"))
sensex = pd.read_csv(os.path.join(OR_DIR, "SENSEX_OR_Raw_FixTP.csv"))
nifty["exit_time"] = pd.to_datetime(nifty["exit_time"])
sensex["exit_time"] = pd.to_datetime(sensex["exit_time"])
nifty["date"] = nifty["exit_time"].dt.date
sensex["date"] = sensex["exit_time"].dt.date

CAP = 100000; NLOT = 50; SLOT = 10; CHG = 20

def analyze(df_n, df_s, label):
    df_n = df_n.copy(); df_s = df_s.copy()
    df_n["pnl"] = df_n["points"] * NLOT - CHG
    df_s["pnl"] = df_s["points"] * SLOT - CHG
    
    n_net = df_n["pnl"].sum(); s_net = df_s["pnl"].sum(); tot = n_net + s_net
    
    n_w = len(df_n[df_n["pnl"]>0]); n_l = len(df_n[df_n["pnl"]<=0])
    s_w = len(df_s[df_s["pnl"]>0]); s_l = len(df_s[df_s["pnl"]<=0])
    
    n_gp = df_n[df_n["pnl"]>0]["pnl"].sum(); n_gl = abs(df_n[df_n["pnl"]<=0]["pnl"].sum())
    s_gp = df_s[df_s["pnl"]>0]["pnl"].sum(); s_gl = abs(df_s[df_s["pnl"]<=0]["pnl"].sum())
    
    n_daily = df_n.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl":"n_pnl"})
    s_daily = df_s.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl":"s_pnl"})
    comb = pd.merge(n_daily, s_daily, on="date", how="outer").fillna(0).sort_values("date").reset_index(drop=True)
    comb["total"] = comb["n_pnl"] + comb["s_pnl"]
    comb["cum"] = comb["total"].cumsum()
    peak = comb["cum"].cummax(); dd = peak - comb["cum"]; mdd = dd.max()
    
    returns = comb["total"] / (CAP * 2)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    cagr = ((1 + tot/(CAP*2)) ** (1/10) - 1) * 100 if tot > -(CAP*2) else -100
    
    print(f"\n{label}")
    print(f"  {'':20s} {'NIFTY50':>12s} {'SENSEX':>12s} {'COMBINED':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'Trades':20s} {len(df_n):>8d}      {len(df_s):>8d}      {len(df_n)+len(df_s):>8d}")
    print(f"  {'Trading Days':20s} {df_n['date'].nunique():>8d}      {df_s['date'].nunique():>8d}      {comb['date'].nunique():>8d}")
    print(f"  {'Net P&L':20s} Rs{n_net:>+10,.0f}   Rs{s_net:>+10,.0f}   Rs{tot:>+10,.0f}")
    print(f"  {'Win Rate':20s} {n_w/len(df_n)*100:>7.1f}%     {s_w/len(df_s)*100:>7.1f}%     {(n_w+s_w)/(len(df_n)+len(df_s))*100:>7.1f}%")
    print(f"  {'Profit Factor':20s} {n_gp/n_gl:>7.2f}     {s_gp/s_gl:>7.2f}     {(n_gp+s_gp)/(n_gl+s_gl):>7.2f}")
    print(f"  {'Max DD':20s} {'':>11s} {'':>11s} Rs{mdd:>+10,.0f} ({mdd/(CAP*2)*100:.1f}%)")
    print(f"  {'Sharpe (daily)':20s} {'':>11s} {'':>11s} {sharpe:>10.2f}")
    print(f"  {'Return':20s} {'':>11s} {'':>11s} {tot/(CAP*2)*100:+6.1f}% (CAGR {cagr:+.1f}%)")
    
    # Best/worst days
    top = comb.nlargest(1, "total").iloc[0]
    bot = comb.nsmallest(1, "total").iloc[0]
    print(f"  {'Best Day':20s}: {top['date']} (Rs{top['total']:+,.0f})")
    print(f"  {'Worst Day':20s}: {bot['date']} (Rs{bot['total']:+,.0f})")
    
    # Monthly analysis
    comb["month"] = pd.to_datetime(comb["date"]).dt.to_period("M")
    monthly = comb.groupby("month")["total"].sum()
    win_months = (monthly > 0).sum()
    total_months = len(monthly)
    print(f"  {'Winning Months':20s}: {win_months}/{total_months} ({win_months/total_months*100:.0f}%)")
    
    # Consecutive losses
    comb["win"] = comb["total"] > 0
    max_cl = 0; cur_cl = 0
    for w in comb["win"]:
        if not w: cur_cl += 1
        else: cur_cl = 0
        max_cl = max(max_cl, cur_cl)
    print(f"  {'Max Cons Loss Days':20s}: {max_cl}")
    
    return {"net":tot, "mdd":mdd, "trades":len(df_n)+len(df_s)}

# ── Full period ──
print("=" * 75)
print("OR RAW+FIXTP - WITH March 2020")
print("=" * 75)
full = analyze(nifty, sensex, "FULL PERIOD")

# ── Exclude March 2020 ──
print(f"\n{'='*75}")
print("OR RAW+FIXTP - EXCLUDING March 2020")
print(f"{'='*75}")
mar2020 = pd.Timestamp("2020-03-01").date()
apr2020 = pd.Timestamp("2020-03-31").date()
n_ex = nifty[~((nifty["date"] >= mar2020) & (nifty["date"] <= apr2020))]
s_ex = sensex[~((sensex["date"] >= mar2020) & (sensex["date"] <= apr2020))]
ex = analyze(n_ex, s_ex, "EXCLUDED March 2020")

# ── Exclude ONLY the single best day (2020-03-13) ──
print(f"\n{'='*75}")
print("OR RAW+FIXTP - EXCLUDING ONLY 2020-03-13")
print(f"{'='*75}")
mar13 = pd.Timestamp("2020-03-13").date()
n_mar13 = nifty[nifty["date"] != mar13]
s_mar13 = sensex[sensex["date"] != mar13]
ex13 = analyze(n_mar13, s_mar13, "EXCLUDED 2020-03-13 only")

# ── Exclude ALL March 2020 AND April 2020 (entire COVID crash/recovery) ──
print(f"\n{'='*75}")
print("OR RAW+FIXTP - EXCLUDING March-April 2020")
print(f"{'='*75}")
may2020 = pd.Timestamp("2020-05-01").date()
n_covid = nifty[~((nifty["date"] >= mar2020) & (nifty["date"] < may2020))]
s_covid = sensex[~((sensex["date"] >= mar2020) & (sensex["date"] < may2020))]
ex_covid = analyze(n_covid, s_covid, "EXCLUDED Mar-Apr 2020")

print(f"\n{'='*75}")
print("SUMMARY TABLE")
print(f"{'='*75}")
print(f"  {'Scenario':<35s} {'Net P&L':>12s} {'MDD':>12s} {'Trades':>8s}")
print(f"  {'-'*67}")
for label, d in [("Full Period", full), ("Excl Mar 2020", ex), ("Excl 2020-03-13 only", ex13), ("Excl Mar-Apr 2020", ex_covid)]:
    print(f"  {label:<35s} Rs{d['net']:>+8,.0f}   Rs{d['mdd']:>+8,.0f}   {d['trades']:>5d}")

print(f"\n  KEY INSIGHT:")
mar13_profit = full["net"] - ex13["net"]
print(f"  2020-03-13 alone contributed +Rs{mar13_profit:+,.0f} of Rs{full['net']:+,.0f} total ({mar13_profit/full['net']*100:.0f}%)")
print(f"  Without this single day: Rs{ex13['net']:+,.0f} on Rs{200000:,} capital ({ex13['net']/200000*100:.1f}%)")
print(f"  Without all COVID volatility: Rs{ex_covid['net']:+,.0f} ({ex_covid['net']/200000*100:.1f}%)")
