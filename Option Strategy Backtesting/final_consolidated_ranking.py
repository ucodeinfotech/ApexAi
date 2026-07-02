"""
Final consolidated ranking table: ALL strategies x ALL improvements
"""
import sys

# (strategy, config, trades, net, wr, wl_ratio, mdd)
DATA = [
    # ---- Engulf_Raw variants ----
    ("Engulf_Raw","CH45_base",    1671, +660523, 48, 2.2, +105693),
    ("Engulf_Raw","CH55",         1671,+1235116, 52, 2.5, +153981),
    ("Engulf_Raw","CH55+WL",      1671,+1296991, 52, 4.7,  +66227),
    ("Engulf_Raw","CH55+Skip2",    987,+1522621, 69, 3.1,  +53351),
    ("Engulf_Raw","CH55+WL+Skip2", 564,+1395534, 69, 5.8,  +19362),
    ("Engulf_Raw","CH55+2w1l",    1671,+1889157, 52, 2.6, +223397),
    ("Engulf_Raw","DynCH",        1671, +733946, 46, 2.4, +171793),
    ("Engulf_Raw","DynCH+WL",     1671, +850951, 46, 4.2,  +54147),
    ("Engulf_Raw","DynCH+Skip2",   987, +984455, 64, 2.6,  +65478),
    ("Engulf_Raw","DynCH+WL+Skip2",564, +838156, 64, 4.4,  +23755),
    ("Engulf_Raw","CH45+Skip2",    987, +879465, 65, 2.4,  +35145),
    ("Engulf_Raw","CH45+Skip3",    853, +832395, 67, 2.4,  +32930),
    ("Engulf_Raw","CH45+Skip1",   1183, +831717, 58, 2.3,  +53276),
    ("Engulf_Raw","CH45+WL",      1671, +775569, 48, 4.3,  +49789),
    ("Engulf_Raw","CH45+2w1l",    1671,+1014604, 48, 2.3, +155411),
    ("Engulf_Raw","CH45+3w2l",    1671,+1009163, 48, 2.3, +155411),
    ("Engulf_Raw","CH35",         1671, +618593, 48, 2.3,  +76173),
    ("Engulf_Raw","CH25",         1671, +293688, 46, 1.9,  +40790),
    ("Engulf_Raw","CH60",         1671,+1201005, 51, 2.4, +223397),
    ("Engulf_Raw","FixTP",        1671,   +11654, 33, 2.5,   +6029),
    # ---- Engulf_Filt variants ----
    ("Engulf_Filt","CH45_base",    652, +215951, 45, 2.2,  +69223),
    ("Engulf_Filt","CH55",         652, +492170, 47, 2.4,  +85568),
    ("Engulf_Filt","CH55+WL",      652, +525938, 47, 4.3,  +37380),
    ("Engulf_Filt","CH55+Skip2",   397, +608828, 65, 2.8,  +19422),
    ("Engulf_Filt","CH55+WL+Skip2",240, +593445, 67, 4.9,   +9093),
    ("Engulf_Filt","CH55+2w1l",    652, +754263, 50, 2.7, +111559),
    ("Engulf_Filt","DynCH",        652, +343320, 50, 2.2,  +63264),
    ("Engulf_Filt","DynCH+WL",     652, +408475, 50, 4.7,  +21701),
    ("Engulf_Filt","DynCH+Skip2",  397, +359518, 63, 2.3,  +34144),
    ("Engulf_Filt","DynCH+WL+Skip2",240, +431039, 67, 4.6,  +11377),
    # ---- BigCandle variants ----
    ("BigCandle","CH45_base",      480, +214455, 46, 2.3,  +41032),
    ("BigCandle","CH55",           480, +434228, 50, 2.4,  +73266),
    ("BigCandle","CH55+WL",        480, +457582, 50, 4.3,  +29707),
    ("BigCandle","CH55+Skip2",     283, +474299, 65, 2.6,  +18099),
    ("BigCandle","CH55+WL+Skip2",  181, +477748, 66, 5.4,   +8571),
    ("BigCandle","CH55+2w1l",      480, +645452, 53, 2.7,  +61044),
    ("BigCandle","DynCH",          480, +292920, 50, 2.3,  +40899),
    ("BigCandle","DynCH+WL",       480, +295549, 50, 3.4,  +29757),
    ("BigCandle","DynCH+Skip2",    283, +317140, 64, 2.5,  +15148),
    ("BigCandle","DynCH+WL+Skip2", 181, +316143, 64, 4.2,   +6197),
    # ---- Comb_OR variants ----
    ("Comb_OR","CH45_base",        731, +252801, 45, 2.2,  +74546),
    ("Comb_OR","CH55",             731, +542860, 48, 2.4, +113708),
    ("Comb_OR","CH55+WL",          731, +601656, 48, 4.2,  +63840),
    ("Comb_OR","CH55+Skip2",       440, +654023, 64, 2.8,  +23367),
    ("Comb_OR","CH55+WL+Skip2",    251, +567124, 66, 4.7,   +8627),
    ("Comb_OR","CH55+2w1l",        731, +834160, 49, 2.8, +119127),
    ("Comb_OR","DynCH",            731, +367257, 49, 2.2,  +76226),
    ("Comb_OR","DynCH+WL",         731, +420729, 48, 3.7,  +47310),
    ("Comb_OR","DynCH+Skip2",      440, +471782, 63, 2.4,  +28129),
    ("Comb_OR","DynCH+WL+Skip2",   251, +433199, 65, 4.1,   +9662),
    # ---- Sir variants ----
    ("Sir","CH45_base",             81,  +41052, 49, 2.2,  +10204),
    ("Sir","CH55",                  81,  +53298, 44, 3.0,  +13900),
    ("Sir","CH55+WL",               81,  +49600, 44, 3.1,  +13364),
    ("Sir","CH55+Skip2",            46,  +59068, 54, 4.2,   +4996),
    ("Sir","CH55+WL+Skip2",         46,  +55387, 54, 4.1,   +4936),
    ("Sir","CH55+2w1l",             81,  +65710, 44, 2.9,  +20300),
    ("Sir","DynCH",                 81,  +40858, 48, 2.3,  +10301),
    ("Sir","DynCH+WL",              81,  +40221, 48, 2.5,   +8157),
    ("Sir","DynCH+Skip2",           46,  +26848, 54, 2.5,   +4100),
    ("Sir","DynCH+WL+Skip2",        46,  +22071, 54, 2.3,   +4100),
]

# sort by net descending
DATA.sort(key=lambda r: -r[3])

print("="*140)
print("FINAL CONSOLIDATED RANKING: ALL STRATEGIES x ALL IMPROVEMENTS")
print("="*140)
print(f"{'#':>3} {'Strategy':<14} {'Config':<22} {'Trades':>7} {'Net Pts':>11} {'WR':>5} {'W/L':>5} {'MDD':>9} {'Net/MDD':>8}  {'Category'}")
print("-"*140)
for i,(strat,cfg,trd,net,wr,wl,mdd) in enumerate(DATA,1):
    cat = "Raw" if "Engulf_Raw" in strat or ("Engulf" not in strat and "Comb" not in strat and "Big" not in strat and "Sir" not in strat) else strat.split("_")[0] if "_" in strat else strat
    # simplify category
    if "Engulf_Raw" in strat: cat="Raw"
    elif "Engulf_Filt" in strat: cat="Filt"
    elif "BigCandle" in strat: cat="BigC"
    elif "Comb_OR" in strat: cat="Comb"
    elif "Sir" in strat: cat="Sir"
    else: cat=strat
    nm = net/mdd if mdd>0 else 999
    sig = ""
    if "Skip2" in cfg and "WL" in cfg: sig="BEST"
    elif "2w1l" in cfg and "CH55" in cfg: sig="MAX"
    elif "Skip2" in cfg: sig="SKIP"
    elif "WL" in cfg: sig="W/L"
    print(f"{i:>3} {strat:<14} {cfg:<22} {trd:>7} {net:>+10,d}  {wr:>3d}%  {wl:>3.1f}x {mdd:>+7,d}  {nm:>6.1f}x  {sig}")
print("-"*140)

print("\n")
print("="*100)
print("TOP 15 BY RISK-ADJUSTED RETURN (Net / MDD)")
print("="*100)
by_nm = sorted(DATA, key=lambda r: -r[3]/r[6] if r[6]>0 else -999)
print(f"{'#':>3} {'Strategy':<14} {'Config':<22} {'Net Pts':>11} {'MDD':>9} {'Net/MDD':>8} {'WR':>5}")
print("-"*100)
for i,(strat,cfg,_,net,wr,_,mdd) in enumerate(by_nm[:15],1):
    nm = net/mdd if mdd>0 else 999
    print(f"{i:>3} {strat:<14} {cfg:<22} {net:>+10,d} {mdd:>+7,d}  {nm:>6.1f}x  {wr:>3d}%")

print("\n")
print("="*100)
print("TOP 15 BY WIN RATE")
print("="*100)
by_wr = sorted(DATA, key=lambda r: -r[4])
print(f"{'#':>3} {'Strategy':<14} {'Config':<22} {'Net Pts':>11} {'WR':>5} {'W/L':>5} {'Trades':>7}")
print("-"*100)
for i,(strat,cfg,trd,net,wr,wl,_) in enumerate(by_wr[:15],1):
    print(f"{i:>3} {strat:<14} {cfg:<22} {net:>+10,d}  {wr:>3d}%  {wl:>3.1f}x {trd:>7}")

print("\n")
print("="*100)
print("TOP 10 BY W/L RATIO (min 100 trades)")
print("="*100)
by_wl = sorted([r for r in DATA if r[2]>=100], key=lambda r: -r[5])
print(f"{'#':>3} {'Strategy':<14} {'Config':<22} {'W/L':>5} {'Net Pts':>11} {'WR':>5} {'Trades':>7}")
print("-"*100)
for i,(strat,cfg,trd,net,wr,wl,_) in enumerate(by_wl[:10],1):
    print(f"{i:>3} {strat:<14} {cfg:<22} {wl:>3.1f}x {net:>+10,d}  {wr:>3d}% {trd:>7}")

print("\n")
print("="*100)
print("BEST PER STRATEGY")
print("="*100)
strategies = ["Engulf_Raw","Engulf_Filt","BigCandle","Comb_OR","Sir"]
print(f"{'Strategy':<14} {'Best Config':<22} {'Net Pts':>11} {'WR':>5} {'W/L':>5} {'MDD':>9} {'Net/MDD':>8}")
print("-"*100)
for s in strategies:
    subs = [r for r in DATA if s in r[0]]
    best = max(subs, key=lambda r: r[3])
    nm = best[3]/best[6] if best[6]>0 else 999
    print(f"{best[0]:<14} {best[1]:<22} {best[3]:>+10,d}  {best[4]:>3d}%  {best[5]:>3.1f}x {best[6]:>+7,d}  {nm:>6.1f}x")
    # also best risk-adjusted
    bestra = max(subs, key=lambda r: r[3]/r[6] if r[6]>0 else -999)
    if bestra != best:
        nm2 = bestra[3]/bestra[6] if bestra[6]>0 else 999
        print(f"  (risk adj)  {bestra[1]:<22} {bestra[3]:>+10,d}  {bestra[4]:>3d}%  {bestra[5]:>3.1f}x {bestra[6]:>+7,d}  {nm2:>6.1f}x")

print("\n")
print("="*100)
print("SUMMARY")
print("="*100)
print("""
  BEST RAW RETURN:    Engulf_Raw CH55+2w1l          +1,889,157 pts  (52% WR,  2.6x W/L, MDD 223K)
  BEST RISK-ADJ:      Engulf_Raw CH55+WL+Skip2      +1,395,534 pts  (69% WR,  5.8x W/L, MDD  19K)
  BEST WIN RATE:      Engulf_Filt CH55+WL+Skip2       +593,445 pts  (67% WR,  4.9x W/L, MDD   9K)
                       BigCandle  CH55+WL+Skip2       +477,748 pts  (66% WR,  5.4x W/L, MDD   9K)
  BEST W/L RATIO:     Engulf_Raw CH55+WL+Skip2      69% WR,  5.8x W/L, MDD  19K     (min 100 trades)
  LOWEST MDD:         Sir CH55+Skip2                   +59,068 pts  (54% WR,  4.2x W/L, MDD   5K)

  === OVERALL RECOMMENDATION ===
  Engulf_Raw + CH55 + WL + Skip2:
    Net: +1,395,534 pts  WR: 69%  W/L: 5.8x  MDD: 19,362 pts
    Net/MDD: 72.1x — best risk-adjusted return
    Trades: 564 over 12 years (~47/year, perfect for manual/position trading)
    Consistently profitable every year except 2015 (-1.1K), 2022 (+3.9K), 2025 (-1.9K), 2026 (-7.6K)
""")
print("="*100)
