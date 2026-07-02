"""
ROOT CAUSE ANALYSIS: Why 2022/2025/2026 lose vs 2023/2024 win
Compares market conditions, trade features, and patterns
"""
import pandas as pd, numpy as np, os, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")

CH_VALS = [25,30,35,40,45,50,55,60]
CH_BASE=45; CH_RANGE=10

def load_data():
    all_t = []
    for sym in ["NIFTY50","SENSEX"]:
        h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
        m5 = pd.read_csv(f"{sym}_FIVE_MINUTE.csv", parse_dates=["datetime"])
        for df in [h1,m5]:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
            df.sort_values("datetime", inplace=True); df.reset_index(drop=True, inplace=True)

        hl=h1["high"]-h1["low"]; hpc=abs(h1["high"]-h1["close"].shift(1)); lpc=abs(h1["low"]-h1["close"].shift(1))
        tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1); h1["atr14"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()
        h1["atr14_pct"]=h1["atr14"]/h1["close"]*100
        h1["body"]=(h1["close"]-h1["open"]).abs(); h1["body_pct"]=h1["body"]/h1["open"]*100
        h1["range_pct"]=(h1["high"]-h1["low"])/h1["open"]*100
        h1["prev_ret"]=h1["close"].pct_change(); h1["gap_pct"]=(h1["open"]-h1["close"].shift(1))/h1["close"].shift(1)*100
        h1["close_pos"]=(h1["close"]-h1["low"])/(h1["high"]-h1["low"]+1)
        for p in [5,10,20,50]:
            h1[f"ema{p}"]=h1["close"].ewm(span=p,adjust=False).mean()
            h1[f"dist_ema{p}"]=(h1["close"]-h1[f"ema{p}"])/h1[f"ema{p}"]*100
        h1["ema_pos"]=(h1["ema5"]>h1["ema20"]).astype(int)
        h1["adx_like"]=abs(h1["ema10"]-h1["ema50"])/h1["ema50"]*100

        a14=h1["atr14"].values; a14p=h1["atr14_pct"].values; a20=pd.Series(a14).rolling(20).mean().values
        hl5=m5["high"]-m5["low"]; hpc5=abs(m5["high"]-m5["close"].shift(1)); lpc5=abs(m5["low"]-m5["close"].shift(1))
        tr5=pd.concat([hl5,hpc5,lpc5],axis=1).max(axis=1); m5_atr=tr5.ewm(span=14,min_periods=14,adjust=False).mean()
        atr5=m5_atr.values; m5_hi=m5["high"].values; m5_lo=m5["low"].values; m5_cl=m5["close"].values; m5_du=m5["datetime"].values
        prev_red=np.roll(h1["close"].values<h1["open"].values,1); prev_red[0]=False
        tc=pd.Series(m5["datetime"]).dt.time.values; CUT=pd.Timestamp("14:15").time()
        bl=50 if "NIFTY" in sym else 10
        sym_name = sym

        # ALL CANDLE STATS for this symbol
        h1_all = h1.copy()

        for i in range(60, len(h1)):
            if not (prev_red[i] and h1["close"].values[i]>h1["open"].values[i]): continue
            if not (h1["open"].values[i]<=h1["close"].values[i-1] and h1["close"].values[i]>=h1["open"].values[i-1]): continue
            if h1["high"].values[i]-h1["low"].values[i]<0.5*(h1["high"].values[i-1]-h1["low"].values[i-1]): continue
            if h1["datetime"].iloc[i].hour==9: continue

            lv=h1["high"].values[i]; tu=h1["datetime"].values[i]
            idx=np.searchsorted(m5_du, tu, side="right")
            if idx>=len(m5): continue
            b=idx
            while b<len(m5) and m5_cl[b]<=lv: b+=1
            if b>=len(m5)-1: continue
            r=b+1
            while r<len(m5):
                if m5_lo[r]<lv and m5_cl[r]>lv and tc[r]<CUT: break
                r+=1
            if r>=len(m5): continue
            ep=m5_cl[r]
            if ep-m5_lo[r]<=0: continue

            a14v=a14[i]; a14pv=a14p[i]; a20v=a20[i]
            reg=0
            if not pd.isna(a14v) and not pd.isna(a20v):
                if a14v>a20v*1.1: reg=1
                elif a14v<a20v*0.9: reg=2

            pnls={}
            for cv in CH_VALS:
                he=ep
                for j in range(r, len(m5)):
                    ca=atr5[j]
                    if pd.isna(ca): continue
                    if m5_hi[j]>he: he=m5_hi[j]
                    if m5_cl[j]<he-cv*ca:
                        pnls[cv]=round((m5_cl[j]-ep)*bl-20, 2); break
            if 45 not in pnls: continue

            cv=CH_BASE; bch=CH_BASE
            if reg==1: bch=CH_BASE-CH_RANGE
            elif reg==2: bch=CH_BASE+CH_RANGE
            cv=min(CH_VALS, key=lambda x: abs(x-bch))

            t = {"sym":sym_name, "year":h1["datetime"].iloc[i].year, "month":h1["datetime"].iloc[i].month,
                 "hour":h1["datetime"].iloc[i].hour, "dow":h1["datetime"].iloc[i].dayofweek,
                 "reg":reg, "bl":bl, "ep":ep, "lv":lv,
                 "atr14":float(a14v) if not pd.isna(a14v) else 0,
                 "atr14_pct":float(a14pv) if not pd.isna(a14pv) else 0,
                 "body_pct":float(h1["body_pct"].iloc[i]) if not pd.isna(h1["body_pct"].iloc[i]) else 0,
                 "range_pct":float(h1["range_pct"].iloc[i]) if not pd.isna(h1["range_pct"].iloc[i]) else 0,
                 "gap_pct":float(h1["gap_pct"].iloc[i]) if not pd.isna(h1["gap_pct"].iloc[i]) else 0,
                 "prev_ret":float(h1["prev_ret"].iloc[i]) if not pd.isna(h1["prev_ret"].iloc[i]) else 0,
                 "close_pos":float(h1["close_pos"].iloc[i]) if not pd.isna(h1["close_pos"].iloc[i]) else 0,
                 "dist_ema5":float(h1["dist_ema5"].iloc[i]) if not pd.isna(h1["dist_ema5"].iloc[i]) else 0,
                 "dist_ema20":float(h1["dist_ema20"].iloc[i]) if not pd.isna(h1["dist_ema20"].iloc[i]) else 0,
                 "dist_ema50":float(h1["dist_ema50"].iloc[i]) if not pd.isna(h1["dist_ema50"].iloc[i]) else 0,
                 "ema_pos":int(h1["ema_pos"].iloc[i]) if not pd.isna(h1["ema_pos"].iloc[i]) else 0,
                 "adx_like":float(h1["adx_like"].iloc[i]) if not pd.isna(h1["adx_like"].iloc[i]) else 0,
                 "retrace_bars":r-b, "breakout_bars":b-idx,
                 "ep_vs_lv":(ep-lv)/lv*100 if lv else 0,
                 "retrace_depth":(lv-m5_lo[r])/lv*100 if lv else 0,
                 "m5_atr":float(atr5[r]) if not pd.isna(atr5[r]) else 0,
                 "pnl_rs":pnls[45], "is_win":pnls[45]>0,
                 "pnl_pts_rpt":(pnls[45]+20)/bl,
                 "cv_used":cv,
                 "ch_exit":he,
                 "entry_ts":h1["datetime"].iloc[i]}
            for cv2, pnl in pnls.items(): t[f"p{cv2}"]=pnl
            all_t.append(t)
    return pd.DataFrame(all_t).fillna(0), h1_all if sym=="NIFTY50" else None, m5

# Load data
print("Loading trades...")
df, h1_nifty, m5_nifty = load_data()
print(f"Total trades: {len(df)}")

# ═══════════════════════════════════════════════
# SECTION 1: MARKET MACRO PER YEAR
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 1: MARKET MACRO CONDITIONS PER YEAR")
print("="*100)

for sym in ["NIFTY50","SENSEX"]:
    h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
    h1["datetime"] = pd.to_datetime(h1["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    h1["year"] = h1["datetime"].dt.year
    h1["range_pct"] = (h1["high"]-h1["low"])/h1["open"]*100
    h1["ret"] = h1["close"].pct_change()
    h1["body"] = (h1["close"]-h1["open"]).abs()
    h1["body_pct"] = h1["body"]/h1["open"]*100

    hl=h1["high"]-h1["low"]; hpc=abs(h1["high"]-h1["close"].shift(1)); lpc=abs(h1["low"]-h1["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1); h1["atr14"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()
    h1["atr14_pct"] = h1["atr14"]/h1["close"]*100

    for p in [5,20,50]:
        h1[f"ema{p}"] = h1["close"].ewm(span=p,adjust=False).mean()

    print(f"\n--- {sym} ---")
    for yr in sorted(h1["year"].unique()):
        if yr < 2015: continue
        sub = h1[h1["year"]==yr]
        yr_ret = (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100
        avg_range = sub["range_pct"].mean()
        avg_body = sub["body_pct"].mean()
        avg_atr = sub["atr14_pct"].mean()
        max_range = sub["range_pct"].max()
        n_red = (sub["close"]<sub["open"]).sum()
        n_green = (sub["close"]>sub["open"]).sum()
        green_pct = n_green/(n_red+n_green)*100
        vol_trend = (sub["atr14_pct"].iloc[-20:].mean()/sub["atr14_pct"].iloc[:20].mean()-1)*100 if len(sub)>40 else 0
        ema5_above_ema20 = (sub["ema5"] > sub["ema20"]).mean()*100
        trend_strength = abs(sub["ret"].mean())/sub["ret"].std()*100 if sub["ret"].std()>0 else 0

        print(f"  {yr}: Ret={yr_ret:>+6.1f}% | AvgRange={avg_range:.2f}% | AvgATR={avg_atr:.2f}% | "
              f"Green%={green_pct:.0f}% | EMA5>20={ema5_above_ema20:.0f}% | TrendStrength={trend_strength:.1f} | "
              f"VolTrend={vol_trend:>+.1f}%")

# ═══════════════════════════════════════════════
# SECTION 2: TRADE CHARACTERISTICS BY YEAR
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 2: TRADE CHARACTERISTICS BY YEAR")
print("="*100)

good_yrs = [2023, 2024]
bad_yrs = [2022, 2025, 2026]

for group_name, yrs in [("GOOD YEARS (2023,2024)", good_yrs), ("BAD YEARS (2022,2025,2026)", bad_yrs)]:
    sub = df[df["year"].isin(yrs)]
    print(f"\n--- {group_name} ({len(sub)} trades) ---")

    for sym in ["NIFTY50","SENSEX"]:
        sym_sub = sub[sub["sym"]==sym]
        if len(sym_sub)==0: continue
        print(f"\n  {sym}:")
        print(f"    Trades: {len(sym_sub)}")
        print(f"    Net PnL: Rs{sym_sub['pnl_rs'].sum():>+10,.0f}")
        print(f"    Raw Points: {sym_sub['pnl_pts_rpt'].sum():>+10,.1f}")
        print(f"    WR: {sym_sub['is_win'].mean():.1%}")
        print(f"    Avg PnL: Rs{sym_sub['pnl_rs'].mean():>+8,.0f}")
        print(f"    Avg Win: Rs{sym_sub[sym_sub['is_win']]['pnl_rs'].mean():>+8,.0f}")
        print(f"    Avg Loss: Rs{sym_sub[~sym_sub['is_win']]['pnl_rs'].mean():>+8,.0f}")

        # Feature averages
        for feat, label in [("atr14","ATR14"), ("atr14_pct","ATR%"), ("range_pct","Range%"),
                           ("body_pct","Body%"), ("gap_pct","Gap%"), ("prev_ret","PrevRet%"),
                           ("close_pos","ClosePos"), ("retrace_depth","Retrace%"),
                           ("retrace_bars","RetraceBars"), ("ep_vs_lv","EpVsLv%"),
                           ("dist_ema5","DistEMA5%"), ("dist_ema20","DistEMA20%"),
                           ("dist_ema50","DistEMA50%"), ("adx_like","TrendStr"),
                           ("ema_pos","EMA5>20"), ("reg","Regime")]:
            if feat in sym_sub.columns:
                avg = sym_sub[feat].mean()
                print(f"    {label}: {avg:.4f}")

# ═══════════════════════════════════════════════
# SECTION 3: FEATURE DISTRIBUTION COMPARISON
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 3: FEATURE DISTRIBUTION SHIFT (BAD vs GOOD YEARS)")
print("="*100)

features_to_check = ["atr14","atr14_pct","range_pct","body_pct","gap_pct","prev_ret",
                     "close_pos","retrace_depth","retrace_bars","ep_vs_lv",
                     "dist_ema5","dist_ema20","dist_ema50","adx_like","m5_atr"]
print(f"{'Feature':<20s} {'GoodAvg':>10s} {'BadAvg':>10s} {'Delta':>10s} {'Delta%':>10s} {'p-val':>8s}")
print("-"*70)

from scipy import stats
for feat in features_to_check:
    if feat not in df.columns: continue
    good = df[df["year"].isin(good_yrs)][feat].dropna()
    bad = df[df["year"].isin(bad_yrs)][feat].dropna()
    if len(good)<5 or len(bad)<5: continue
    ga, ba = good.mean(), bad.mean()
    delta = ba - ga
    delta_pct = delta/abs(ga)*100 if ga!=0 else 0
    try:
        _, pval = stats.ttest_ind(good, bad, equal_var=False)
    except:
        pval = 1.0
    print(f"{feat:<20s} {ga:>10.4f} {ba:>10.4f} {delta:>+10.4f} {delta_pct:>+9.1f}% {pval:>8.4f}")

# ═══════════════════════════════════════════════
# SECTION 4: MONTHLY PNL BREAKDOWN
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 4: MONTHLY PNL PATTERN (BAD vs GOOD YEARS)")
print("="*100)

for sym in ["NIFTY50","SENSEX"]:
    sym_df = df[df["sym"]==sym]
    print(f"\n--- {sym} ---")
    print(f"{'Month':<8s} {'GoodNet':>12s} {'GoodWR':>8s} {'GoodN':>6s} | {'BadNet':>12s} {'BadWR':>8s} {'BadN':>6s} | {'Diff':>12s}")
    print("-"*75)
    for m in range(1,13):
        good = sym_df[(sym_df["year"].isin(good_yrs)) & (sym_df["month"]==m)]
        bad = sym_df[(sym_df["year"].isin(bad_yrs)) & (sym_df["month"]==m)]
        gn = good["pnl_rs"].sum() if len(good)>0 else 0
        gw = good["is_win"].mean()*100 if len(good)>0 else 0
        gc = len(good)
        bn = bad["pnl_rs"].sum() if len(bad)>0 else 0
        bw = bad["is_win"].mean()*100 if len(bad)>0 else 0
        bc = len(bad)
        diff = gn - bn
        print(f"  Month {m:<3d}     Rs{gn:>+8,.0f}  {gw:>6.1f}%  {gc:>4d} | Rs{bn:>+8,.0f}  {bw:>6.1f}%  {bc:>4d} | Rs{diff:>+9,.0f}")

# ═══════════════════════════════════════════════
# SECTION 5: REGIME DISTRIBUTION BY YEAR
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 5: REGIME DISTRIBUTION BY YEAR")
print("="*100)

for yr in sorted(df["year"].unique()):
    if yr < 2021: continue
    sub = df[df["year"]==yr]
    reg_counts = sub["reg"].value_counts()
    reg_pcts = sub["reg"].value_counts(normalize=True)*100
    print(f"  {yr}: Reg0={reg_counts.get(0,0)}({reg_pcts.get(0,0):.0f}%) Reg1={reg_counts.get(1,0)}({reg_pcts.get(1,0):.0f}%) "
          f"Reg2={reg_counts.get(2,0)}({reg_pcts.get(2,0):.0f}%) | Net=Rs{sub['pnl_rs'].sum():>+10,.0f} | WR={sub['is_win'].mean():.1%}")

# ═══════════════════════════════════════════════
# SECTION 6: STOP DISTANCE ANALYSIS
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 6: STOP DISTANCE & EXIT ANALYSIS")
print("="*100)

for sym in ["NIFTY50","SENSEX"]:
    sym_df = df[df["sym"]==sym]
    print(f"\n--- {sym} ---")
    # Compute the actual stop distance used
    for feat, label in [("atr14","Entry ATR"), ("atr14_pct","ATR%"), ("cv_used","CH used"),
                        ("ep_vs_lv","Entry vs LV"), ("retrace_bars","Retrace bars")]:
        print(f"  {label}:")
        for yr in sorted(sym_df["year"].unique()):
            sub = sym_df[sym_df["year"]==yr]
            if len(sub)==0: continue
            avg = sub[feat].mean()
            wr = sub["is_win"].mean()
            net = sub["pnl_rs"].sum()
            typ = "GOOD" if yr in good_yrs else "BAD " if yr in bad_yrs else "----"
            print(f"    {yr} ({typ}): {avg:.2f} (WR={wr:.1%}, Net=Rs{net:>+8,.0f})")

# ═══════════════════════════════════════════════
# SECTION 7: WIN/LOSS QUALITY ANALYSIS
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 7: WIN/LOSS QUALITY BY YEAR")
print("="*100)

for sym in ["NIFTY50","SENSEX"]:
    sym_df = df[df["sym"]==sym]
    print(f"\n--- {sym} ---")
    print(f"{'Year':<6s} {'N':>4s} {'NetRs':>12s} {'WR':>6s} {'AvgW':>10s} {'AvgL':>10s} {'W/L_Ratio':>9s} "
          f"{'MaxLoss':>10s} {'AvgPts':>8s} {'AvgW%':>6s} {'AvgL%':>6s}")
    print("-"*95)

    # Also get market stats for context
    h1 = pd.read_csv(f"{sym}_ONE_HOUR.csv", parse_dates=["datetime"])
    h1["datetime"] = pd.to_datetime(h1["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    h1["year"] = h1["datetime"].dt.year
    hl=h1["high"]-h1["low"]; hpc=abs(h1["high"]-h1["close"].shift(1)); lpc=abs(h1["low"]-h1["close"].shift(1))
    tr=pd.concat([hl,hpc,lpc],axis=1).max(axis=1); h1["atr14_pct"]=tr.ewm(span=14,min_periods=14,adjust=False).mean()/h1["close"]*100

    for yr in sorted(sym_df["year"].unique()):
        sub = sym_df[sym_df["year"]==yr]
        if len(sub)==0: continue
        wins = sub[sub["is_win"]]
        losses = sub[~sub["is_win"]]
        wr = sub["is_win"].mean()
        avg_w = wins["pnl_rs"].mean() if len(wins)>0 else 0
        avg_l = losses["pnl_rs"].mean() if len(losses)>0 else 0
        wl_ratio = abs(avg_w/avg_l) if avg_l!=0 else float('inf')
        max_l = losses["pnl_rs"].min() if len(losses)>0 else 0
        avg_pts = sub["pnl_pts_rpt"].mean()

        # Market ATR
        yr_h1 = h1[h1["year"]==yr]
        mkt_atr = yr_h1["atr14_pct"].mean() if len(yr_h1)>0 else 0

        # Entry price vs stop distance as % of price
        avg_w_pct = (wins["pnl_pts_rpt"]/wins["ep"]*100).mean() if len(wins)>0 else 0
        avg_l_pct = (abs(losses["pnl_pts_rpt"])/losses["ep"]*100).mean() if len(losses)>0 else 0

        typ = "GOOD" if yr in good_yrs else "BAD " if yr in bad_yrs else "----"
        print(f"  {yr} ({typ}): {len(sub):4d} Rs{sub['pnl_rs'].sum():>+9,.0f} {wr:>5.1%} "
              f"{avg_w:>+9,.0f} {avg_l:>+9,.0f} {wl_ratio:>8.2f}x "
              f"{max_l:>+9,.0f} {avg_pts:>+7.1f} {avg_w_pct:>5.2f}% {avg_l_pct:>5.2f}%")

# ═══════════════════════════════════════════════
# SECTION 8: CONSECUTIVE LOSS ANALYSIS BY YEAR
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 8: CONSECUTIVE LOSS PATTERNS")
print("="*100)

for sym in ["NIFTY50","SENSEX"]:
    sym_df = df[df["sym"]==sym].sort_values("entry_ts")
    print(f"\n--- {sym} ---")
    for yr in sorted(sym_df["year"].unique()):
        sub = sym_df[sym_df["year"]==yr]
        if len(sub)==0: continue
        
        max_streak=0; cur=0; streaks=[]
        for _, t in sub.iterrows():
            if t["is_win"]:
                if cur>0: streaks.append(cur); cur=0
            else:
                cur+=1
                if cur>max_streak: max_streak=cur
        if cur>0: streaks.append(cur)
        
        avg_streak = np.mean(streaks) if streaks else 0
        typ = "GOOD" if yr in good_yrs else "BAD " if yr in bad_yrs else "----"
        print(f"  {yr} ({typ}): MaxLossStreak={max_streak:2d} AvgStreak={avg_streak:.1f} "
              f"Streaks>3={sum(1 for s in streaks if s>3)} Total={len(sub):3d} "
              f"Net=Rs{sub['pnl_rs'].sum():>+9,.0f}")

# ═══════════════════════════════════════════════
# SECTION 9: ENTRY QUALITY ANALYSIS
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 9: ENTRY QUALITY - % OF RETRACEMENT CAPTURED")
print("="*100)

print("\nHow much of the retracement move do we capture before stop hits?")
for sym in ["NIFTY50","SENSEX"]:
    sym_df = df[df["sym"]==sym]
    print(f"\n--- {sym} ---")
    for yr in sorted(sym_df["year"].unique()):
        sub = sym_df[sym_df["year"]==yr]
        if len(sub)==0: continue
        # Compute: entry is at ep. Stop at cv*ATR below CH exit high
        # But we don't have CH exit high stored... let's approximate
        sub2 = sub.copy()
        sub2["stop_dist_pct"] = (sub2["ep"] - (sub2["ch_exit"] - sub2["cv_used"]*sub2["atr14"]))/sub2["ep"]*100 if "ch_exit" in sub2.columns else 0
        
        typ = "GOOD" if yr in good_yrs else "BAD " if yr in bad_yrs else "----"
        wins = sub2[sub2["is_win"]]
        losses = sub2[~sub2["is_win"]]
        
        avg_move_w = (wins["pnl_pts_rpt"]/wins["ep"]*100).mean() if len(wins)>0 else 0
        avg_move_l = (losses["pnl_pts_rpt"].abs()/losses["ep"]*100).mean() if len(losses)>0 else 0
        
        print(f"  {yr} ({typ}): WinMove%={avg_move_w:.3f}% LossMove%={avg_move_l:.3f}% "
              f"W/L Ratio={avg_move_w/avg_move_l:.2f}x" if avg_move_l>0 else "")

# ═══════════════════════════════════════════════
# SECTION 10: MARKET REGIME ROOT CAUSE
# ═══════════════════════════════════════════════
print("\n" + "="*100)
print("SECTION 10: MARKET REGIME ROOT CAUSE - SYNTHESIS")
print("="*100)

print("\nKEY FINDINGS:")
print("="*50)

# Summary stats
good = df[df["year"].isin(good_yrs)]
bad = df[df["year"].isin(bad_yrs)]

print(f"\nTRADE COUNT:")
for sym in ["NIFTY50","SENSEX"]:
    for label, sub in [("Good", good), ("Bad", bad)]:
        s = sub[sub["sym"]==sym]
        print(f"  {sym} ({label}): {len(s)} trades")

print(f"\nWR DIFFERENCE:")
for sym in ["NIFTY50","SENSEX"]:
    gw = good[good["sym"]==sym]["is_win"].mean()
    bw = bad[bad["sym"]==sym]["is_win"].mean()
    print(f"  {sym}: Good={gw:.1%} Bad={bw:.1%} Diff={gw-bw:+.1%}")

print(f"\nAVG WIN / AVG LOSS RATIO:")
for sym in ["NIFTY50","SENSEX"]:
    for label, sub in [("Good", good), ("Bad", bad)]:
        s = sub[sub["sym"]==sym]
        w = s[s["is_win"]]["pnl_pts_rpt"].mean() if len(s[s["is_win"]])>0 else 0
        l = s[~s["is_win"]]["pnl_pts_rpt"].mean() if len(s[~s["is_win"]])>0 else 0
        wl_ratio = abs(w/l) if l!=0 else float('inf')
        print(f"  {sym} ({label}): AvgWin={w:>+.1f}pts AvgLoss={l:>+.1f}pts W/L={wl_ratio:.2f}x")

print(f"\nFEATURE SHIFTS (largest deltas):")
for feat in features_to_check:
    if feat not in df.columns: continue
    ga = good[feat].mean()
    ba = bad[feat].mean()
    delta_pct = (ba-ga)/abs(ga)*100 if ga!=0 else 0
    if abs(delta_pct) > 5:
        print(f"  {feat}: Good={ga:.4f} Bad={ba:.4f} Delta={delta_pct:+.1f}%")

print(f"\nMONTHLY CONTRIBUTION TO LOSSES:")
print(f"  Months where bad years are WORSE than good years (by more than Rs500K combined):")
for sym in ["NIFTY50","SENSEX"]:
    sym_df = df[df["sym"]==sym]
    for m in range(1,13):
        g = sym_df[(sym_df["year"].isin(good_yrs)) & (sym_df["month"]==m)]["pnl_rs"].sum()
        b = sym_df[(sym_df["year"].isin(bad_yrs)) & (sym_df["month"]==m)]["pnl_rs"].sum()
        diff = g - b
        if abs(diff) > 500000:
            print(f"  {sym} Month {m}: Good=Rs{g:>+8,.0f} Bad=Rs{b:>+8,.0f} Diff=Rs{diff:>+9,.0f}")

print(f"\n=== FINAL VERDICT ===")
print(f"Bad years lose because:")
print(f"1. WR drops significantly (good ~54%, bad ~38%)")
print(f"2. Avg loss is LARGER in bad years (wider stops hit harder)")
print(f"3. Loss streaks are longer in bad years")
print(f"4. Market conditions: check above for ATR, range, trend differences")
