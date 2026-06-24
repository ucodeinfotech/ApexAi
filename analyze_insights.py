"""Comprehensive statistical & quantitative analysis of all 175 stocks"""
import os, json, warnings, statistics
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

ALL_DIRS = ["nifty50_full_history", "comprehensive_data"]

# Get all stocks
all_stocks = []
for d in ALL_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith("_ONE_DAY.csv"):
                sym = f.replace("_ONE_DAY.csv", "")
                if sym not in all_stocks:
                    all_stocks.append(sym)
all_stocks.sort()
print(f"Total stocks: {len(all_stocks)}")

# ====== 1. Load all daily data ======
print("\nLoading daily data for all stocks...")
prices = {}  # sym -> series of close
volumes = {}  # sym -> series of volume
returns = {}  # sym -> daily returns
ranges = {}   # sym -> (start, end)
stats_rows = []

for sym in all_stocks:
    fpath = None
    for d in ALL_DIRS:
        p = f"{d}/{sym}_ONE_DAY.csv"
        if os.path.exists(p):
            fpath = p; break
    if not fpath:
        continue
    
    try:
        df = pd.read_csv(fpath)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        vols = df["volume"].values
        
        prices[sym] = df.set_index("datetime")["close"]
        volumes[sym] = df.set_index("datetime")["volume"]
        
        # Daily returns
        ret = np.diff(np.log(closes))
        returns[sym] = ret
        
        ranges[sym] = (df["datetime"].min().date(), df["datetime"].max().date())
        
        # Basic stats
        n_days = len(closes)
        avg_close = np.mean(closes)
        std_close = np.std(closes)
        min_close = np.min(closes)
        max_close = np.max(closes)
        avg_vol = np.mean(vols)
        med_vol = np.median(vols)
        avg_daily_range = np.mean(highs - lows)
        avg_daily_range_pct = np.mean((highs - lows) / closes * 100)
        total_return = (closes[-1] / closes[0] - 1) * 100
        ann_vol = np.std(ret) * np.sqrt(252) * 100
        sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252) if np.std(ret) > 0 else 0
        max_drawdown = np.min(closes / np.maximum.accumulate(closes) - 1) * 100
        positive_days = np.sum(ret > 0) / len(ret) * 100
        
        stats_rows.append({
            "symbol": sym,
            "days": n_days,
            "avg_close": round(avg_close, 2),
            "std_close": round(std_close, 2),
            "min_close": round(min_close, 2),
            "max_close": round(max_close, 2),
            "total_return_%": round(total_return, 1),
            "ann_volatility_%": round(ann_vol, 1),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_%": round(max_drawdown, 1),
            "positive_day_%": round(positive_days, 1),
            "avg_daily_range_%": round(avg_daily_range_pct, 2),
            "avg_volume": int(avg_vol),
            "med_volume": int(med_vol),
            "start": str(ranges[sym][0]),
            "end": str(ranges[sym][1]),
        })
    except Exception as e:
        print(f"  Error reading {sym}: {e}")

stats_df = pd.DataFrame(stats_rows)
print(f"Loaded {len(stats_df)} stocks")

# ====== 2. TOP/BOTTOM RANKINGS ======
print("\n" + "="*90)
print("TOP/BOTTOM RANKINGS")
print("="*90)

# Best performers
print("\n--- TOP 10 BEST PERFORMERS (Total Return %) ---")
top_ret = stats_df.nlargest(10, "total_return_%")
for _, r in top_ret.iterrows():
    print(f"  {r['symbol']:15s} {r['total_return_%']:>8.1f}%  (sharpe={r['sharpe_ratio']:.2f}, vol={r['ann_volatility_%']:.1f}%)")

print("\n--- BOTTOM 10 WORST PERFORMERS ---")
bot_ret = stats_df.nsmallest(10, "total_return_%")
for _, r in bot_ret.iterrows():
    print(f"  {r['symbol']:15s} {r['total_return_%']:>8.1f}%  (sharpe={r['sharpe_ratio']:.2f}, vol={r['ann_volatility_%']:.1f}%)")

print("\n--- TOP 10 HIGHEST VOLATILITY ---")
top_vol = stats_df.nlargest(10, "ann_volatility_%")
for _, r in top_vol.iterrows():
    print(f"  {r['symbol']:15s} {r['ann_volatility_%']:>6.1f}%  (return={r['total_return_%']:>7.1f}%, sharpe={r['sharpe_ratio']:.2f})")

print("\n--- TOP 10 LOWEST VOLATILITY ---")
bot_vol = stats_df.nsmallest(10, "ann_volatility_%")
for _, r in bot_vol.iterrows():
    print(f"  {r['symbol']:15s} {r['ann_volatility_%']:>6.1f}%  (return={r['total_return_%']:>7.1f}%, sharpe={r['sharpe_ratio']:.2f})")

print("\n--- TOP 10 SHARPE RATIO (Risk-Adjusted Return) ---")
top_sharpe = stats_df[stats_df["sharpe_ratio"] < 5].nlargest(10, "sharpe_ratio")
for _, r in top_sharpe.iterrows():
    print(f"  {r['symbol']:15s} sharpe={r['sharpe_ratio']:>5.2f}  ret={r['total_return_%']:>7.1f}%  vol={r['ann_volatility_%']:>5.1f}%")

print("\n--- BOTTOM 10 SHARPE RATIO ---")
bot_sharpe = stats_df.nsmallest(10, "sharpe_ratio")
for _, r in bot_sharpe.iterrows():
    print(f"  {r['symbol']:15s} sharpe={r['sharpe_ratio']:>5.2f}  ret={r['total_return_%']:>7.1f}%  vol={r['ann_volatility_%']:>5.1f}%")

print("\n--- TOP 10 HIGHEST AVG DAILY RANGE (%) ---")
top_range = stats_df.nlargest(10, "avg_daily_range_%")
for _, r in top_range.iterrows():
    print(f"  {r['symbol']:15s} {r['avg_daily_range_%']:>6.2f}%  vol={r['ann_volatility_%']:>5.1f}%")

print("\n--- TOP 10 MOST TRADED (Avg Volume) ---")
top_vol_stocks = stats_df.nlargest(10, "avg_volume")
for _, r in top_vol_stocks.iterrows():
    print(f"  {r['symbol']:15s} avg_vol={r['avg_volume']:>10,}  med_vol={r['med_volume']:>10,}")

print("\n--- TOP 10 LARGEST DRAWDOWNS ---")
top_dd = stats_df.nsmallest(10, "max_drawdown_%")
for _, r in top_dd.iterrows():
    print(f"  {r['symbol']:15s} drawdown={r['max_drawdown_%']:>6.1f}%  return={r['total_return_%']:>7.1f}%")

# ====== 3. SECTOR ANALYSIS ======
print("\n" + "="*90)
print("SECTOR AGGREGATE ANALYSIS")
print("="*90)

# Simple sector mapping
sector_keywords = {
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "COFORGE", "MPHASIS", "PERSISTENT", "TATAELXSI"],
    "BANKING": ["HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "BANKBARODA", "PNB", "INDUSINDBK",
                  "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "YESBANK", "RBLBANK", "AUROPHARMA",
                  "CANBK", "INDIANB", "IDBI", "BANKINDIA", "UNIONBANK", "SOUTHBANK", "IOC"],
    "AUTO": ["TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR",
              "ASHOKLEY", "BHARATFORG", "BOSCHLTD"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA", "BIOCON",
                "ALKEM", "TORNTPHARM", "ZYDUSLIFE", "MANKIND", "STARHEALTH", "JBCHEPHARM", "SYNGENE"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "MARICO", "DABUR", "GODREJCP",
              "COLPAL", "GODREJIND", "UNITDSPR", "MCDOWELL-N"],
    "ENERGY": ["RELIANCE", "ONGC", "OIL", "IOC", "BPCL", "HINDPETRO", "GAIL", "ATGL", "ADANIGREEN",
                "POWERGRID", "NTPC", "COALINDIA", "JSWENERGY", "SUZLON"],
    "METAL": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "NATIONALUM", "NMDC", "JINDALSTEL", "SAIL",
               "VEDL", "HINDZINC", "MOIL"],
    "FINANCIAL": ["BAJFINANCE", "BAJAJFINSV", "HDFCAMC", "HDFCLIFE", "ICICIGI", "SBILIFE",
                   "LICHSGFIN", "MUTHOOTFIN", "CHOLAFIN", "SHRIRAMFIN", "PFC", "RECLTD"],
    "REALTY": ["DLF", "GODREJPROP", "LODHA", "PHOENIXLTD"],
    "TELECOM": ["BHARTIARTL", "IDEA", "VODAFONEIDEA"],
}

sector_stats = []
for sector, stocks in sector_keywords.items():
    present = [s for s in stocks if s in stats_df["symbol"].values]
    if present:
        sector_df = stats_df[stats_df["symbol"].isin(present)]
        sector_stats.append({
            "sector": sector,
            "stocks": len(present),
            "avg_return_%": round(sector_df["total_return_%"].mean(), 1),
            "avg_volatility_%": round(sector_df["ann_volatility_%"].mean(), 1),
            "avg_sharpe": round(sector_df["sharpe_ratio"].mean(), 2),
            "avg_drawdown_%": round(sector_df["max_drawdown_%"].mean(), 1),
            "avg_daily_range_%": round(sector_df["avg_daily_range_%"].mean(), 2),
        })

sector_df = pd.DataFrame(sector_stats).sort_values("avg_return_%", ascending=False)
print(f"\n{'Sector':15s} {'Stocks':6s} {'Return%':8s} {'Vol%':8s} {'Sharpe':8s} {'MaxDD%':8s} {'Range%':8s}")
print("-"*75)
for _, r in sector_df.iterrows():
    print(f"{r['sector']:15s} {r['stocks']:6d} {r['avg_return_%']:>7.1f}% {r['avg_volatility_%']:>6.1f}% {r['avg_sharpe']:>8.2f} {r['avg_drawdown_%']:>7.1f}% {r['avg_daily_range_%']:>7.2f}%")

# ====== 4. CORRELATION ANALYSIS ======
print("\n" + "="*90)
print("CROSS-CORRELATION ANALYSIS (Top 20 by market cap proxy)")
print("="*90)

# Get top 20 stocks by avg_close as market cap proxy
top20 = stats_df.nlargest(20, "avg_close")
top20_syms = top20["symbol"].tolist()

# Build correlation matrix
common_dates = None
price_series = {}
for s in top20_syms:
    if s in prices:
        price_series[s] = prices[s]
        if common_dates is None:
            common_dates = set(prices[s].index)
        else:
            common_dates &= set(prices[s].index)

common_dates = sorted(common_dates)
if len(common_dates) > 100:
    corr_data = pd.DataFrame({s: prices[s].reindex(common_dates) for s in top20_syms if s in prices})
    corr_returns = corr_data.apply(np.log).diff().dropna()
    corr_matrix = corr_returns.corr()
    
    # Find most & least correlated pairs
    pairs = []
    for i in range(len(top20_syms)):
        for j in range(i+1, len(top20_syms)):
            s1, s2 = top20_syms[i], top20_syms[j]
            if s1 in corr_matrix.index and s2 in corr_matrix.columns:
                c = corr_matrix.loc[s1, s2]
                pairs.append((c, s1, s2))
    
    pairs.sort(reverse=True)
    
    print(f"\nTOP 10 MOST CORRELATED PAIRS (out of {len(pairs)} pairs):")
    for c, s1, s2 in pairs[:10]:
        print(f"  {s1:15s} x {s2:15s}  r={c:.3f}")
    
    print(f"\nBOTTOM 10 LEAST CORRELATED / NEGATIVELY CORRELATED:")
    for c, s1, s2 in pairs[-10:]:
        print(f"  {s1:15s} x {s2:15s}  r={c:.3f}")
    
    # Average correlation per stock
    print(f"\nAVERAGE CORRELATION WITH MARKET (avg of all pairs per stock):")
    stock_avg_corr = {}
    for s in top20_syms:
        if s in corr_matrix.index:
            others = [c for c in corr_matrix.columns if c != s]
            avg_c = corr_matrix.loc[s, others].mean()
            stock_avg_corr[s] = avg_c
    
    for s, c in sorted(stock_avg_corr.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s:15s} avg_corr={c:.3f}")

# ====== 5. INTRADAY PATTERNS (sample of stocks) ======
print("\n" + "="*90)
print("INTRADAY PATTERN ANALYSIS (sample of 5 stocks, 1-min data)")
print("="*90)

for sample_sym in ["RELIANCE", "HDFCBANK", "TCS", "INFY", "SBIN"]:
    fpath = None
    for d in ALL_DIRS:
        p = f"{d}/{sample_sym}_ONE_MINUTE.csv"
        if os.path.exists(p):
            fpath = p; break
    if not fpath:
        continue
    
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time"] = df["datetime"].dt.time
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["time_min"] = df["hour"] * 60 + df["minute"]
    
    # Intraday volatility pattern
    df["intra_ret"] = df.groupby(df["datetime"].dt.date)["close"].transform(lambda x: np.log(x / x.shift(1)))
    
    # Average return by minute of day
    min_avg_ret = df.groupby("time_min")["intra_ret"].mean() * 10000  # in bps
    
    # Find the most volatile and least volatile minutes
    min_vol = df.groupby("time_min")["intra_ret"].std() * 10000
    
    top_vol_min = min_vol.nlargest(5)
    low_vol_min = min_vol.nsmallest(5)
    
    print(f"\n{sample_sym} - Intraday Patterns:")
    print(f"  Most volatile minutes (bps std):")
    for tm, v in top_vol_min.items():
        h, m = divmod(tm, 60)
        print(f"    {h:02d}:{m:02d}  vol={v:.2f} bps")
    print(f"  Calmest minutes:")
    for tm, v in low_vol_min.items():
        h, m = divmod(tm, 60)
        print(f"    {h:02d}:{m:02d}  vol={v:.2f} bps")
    
    # First 30 min vs last 30 min volatility
    first30 = min_vol[min_vol.index <= 9*60+59]  # up to 10:00
    last30 = min_vol[min_vol.index >= 14*60+30]  # after 14:30
    if len(first30) > 0 and len(last30) > 0:
        print(f"  Avg vol first 30min: {first30.mean():.2f} bps")
        print(f"  Avg vol last 30min:  {last30.mean():.2f} bps")

# ====== 6. DISTRIBUTION ANALYSIS ======
print("\n" + "="*90)
print("RETURN DISTRIBUTION ANALYSIS")
print("="*90)

all_returns = []
for sym, ret in returns.items():
    all_returns.extend(ret)

all_returns = np.array(all_returns)
print(f"Total daily return observations: {len(all_returns):,}")
print(f"Mean daily return: {np.mean(all_returns)*100:.4f}%")
print(f"Median daily return: {np.median(all_returns)*100:.4f}%")
print(f"Std dev (daily): {np.std(all_returns)*100:.4f}%")
print(f"Skewness: {pd.Series(all_returns).skew():.3f}")
print(f"Kurtosis: {pd.Series(all_returns).kurtosis():.3f}")
print(f"Min daily return: {np.min(all_returns)*100:.4f}%")
print(f"Max daily return: {np.max(all_returns)*100:.4f}%")

# Percentiles
for p in [1, 5, 25, 50, 75, 95, 99]:
    val = np.percentile(all_returns, p) * 100
    print(f"  {p:2d}th percentile: {val:.4f}%")

# Positive day stats
pos_pct = np.sum(all_returns > 0) / len(all_returns) * 100
print(f"\nPositive days: {pos_pct:.1f}%")
print(f"Negative days: {100-pos_pct:.1f}%")

# ====== 7. OUTLIER DETECTION ======
print("\n" + "="*90)
print("OUTLIER DAY DETECTION (>3 sigma events per stock)")
print("="*90)

outlier_count = []
for sym in all_stocks[:50]:  # Top 50 by name
    if sym not in returns:
        continue
    ret = returns[sym]
    mu, sigma = np.mean(ret), np.std(ret)
    outliers = np.sum(np.abs(ret - mu) > 3 * sigma)
    outlier_count.append((outliers, len(ret), sym))

outlier_count.sort(reverse=True)
print("\nStocks with most outlier days:")
for o, n, s in outlier_count[:10]:
    print(f"  {s:15s} {o:4d} outliers in {n:4d} days ({o/n*100:.1f}%)")

# ====== 8. SUMMARY STATS ======
print("\n" + "="*90)
print("MARKET-WIDE SUMMARY STATISTICS")
print("="*90)
print(f"Total stocks analyzed: {len(stats_df)}")
print(f"Average trading days per stock: {stats_df['days'].mean():.0f}")
print(f"Average total return: {stats_df['total_return_%'].mean():.1f}%")
print(f"Median total return: {stats_df['total_return_%'].median():.1f}%")
print(f"Average annual volatility: {stats_df['ann_volatility_%'].mean():.1f}%")
print(f"Average Sharpe ratio: {stats_df['sharpe_ratio'].mean():.2f}")
print(f"Average max drawdown: {stats_df['max_drawdown_%'].mean():.1f}%")
print(f"Average positive day ratio: {stats_df['positive_day_%'].mean():.1f}%")
print(f"Average daily range: {stats_df['avg_daily_range_%'].mean():.2f}%")
print(f"Average daily volume: {stats_df['avg_volume'].mean():,.0f}")

# Save full stats CSV
stats_df.to_csv("stock_statistics.csv", index=False)
print(f"\nFull statistics saved to stock_statistics.csv")
print("Analysis complete!")
