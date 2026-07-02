import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting"
CUTOFF = pd.Timestamp("14:15").time()
def A(df, p=14):
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()
CH_VALS = [25, 30, 35, 40, 45, 50, 55, 60]

entries_total = 0
ch_pass_full = 0
ch_pass_2000 = 0

for sym in ["NIFTY50", "SENSEX"]:
    h1 = pd.read_csv(os.path.join(BASE, f"{sym}_ONE_HOUR.csv"))
    m5 = pd.read_csv(os.path.join(BASE, f"{sym}_FIVE_MINUTE.csv"))
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    m5 = m5.sort_values("datetime").reset_index(drop=True)
    body = (h1["close"] - h1["open"]).abs()
    is_red = h1["close"] < h1["open"]
    is_green = h1["close"] > h1["open"]
    du = m5["datetime"].apply(lambda x: int(x.timestamp())).values
    hi = m5["high"].values; lo = m5["low"].values; cl = m5["close"].values
    atr5 = A(m5, 14).values
    tc = m5["datetime"].dt.time
    
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
        entries_total += 1
        
        # Full scan for CH=45
        he = ep
        found45 = False
        for j in range(r, len(m5)):
            ca = atr5[j]
            if pd.isna(ca):
                continue
            if hi[j] > he:
                he = hi[j]
            if cl[j] < he - 45 * ca:
                found45 = True
                break
        if found45:
            ch_pass_full += 1
        
        # Limited scan (2000 candles)
        he = ep
        found45_2000 = False
        for j in range(r, min(r + 2000, len(m5))):
            ca = atr5[j]
            if pd.isna(ca):
                continue
            if hi[j] > he:
                he = hi[j]
            if cl[j] < he - 45 * ca:
                found45_2000 = True
                break
        if found45_2000:
            ch_pass_2000 += 1

print(f"Entries before CH: {entries_total}")
print(f"CH=45 passes (full scan): {ch_pass_full}")
print(f"CH=45 passes (2000 limit): {ch_pass_2000}")
