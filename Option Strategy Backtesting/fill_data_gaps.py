"""Fill gaps: continuous 5-min bars per (strike, expiry_date) with ffill. One strike at a time."""
import duckdb, pandas as pd, numpy as np
from datetime import timedelta, time
from tqdm import tqdm

DB_PATH = r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb"
TABLE_IN = "options_data_clean"
TABLE_OUT = "options_data_filled"

con = duckdb.connect(str(DB_PATH))
tables = set(r[0] for r in con.execute("SHOW TABLES").fetchall())

if TABLE_OUT in tables:
    cnt = con.execute(f"SELECT count(*) FROM \"{TABLE_OUT}\"").fetchone()[0]
    print(f"{TABLE_OUT} already exists: {cnt:,} rows. Dropping and recreating...")
    con.execute(f"DROP TABLE \"{TABLE_OUT}\"")

def get_weekly_expiry(ts):
    dt = pd.Timestamp(ts)
    days_ahead = (3 - dt.weekday()) % 7
    expiry = dt + timedelta(days=days_ahead)
    if dt.weekday() == 3 and dt.time() >= time(15, 30):
        expiry += timedelta(days=7)
    return expiry.date()

# Get unique strikes
strikes = con.execute(f"""
    SELECT DISTINCT strike FROM {TABLE_IN}
    WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
    ORDER BY strike
""").fetchdf()["strike"].tolist()
print(f"Strikes: {len(strikes)}")

created = False
for stk in tqdm(strikes, desc="Strikes"):
    df = con.execute(f"""
        SELECT timestamp, close, open, high, low, volume, oi, iv, spot, atm_distance
        FROM {TABLE_IN}
        WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike = {stk}
        ORDER BY timestamp
    """).fetchdf()
    if df.empty: continue
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    
    # Compute expiry per bar
    df["expiry_date"] = df["timestamp"].apply(get_weekly_expiry)
    
    rows = []
    for exp_date, grp in df.groupby("expiry_date", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < 2: continue
        grp = grp.drop_duplicates(subset="timestamp", keep="first")
        grp = grp.set_index("timestamp")
        filled = grp.resample("5min").ffill()
        filled = filled.dropna(subset=["close"])
        
        # Add constant cols
        filled["strike"] = stk
        filled["option_type"] = "CALL"
        filled["expiry_code"] = 1
        filled["expiry_flag"] = "WEEK"
        filled["expiry_date"] = exp_date
        rows.append(filled.reset_index())
    
    if not rows:
        continue
    batch = pd.concat(rows, ignore_index=True)
    
    # Write
    con.register("_batch", batch)
    if not created:
        con.execute(f"""
            CREATE TABLE \"{TABLE_OUT}\" AS
            SELECT timestamp, strike, expiry_date, option_type, expiry_code, expiry_flag,
                   close, open, high, low, volume, oi, iv, spot, atm_distance
            FROM _batch
        """)
        created = True
    else:
        con.execute(f"""
            INSERT INTO \"{TABLE_OUT}\" (timestamp, strike, expiry_date, option_type, expiry_code, expiry_flag,
                                         close, open, high, low, volume, oi, iv, spot, atm_distance)
            SELECT timestamp, strike, expiry_date, option_type, expiry_code, expiry_flag,
                   close, open, high, low, volume, oi, iv, spot, atm_distance
            FROM _batch
        """)

# Stats
cnt = con.execute(f"SELECT count(*) FROM \"{TABLE_OUT}\"").fetchone()[0]
stk_cnt = con.execute(f"SELECT count(DISTINCT strike) FROM \"{TABLE_OUT}\"").fetchone()[0]
exp_cnt = con.execute(f"SELECT count(DISTINCT expiry_date) FROM \"{TABLE_OUT}\"").fetchone()[0]
print(f"\n{TABLE_OUT}: {cnt:,} rows, {stk_cnt} strikes, {exp_cnt} expiry dates")
print(con.execute(f"SELECT min(timestamp) first, max(timestamp) last FROM \"{TABLE_OUT}\"").fetchdf().to_string(index=False))

con.close()
