"""Import TCIEXP, TRIVENI, UTIAMC into warehouse and audit all data"""
import duckdb, pandas as pd, os, re
from pathlib import Path

BASE = Path(r'C:\Users\pc\Downloads\stock hist data')
DB = BASE / 'warehouse' / 'market_data.duckdb'
CSV_DIR = BASE / 'comprehensive_data'

TF_MAP = {
    'ONE_DAY': '1day', 'ONE_HOUR': '60min',
    'FIFTEEN_MINUTE': '15min', 'FIVE_MINUTE': '5min', 'ONE_MINUTE': '1min'
}

con = duckdb.connect(str(DB))

# Step 1: Import 3 missing stocks
missing = ['TCIEXP', 'TRIVENI', 'UTIAMC']
print("=== Importing 3 missing stocks ===")
imported_total = 0
for sym in missing:
    for tf_raw, tf in TF_MAP.items():
        fp = CSV_DIR / f"{sym}_{tf_raw}.csv"
        if not fp.exists():
            print(f"  {sym}_{tf_raw}: CSV not found, skip")
            continue
        df = pd.read_csv(fp)
        if len(df) == 0: continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['symbol'] = sym
        df['timeframe'] = tf
        df = df[['symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Check existing
        already = con.execute("SELECT datetime FROM raw_market WHERE symbol=? AND timeframe=?", [sym, tf]).fetchdf()
        if len(already) > 0:
            existing_dts = set(pd.to_datetime(already['datetime']))
            df_new = df[~df['datetime'].isin(existing_dts)]
            if len(df_new) > 0:
                con.register('df_new', df_new)
                con.execute("""
                    INSERT INTO raw_market (symbol,timeframe,datetime,open,high,low,close,volume,source)
                    SELECT symbol,timeframe,datetime,open,high,low,close,volume,'csv_import' FROM df_new
                """)
                con.unregister('df_new')
                print(f"  {sym}_{tf}: {len(df_new):,} new rows (had {len(already):,})")
            else:
                print(f"  {sym}_{tf}: already up to date ({len(already):,} rows)")
        else:
            con.register('df_sub', df)
            con.execute("""
                INSERT INTO raw_market (symbol,timeframe,datetime,open,high,low,close,volume,source)
                SELECT symbol,timeframe,datetime,open,high,low,close,volume,'csv_import' FROM df_sub
            """)
            con.unregister('df_sub')
            print(f"  {sym}_{tf}: {len(df):,} rows inserted")
        imported_total += 1

print(f"\nImported {imported_total} files")

# Step 2: Comprehensive audit of all symbols x timeframes
print("\n=== Full Data Audit ===")
syms = sorted(r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw_market").fetchall())
print(f"Total symbols in DB: {len(syms)}")

issues = []
for sym in syms:
    for tf in ['1min', '5min', '15min', '60min', '1day']:
        row = con.execute("""
            SELECT COUNT(*) as cnt, MIN(datetime) as first, MAX(datetime) as last
            FROM raw_market WHERE symbol=? AND timeframe=?
        """, [sym, tf]).fetchone()
        cnt, first, last = row
        if cnt == 0:
            issues.append(f"{sym} {tf}: NO DATA")
        elif tf == '1min' and last:
            # Check recency - should be within 7 days
            days_old = (pd.Timestamp.now(tz='Asia/Kolkata') - pd.Timestamp(last)).days
            if days_old > 7:
                issues.append(f"{sym} {tf}: stale - ends {last.date()} ({days_old}d old, {cnt:,} rows)")

if issues:
    print(f"\n{len(issues)} issues found:")
    for i in issues[:20]:
        print(f"  {i}")
    if len(issues) > 20:
        print(f"  ... and {len(issues)-20} more")
else:
    print("No issues found!")

# Step 3: Summary stats
print("\n=== Database Summary ===")
total_rows = con.execute("SELECT COUNT(*) FROM raw_market").fetchone()[0]
print(f"Total rows: {total_rows:,}")
print("Rows per timeframe:")
for r in con.execute("SELECT timeframe, COUNT(*), COUNT(DISTINCT symbol) FROM raw_market GROUP BY timeframe ORDER BY timeframe").fetchall():
    print(f"  {r[0]}: {r[1]:,} rows, {r[2]} symbols")

con.close()
print("\nDone!")
