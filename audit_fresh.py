import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")

# Check 1day freshness
print("=== 1day freshness ===")
stale_1day = con.execute("""
    SELECT symbol, CAST(MAX(datetime) AS DATE) as last_dt
    FROM raw_market WHERE timeframe='1day'
    GROUP BY symbol
    HAVING MAX(datetime) < CAST('2026-06-20' AS TIMESTAMP)
    ORDER BY last_dt
""").fetchall()
print(f"Stale 1day symbols: {len(stale_1day)}")
for s in stale_1day[:10]:
    print(f"  {s[0]}: ends {s[1]}")

# Check 1min freshness
print(f"\n=== 1min freshness ===")
stale_1min = con.execute("""
    SELECT symbol, CAST(MAX(datetime) AS DATE) as last_dt
    FROM raw_market WHERE timeframe='1min'
    GROUP BY symbol
    HAVING MAX(datetime) < CAST('2026-06-20' AS TIMESTAMP)
    ORDER BY last_dt
""").fetchall()
print(f"Stale 1min symbols: {len(stale_1min)}")

# Group by end date
from collections import Counter
ends = Counter(s[1] for s in stale_1min)
for dt, cnt in sorted(ends.items()):
    print(f"  ends {dt}: {cnt} symbols")

# Check if any of the 30 new stocks have stale 1min
new_30 = ["ABSLAMC","AMBER","ANGELONE","ASTRAMICRO","AURIONPRO","AWL","BSOFT",
    "GPTINFRA","ICICIAMC","IREDA","KALPATARU","KPITTECH","LATENTVIEW","MAZDA",
    "MUTHOOTCAP","NAM-INDIA","NUVOCO","NYKAA","OLAELEC","PATANJALI","PAYTM",
    "POLICYBZR","RVNL","SAFARI","SONACOMS","SWIGGY","TCI","TCIEXP","TRIVENI","UTIAMC"]
for sym in new_30:
    r = con.execute("SELECT CAST(MAX(datetime) AS DATE) FROM raw_market WHERE symbol=? AND timeframe='1min'", [sym]).fetchone()
    if r[0] and r[0] < "2026-06-20":
        print(f"  NEW STOCK STALE: {sym} ends {r[0]}")

con.close()
