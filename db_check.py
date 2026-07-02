import duckdb
c = duckdb.connect("warehouse/market_data.duckdb")

# Check RELIANCE
r = c.execute("SELECT MAX(datetime) FROM raw_market WHERE symbol='RELIANCE' AND timeframe='1min'").fetchone()
print("RELIANCE 1min last:", r[0])

# Check 27 new symbols in 1min
new = ["ABSLAMC","AMBER","ANGELONE","ASTRAMICRO","AURIONPRO","AWL","BSOFT",
    "GPTINFRA","ICICIAMC","IREDA","KALPATARU","KPITTECH","LATENTVIEW","MAZDA",
    "MUTHOOTCAP","NAM-INDIA","NUVOCO","NYKAA","OLAELEC","PATANJALI","PAYTM",
    "POLICYBZR","RVNL","SAFARI","SONACOMS","SWIGGY","TCI"]
r2 = c.execute("SELECT symbol, CAST(MAX(datetime) AS DATE) FROM raw_market WHERE timeframe='1min' AND symbol IN (?) GROUP BY symbol", [new]).fetchall()
print("New symbols 1min last dates:")
for s, dt in r2:
    print(f"  {s}: {dt}")

# Check RELIANCE 1day
r3 = c.execute("SELECT MAX(datetime) FROM raw_market WHERE symbol='RELIANCE' AND timeframe='1day'").fetchone()
print("RELIANCE 1day last:", r3[0])

c.close()
