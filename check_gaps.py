import duckdb
con = duckdb.connect("warehouse/market_data.duckdb")

# Check RKFORGE 1day gaps
rows = con.execute("""
    SELECT datetime::DATE as dt
    FROM raw_market WHERE symbol='RKFORGE' AND timeframe='1day'
    ORDER BY datetime
""").fetchall()

gaps = [(rows[i][0], (rows[i][0]-rows[i-1][0]).days) for i in range(1, len(rows)) if (rows[i][0]-rows[i-1][0]).days > 1]
print(f"RKFORGE: {len(rows)} trading days, {len(gaps)} gaps")
# Show gap distribution
from collections import Counter
gap_dist = Counter(g[1] for g in gaps)
for d, c in sorted(gap_dist.items()):
    print(f"  {d-1}d gap: {c} occurrences")
print(f"First: {rows[0][0]}, Last: {rows[-1][0]}")

# Check if gaps are just weekends/holidays
print("\nSample 10 gaps:")
for dt, gap in gaps[:10]:
    prev = [rows[i-1][0] for i in range(1, len(rows)) if (rows[i][0]-rows[i-1][0]).days > 1][:10]
    idx = [i for i in range(1, len(rows)) if (rows[i][0]-rows[i-1][0]).days > 1][:10]
    print(f"  {rows[idx[0]-1][0]} -> {rows[idx[0]][0]} ({gap-1}d gap)")
    
con.close()
