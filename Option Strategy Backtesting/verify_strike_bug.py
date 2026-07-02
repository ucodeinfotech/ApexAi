"""
Verify if the backtest has a strike-switching bug
"""
import duckdb, pandas as pd

con=duckdb.connect(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")

print("=== atm_distance=0 data at key timestamps ===")
q = """SELECT timestamp, close, strike, spot, atm_distance
FROM options_data_dedup
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
AND timestamp IN ('2021-11-17 09:55:00', '2021-11-17 10:00:00', '2021-11-17 14:00:00', '2021-11-18 09:15:00', '2021-11-22 09:15:00')
ORDER BY timestamp"""
print(con.execute(q).df().to_string())

print("\n=== Fixed strike data for Trade #532 (entry 2021-11-17 10:00) ===")
# Find the ATM strike at entry
q0 = "SELECT strike FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 AND timestamp='2021-11-17 10:00:00'"
entry_strike = con.execute(q0).fetchone()[0]
print(f"ATM strike at entry: {entry_strike}")

# Now get data for that fixed strike across time
q = f"""SELECT timestamp, close, strike, spot, atm_distance
FROM options_data_dedup
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK'
AND strike={entry_strike}
AND timestamp BETWEEN '2021-11-17 09:55:00' AND '2021-11-22 10:00:00'
ORDER BY timestamp"""
res = con.execute(q).df()
for _, r in res.iterrows():
    print(f"  {r['timestamp']} close={r['close']:.1f} spot={r['spot']:.1f} atm_dist={r['atm_distance']}")

print(f"\n=== What should the option PnL be? ===")
# Entry: 2021-11-17 10:00, atm_distance=0 -> does it match our strike?
q1 = "SELECT close, strike, spot FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 AND timestamp='2021-11-17 10:00:00'"
r1 = con.execute(q1).fetchone()
print(f"Entry (atm_dist=0): close={r1[0]:.1f} strike={r1[1]:.0f} spot={r1[2]:.0f}")

# Exit: 2021-11-22 09:15, but for the FIXED strike
q2 = f"SELECT close, strike, spot FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike={r1[1]} AND timestamp='2021-11-22 09:15:00'"
r2 = con.execute(q2).fetchone()
print(f"Exit (fixed strike={r1[1]:.0f}): close={r2[0]:.1f} spot={r2[1]:.0f}")

real_pnl = r2[0] - r1[0]
print(f"Real PnL (fixed strike): {real_pnl:.1f}")

# What does atm_distance=0 give at exit?
q3 = "SELECT close, strike, spot FROM options_data_dedup WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0 AND timestamp='2021-11-22 09:15:00'"
r3 = con.execute(q3).fetchone()
print(f"Exit (atm_dist=0): close={r3[0]:.1f} strike={r3[1]:.0f} spot={r3[2]:.0f}")
bug_pnl = r3[0] - r1[0]
print(f"Bug PnL (atm_dist=0 change): {bug_pnl:.1f}")

# Verify: how often does atm_distance=0 switch strikes?
print("\n=== How often does atm_distance=0 switch strikes? ===")
q = """SELECT timestamp, strike
FROM options_data_dedup
WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
AND timestamp BETWEEN '2021-11-17 09:55:00' AND '2021-11-18 15:30:00'
ORDER BY timestamp"""
res = con.execute(q).df()
strikes = res['strike'].unique()
print(f"Unique strikes in one day: {len(strikes)} -> {sorted(int(s) for s in strikes)}")
print(f"Strike changes: {(res['strike'] != res['strike'].shift(1)).sum()} times")

con.close()
