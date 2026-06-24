"""
Load delivery % data for all 90 stocks from NSE via nselib.
Stores in warehouse as `delivery_data` table.
"""
import duckdb
import pandas as pd
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nselib import capital_market

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def load_all():
    con = duckdb.connect(str(DB_PATH))

    # Get symbols
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM feature_store WHERE timeframe='1day' ORDER BY symbol"
    ).fetchall()]
    print(f"Symbols to load: {len(symbols)}")

    # Create table
    con.execute("""
        CREATE TABLE IF NOT EXISTS delivery_data (
            symbol VARCHAR,
            date DATE,
            traded_qty BIGINT,
            deliverable_qty BIGINT,
            delivery_pct DOUBLE
        )
    """)

    total = 0
    errors = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        try:
            df = capital_market.deliverable_position_data(
                sym, from_date="01-01-2016", to_date="01-06-2026"
            )
            if len(df) == 0:
                continue

            # Clean and standardize
            rows = []
            for _, r in df.iterrows():
                date_str = r["Date"]
                # Parse Indian date format: "18-Jun-2026"
                try:
                    dt = pd.to_datetime(date_str, format="%d-%b-%Y")
                except:
                    try:
                        dt = pd.to_datetime(date_str, format="%d-%m-%Y")
                    except:
                        continue

                traded = str(r["TradedQty"]).replace(",", "")
                delivered = str(r["%DlyQttoTradedQty"]).replace(",", "")
                del_qty = str(r["DeliverableQty"]).replace(",", "")

                rows.append((
                    sym, dt.date(),
                    int(traded) if traded.isdigit() else 0,
                    int(del_qty) if del_qty.isdigit() else 0,
                    float(delivered) if delivered.replace(".", "").isdigit() else 0.0
                ))

            if rows:
                con.executemany(
                    "INSERT INTO delivery_data VALUES (?, ?, ?, ?, ?)",
                    rows
                )
                total += len(rows)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(symbols)}] {sym}: {len(rows)} rows ({total} total, {rate:.1f} sym/s)")

        except Exception as e:
            errors += 1
            print(f"  ERROR [{sym}]: {str(e)[:100]}")

    elapsed = time.time() - t0
    print(f"\nDone: {total} rows from {len(symbols)} symbols ({errors} errors) in {elapsed:.0f}s")

    # Verify
    cnt = con.execute("SELECT COUNT(1) FROM delivery_data").fetchone()[0]
    sym_cnt = con.execute("SELECT COUNT(DISTINCT symbol) FROM delivery_data").fetchone()[0]
    date_range = con.execute("SELECT MIN(date), MAX(date) FROM delivery_data").fetchone()
    print(f"Stored: {cnt} rows, {sym_cnt} symbols")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    con.close()


if __name__ == "__main__":
    load_all()
