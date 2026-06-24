"""
Phase 20a — Daily Scanner (Range-based).
Shows probability of next-day wide-range candles.
"""
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def run_scanner(timeframe="1day", top_n=15):
    from src.discovery.engine import run as discovery_run
    discovery_run([timeframe])

    con = duckdb.connect(str(DB_PATH))

    today = datetime.now()
    cols = "symbol, composite_score, confidence, range_2pct_prob, range_5pct_prob, range_6pct_prob, expected_range_pct, structure_score, regime_alignment, wide_bullish_prob, wide_bearish_prob, net_directional"
    top = con.execute(f"""
        SELECT {cols}
        FROM discovery_scores
        WHERE timeframe=? AND datetime::DATE=?
        ORDER BY composite_score DESC LIMIT ?
    """, [timeframe, today.date(), top_n]).fetchdf()

    if len(top) == 0:
        top = con.execute(f"""
            SELECT {cols}
            FROM discovery_scores
            WHERE timeframe=?
            ORDER BY datetime DESC, composite_score DESC LIMIT ?
        """, [timeframe, top_n]).fetchdf()

    mkt = con.execute(
        "SELECT regime_label, volatility_regime FROM market_regimes WHERE timeframe=? ORDER BY datetime DESC LIMIT 1",
        [timeframe]
    ).fetchone()
    con.close()

    if len(top) == 0:
        print("No discovery scores available.")
        return top

    print("=" * 85)
    print(f"  WIDE-RANGE SCANNER — {today.date()}")
    if mkt:
        print(f"  Market: {mkt[0].upper()} | Volatility: {mkt[1]}")
    print("=" * 85)
    print(f"  {'#':3s} {'Symbol':14s} {'Score':6s} {'Conf':8s} {'>2%':6s} {'>5%':6s} {'>6%':6s} {'Dir':8s} {'Exp%':6s}")
    print(f"  {'-'*3} {'-'*14} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for i, (_, r) in enumerate(top.iterrows()):
        nd = r.get("net_directional", 0)
        dir_lbl = f"BULL:{r.get('wide_bullish_prob',0):.0%}" if nd > 0.1 else f"BEAR:{r.get('wide_bearish_prob',0):.0%}" if nd < -0.1 else "NEUT"
        print(f"  {i+1:3d} {r['symbol']:14s} {r['composite_score']:5.0f}  {r['confidence']:8s} {r['range_2pct_prob']:.0%}  {r['range_5pct_prob']:.0%}  {r['range_6pct_prob']:.0%}  {dir_lbl:8s} {r['expected_range_pct']:.1f}")

    print(f"\n  Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return top


if __name__ == "__main__":
    run_scanner()
