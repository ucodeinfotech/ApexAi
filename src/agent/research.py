"""
Phase 20c — AI Research Agent.
CLI tool that answers questions about stocks using the warehouse.
Usage: python -m src.agent.research "What is the current score for IDBI?"
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def query(q):
    """Answer a natural language question about stocks using warehouse data."""
    con = duckdb.connect(str(DB_PATH))
    q_lower = q.lower()
    results = []
    explanation = ""

    # Current regime
    mkt = con.execute("SELECT regime_label, volatility_regime FROM market_regimes WHERE timeframe='1day' ORDER BY datetime DESC LIMIT 1").fetchone()
    regime_str = f"{mkt[0]} market with {mkt[1]} volatility" if mkt else "unknown"

    # Check if asking about a specific symbol
    known_symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM discovery_scores ORDER BY symbol").fetchall()]
    mentioned = [s for s in known_symbols if s.lower() in q_lower]

    if "score" in q_lower or "rank" in q_lower or "pick" in q_lower or "recommend" in q_lower:
        if mentioned:
            sym = mentioned[0]
            row = con.execute("""
                SELECT composite_score, confidence, hg_probability, expected_return,
                       risk_score, pattern_score, structure_score, datetime
                FROM discovery_scores WHERE symbol=? ORDER BY datetime DESC LIMIT 1
            """, [sym]).fetchone()
            if row:
                results.append(f"{sym}: Score {row[0]:.0f}/100 ({row[1]}), ML prob {row[2]:.1%}, expected return {row[3]:+.2f}%")
                explanation = f"Risk {row[4]:.0f}, Pattern {row[5]:.0f}, Structure {row[6]:.0f} | Regime: {regime_str}"
                # Get explanation text
                exp = con.execute("SELECT explanation FROM ml_explanations WHERE symbol=? LIMIT 1", [sym]).fetchone()
                if exp and exp[0]:
                    explanation += f"\n{exp[0]}"
        else:
            top = con.execute("""
                SELECT symbol, composite_score, confidence FROM discovery_scores
                ORDER BY datetime DESC, composite_score DESC LIMIT 5
            """).fetchall()
            if top:
                results.append("Top picks:")
                for i, r in enumerate(top):
                    results.append(f"  {i+1}. {r[0]} — Score {r[1]:.0f}/100 ({r[2]})")
                explanation = f"Market regime: {regime_str}"

    elif "bullish" in q_lower or "pattern" in q_lower:
        if mentioned:
            sym = mentioned[0]
            pats = con.execute("""
                SELECT pattern, direction, COUNT(1) as cnt
                FROM pattern_occurrences
                WHERE symbol=? AND timeframe='1day'
                GROUP BY pattern, direction
                ORDER BY cnt DESC LIMIT 5
            """, [sym]).fetchall()
            if pats:
                results.append(f"Patterns for {sym}:")
                for p in pats:
                    results.append(f"  {p[0]} ({p[1]}): {p[2]} occurrences")
        else:
            top_pat = con.execute("""
                SELECT po.symbol, po.pattern, po.direction, COUNT(1) as cnt
                FROM pattern_occurrences po
                WHERE po.timeframe='1day' AND po.direction='bullish'
                GROUP BY po.symbol, po.pattern, po.direction
                ORDER BY cnt DESC LIMIT 5
            """).fetchall()
            results.append("Most bullish patterns:")
            for p in top_pat:
                results.append(f"  {p[0]} — {p[1]} ({p[2]}): {p[3]}")

    elif "regime" in q_lower or "market" in q_lower:
        results.append(f"Current market regime: {regime_str}")
        regimes = con.execute("""
            SELECT regime_label, COUNT(1) as days FROM market_regimes
            WHERE timeframe='1day' GROUP BY regime_label ORDER BY days DESC
        """).fetchall()
        results.append("Regime history:")
        for r_item in regimes:
            results.append(f"  {r_item[0]}: {r_item[1]} days")

    elif "perform" in q_lower or "backtest" in q_lower or "result" in q_lower:
        bt = con.execute("""
            SELECT strategy, total_return, annual_return, sharpe, win_rate
            FROM backtest_results ORDER BY sharpe DESC LIMIT 5
        """).fetchall()
        if bt:
            results.append("Backtest results (best Sharpe):")
            for b in bt:
                results.append(f"  {b[0]}: ret={float(b[1]):.1%}, ann={float(b[2]):.1%}, Sharpe={float(b[3]):.2f}, win={float(b[4]):.1%}")
        else:
            results.append("No backtest results available yet.")

    elif "decile" in q_lower or "signal" in q_lower:
        pat_scores = con.execute("""
            SELECT po.pattern, COUNT(1) as cnt
            FROM pattern_occurrences po
            WHERE po.timeframe='1day' AND po.direction='bullish'
            GROUP BY po.pattern ORDER BY cnt DESC LIMIT 5
        """).fetchall()
        if pat_scores:
            results.append("Most frequent bullish patterns:")
            for p in pat_scores:
                results.append(f"  {p[0]}: {p[1]} occurrences")

    elif "help" in q_lower or "what can" in q_lower:
        results.append("Available queries:")
        results.append('  "score for [SYMBOL]" — get latest score and prediction')
        results.append('  "top picks" — today\'s best ranked stocks')
        results.append('  "bullish patterns for [SYMBOL]" — pattern breakdown')
        results.append('  "market regime" — current market conditions')
        results.append('  "backtest results" — strategy performance')
        results.append('  "signal quality" — best patterns')

    else:
        results.append(f"I couldn't understand your question. Try asking about scores, patterns, regime, or backtest results. Regime: {regime_str}")

    con.close()
    return "\n".join(results)


def cli():
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not q:
        print("Usage: python -m src.agent.research \"your question about stocks\"")
        print("Example: python -m src.agent.research \"top picks\"")
        sys.exit(1)
    result = query(q)
    print(result)


if __name__ == "__main__":
    cli()
