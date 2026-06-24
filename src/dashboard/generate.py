"""
Phase 20b — Dashboard Generator.
Creates an HTML dashboard showing top picks, decile analysis, and market context.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"


def generate_html(con, timeframe="1day"):
    """Generate the HTML dashboard."""
    today = datetime.now()

    # Top picks
    top = con.execute(f"""
        SELECT symbol, composite_score, confidence, hg_probability, expected_return,
               risk_score, pattern_score, structure_score, regime_alignment
        FROM discovery_scores
        WHERE timeframe=? AND datetime::DATE=?
        ORDER BY composite_score DESC LIMIT 20
    """, [timeframe, today.date()]).fetchdf()
    if len(top) == 0:
        top = con.execute(f"""
            SELECT symbol, composite_score, confidence, hg_probability, expected_return,
                   risk_score, pattern_score, structure_score, regime_alignment
            FROM discovery_scores
            WHERE timeframe=?
            ORDER BY datetime DESC, composite_score DESC LIMIT 20
        """, [timeframe]).fetchdf()

    # Market regime
    mkt = con.execute(
        "SELECT datetime, regime_label, volatility_regime FROM market_regimes WHERE timeframe=? ORDER BY datetime DESC LIMIT 1",
        [timeframe]
    ).fetchone()

    # Backtest summary
    bt_best = con.execute("""
        SELECT strategy, total_return, annual_return, sharpe, max_drawdown, win_rate
        FROM backtest_results WHERE timeframe=? ORDER BY sharpe DESC LIMIT 3
    """, [timeframe]).fetchdf()

    # Decile analysis from backtest
    bt_detail = con.execute("SELECT * FROM backtest_results WHERE timeframe=? ORDER BY strategy", [timeframe]).fetchdf()

    # ML metrics
    ml_metrics = con.execute("SELECT * FROM ml_model_metrics WHERE timeframe=?", [timeframe]).fetchdf()

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>High-Gainer Discovery Dashboard — {today.date()}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1a1a2e; color: white; padding: 10px 12px; text-align: left; font-size: 13px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tr:hover {{ background: #f0f0f0; }}
  .high {{ background: #d4edda; }}
  .medium {{ background: #fff3cd; }}
  .low {{ background: #f8d7da; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .conf-high {{ color: #28a745; font-weight: bold; }}
  .conf-med {{ color: #ffc107; font-weight: bold; }}
  .conf-low {{ color: #dc3545; font-weight: bold; }}
  .footer {{ text-align: center; margin-top: 40px; color: #888; font-size: 12px; }}
  .score-bar {{ height: 8px; background: #e9ecef; border-radius: 4px; margin-top: 3px; }}
  .score-fill {{ height: 8px; border-radius: 4px; background: linear-gradient(90deg, #28a745, #20c997); }}
</style>
</head>
<body>
<div class="container">
<h1> High-Gainer Discovery Dashboard</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Timeframe: {timeframe}</p>
"""
    # Market context
    if mkt:
        html += f"""
<div class="card">
<h2> Market Context</h2>
<table>
  <tr><td>Date</td><td>{mkt[0]}</td></tr>
  <tr><td>Regime</td><td><strong>{mkt[1].upper()}</strong></td></tr>
  <tr><td>Volatility</td><td>{mkt[2]}</td></tr>
</table>
</div>
"""
    # Top picks
    html += """
<div class="card">
<h2> Top Picks</h2>
<table>
<tr><th>#</th><th>Symbol</th><th>Score</th><th>Confidence</th><th>ML Prob</th><th>Exp Ret</th><th>Risk</th><th>Pattern</th><th>Structure</th></tr>"""
    for i, (_, r) in enumerate(top.iterrows()):
        cls = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(r["confidence"], "")
        conf_cls = {"HIGH": "conf-high", "MEDIUM": "conf-med", "LOW": "conf-low"}.get(r["confidence"], "")
        html += f"""
<tr class="{cls}">
  <td>{i+1}</td>
  <td><strong>{r['symbol']}</strong></td>
  <td>{r['composite_score']:.0f}</td>
  <td class="{conf_cls}">{r['confidence']}</td>
  <td>{r['hg_probability']:.1%}</td>
  <td>{r['expected_return']:+.2f}%</td>
  <td>{r['risk_score']:.0f}</td>
  <td>{r['pattern_score']:.0f}</td>
  <td>{r['structure_score']:.0f}</td>
</tr>
<div class="score-bar"><div class="score-fill" style="width:{r['composite_score']:.0f}%"></div></div>"""
    html += "</table></div>"

    # ML metrics
    if len(ml_metrics) > 0:
        html += '<div class="card"><h2> ML Model Performance</h2><table><tr><th>Model</th><th>Metric</th><th>Value</th></tr>'
        for _, r in ml_metrics.iterrows():
            val = f"{r['value']:.4f}" if isinstance(r['value'], (int, float)) else str(r['value'])
            html += f"<tr><td>{r['model_name']}</td><td>{r['metric']}</td><td>{val}</td></tr>"
        html += "</table></div>"

    # Backtest results
    if len(bt_detail) > 0:
        html += '<div class="card"><h2> Backtest Results</h2><table><tr><th>Strategy</th><th>Total Ret</th><th>Ann Ret</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>Trades</th></tr>'
        for _, r in bt_detail.iterrows():
            html += f"<tr><td>{r['strategy']}</td><td>{float(r['total_return']):.1%}</td><td>{float(r['annual_return']):.1%}</td><td>{float(r['sharpe']):.2f}</td><td>{float(r['max_drawdown']):.1%}</td><td>{float(r['win_rate']):.1%}</td><td>{int(r['num_trades'])}</td></tr>"
        html += "</table></div>"

    html += """
<div class="footer">
High-Gainer Pattern Discovery System | 20-Phase Quantitative Research Pipeline
</div>
</div>
</body>
</html>"""
    return html


def generate(timeframe="1day"):
    con = duckdb.connect(str(DB_PATH))
    html = generate_html(con, timeframe)
    con.close()
    path = BASE_DIR / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {path}")
    return path


if __name__ == "__main__":
    generate()
