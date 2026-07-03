import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const AGGREGATED_FILE = path.join(process.cwd(), "src/app/api/scanner/history/aggregated.json");
const SEEN_FILE = "C:/Users/pc/Downloads/stock hist data/backtest_results/seen_patterns_bcc.csv";

export async function GET() {
  try {
    // Use pre-aggregated JSON cache if available (~5ms vs ~15s for CSV parsing)
    if (fs.existsSync(AGGREGATED_FILE)) {
      const cached = fs.readFileSync(AGGREGATED_FILE, "utf-8");
      return NextResponse.json(JSON.parse(cached));
    }

    // Fallback: parse CSV on-the-fly
    const sectorMap: Record<string, string> = {
      HDFCBANK:"Banking",ICICIBANK:"Banking",SBIN:"Banking",KOTAKBANK:"Banking",AXISBANK:"Banking",INDUSINDBK:"Banking",BANDHANBNK:"Banking",FEDERALBNK:"Banking",RBLBANK:"Banking",IDFCFIRSTB:"Banking",PNB:"Banking",BANKBARODA:"Banking",CANBK:"Banking",INDIANB:"Banking",MAHABANK:"Banking",YESBANK:"Banking",IDBI:"Banking",
      TCS:"IT",INFY:"IT",WIPRO:"IT",HCLTECH:"IT",TECHM:"IT",MPHASIS:"IT",COFORGE:"IT",PERSISTENT:"IT",LTTS:"IT",BSOFT:"IT",KPITTECH:"IT",
      SUNPHARMA:"Pharma",DRREDDY:"Pharma",CIPLA:"Pharma",DIVISLAB:"Pharma",LUPIN:"Pharma",ALKEM:"Pharma",MANKIND:"Pharma",TORNTPHARM:"Pharma",
      MARUTI:"Auto",M_M:"Auto",TATAMOTORS:"Auto",EICHERMOT:"Auto",HEROMOTOCO:"Auto",TVSMOTOR:"Auto",
      RELIANCE:"Energy",ONGC:"Energy",BPCL:"Energy",IOC:"Energy",HINDPETRO:"Energy",GAIL:"Energy",
      NTPC:"Power",POWERGRID:"Power",TATAPOWER:"Power",
      HINDALCO:"Metal",JSWSTEEL:"Metal",TATASTEEL:"Metal",COALINDIA:"Metal",SAIL:"Metal",JINDALSTEL:"Metal",NMDC:"Metal",
      BHARTIARTL:"Telecom",IDEA:"Telecom",
      DLF:"Realty",OBEROIRLTY:"Realty",PRESTIGE:"Realty",GODREJPROP:"Realty",
    };

    if (!fs.existsSync(SEEN_FILE)) {
      return NextResponse.json({ stocks: [], total: 0, error: "No history file found" });
    }

    const raw = fs.readFileSync(SEEN_FILE, "utf-8");
    const lines = raw.trim().split("\n");
    const header = lines[0].split(",").map((h) => h.trim());
    const records: any[] = [];
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(",");
      const row: any = {};
      for (let j = 0; j < header.length; j++) row[header[j]] = vals[j]?.trim() || "";
      records.push(row);
    }

    const stockPatterns = new Map<string, any[]>();
    for (const r of records) {
      const sym = r.symbol;
      if (!stockPatterns.has(sym)) stockPatterns.set(sym, []);
      stockPatterns.get(sym)!.push(r);
    }

    const results: any[] = [];
    for (const [sym, patterns] of stockPatterns) {
      const totalPatterns = patterns.length;
      patterns.sort((a, b) => b.trigger_date.localeCompare(a.trigger_date));
      const latest = patterns[0];
      const price = parseFloat(latest.last_close) || 0;
      const bullish = patterns.filter((p: any) => p.trigger_type === "BULLISH").length;
      const bearish = patterns.filter((p: any) => p.trigger_type === "BEARISH").length;
      const brokenUp = patterns.filter((p: any) => p.status === "BROKEN UP").length;
      const successRate = totalPatterns > 0 ? Math.round((brokenUp / totalPatterns) * 100) : 0;
      const bv = parseFloat(latest.body_vs_avg) || 0;
      const vv = parseFloat(latest.vol_vs_avg) || 0;

      results.push({
        ticker: sym, company: sym,
        price: Math.round(price * 100) / 100, change: 0,
        pattern: `BCC ${latest.trigger_type}`,
        patternCount: totalPatterns, bullishCount: bullish, bearishCount: bearish,
        brokenUp, brokenDown: patterns.filter((p: any) => p.status === "BROKEN DOWN").length,
        consolidating: patterns.filter((p: any) => p.status === "CONSOLIDATING").length,
        successRate, aiScore: Math.round(Math.min(30 + (bv > 2 ? Math.min((bv - 2) * 8, 20) : 0) + (vv > 1.5 ? Math.min((vv - 1.5) * 10, 15) : 0) + 10 + (parseInt(latest.consol_candles) >= 3 ? 10 : 0) + (parseInt(latest.consol_candles) >= 5 ? 5 : 0), 98)),
        latestDate: latest.trigger_date, latestStatus: latest.status,
        bodyVsAvg: bv, volVsAvg: vv, rsi: parseInt(latest.rsi) || 50,
        consolCandles: parseInt(latest.consol_candles) || 0,
        sector: sectorMap[sym] || "Other",
        volume: 0, relVolume: 0, similarity: 0, confidence: 0,
        trend: latest.trigger_type === "BULLISH" ? "Bullish" : "Bearish", alerted: true,
      });
    }

    results.sort((a, b) => b.aiScore - a.aiScore);
    results.forEach((r, i) => (r.rank = i + 1));

    return NextResponse.json({ stocks: results, total: results.length, generatedAt: new Date().toISOString() });
  } catch (err) {
    console.error("History scanner error:", err);
    return NextResponse.json({ error: "History scanner failed" }, { status: 500 });
  }
}
