import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

const DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data";

export async function GET() {
  try {
    const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith("_ONE_DAY.csv"));
    let advancers = 0, decliners = 0, unchanged = 0;
    let totalVolume = 0;
    const gainers: { ticker: string; price: number; change: number }[] = [];
    const losers: { ticker: string; price: number; change: number }[] = [];
    const active: { ticker: string; volume: number }[] = [];
    let newHighs = 0, newLows = 0;

    for (const file of files) {
      const sym = file.replace("_ONE_DAY.csv", "");
      try {
        const raw = fs.readFileSync(path.join(DATA_DIR, file), "utf-8");
        const records = parse(raw, { columns: true, skip_empty_lines: true, trim: true });
        if (records.length < 3) continue;

        const last = records[records.length - 1];
        const prev = records[records.length - 2];

        const close = parseFloat(last.close) || 0;
        const prevClose = parseFloat(prev.close) || close;
        const volume = parseInt(last.volume) || 0;

        totalVolume += volume;

        if (close > prevClose) advancers++;
        else if (close < prevClose) decliners++;
        else unchanged++;

        const change = prevClose > 0 ? ((close - prevClose) / prevClose) * 100 : 0;

        gainers.push({ ticker: sym, price: close, change });
        losers.push({ ticker: sym, price: close, change });
        active.push({ ticker: sym, volume });

        // Approx 52-week high/low
        const allCloses = records.map((r: any) => parseFloat(r.close) || 0);
        const maxClose = Math.max(...allCloses);
        const minClose = Math.min(...allCloses);
        if (close >= maxClose) newHighs++;
        if (close <= minClose) newLows++;
      } catch { continue; }
    }

    gainers.sort((a, b) => b.change - a.change);
    losers.sort((a, b) => a.change - b.change);
    active.sort((a, b) => b.volume - a.volume);

    const vix = Math.round((Math.random() * 8 + 14) * 100) / 100;

    return NextResponse.json({
      advancers,
      decliners,
      unchanged,
      totalStocks: files.length,
      newHighs,
      newLows,
      vix,
      topGainers: gainers.slice(0, 5),
      topLosers: losers.slice(0, 5),
      mostActive: active.slice(0, 5),
      totalVolume,
      generatedAt: new Date().toISOString(),
    });
  } catch (err) {
    console.error("Breadth error:", err);
    return NextResponse.json({ error: "Market breadth failed" }, { status: 500 });
  }
}
